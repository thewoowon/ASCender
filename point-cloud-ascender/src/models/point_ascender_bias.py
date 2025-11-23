"""
Point Cloud ASCender Bias Module

Extends ASCender from 1D text sequences to 3D point clouds.
Key differences from original ASCender:
1. Distance: Token position difference → Euclidean distance
2. Alignment: Token similarity → Surface normal similarity
3. Temporal: Static → Dynamic Boids over time
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PointAscenderConfig:
    """Configuration for Point Cloud ASCender Bias"""

    # ----- Component switches -----
    use_alignment: bool = True
    use_separation: bool = True
    use_cohesion: bool = True

    # ----- Component weights -----
    w_align: float = 0.30
    w_sep: float = 0.15
    w_coh: float = 0.25

    # ----- Spatial kernels (3D distance in meters) -----
    sigma_sep: float = 0.05   # Separation range (5cm)
    sigma_coh: float = 0.50   # Cohesion range (50cm)

    # ----- Alignment options -----
    align_source: Literal["normals", "features", "both"] = "normals"
    temperature: float = 1.0

    # ----- Safety clamp -----
    clamp_min: float = -10.0
    clamp_max: float = 10.0

    # ----- Centering control -----
    enable_centering: bool = False

    # ----- Per-head options -----
    per_head_scale: bool = True    # Each head learns different σ
    per_head_gate: bool = True     # Each head learns different γ

    # ----- Learnable gate σ -----
    use_gate: bool = True
    gate_floor: float = 0.20
    gate_ceiling: float = 0.90

    # ----- Gamma (scale) -----
    gamma_min: float = 0.50
    gamma_cap: float = 8.0
    global_scale_init: float = 1.0

    # ----- Temporal dynamics (for dynamic point clouds) -----
    enable_temporal: bool = False
    temporal_momentum: float = 0.9

    # ----- Residual Bias Path -----
    enable_residual_path: bool = True
    alpha_init: float = 0.0  # sigmoid(0) = 0.5 (balanced)


class PointAscenderBias(nn.Module):
    """
    Boids-Inspired Spatial Bias for Point Cloud Transformers

    Pipeline:
      (A) Build raw bias from components:
          - Alignment: sim(normal_i, normal_j) or sim(feat_i, feat_j)
          - Separation: -exp(-||p_i - p_j||²/σ_sep²)
          - Cohesion: exp(-||p_i - p_j||²/σ_coh²)
      (B) Optional centering and clamping
      (C) γ-scale and gate σ
      (D) Optional temporal smoothing (for dynamic clouds)

    Inputs:
      - qh: Query heads (B, H, Tq, d_head)
      - kh: Key heads (B, H, Tk, d_head)
      - xyz_q: Query 3D coordinates (B, Tq, 3)
      - xyz_k: Key 3D coordinates (B, Tk, 3)
      - normals_q: Query surface normals (B, Tq, 3), optional
      - normals_k: Key surface normals (B, Tk, 3), optional

    Output:
      - bias: (B, H, Tq, Tk) additive bias for attention logits
    """

    def __init__(self, cfg: PointAscenderConfig, n_heads: int = 8):
        super().__init__()
        self.cfg = cfg
        self.n_heads = n_heads

        # ----- γ (log-param for scale) -----
        if cfg.per_head_scale:
            init_log = math.log(cfg.global_scale_init)
            self.gamma_log = nn.Parameter(
                torch.full((n_heads,), init_log, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "gamma_fixed",
                torch.tensor(cfg.global_scale_init, dtype=torch.float32)
            )

        # ----- σ (gate via sigmoid) -----
        if cfg.use_gate:
            if cfg.per_head_gate:
                self.gate_param = nn.Parameter(
                    torch.zeros(n_heads, dtype=torch.float32)
                )
            else:
                self.register_buffer(
                    "gate_fixed",
                    torch.tensor(0.5 * (cfg.gate_floor + cfg.gate_ceiling))
                )

        # ----- Learnable σ_sep, σ_coh per head -----
        if cfg.per_head_scale:
            self.log_sigma_sep = nn.Parameter(
                torch.full((n_heads,), math.log(cfg.sigma_sep))
            )
            self.log_sigma_coh = nn.Parameter(
                torch.full((n_heads,), math.log(cfg.sigma_coh))
            )
        else:
            self.register_buffer("sigma_sep", torch.tensor(cfg.sigma_sep))
            self.register_buffer("sigma_coh", torch.tensor(cfg.sigma_coh))

        # ----- Temporal state (for dynamic point clouds) -----
        if cfg.enable_temporal:
            self.register_buffer("prev_bias", None, persistent=False)

    def _get_gamma(self) -> torch.Tensor:
        """Get γ scale: (H,) or scalar"""
        if self.cfg.per_head_scale:
            gamma = torch.exp(self.gamma_log).clamp(
                min=self.cfg.gamma_min,
                max=self.cfg.gamma_cap
            )
        else:
            gamma = self.gamma_fixed
        return gamma

    def _get_gate(self) -> torch.Tensor:
        """Get gate σ: (H,) or scalar"""
        if not self.cfg.use_gate:
            return torch.tensor(1.0, device=self.gamma_log.device)

        if self.cfg.per_head_gate:
            gate = torch.sigmoid(self.gate_param)
            gate = self.cfg.gate_floor + (self.cfg.gate_ceiling - self.cfg.gate_floor) * gate
        else:
            gate = self.gate_fixed
        return gate

    def _get_sigmas(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get σ_sep, σ_coh: (H,) or scalar"""
        if self.cfg.per_head_scale:
            sigma_sep = torch.exp(self.log_sigma_sep).clamp(min=0.01, max=5.0)
            sigma_coh = torch.exp(self.log_sigma_coh).clamp(min=0.01, max=5.0)
        else:
            sigma_sep = self.sigma_sep
            sigma_coh = self.sigma_coh
        return sigma_sep, sigma_coh

    def compute_alignment(
        self,
        qh: torch.Tensor,  # (B, H, Tq, d_head)
        kh: torch.Tensor,  # (B, H, Tk, d_head)
        normals_q: Optional[torch.Tensor],  # (B, Tq, 3)
        normals_k: Optional[torch.Tensor],  # (B, Tk, 3)
    ) -> torch.Tensor:
        """
        Compute alignment bias: Points with similar normals/features attract

        Returns: (B, H, Tq, Tk)
        """
        B, H, Tq, d_head = qh.shape
        Tk = kh.size(2)

        if self.cfg.align_source == "normals" and normals_q is not None and normals_k is not None:
            # Normal-based alignment (most interpretable)
            # (B, Tq, 3) @ (B, Tk, 3).T → (B, Tq, Tk)
            align = torch.bmm(
                F.normalize(normals_q, dim=-1),  # (B, Tq, 3)
                F.normalize(normals_k, dim=-1).transpose(1, 2)  # (B, 3, Tk)
            )  # (B, Tq, Tk)

            # Expand to heads: (B, 1, Tq, Tk) → (B, H, Tq, Tk)
            align = align.unsqueeze(1).expand(B, H, Tq, Tk)

        elif self.cfg.align_source == "features":
            # Feature-based alignment (like original transformer)
            # (B, H, Tq, d_head) @ (B, H, d_head, Tk) → (B, H, Tq, Tk)
            align = torch.matmul(
                F.normalize(qh, dim=-1),
                F.normalize(kh, dim=-1).transpose(-2, -1)
            )

        elif self.cfg.align_source == "both" and normals_q is not None:
            # Combine both (50-50 mix)
            normal_align = torch.bmm(
                F.normalize(normals_q, dim=-1),
                F.normalize(normals_k, dim=-1).transpose(1, 2)
            ).unsqueeze(1).expand(B, H, Tq, Tk)

            feat_align = torch.matmul(
                F.normalize(qh, dim=-1),
                F.normalize(kh, dim=-1).transpose(-2, -1)
            )

            align = 0.5 * normal_align + 0.5 * feat_align

        else:
            # Fallback to feature-based
            align = torch.matmul(
                F.normalize(qh, dim=-1),
                F.normalize(kh, dim=-1).transpose(-2, -1)
            )

        # Temperature scaling
        align = align / self.cfg.temperature

        return align

    def compute_separation(
        self,
        xyz_q: torch.Tensor,  # (B, Tq, 3)
        xyz_k: torch.Tensor,  # (B, Tk, 3)
        sigma_sep: torch.Tensor,  # (H,) or scalar
    ) -> torch.Tensor:
        """
        Compute separation bias: Points too far apart repel

        Returns: (B, H, Tq, Tk)
        """
        B, Tq, _ = xyz_q.shape
        Tk = xyz_k.size(1)

        # Euclidean distance: (B, Tq, Tk)
        dist = torch.cdist(xyz_q, xyz_k, p=2)  # ||p_i - p_j||

        # Separation kernel: -exp(-dist²/σ²)
        # Far points → large negative bias (suppress)
        if sigma_sep.ndim == 0:  # scalar
            sep = -torch.exp(-dist.pow(2) / (sigma_sep.pow(2) + 1e-8))
            sep = sep.unsqueeze(1).expand(B, self.n_heads, Tq, Tk)
        else:  # (H,)
            sep = -torch.exp(
                -dist.unsqueeze(1) / (sigma_sep.view(1, -1, 1, 1).pow(2) + 1e-8)
            )  # (B, H, Tq, Tk)

        return sep

    def compute_cohesion(
        self,
        xyz_q: torch.Tensor,  # (B, Tq, 3)
        xyz_k: torch.Tensor,  # (B, Tk, 3)
        sigma_coh: torch.Tensor,  # (H,) or scalar
    ) -> torch.Tensor:
        """
        Compute cohesion bias: Nearby points attract

        Returns: (B, H, Tq, Tk)
        """
        B, Tq, _ = xyz_q.shape
        Tk = xyz_k.size(1)

        # Euclidean distance: (B, Tq, Tk)
        dist = torch.cdist(xyz_q, xyz_k, p=2)

        # Cohesion kernel: exp(-dist²/σ²)
        # Close points → large positive bias (boost)
        if sigma_coh.ndim == 0:  # scalar
            coh = torch.exp(-dist.pow(2) / (sigma_coh.pow(2) + 1e-8))
            coh = coh.unsqueeze(1).expand(B, self.n_heads, Tq, Tk)
        else:  # (H,)
            coh = torch.exp(
                -dist.unsqueeze(1) / (sigma_coh.view(1, -1, 1, 1).pow(2) + 1e-8)
            )  # (B, H, Tq, Tk)

        return coh

    def forward(
        self,
        qh: torch.Tensor,
        kh: torch.Tensor,
        xyz_q: torch.Tensor,
        xyz_k: torch.Tensor,
        normals_q: Optional[torch.Tensor] = None,
        normals_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: Compute Boids-inspired spatial bias

        Args:
            qh: Query heads (B, H, Tq, d_head)
            kh: Key heads (B, H, Tk, d_head)
            xyz_q: Query 3D coordinates (B, Tq, 3)
            xyz_k: Key 3D coordinates (B, Tk, 3)
            normals_q: Query surface normals (B, Tq, 3), optional
            normals_k: Key surface normals (B, Tk, 3), optional

        Returns:
            bias: (B, H, Tq, Tk) additive bias for attention logits
        """
        B, H, Tq, d_head = qh.shape
        Tk = kh.size(2)
        device = qh.device

        # Get learnable parameters
        gamma = self._get_gamma()  # (H,) or scalar
        gate = self._get_gate()    # (H,) or scalar
        sigma_sep, sigma_coh = self._get_sigmas()  # (H,) or scalar

        # Initialize bias
        bias = torch.zeros(B, H, Tq, Tk, device=device, dtype=qh.dtype)

        # (A) Alignment component
        if self.cfg.use_alignment and self.cfg.w_align != 0.0:
            align = self.compute_alignment(qh, kh, normals_q, normals_k)
            bias = bias + self.cfg.w_align * align

        # (S) Separation component
        if self.cfg.use_separation and self.cfg.w_sep != 0.0:
            sep = self.compute_separation(xyz_q, xyz_k, sigma_sep)
            bias = bias + self.cfg.w_sep * sep

        # (C) Cohesion component
        if self.cfg.use_cohesion and self.cfg.w_coh != 0.0:
            coh = self.compute_cohesion(xyz_q, xyz_k, sigma_coh)
            bias = bias + self.cfg.w_coh * coh

        # (B) Optional centering
        if self.cfg.enable_centering:
            # Center per (B, H, Tq, *)
            bias = bias - bias.mean(dim=-1, keepdim=True)

        # Safety clamp
        bias = bias.clamp(self.cfg.clamp_min, self.cfg.clamp_max)

        # (C) Scale by γ and gate σ
        # Reshape gamma if per-head
        if self.cfg.per_head_scale and gamma.numel() > 1:
            gamma = gamma.view(1, -1, 1, 1)
        # Reshape gate if per-head
        if self.cfg.per_head_gate and gate.numel() > 1:
            gate = gate.view(1, -1, 1, 1)

        bias = bias * gamma * gate

        # (D) Temporal smoothing (for dynamic point clouds)
        if self.cfg.enable_temporal and self.training:
            if self.prev_bias is not None and self.prev_bias.shape == bias.shape:
                bias = (1 - self.cfg.temporal_momentum) * bias + \
                       self.cfg.temporal_momentum * self.prev_bias
            self.prev_bias = bias.detach().clone()

        return bias


# Example usage and testing
if __name__ == "__main__":
    # Setup
    B, H, Tq, Tk = 2, 8, 128, 128
    d_head = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    cfg = PointAscenderConfig(
        use_alignment=True,
        use_separation=True,
        use_cohesion=True,
        per_head_scale=True,
        enable_residual_path=True,
    )

    # Model
    biaser = PointAscenderBias(cfg, n_heads=H).to(device)

    # Dummy inputs
    qh = torch.randn(B, H, Tq, d_head, device=device)
    kh = torch.randn(B, H, Tk, d_head, device=device)
    xyz_q = torch.rand(B, Tq, 3, device=device) * 2.0  # 0~2 meters
    xyz_k = torch.rand(B, Tk, 3, device=device) * 2.0
    normals_q = F.normalize(torch.randn(B, Tq, 3, device=device), dim=-1)
    normals_k = F.normalize(torch.randn(B, Tk, 3, device=device), dim=-1)

    # Forward
    bias = biaser(qh, kh, xyz_q, xyz_k, normals_q, normals_k)

    print(f"✅ Bias shape: {bias.shape}")  # (B, H, Tq, Tk)
    print(f"✅ Bias range: [{bias.min().item():.3f}, {bias.max().item():.3f}]")
    print(f"✅ Bias mean: {bias.mean().item():.3f}")
    print(f"✅ Gamma: {biaser._get_gamma()}")
    print(f"✅ Gate: {biaser._get_gate()}")
    print(f"✅ Sigma (sep, coh): {biaser._get_sigmas()}")
