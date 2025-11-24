"""
Point Cloud ASCender v2.0 - Vector Kernel Edition

Three-Level Intervention:
1. Graph Construction: ASC-aware neighbor selection
2. Vector Kernel: B_vec ∈ R^C (not scalar bias)
3. Value Pathway: RBP + spatial gating

Based on Point Transformer (Zhao et al., ICCV 2021)
Formula: y_i = Σ_j ρ(γ(φ(x_i) - ψ(x_j) + δ_ij)) ⊙ (α(x_j) + δ_ij)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ASCenderV2Config:
    """Configuration for ASCender v2.0"""

    # ----- Component switches -----
    use_alignment: bool = True
    use_separation: bool = True
    use_cohesion: bool = True

    # ----- Component weights (for vector kernel) -----
    w_align: float = 0.5
    w_sep: float = 0.3
    w_coh: float = 0.4

    # ----- Spatial scales -----
    sigma_sep: float = 0.05   # 5cm - separation range
    sigma_coh: float = 0.50   # 50cm - cohesion range

    # ----- Level 1: Graph construction -----
    enable_graph_reweight: bool = True
    neighbor_enlarge_factor: float = 1.5  # k_large = k * factor

    # ----- Level 2: Vector kernel -----
    enable_vector_kernel: bool = True
    kernel_bottleneck: int = 4  # C // kernel_bottleneck

    # ----- Level 3: Value pathway -----
    enable_value_rbp: bool = True
    enable_value_gating: bool = True
    rbp_lambda: float = 0.3

    # ----- Learnable parameters -----
    per_component_scale: bool = True  # Each component learns its own scale

    # ----- Safety -----
    clamp_min: float = -10.0
    clamp_max: float = 10.0


class VectorKernelEncoder(nn.Module):
    """
    Encodes Boids components into vector kernels (B, N, k, C)

    Instead of scalar bias, each component generates a C-dimensional kernel
    that modulates the relation vector in Point Transformer.
    """

    def __init__(self, channels: int, cfg: ASCenderV2Config):
        super().__init__()
        self.cfg = cfg
        self.channels = channels
        bottleneck = max(channels // cfg.kernel_bottleneck, 8)

        # Alignment encoder: (scalar similarity) → C-dim vector
        if cfg.use_alignment:
            self.align_encoder = nn.Sequential(
                nn.Linear(1, bottleneck),
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck, channels)
            )

        # Separation encoder: (distance + density) → C-dim vector
        if cfg.use_separation:
            self.sep_encoder = nn.Sequential(
                nn.Linear(2, bottleneck),  # [distance, local_density]
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck, channels)
            )

        # Cohesion encoder: (distance + feature_sim) → C-dim vector
        if cfg.use_cohesion:
            self.coh_encoder = nn.Sequential(
                nn.Linear(2, bottleneck),  # [distance, feature_similarity]
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck, channels)
            )

        # Learnable component scales
        if cfg.per_component_scale:
            self.log_scale_a = nn.Parameter(torch.zeros(1))
            self.log_scale_s = nn.Parameter(torch.zeros(1))
            self.log_scale_c = nn.Parameter(torch.zeros(1))

    def compute_alignment_vector(
        self,
        x_i: torch.Tensor,      # (N, C)
        x_j: torch.Tensor,      # (N, k, C)
        normals_i: Optional[torch.Tensor] = None,  # (N, 3)
        normals_j: Optional[torch.Tensor] = None,  # (N, k, 3)
    ) -> torch.Tensor:
        """
        Compute alignment as vector kernel (N, k, C)

        Alignment: Points with similar normals/features should align
        """
        N, k, C = x_j.shape

        # Use normals if available, else use features
        if normals_i is not None and normals_j is not None:
            # Normal-based alignment: cos(θ)
            norm_i = F.normalize(normals_i, dim=-1)  # (N, 3)
            norm_j = F.normalize(normals_j, dim=-1)  # (N, k, 3)
            align_scalar = (norm_i.unsqueeze(1) * norm_j).sum(dim=-1, keepdim=True)  # (N, k, 1)
        else:
            # Feature-based alignment
            feat_i = F.normalize(x_i, dim=-1).unsqueeze(1)  # (N, 1, C)
            feat_j = F.normalize(x_j, dim=-1)  # (N, k, C)
            align_scalar = (feat_i * feat_j).sum(dim=-1, keepdim=True)  # (N, k, 1)

        # Encode scalar → vector kernel (N, k, C)
        align_vec = self.align_encoder(align_scalar)  # (N, k, C)

        # Apply learnable scale
        if self.cfg.per_component_scale:
            scale = torch.exp(self.log_scale_a).clamp(0.1, 10.0)
            align_vec = align_vec * scale

        return align_vec * self.cfg.w_align

    def compute_separation_vector(
        self,
        p_i: torch.Tensor,  # (N, 3)
        p_j: torch.Tensor,  # (N, k, 3)
        x_i: torch.Tensor,  # (N, C)
        x_j: torch.Tensor,  # (N, k, C)
    ) -> torch.Tensor:
        """
        Compute separation as vector kernel (N, k, C)

        Separation: Repel from overcrowded regions
        """
        N, k, C = x_j.shape

        # Distance
        delta_p = p_i.unsqueeze(1) - p_j  # (N, k, 3)
        dist = delta_p.norm(dim=-1, keepdim=True)  # (N, k, 1)

        # Local density (inverse of mean distance to neighbors)
        mean_dist = dist.mean(dim=1, keepdim=True)  # (N, 1, 1)
        density = 1.0 / (mean_dist + 1e-6)  # (N, 1, 1)
        density_broadcast = density.expand(N, k, 1)  # (N, k, 1)

        # Separation kernel: Gaussian-weighted by distance and density
        sep_kernel = torch.exp(-dist.pow(2) / (self.cfg.sigma_sep ** 2 + 1e-6))  # (N, k, 1)

        # Encode [distance, density] → vector (N, k, C)
        sep_input = torch.cat([dist, density_broadcast], dim=-1)  # (N, k, 2)
        sep_vec = self.sep_encoder(sep_input)  # (N, k, C)

        # Negative kernel (repulsion)
        sep_vec = -sep_vec * sep_kernel

        # Apply learnable scale
        if self.cfg.per_component_scale:
            scale = torch.exp(self.log_scale_s).clamp(0.1, 10.0)
            sep_vec = sep_vec * scale

        return sep_vec * self.cfg.w_sep

    def compute_cohesion_vector(
        self,
        p_i: torch.Tensor,  # (N, 3)
        p_j: torch.Tensor,  # (N, k, 3)
        x_i: torch.Tensor,  # (N, C)
        x_j: torch.Tensor,  # (N, k, C)
    ) -> torch.Tensor:
        """
        Compute cohesion as vector kernel (N, k, C)

        Cohesion: Attract to nearby similar points
        """
        N, k, C = x_j.shape

        # Distance
        delta_p = p_i.unsqueeze(1) - p_j  # (N, k, 3)
        dist = delta_p.norm(dim=-1, keepdim=True)  # (N, k, 1)

        # Feature similarity
        feat_i = F.normalize(x_i, dim=-1).unsqueeze(1)  # (N, 1, C)
        feat_j = F.normalize(x_j, dim=-1)  # (N, k, C)
        feat_sim = (feat_i * feat_j).sum(dim=-1, keepdim=True)  # (N, k, 1)

        # Cohesion kernel: Gaussian-weighted attraction
        coh_kernel = torch.exp(-dist.pow(2) / (self.cfg.sigma_coh ** 2 + 1e-6))  # (N, k, 1)

        # Encode [distance, similarity] → vector (N, k, C)
        coh_input = torch.cat([dist, feat_sim], dim=-1)  # (N, k, 2)
        coh_vec = self.coh_encoder(coh_input)  # (N, k, C)

        # Positive kernel (attraction)
        coh_vec = coh_vec * coh_kernel

        # Apply learnable scale
        if self.cfg.per_component_scale:
            scale = torch.exp(self.log_scale_c).clamp(0.1, 10.0)
            coh_vec = coh_vec * scale

        return coh_vec * self.cfg.w_coh

    def forward(
        self,
        p_i: torch.Tensor,
        p_j: torch.Tensor,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        normals_i: Optional[torch.Tensor] = None,
        normals_j: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate combined vector kernel B_vec ∈ R^{N×k×C}

        Returns:
            B_vec: (N, k, C) vector kernel
        """
        N, k, C = x_j.shape
        device = x_j.device

        B_vec = torch.zeros(N, k, C, device=device, dtype=x_j.dtype)

        # Alignment
        if self.cfg.use_alignment:
            A_vec = self.compute_alignment_vector(x_i, x_j, normals_i, normals_j)
            B_vec = B_vec + A_vec

        # Separation
        if self.cfg.use_separation:
            S_vec = self.compute_separation_vector(p_i, p_j, x_i, x_j)
            B_vec = B_vec + S_vec

        # Cohesion
        if self.cfg.use_cohesion:
            C_vec = self.compute_cohesion_vector(p_i, p_j, x_i, x_j)
            B_vec = B_vec + C_vec

        # Safety clamp
        B_vec = B_vec.clamp(self.cfg.clamp_min, self.cfg.clamp_max)

        return B_vec


class ASCGraphReweight(nn.Module):
    """
    Level 1: ASC-aware neighbor selection

    Instead of fixed k-NN, reweight neighbors based on ASC scores.
    """

    def __init__(self, cfg: ASCenderV2Config):
        super().__init__()
        self.cfg = cfg

        # Score aggregation: [align_score, sep_score, coh_score] → scalar
        self.score_net = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        p_i: torch.Tensor,      # (N, 3)
        p_j_large: torch.Tensor,  # (N, k_large, 3)
        x_i: torch.Tensor,      # (N, C)
        x_j_large: torch.Tensor,  # (N, k_large, C)
        normals_i: Optional[torch.Tensor] = None,
        normals_j_large: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute neighbor importance scores

        Returns:
            scores: (N, k_large) neighbor weights
        """
        N, k_large, C = x_j_large.shape

        # Alignment score
        if normals_i is not None and normals_j_large is not None:
            norm_i = F.normalize(normals_i, dim=-1).unsqueeze(1)  # (N, 1, 3)
            norm_j = F.normalize(normals_j_large, dim=-1)  # (N, k_large, 3)
            align_score = (norm_i * norm_j).sum(dim=-1, keepdim=True)  # (N, k_large, 1)
        else:
            feat_i = F.normalize(x_i, dim=-1).unsqueeze(1)  # (N, 1, C)
            feat_j = F.normalize(x_j_large, dim=-1)  # (N, k_large, C)
            align_score = (feat_i * feat_j).sum(dim=-1, keepdim=True)  # (N, k_large, 1)

        # Separation score (inverse of overcrowding)
        delta_p = p_i.unsqueeze(1) - p_j_large  # (N, k_large, 3)
        dist = delta_p.norm(dim=-1, keepdim=True)  # (N, k_large, 1)
        sep_score = torch.exp(-dist.pow(2) / (self.cfg.sigma_sep ** 2 + 1e-6))  # (N, k_large, 1)
        sep_score = 1.0 - sep_score  # Invert: prefer not-too-close points

        # Cohesion score (prefer nearby similar points)
        coh_score = torch.exp(-dist.pow(2) / (self.cfg.sigma_coh ** 2 + 1e-6))  # (N, k_large, 1)

        # Aggregate scores
        scores_input = torch.cat([align_score, sep_score, coh_score], dim=-1)  # (N, k_large, 3)
        scores = self.score_net(scores_input).squeeze(-1)  # (N, k_large)

        return scores


class ValuePathwayModulator(nn.Module):
    """
    Level 3: RBP + Gating in value pathway

    Instead of just v_ij = α(x_j) + δ_ij,
    we do: v_ij' = gate(B_vec) ⊙ v_ij + λ * B_vec
    """

    def __init__(self, channels: int, cfg: ASCenderV2Config):
        super().__init__()
        self.cfg = cfg

        # Gating network: B_vec → gate
        if cfg.enable_value_gating:
            self.gate_net = nn.Sequential(
                nn.Linear(channels, channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(channels // 4, channels),
                nn.Sigmoid()
            )

        # RBP lambda (learnable)
        if cfg.enable_value_rbp:
            self.log_lambda = nn.Parameter(torch.tensor(math.log(cfg.rbp_lambda)))

    def forward(
        self,
        v: torch.Tensor,      # (N, k, C) original value
        B_vec: torch.Tensor,  # (N, k, C) ASC vector kernel
    ) -> torch.Tensor:
        """
        Modulate value pathway

        Returns:
            v_modulated: (N, k, C)
        """
        v_out = v

        # Multiplicative gating
        if self.cfg.enable_value_gating:
            gate = self.gate_net(B_vec)  # (N, k, C)
            v_out = gate * v_out

        # Additive residual bias path
        if self.cfg.enable_value_rbp:
            lambda_rbp = torch.exp(self.log_lambda).clamp(0.01, 1.0)
            v_out = v_out + lambda_rbp * B_vec

        return v_out


# ============================================================================
# Main Point Transformer Layer with ASCender v2.0
# ============================================================================

class PointTransformerLayerASC(nn.Module):
    """
    Point Transformer Layer with ASCender v2.0

    Three-level intervention:
    1. Graph: ASC-aware neighbor selection (optional)
    2. Kernel: Vector B_vec added to relation
    3. Value: RBP + gating modulation
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        share_planes: int = 8,
        nsample: int = 16,
        asc_config: Optional[ASCenderV2Config] = None,
    ):
        super().__init__()

        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        # Standard Point Transformer components
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes)
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, mid_planes // share_planes),
            nn.BatchNorm1d(mid_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes)
        )
        self.softmax = nn.Softmax(dim=1)

        # ASCender v2.0 components
        self.cfg = asc_config or ASCenderV2Config()
        self.use_ascender = asc_config is not None

        if self.use_ascender:
            # Level 1: Graph reweighting
            if self.cfg.enable_graph_reweight:
                self.graph_reweight = ASCGraphReweight(self.cfg)

            # Level 2: Vector kernel encoder
            if self.cfg.enable_vector_kernel:
                self.vector_kernel = VectorKernelEncoder(out_planes, self.cfg)

            # Level 3: Value pathway modulator
            if self.cfg.enable_value_rbp or self.cfg.enable_value_gating:
                self.value_modulator = ValuePathwayModulator(out_planes, self.cfg)

    def forward(self, pxo, normals=None):
        """
        Forward pass with optional ASCender

        Args:
            pxo: (p, x, o) where
                p: (n, 3) coordinates
                x: (n, c) features
                o: (b,) batch offsets
            normals: (n, 3) surface normals (optional, for ASCender)

        Returns:
            x_out: (n, c) output features
        """
        p, x, o = pxo  # (n, 3), (n, c), (b)

        # Q, K, V projections
        x_q = self.linear_q(x)  # (n, mid_planes)
        x_k = self.linear_k(x)  # (n, mid_planes)
        x_v = self.linear_v(x)  # (n, out_planes)

        # Standard Point Transformer neighbor gathering
        # This uses kNN via pointops library
        from lib.pointops.functions import pointops

        # Gather neighbors (this is k-NN grouping)
        x_k_grouped = pointops.queryandgroup(
            self.nsample, p, p, x_k, None, o, o, use_xyz=True
        )  # (n, nsample, 3+mid_planes)
        x_v_grouped = pointops.queryandgroup(
            self.nsample, p, p, x_v, None, o, o, use_xyz=False
        )  # (n, nsample, out_planes)

        # Split position and features
        p_r_raw = x_k_grouped[:, :, 0:3]  # (n, nsample, 3) - RAW relative positions (PRESERVE THIS!)
        x_k_grouped = x_k_grouped[:, :, 3:]  # (n, nsample, mid_planes)

        # Position encoding δ_ij = θ(p_i - p_j)
        p_r = p_r_raw  # Start with raw positions for encoding
        for i, layer in enumerate(self.linear_p):
            if i == 1:
                p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            else:
                p_r = layer(p_r)
        # p_r now (n, nsample, out_planes) - TRANSFORMED for positional encoding
        # p_r_raw still (n, nsample, 3) - PRESERVED for ASC calculations

        # ===== CORE RELATION: w = x_k - x_q + p_r =====
        # This is: rel_ij = ψ(x_j) - φ(x_i) + δ_ij
        w = x_k_grouped - x_q.unsqueeze(1) + p_r.view(
            p_r.shape[0], p_r.shape[1],
            self.out_planes // self.mid_planes, self.mid_planes
        ).sum(2)  # (n, nsample, mid_planes)

        # ===== ASCender v2.0 Intervention =====
        # Gather normals if available (needed for both Level 1 and Level 2)
        normals_grouped = None
        if self.use_ascender and normals is not None:
            normals_grouped = pointops.queryandgroup(
                self.nsample, p, p, normals, None, o, o, use_xyz=False
            )  # (n, nsample, 3)

        # Level 1: Graph Reweighting (modify neighbors before attention)
        if self.use_ascender and self.cfg.enable_graph_reweight:
            # Compute ASC-based neighbor scores
            p_i = p.unsqueeze(1).expand(-1, self.nsample, -1)  # (n, nsample, 3)
            p_j = p_i + p_r_raw  # Use preserved raw coordinates

            graph_scores = self.graph_reweight(
                p_i=p,  # (n, 3)
                p_j=p_j,  # (n, nsample, 3)
                x_i=x,  # (n, c)
                x_j=x_k_grouped + x.unsqueeze(1),  # Reconstruct absolute features
                normals_i=normals,
                normals_j=normals_grouped,
            )  # (n, nsample)

            # Apply graph scores as multiplicative gating to grouped features
            # This modulates the importance of each neighbor
            x_k_grouped = x_k_grouped * graph_scores.unsqueeze(-1)
            x_v_grouped = x_v_grouped * graph_scores.unsqueeze(-1)

        # Level 2: Vector Kernel (compute B_vec for attention bias)
        if self.use_ascender and self.cfg.enable_vector_kernel:
            # We need x_j at full feature dimension for ASC
            x_j_full = x_v_grouped  # (n, nsample, out_planes)
            x_i_full = x_v.unsqueeze(1).expand_as(x_j_full)  # (n, nsample, out_planes)

            # Positions - USE RAW RELATIVE COORDINATES!
            p_i = p.unsqueeze(1).expand(-1, self.nsample, -1)  # (n, nsample, 3)
            p_j = p_i + p_r_raw  # Use preserved raw coordinates, NOT transformed p_r!

            # Generate B_vec
            B_vec = self.vector_kernel(
                p_i=p.squeeze() if p.dim() > 2 else p,  # (n, 3)
                p_j=p_j,  # (n, nsample, 3)
                x_i=x_v,  # (n, out_planes)
                x_j=x_j_full,  # (n, nsample, out_planes)
                normals_i=normals,
                normals_j=normals_grouped,
            )  # (n, nsample, out_planes)

        # Transform relation through γ (linear_w)
        for i, layer in enumerate(self.linear_w):
            if i % 3 == 0:
                w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            else:
                w = layer(w)
        # w now (n, nsample, out_planes // share_planes)

        # Add B_vec to attention weights (after expansion)
        # Need to match dimensions first
        if self.use_ascender and self.cfg.enable_vector_kernel:
            # Expand w to full out_planes for B_vec addition
            # This is a bit tricky - in original PT, w is compressed
            # We'll add B_vec influence through a projection
            w_expanded = w.repeat(1, 1, self.share_planes)  # (n, nsample, out_planes)
            w_expanded = w_expanded + 0.1 * B_vec  # Small additive term
            w = w_expanded[:, :, ::self.share_planes]  # Downsample back

        # Softmax to get attention weights
        w = self.softmax(w)  # (n, nsample, out_planes // share_planes)

        # Value aggregation: v' = α(x_j) + δ_ij
        n, nsample, c = x_v_grouped.shape
        s = self.share_planes

        v = x_v_grouped + p_r  # (n, nsample, out_planes)

        # ===== Level 3: Value Pathway Modulation =====
        if self.use_ascender and (self.cfg.enable_value_rbp or self.cfg.enable_value_gating):
            v = self.value_modulator(v, B_vec)

        # Final aggregation with vector attention
        x_out = (v.view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)

        return x_out


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("ASCender v2.0 - Vector Kernel Edition")
    print("="*60)

    # Setup
    N, k, C = 1024, 16, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    cfg = ASCenderV2Config(
        use_alignment=True,
        use_separation=True,
        use_cohesion=True,
        enable_graph_reweight=True,
        enable_vector_kernel=True,
        enable_value_rbp=True,
        enable_value_gating=True,
    )

    # Test vector kernel encoder
    print("\n1. Testing Vector Kernel Encoder...")
    encoder = VectorKernelEncoder(C, cfg).to(device)

    # Dummy data
    p_i = torch.rand(N, 3, device=device)
    p_j = torch.rand(N, k, 3, device=device)
    x_i = torch.randn(N, C, device=device)
    x_j = torch.randn(N, k, C, device=device)
    normals_i = F.normalize(torch.randn(N, 3, device=device), dim=-1)
    normals_j = F.normalize(torch.randn(N, k, 3, device=device), dim=-1)

    B_vec = encoder(p_i, p_j, x_i, x_j, normals_i, normals_j)
    print(f"   B_vec shape: {B_vec.shape}")  # (N, k, C)
    print(f"   B_vec range: [{B_vec.min().item():.3f}, {B_vec.max().item():.3f}]")
    print(f"   B_vec mean: {B_vec.mean().item():.3f}")

    # Check learnable scales
    if cfg.per_component_scale:
        print(f"   Scale A: {torch.exp(encoder.log_scale_a).item():.3f}")
        print(f"   Scale S: {torch.exp(encoder.log_scale_s).item():.3f}")
        print(f"   Scale C: {torch.exp(encoder.log_scale_c).item():.3f}")

    # Test value modulator
    print("\n2. Testing Value Pathway Modulator...")
    modulator = ValuePathwayModulator(C, cfg).to(device)
    v = torch.randn(N, k, C, device=device)
    v_mod = modulator(v, B_vec)
    print(f"   v_mod shape: {v_mod.shape}")
    print(f"   Lambda RBP: {torch.exp(modulator.log_lambda).item():.3f}")

    # Test graph reweight
    print("\n3. Testing Graph Reweighting...")
    graph_reweight = ASCGraphReweight(cfg).to(device)
    k_large = int(k * cfg.neighbor_enlarge_factor)
    p_j_large = torch.rand(N, k_large, 3, device=device)
    x_j_large = torch.randn(N, k_large, C, device=device)
    normals_j_large = F.normalize(torch.randn(N, k_large, 3, device=device), dim=-1)

    scores = graph_reweight(p_i, p_j_large, x_i, x_j_large, normals_i, normals_j_large)
    print(f"   Scores shape: {scores.shape}")  # (N, k_large)
    print(f"   Scores range: [{scores.min().item():.3f}, {scores.max().item():.3f}]")

    print("\n" + "="*60)
    print("All components working! ✅")
    print("="*60)
