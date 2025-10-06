from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AscenderBiasConfig:
    # component switches
    use_alignment: bool = True
    use_separation: bool = True
    use_cohesion: bool = True

    # weights (additive to attention scores)
    w_align: float = 0.2
    w_sep: float = 0.15
    w_coh: float = 0.1

    # positional kernels (in tokens)
    sigma_sep: float = 1.0     # small-scale repulsion
    sigma_coh: float = 4.0     # mid-range attraction

    # alignment options
    align_source: Literal["qk", "preproj"] = "qk"
    # "qk": cosine(q,k) per-head; "preproj": cosine on pre-projection hidden (shared across heads)

    # temperature for optional normalization
    temperature: float = 1.0

    # safety clamp
    clamp_min: float = -2.0
    clamp_max: float = 2.0


class AscenderBias(nn.Module):
    """
    Returns an additive bias tensor for attention scores: (B, h, T, S).
    Components:
      - Alignment  : + w_align * cos_sim(content)
      - Separation : - w_sep   * exp(- (Δpos)^2 / 2σ_sep^2)
      - Cohesion   : + w_coh   * exp(- (Δpos)^2 / 2σ_coh^2)
    Notes:
      • Separation discourages over-focusing on immediate neighbors (small σ).
      • Cohesion rewards mid-range grouping (larger σ).
      • Alignment uses content similarity; choose "qk" (per-head) or "preproj".
    """
    def __init__(self, cfg: AscenderBiasConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _relative_pos_bias(T: int, S: int, device) -> torch.Tensor:
        """Return Δpos matrix (T,S) as float."""
        t = torch.arange(T, device=device).unsqueeze(1)  # (T,1)
        s = torch.arange(S, device=device).unsqueeze(0)  # (1,S)
        return (t - s).abs().float()  # (T,S)

    def _sep_kernel(self, relpos: torch.Tensor) -> torch.Tensor:
        σ = max(1e-6, self.cfg.sigma_sep)
        return torch.exp(- (relpos ** 2) / (2.0 * σ * σ))

    def _coh_kernel(self, relpos: torch.Tensor) -> torch.Tensor:
        σ = max(1e-6, self.cfg.sigma_coh)
        return torch.exp(- (relpos ** 2) / (2.0 * σ * σ))

    def forward(
        self,
        qh: torch.Tensor,         # (B,h,T,dh)  projected queries
        kh: torch.Tensor,         # (B,h,S,dh)  projected keys
        *,
        pre_q: Optional[torch.Tensor] = None,  # (B,T,d_model) pre-proj (for align_source="preproj")
        pre_k: Optional[torch.Tensor] = None,  # (B,S,d_model)
    ) -> torch.Tensor:
        B, h, T, dh = qh.shape
        S = kh.size(2)
        device = qh.device
        bias = torch.zeros((B, h, T, S), device=device)

        # --- Alignment (content similarity) ---
        if self.cfg.use_alignment and self.cfg.w_align != 0.0:
            if self.cfg.align_source == "qk":
                qn = F.normalize(qh, dim=-1)                # (B,h,T,dh)
                kn = F.normalize(kh, dim=-1)                # (B,h,S,dh)
                align = torch.matmul(qn, kn.transpose(-2, -1))  # (B,h,T,S) cosine
            else:
                assert pre_q is not None and pre_k is not None, \
                    "pre_q/pre_k required for align_source='preproj'"
                qn = F.normalize(pre_q, dim=-1).unsqueeze(1).expand(B, h, T, -1)  # (B,h,T,d)
                kn = F.normalize(pre_k, dim=-1).unsqueeze(1).expand(B, h, S, -1)  # (B,h,S,d)
                align = torch.matmul(qn, kn.transpose(-2, -1))                    # (B,h,T,S)

            bias = bias + self.cfg.w_align * (align / max(1e-6, self.cfg.temperature))

        # --- Separation / Cohesion (relative-position kernels) ---
        relpos = self._relative_pos_bias(T, S, device)  # (T,S)
        if self.cfg.use_separation and self.cfg.w_sep != 0.0:
            sep = self._sep_kernel(relpos)  # (T,S) in [0,1]
            sep = sep.unsqueeze(0).unsqueeze(0).expand(B, h, T, S)
            bias = bias - self.cfg.w_sep * sep

        if self.cfg.use_cohesion and self.cfg.w_coh != 0.0:
            coh = self._coh_kernel(relpos)  # (T,S)
            coh = coh.unsqueeze(0).unsqueeze(0).expand(B, h, T, S)
            bias = bias + self.cfg.w_coh * coh

        # Clamp for numerical safety
        bias = bias.clamp_(self.cfg.clamp_min, self.cfg.clamp_max)
        return bias
