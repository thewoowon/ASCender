from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AscenderBiasV2Config:
    use_alignment: bool = True
    use_separation: bool = True
    use_cohesion: bool = True

    # initial learnable weights
    init_gamma_A: float = 0.1
    init_gamma_S: float = 0.1
    init_gamma_C: float = 0.1

    sigma_sep: float = 1.0
    sigma_coh: float = 4.0

    clamp_min: float = -2.0
    clamp_max: float = 2.0


class AscenderBiasV2(nn.Module):
    """Learnable, differentiable bias for ASCender v2."""

    def __init__(self, cfg: AscenderBiasV2Config, d_model: int):
        super().__init__()
        self.cfg = cfg

        # Learnable contribution weights
        self.gamma_A = nn.Parameter(torch.tensor(cfg.init_gamma_A))
        self.gamma_S = nn.Parameter(torch.tensor(cfg.init_gamma_S))
        self.gamma_C = nn.Parameter(torch.tensor(cfg.init_gamma_C))

        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.proj_q.weight)
        nn.init.xavier_uniform_(self.proj_k.weight)

    @staticmethod
    def _relative_pos(T: int, S: int, device) -> torch.Tensor:
        t = torch.arange(T, device=device).unsqueeze(1)
        s = torch.arange(S, device=device).unsqueeze(0)
        return (t - s).abs().float()

    @staticmethod
    def _pairwise_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x2 = (x ** 2).sum(dim=-1, keepdim=True)
        y2 = (y ** 2).sum(dim=-1, keepdim=True).transpose(-2, -1)
        xy = x @ y.transpose(-2, -1)
        dist2 = (x2 + y2 - 2 * xy).clamp_min(0.0)
        return torch.sqrt(dist2 + 1e-9)

    def _sep_kernel(self, relpos: torch.Tensor) -> torch.Tensor:
        σ = max(1e-6, self.cfg.sigma_sep)
        return torch.exp(- (relpos ** 2) / (2.0 * σ * σ))

    def _coh_kernel(self, relpos: torch.Tensor) -> torch.Tensor:
        σ = max(1e-6, self.cfg.sigma_coh)
        return torch.exp(- (relpos ** 2) / (2.0 * σ * σ))

    def forward(self, qh: torch.Tensor, kh: torch.Tensor) -> torch.Tensor:
        B, h, T, dh = qh.shape
        S = kh.size(2)
        device = qh.device

        bias = torch.zeros((B, h, T, S), device=device)

        if self.cfg.use_alignment:
            qn = F.normalize(self.proj_q(qh), dim=-1)
            kn = F.normalize(self.proj_k(kh), dim=-1)
            align = torch.matmul(qn, kn.transpose(-2, -1))
            bias += self.gamma_A * align

        relpos = self._relative_pos(T, S, device)

        if self.cfg.use_separation:
            sep = self._sep_kernel(relpos)
            sep = sep.unsqueeze(0).unsqueeze(0).expand(B, h, T, S)
            bias -= self.gamma_S * sep

        if self.cfg.use_cohesion:
            coh = self._coh_kernel(relpos)
            coh = coh.unsqueeze(0).unsqueeze(0).expand(B, h, T, S)
            bias += self.gamma_C * coh

        bias = bias.clamp_(self.cfg.clamp_min, self.cfg.clamp_max)
        return bias
