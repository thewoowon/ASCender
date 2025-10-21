from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AscenderBiasConfig:
    # ===== Component switches =====
    use_alignment: bool = True
    use_separation: bool = True
    use_cohesion: bool = True

    # ===== Component weights (pre-scale, later multiplied by γ) =====
    w_align: float = 0.2
    w_sep: float = 0.15
    w_coh: float = 0.1

    # ===== Positional kernels (in tokens) =====
    sigma_sep: float = 1.0     # near-field repulsion
    sigma_coh: float = 3.0     # mid-range attraction

    # ===== Alignment options =====
    align_source: Literal["qk", "preproj"] = "qk"
    temperature: float = 1.0   # optional temperature on alignment

    # ===== Safety clamp (applied before γ) =====
    clamp_min: float = -2.0
    clamp_max: float = 2.0

    # ===== Directionality & scaling =====
    past_only: bool = True          # only allow s <= t (self-attn typical)
    per_head_scale: bool = False    # γ per-head or scalar
    global_scale_init: float = 1.2  # γ init

    # ===== Optional band-pass on |Δpos| =====
    band_min: Optional[int] = None  # keep if |Δpos| >= band_min
    band_max: Optional[int] = None  # keep if |Δpos| <= band_max

    # ===== New: learnable gate (0..1 via sigmoid) =====
    use_gate: bool = True
    per_head_gate: bool = False     # set True to have gate per head
    gate_init: float = 0.0          # gate pre-sigmoid init (σ(0)=0.5)

    # ===== New (optional): EMA auto-calibration of γ toward target ratio =====
    use_auto_calibrate: bool = False  # requires scores_std passed from caller
    target_ratio: float = 0.5          # target std(bias)/std(scores)
    calibrate_step_clamp_lo: float = 0.9
    calibrate_step_clamp_hi: float = 1.1
    ema_momentum: float = 0.9          # EMA for observed ratio (cosmetic)


class AscenderBias(nn.Module):
    """
    Additive bias for attention logits (B,h,T,S).

    Key changes (amplify):
      • Center-only normalization (per B,h,T): subtract mean along key axis; no std division.
      • Learnable global scale γ (scalar or per-head).
      • Learnable gate g in (0,1) via sigmoid (scalar or per-head).
      • Directional past-only mask for self-attn (s <= t).
      • Optional band-pass window on |Δpos|.
      • Optional EMA auto-calibration of γ toward a target std ratio if scores_std provided.

    Forward signature is backward compatible; pass scores_std to enable auto-calibration.
    """
    def __init__(self, cfg: AscenderBiasConfig):
        super().__init__()
        self.cfg = cfg

        # γ (learnable), scalar or per-head (lazy if per-head)
        if cfg.per_head_scale:
            self.register_parameter("gamma", None)  # lazy init on first forward (depends on h)
            self._per_head_pending_init = True
        else:
            self.gamma = nn.Parameter(torch.tensor(float(cfg.global_scale_init)))
            self._per_head_pending_init = False

        # gate (learnable), scalar or per-head (lazy too if per-head)
        if cfg.use_gate:
            if cfg.per_head_gate:
                self.register_parameter("gate_param", None)  # lazy per-head
                self._per_head_gate_pending_init = True
            else:
                self.gate_param = nn.Parameter(torch.tensor(float(cfg.gate_init)))
                self._per_head_gate_pending_init = False
        else:
            self.register_parameter("gate_param", None)
            self._per_head_gate_pending_init = False

        # For optional auto-calibration
        if cfg.use_auto_calibrate:
            self.register_buffer("ema_ratio", torch.tensor(0.0), persistent=False)
            self._ema_initialized = False

    # ----- helpers -----
    @staticmethod
    def _relative_pos_signed(T: int, S: int, device) -> torch.Tensor:
        t = torch.arange(T, device=device).unsqueeze(1)  # (T,1)
        s = torch.arange(S, device=device).unsqueeze(0)  # (1,S)
        return (t - s).float()

    @staticmethod
    def _gauss(rel_abs: torch.Tensor, sigma: float) -> torch.Tensor:
        σ = max(1e-6, sigma)
        return torch.exp(- (rel_abs ** 2) / (2.0 * σ * σ))

    def _apply_bandpass(self, rel_abs: torch.Tensor) -> torch.Tensor:
        m = torch.ones_like(rel_abs, dtype=torch.float32)
        if self.cfg.band_min is not None:
            m = m * (rel_abs >= float(self.cfg.band_min))
        if self.cfg.band_max is not None:
            m = m * (rel_abs <= float(self.cfg.band_max))
        return m

    def _direction_mask(self, T: int, S: int, device) -> torch.Tensor:
        if not self.cfg.past_only or T != S:
            return torch.ones((T, S), device=device, dtype=torch.float32)
        return torch.tril(torch.ones((T, S), device=device, dtype=torch.float32))  # s <= t

    # ----- main -----
    def forward(
        self,
        qh: torch.Tensor,         # (B,h,T,dh)
        kh: torch.Tensor,         # (B,h,S,dh)
        *,
        pre_q: Optional[torch.Tensor] = None,  # (B,T,d_model) for align_source="preproj"
        pre_k: Optional[torch.Tensor] = None,  # (B,S,d_model)
        scores_std: Optional[torch.Tensor] = None,  # scalar/tensor; if provided & use_auto_calibrate=True → γ autotune
    ) -> torch.Tensor:
        B, h, T, _ = qh.shape
        S = kh.size(2)
        device = qh.device

        # --- lazy init for per-head γ and per-head gate ---
        if self._per_head_pending_init:
            self.gamma = nn.Parameter(torch.full((h,), float(self.cfg.global_scale_init), device=device))
            self._per_head_pending_init = False
        if self.cfg.use_gate and self._per_head_gate_pending_init:
            self.gate_param = nn.Parameter(torch.full((h,), float(self.cfg.gate_init), device=device))
            self._per_head_gate_pending_init = False

        # ---------- Components ----------
        bias = torch.zeros((B, h, T, S), device=device)

        # 1) Alignment (content-based)
        if self.cfg.use_alignment and self.cfg.w_align != 0.0:
            if self.cfg.align_source == "qk":
                qn = F.normalize(qh, dim=-1)
                kn = F.normalize(kh, dim=-1)
                align = torch.matmul(qn, kn.transpose(-2, -1))  # (B,h,T,S)
            else:
                assert pre_q is not None and pre_k is not None, "pre_q/pre_k needed for align_source='preproj'"
                qn = F.normalize(pre_q, dim=-1).unsqueeze(1).expand(B, h, T, -1)
                kn = F.normalize(pre_k, dim=-1).unsqueeze(1).expand(B, h, S, -1)
                align = torch.matmul(qn, kn.transpose(-2, -1))
            if self.cfg.temperature != 1.0:
                align = align / max(1e-6, self.cfg.temperature)
            bias = bias + self.cfg.w_align * align

        # 2) Positional fields (separation/cohesion) with direction & band-pass
        rel_signed = self._relative_pos_signed(T, S, device)
        rel_abs = rel_signed.abs()
        band = self._apply_bandpass(rel_abs)
        dirmask = self._direction_mask(T, S, device)

        if self.cfg.use_separation and self.cfg.w_sep != 0.0:
            sep = self._gauss(rel_abs, self.cfg.sigma_sep) * band * dirmask
            sep = sep.unsqueeze(0).unsqueeze(0).expand(B, h, T, S)
            bias = bias - self.cfg.w_sep * sep  # repulsion

        if self.cfg.use_cohesion and self.cfg.w_coh != 0.0:
            coh = self._gauss(rel_abs, self.cfg.sigma_coh) * band * dirmask
            coh = coh.unsqueeze(0).unsqueeze(0).expand(B, h, T, S)
            bias = bias + self.cfg.w_coh * coh  # attraction

        # ---------- Stabilization ----------
        # Center-only: subtract mean over key axis
        bias = bias - bias.mean(dim=-1, keepdim=True)
        # Clamp (pre-scale)
        bias = bias.clamp_(self.cfg.clamp_min, self.cfg.clamp_max)

        # ---------- Scale (γ) ----------
        if self.cfg.per_head_scale:
            gamma = self.gamma.view(1, h, 1, 1)
        else:
            gamma = self.gamma  # scalar
        scaled = gamma * bias  # (B,h,T,S)

        # ---------- Gate (σ) ----------
        if self.cfg.use_gate and self.gate_param is not None:
            if self.cfg.per_head_gate:
                g = torch.sigmoid(self.gate_param).view(1, h, 1, 1)
            else:
                g = torch.sigmoid(self.gate_param)  # scalar in (0,1)
            scaled = g * scaled  # gated bias

        # ---------- Optional: auto-calibrate γ toward target ratio ----------
        # Only if: cfg.use_auto_calibrate == True and scores_std provided.
        if self.cfg.use_auto_calibrate and (scores_std is not None):
            with torch.no_grad():
                # reduce to scalar std (robust to shape)
                bias_std = scaled.std().clamp_min(1e-6)
                if isinstance(scores_std, torch.Tensor):
                    sstd = scores_std.detach()
                    sstd = sstd.mean() if sstd.ndim > 0 else sstd
                else:
                    sstd = torch.tensor(float(scores_std), device=device)
                sstd = sstd.clamp_min(1e-6)
                ratio = (bias_std / sstd).item()

                # EMA (cosmetic monitoring)
                if not self._ema_initialized:
                    self.ema_ratio = torch.tensor(ratio, device=device)
                    self._ema_initialized = True
                else:
                    self.ema_ratio.mul_(self.cfg.ema_momentum).add_(
                        (1.0 - self.cfg.ema_momentum) * ratio
                    )

                # multiplicative nudging of γ toward target_ratio
                target = float(self.cfg.target_ratio)
                step_lo = float(self.cfg.calibrate_step_clamp_lo)
                step_hi = float(self.cfg.calibrate_step_clamp_hi)
                adj = max(step_lo, min(step_hi, target / max(1e-6, ratio)))

                # apply to γ (keep shape)
                if self.cfg.per_head_scale:
                    self.gamma.data.mul_(adj)
                else:
                    self.gamma.data.mul_(adj)

        return scaled
