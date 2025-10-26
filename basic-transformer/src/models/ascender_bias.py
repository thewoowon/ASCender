from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import math
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
    sigma_sep: float = 1.0
    sigma_coh: float = 3.0

    # ===== Alignment options =====
    align_source: Literal["qk", "preproj"] = "qk"
    temperature: float = 1.0

    # ===== Safety clamp (applied before γ) =====
    clamp_min: float = -2.0
    clamp_max: float = 2.0

    # ===== Directionality & scaling =====
    past_only: bool = True
    per_head_scale: bool = False

    # log-γ reparameterization init (γ0>0)
    global_scale_init: float = 0.5

    # ===== Optional band-pass on |Δpos| =====
    band_min: Optional[int] = None
    band_max: Optional[int] = None

    # ===== Learnable gate (0..1 via sigmoid) =====
    use_gate: bool = True
    per_head_gate: bool = False
    gate_init: float = -2.2
    gate_floor: float = 0.20
    gate_ceiling: float = 0.95   # 살짝 올림(포화 방지용 상한)

    # ===== EMA auto-calibration of γ toward target ratio =====
    use_auto_calibrate: bool = True
    target_ratio: float = 0.30
    calibrate_step_clamp_lo: float = 0.95
    calibrate_step_clamp_hi: float = 1.05
    ema_momentum: float = 0.9

    # ===== Global runtime safety =====
    hard_max_ratio: float = 0.60
    hard_target_ratio: float = 0.30

    # ===== Gamma soft-cap =====
    gamma_cap: float = 3.0

    # ===== Symmetry break =====
    jitter_std: float = 1e-3      # per-head 초기값에 아주 작은 노이즈


class AscenderBias(nn.Module):
    """
    Additive bias for attention logits (B,h,T,S) with head-wise stabilization.

      • Center-only normalization (+ clamp).
      • log-γ (per-head optional) + soft-cap.
      • Gate σ in (gate_floor..gate_ceiling).
      • Directional/band-pass options.
      • Head-wise EMA auto-calibration toward target ratio.
      • Hard limiter on ratio.

    If per_head_* is on, we compute ratios and apply nudges per-head.
    """
    def __init__(self, cfg: AscenderBiasConfig):
        super().__init__()
        self.cfg = cfg

        # === γ: log-parameterization (lazy for per-head) ===
        if cfg.per_head_scale:
            self.register_parameter("gamma_log", None)
            self._per_head_pending_init = True
        else:
            g0 = max(1e-6, float(cfg.global_scale_init))
            self.gamma_log = nn.Parameter(torch.tensor(math.log(g0)))
            self._per_head_pending_init = False

        # === gate (lazy for per-head) ===
        if cfg.use_gate:
            if cfg.per_head_gate:
                self.register_parameter("gate_param", None)
                self._per_head_gate_pending_init = True
            else:
                self.gate_param = nn.Parameter(torch.tensor(float(cfg.gate_init)))
                self._per_head_gate_pending_init = False
        else:
            self.register_parameter("gate_param", None)
            self._per_head_gate_pending_init = False

        # === EMA buffer ===
        if cfg.use_auto_calibrate:
            self.register_buffer("ema_ratio", torch.tensor(0.0), persistent=False)
            self._ema_initialized = False

    # ----- helpers -----
    @staticmethod
    def _relative_pos_signed(T: int, S: int, device) -> torch.Tensor:
        t = torch.arange(T, device=device).unsqueeze(1)
        s = torch.arange(S, device=device).unsqueeze(0)
        return (t - s).float()

    @staticmethod
    def _gauss(rel_abs: torch.Tensor, sigma: float) -> torch.Tensor:
        σ = max(1e-6, sigma)
        return torch.exp(- (rel_abs ** 2) / (2.0 * σ * σ))

    def _ensure_gamma_init(self, h: int, device):
        if self.cfg.per_head_scale:
            if (getattr(self, "gamma_log", None) is None) or getattr(self, "_per_head_pending_init", False):
                g0 = max(1e-6, float(self.cfg.global_scale_init))
                base = math.log(g0)
                jitter = torch.randn(h, device=device) * float(self.cfg.jitter_std)
                self.gamma_log = nn.Parameter(torch.full((h,), base, device=device) + jitter)
                self._per_head_pending_init = False
        else:
            if getattr(self, "gamma_log", None) is None:
                g0 = max(1e-6, float(self.cfg.global_scale_init))
                self.gamma_log = nn.Parameter(torch.tensor(math.log(g0), device=device))
                self._per_head_pending_init = False

    def _ensure_gate_init(self, h: int, device):
        if not self.cfg.use_gate:
            return
        if self.cfg.per_head_gate:
            if (getattr(self, "gate_param", None) is None) or getattr(self, "_per_head_gate_pending_init", False):
                base = float(self.cfg.gate_init)
                jitter = torch.randn(h, device=device) * float(self.cfg.jitter_std)
                self.gate_param = nn.Parameter(torch.full((h,), base, device=device) + jitter)
                self._per_head_gate_pending_init = False
        else:
            if getattr(self, "gate_param", None) is None:
                self.gate_param = nn.Parameter(torch.tensor(float(self.cfg.gate_init), device=device))
                self._per_head_gate_pending_init = False

    def _eff_gamma(self, h: int, device) -> torch.Tensor:
        self._ensure_gamma_init(h, device)
        gamma = torch.exp(self.gamma_log)
        gamma = torch.clamp(gamma, max=float(self.cfg.gamma_cap))
        return gamma.view(1, -1, 1, 1) if self.cfg.per_head_scale else gamma

    def _eff_gate(self, h: int, device) -> Optional[torch.Tensor]:
        if not self.cfg.use_gate:
            return None
        self._ensure_gate_init(h, device)
        g_raw = torch.sigmoid(self.gate_param)
        g = self.cfg.gate_floor + (1.0 - self.cfg.gate_floor) * g_raw
        g = torch.minimum(g, torch.as_tensor(float(self.cfg.gate_ceiling), device=g.device))
        return g.view(1, -1, 1, 1) if self.cfg.per_head_gate else g

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
        return torch.tril(torch.ones((T, S), device=device, dtype=torch.float32))

    # ----- main -----
    def forward(
        self,
        qh: torch.Tensor,         # (B,h,T,dh)
        kh: torch.Tensor,         # (B,h,S,dh)
        *,
        pre_q: Optional[torch.Tensor] = None,  # (B,T,d_model)
        pre_k: Optional[torch.Tensor] = None,  # (B,S,d_model)
        scores_std: Optional[torch.Tensor] = None,  # scalar or (H,)
    ) -> torch.Tensor:
        B, h, T, _ = qh.shape
        S = kh.size(2)
        device = qh.device

        self._ensure_gamma_init(h, device)
        if self.cfg.use_gate:
            self._ensure_gate_init(h, device)

        # ---------- Components ----------
        bias = torch.zeros((B, h, T, S), device=device)

        # Alignment
        if self.cfg.use_alignment and self.cfg.w_align != 0.0:
            if self.cfg.align_source == "qk":
                qn = F.normalize(qh, dim=-1)
                kn = F.normalize(kh, dim=-1)
                align = torch.matmul(qn, kn.transpose(-2, -1))
            else:
                assert pre_q is not None and pre_k is not None, "pre_q/pre_k needed for align_source='preproj'"
                qn = F.normalize(pre_q, dim=-1).unsqueeze(1).expand(B, h, T, -1)
                kn = F.normalize(pre_k, dim=-1).unsqueeze(1).expand(B, h, S, -1)
                align = torch.matmul(qn, kn.transpose(-2, -1))
            if self.cfg.temperature != 1.0:
                align = align / max(1e-6, self.cfg.temperature)
            bias = bias + self.cfg.w_align * align

        # Positional (S/C)
        rel_signed = self._relative_pos_signed(T, S, device)
        rel_abs = rel_signed.abs()
        band = self._apply_bandpass(rel_abs)
        dirmask = self._direction_mask(T, S, device)

        if self.cfg.use_separation and self.cfg.w_sep != 0.0:
            sep = self._gauss(rel_abs, self.cfg.sigma_sep) * band * dirmask
            bias = bias - self.cfg.w_sep * sep.view(1, 1, T, S)

        if self.cfg.use_cohesion and self.cfg.w_coh != 0.0:
            coh = self._gauss(rel_abs, self.cfg.sigma_coh) * band * dirmask
            bias = bias + self.cfg.w_coh * coh.view(1, 1, T, S)

        # Stabilization
        bias = bias - bias.mean(dim=-1, keepdim=True)
        bias = bias.clamp_(self.cfg.clamp_min, self.cfg.clamp_max)

        # Scale & Gate
        gamma_eff = self._eff_gamma(h, device)     # (1,h,1,1) or scalar
        scaled = gamma_eff * bias

        g_eff = self._eff_gate(h, device) if self.cfg.use_gate else None
        if g_eff is not None:
            scaled = scaled * g_eff

        # Guard
        if not torch.isfinite(scaled).all():
            scaled = torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- Auto-calibration (head-wise when possible) ----------
        if self.cfg.use_auto_calibrate and (scores_std is not None) and self.training:
            with torch.no_grad():
                # bias std per head: (H,)
                bstd_h = scaled.float().std(dim=(0, 2, 3)).clamp_min(1e-6)

                # scores std: allow scalar or (H,)
                if isinstance(scores_std, torch.Tensor):
                    sstd = scores_std.detach().float()
                    if sstd.ndim == 0:
                        sstd_h = sstd.expand_as(bstd_h)
                    else:
                        assert sstd.shape[0] == h, f"scores_std shape {sstd.shape} expected (H,) with H={h}"
                        sstd_h = sstd.clamp_min(1e-6)
                else:
                    sstd_h = torch.full_like(bstd_h, float(scores_std), device=device).clamp_min(1e-6)

                ratio_h = (bstd_h / sstd_h)  # (H,)

                # EMA (scalar cosmetic: mean of heads)
                r_mean = float(ratio_h.mean().item())
                if not self._ema_initialized:
                    self.ema_ratio = torch.tensor(r_mean, device=device)
                    self._ema_initialized = True
                else:
                    self.ema_ratio.mul_(self.cfg.ema_momentum).add_(
                        (1.0 - self.cfg.ema_momentum) * r_mean
                    )

                # Head-wise nudging in logγ and gate
                target = float(self.cfg.target_ratio)
                step_lo = float(self.cfg.calibrate_step_clamp_lo)
                step_hi = float(self.cfg.calibrate_step_clamp_hi)

                adj_h = torch.clamp(target / torch.clamp(ratio_h, min=1e-6), min=step_lo, max=step_hi)  # (H,)
                # log-γ update per head
                if self.cfg.per_head_scale:
                    self.gamma_log.data.add_(adj_h.log())
                else:
                    # fallback to scalar: use mean adj
                    self.gamma_log.data.add_(float(adj_h.mean().item()).__float__()).add_(0.0)  # ensure tensor-like op
                    # 실제로는 위 한 줄이면 충분하지만 torchscript 모양새 유지용

                # gate per head (proportional control)
                if self.cfg.use_gate:
                    k = 0.15  # responsiveness
                    err_h = (ratio_h - target)  # >0: too strong → close; <0: too weak → open
                    delta = (-k * err_h)
                    if self.cfg.per_head_gate:
                        self.gate_param.data.add_(delta)
                    else:
                        self.gate_param.data.add_(float(delta.mean().item()))

        # ---- Hard runtime limiter ----
        with torch.no_grad():
            bstd = scaled.float().std().clamp_min(1e-6)
            # tolerate scalar/tensor arg
            if isinstance(scores_std, torch.Tensor):
                sstd2 = scores_std.detach().float()
                sstd2 = sstd2.mean() if sstd2.ndim > 0 else sstd2
            elif scores_std is None:
                sstd2 = torch.tensor(1.0, device=device)
            else:
                sstd2 = torch.tensor(float(scores_std), device=device)
            sstd2 = sstd2.clamp_min(1e-6)

            ratio_now = float((bstd / sstd2).item())
            if ratio_now > float(self.cfg.hard_max_ratio):
                sf = float(self.cfg.hard_target_ratio) / max(1e-6, ratio_now)
                scaled.mul_(sf)

        return scaled

    # ----- helpers for logging -----
    @property
    def gamma_effective(self) -> float:
        g = torch.exp(self.gamma_log.detach())
        g = torch.clamp(g, max=float(self.cfg.gamma_cap))
        return float(g.mean().item())

    @property
    def gate_effective(self) -> Optional[float]:
        if not self.cfg.use_gate or getattr(self, "gate_param", None) is None:
            return None
        g_raw = torch.sigmoid(self.gate_param.detach())
        g = self.cfg.gate_floor + (1.0 - self.cfg.gate_floor) * g_raw
        g = torch.minimum(g, torch.as_tensor(float(self.cfg.gate_ceiling), device=g.device))
        return float(g.mean().item())
