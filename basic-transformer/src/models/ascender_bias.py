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

    # log-γ reparameterization init (γ0>0)
    global_scale_init: float = 0.5  # recommended: 0.2~0.5

    # ===== Optional band-pass on |Δpos| =====
    band_min: Optional[int] = None
    band_max: Optional[int] = None

    # ===== Learnable gate (0..1 via sigmoid) =====
    use_gate: bool = True
    per_head_gate: bool = False
    gate_init: float = -2.2  # σ(-2.2)≈0.10
    gate_floor: float = 0.20 # floor for effective gate (0.15~0.30 rec.)
    gate_ceiling: float = 0.85  # ★ 추가: 게이트 천장(포화 방지)

    # ===== EMA auto-calibration of γ toward target ratio =====
    use_auto_calibrate: bool = True
    target_ratio: float = 0.3             # target std(bias)/std(scores)
    calibrate_step_clamp_lo: float = 0.95
    calibrate_step_clamp_hi: float = 1.05
    ema_momentum: float = 0.9

    # ===== Global runtime safety =====
    hard_max_ratio: float = 0.6
    hard_target_ratio: float = 0.3

    # ===== Gamma soft-cap =====
    gamma_cap: float = 3.0        # cap on exp(logγ)



class AscenderBias(nn.Module):
    """
    Additive bias for attention logits (B,h,T,S).

    Stabilization pack:
      • Center-only normalization (per B,h,T) then clamp.
      • log-parameterized γ with soft-cap; per-head optional.
      • Learnable gate with floor mapping; per-head optional.
      • Directional past-only mask + optional band-pass.
      • Optional EMA auto-calibration of γ (and gentle gate nudging).
      • Hard runtime limiter on ratio to avoid explosions.
    """
    def __init__(self, cfg: AscenderBiasConfig):
        super().__init__()
        self.cfg = cfg

        # === γ: log-parameterization for stability ===
        if cfg.per_head_scale:
            self.register_parameter("gamma_log", None)  # lazy (depends on h)
            self._per_head_pending_init = True
        else:
            g0 = max(1e-6, float(cfg.global_scale_init))
            self.gamma_log = nn.Parameter(torch.tensor(math.log(g0)))
            self._per_head_pending_init = False

        # === gate ===
        if cfg.use_gate:
            if cfg.per_head_gate:
                self.register_parameter("gate_param", None)  # lazy (depends on h)
                self._per_head_gate_pending_init = True
            else:
                self.gate_param = nn.Parameter(torch.tensor(float(cfg.gate_init)))
                self._per_head_gate_pending_init = False
        else:
            self.register_parameter("gate_param", None)
            self._per_head_gate_pending_init = False

        # === auto-calibration buffers ===
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

    def _ensure_gamma_init(self, h: int, device):
        if self.cfg.per_head_scale:
            if (getattr(self, "gamma_log", None) is None) or getattr(self, "_per_head_pending_init", False):
                g0 = max(1e-6, float(self.cfg.global_scale_init))
                self.gamma_log = nn.Parameter(torch.full((h,), math.log(g0), device=device))
                self._per_head_pending_init = False
        else:
            if getattr(self, "gamma_log", None) is None:
                g0 = max(1e-6, float(self.cfg.global_scale_init))
                self.gamma_log = nn.Parameter(torch.tensor(math.log(g0), device=device))
                self._per_head_pending_init = False

    def _eff_gamma(self, h: int, device) -> torch.Tensor:
        self._ensure_gamma_init(h, device)
        gamma = torch.exp(self.gamma_log)
        gamma = torch.clamp(gamma, max=float(self.cfg.gamma_cap))
        return gamma.view(1, -1, 1, 1) if self.cfg.per_head_scale else gamma  # broadcastable

    def _ensure_gate_init(self, h: int, device):
        """Make sure gate_param exists before any use (per-head or scalar)."""
        if not self.cfg.use_gate:
            return
        if self.cfg.per_head_gate:
            if (getattr(self, "gate_param", None) is None) or getattr(self, "_per_head_gate_pending_init", False):
                self.gate_param = nn.Parameter(torch.full((h,), float(self.cfg.gate_init), device=device))
                self._per_head_gate_pending_init = False
        else:
            if getattr(self, "gate_param", None) is None:
                self.gate_param = nn.Parameter(torch.tensor(float(self.cfg.gate_init), device=device))
                self._per_head_gate_pending_init = False

    def _eff_gate(self, h: int, device) -> Optional[torch.Tensor]:
        if not self.cfg.use_gate:
            return None
        self._ensure_gate_init(h, device)
        g_raw = torch.sigmoid(self.gate_param)
        # floor mapping: g ∈ [gate_floor, 1]
        g = self.cfg.gate_floor + (1.0 - self.cfg.gate_floor) * g_raw
        ceiling = getattr(self.cfg, "gate_ceiling", 0.85)
        g = torch.minimum(g, torch.as_tensor(ceiling, device=g.device))
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
        return torch.tril(torch.ones((T, S), device=device, dtype=torch.float32))  # s <= t

    # ----- main -----
    def forward(
        self,
        qh: torch.Tensor,         # (B,h,T,dh)
        kh: torch.Tensor,         # (B,h,S,dh)
        *,
        pre_q: Optional[torch.Tensor] = None,  # (B,T,d_model) for align_source="preproj"
        pre_k: Optional[torch.Tensor] = None,  # (B,S,d_model)
        scores_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, h, T, _ = qh.shape
        S = kh.size(2)
        device = qh.device

        # ensure lazy params exist before any use
        self._ensure_gamma_init(h, device)
        if self.cfg.use_gate:
            self._ensure_gate_init(h, device)

        # ---------- Components ----------
        bias = torch.zeros((B, h, T, S), device=device)

        # 1) Alignment
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

        # 2) Positional (separation / cohesion)
        rel_signed = self._relative_pos_signed(T, S, device)
        rel_abs = rel_signed.abs()
        band = self._apply_bandpass(rel_abs)
        dirmask = self._direction_mask(T, S, device)

        if self.cfg.use_separation and self.cfg.w_sep != 0.0:
            sep = self._gauss(rel_abs, self.cfg.sigma_sep) * band * dirmask
            bias = bias - self.cfg.w_sep * sep.view(1, 1, T, S)  # broadcast

        if self.cfg.use_cohesion and self.cfg.w_coh != 0.0:
            coh = self._gauss(rel_abs, self.cfg.sigma_coh) * band * dirmask
            bias = bias + self.cfg.w_coh * coh.view(1, 1, T, S)

        # ---------- Stabilization ----------
        bias = bias - bias.mean(dim=-1, keepdim=True)                     # center-only
        bias = bias.clamp_(self.cfg.clamp_min, self.cfg.clamp_max)        # clip

        # ---------- Scale (γ) then Gate (σ) ----------
        gamma_eff = self._eff_gamma(h, device)                            # (1,h,1,1) or scalar
        scaled = gamma_eff * bias                                         # define first!

        g_eff = self._eff_gate(h, device) if self.cfg.use_gate else None  # (1,h,1,1) or scalar
        if g_eff is not None:
            scaled = scaled * g_eff

        # NaN/Inf guard
        if not torch.isfinite(scaled).all():
            scaled = torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- Early training: gently open gate if almost closed ----------
        if self.training and self.cfg.use_gate:
            with torch.no_grad():
                g_now = (g_eff if g_eff is not None else self._eff_gate(h, device)).mean()
                if float(g_now) < 0.08:
                    self.gate_param.add_(0.01)

        # ---------- Optional: auto-calibrate γ (and gate) ----------
        if self.cfg.use_auto_calibrate and (scores_std is not None):
            with torch.no_grad():
                # reduce to scalar stds
                bstd = scaled.std().clamp_min(1e-6)
                if isinstance(scores_std, torch.Tensor):
                    sstd = scores_std.detach()
                    sstd = sstd.mean() if sstd.ndim > 0 else sstd
                else:
                    sstd = torch.tensor(float(scores_std), device=device)
                sstd = sstd.clamp_min(1e-6)
                ratio = float((bstd / sstd).item())

                # EMA (for logging)
                if not self._ema_initialized:
                    self.ema_ratio = torch.tensor(ratio, device=device)
                    self._ema_initialized = True
                else:
                    self.ema_ratio.mul_(self.cfg.ema_momentum).add_(
                        (1.0 - self.cfg.ema_momentum) * ratio
                    )

                # multiplicative nudging in log-space
                target = float(self.cfg.target_ratio)
                step_lo = float(self.cfg.calibrate_step_clamp_lo)
                step_hi = float(self.cfg.calibrate_step_clamp_hi)

                err = ratio - target                       # >0: bias가 과함, <0: 약함
                deadband = 0.01                            # 작은 오차는 무시
                if abs(err) > deadband:
                    # γ: log-space multiplicative adjust
                    adj = max(step_lo, min(step_hi, target / max(1e-6, ratio)))
                    self.gamma_log.data.add_(math.log(adj))

                    # gate: 비례 제어 (과하면 닫고, 약하면 열기)
                    if self.cfg.use_gate:
                        k = 0.15                           # 게이트 반응계수 (0.10~0.20)
                        self.gate_param.data.add_(-k * err)  # err>0 → 닫힘(음수), err<0 → 열림(양수)

                adj = max(step_lo, min(step_hi, target / max(1e-6, ratio)))
                # log-γ update: logγ ← logγ + log(adj)
                self._ensure_gamma_init(h, device)
                self.gamma_log.data.add_(math.log(adj))

                # gentle gate nudging if too low
                if self.cfg.use_gate and (ratio < target):
                    self._ensure_gate_init(h, device)
                    k = 0.10  # small safe coefficient
                    delta_g = k * math.log(target / max(1e-6, ratio))
                    self.gate_param.data.add_(delta_g)

                # re-apply gamma cap effect by clamping effective γ implicitly next forward

        # ---- Hard runtime limiter (airbag) ----
        with torch.no_grad():
            bstd = scaled.std().clamp_min(1e-6)
            if isinstance(scores_std, torch.Tensor):
                sstd2 = scores_std.detach()
                sstd2 = sstd2.mean() if sstd2.ndim > 0 else sstd2
            elif scores_std is None:
                sstd2 = torch.tensor(1.0, device=device)  # conservative fallback
            else:
                sstd2 = torch.tensor(float(scores_std), device=device)
            sstd2 = sstd2.clamp_min(1e-6)

            ratio_now = float((bstd / sstd2).item())
            if ratio_now > float(self.cfg.hard_max_ratio):
                sf = float(self.cfg.hard_target_ratio) / max(1e-6, ratio_now)
                scaled.mul_(sf)  # immediate shrink; keeps gradients upstream

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
        return float(g.mean().item())
