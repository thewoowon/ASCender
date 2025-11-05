# ascender_bias.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
from math import isfinite


# ============================================================
# Config
# ============================================================

@dataclass
class AscenderBiasConfig:
    # ----- Component switches -----
    use_alignment: bool = True
    use_separation: bool = False     # default OFF — early collapse prevention
    use_cohesion: bool = True

    # ----- Component weights (pre-scale; multiplied by γ later) -----
    w_align: float = 0.22
    w_sep: float = 0.00
    w_coh: float = 0.10

    # ----- Positional kernels (token distance) -----
    sigma_sep: float = 1.0
    sigma_coh: float = 3.0

    # ----- Alignment options -----
    align_source: Literal["qk", "preproj"] = "qk"   # use q·k or pre-projection embeddings
    temperature: float = 1.0                        # alignment temperature

    # ----- Safety clamp (before γ) -----
    clamp_min: float = -10.0
    clamp_max: float = 10.0

    # ----- Centering control (disable to preserve global structure) -----
    enable_centering: bool = False

    # ----- Directionality & band-pass -----
    past_only: bool = True
    band_min: Optional[int] = 0
    band_max: Optional[int] = 96

    # ----- Scaling & per-head options -----
    per_head_scale: bool = False
    global_scale_init: float = 1.0                 # γ init (log-param)
    calibrate_warmup_steps: int = 0

    # ----- Learnable gate σ in [floor..ceiling] via sigmoid -----
    use_gate: bool = True
    per_head_gate: bool = False
    gate_init: float = 0.0
    gate_floor: float = 0.15
    gate_ceiling: float = 0.85

    # ----- Auto-calibration toward target bias/std ratio -----
    use_auto_calibrate: bool = False
    target_ratio: float = 0.30
    calibrate_step_clamp_lo: float = 0.90
    calibrate_step_clamp_hi: float = 1.12
    ema_momentum: float = 0.90

    # ----- Global runtime safety limits -----
    hard_max_ratio: float = 0.85
    hard_target_ratio: float = 0.55

    # ----- Gamma clip (soft cap) -----
    gamma_min: float = 0.90
    gamma_cap: float = 6.0
    # gamma_max: float = 1.2  # (not used; informational)

    # ----- Symmetry break -----
    jitter_std: float = 1e-3

    # ----- ALiBi convex mix (ASC ↔ ALiBi) -----
    use_alibi_mix: bool = True
    alpha_start: float = 0.5
    alpha_end: float = 0.6
    alpha_schedule: Literal["none", "cosine"] = "none"
    alpha_total_steps: int = 0

    # ----- Batch std aggregation for bias scaling (used upstream MHA) -----
    std_batch_mean: bool = True

    # --- type guards / coercion ---
    def coerce(self):
        # jitter_std may arrive as str from YAML; coerce to float
        try:
            if isinstance(self.jitter_std, str):
                self.jitter_std = float(self.jitter_std)
        except Exception:
            self.jitter_std = 1e-3
        # clamp gates
        if not isfinite(self.gate_floor): self.gate_floor = 0.35
        if not isfinite(self.gate_ceiling): self.gate_ceiling = 0.65
        self.gate_floor   = max(0.0, min(1.0, self.gate_floor))
        self.gate_ceiling = max(self.gate_floor, min(1.0, self.gate_ceiling))


# ============================================================
# Bias Module
# ============================================================

class AscenderBias(nn.Module):
    """
    Additive bias B for attention logits (B,H,T,S) injected pre-softmax.

    Pipeline:
      (A) Build raw bias from components:
          - Alignment: sim(q, k) or sim(pre_q, pre_k)
          - Separation/Cohesion: Gaussian positional kernels on |Δpos|
      (B) Center-only normalize (per (B,H,T,*)) and clamp
      (C) Optional ALiBi convex mix with α schedule
      (D) γ-scale and (optional) gate σ
      (E) Optional auto-calibration of γ/σ toward target bias/std ratio
      (F) Hard limiter on global ratio (std(bias)/std(scores))

    Drift-safety:
      - _attach_mha(mha, role) wires to hosting MHA and stores expected τ/topk/r/ε snapshot
      - _maybe_drift_warn() prints once if runtime deviates from expected (debug aid)
      - lock_runtime_controls() can freeze γ/σ trainability for deployment runs
    """

    def __init__(self, cfg: AscenderBiasConfig):
        super().__init__()
        self.cfg = cfg

        # ---- wiring & runtime-expectation snapshot (for DRIFT checks) ----
        self._attached_mha = None
        self.role: str = "unknown"
        self.expected_tau: Optional[float] = None
        self.expected_topk: Optional[float] = None
        self.expected_r: Optional[float] = None
        self.expected_v_eps: Optional[float] = None
        self.register_buffer("drift_warned_once", torch.tensor(0, dtype=torch.int32), persistent=False)

        # ---- α schedule step, calibration step ----
        self.register_buffer("_alpha_step", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("calib_steps", torch.tensor(0, dtype=torch.long), persistent=False)

        # ---- γ (log-param) ----
        if cfg.per_head_scale:
            self.register_parameter("gamma_log", None)   # lazy init on first forward (needs H)
            self._per_head_pending_init = True
        else:
            g0 = max(1e-6, float(cfg.global_scale_init))
            self.gamma_log = nn.Parameter(torch.tensor(math.log(g0)))
            self._per_head_pending_init = False

        # ---- σ gate (logit-param) ----
        if cfg.use_gate:
            if cfg.per_head_gate:
                self.register_parameter("gate_param", None)  # lazy init
                self._per_head_gate_pending_init = True
            else:
                self.gate_param = nn.Parameter(torch.tensor(float(cfg.gate_init)))
                self._per_head_gate_pending_init = False
        else:
            self.register_parameter("gate_param", None)
            self._per_head_gate_pending_init = False

        # ---- EMA buffer for ratio ----
        if cfg.use_auto_calibrate:
            self.register_buffer("ema_ratio", torch.tensor(0.0), persistent=False)
            self._ema_initialized = False

        # ---- ALiBi cache ----
        self._cached_alibi = {"H": None, "T": None, "S": None, "device": None, "bias": None}

    # ============================================================
    # Wiring & Runtime Drift
    # ============================================================

    def _attach_mha(self, mha, *, role: Optional[str] = None):
        """MHA를 강한 참조로 잡지 말고 weakref로만 들고 있어 순환 참조를 방지한다."""
        # ✨ 중요: nn.Module 인스턴스를 직접 속성에 달지 말 것!
        try:
            self._attached_mha_ref = weakref.ref(mha)   # SAFE: Module 아님(프록시/콜러블)
        except TypeError:
            self._attached_mha_ref = None               # 실패 시 그냥 포기(경고만)
        if role is not None:
            self.role = role

        # 아래는 순수 값만 복사 — 모듈을 잡지 않음
        self.expected_tau   = float(getattr(mha, "attn_temperature", 1.0))
        self.expected_topk  = float(getattr(mha, "sparsify_k_frac", 0.0))
        self.expected_r     = float(getattr(mha, "std_match_ratio", 1.0))
        self.expected_v_eps = float(getattr(mha, "v_gain_epsilon", 0.0))

        # MHA에 기본 필드 없으면 값만 채워둠(모듈 할당 금지!)
        for k, v in (
            ("attn_temperature", 1.0),
            ("sparsify_k_frac",  0.0),
            ("std_match_ratio",  1.0),
            ("v_gain_epsilon",   0.0),
        ):
            if not hasattr(mha, k):
                setattr(mha, k, v)

    def lock_runtime_controls(self):
        """Freeze γ/σ (no learning) — useful for 'operational' runs."""
        for p in (getattr(self, "gamma_log", None), getattr(self, "gate_param", None)):
            if p is not None:
                p.requires_grad_(False)

    def _maybe_drift_warn(self):
        # 약한 참조 사용
        mha = None
        if hasattr(self, "_attached_mha_ref") and self._attached_mha_ref is not None:
            try:
                mha = self._attached_mha_ref()
            except Exception:
                mha = None

        if mha is None or self.drift_warned_once.item() == 1:
            return

        exp = (self.expected_tau, self.expected_topk, self.expected_r, self.expected_v_eps)
        cur = (getattr(mha, "attn_temperature", None),
               getattr(mha, "sparsify_k_frac", None),
               getattr(mha, "std_match_ratio", None),
               getattr(mha, "v_gain_epsilon", None))
        if any(x is None for x in exp) or any(x is None for x in cur):
            return
        tol = 1e-6
        drift = (abs(exp[0]-cur[0])>tol) or (abs(exp[1]-cur[1])>tol) or (abs(exp[2]-cur[2])>tol) or (abs(exp[3]-cur[3])>tol)
        if drift:
            print(f"[ASC RUNTIME DRIFT][{self.role}] expected tau/topk/r/eps=({exp[0]:.2f}, {exp[1]:.2f}, {exp[2]:.1f}, {exp[3]:.2f}) "
                  f"but got ({cur[0]:.2f}, {cur[1]:.2f}, {cur[2]:.1f}, {cur[3]:.2f}) — using current values.")
            self.drift_warned_once.fill_(1)


    # ============================================================
    # Small utilities
    # ============================================================

    @staticmethod
    def _relative_pos_signed(T: int, S: int, device) -> torch.Tensor:
        t = torch.arange(T, device=device).unsqueeze(1)  # (T,1)
        s = torch.arange(S, device=device).unsqueeze(0)  # (1,S)
        return (t - s).float()                           # (T,S)

    @staticmethod
    def _gauss(rel_abs: torch.Tensor, sigma: float) -> torch.Tensor:
        σ = max(1e-6, float(sigma))
        return torch.exp(-(rel_abs ** 2) / (2.0 * σ * σ))

    @staticmethod
    def _alibi_slopes(H: int, device) -> torch.Tensor:
        def get_slopes(n):
            import math as _m
            def p2(x): return 2 ** _m.floor(_m.log2(x))
            m = p2(n)
            slopes = torch.pow(2, torch.arange(1, m + 1, device=device).float() * (-2.0 / m))
            if m < n:
                extra = torch.pow(2, torch.arange(1, 2 * (n - m) + 1, 2, device=device).float() * (-1.0 / m))
                slopes = torch.cat([slopes, extra], dim=0)
            return slopes
        return get_slopes(H).view(H)

    def _alibi_bias(self, H: int, T: int, S: int, device, past_only: bool) -> torch.Tensor:
        cache = self._cached_alibi
        if (cache["H"] == H and cache["T"] == T and cache["S"] == S and cache["device"] == device
                and cache["bias"] is not None):
            return cache["bias"]
        slopes = self._alibi_slopes(H, device)                   # (H,)
        rel = self._relative_pos_signed(T, S, device)            # (T,S) = i-j
        if past_only:
            rel = torch.clamp(rel, min=0.0)                      # penalize future only
        base = (-rel).unsqueeze(0).unsqueeze(0)                  # (1,1,T,S)
        bias = base * slopes.view(1, H, 1, 1)                    # (1,H,T,S)
        self._cached_alibi = {"H": H, "T": T, "S": S, "device": device, "bias": bias}
        return bias

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

    def _alpha_now(self) -> float:
        if not self.cfg.use_alibi_mix:
            return 1.0
        if self.cfg.alpha_schedule == "none" or self.cfg.alpha_total_steps <= 0:
            return float(self.cfg.alpha_start)
        step = int(self._alpha_step.item())
        total = max(1, int(self.cfg.alpha_total_steps))
        t = min(max(step / total, 0.0), 1.0)
        a0, a1 = float(self.cfg.alpha_start), float(self.cfg.alpha_end)
        # cosine from a0@t=0 → a1@t=1
        return float(a1 + 0.5 * (a0 - a1) * (1 + math.cos(math.pi * t)))

    # ---- lazy param inits (need H) ----
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
        g = torch.exp(self.gamma_log)
        g = torch.clamp(g, min=float(self.cfg.gamma_min), max=float(self.cfg.gamma_cap))
        return g.view(1, -1, 1, 1) if self.cfg.per_head_scale else g

    def _eff_gate(self, h: int, device) -> Optional[torch.Tensor]:
        if not self.cfg.use_gate:
            return None
        self._ensure_gate_init(h, device)
        g_raw = torch.sigmoid(self.gate_param)
        g = self.cfg.gate_floor + (1.0 - self.cfg.gate_floor) * g_raw
        g = torch.minimum(g, torch.as_tensor(float(self.cfg.gate_ceiling), device=g_raw.device))
        return g.view(1, -1, 1, 1) if self.cfg.per_head_gate else g

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        qh: torch.Tensor,         # (B,H,T,dh)
        kh: torch.Tensor,         # (B,H,S,dh)
        *,
        pre_q: Optional[torch.Tensor] = None,  # (B,T,d_model)
        pre_k: Optional[torch.Tensor] = None,  # (B,S,d_model)
        scores_std: Optional[torch.Tensor] = None,  # scalar or (H,)
    ) -> torch.Tensor:
        B, H, T, _ = qh.shape
        S = kh.size(2)
        device = qh.device

        # Drift diagnostic (once)
        self._maybe_drift_warn()

        # ensure params ready
        self._ensure_gamma_init(H, device)
        if self.cfg.use_gate:
            self._ensure_gate_init(H, device)

        # === (A) Build raw bias ===
        bias = torch.zeros((B, H, T, S), device=device)

        # Alignment
        if self.cfg.use_alignment and self.cfg.w_align != 0.0:
            if self.cfg.align_source == "qk":
                qn = F.normalize(qh, dim=-1)
                kn = F.normalize(kh, dim=-1)
                align = torch.matmul(qn, kn.transpose(-2, -1))  # (B,H,T,S)
            else:
                assert pre_q is not None and pre_k is not None, \
                    "pre_q/pre_k required for align_source='preproj'"
                qn = F.normalize(pre_q, dim=-1).unsqueeze(1).expand(B, H, T, -1)
                kn = F.normalize(pre_k, dim=-1).unsqueeze(1).expand(B, H, S, -1)
                align = torch.matmul(qn, kn.transpose(-2, -1))  # (B,H,T,S)
            if self.cfg.temperature != 1.0:
                align = align / max(1e-6, float(self.cfg.temperature))
            bias = bias + float(self.cfg.w_align) * align

        # Positional kernels
        rel = self._relative_pos_signed(T, S, device)
        rel_abs = rel.abs()
        band = self._apply_bandpass(rel_abs)
        dirmask = self._direction_mask(T, S, device)

        if self.cfg.use_separation and self.cfg.w_sep != 0.0:
            sep = self._gauss(rel_abs, self.cfg.sigma_sep) * band * dirmask
            bias = bias - float(self.cfg.w_sep) * sep.view(1, 1, T, S)

        if self.cfg.use_cohesion and self.cfg.w_coh != 0.0:
            coh = self._gauss(rel_abs, self.cfg.sigma_coh) * band * dirmask
            bias = bias + float(self.cfg.w_coh) * coh.view(1, 1, T, S)

        # === (B) Stabilize: optional centering & clamp ===
        # IMPORTANT: Centering removes global structure! Only use for debugging.
        if getattr(self.cfg, "enable_centering", False):
            bias = bias - bias.mean(dim=-1, keepdim=True)
        bias = torch.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0)
        bias = bias.clamp_(float(self.cfg.clamp_min), float(self.cfg.clamp_max))

        # === (C) ALiBi convex mix ===
        if self.cfg.use_alibi_mix:
            alibi = self._alibi_bias(H, T, S, device, past_only=self.cfg.past_only)  # (1,H,T,S)
            alpha = self._alpha_now()
            bias = alpha * bias + (1.0 - alpha) * alibi
            if self.training and self.cfg.alpha_schedule != "none" and self.cfg.alpha_total_steps > 0:
                self._alpha_step += 1

        # === (D) γ-scale & Gate ===
        gamma_eff = self._eff_gamma(H, device)  # (1,H,1,1) or scalar
        scaled = gamma_eff * bias

        g_eff = self._eff_gate(H, device) if self.cfg.use_gate else None
        if g_eff is not None:
            scaled = scaled * g_eff

        # NaN/Inf guard
        if not torch.isfinite(scaled).all():
            scaled = torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # === (E) Auto-calibration (per head when possible) ===
        if self.cfg.use_auto_calibrate and (scores_std is not None) and self.training:
            with torch.no_grad():
                self.calib_steps += 1
                if int(self.calib_steps.item()) >= int(self.cfg.calibrate_warmup_steps):
                    # bias std per head: (H,)
                    bstd_h = scaled.float().std(dim=(0, 2, 3)).clamp_min(1e-6)

                    # scores_std → (H,)
                    if isinstance(scores_std, torch.Tensor):
                        sstd = scores_std.detach().float()
                        if sstd.ndim == 0:
                            sstd_h = sstd.expand_as(bstd_h)
                        else:
                            assert sstd.shape[0] == H, f"scores_std {sstd.shape} expected (H,) with H={H}"
                            sstd_h = sstd.clamp_min(1e-6)
                    else:
                        sstd_h = torch.full_like(bstd_h, float(scores_std), device=device).clamp_min(1e-6)

                    ratio_h = (bstd_h / sstd_h)  # (H,)

                    # EMA on head-mean
                    r_mean = float(ratio_h.mean().item())
                    if not getattr(self, "_ema_initialized", False):
                        self.ema_ratio = torch.tensor(r_mean, device=device)
                        self._ema_initialized = True
                    else:
                        self.ema_ratio.mul_(self.cfg.ema_momentum).add_(
                            (1.0 - self.cfg.ema_momentum) * r_mean
                        )

                    # Gentle log-γ update toward target ratio (reduced interference)
                    target = float(self.cfg.target_ratio)
                    lo = float(self.cfg.calibrate_step_clamp_lo)
                    hi = float(self.cfg.calibrate_step_clamp_hi)
                    adj_h = torch.clamp(target / torch.clamp(ratio_h, min=1e-6), min=lo, max=hi)  # (H,)

                    # Reduce step size to 0.1x (gentler calibration, preserve learning)
                    gentle_adj = (adj_h - 1.0) * 0.1 + 1.0

                    if self.cfg.per_head_scale:
                        self.gamma_log.data.add_(gentle_adj.log())
                    else:
                        self.gamma_log.data.add_(float(gentle_adj.mean().item()))

                    # Proportional control on gate (reduced gain)
                    if self.cfg.use_gate and getattr(self, "gate_param", None) is not None:
                        k = 0.02  # Reduced from 0.08 for gentler adjustment
                        err_h = (ratio_h - target)    # >0 too strong → close; <0 too weak → open
                        delta = (-k * err_h)
                        if self.cfg.per_head_gate:
                            self.gate_param.data.add_(delta)
                        else:
                            self.gate_param.data.add_(float(delta.mean().item()))

        # γ bounds re-apply (numerical drift guard)
        if getattr(self, "gamma_log", None) is not None:
            g_eff_now = torch.exp(self.gamma_log).clamp(min=float(self.cfg.gamma_min), max=float(self.cfg.gamma_cap))
            self.gamma_log.data.copy_(torch.log(g_eff_now))

        # === (F) Hard runtime limiter on global ratio ===
        with torch.no_grad():
            bstd_all = scaled.float().std().clamp_min(1e-6)
            if isinstance(scores_std, torch.Tensor):
                sstd2 = scores_std.detach().float()
                sstd2 = sstd2.mean() if sstd2.ndim > 0 else sstd2
            elif scores_std is None:
                sstd2 = torch.tensor(1.0, device=device)
            else:
                sstd2 = torch.tensor(float(scores_std), device=device)
            sstd2 = sstd2.clamp_min(1e-6)

            ratio_now = float((bstd_all / sstd2).item())
            if ratio_now > float(self.cfg.hard_max_ratio):
                sf = float(self.cfg.hard_target_ratio) / max(1e-6, ratio_now)
                scaled.mul_(sf)  # in-place, graph-safe

        scaled = scaled.to(qh.dtype)
        return scaled

    # ============================================================
    # Logging helpers
    # ============================================================

    @property
    def gamma_effective(self) -> float:
        g = torch.exp(self.gamma_log.detach())
        g = torch.clamp(g, min=float(self.cfg.gamma_min), max=float(self.cfg.gamma_cap))
        return float(g.mean().item())

    @property
    def gate_effective(self) -> Optional[float]:
        if not self.cfg.use_gate or getattr(self, "gate_param", None) is None:
            return None
        g_raw = torch.sigmoid(self.gate_param.detach())
        g = self.cfg.gate_floor + (1.0 - self.cfg.gate_floor) * g_raw
        g = torch.minimum(g, torch.as_tensor(float(self.cfg.gate_ceiling), device=g.device))
        return float(g.mean().item())
