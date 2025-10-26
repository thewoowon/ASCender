# src/utils/sched_ascender.py
import math

# ---------- low-level setters ----------
def set_std_match_ratio(mha, r: float):
    if hasattr(mha, "std_match_ratio"):
        mha.std_match_ratio = float(r)

def set_attn_temperature(mha, tau: float):
    if hasattr(mha, "attn_temperature"):
        mha.attn_temperature = float(tau)

def set_sparsify_k_frac(mha, frac: float):
    if hasattr(mha, "sparsify_k_frac"):
        mha.sparsify_k_frac = float(frac)

# ---------- simple warmup ramp for r ----------
def ascender_layerwise_r(model, step, warmup=1000, r0_max=1.2, r1_max=0.7):
    """
    step 기반으로 디코더 self-attn L0/L1의 r을 램프업.
    - warmup 동안 0→r_max 코사인 램프업
    - 이후 r_max 유지
    """
    def ramp(maxv):
        if step >= warmup: return maxv
        t = (1 - math.cos(math.pi * step / warmup)) * 0.5  # 0→1
        return maxv * t

    if len(model.decoder.layers) >= 1:
        set_std_match_ratio(model.decoder.layers[0].self_attn, ramp(r0_max))
    if len(model.decoder.layers) >= 2:
        set_std_match_ratio(model.decoder.layers[1].self_attn, ramp(r1_max))
    for i in range(2, len(model.decoder.layers)):
        set_std_match_ratio(model.decoder.layers[i].self_attn, 0.0)

# ---------- optional: τ/Top-k warmup (포화 완화/집중력 강화) ----------
def ascender_layerwise_extra(model, step, warmup_tau=800, warmup_sparse=800,
                             tau0=1.0, tau1=1.3, k0=0.0, k1=0.2):
    """
    L0에 한해: τ(softmax 온도)와 Top-k sparsify를 코사인 램프로 켬.
    - τ: 1.0 → 1.3 (기본)
    - Top-k: 0.0 → 0.2 (상위 20% 위치만 bias 유지)
    """
    def cosr(step, warmup, v0, v1):
        if warmup <= 0: return v1
        if step >= warmup: return v1
        t = (1 - math.cos(math.pi * step / warmup)) * 0.5
        return v0 + (v1 - v0) * t

    if len(model.decoder.layers) >= 1:
        mha0 = model.decoder.layers[0].self_attn
        set_attn_temperature(mha0, cosr(step, warmup_tau, tau0, tau1))
        set_sparsify_k_frac(mha0, cosr(step, warmup_sparse, k0, k1))

    # L1은 보통 off 유지가 안전
    if len(model.decoder.layers) >= 2:
        mha1 = model.decoder.layers[1].self_attn
        set_attn_temperature(mha1, 1.0)
        set_sparsify_k_frac(mha1, 0.0)

# ---------- keep mean|Δp| in a target band (곱셈형 컨트롤러) ----------
def _nudger(current, target=0.0028, tol=0.001, up=1.05, down=0.95):
    if current < (target - tol): return up
    if current > (target + tol): return down
    return 1.0

def keep_delta_p_band(model, layer_idx=0, target=0.0028, tol=0.001,
                      up=1.05, down=0.95, r_min=0.1, r_max=1.6):
    """
    AttnProbe가 기록한 mean|Δp|을 읽어와서 std_match_ratio(r)를 미세 보정.
    - layer_idx: 디코더 self-attn 레이어 인덱스 (기본 L0)
    - r ∈ [r_min, r_max] 범위로 클램프
    """
    if layer_idx >= len(model.decoder.layers): return
    mha = model.decoder.layers[layer_idx].self_attn
    probe = getattr(mha, "probe", None)
    cache = getattr(probe, "cache", None)
    if not cache or "mean|Δp|" not in cache: return

    cur = float(cache["mean|Δp|"])
    factor = _nudger(cur, target=target, tol=tol, up=up, down=down)

    r = float(getattr(mha, "std_match_ratio", 1.0)) * factor
    r = max(r_min, min(r, r_max))
    set_std_match_ratio(mha, r)
