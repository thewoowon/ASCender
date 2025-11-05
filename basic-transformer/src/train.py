# train.py
# (clean, DRIFT-safe, single-source YAML config, τ-schedule, Δp/KL logging)

import os
import csv
import math
import yaml
import argparse
import importlib
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import torch

# ---- Optional helpers (graceful fallback if utils are absent) ----
try:
    from src.utils.sched_ascender import (
        ascender_layerwise_r,
        ascender_layerwise_extra,
        keep_delta_p_band,
    )
except Exception:
    def ascender_layerwise_r(*a, **k): ...
    def ascender_layerwise_extra(*a, **k): ...
    def keep_delta_p_band(*a, **k): ...

# ---- Matplotlib (offscreen) for bias heatmaps ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Loader / Device / I/O
# ============================================================

def load_transformer(mode: str):
    """
    Dynamically load the additive (ASCender) or multiplicative variant
    without branching elsewhere in training code.
    """
    if mode == "multiplicative":
        m = importlib.import_module("src.models.multiplicative_transformer")
    elif mode == "additive":
        m = importlib.import_module("src.models.transformer")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return m.Transformer, m.TransformerConfig, m.LabelSmoothingLoss, m.NoamLR


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_result(csv_path: str, fields: dict):
    header = ["timestamp", "mode", "use_ascender", "bias_combo", "seed", "epoch", "avg_loss"]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    newfile = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if newfile:
            w.writeheader()
        w.writerow(fields)


# ============================================================
# τ schedule & Attention metrics
# ============================================================

def cosine_tau(step: int, total_steps: int, start_tau: float, end_tau: float) -> float:
    if total_steps <= 0:
        return float(start_tau)
    t = min(max(step, 0) / float(total_steps), 1.0)
    return float(end_tau + 0.5 * (start_tau - end_tau) * (1 + math.cos(math.pi * t)))


def apply_tau_to_asc_heads(model, tau: float):
    """
    Apply τ to decoder self-attn heads where ASCender is attached (L0, L1).
    """
    if not hasattr(model, "decoder"):
        return
    layers = getattr(model.decoder, "layers", [])
    for li in (0, 1):
        if li < len(layers):
            mha = layers[li].self_attn
            if hasattr(mha, "attn_temperature"):
                mha.attn_temperature = float(tau)


@torch.no_grad()
def _safe_softmax(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=-1e9, posinf=80.0, neginf=-80.0).clamp(-80, 80)
    return torch.softmax(x, dim=-1)


@torch.no_grad()
def collect_attn_metrics(model) -> dict:
    """
    Uses MHA snapshots (attn_pre_masked / attn_post_masked) captured inside
    the forward pass to compute mean Δp and mean KL across L0/L1 decoder self-attn.
    """
    deltas, kls = [], []
    if not hasattr(model, "decoder"):
        return {"delta_p_mean": 0.0, "kl_mean": 0.0}

    for li in (0, 1):
        if li >= len(model.decoder.layers):
            continue
        mha = model.decoder.layers[li].self_attn
        pre = getattr(mha, "attn_pre_masked", None)
        post = getattr(mha, "attn_post_masked", None)
        if pre is None or post is None:
            continue

        p0, p1 = _safe_softmax(pre.float()), _safe_softmax(post.float())
        deltas.append(torch.mean(torch.abs(p1 - p0)).item())

        eps = 1e-8
        kls.append(torch.mean((p1 + eps) * (torch.log(p1 + eps) - torch.log(p0 + eps))).item())

    if not deltas:
        return {"delta_p_mean": 0.0, "kl_mean": 0.0}
    return {
        "delta_p_mean": float(sum(deltas) / len(deltas)),
        "kl_mean": float(sum(kls) / len(kls)),
    }

@torch.no_grad()
def _row_entropy(P: torch.Tensor) -> torch.Tensor:
    # P: (Tq, Tk)
    P = P.clamp_min(1e-12)
    return (-P * torch.log(P)).sum(dim=-1)

@torch.no_grad()
def _plot_attn_map(A_2d: torch.Tensor, epoch: int, layer: int, head: int, tag: str):
    """
    A_2d: (Tq, Tk) attention probability
    """
    H = _row_entropy(A_2d).mean().item()
    os.makedirs("logs/attn", exist_ok=True)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(A_2d.cpu().numpy(), aspect='auto', cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
    plt.title(f"{tag} | L{layer} H{head} | Epoch {epoch} | H={H:.3f}", fontsize=14, fontweight='bold')
    plt.xlabel("Key index (j)", fontsize=12)
    plt.ylabel("Query index (i)", fontsize=12)
    cbar = plt.colorbar(im, label='Attention Probability')
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    sp = f"logs/attn/{tag}_L{layer}_H{head}_E{epoch:02d}.png"
    plt.savefig(sp, dpi=300); plt.close()
    print(f"[Saved] attn heatmap → {sp}")

@torch.no_grad()
def save_attn_heatmaps(model, batch, device, epoch_idx: int, layers=(0,1), heads=(0,1), tag="train"):
    """
    1) 배치를 한 번 forward 해 attn snapshot을 최신화
    2) 각 레이어/헤드의 post-softmax 확률(attn_post_masked)을 읽어 히트맵 저장
    """
    if not hasattr(model, "decoder") or batch is None:
        return

    # 작은 배치만 사용 (메모리/속도)
    src, tgt_inp, _ = batch
    src, tgt_inp = src.to(device), tgt_inp.to(device)

    # forward 한 번으로 self_attn의 스냅샷(attn_post_masked 등) 갱신
    _ = model(src, tgt_inp)

    for li in layers:
        if li >= len(model.decoder.layers):
            continue
        mha = model.decoder.layers[li].self_attn

        # 선호 순서: attn_post_masked -> last_attn -> attn_probs
        P = getattr(mha, "attn_post_masked", None)
        if P is None:
            P = getattr(mha, "last_attn", None)
        if P is None:
            P = getattr(mha, "attn_probs", None)
        if P is None:
            print(f"[Warn] No attn snapshot on L{li}. Skipping.")
            continue

        # 텐서 형태: (B, n_heads, Tq, Tk)
        if P.dim() != 4:
            print(f"[Warn] Unexpected attn shape on L{li}: {tuple(P.shape)}")
            continue

        # 첫 샘플만 시각화
        P = P[0]  # (n_heads, Tq, Tk)
        for h in heads:
            if h >= P.size(0):
                continue
            A_2d = P[h].float()
            # 혹시 확률이 아니면 softmax로 보정
            if not torch.all((A_2d >= 0) & (A_2d <= 1)) or not torch.allclose(A_2d.sum(-1), torch.ones_like(A_2d.sum(-1)), atol=1e-3):
                A_2d = torch.softmax(A_2d, dim=-1)
            _plot_attn_map(A_2d, epoch=epoch_idx, layer=li, head=h, tag=tag)


# ============================================================
# DRIFT suppressor: sync expected <- current (once per epoch)
# ============================================================

def sync_expected_to_runtime(model):
    """
    Overwrite biaser.expected_* with current runtime hyper-params to suppress
    one-off DRIFT warnings. Call after applying YAML/schedules.
    """
    if not hasattr(model, "decoder"):
        return
    for li in (0, 1):
        if li >= len(model.decoder.layers):
            continue
        b = getattr(model.decoder.layers[li], "biaser_self", None)
        a = model.decoder.layers[li].self_attn if li < len(model.decoder.layers) else None
        if b is None or a is None:
            continue
        # copy current runtime → expected_*
        mappings = [
            ("expected_tau", getattr(a, "attn_temperature", None)),
            ("expected_topk", getattr(a, "sparsify_k_frac", None)),
            ("expected_r", getattr(a, "std_match_ratio", None)),
            ("expected_v_eps", getattr(a, "v_gain_epsilon", None)),
        ]
        for name, cur in mappings:
            if cur is not None and hasattr(b, name):
                setattr(b, name, float(cur))
        # optionally disable repeated drift warnings if config supports it
        if hasattr(b, "cfg") and hasattr(b.cfg, "enable_drift_warn"):
            b.cfg.enable_drift_warn = False


# ============================================================
# A/B quick check (bias ON vs OFF) — diagnostic only
# ============================================================

@torch.no_grad()
def log_alpha_values(model, epoch, seed):
    """
    Log learned alpha mixing weights from Residual Bias Path.
    Shows how much each head uses spatial bias vs. learned attention.
    """
    alpha_data = {"epoch": epoch, "seed": seed, "encoder": [], "decoder": []}

    # Check if residual path is enabled
    has_residual = getattr(model.cfg, "enable_residual_path", False)
    if not has_residual:
        return  # Skip if not using residual path

    # Encoder
    for i, layer in enumerate(model.encoder.layers):
        if hasattr(layer.self_attn, "alpha_logit"):
            alpha = torch.sigmoid(layer.self_attn.alpha_logit.detach()).cpu()
            has_bias = layer.self_attn.biaser is not None
            alpha_data["encoder"].append({
                "layer": i,
                "alpha": alpha.numpy().tolist(),
                "mean": float(alpha.mean()),
                "has_bias": has_bias
            })

    # Decoder
    for i, layer in enumerate(model.decoder.layers):
        if hasattr(layer.self_attn, "alpha_logit"):
            alpha = torch.sigmoid(layer.self_attn.alpha_logit.detach()).cpu()
            has_bias = layer.self_attn.biaser is not None
            alpha_data["decoder"].append({
                "layer": i,
                "type": "self",
                "alpha": alpha.numpy().tolist(),
                "mean": float(alpha.mean()),
                "has_bias": has_bias
            })

    # Save to JSON
    os.makedirs("logs/alpha", exist_ok=True)
    import json
    alpha_path = f"logs/alpha/alpha_epoch{epoch:02d}_seed{seed}.json"
    with open(alpha_path, "w") as f:
        json.dump(alpha_data, f, indent=2)

    # Print summary
    all_means = []
    for enc in alpha_data["encoder"]:
        if enc["has_bias"]:
            all_means.append(enc["mean"])
    for dec in alpha_data["decoder"]:
        if dec["has_bias"]:
            all_means.append(dec["mean"])

    if all_means:
        import numpy as np
        mean_alpha = np.mean(all_means)
        print(f"[Alpha] Epoch {epoch} | Mean α across all heads: {mean_alpha:.4f}")
        if mean_alpha > 0.9:
            print(f"        ⚠️  Model ignoring spatial bias (α > 0.9)")
        elif mean_alpha > 0.7:
            print(f"        📊 Weak spatial bias influence (α > 0.7)")
        elif mean_alpha > 0.3:
            print(f"        ⚖️  Balanced mixing (0.3 < α < 0.7)")
        else:
            print(f"        🎯 Strong spatial bias influence (α < 0.3)")


def quick_ab_check(model, batch, device):
    was_training = model.training
    model.eval()
    src, tgt_inp, tgt_out = (x.to(device) for x in batch)

    logits_on = model(src, tgt_inp)

    keep = []
    for l in model.decoder.layers:
        sa, ca = getattr(l, "self_attn", None), getattr(l, "cross_attn", None)
        keep.append((getattr(sa, "biaser", None), getattr(l, "biaser_self", None),
                     getattr(ca, "biaser", None), getattr(l, "biaser_cross", None)))
        if sa is not None:
            setattr(sa, "biaser", None)
        if hasattr(l, "biaser_self"):
            setattr(l, "biaser_self", None)
        if ca is not None:
            setattr(ca, "biaser", None)
        if hasattr(l, "biaser_cross"):
            setattr(l, "biaser_cross", None)

    # clear snapshots so "off" path is clean
    for l in model.decoder.layers:
        for a in ("self_attn", "cross_attn"):
            mha = getattr(l, a, None)
            if mha is None:
                continue
            for fld in ("attn_pre", "attn_pre_masked", "attn_post_masked",
                        "attn_bias", "attn_logits", "attn_probs", "last_attn"):
                if hasattr(mha, fld):
                    setattr(mha, fld, None)

    logits_off = model(src, tgt_inp)

    # restore
    for l, (sa_b, l_sa_b, ca_b, l_ca_b) in zip(model.decoder.layers, keep):
        if hasattr(l, "self_attn"):
            setattr(l.self_attn, "biaser", sa_b)
        if hasattr(l, "biaser_self"):
            setattr(l, "biaser_self", l_sa_b)
        if hasattr(l, "cross_attn"):
            setattr(l.cross_attn, "biaser", ca_b)
        if hasattr(l, "biaser_cross"):
            setattr(l, "biaser_cross", l_ca_b)

    # NLL compare (token level on tgt_out != pad)
    logp_on = torch.log_softmax(logits_on.float(), dim=-1)
    logp_off = torch.log_softmax(logits_off.float(), dim=-1)
    idx = tgt_out != model.cfg.pad_id
    nll_on = -logp_on.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)[idx].mean().item()
    nll_off = -logp_off.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)[idx].mean().item()
    print(f"[AB] NLL on={nll_on:.4f} | off={nll_off:.4f}")

    if was_training:
        model.train()


# ============================================================
# One training epoch
# ============================================================

def run_epoch(model, data_loader, optimizer, scheduler, criterion, device, clip_grad, epoch_idx=None, ASC_IDX=2):
    model.train()
    total_loss, steps = 0.0, 0
    LOG_EVERY = 100
    last_batch_for_vis = None

    if not hasattr(model, "_global_step"):
        model._global_step = 0
    if not hasattr(model, "_total_steps_all"):
        model._total_steps_all = len(data_loader) if hasattr(data_loader, "__len__") else 1000

    # epoch start: apply τ-snapshot and sync DRIFT expectations
    if getattr(model.cfg, "use_ascender", False):
        # DISABLED: tau schedule can interfere with learning
        # Keep tau at 1.0 for stable training
        # tau0 = cosine_tau(model._global_step, model._total_steps_all, 1.0, 1.10)
        # apply_tau_to_asc_heads(model, tau0)
        sync_expected_to_runtime(model)

    for step, batch in enumerate(data_loader, 1):
        # τ schedule DISABLED - keep constant at 1.0
        # if getattr(model.cfg, "use_ascender", False):
        #     tau = cosine_tau(model._global_step, model._total_steps_all, 1.00, 1.10)
        #     apply_tau_to_asc_heads(model, tau)

        # 1-time wire print for quick sanity
        if step == 1 and getattr(model.cfg, "use_ascender", False):
            for li in range(min(2, len(model.decoder.layers))):
                m = model.decoder.layers[li].self_attn
                print(f"[ASC wire][L{li}] biaser={type(m.biaser).__name__ if m.biaser else None}, "
                      f"r={getattr(m,'std_match_ratio',None)}, "
                      f"tau={getattr(m,'attn_temperature',None)}, "
                      f"topk={getattr(m,'sparsify_k_frac',None)}")

        src, tgt_inp, tgt_out = (t.to(device) for t in batch)
        optimizer.zero_grad(set_to_none=True)

        last_batch_for_vis = (src[:1], tgt_inp[:1], tgt_out[:1])

        logits = model(src, tgt_inp)
        if torch.isnan(logits).any():
            print("⚠️ NaN logits — skipping step")
            model._global_step += 1
            continue

        loss = criterion(logits, tgt_out)
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ NaN/Inf loss — skipping step")
            model._global_step += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()

        # Optional Δp control (kept OFF by default)
        # if step % 100 == 0 and getattr(model.cfg, "use_ascender", False):
        #     keep_delta_p_band(model, layer_idx=0, target=0.0120, tol=0.0020, up=1.12, down=0.96, r_min=0.5, r_max=30.0)
        #     keep_delta_p_band(model, layer_idx=1, target=0.0070, tol=0.0015, up=1.10, down=0.97, r_min=0.4, r_max=12.0)

        # Keep ASC LR constant (if user set a separate lr_asc)
        if hasattr(model, "_asc_lr") and ASC_IDX < len(optimizer.param_groups):
            optimizer.param_groups[ASC_IDX]["lr"] = model._asc_lr

        total_loss += float(loss.detach())
        steps += 1
        model._global_step += 1

        if step % LOG_EVERY == 0:
            metrics = {}
            try:
                metrics = collect_attn_metrics(model)
            except Exception:
                pass
            print(f"Step {step:03d} | Loss {float(loss):.4f} | Δp={metrics.get('delta_p_mean', 0.0):.4f} "
                  f"KL={metrics.get('kl_mean', 0.0):.4f}")
            # mini AB diagnostic
            # quick_ab_check(model, (src, tgt_inp, tgt_out), device)

            # Head-wise 상태
            # for li in [0, 1]:
            #     biaser = getattr(model.decoder.layers[li], "biaser_self", None)
            #     if biaser is None or not hasattr(biaser, "gamma_log"):
            #         continue
            #     g_h = torch.exp(biaser.gamma_log.detach()).clamp(max=float(biaser.cfg.gamma_cap))
            #     gt = None
            #     if getattr(biaser, "gate_param", None) is not None:
            #         gt = torch.sigmoid(biaser.gate_param.detach())
            #         gt = biaser.cfg.gate_floor + (1.0 - biaser.cfg.gate_floor) * gt
            #         gt = torch.minimum(gt, torch.as_tensor(float(biaser.cfg.gate_ceiling), device=gt.device))
            #     def mm(x): return float(x.min()), float(x.median()), float(x.max())
            #     if gt is not None:
            #         a, bmd, c = mm(gt); d, e, f = mm(g_h)
            #         print(f"[ASC head][L{li}] gate(min/med/max)={a:.2f}/{bmd:.2f}/{c:.2f} | "
            #               f"γ(min/med/max)={d:.2f}/{e:.2f}/{f:.2f}")

            # if getattr(model.cfg, "use_ascender", False):
            #     b0 = getattr(model.decoder.layers[0], "biaser_self", None)
            #     if b0 is not None:
            #         try:
            #             if getattr(b0.cfg, "per_head_gate", False) and getattr(b0, "gate_param", None) is not None:
            #                 g = torch.sigmoid(b0.gate_param.detach()).cpu()
            #                 gmn, gmd, gmx = g.min().item(), g.median().item(), g.max().item()
            #             else:
            #                 gmn = gmd = gmx = float(getattr(b0, "gate_effective", 0.0))
            #             if getattr(b0.cfg, "per_head_scale", False) and getattr(b0, "gamma_log", None) is not None:
            #                 gam = torch.exp(b0.gamma_log.detach()).clamp(max=b0.cfg.gamma_cap).cpu()
            #                 amn, amd, amx = gam.min().item(), gam.median().item(), gam.max().item()
            #             else:
            #                 ge = float(getattr(b0, "gamma_effective", 0.0))
            #                 amn = amd = amx = ge
            #             print(f"[ASC head] gate(min/med/max)={gmn:.2f}/{gmd:.2f}/{gmx:.2f} | "
            #                   f"γ(min/med/max)={amn:.2f}/{amd:.2f}/{amx:.2f}")
            #         except Exception as e:
            #             print(f"[ASC head] log failed: {e}")
            #     if b0 is not None and hasattr(b0, "ema_ratio"):
            #         try:
            #             gamma_eff = getattr(b0, "gamma_effective", None)
            #             if gamma_eff is None and hasattr(b0, "gamma"):
            #                 gamma_eff = float(b0.gamma.mean().item())
            #         except Exception:
            #             gamma_eff = None
            #         try:
            #             if hasattr(b0, "gate_effective"):
            #                 gate_eff = b0.gate_effective
            #             elif hasattr(b0, "gate_param") and b0.gate_param is not None:
            #                 gate_eff = float(torch.sigmoid(b0.gate_param).mean().item())
            #             else:
            #                 gate_eff = None
            #         except Exception:
            #             gate_eff = None
            #         print(f"[ASC dbg] ratio(ema)={float(b0.ema_ratio):.3f} | "
            #               f"γ={('None' if gamma_eff is None else f'{gamma_eff:.3f}')} | "
            #               f"gate={('None' if gate_eff is None else f'{gate_eff:.3f}')}")

    avg_loss = total_loss / max(steps, 1)

    # === NEW: epoch-end attention heatmaps for L0/L1, heads 0/1 ===
    if getattr(model.cfg, "use_ascender", False) and (epoch_idx is not None) and (last_batch_for_vis is not None):
        try:
            model.eval()
            save_attn_heatmaps(
                model,
                last_batch_for_vis,
                device,
                epoch_idx=epoch_idx,
                layers=(0,1),
                heads=(0,1),
                tag="train"
            )
        except Exception as e:
            print(f"[Warning] attn heatmap save failed: {e}")
    # === /NEW ===

    # Save bias heatmap for L0 at epoch end (if ASC enabled)
    # NOTE: Visualize ENCODER bias (decoder bias may be disabled in config)
    if getattr(model.cfg, "use_ascender", False) and (epoch_idx is not None):
        try:
            os.makedirs("logs/heatmaps", exist_ok=True)
            # Try encoder first, fallback to decoder if encoder has no bias
            first_layer = model.encoder.layers[0]
            biaser = getattr(first_layer.self_attn, "biaser", None)
            if biaser is None:
                first_layer = model.decoder.layers[0]
                biaser = getattr(first_layer, "biaser_self", None)

            if biaser is not None:
                T = 20
                h = torch.randn((1, T, model.cfg.d_model), device=device, dtype=torch.float32) * 0.01
                qh = first_layer.self_attn._shape(first_layer.self_attn.q_proj(h))
                kh = first_layer.self_attn._shape(first_layer.self_attn.k_proj(h))
                # Call biaser (handle different signatures)
                try:
                    bias_full = biaser(qh, kh, pre_q=h, pre_k=h)
                except:
                    bias_full = biaser(qh, kh)

                # Extract first head from first batch
                if bias_full.dim() == 4:  # (B, H, T, T)
                    bias = bias_full[0, 0].detach().cpu()
                else:  # (B, T, T) or (T, T)
                    bias = bias_full[0].detach().cpu() if bias_full.dim() == 3 else bias_full.detach().cpu()

                # 통계 기반 동적 범위 (mean ± 3σ로 대비 극대화)
                bias_np = bias.numpy()
                bias_mean = bias_np.mean()
                bias_std = bias_np.std()

                # 표준화 후 시각화 (z-score normalization으로 패턴 강조)
                if bias_std > 1e-6:
                    bias_normalized = (bias_np - bias_mean) / bias_std
                    vmin, vmax = -3, 3  # ±3σ 범위로 고정
                else:
                    bias_normalized = bias_np
                    vmin, vmax = bias_np.min(), bias_np.max()

                fig, axes = plt.subplots(1, 2, figsize=(16, 7))

                # Determine layer type for title
                layer_name = "Encoder L0" if hasattr(model.encoder.layers[0].self_attn, 'biaser') and model.encoder.layers[0].self_attn.biaser else "Decoder L0"

                # 왼쪽: 정규화된 전체 히트맵
                im1 = axes[0].imshow(bias_normalized, cmap="RdBu_r", interpolation="nearest", vmin=vmin, vmax=vmax, aspect='auto')
                cbar1 = plt.colorbar(im1, ax=axes[0], label="Z-score (σ)")
                cbar1.ax.tick_params(labelsize=10)
                axes[0].set_title(f"{layer_name} Bias (Normalized) - Epoch {epoch_idx}", fontsize=12, fontweight='bold')
                axes[0].set_xlabel("Key Position", fontsize=11)
                axes[0].set_ylabel("Query Position", fontsize=11)

                # 오른쪽: 원본 값 (mean ± 3σ 범위로 클램핑해서 대비 강화)
                clamp_min = bias_mean - 3 * bias_std
                clamp_max = bias_mean + 3 * bias_std
                bias_clamped = np.clip(bias_np, clamp_min, clamp_max)

                im2 = axes[1].imshow(bias_clamped, cmap="coolwarm", interpolation="nearest",
                                     vmin=clamp_min, vmax=clamp_max, aspect='auto')
                cbar2 = plt.colorbar(im2, ax=axes[1], label="Bias Value (clamped)")
                cbar2.ax.tick_params(labelsize=10)
                axes[1].set_title(f"{layer_name} Raw Values (±3σ) - Epoch {epoch_idx}", fontsize=12, fontweight='bold')
                axes[1].set_xlabel("Key Position", fontsize=11)
                axes[1].set_ylabel("Query Position", fontsize=11)

                # 통계 정보 추가
                stats_text = f"μ={bias_mean:.3f}, σ={bias_std:.3f}, min={bias_np.min():.3f}, max={bias_np.max():.3f}"
                fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, style='italic')

                plt.tight_layout(rect=[0, 0.03, 1, 1])
                sp = f"logs/heatmaps/bias_epoch_{epoch_idx:02d}.png"
                plt.savefig(sp, dpi=300); plt.close()
                print(f"[Saved] Bias heatmap → {sp} | {stats_text}")
        except Exception as e:
            print(f"[Warning] Heatmap save failed: {e}")

    return avg_loss


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    raw = load_config(args.config)

    # SimpleNamespace builder with defaults
    def ns(d, defaults=None):
        d = {} if d is None else d
        if defaults:
            for k, v in defaults.items():
                d.setdefault(k, v)
        return SimpleNamespace(**d)

    cfg = ns(raw)
    cfg.dataset = ns(getattr(cfg, "dataset", None))
    cfg.experiment = ns(getattr(cfg, "experiment", None))
    cfg.model = ns(getattr(cfg, "model", None))
    cfg.model.asc_cfg = ns(getattr(cfg.model, "asc_cfg", None))

    # Global defaults
    cfg.mode = getattr(cfg, "mode", "additive")
    cfg.experiment.seeds = getattr(cfg.experiment, "seeds", [42])
    cfg.experiment.epochs = getattr(cfg.experiment, "epochs", 3)
    cfg.experiment.lr = float(getattr(cfg.experiment, "lr", 5e-4))
    cfg.experiment.lr_asc = float(getattr(cfg.experiment, "lr_asc", cfg.experiment.lr))
    cfg.experiment.warmup_steps = int(getattr(cfg.experiment, "warmup_steps", 200))
    cfg.experiment.clip_grad = float(getattr(cfg.experiment, "clip_grad", 0.8))
    cfg.experiment.smoothing = float(getattr(cfg.experiment, "smoothing", 0.05))

    Transformer, TransformerConfig, LabelSmoothingLoss, NoamLR = load_transformer(cfg.mode)

    # Build AscenderBias Config object explicitly to ensure type-correct init
    from src.models.ascender_bias import AscenderBiasConfig
    asc_cfg_obj = AscenderBiasConfig(**vars(cfg.model.asc_cfg))

    # 타입/범위 정리
    if hasattr(asc_cfg_obj, "coerce"):
        asc_cfg_obj.coerce()

    model_kwargs = vars(cfg.model).copy()
    model_kwargs["asc_cfg"] = asc_cfg_obj
    model_cfg = TransformerConfig(**model_kwargs)

    device = get_device()
    print(f"[Device] {device}")

    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/results_summary.csv"

    # ---- Data (try real dataloader, else dummy) ----
    try:
        from src.data.wikitext_loader import get_dataloader
        train_loader, _ = get_dataloader(cfg, split="train")
    except Exception as e:
        print(f"[WARN] get_dataloader failed ({e}). Using dummy data.")
        def make_dummy(vocab_size, pad_id, batch_size, seq_len=20, num_batches=64):
            for _ in range(num_batches):
                src = torch.randint(1, vocab_size, (batch_size, seq_len))
                tgt_inp = torch.randint(1, vocab_size, (batch_size, seq_len))
                tgt_out = torch.randint(1, vocab_size, (batch_size, seq_len))
                src[:, -1] = pad_id
                tgt_inp[:, -1] = pad_id
                tgt_out[:, -1] = pad_id
                yield (src, tgt_inp, tgt_out)

        bs = int(getattr(cfg.dataset, "batch_size", 16))
        vs = int(getattr(cfg.dataset, "vocab_size", 30000))
        pad = int(getattr(model_cfg, "pad_id", 0))
        sl = int(getattr(cfg.dataset, "seq_len", 128))
        train_loader = list(make_dummy(vs, pad, bs, seq_len=sl, num_batches=64))

    all_losses = []

    # ---- Multi-seed loop ----
    for seed in cfg.experiment.seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print("\n==============================")
        print(f"🚀 Starting training for seed={seed}")
        print("==============================")

        model = Transformer(model_cfg).to(device)

        total_steps_all = (len(train_loader) * cfg.experiment.epochs) if hasattr(train_loader, "__len__") else 0
        setattr(model, "_total_steps_all", int(total_steps_all))
        setattr(model, "_global_step", 0)

        # Wiring verification (optional)
        try:
            from src.utils.debug_ascender import verify_ascender_wiring
            # verify_ascender_wiring(model)
            pass
        except Exception as _e:
            print("[ASC VERIFY] skipped:", _e)

        # ---- Optimizer param groups (LN/bias no-decay; ASC separate LR) ----
        main_decay, main_nodc, asc_params = [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_asc = any(k in n for k in ["biaser", "gamma_log", "gate_param"])
            if is_asc:
                asc_params.append(p)
            else:
                is_ln = ("ln" in n.lower()) or ("layernorm" in n.lower()) or ("norm" in n.lower())
                is_bias = n.endswith(".bias")
                (main_nodc if (is_ln or is_bias) else main_decay).append(p)

        lr_base = float(cfg.experiment.lr)
        lr_asc = float(cfg.experiment.lr_asc)

        optimizer = torch.optim.AdamW(
            [
                {"params": main_decay, "lr": lr_base, "weight_decay": 0.01},
                {"params": main_nodc,  "lr": lr_base, "weight_decay": 0.00},
                {"params": asc_params, "lr": lr_asc,  "weight_decay": 0.00},
            ],
            betas=(0.9, 0.98), eps=1e-9
        )
        model._asc_lr = lr_asc

        scheduler = NoamLR(optimizer, d_model=model_cfg.d_model, warmup_steps=cfg.experiment.warmup_steps)
        criterion = LabelSmoothingLoss(model_cfg.tgt_vocab_size, cfg.experiment.smoothing, ignore_index=model_cfg.pad_id)

        # bias combo tag for CSV logging
        asc = model_cfg.asc_cfg
        combo = []
        def on(flag, w):
            return getattr(asc, flag, True) and float(getattr(asc, w, 0.0)) != 0.0
        if on("use_alignment", "w_align"): combo.append("A")
        if on("use_separation", "w_sep"):  combo.append("S")
        if on("use_cohesion", "w_coh"):    combo.append("C")
        bias_combo = "+".join(combo) if combo else "None"

        # ---- Epoch loop ----
        for epoch in range(1, cfg.experiment.epochs + 1):
            # Optional: after epoch 1, cap gate ceiling to avoid over-opening
            # if getattr(model.cfg, "use_ascender", False) and epoch >= 2:
            #     for li in (0, 1):
            #         if li >= len(model.decoder.layers):
            #             continue
            #         b = getattr(model.decoder.layers[li], "biaser_self", None)
            #         if b is not None and hasattr(b, "cfg"):
            #             if not hasattr(b.cfg, "gate_ceiling") or b.cfg.gate_ceiling is None:
            #                 b.cfg.gate_ceiling = 0.65
            #             elif b.cfg.gate_ceiling > 0.65:
            #                 b.cfg.gate_ceiling = 0.65

            print(f"\n🧭 Epoch {epoch}/{cfg.experiment.epochs} | seed={seed}")
            # resync expected_* at epoch start (after τ snapshot)
            if getattr(model.cfg, "use_ascender", False):
                sync_expected_to_runtime(model)

            avg_loss = run_epoch(
                model, train_loader, optimizer, scheduler, criterion,
                device, cfg.experiment.clip_grad, epoch_idx=epoch, ASC_IDX=2
            )
            print(f"✅ Epoch {epoch} done. AvgLoss={avg_loss:.4f}")
            all_losses.append(avg_loss)

            log_result("logs/results_summary.csv", {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": cfg.mode,
                "use_ascender": model_cfg.use_ascender,
                "bias_combo": bias_combo,
                "seed": seed,
                "epoch": epoch,
                "avg_loss": avg_loss,
            })

            # Log alpha values if using residual path
            if getattr(model_cfg, "enable_residual_path", False):
                log_alpha_values(model, epoch, seed)

        print(f"🏁 Finished seed={seed}")

    # ============================================================
    # Save model checkpoint
    # ============================================================
    checkpoint_dir = "logs/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"{checkpoint_dir}/model_{cfg.experiment.name}_seed{seed}_{timestamp}.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model_cfg,
        "losses": all_losses,
        "timestamp": timestamp,
    }, checkpoint_path)
    print(f"\n✅ Model checkpoint saved → {checkpoint_path}")

    # ============================================================
    # Final Alpha Analysis (if residual path enabled)
    # ============================================================
    if getattr(model_cfg, "enable_residual_path", False) and getattr(model_cfg, "use_ascender", False):
        print("\n" + "="*80)
        print("FINAL ALPHA ANALYSIS")
        print("="*80)

        all_alphas = []
        layer_info = []

        # Encoder
        for i, layer in enumerate(model.encoder.layers):
            if hasattr(layer.self_attn, "alpha_logit") and layer.self_attn.biaser is not None:
                alpha = torch.sigmoid(layer.self_attn.alpha_logit.detach()).cpu().numpy()
                all_alphas.extend(alpha.tolist())
                layer_info.append(f"Encoder L{i}: mean α = {alpha.mean():.4f} [{alpha.min():.3f}, {alpha.max():.3f}]")

        # Decoder
        for i, layer in enumerate(model.decoder.layers):
            if hasattr(layer.self_attn, "alpha_logit") and layer.self_attn.biaser is not None:
                alpha = torch.sigmoid(layer.self_attn.alpha_logit.detach()).cpu().numpy()
                all_alphas.extend(alpha.tolist())
                layer_info.append(f"Decoder L{i} (self): mean α = {alpha.mean():.4f} [{alpha.min():.3f}, {alpha.max():.3f}]")

        if all_alphas:
            all_alphas = np.array(all_alphas)
            mean_alpha = all_alphas.mean()

            print("\nPer-layer α summary:")
            for info in layer_info:
                print(f"  {info}")

            print(f"\n📊 Overall Statistics:")
            print(f"  Mean α:   {mean_alpha:.4f}")
            print(f"  Median α: {np.median(all_alphas):.4f}")
            print(f"  Std α:    {all_alphas.std():.4f}")
            print(f"  Min α:    {all_alphas.min():.4f}")
            print(f"  Max α:    {all_alphas.max():.4f}")

            print(f"\n🎯 INTERPRETATION:")
            if mean_alpha > 0.95:
                print("  ❌ Spatial bias COMPLETELY IGNORED (α > 0.95)")
                print("  → Model learned spatial structure provides ZERO value")
                print("  → Boids-inspired biases don't help this task")
            elif mean_alpha > 0.85:
                print("  ⚠️  Spatial bias MOSTLY IGNORED (α > 0.85)")
                print("  → Model learned spatial structure provides minimal value")
            elif mean_alpha > 0.70:
                print("  📊 WEAK spatial bias influence (α > 0.70)")
                print("  → Model prefers learned attention with minor spatial hints")
            elif mean_alpha > 0.30:
                print("  ⚖️  BALANCED mixing (0.30 < α < 0.70)")
                print("  → Model finds value in combining both signals")
            elif mean_alpha > 0.15:
                print("  📈 STRONG spatial bias influence (α < 0.30)")
                print("  → Model relies heavily on spatial structure")
            else:
                print("  🎯 Spatial bias DOMINATES (α < 0.15)")
                print("  → Spatial structure more valuable than learned patterns!")

            print("="*80)

            # Save comprehensive report
            report_path = f"logs/alpha/FINAL_ANALYSIS_seed{seed}.txt"
            with open(report_path, "w") as f:
                f.write("="*80 + "\n")
                f.write("FINAL ALPHA ANALYSIS - RESIDUAL BIAS PATH\n")
                f.write("="*80 + "\n\n")
                f.write(f"Experiment: {cfg.experiment.name}\n")
                f.write(f"Seed: {seed}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                f.write("Per-layer α summary:\n")
                for info in layer_info:
                    f.write(f"  {info}\n")
                f.write(f"\nOverall Statistics:\n")
                f.write(f"  Mean α:   {mean_alpha:.4f}\n")
                f.write(f"  Median α: {np.median(all_alphas):.4f}\n")
                f.write(f"  Std α:    {all_alphas.std():.4f}\n")
                f.write(f"  Min α:    {all_alphas.min():.4f}\n")
                f.write(f"  Max α:    {all_alphas.max():.4f}\n")
                f.write(f"\nFinal losses: {all_losses}\n")
            print(f"✅ Detailed analysis saved → {report_path}\n")

    # Save aggregated metrics
    metrics_dir = f"logs/{cfg.mode}_logs"
    os.makedirs(metrics_dir, exist_ok=True)
    torch.save({"losses": all_losses}, f"{metrics_dir}/metrics.pt")
    print(f"✅ Saved metrics.pt → {metrics_dir}/metrics.pt")

    # Final bias snapshot (optional)
    if getattr(model_cfg, "use_ascender", False):
        print("\n[DEBUG] Checking one sample Ascender bias matrix stats...")
        # Try encoder first, fallback to decoder
        first_layer = model.encoder.layers[0]
        biaser = getattr(first_layer.self_attn, "biaser", None)
        if biaser is None:
            first_layer = model.decoder.layers[0]
            biaser = getattr(first_layer, "biaser_self", None)

        if biaser is not None:
            T = 20
            h = torch.randn((1, T, model.cfg.d_model), device=device, dtype=torch.float32) * 0.01
            qh = first_layer.self_attn._shape(first_layer.self_attn.q_proj(h))
            kh = first_layer.self_attn._shape(first_layer.self_attn.k_proj(h))

            # Call biaser (handle different signatures)
            try:
                bias_full = biaser(qh, kh, pre_q=h, pre_k=h)
            except:
                bias_full = biaser(qh, kh)

            # Extract first head from first batch
            if bias_full.dim() == 4:  # (B, H, T, T)
                bias = bias_full[0, 0].detach().cpu()
            else:  # (B, T, T) or (T, T)
                bias = bias_full[0].detach().cpu() if bias_full.dim() == 3 else bias_full.detach().cpu()

            os.makedirs("logs/heatmaps", exist_ok=True)

            # 통계 기반 동적 범위 (mean ± 3σ로 대비 극대화)
            bias_np = bias.numpy()
            bias_mean = bias_np.mean()
            bias_std = bias_np.std()

            # 표준화 후 시각화 (z-score normalization으로 패턴 강조)
            if bias_std > 1e-6:
                bias_normalized = (bias_np - bias_mean) / bias_std
                vmin, vmax = -3, 3  # ±3σ 범위로 고정
            else:
                bias_normalized = bias_np
                vmin, vmax = bias_np.min(), bias_np.max()

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            # Determine layer type for title
            layer_name = "Encoder L0" if hasattr(model.encoder.layers[0].self_attn, 'biaser') and model.encoder.layers[0].self_attn.biaser else "Decoder L0"

            # 왼쪽: 정규화된 전체 히트맵
            im1 = axes[0].imshow(bias_normalized, cmap="RdBu_r", interpolation="nearest", vmin=vmin, vmax=vmax, aspect='auto')
            cbar1 = plt.colorbar(im1, ax=axes[0], label="Z-score (σ)")
            cbar1.ax.tick_params(labelsize=10)
            axes[0].set_title(f"{layer_name} Bias (Normalized) - Final", fontsize=12, fontweight='bold')
            axes[0].set_xlabel("Key Position", fontsize=11)
            axes[0].set_ylabel("Query Position", fontsize=11)

            # 오른쪽: 원본 값 (mean ± 3σ 범위로 클램핑해서 대비 강화)
            clamp_min = bias_mean - 3 * bias_std
            clamp_max = bias_mean + 3 * bias_std
            bias_clamped = np.clip(bias_np, clamp_min, clamp_max)

            im2 = axes[1].imshow(bias_clamped, cmap="coolwarm", interpolation="nearest",
                                 vmin=clamp_min, vmax=clamp_max, aspect='auto')
            cbar2 = plt.colorbar(im2, ax=axes[1], label="Bias Value (clamped)")
            cbar2.ax.tick_params(labelsize=10)
            axes[1].set_title(f"{layer_name} Raw Values (±3σ) - Final", fontsize=12, fontweight='bold')
            axes[1].set_xlabel("Key Position", fontsize=11)
            axes[1].set_ylabel("Query Position", fontsize=11)

            # 통계 정보 추가
            stats_text = f"μ={bias_mean:.3f}, σ={bias_std:.3f}, min={bias_np.min():.3f}, max={bias_np.max():.3f}"
            fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, style='italic')

            plt.tight_layout(rect=[0, 0.03, 1, 1])
            sp = "logs/heatmaps/bias_final.png"
            plt.savefig(sp, dpi=300); plt.close()
            print(f"[Saved] Final bias heatmap → {sp}")
            print(f"  Bias stats — mean={bias_mean:.4f}, std={bias_std:.4f}, "
                  f"min={bias_np.min():.4f}, max={bias_np.max():.4f}")

    print("\nTraining complete ✅")


if __name__ == "__main__":
    main()
