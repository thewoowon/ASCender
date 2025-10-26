# train.py (clean & robust)

import csv
import math
import os
import argparse
import yaml
import torch
import importlib
from datetime import datetime
from types import SimpleNamespace
from src.utils.sched_ascender import (
    ascender_layerwise_r,
    ascender_layerwise_extra,
    keep_delta_p_band,
)

# === optional: non-interactive backend (server/CLI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# IO / Config / Device
# -----------------------------

def load_transformer(mode: str):
    if mode == "multiplicative":
        module = importlib.import_module("src.models.multiplicative_transformer")
    elif mode == "additive":
        module = importlib.import_module("src.models.transformer")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return module.Transformer, module.TransformerConfig, module.LabelSmoothingLoss, module.NoamLR


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def log_result(csv_path, fields):
    header = ["timestamp", "mode", "use_ascender", "bias_combo", "seed", "epoch", "avg_loss"]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    newfile = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if newfile:
            w.writeheader()
        w.writerow(fields)


# -----------------------------
# AB check (Ascender ON vs OFF)
# -----------------------------

@torch.no_grad()
def quick_ab_check(model, batch, device):
    was_training = model.training
    model.eval()

    src, tgt_inp, tgt_out = (x.to(device) for x in batch)

    # === A) ASC ON ===
    logits_on = model(src, tgt_inp)

    # === B) ASC OFF ===
    keep = []
    for l in model.decoder.layers:
        sa = getattr(l, "self_attn", None)
        ca = getattr(l, "cross_attn", None)
        keep.append((
            getattr(sa, "biaser", None), getattr(l, "biaser_self", None),
            getattr(ca, "biaser", None), getattr(l, "biaser_cross", None),
        ))
        if sa is not None: setattr(sa, "biaser", None)
        if hasattr(l, "biaser_self"): setattr(l, "biaser_self", None)
        if ca is not None: setattr(ca, "biaser", None)
        if hasattr(l, "biaser_cross"): setattr(l, "biaser_cross", None)

    # 내부 스냅샷/캐시 제거 (공정 비교)
    for l in model.decoder.layers:
        for a in ("self_attn", "cross_attn"):
            mha = getattr(l, a, None)
            if mha is None:
                continue
            for fld in (
                "attn_pre", "attn_pre_masked", "attn_post_masked",
                "attn_bias", "attn_logits", "attn_probs", "last_attn"
            ):
                if hasattr(mha, fld):
                    setattr(mha, fld, None)

    logits_off = model(src, tgt_inp)

    # 복구
    for l, (sa_b, l_sa_b, ca_b, l_ca_b) in zip(model.decoder.layers, keep):
        if hasattr(l, "self_attn"):  setattr(l.self_attn,  "biaser", sa_b)
        if hasattr(l, "biaser_self"): setattr(l, "biaser_self", l_sa_b)
        if hasattr(l, "cross_attn"): setattr(l.cross_attn, "biaser", ca_b)
        if hasattr(l, "biaser_cross"): setattr(l, "biaser_cross", l_ca_b)

    # === NLL ===
    logp_on  = torch.log_softmax(logits_on.float(), dim=-1)
    logp_off = torch.log_softmax(logits_off.float(), dim=-1)
    idx = tgt_out != model.cfg.pad_id
    nll_on  = -logp_on.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)[idx].mean().item()
    nll_off = -logp_off.gather(-1, tgt_out.unsqueeze(-1)).squeeze(-1)[idx].mean().item()
    print(f"[AB] NLL on={nll_on:.4f} | off={nll_off:.4f}")

    if was_training:
        model.train()


# -----------------------------
# Training
# -----------------------------

def run_epoch(model, data_loader, optimizer, scheduler, criterion, device, clip_grad: float, epoch_idx=None, ASC_IDX=2):
    model.train()
    total_loss, valid_steps = 0.0, 0
    LOG_EVERY = 10

    total_steps = len(data_loader) if hasattr(data_loader, "__len__") else 1000

    for step, batch in enumerate(data_loader, 1):
        if step <= 1500:
            u = step / 1500.0
            # L0: 1.9 -> 1.2
            t0 = 1.9 + (1.2 - 1.9) * u
            # L1: 1.30 -> 1.10
            t1 = 1.30 + (1.10 - 1.30) * u
            try:
                model.decoder.layers[0].self_attn.attn_temperature = float(t0)
                model.decoder.layers[1].self_attn.attn_temperature = float(t1)
            except Exception:
                pass

        if step == 1 and getattr(model.cfg, "use_ascender", False):
            for li in range(min(2, len(model.decoder.layers))):
                m = model.decoder.layers[li].self_attn
                print(
                    f"[ASC wire][L{li}] biaser={type(m.biaser).__name__ if m.biaser is not None else None}, "
                    f"r={getattr(m, 'std_match_ratio', None)}, "
                    f"tau={getattr(m, 'attn_temperature', None)}, "
                    f"topk={getattr(m, 'sparsify_k_frac', None)}"
                )

        src, tgt_inp, tgt_out = batch
        src, tgt_inp, tgt_out = src.to(device), tgt_inp.to(device), tgt_out.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(src, tgt_inp)
        if torch.isnan(logits).any():
            print(f"⚠️ NaN logits at step {step}")
            continue

        loss = criterion(logits, tgt_out)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf loss at step {step}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # 스케줄러는 모든 group에 적용되므로, step 후 ASC LR를 고정으로 되돌림(원하면)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 and getattr(model.cfg, "use_ascender", False):
            # L0: mean|Δp| 타깃을 살짝↑
            keep_delta_p_band(model, layer_idx=0, target=0.0045, tol=0.0007,
                            up=1.08, down=0.97, r_min=0.2, r_max=3.00)
            # L1: 약하게
            keep_delta_p_band(model, layer_idx=1, target=0.0025, tol=0.0005,
                            up=1.06, down=0.98, r_min=0.2, r_max=2.2)

        if hasattr(model, "_asc_lr") and ASC_IDX < len(optimizer.param_groups):
            optimizer.param_groups[ASC_IDX]["lr"] = model._asc_lr  # ASC 고정 LR

        total_loss += float(loss.detach())
        valid_steps += 1

        if step % 100 == 0:
            g_gamma, g_gate = [], []
            for l in model.decoder.layers:
                b = getattr(l, "biaser_self", None)
                if b is None:
                    b = getattr(l.self_attn, "biaser", None) if hasattr(l, "self_attn") else None
                if b is None:
                    continue
                if getattr(b, "gamma_log", None) is not None and b.gamma_log.grad is not None:
                    g_gamma.append(b.gamma_log.grad.abs().mean().item())
                if getattr(b, "gate_param", None) is not None and b.gate_param.grad is not None:
                    g_gate.append(b.gate_param.grad.abs().mean().item())
            print(f"[ASC grad] gamma={(sum(g_gamma)/len(g_gamma)) if g_gamma else 0:.3e}, "
                  f"gate={(sum(g_gate)/len(g_gate)) if g_gate else 0:.3e}")
            print("[LR]", [pg["lr"] for pg in optimizer.param_groups])

        if step % LOG_EVERY == 0:
            print(f"Step {step:03d} | Loss {float(loss):.4f}")
            quick_ab_check(model, (src, tgt_inp, tgt_out), device)

            # Head-wise 상태
            for li in [0, 1]:
                biaser = getattr(model.decoder.layers[li], "biaser_self", None)
                if biaser is None or not hasattr(biaser, "gamma_log"):
                    continue
                g_h = torch.exp(biaser.gamma_log.detach()).clamp(max=float(biaser.cfg.gamma_cap))
                gt = None
                if getattr(biaser, "gate_param", None) is not None:
                    gt = torch.sigmoid(biaser.gate_param.detach())
                    gt = biaser.cfg.gate_floor + (1.0 - biaser.cfg.gate_floor) * gt
                    gt = torch.minimum(gt, torch.as_tensor(float(biaser.cfg.gate_ceiling), device=gt.device))
                def mm(x): return float(x.min()), float(x.median()), float(x.max())
                if gt is not None:
                    a, bmd, c = mm(gt); d, e, f = mm(g_h)
                    print(f"[ASC head][L{li}] gate(min/med/max)={a:.2f}/{bmd:.2f}/{c:.2f} | "
                          f"γ(min/med/max)={d:.2f}/{e:.2f}/{f:.2f}")

            if getattr(model.cfg, "use_ascender", False):
                b0 = getattr(model.decoder.layers[0], "biaser_self", None)
                if b0 is not None:
                    try:
                        if getattr(b0.cfg, "per_head_gate", False) and getattr(b0, "gate_param", None) is not None:
                            g = torch.sigmoid(b0.gate_param.detach()).cpu()
                            gmn, gmd, gmx = g.min().item(), g.median().item(), g.max().item()
                        else:
                            gmn = gmd = gmx = float(getattr(b0, "gate_effective", 0.0))
                        if getattr(b0.cfg, "per_head_scale", False) and getattr(b0, "gamma_log", None) is not None:
                            gam = torch.exp(b0.gamma_log.detach()).clamp(max=b0.cfg.gamma_cap).cpu()
                            amn, amd, amx = gam.min().item(), gam.median().item(), gam.max().item()
                        else:
                            ge = float(getattr(b0, "gamma_effective", 0.0))
                            amn = amd = amx = ge
                        print(f"[ASC head] gate(min/med/max)={gmn:.2f}/{gmd:.2f}/{gmx:.2f} | "
                              f"γ(min/med/max)={amn:.2f}/{amd:.2f}/{amx:.2f}")
                    except Exception as e:
                        print(f"[ASC head] log failed: {e}")
                if b0 is not None and hasattr(b0, "ema_ratio"):
                    try:
                        gamma_eff = getattr(b0, "gamma_effective", None)
                        if gamma_eff is None and hasattr(b0, "gamma"):
                            gamma_eff = float(b0.gamma.mean().item())
                    except Exception:
                        gamma_eff = None
                    try:
                        if hasattr(b0, "gate_effective"):
                            gate_eff = b0.gate_effective
                        elif hasattr(b0, "gate_param") and b0.gate_param is not None:
                            gate_eff = float(torch.sigmoid(b0.gate_param).mean().item())
                        else:
                            gate_eff = None
                    except Exception:
                        gate_eff = None
                    print(f"[ASC dbg] ratio(ema)={float(b0.ema_ratio):.3f} | "
                          f"γ={('None' if gamma_eff is None else f'{gamma_eff:.3f}')} | "
                          f"gate={('None' if gate_eff is None else f'{gate_eff:.3f}')}")

    avg_loss = total_loss / max(valid_steps, 1)

    # Heatmap (optional)
    if getattr(model.cfg, "use_ascender", False) and (epoch_idx is not None):
        try:
            os.makedirs("logs/heatmaps", exist_ok=True)
            first_layer = model.decoder.layers[0]
            if getattr(first_layer, "biaser_self", None) is not None:
                T = 20
                h = torch.zeros((1, T, model.cfg.d_model), device=device, dtype=torch.float32)
                qh = first_layer.self_attn._shape(first_layer.self_attn.q_proj(h))
                kh = first_layer.self_attn._shape(first_layer.self_attn.k_proj(h))
                bias = first_layer.biaser_self(qh, kh, pre_q=h, pre_k=h)[0, 0].detach().cpu()

                plt.figure(figsize=(5, 4))
                im = plt.imshow(bias, cmap="coolwarm", interpolation="nearest")
                plt.colorbar(im, label="Bias Value")
                plt.title(f"Decoder[0] Self-Attn Bias (Epoch {epoch_idx})")
                plt.xlabel("Key Position"); plt.ylabel("Query Position")
                plt.tight_layout()
                save_path = f"logs/heatmaps/bias_epoch_{epoch_idx:02d}.png"
                plt.savefig(save_path); plt.close()
                print(f"[Saved] Bias heatmap → {save_path}")
        except Exception as e:
            print(f"[Warning] Heatmap save failed: {e}")

    return avg_loss


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_raw = load_config(args.config)
    cfg_raw = SimpleNamespace(**cfg_raw)
    cfg_raw.dataset = SimpleNamespace(**cfg_raw.dataset)
    cfg_raw.experiment = SimpleNamespace(**cfg_raw.experiment)
    cfg_raw.model = SimpleNamespace(**cfg_raw.model)
    cfg_raw.model.asc_cfg = SimpleNamespace(**cfg_raw.model.asc_cfg)

    exp_cfg = cfg_raw.experiment
    seeds = getattr(exp_cfg, "seeds", [42])
    mode = getattr(cfg_raw, "mode", "additive")

    Transformer, TransformerConfig, LabelSmoothingLoss, NoamLR = load_transformer(mode)
    from src.models.ascender_bias import AscenderBiasConfig
    asc_cfg_obj = AscenderBiasConfig(**vars(cfg_raw.model.asc_cfg))

    model_kwargs = vars(cfg_raw.model).copy()
    model_kwargs["asc_cfg"] = asc_cfg_obj
    model_cfg = TransformerConfig(**model_kwargs)

    device = get_device()
    print(f"[Device] {device}")

    csv_path = "logs/results_summary.csv"
    os.makedirs("logs", exist_ok=True)

    # Data
    try:
        from src.data.wikitext_loader import get_dataloader
        train_loader, _ = get_dataloader(cfg_raw, split="train")
    except Exception as e:
        print(f"[WARN] get_dataloader failed ({e}). Falling back to dummy data.")
        def make_dummy_data(vocab_size: int, pad_id: int, batch_size: int, seq_len: int = 20, num_batches: int = 64):
            for _ in range(num_batches):
                src = torch.randint(1, vocab_size, (batch_size, seq_len))
                tgt_inp = torch.randint(1, vocab_size, (batch_size, seq_len))
                tgt_out = torch.randint(1, vocab_size, (batch_size, seq_len))
                src[:, -1] = pad_id
                tgt_inp[:, -1] = pad_id
                tgt_out[:, -1] = pad_id
                yield (src, tgt_inp, tgt_out)
        bs = cfg_raw.dataset.batch_size
        vs = cfg_raw.dataset.vocab_size
        pad = model_cfg.pad_id
        train_loader = list(make_dummy_data(vs, pad, bs, seq_len=cfg_raw.dataset.seq_len, num_batches=64))

    all_losses = []

    for seed in seeds:
        torch.manual_seed(seed)
        print("\n==============================")
        print(f"🚀 Starting training for seed={seed}")
        print("==============================")

        model = Transformer(model_cfg).to(device)

        # ★ 여기서 YAML 설정을 존중한다: (덮어쓰기 금지)
        #   - use_auto_calibrate, use_gate 등을 강제로 끄지 않음
        #   - 단, MHA 튜닝(r/τ/topk)은 필요 시만 지정
        try:
            m0 = model.decoder.layers[0].self_attn
            m1 = model.decoder.layers[1].self_attn

            # --- L0: r 살짝 상향, τ는 약간 낮춰 포화 완화(=스코어 영향↑) ---
            m0.std_match_ratio = getattr(m0, "std_match_ratio", 2.2)  # was ~2.0
            m0.attn_temperature = getattr(m0, "attn_temperature", 1.9)  # was ~2.2
            m0.sparsify_k_frac = getattr(m0, "sparsify_k_frac", 0.30)   # 유지

            # --- L1: r 소폭↑, τ 살짝↓, 희소화 약하게 ON ---
            m1.std_match_ratio = getattr(m1, "std_match_ratio", 1.55)  # was ~1.4
            m1.attn_temperature = getattr(m1, "attn_temperature", 1.30) # was ~1.4
            m1.sparsify_k_frac = getattr(m1, "sparsify_k_frac", 0.05)   # was 0.10 유지(없으면 켬)
        except Exception:
            pass

        # === Plan-B wiring override: use L0 only, disable L1 biaser ===
        try:
            # L0는 그대로(정렬 전용이면 L0만 씀)
            if len(model.decoder.layers) >= 1:
                l0 = model.decoder.layers[0]
                if getattr(l0, "biaser_self", None) is not None:
                    l0.self_attn.biaser = l0.biaser_self  # 확실히 self_attn 경로에 부착

            # L1 biaser 완전 비활성화 (모듈 속성 + MHA 경로 둘 다)
            if len(model.decoder.layers) >= 2:
                l1 = model.decoder.layers[1]
                l1.biaser_self = None
                l1.self_attn.biaser = None
                # 혹시 모를 cross 경로도 확실히 OFF
                if hasattr(l1, "biaser_cross"):
                    l1.biaser_cross = None
                if hasattr(l1, "cross_attn"):
                    l1.cross_attn.biaser = None
        except Exception:
            pass

        # 배선 검증 (있으면)
        try:
            from src.utils.debug_ascender import verify_ascender_wiring
            verify_ascender_wiring(model)
        except Exception as _e:
            print("[ASC VERIFY] skipped:", _e)

        # Optimizer: [0]=main_decay, [1]=main_no_decay, [2]=asc (wd=0, lr 별도)
        main_decay, main_nodc, asc_params = [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_asc = any(k in n for k in ["biaser", "gamma_log", "gate_param", "wA", "wS", "wC"])
            if is_asc:
                asc_params.append(p)
            else:
                is_ln = ("ln" in n.lower()) or ("layernorm" in n.lower()) or ("norm" in n.lower())
                is_bias = n.endswith(".bias")
                (main_nodc if (is_ln or is_bias) else main_decay).append(p)

        lr_base = float(exp_cfg.lr)
        lr_asc  = float(getattr(exp_cfg, "lr_asc", lr_base))  # 필요하면 별도 튜닝
        optimizer = torch.optim.AdamW(
            [
                {"params": main_decay, "lr": lr_base, "weight_decay": 0.01},
                {"params": main_nodc, "lr": lr_base, "weight_decay": 0.00},
                {"params": asc_params, "lr": lr_asc,  "weight_decay": 0.00},
            ],
            betas=(0.9, 0.98), eps=1e-9
        )
        model._asc_lr = lr_asc
        scheduler = NoamLR(optimizer, d_model=model_cfg.d_model, warmup_steps=exp_cfg.warmup_steps)
        criterion = LabelSmoothingLoss(model_cfg.tgt_vocab_size, exp_cfg.smoothing, ignore_index=model_cfg.pad_id)

        # Bias 조합 태그
        asc = model_cfg.asc_cfg
        combo = []
        def on(flag, w):
            return getattr(asc, flag, True) and float(getattr(asc, w, 0.0)) != 0.0
        if on("use_alignment", "w_align"): combo.append("A")
        if on("use_separation", "w_sep"):  combo.append("S")
        if on("use_cohesion",   "w_coh"):  combo.append("C")
        bias_combo = "+".join(combo) if combo else "None"

        for epoch in range(1, exp_cfg.epochs + 1):
            # 필요하면 epoch-based anneal을 켤 수 있음 (지금은 보수적 기본 유지)
            # if hasattr(asc, "target_ratio"):
            #     if epoch >= 8:  model.cfg.asc_cfg.target_ratio = 0.26
            #     elif epoch >= 5: model.cfg.asc_cfg.target_ratio = 0.24

            if getattr(model.cfg, "use_ascender", False) and epoch >= 2:
                for li in (0, 1):
                    b = getattr(model.decoder.layers[li], "biaser_self", None)
                    if b is not None and hasattr(b, "cfg"):
                        # YAML에 다른 값이 이미 있으면 그대로 두고, 없을 때만 살짝 내리기
                        if not hasattr(b.cfg, "gate_ceiling") or b.cfg.gate_ceiling is None:
                            b.cfg.gate_ceiling = 0.65
                        else:
                            # 이미 설정된 상한이 0.7 이상이면 0.65로 보수적으로 낮춤
                            if b.cfg.gate_ceiling > 0.65:
                                b.cfg.gate_ceiling = 0.65

            print(f"\n🧭 Epoch {epoch}/{exp_cfg.epochs} | seed={seed}")
            avg_loss = run_epoch(
                model, train_loader, optimizer, scheduler, criterion,
                device, exp_cfg.clip_grad, epoch_idx=epoch, ASC_IDX=2
            )
            print(f"✅ Epoch {epoch} done. AvgLoss={avg_loss:.4f}")

            all_losses.append(avg_loss)
            log_result(csv_path, {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": mode,
                "use_ascender": model_cfg.use_ascender,
                "bias_combo": bias_combo,
                "seed": seed,
                "epoch": epoch,
                "avg_loss": avg_loss,
            })

        print(f"🏁 Finished seed={seed}")

    # Save metrics
    metrics_dir = f"logs/{mode}_logs"
    os.makedirs(metrics_dir, exist_ok=True)
    torch.save({"losses": all_losses}, f"{metrics_dir}/metrics.pt")
    print(f"✅ Saved metrics.pt → {metrics_dir}/metrics.pt")

    # Final bias snapshot (optional)
    if getattr(model_cfg, "use_ascender", False):
        print("\n[DEBUG] Checking one sample Ascender bias matrix stats...")
        first_layer = model.decoder.layers[0]
        if getattr(first_layer, "biaser_self", None) is not None:
            T = 20
            h = torch.zeros((1, T, model.cfg.d_model), device=device, dtype=torch.float32)
            qh = first_layer.self_attn._shape(first_layer.self_attn.q_proj(h))
            kh = first_layer.self_attn._shape(first_layer.self_attn.k_proj(h))
            bias = first_layer.biaser_self(qh, kh, pre_q=h, pre_k=h)[0, 0].detach().cpu()

            os.makedirs("logs/heatmaps", exist_ok=True)
            plt.figure(figsize=(5, 4))
            im = plt.imshow(bias, cmap="coolwarm", interpolation="nearest")
            plt.colorbar(im, label="Bias Value")
            plt.title("Decoder[0] Self-Attn Bias Heatmap")
            plt.xlabel("Key Position"); plt.ylabel("Query Position")
            plt.tight_layout()
            save_path = "logs/heatmaps/bias_final.png"
            plt.savefig(save_path); plt.close()
            print(f"[Saved] Final bias heatmap → {save_path}")
            print(f"  Bias stats — mean={float(bias.mean()):.4f}, std={float(bias.std()):.4f}, "
                  f"min={float(bias.min()):.4f}, max={float(bias.max()):.4f}")

    print("\nTraining complete ✅")


if __name__ == "__main__":
    main()
