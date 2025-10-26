import csv
import os
import argparse
import yaml
import torch
import importlib
from datetime import datetime
from types import SimpleNamespace

# === optional: 안전한 non-interactive backend (서버/CLI 환경)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def log_result(csv_path, fields):
    header = ["timestamp", "mode", "use_ascender", "bias_combo", "seed", "epoch", "avg_loss"]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    newfile = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if newfile:
            writer.writeheader()
        writer.writerow(fields)


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
    # CUDA 우선 → MPS → CPU 순서를 권장 (성능면에서)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def anneal_ascender(model, cur_step, total_steps):
    # 스케줄 파라미터
    ramp = 0.4               # 초반 비율(40%)은 강하게
    t = cur_step / max(1, total_steps)
    # target_ratio: 0.30 -> 0.06
    hi, lo = 0.30, 0.06
    if t <= ramp:
        tr = hi
        gc = 1.00            # gate ceiling
    else:
        u = (t - ramp) / (1 - ramp)
        tr = hi + (lo - hi) * 0.5*(1 - math.cos(math.pi*u))   # cosine
        gc = 1.00 + (0.70 - 1.00) * u                         # 1.0 -> 0.70

    # γ cap도 살짝 내리기: 4.0 -> 2.5
    gcap_hi, gcap_lo = 4.0, 2.5
    gcap = gcap_hi + (gcap_lo - gcap_hi) * max(0.0, (t - ramp)/(1 - ramp))

    # 모델에 반영
    for dec_l in model.decoder.layers[:2]:
        b = getattr(dec_l, "biaser_self", None)
        if b is None: continue
        b.cfg.target_ratio = float(tr)
        if hasattr(b.cfg, "gate_ceiling"):
            b.cfg.gate_ceiling = float(gc)
        b.cfg.gamma_cap = float(gcap)



def run_epoch(model, data_loader, optimizer, scheduler, criterion, device, clip_grad: float, epoch_idx=None):
    model.train()
    total_loss = 0.0
    valid_steps = 0

    LOG_EVERY = 10  # ← 원하는 주기

    for step, batch in enumerate(data_loader, 1):
        # get_dataloader()가 (src, tgt_inp, tgt_out) 튜플(batch)을 반환한다고 가정
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
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.detach())
        valid_steps += 1

        if step % 200 == 0 and getattr(model.cfg, "use_ascender", False):
            model.eval(); 
            with torch.no_grad():
                logits_on = model(src, tgt_inp); loss_on = criterion(logits_on, tgt_out).item()
                saved = []
                for L in model.decoder.layers[:2]:
                    b=L.biaser_self
                    if b is not None and hasattr(b,"gate_param"):
                        saved.append(b.gate_param.detach().clone()); b.gate_param.fill_(-99.0)
                logits_off = model(src, tgt_inp); loss_off = criterion(logits_off, tgt_out).item()
                print(f"[AB] Δloss(ON-OFF)={loss_on - loss_off:+.4f}")
                for (L,w) in zip(model.decoder.layers[:2], saved):
                    L.biaser_self.gate_param.data.copy_(w)
            model.train()


        if step % LOG_EVERY == 0:
            print(f"Step {step:03d} | Loss {float(loss):.4f}")

            # ==== ASCender head-wise quick log ====
            # 여러 레이어 로깅: L0/L1 구분
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
                    a,b,c = mm(gt); d,e,f = mm(g_h)
                    print(f"[ASC head][L{li}] gate(min/med/max)={a:.2f}/{b:.2f}/{c:.2f} | γ(min/med/max)={d:.2f}/{e:.2f}/{f:.2f}")



            # ASC dbg도 동일 주기로만 찍기
            if getattr(model.cfg, "use_ascender", False):
                b = getattr(model.decoder.layers[0], "biaser_self", None)
                if b is not None:
                    try:
                        if getattr(b.cfg, "per_head_gate", False) and getattr(b, "gate_param", None) is not None:
                            g = torch.sigmoid(b.gate_param.detach()).cpu()
                            gmn, gmd, gmx = g.min().item(), g.median().item(), g.max().item()
                        else:
                            gmn = gmd = gmx = float(b.gate_effective or 0.0)

                        if getattr(b.cfg, "per_head_scale", False) and getattr(b, "gamma_log", None) is not None:
                            gam = torch.exp(b.gamma_log.detach()).clamp(max=b.cfg.gamma_cap).cpu()
                            amn, amd, amx = gam.min().item(), gam.median().item(), gam.max().item()
                        else:
                            ge = float(b.gamma_effective)
                            amn = amd = amx = ge

                        print(f"[ASC head] gate(min/med/max)={gmn:.2f}/{gmd:.2f}/{gmx:.2f} | "
                            f"γ(min/med/max)={amn:.2f}/{amd:.2f}/{amx:.2f}")
                    except Exception as e:
                        print(f"[ASC head] log failed: {e}")

                if b is not None and hasattr(b, "ema_ratio"):
                    # 안전 추출 (구버전/신버전 모두 호환)
                    try:
                        gamma_eff = b.gamma_effective if hasattr(b, "gamma_effective") else float(
                            (b.gamma.mean() if hasattr(b, "gamma") else torch.tensor(0.0)).item()
                        )
                    except Exception:
                        gamma_eff = None
                    try:
                        if hasattr(b, "gate_effective"):
                            gate_eff = b.gate_effective
                        elif hasattr(b, "gate_param") and b.gate_param is not None:
                            gate_eff = float(torch.sigmoid(b.gate_param).mean().item())
                        else:
                            gate_eff = None
                    except Exception:
                        gate_eff = None

                    print(f"[ASC dbg] ratio(ema)={float(b.ema_ratio):.3f} | "
                          f"γ={('None' if gamma_eff is None else f'{gamma_eff:.3f}')} | "
                          f"gate={('None' if gate_eff is None else f'{gate_eff:.3f}')}")


    avg_loss = total_loss / max(valid_steps, 1)

    # 🔥 Bias heatmap 저장 (Ascender 활성 시)
    if getattr(model.cfg, "use_ascender", False) and (epoch_idx is not None):
        try:
            os.makedirs("logs/heatmaps", exist_ok=True)
            first_layer = model.decoder.layers[0]
            if getattr(first_layer, "biaser_self", None) is not None:
                T = 20  # 시각화 길이
                h = torch.zeros((1, T, model.cfg.d_model), device=device, dtype=torch.float32)
                qh = first_layer.self_attn._shape(first_layer.self_attn.q_proj(h))
                kh = first_layer.self_attn._shape(first_layer.self_attn.k_proj(h))
                bias = first_layer.biaser_self(qh, kh, pre_q=h, pre_k=h)[0, 0].detach().cpu()

                plt.figure(figsize=(5, 4))
                im = plt.imshow(bias, cmap="coolwarm", interpolation="nearest")
                plt.colorbar(im, label="Bias Value")
                plt.title(f"Decoder[0] Self-Attn Bias (Epoch {epoch_idx})")
                plt.xlabel("Key Position")
                plt.ylabel("Query Position")
                plt.tight_layout()
                save_path = f"logs/heatmaps/bias_epoch_{epoch_idx:02d}.png"
                plt.savefig(save_path)
                plt.close()
                print(f"[Saved] Bias heatmap → {save_path}")
        except Exception as e:
            print(f"[Warning] Heatmap save failed: {e}")

    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # --- Load Config ---
    cfg_raw = load_config(args.config)

    # dict→SimpleNamespace 변환
    cfg_raw = SimpleNamespace(**cfg_raw)
    cfg_raw.dataset = SimpleNamespace(**cfg_raw.dataset)
    cfg_raw.experiment = SimpleNamespace(**cfg_raw.experiment)
    cfg_raw.model = SimpleNamespace(**cfg_raw.model)
    cfg_raw.model.asc_cfg = SimpleNamespace(**cfg_raw.model.asc_cfg)

    exp_cfg = cfg_raw.experiment
    seeds = getattr(exp_cfg, "seeds", [42, 43, 44])
    mode = getattr(cfg_raw, "mode", "additive")  # 기본값 additive 권장

    # 모듈 로드 (주의: additive/multiplicative에 따라 import 대상 달라짐)
    Transformer, TransformerConfig, LabelSmoothingLoss, NoamLR = load_transformer(mode)

    # ✅ AscenderBiasConfig(dataclass)로 강제 변환
    #    - transformer.TransformerConfig가 AscenderBiasConfig 타입을 기대
    from src.models.ascender_bias import AscenderBiasConfig
    asc_cfg_obj = AscenderBiasConfig(**vars(cfg_raw.model.asc_cfg))

    # 최종 모델 설정 객체 생성
    model_kwargs = vars(cfg_raw.model).copy()
    model_kwargs["asc_cfg"] = asc_cfg_obj
    model_cfg = TransformerConfig(**model_kwargs)

    device = get_device()
    print(f"[Device] {device}")

    csv_path = "logs/results_summary.csv"
    os.makedirs("logs", exist_ok=True)

    # 데이터 로더
    try:
        from src.data.wikitext_loader import get_dataloader
        train_loader, _ = get_dataloader(cfg_raw, split="train")
    except Exception as e:
        print(f"[WARN] get_dataloader failed ({e}). Falling back to dummy data.")
        # 더미 데이터로 폴백 (배치 수는 대략 64 스텝)
        def make_dummy_data(vocab_size: int, pad_id: int, batch_size: int, seq_len: int = 20, num_batches: int = 64):
            for _ in range(num_batches):
                src = torch.randint(1, vocab_size, (batch_size, seq_len))
                tgt_inp = torch.randint(1, vocab_size, (batch_size, seq_len))
                tgt_out = torch.randint(1, vocab_size, (batch_size, seq_len))
                src[:, -1] = pad_id
                tgt_inp[:, -1] = pad_id
                tgt_out[:, -1] = pad_id
                yield (src, tgt_inp, tgt_out)
        # iterable로 사용
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

        # (기존) optimizer = torch.optim.AdamW(model.parameters(), lr=exp_cfg.lr, betas=(0.9, 0.98), eps=1e-9)
        # ──> (교체) biaser/LayerNorm/bias는 weight decay 0, 나머지는 decay
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_asc = ("biaser" in n) or ("gamma_log" in n) or ("gate_param" in n)
            is_ln  = ("ln" in n.lower()) or ("layernorm" in n.lower()) or ("norm" in n.lower())
            is_bias = n.endswith(".bias")
            if is_asc or is_ln or is_bias:
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0}],
            lr=exp_cfg.lr, betas=(0.9, 0.98), eps=1e-9
        )

        # optimizer = torch.optim.AdamW(model.parameters(), lr=exp_cfg.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamLR(optimizer, d_model=model_cfg.d_model, warmup_steps=exp_cfg.warmup_steps)
        criterion = LabelSmoothingLoss(model_cfg.tgt_vocab_size, exp_cfg.smoothing, ignore_index=model_cfg.pad_id)

        # Bias 조합 태그 (로그용)
        asc = model_cfg.asc_cfg
        
        combo = []
        def on(flag, w):
            return getattr(asc, flag, True) and float(getattr(asc, w, 0.0)) != 0.0
        if on("use_alignment", "w_align"):   combo.append("A")
        if on("use_separation", "w_sep"):    combo.append("S")
        if on("use_cohesion",   "w_coh"):    combo.append("C")
        bias_combo = "+".join(combo) if combo else "None"

        for epoch in range(1, exp_cfg.epochs + 1):
            if hasattr(asc, "target_ratio"):
                if epoch >= 8:  model.cfg.asc_cfg.target_ratio = 0.26
                elif epoch >= 5: model.cfg.asc_cfg.target_ratio = 0.24

            print(f"\n🧭 Epoch {epoch}/{exp_cfg.epochs} | seed={seed}")
            avg_loss = run_epoch(model, train_loader, optimizer, scheduler, criterion, device, exp_cfg.clip_grad, epoch_idx=epoch)
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

    # === Save summarized metrics for comparison ===
    metrics_dir = f"logs/{mode}_logs"
    os.makedirs(metrics_dir, exist_ok=True)
    torch.save({
        "losses": all_losses,
        "attn_entropies": [],
        "grad_norms": [],
        "bias_stats": []
    }, f"{metrics_dir}/metrics.pt")
    print(f"✅ Saved metrics.pt → {metrics_dir}/metrics.pt")

    # --- Optional Bias Debug Info (마지막 모델로 샘플) ---
    if getattr(model_cfg, "use_ascender", False):
        print("\n[DEBUG] Checking one sample Ascender bias matrix stats...")
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
            plt.title("Decoder[0] Self-Attn Bias Heatmap")
            plt.xlabel("Key Position")
            plt.ylabel("Query Position")
            plt.tight_layout()
            save_path = "logs/heatmaps/bias_final.png"
            os.makedirs("logs/heatmaps", exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            print(f"[Saved] Final bias heatmap → {save_path}")
            print(f"  Bias stats — mean={float(bias.mean()):.4f}, std={float(bias.std()):.4f}, "
                  f"min={float(bias.min()):.4f}, max={float(bias.max()):.4f}")

    print("\nTraining complete ✅")


if __name__ == "__main__":
    main()
