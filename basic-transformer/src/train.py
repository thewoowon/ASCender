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


def run_epoch(model, data_loader, optimizer, scheduler, criterion, device, clip_grad: float, epoch_idx=None):
    model.train()
    total_loss = 0.0
    valid_steps = 0

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

        if step % 10 == 0:
            print(f"Step {step:03d} | Loss {float(loss):.4f}")

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

        optimizer = torch.optim.AdamW(model.parameters(), lr=exp_cfg.lr, betas=(0.9, 0.98), eps=1e-9)
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
