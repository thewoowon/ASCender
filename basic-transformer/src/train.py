import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.models.multiplicative_transformer import Transformer, TransformerConfig, LabelSmoothingLoss, NoamLR


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def make_dummy_data(vocab_size: int, pad_id: int, batch_size: int, seq_len: int = 20, num_batches: int = 64):
    data = []
    for _ in range(num_batches):
        src = torch.randint(1, vocab_size, (batch_size, seq_len))
        tgt_inp = torch.randint(1, vocab_size, (batch_size, seq_len))
        tgt_out = torch.randint(1, vocab_size, (batch_size, seq_len))
        src[:, -1] = pad_id
        tgt_inp[:, -1] = pad_id
        tgt_out[:, -1] = pad_id
        data.append((src, tgt_inp, tgt_out))
    return data


def run_epoch(model, data, optimizer, scheduler, criterion, device, clip_grad: float,epoch_idx=None):
    model.train()
    total_loss = 0.0
    valid_steps = 0

    for step, (src, tgt_inp, tgt_out) in enumerate(data, 1):
        src, tgt_inp, tgt_out = src.to(device), tgt_inp.to(device), tgt_out.to(device)
        optimizer.zero_grad()

        logits = model(src, tgt_inp)
        if torch.isnan(logits).any():
            print(f"⚠️ NaN logits at step {step}")
            continue

        loss = criterion(logits, tgt_out)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN loss detected at step {step}")
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
    if model.cfg.use_ascender and (epoch_idx is not None):
        try:
            import matplotlib.pyplot as plt
            import os
            os.makedirs("logs/heatmaps", exist_ok=True)

            first_layer = model.decoder.layers[0]
            if first_layer.biaser_self is not None:
                T = 20  # token length (dataset에 맞게)
                h = torch.zeros((1, T, model.cfg.d_model), device=device)
                qh = first_layer.self_attn._shape(first_layer.self_attn.q_proj(h))
                kh = first_layer.self_attn._shape(first_layer.self_attn.k_proj(h))
                bias = first_layer.biaser_self(qh, kh, pre_q=h, pre_k=h)[0, 0].detach().cpu()

                plt.style.use("seaborn-v0_8")
                plt.figure(figsize=(5, 4))
                plt.imshow(bias, cmap="coolwarm", interpolation="nearest")
                plt.colorbar(label="Bias Value")
                plt.title(f"Decoder[0] Self-Attn Bias (Epoch {epoch_idx+1})")
                plt.xlabel("Key Position")
                plt.ylabel("Query Position")
                plt.tight_layout()
                save_path = f"logs/heatmaps/bias_epoch_{epoch_idx+1:02d}.png"
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
    exp_cfg = cfg_raw["experiment"]
    model_cfg = TransformerConfig(**cfg_raw["model"])

    # --- Device ---
    device = get_device()
    print(f"\nUsing device: {device}")
    print(f"[Init] use_ascender={model_cfg.use_ascender}\n")

    # --- Model ---
    torch.manual_seed(42)
    model = Transformer(model_cfg).to(device)

    # --- Optimizer / Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_cfg["lr"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, d_model=model_cfg.d_model, warmup_steps=exp_cfg["warmup_steps"])
    criterion = LabelSmoothingLoss(model_cfg.tgt_vocab_size, exp_cfg["smoothing"], ignore_index=model_cfg.pad_id)

    # --- Dummy Data (or replace with real loader) ---
    data = make_dummy_data(model_cfg.src_vocab_size, model_cfg.pad_id, exp_cfg["batch_size"], seq_len=20)

    # --- Training Loop ---
    for epoch in range(1, exp_cfg["epochs"] + 1):
        print(f"\n🧭 Epoch {epoch}/{exp_cfg['epochs']}")
        avg_loss = run_epoch(model, data, optimizer, scheduler, criterion, device, exp_cfg["clip_grad"], epoch_idx=epoch)
        print(f"✅ Epoch {epoch} done. AvgLoss={avg_loss:.4f}")

    # --- Optional Bias Debug Info ---
    if model_cfg.use_ascender:
        print("\n[DEBUG] Checking one sample Ascender bias matrix stats...")
        first_layer = model.decoder.layers[0]
        if first_layer.biaser_self is not None:
            # 샘플 토큰 길이 정의 (20이 아니면 데이터셋 길이에 맞게)
            T = 20
            # 임의의 qh, kh를 biaser_self에 전달
            h = torch.zeros((1, T, model.cfg.d_model), device=device)
            qh = first_layer.self_attn._shape(first_layer.self_attn.q_proj(h))
            kh = first_layer.self_attn._shape(first_layer.self_attn.k_proj(h))
            bias = first_layer.biaser_self(qh, kh, pre_q=h, pre_k=h)[0, 0].detach().cpu()  # (T,T)
            
            import matplotlib.pyplot as plt
            plt.style.use("seaborn-v0_8")
            plt.figure(figsize=(5, 4))
            plt.imshow(bias, cmap="coolwarm", interpolation="nearest")
            plt.colorbar(label="Bias Value")
            plt.title("Decoder[0] Self-Attn Bias Heatmap")
            plt.xlabel("Key Position")
            plt.ylabel("Query Position")
            plt.tight_layout()
            plt.show()

            print(f"  Bias stats — mean={bias.mean():.4f}, std={bias.std():.4f}, "
                f"min={bias.min():.4f}, max={bias.max():.4f}")

    print("\nTraining complete ✅")


if __name__ == "__main__":
    main()
