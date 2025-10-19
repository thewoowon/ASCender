import os
import yaml
import torch
import matplotlib.pyplot as plt
from src.train import (
    load_config, get_device, load_transformer
)
from torch.nn.utils import clip_grad_norm_
from src.data.wikitext_loader import get_dataloader

plt.style.use("seaborn-v0_8")

def run_training(mode: str, cfg_path: str, save_dir: str):
    cfg_raw = load_config(cfg_path)
    cfg_raw["mode"] = mode
    exp_cfg = cfg_raw["experiment"]
    model_cfg = cfg_raw["model"]

    Transformer, TransformerConfig, LabelSmoothingLoss, NoamLR = load_transformer(mode)
    model_cfg = TransformerConfig(**model_cfg)

    device = get_device()
    print(f"\n[Running {mode.upper()}] Device={device}")

    torch.manual_seed(42)
    model = Transformer(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_cfg["lr"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, d_model=model_cfg.d_model, warmup_steps=exp_cfg["warmup_steps"])
    criterion = LabelSmoothingLoss(model_cfg.tgt_vocab_size, exp_cfg["smoothing"], ignore_index=model_cfg.pad_id)

    train_loader, train_ds = get_dataloader(cfg_raw, split="train")

    os.makedirs(f"{save_dir}/{mode}_bias_maps", exist_ok=True)
    os.makedirs(f"{save_dir}/{mode}_logs", exist_ok=True)

    losses, grad_norms, bias_stats = [], [], []

    for epoch in range(1, exp_cfg["epochs"] + 1):
        model.train()
        total_loss, total_grad, valid_steps = 0.0, 0.0, 0

        for step, (src, tgt_inp, tgt_out) in enumerate(train_loader, 1):
            src, tgt_inp, tgt_out = src.to(device), tgt_inp.to(device), tgt_out.to(device)
            optimizer.zero_grad()
            logits = model(src, tgt_inp)
            loss = criterion(logits, tgt_out)

            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), exp_cfg["clip_grad"])
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.detach())
            total_grad += float(grad_norm)
            valid_steps += 1

        avg_loss = total_loss / max(valid_steps, 1)
        avg_grad = total_grad / max(valid_steps, 1)
        losses.append(avg_loss)
        grad_norms.append(avg_grad)

        print(f"✅ [{mode}] Epoch {epoch}: Loss={avg_loss:.4f}, GradNorm={avg_grad:.3f}")

        # Bias heatmap 저장 (첫 디코더 레이어)
        try:
            first_layer = model.decoder.layers[0]
            if getattr(first_layer, "biaser_self", None) is not None:
                T = 20
                h = torch.zeros((1, T, model.cfg.d_model), device=device)
                qh = first_layer.self_attn._shape(first_layer.self_attn.q_proj(h))
                kh = first_layer.self_attn._shape(first_layer.self_attn.k_proj(h))
                bias = first_layer.biaser_self(qh, kh, pre_q=h, pre_k=h)[0, 0].detach().cpu()

                stats = {
                    "mean": float(bias.mean()), "std": float(bias.std()),
                    "min": float(bias.min()), "max": float(bias.max())
                }
                bias_stats.append(stats)

                plt.figure(figsize=(5, 4))
                plt.imshow(bias, cmap="coolwarm", interpolation="nearest")
                plt.colorbar(label="Bias Value")
                plt.title(f"{mode.upper()} Bias (Epoch {epoch})")
                plt.xlabel("Key Position"); plt.ylabel("Query Position")
                plt.tight_layout()
                plt.savefig(f"{save_dir}/{mode}_bias_maps/bias_epoch_{epoch:02d}.png")
                plt.close()
        except Exception as e:
            print(f"[Warning] Heatmap save failed: {e}")

    # bias 통계 저장
    torch.save({
        "losses": losses,
        "grad_norms": grad_norms,
        "bias_stats": bias_stats
    }, f"{save_dir}/{mode}_logs/metrics.pt")

    return losses, grad_norms, bias_stats


def plot_comparisons(results_add, results_mul, save_dir):
    losses_a, grads_a, bias_a = results_add
    losses_m, grads_m, bias_m = results_mul

    os.makedirs(save_dir, exist_ok=True)

    # 1️⃣ Loss curve
    plt.figure(figsize=(7,5))
    plt.plot(losses_a, label="Additive", marker="o")
    plt.plot(losses_m, label="Multiplicative", marker="s")
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_comparison.png")
    plt.close()

    # 2️⃣ Gradient norm
    plt.figure(figsize=(7,5))
    plt.plot(grads_a, label="Additive", marker="o")
    plt.plot(grads_m, label="Multiplicative", marker="s")
    plt.title("Gradient Norm Comparison")
    plt.xlabel("Epoch"); plt.ylabel("Grad Norm"); plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/grad_comparison.png")
    plt.close()

    # 3️⃣ Bias mean/std evolution
    if bias_a and bias_m:
        plt.figure(figsize=(7,5))
        plt.plot([b["mean"] for b in bias_a], label="Additive Mean", marker="o")
        plt.plot([b["mean"] for b in bias_m], label="Multiplicative Mean", marker="s")
        plt.title("Bias Mean Evolution")
        plt.xlabel("Epoch"); plt.ylabel("Mean Bias Value"); plt.legend(); plt.grid()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/bias_mean_comparison.png")
        plt.close()

        plt.figure(figsize=(7,5))
        plt.plot([b["std"] for b in bias_a], label="Additive Std", marker="o")
        plt.plot([b["std"] for b in bias_m], label="Multiplicative Std", marker="s")
        plt.title("Bias Std Evolution")
        plt.xlabel("Epoch"); plt.ylabel("Std Dev"); plt.legend(); plt.grid()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/bias_std_comparison.png")
        plt.close()

    print(f"📊 Saved all comparisons → {save_dir}")


def compare_modes_plus(cfg_path="configs/ascender_test.yaml"):
    save_dir = "logs/compare_full"
    os.makedirs(save_dir, exist_ok=True)

    print("🚀 Running ADDITIVE baseline...")
    results_add = run_training("additive", cfg_path, save_dir)

    print("\n🚀 Running MULTIPLICATIVE ASCender...")
    results_mul = run_training("multiplicative", cfg_path, save_dir)

    plot_comparisons(results_add, results_mul, save_dir)


if __name__ == "__main__":
    compare_modes_plus("configs/ascender_test.yaml")
