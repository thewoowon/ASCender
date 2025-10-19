import os
import yaml
import torch
import matplotlib.pyplot as plt
from src.train import load_config, get_device, run_epoch, get_dataloader, load_transformer

def run_training(mode: str, cfg_path: str):
    # Load config and set mode
    cfg_raw = load_config(cfg_path)
    cfg_raw["mode"] = mode
    exp_cfg = cfg_raw["experiment"]
    model_cfg = cfg_raw["model"]

    # Load Transformer module dynamically
    Transformer, TransformerConfig, LabelSmoothingLoss, NoamLR = load_transformer(mode)
    model_cfg = TransformerConfig(**model_cfg)

    # Device
    device = get_device()
    print(f"\n[Running {mode.upper()}] Device={device}")

    # Model setup
    torch.manual_seed(42)
    model = Transformer(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_cfg["lr"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, d_model=model_cfg.d_model, warmup_steps=exp_cfg["warmup_steps"])
    criterion = LabelSmoothingLoss(model_cfg.tgt_vocab_size, exp_cfg["smoothing"], ignore_index=model_cfg.pad_id)

    # Dataset
    train_loader, dataset = get_dataloader(
        batch_size=exp_cfg["batch_size"],
        seq_len=exp_cfg.get("seq_len", 64),
        split="train"
    )

    # Train loop
    losses = []
    for epoch in range(1, exp_cfg["epochs"] + 1):
        avg_loss = run_epoch(model, train_loader, optimizer, scheduler, criterion, device, exp_cfg["clip_grad"], epoch_idx=epoch)
        losses.append(avg_loss)
        print(f"✅ [{mode}] Epoch {epoch} done. AvgLoss={avg_loss:.4f}")

    return losses


def compare_modes(cfg_path="configs/ascender_test.yaml"):
    os.makedirs("logs/compare", exist_ok=True)

    print("🚀 Running ADDITIVE baseline...")
    additive_losses = run_training("additive", cfg_path)

    print("\n🚀 Running MULTIPLICATIVE ASCENDER...")
    multiplicative_losses = run_training("multiplicative", cfg_path)

    # Plot comparison
    plt.figure(figsize=(7,5))
    plt.plot(additive_losses, label="Additive (Baseline)", marker="o")
    plt.plot(multiplicative_losses, label="Multiplicative (ASCender)", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Additive vs Multiplicative Training Loss (WikiText-2)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/compare/loss_comparison.png")
    plt.show()
    print("📊 Saved comparison → logs/compare/loss_comparison.png")


if __name__ == "__main__":
    compare_modes("configs/ascender_test.yaml")
