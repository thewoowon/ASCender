import os
import yaml
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.train import (
    get_device, load_transformer
)
from torch.nn.utils import clip_grad_norm_
from src.data.wikitext_loader import get_dataloader
from types import SimpleNamespace

plt.style.use("seaborn-v0_8")

# =========================
# Utils
# =========================

def safe_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _compute_entropy(attn: torch.Tensor) -> float:
    """
    Shannon entropy of attention over last dim.
    attn: (B, H, T, S) with sum over S == 1
    """
    if attn is None or not torch.isfinite(attn).all():
        return 0.0
    attn = torch.clamp(attn, 1e-9, 1.0)
    ent = -(attn * torch.log(attn)).sum(dim=-1)  # (B,H,T)
    return float(ent.mean().detach().cpu())


def _try_get_attn_from_model(model, src, tgt_inp):
    """
    1) 우선 model(..., return_attn=True)을 시도
    2) 안 되면 None 반환 (fallback에서 수동 계산)
    """
    try:
        out = model(src, tgt_inp, return_attn=True)
        if isinstance(out, tuple) and len(out) >= 2:
            logits, attn_maps = out[0], out[1]  # attn_maps: List[(B,H,T,S)] or (B,H,T,S)
            return logits, attn_maps
        return out, None
    except TypeError:
        # 모델이 return_attn 인자를 지원하지 않는 경우
        logits = model(src, tgt_inp)
        return logits, None


def _manual_attn_first_decoder_layer(model, tokens: torch.Tensor):
    """
    첫 디코더 레이어의 self-attn q/k로 '근사' 점수→softmax→attn 계산
    - 정확한 마스크/바이어스는 모델 내부와 다를 수 있지만, 비교용 추세 확인엔 충분
    - 스케일은 반드시 √d_head 사용
    """
    try:
        first = model.decoder.layers[0]
        sa = first.self_attn
        # project -> shape
        qh = sa._shape(sa.q_proj(tokens))  # (B,H,T,dh)
        kh = sa._shape(sa.k_proj(tokens))  # (B,H,S,dh)
        d_head = qh.size(-1)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / (d_head ** 0.5)  # (B,H,T,S)
        scores = scores.clamp(-80, 80)
        attn = F.softmax(scores, dim=-1)
        return attn
    except Exception:
        return None


# =========================
# Core Runner
# =========================

def run_training(mode: str, cfg_path: str, save_dir: str):
    print(f"\n🚀 Running {mode.upper()} baseline...\n")

    # --- Load Config ---
    with open(cfg_path, "r") as f:
        cfg_raw = yaml.safe_load(f)

    # ✅ 1. SimpleNamespace 변환 (train.py와 동일하게)
    cfg_raw = SimpleNamespace(**cfg_raw)
    cfg_raw.dataset = SimpleNamespace(**cfg_raw.dataset)
    cfg_raw.experiment = SimpleNamespace(**cfg_raw.experiment)
    cfg_raw.model = SimpleNamespace(**cfg_raw.model)
    cfg_raw.model.asc_cfg = SimpleNamespace(**cfg_raw.model.asc_cfg)

    exp_cfg = cfg_raw.experiment
    model_cfg_raw = cfg_raw.model

    # ✅ 2. TransformerConfig 생성 (asc_cfg 포함)
    Transformer, TransformerConfig, LabelSmoothingLoss, NoamLR = load_transformer(mode)
    model_cfg = TransformerConfig(**vars(model_cfg_raw))

    device = get_device()
    print(f"[Running {mode.upper()}] Device={device}")
    print(f"[Init] use_ascender={model_cfg.use_ascender}")

    torch.manual_seed(42)
    model = Transformer(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp_cfg.lr,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    scheduler = NoamLR(optimizer, d_model=model_cfg.d_model, warmup_steps=exp_cfg.warmup_steps)
    criterion = LabelSmoothingLoss(model_cfg.tgt_vocab_size, exp_cfg.smoothing, ignore_index=model_cfg.pad_id)

    # ✅ 3. dataset loader
    train_loader, train_ds = get_dataloader(cfg_raw, split="train")

    os.makedirs(f"{save_dir}/{mode}_bias_maps", exist_ok=True)
    os.makedirs(f"{save_dir}/{mode}_logs", exist_ok=True)

    losses, grad_norms, bias_stats, attn_entropies = [], [], [], []

    log_path = f"{save_dir}/{mode}_logs/training_log.txt"
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Mode: {mode}\n\n")

        for epoch in range(1, exp_cfg.epochs + 1):
            model.train()
            total_loss, total_grad, total_entropy, valid_steps = 0.0, 0.0, 0.0, 0

            for step, (src, tgt_inp, tgt_out) in enumerate(train_loader, 1):
                src, tgt_inp, tgt_out = src.to(device), tgt_inp.to(device), tgt_out.to(device)
                optimizer.zero_grad()

                logits, attn_maps = _try_get_attn_from_model(model, src, tgt_inp)
                loss = criterion(logits, tgt_out)
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), exp_cfg.clip_grad)
                optimizer.step()
                scheduler.step()

                # --- Entropy 측정 ---
                try:
                    if attn_maps is not None:
                        ents = []
                        for a in attn_maps if isinstance(attn_maps, (list, tuple)) else [attn_maps]:
                            if a is not None and a.dim() == 4:
                                ents.append(_compute_entropy(a))
                        if ents:
                            total_entropy += sum(ents) / len(ents)
                    else:
                        approx = _manual_attn_first_decoder_layer(model, tgt_inp)
                        total_entropy += _compute_entropy(approx)
                except Exception:
                    pass

                total_loss += float(loss.detach())
                total_grad += float(grad_norm)
                valid_steps += 1

            avg_loss = total_loss / max(valid_steps, 1)
            avg_grad = total_grad / max(valid_steps, 1)
            avg_entropy = total_entropy / max(valid_steps, 1)
            losses.append(avg_loss)
            grad_norms.append(avg_grad)
            attn_entropies.append(avg_entropy)

            print(f"✅ [{mode}] Epoch {epoch}: Loss={avg_loss:.4f}, GradNorm={avg_grad:.3f}, Entropy={avg_entropy:.5f}")
            log_file.write(f"Epoch {epoch:02d} | Loss={avg_loss:.4f}, GradNorm={avg_grad:.3f}, Entropy={avg_entropy:.5f}\n")

            # --- Bias heatmap ---
            try:
                first = model.decoder.layers[0]
                if getattr(first, "biaser_self", None) is not None:
                    T = safe_get(cfg_raw.dataset, "seq_len", 64)
                    h = torch.zeros((1, T, model.cfg.d_model), device=device)
                    qh = first.self_attn._shape(first.self_attn.q_proj(h))
                    kh = first.self_attn._shape(first.self_attn.k_proj(h))
                    bias = first.biaser_self(qh, kh, pre_q=h, pre_k=h)[0, 0].detach().cpu()

                    stats = {
                        "mean": float(bias.mean()),
                        "std": float(bias.std()),
                        "min": float(bias.min()),
                        "max": float(bias.max()),
                        "range": float(bias.max() - bias.min())
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

    torch.save({
        "losses": losses,
        "grad_norms": grad_norms,
        "bias_stats": bias_stats,
        "attn_entropies": attn_entropies
    }, f"{save_dir}/{mode}_logs/metrics.pt")

    return losses, grad_norms, bias_stats, attn_entropies



# =========================
# Visualization
# =========================

def plot_advanced_comparisons(results_add, results_mul, save_dir):
    losses_a, grads_a, bias_a, ent_a = results_add
    losses_m, grads_m, bias_m, ent_m = results_mul

    os.makedirs(save_dir, exist_ok=True)

    # 1) Loss
    plt.figure(figsize=(7,5))
    plt.plot(losses_a, label="Additive", marker="o")
    plt.plot(losses_m, label="Multiplicative", marker="s")
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid()
    plt.tight_layout(); plt.savefig(f"{save_dir}/loss_comparison.png"); plt.close()

    # 2) Grad Norm
    plt.figure(figsize=(7,5))
    plt.plot(grads_a, label="Additive", marker="o")
    plt.plot(grads_m, label="Multiplicative", marker="s")
    plt.title("Gradient Norm Comparison")
    plt.xlabel("Epoch"); plt.ylabel("Grad Norm"); plt.legend(); plt.grid()
    plt.tight_layout(); plt.savefig(f"{save_dir}/grad_comparison.png"); plt.close()

    # 3) Bias stats
    if bias_a and bias_m:
        for metric, label in [("mean", "Mean"), ("std", "Std"), ("range", "Range")]:
            plt.figure(figsize=(7,5))
            plt.plot([b[metric] for b in bias_a], label=f"Additive {label}", marker="o")
            plt.plot([b[metric] for b in bias_m], label=f"Multiplicative {label}", marker="s")
            plt.title(f"Bias {label} Evolution")
            plt.xlabel("Epoch"); plt.ylabel(label); plt.legend(); plt.grid()
            plt.tight_layout(); plt.savefig(f"{save_dir}/bias_{metric}_comparison.png"); plt.close()

    # 4) Entropy
    if ent_a and ent_m:
        plt.figure(figsize=(7,5))
        plt.plot(ent_a, label="Additive", marker="o")
        plt.plot(ent_m, label="Multiplicative", marker="s")
        plt.title("Attention Entropy (Lower → More Focused)")
        plt.xlabel("Epoch"); plt.ylabel("Entropy"); plt.legend(); plt.grid()
        plt.tight_layout(); plt.savefig(f"{save_dir}/entropy_comparison.png"); plt.close()

    print(f"📊 All advanced comparisons saved → {save_dir}")


# =========================
# Main
# =========================

def compare_modes_plus_pro(cfg_path="configs/ascender_test.yaml"):
    save_dir = "logs/compare_pro"
    os.makedirs(save_dir, exist_ok=True)

    print("🚀 Running ADDITIVE baseline...")
    results_add = run_training("additive", cfg_path, save_dir)

    print("\n🚀 Running MULTIPLICATIVE ASCender...")
    results_mul = run_training("multiplicative", cfg_path, save_dir)

    plot_advanced_comparisons(results_add, results_mul, save_dir)


if __name__ == "__main__":
    compare_modes_plus_pro("configs/ascender_test.yaml")
