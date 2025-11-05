import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def visualize_attention_entropy(model, save_path="logs/heatmaps/attn_entropy.png"):
    """
    디코더 self-attention의 헤드별 평균 엔트로피를 시각화.
    각 디코더 레이어별 (head × layer) heatmap 출력.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    entropy_per_layer = []

    with torch.no_grad():
        for li, layer in enumerate(model.decoder.layers):
            attn = getattr(layer.self_attn, "last_attn", None)  # (B, H, T, S)
            if attn is None:
                print(f"[Warn] No stored attention map for layer {li}")
                continue

            # 확률 안정화 (NaN 회피)
            attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

            # 엔트로피 계산: H = -sum(p log p)
            entropy = - (attn * (attn.clamp(min=1e-8).log())).sum(dim=-1)  # (B, H, T)
            entropy = entropy.mean(dim=(0, 2))  # (H,) — 배치, 시퀀스 평균
            entropy_per_layer.append(entropy.cpu())

    if not entropy_per_layer:
        print("[Warn] No attention maps recorded.")
        return

    entropy_tensor = torch.stack(entropy_per_layer)  # (L, H)
    plt.figure(figsize=(8, 4))
    plt.imshow(entropy_tensor, cmap="viridis", aspect="auto")
    plt.colorbar(label="Average Attention Entropy")
    plt.title("Decoder Self-Attention Entropy per Head/Layer")
    plt.xlabel("Head Index")
    plt.ylabel("Decoder Layer")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[Saved] Head-wise attention entropy heatmap → {save_path}")
