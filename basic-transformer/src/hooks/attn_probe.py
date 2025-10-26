# hooks/attn_probe.py (또는 transformer.py 상단 정의부)
import torch
import torch.nn.functional as F

class AttnProbe:
    def __init__(self, name=""):
        self.name = name
        self.cache = {}

    def __call__(self, module, input, output):
        # ✅ 같은 마스크 기준으로 저장해 둔 스냅샷 사용
        pre  = getattr(module, "attn_pre_masked", None)   # (B,H,T,S)
        post = getattr(module, "attn_post_masked", None)  # (B,H,T,S)
        bias = getattr(module, "attn_bias", None)
        logits = getattr(module, "attn_logits", None)

        if pre is None or post is None or logits is None:
            print(f"[Probe:{self.name}] missing masked snapshots")
            return

        # 수치 안전을 위해 clamp (scores는 이미 clamp되어 있지만 확실하게)
        pre_c  = pre.clamp(-80, 80)
        post_c = post.clamp(-80, 80)

        # KL(pre||post) on same support
        p0 = F.softmax(pre_c.reshape(-1, pre_c.size(-1)), dim=-1)
        p1 = F.softmax(post_c.reshape(-1, post_c.size(-1)), dim=-1)
        kl = (p0 * (p0.clamp_min(1e-12).log() - p1.clamp_min(1e-12).log())).sum(-1).mean()

        # mean|Δp|
        mean_abs_dp = (p1 - p0).abs().mean()

        self.cache = {
            "pre_std":  float(pre_c.std().item()),
            "bias_std": float(bias.std().item()) if bias is not None else None,
            "post_std": float(post_c.std().item()),
            "KL_same_mask": float(kl.item()),
            "mean|Δp|": float(mean_abs_dp.item()),
        }
        print(f"[Probe:{self.name}] {self.cache}")
