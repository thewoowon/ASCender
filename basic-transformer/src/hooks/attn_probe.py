# hooks/attn_probe.py (또는 transformer.py 상단 정의부)
import torch
import torch.nn.functional as F

class AttnProbe:
    def __init__(self, name="", every=50):
        self.name = name
        self.every = every
        self.cache = {}
        self._step = 0
    def __call__(self, module, input, output):
        self._step += 1
        if self._step % self.every != 0:
            return

        pre    = getattr(module, "attn_pre_masked", None)   # (B,H,T,S)
        post   = getattr(module, "attn_post_masked", None)  # (B,H,T,S)
        bias   = getattr(module, "probe_bias_snapshot", None)
        logits = getattr(module, "attn_logits", None)       # (B,H,T,S)
        all_masked = getattr(module, "probe_all_masked", None)  # (B,H,T,1) or None

        if pre is None or post is None or logits is None:
            print(f"[Probe:{self.name}] missing masked snapshots")
            return

        # 수치 안전을 위해 clamp (scores는 이미 clamp되어 있지만 확실하게)
        pre_c  = pre.clamp(-80, 80)
        post_c = post.clamp(-80, 80)

        # === 유효 행(row) 선별 ===
        B,H,T,S = pre_c.shape
        row_mask = torch.ones((B,H,T), dtype=torch.bool, device=pre_c.device)
        if all_masked is not None:
            # 모든 key가 마스크된 행 제외
            row_mask &= ~all_masked.squeeze(-1)
        if bias is not None:
            # bias가 전부 0인 행(스파르시파이/마스크) 제외
            row_mask &= (bias.abs().amax(dim=-1) > 0) 

        # 행 마스크를 펼친 인덱스에 적용
        row_mask_flat = row_mask.reshape(-1)                # (B*H*T,)
        pre_rows  = pre_c.reshape(B*H*T, S)[row_mask_flat]  # (R,S)
        post_rows = post_c.reshape(B*H*T, S)[row_mask_flat] # (R,S 

        if pre_rows.numel() == 0:
            print(f"[Probe:{self.name}] no valid rows")
            return

        # KL(pre||post) on same support
        p0 = F.softmax(pre_rows,  dim=-1)
        p1 = F.softmax(post_rows, dim=-1)
        kl = (p0 * (p0.clamp_min(1e-12).log() - p1.clamp_min(1e-12).log())).sum(-1).mean()

        # mean|Δp|
        mean_abs_dp = (p1 - p0).abs().mean()

        # === logits_std: 유한값만 집계 (attn_logits에는 -inf가 들어있을 수 있음)
        finite = torch.isfinite(logits)
        logits_std = logits[finite].std().item() if finite.any() else float("nan")

        self.cache = {
            "pre_std":  float(pre_c.std().item()),
            "bias_std": float(bias.std().item()) if bias is not None else None,
            "post_std": float(post_c.std().item()),
            "logits_std": float(logits_std),
            "bias_ratio": round(float(bias.std().item()) / (pre_c.std().item() + 1e-12), 6) if bias is not None else None,
            "KL_same_mask": float(kl.item()),
            "mean|Δp|": float(mean_abs_dp.item()),
        }
        print(f"[Probe:{self.name}] {self.cache}")
