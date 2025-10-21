# ===============================
# train_hooks_v2.py (snippets to integrate into your loop)
# ===============================
"""
Usage notes:
- Replace your existing additive/multiplicative attention module with MultiHeadAttentionASCv2.
- Expose cfg.ascender.max_beta, cfg.ascender.beta_warmup_steps in your YAML.
- Log beta(t), entropy(A), and γ parameters periodically.
"""
import torch


def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    # attn: (B,H,T,T), return mean entropy per head
    eps = 1e-9
    p = attn.clamp_min(eps)
    H = -(p * (p + eps).log()).sum(dim=-1) # (B,H,T)
    return H.mean()


@torch.no_grad()
def log_ascender_stats(writer, step: int, attn: torch.Tensor, module: nn.Module, beta: float):
    ent = attention_entropy(attn).item()
    writer.add_scalar('asc/entropy', ent, step)
    # log gammas if available
    if hasattr(module, 'bias_mod'):
        bm = module.bias_mod
        for name in ['gamma_A', 'gamma_S', 'gamma_C']:
            if hasattr(bm, name):
                writer.add_scalar(f'asc/{name}', float(getattr(bm, name).data.item()), step)
    writer.add_scalar('asc/beta', beta, step)