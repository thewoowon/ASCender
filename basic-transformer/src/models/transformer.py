# transformer.py
from __future__ import annotations
import math
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.architectural_mods import (
    MultiHeadAttentionWithResidualBias,
    GatedBiasAttention,
    MultiScaleBiasAttention,
    BiasConditionedValueAttention,
    HierarchicalBiasAttention,
)

from src.models.ascender_bias import AscenderBias, AscenderBiasConfig
from src.hooks.attn_probe import AttnProbe

# ============================================================
# Utilities: Probes & Masks
# ============================================================

def attach_probes(model, layers=(0,1)):
    # SAFE: store in plain list on the model, not as submodules
    handles = []
    refs = []
    for li in layers:
        if li < len(model.decoder.layers):
            mha = model.decoder.layers[li].self_attn
            probe = AttnProbe(f"decoder.self_attn.layer{li}")
            h = mha.register_forward_hook(probe)
            handles.append(h)
            refs.append((probe, h))
    model._probe_handles = handles
    model._probe_refs = refs  # keep strong refs (not modules), avoid GC

def detach_probes(model, layers=(0,1)):
    # remove handles; the plain Python refs will be GC'd
    if hasattr(model, "_probe_handles"):
        for h in model._probe_handles:
            try:
                h.remove()
            except Exception:
                pass
        model._probe_handles = []
    if hasattr(model, "_probe_refs"):
        model._probe_refs = []

def make_padding_mask(seq: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    seq: (B, S) int
    return: (B,1,1,S) bool — True = masked
    """
    assert seq.dim() == 2, f"Expected (B, S), got {seq.shape}"
    return (seq == pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    (1,1,T,T) bool — True = masked (future j>i)
    """
    m = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
    return m.unsqueeze(0).unsqueeze(0)


# ============================================================
# LR Scheduler — Noam
# ============================================================

class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Vaswani Noam LR:
      lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    def __init__(self, optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


# ============================================================
# Embeddings & Positional Encodings
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        s = x.size(1)
        return x + self.pe[start_pos:start_pos + s].unsqueeze(0)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


# ============================================================
# Core: Multi-Head Attention (ASC-ready)
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Standard pre-LN MHA with additive bias before softmax.

    Mask semantics:
      - attn_mask: (B,1,T,S) or (B,H,T,S), True=mask
      - attn_bias: (B,H,T,S) additive logits bias

    ASC extensions:
      - biaser: AscenderBias or None
      - std_match_ratio (r): match bias std to scores std
      - attn_temperature (tau)
      - sparsify_k_frac: keep top-k% |bias| along key dim
      - v_gain_epsilon: micro gain on V-path based on |bias|
      - runtime lock: lock_runtime_controls() — restore hyper snapshot before each fwd
      - enable_std_match: turn on/off score-std matching
      - bias_softcap: cap raw bias when std-match disabled
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # ASC params (public)
        self.biaser: Optional[Any] = None
        self.std_match_ratio: float = 1.0
        self.attn_temperature: float = 1.0
        self.sparsify_k_frac: float = 0.0
        self.v_gain_epsilon: float = 0.0

        # self.std_match_ratio: float = getattr(self, "std_match_ratio", 1.0)
        # self.attn_temperature: float = getattr(self, "attn_temperature", 1.0)
        # self.sparsify_k_frac: float = getattr(self, "sparsify_k_frac", 0.0)
        # self.v_gain_epsilon: float = getattr(self, "v_gain_epsilon", 0.0)

        # std-match & safety
        self.enable_std_match: bool = True
        self.bias_softcap: float = 6.0

        # ══════════════════════════════════════════════════════════════════
        # RESIDUAL BIAS PATH (architectural_mods.py Option 1)
        # ══════════════════════════════════════════════════════════════════
        # Solves the softmax problem: bias isn't overwhelmed by normalization!
        # Instead of: attn = softmax(scores + bias)
        # We compute: α * softmax(scores) + (1-α) * softmax(scores + bias)
        # Model learns per-head mixing weight α ∈ [0,1]
        # ══════════════════════════════════════════════════════════════════
        self.alpha_logit = nn.Parameter(torch.zeros(n_heads))  # per-head mixing
        self.enable_residual_path: bool = False  # Set True in config to activate

        # runtime control (lock)
        self._locked_runtime: bool = False
        self._init_runtime_snapshot = None
        self._freeze_after_lock = None
        self._warned_runtime_drift = False

        # one-off logs
        self._once_wire_log = False
        self._once_std_log = False

        # 신규: 스냅샷/프로브 토글
        self.capture_snapshots: bool = True

    # ---------- runtime helpers ----------
    def set_runtime_controls(self, *, tau=None, topk=None, r=None, v_gain_eps=None):
        if tau is not None:        self.attn_temperature = float(tau)
        if topk is not None:       self.sparsify_k_frac = float(topk)
        if r is not None:          self.std_match_ratio = float(r)
        if v_gain_eps is not None: self.v_gain_epsilon = float(v_gain_eps)

    def lock_runtime_controls(self):
        """
        Freeze current runtime hyperparameters; any later changes are reverted
        right before forward() uses them (source of truth = snapshot).
        """
        self._locked_runtime = True
        self._init_runtime_snapshot = (
            float(self.attn_temperature),
            float(self.sparsify_k_frac),
            float(self.std_match_ratio),
            float(self.v_gain_epsilon),
        )
        self._freeze_after_lock = {
            "attn_temperature": self.attn_temperature,
            "sparsify_k_frac":  self.sparsify_k_frac,
            "std_match_ratio":  self.std_match_ratio,
            "v_gain_epsilon":   self.v_gain_epsilon,
        }

    def _restore_locked_runtime(self):
        if self._locked_runtime and self._freeze_after_lock is not None:
            for k, v in self._freeze_after_lock.items():
                setattr(self, k, v)

    # ---------- shapes & masks ----------
    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,S,dh)

    @staticmethod
    def _expand_valid(mask_3d: torch.Tensor, like_4d: torch.Tensor) -> torch.Tensor:
        # mask_3d: (B,1,T,S) or (B,T,S), like_4d: (B,H,T,S)
        m = mask_3d
        if m.dim() == 3:
            m = m.unsqueeze(1)  # (B,1,T,S)
        if m.size(1) == 1:
            m = m.expand(like_4d.size(0), 1, like_4d.size(2), like_4d.size(3)).expand_as(like_4d)
        elif m.size(1) != like_4d.size(1):
            # head-mismatch: broadcast first head
            m = m[:, :1, :, :].expand_as(like_4d)
        return m

    def _masked_std_scores(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], per_head: bool) -> torch.Tensor:
        x = x.float().detach()
        if attn_mask is None:
            return x.std(dim=(0, 2, 3)).clamp_min(1e-6) if per_head else x.std().clamp_min(1e-6)
        vexp = self._expand_valid(~attn_mask, x).float()
        if per_head:
            num = vexp.sum(dim=(0, 2, 3)).clamp_min(1.0)
            mu  = (x * vexp).sum(dim=(0, 2, 3)) / num
            var = (((x - mu.view(1, -1, 1, 1)) ** 2) * vexp).sum(dim=(0, 2, 3)) / num
            return var.sqrt().clamp_min(1e-6)
        else:
            num = vexp.sum().clamp_min(1.0)
            mu  = (x * vexp).sum() / num
            var = (((x - mu) ** 2) * vexp).sum() / num
            return var.sqrt().clamp_min(1e-6)

    def _masked_std_bias(self, b: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        b = b.float()
        if attn_mask is None:
            return b.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        vexp = self._expand_valid(~attn_mask, b).float()
        num  = vexp.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        mu   = (b * vexp).sum(dim=(-2, -1), keepdim=True) / num
        var  = (((b - mu) ** 2) * vexp).sum(dim=(-2, -1), keepdim=True) / num
        return var.sqrt().clamp_min(1e-6)

    @staticmethod
    def _sparsify_last_dim(bias: torch.Tensor, k_frac: float, use_abs: bool = True) -> torch.Tensor:
        if not (0.0 < k_frac < 1.0):
            return bias
        B, H, T, S = bias.shape
        k = max(1, int(S * k_frac))
        sel = bias.abs() if use_abs else bias
        topv, topi = torch.topk(sel, k, dim=-1)
        mask = torch.zeros_like(bias, dtype=torch.bool).scatter_(-1, topi, True)
        return torch.where(mask, bias, torch.zeros_like(bias))

    @staticmethod
    def _broadcast_mask(attn_mask: Optional[torch.Tensor], like: torch.Tensor) -> Optional[torch.Tensor]:
        if attn_mask is None:
            return None
        return MultiHeadAttention._expand_valid(attn_mask, like)

    # ---------- forward ----------
    def forward(
        self,
        q: torch.Tensor,  # (B,T,d)
        k: torch.Tensor,  # (B,S,d)
        v: torch.Tensor,  # (B,S,d)
        attn_mask: Optional[torch.Tensor] = None,  # (B,1,T,S) or (B,H,T,S) True=mask
        attn_bias: Optional[torch.Tensor] = None,  # (B,H,T,S)
        *,
        pre_q: Optional[torch.Tensor] = None,
        pre_k: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # restore snapshot if locked
        self._restore_locked_runtime()

        B, T, _ = q.size()
        S = k.size(1)

        # one-off wire log
        if not self._once_wire_log:
            print(f"[ASC wire][{getattr(self, 'role', 'mha')}] "
                  f"biaser={type(self.biaser).__name__ if self.biaser else None}, "
                  f"r={self.std_match_ratio}, tau={self.attn_temperature}, "
                  f"topk={self.sparsify_k_frac}, v_eps={self.v_gain_epsilon}")
            self._once_wire_log = True

        # QKV
        qh = self._shape(self.q_proj(q))  # (B,H,T,dh)
        kh = self._shape(self.k_proj(k))  # (B,H,S,dh)
        vh = self._shape(self.v_proj(v))  # (B,H,S,dh)

        # logits
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,T,S)

        # temperature
        tau = float(getattr(self, "attn_temperature", 1.0))
        if tau != 1.0:
            scores = scores / tau

        scores = torch.nan_to_num(scores, nan=0.0, posinf=80.0, neginf=-80.0).clamp(-80, 80)
        self.attn_pre = scores.detach()

        # std of scores (valid positions only)
        per_head_mode = (self.biaser is not None) and (
            getattr(self.biaser.cfg, "per_head_scale", False) or getattr(self.biaser.cfg, "per_head_gate", False)
        )
        scores_std = self._masked_std_scores(scores, attn_mask, per_head=per_head_mode)

        if not self._once_std_log:
            if isinstance(scores_std, torch.Tensor):
                med = float(scores_std.median().item())
                print(f"[ASC std-check][{getattr(self,'role','mha')}] scores_std≈{med:.3f} "
                      f"(per-head={scores_std.numel()})")
            else:
                print(f"[ASC std-check][{getattr(self,'role','mha')}] scores_std={float(scores_std):.3f}")
            self._once_std_log = True

        # bias generation
        runtime_bias = None
        if self.biaser is not None:
            _pre_q = pre_q if pre_q is not None else q
            _pre_k = pre_k if pre_k is not None else k
            runtime_bias = self.biaser(qh, kh, pre_q=_pre_q, pre_k=_pre_k, scores_std=scores_std)
        elif attn_bias is not None:
            runtime_bias = attn_bias
        else:
            if getattr(self, "expect_bias", False) and not getattr(self, "_warned_no_bias", False):
                print("[MHA] No bias injected: biaser=None and attn_bias=None")
                self._warned_no_bias = True

        if runtime_bias is not None:
            runtime_bias = torch.nan_to_num(runtime_bias, nan=0.0, posinf=80.0, neginf=-80.0)
            assert runtime_bias.shape == scores.shape, f"bias {runtime_bias.shape} vs scores {scores.shape}"

            # mask out bias on padded/future (pre-add)
            if attn_mask is not None:
                mask_bh = self._broadcast_mask(attn_mask, runtime_bias)
                runtime_bias = runtime_bias.masked_fill(mask_bh, 0.0)

            # sparsify if requested
            k_frac = float(getattr(self, "sparsify_k_frac", 0.0))
            if 0.0 < k_frac < 1.0:
                runtime_bias = self._sparsify_last_dim(runtime_bias, k_frac=k_frac)

            # std-match normalization to target r * std(scores)
            if isinstance(scores_std, torch.Tensor):
                t_std = scores_std.view(1, -1, 1, 1)
            else:
                t_std = torch.tensor(scores_std, device=scores.device, dtype=scores.dtype).view(1, 1, 1, 1)
            t_std = t_std.detach().clamp_min(1e-6)

            if getattr(self, "enable_std_match", True):
                b_std = self._masked_std_bias(runtime_bias, attn_mask).detach().clamp_min(1e-6)
                if self.biaser is not None and getattr(self.biaser.cfg, "std_batch_mean", True):
                    b_std = b_std.mean(dim=0, keepdim=True)  # (1,H,1,1)
                r = float(getattr(self, "std_match_ratio", 1.0))
                runtime_bias = (runtime_bias / b_std) * (t_std * r)
            else:
                cap = float(getattr(self, "bias_softcap", 6.0))
                runtime_bias = runtime_bias.clamp(min=-cap, max=cap)

            # snapshots for logging
            self.attn_bias = runtime_bias
            self.probe_bias_snapshot = runtime_bias.detach()

            pre_for_probe = scores
            post_for_probe = scores + runtime_bias
            if attn_mask is not None:
                mask_bh = self._broadcast_mask(attn_mask, scores)
                pre_for_probe  = pre_for_probe.masked_fill(mask_bh, float("-inf"))
                post_for_probe = post_for_probe.masked_fill(mask_bh, float("-inf"))

            if getattr(self, "capture_snapshots", False):
                self.attn_pre_masked  = pre_for_probe.detach()
                self.attn_post_masked = post_for_probe.detach()
            else:
                self.attn_pre_masked = self.attn_post_masked = None

            # ══════════════════════════════════════════════════════════════
            # RESIDUAL BIAS PATH: Compute both paths if enabled
            # ══════════════════════════════════════════════════════════════
            if getattr(self, "enable_residual_path", False):
                # Path 1: Normal (unbiased) attention
                scores_normal = scores.clone()
                if attn_mask is not None:
                    mask_bh = self._broadcast_mask(attn_mask, scores_normal)
                    scores_normal = scores_normal.masked_fill(mask_bh, float("-inf"))
                attn_normal = F.softmax(scores_normal, dim=-1)
                attn_normal = self.dropout(attn_normal)

                # Path 2: Biased attention
                scores_biased = scores + runtime_bias
                if attn_mask is not None:
                    mask_bh = self._broadcast_mask(attn_mask, scores_biased)
                    scores_biased = scores_biased.masked_fill(mask_bh, float("-inf"))
                attn_biased = F.softmax(scores_biased, dim=-1)
                attn_biased = self.dropout(attn_biased)

                # Learnable per-head mixing: α ∈ [0,1]
                alpha = torch.sigmoid(self.alpha_logit).view(1, -1, 1, 1)  # (1,H,1,1)

                # Mix attention patterns
                attn = alpha * attn_normal + (1.0 - alpha) * attn_biased

                # Store for logging (use biased path for probes)
                self.attn_logits = scores_biased.detach()
                self.attn_probs = attn.detach()
                self._alpha_effective = alpha.detach().view(-1)  # (H,) for logging

            else:
                # Standard path: add bias directly
                scores = scores + runtime_bias
                if attn_mask is not None:
                    mask_bh = self._broadcast_mask(attn_mask, scores)
                    scores = scores.masked_fill(mask_bh, float("-inf"))
                self.attn_logits = scores.detach()
                attn = F.softmax(scores, dim=-1)
                attn = self.dropout(attn)
                self.attn_probs = attn.detach()
        else:
            # No bias case
            self.attn_bias = None
            if attn_mask is not None:
                pre_masked = scores.masked_fill(attn_mask if attn_mask.dim() == 4 else attn_mask.unsqueeze(1),
                                                float("-inf"))
            else:
                pre_masked = scores
            self.attn_pre_masked = pre_masked.detach()
            self.attn_post_masked = pre_masked.detach()

            # apply mask before softmax
            if attn_mask is not None:
                mask_bh = self._broadcast_mask(attn_mask, scores)
                scores = scores.masked_fill(mask_bh, float("-inf"))

            self.attn_logits = scores.detach()
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            self.attn_probs = attn.detach()

        # V-path micro gain
        if getattr(self, "v_gain_epsilon", 0.0) > 0.0 and getattr(self, "attn_bias", None) is not None:
            with torch.no_grad():
                b = self.attn_bias.detach().abs()    # (B,H,T,S)
                if attn_mask is not None:
                    mask_bh = self._broadcast_mask(attn_mask, b)
                    b = b.masked_fill(mask_bh, 0.0)
                m = b.mean(dim=2)                    # (B,H,S)
                denom = (m.mean(dim=-1, keepdim=True) + 1e-6)
                m_norm = (m / denom).unsqueeze(-1)   # (B,H,S,1)
                gain = 1.0 + float(self.v_gain_epsilon) * m_norm
            vh = vh * gain.detach()

        # output
        out = torch.matmul(attn, vh).transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)

        self.last_attn = attn.detach()
        return out, attn


# ============================================================
# Position-wise FFN
# ============================================================

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # <- ReLU → GELU for stability/perf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


# ============================================================
# Encoder / Decoder Layers
# ============================================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, biaser: Optional[nn.Module] = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.resid1_scale: float = 1.0

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.biaser: Optional[AscenderBias] = biaser
        self.self_attn.biaser = self.biaser

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.ln1(x)
        # Use (B,1,S,S) mask — both query/key pads masked for self-attn
        attn_out, _ = self.self_attn(h, h, h, attn_mask=src_mask, pre_q=h, pre_k=h)
        x = x + self.resid1_scale * self.dropout1(attn_out)

        h2 = self.ln2(x)
        x = x + self.dropout2(self.ffn(h2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 biaser_self: Optional[nn.Module] = None,
                 biaser_cross: Optional[nn.Module] = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.resid1_scale: float = 1.0

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resid2_scale: float = 1.0

        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.biaser_self: Optional[AscenderBias] = biaser_self
        self.biaser_cross: Optional[AscenderBias] = biaser_cross

        self.self_attn.biaser = self.biaser_self
        self.cross_attn.biaser = self.biaser_cross

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self-attention
        h = self.ln1(x)
        sa_out, _ = self.self_attn(h, h, h, attn_mask=tgt_mask, pre_q=h, pre_k=h)
        x = x + self.resid1_scale * self.dropout1(sa_out)

        # Cross-attention
        h2 = self.ln2(x)
        ca_out, _ = self.cross_attn(h2, memory, memory, attn_mask=memory_mask, pre_q=h2, pre_k=memory)
        x = x + self.resid2_scale * self.dropout2(ca_out)

        # FFN
        h3 = self.ln3(x)
        x = x + self.dropout3(self.ffn(h3))
        return x


# ============================================================
# Encoder / Decoder Stacks
# ============================================================

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 dropout: float, pad_id: int, max_len: int = 5000, layers: Optional[nn.ModuleList] = None):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = layers if layers is not None else nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.pad_id = pad_id

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        src: (B, S)
        returns:
          memory: (B,S,d)
          src_pad_mask: (B,1,1,S) — compact for cross-attn
        """
        x = self.dropout(self.pos_enc(self.tok_emb(src)))
        src_pad_mask = make_padding_mask(src, self.pad_id)          # (B,1,1,S)
        src_self_mask = src_pad_mask.expand(-1, 1, src.size(1), -1) # (B,1,S,S)
        for layer in self.layers:
            x = layer(x, src_self_mask)
        x = self.ln(x)
        return x, src_pad_mask


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 dropout: float, pad_id: int, max_len: int = 5000, tie_embeddings: bool = True,
                 layers: Optional[nn.ModuleList] = None):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = layers if layers is not None else nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.proj.weight = self.tok_emb.emb.weight
        self.pad_id = pad_id
        self.d_model = d_model

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        B, T = tgt.size()
        x = self.dropout(self.pos_enc(self.tok_emb(tgt)))
        device = tgt.device
        causal = make_causal_mask(T, device)                  # (1,1,T,T)
        tgt_pad = make_padding_mask(tgt, self.pad_id)         # (B,1,1,T)
        tgt_mask = (causal | tgt_pad.expand(-1, 1, T, -1))    # (B,1,T,T)
        memory_mask = src_pad_mask.expand(B, 1, T, -1)        # (B,1,T,S)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)

        x = self.ln(x)
        logits = self.proj(x)
        return logits


# ============================================================
# Full Model
# ============================================================

@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers_enc: int = 6
    n_layers_dec: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    pad_id: int = 0
    max_len: int = 5000
    tie_embeddings: bool = True

    # ASC
    use_ascender: bool = False
    asc_bias_enc: bool = True
    asc_bias_dec_self: bool = True
    asc_bias_dec_cross: bool = True
    asc_cfg: AscenderBiasConfig = field(default_factory=AscenderBiasConfig)

    # Residual Bias Path (architectural modification)
    enable_residual_path: bool = False  # Dual-path attention architecture

    probe_every: int = 0  # steps

    std_match_ratio_override: Optional[float] = 0.30  # global override


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder / Decoder
        self.encoder = Encoder(
            vocab_size=cfg.src_vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers_enc,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            pad_id=cfg.pad_id,
            max_len=cfg.max_len,
        )

        self.decoder = Decoder(
            vocab_size=cfg.tgt_vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers_dec,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            pad_id=cfg.pad_id,
            max_len=cfg.max_len,
            tie_embeddings=cfg.tie_embeddings,
        )

        # Base per-layer std-match defaults for decoder self-attn
        # std_match_ratio controls bias magnitude relative to attention scores std
        # Rule of thumb: 0.1-0.2 gentle, 0.3-0.5 moderate, 0.6-0.9 aggressive, >1.0 very aggressive

        # Check if config specifies a global std_match_ratio override
        global_r = getattr(cfg, "std_match_ratio_override", None)

        if len(self.decoder.layers) >= 1:
            # Default values (can be overridden by config or experiment)
            self.decoder.layers[0].self_attn.std_match_ratio = global_r if global_r else 0.15
            self.decoder.layers[0].self_attn.attn_temperature = 1.00
            self.decoder.layers[0].self_attn.sparsify_k_frac = 0.0
            self.decoder.layers[0].self_attn.v_gain_epsilon = 0.0
            self.decoder.layers[0].self_attn.bias_softcap = 6.0  # Allow higher for aggressive configs
        if len(self.decoder.layers) >= 2:
            self.decoder.layers[1].self_attn.std_match_ratio = global_r if global_r else 0.10
            self.decoder.layers[1].self_attn.attn_temperature = 1.00
            self.decoder.layers[1].self_attn.sparsify_k_frac = 0.0
            self.decoder.layers[1].self_attn.v_gain_epsilon = 0.0
            self.decoder.layers[1].self_attn.bias_softcap = 6.0

        # === ASCender attachment policy ===
        if cfg.use_ascender:
            print("[Init] ASCender ON (additive). Attach policy: decoder self-attn first 2 layers only.")
        else:
            print("[Init] ASCender OFF (baseline).")

        # Encoder biaser (default OFF per YAML)
        for i, layer in enumerate(self.encoder.layers):
            layer.biaser = AscenderBias(cfg.asc_cfg) if (cfg.use_ascender and cfg.asc_bias_enc) else None
            layer.self_attn.biaser = layer.biaser
            if layer.biaser is not None:
                # tag & expect_bias
                setattr(layer.self_attn, "role", f"enc.self.L{i}")
                setattr(layer.self_attn, "expect_bias", True)
                print(f"[Encoder] Layer {i} — biaser attached")

        # Decoder biasers — self: L0~L1 only; cross: YAML flag
        for i, layer in enumerate(self.decoder.layers):
            # self
            if cfg.use_ascender and cfg.asc_bias_dec_self and (i < 2):
                layer.biaser_self = AscenderBias(cfg.asc_cfg)
                layer.self_attn.biaser = layer.biaser_self
                role = f"dec.self.L{i}"
                # hard-wire expected snapshot FROM current runtime
                layer.biaser_self.expected_tau   = getattr(layer.self_attn, "attn_temperature", 1.0)
                layer.biaser_self.expected_topk  = getattr(layer.self_attn, "sparsify_k_frac", 0.0)
                layer.biaser_self.expected_r     = getattr(layer.self_attn, "std_match_ratio", 1.0)
                layer.biaser_self.expected_v_eps = getattr(layer.self_attn, "v_gain_epsilon", 0.0)

                # attach back-reference (enables DRIFT check)
                layer.biaser_self._attach_mha(layer.self_attn, role=role)

                # mild residual scale to keep stability with aggressive settings
                layer.resid1_scale = 0.9
                print(f"[Decoder] Layer {i} — self-attn biaser attached")
            else:
                layer.biaser_self = None
                layer.self_attn.biaser = None

            # cross
            if cfg.use_ascender and cfg.asc_bias_dec_cross:
                cross_cfg = dataclasses.replace(cfg.asc_cfg, past_only=False)
                layer.biaser_cross = AscenderBias(cross_cfg)
                layer.cross_attn.biaser = layer.biaser_cross
                layer.biaser_cross._attach_mha(layer.cross_attn, role=f"dec.cross.L{i}")
                print(f"[Decoder] Layer {i} — cross-attn biaser attached (past_only=False)")
            else:
                layer.biaser_cross = None
                layer.cross_attn.biaser = None

        # ══════════════════════════════════════════════════════════════════
        # RESIDUAL BIAS PATH: Enable dual-path architecture if requested
        # ══════════════════════════════════════════════════════════════════
        if getattr(cfg, "enable_residual_path", False):
            print("[Init] Residual Bias Path ENABLED (dual-path architecture)")
            for i, layer in enumerate(self.decoder.layers):
                if hasattr(layer, "self_attn"):
                    layer.self_attn.enable_residual_path = True
                if hasattr(layer, "cross_attn"):
                    layer.cross_attn.enable_residual_path = True
            for i, layer in enumerate(self.encoder.layers):
                if hasattr(layer, "self_attn"):
                    layer.self_attn.enable_residual_path = True
        else:
            print("[Init] Residual Bias Path OFF (standard additive bias)")

        # ---- Role tags & expect_bias flags for logging ----
        def _tag(mha: MultiHeadAttention, role: str, expect_bias: bool):
            setattr(mha, "role", role)
            setattr(mha, "expect_bias", bool(expect_bias))
            if not hasattr(mha, "_warned_no_bias"):
                mha._warned_no_bias = False

        for i, layer in enumerate(self.encoder.layers):
            _tag(layer.self_attn, role=f"enc.self.L{i}",
                 expect_bias=(self.cfg.use_ascender and self.cfg.asc_bias_enc))

        for i, layer in enumerate(self.decoder.layers):
            _tag(layer.self_attn, role=f"dec.self.L{i}",
                 expect_bias=(self.cfg.use_ascender and self.cfg.asc_bias_dec_self and i < 2))
            _tag(layer.cross_attn, role=f"dec.cross.L{i}",
                 expect_bias=(self.cfg.use_ascender and self.cfg.asc_bias_dec_cross))

        # ---- Wiring printout ----
        # for i in range(len(self.decoder.layers)):
        #     sa = self.decoder.layers[i].self_attn
        #     ca = self.decoder.layers[i].cross_attn
        #     print(f"[WIRE] L{i}.self_attn: biaser={type(sa.biaser).__name__ if sa.biaser is not None else None} "
        #           f"| expect_bias={getattr(sa, 'expect_bias', None)} "
        #           f"| r={getattr(sa, 'std_match_ratio', None)} "
        #           f"| tau={getattr(sa, 'attn_temperature', None)} "
        #           f"| topk={getattr(sa, 'sparsify_k_frac', None)} "
        #           f"| v_eps={getattr(sa, 'v_gain_epsilon', None)}")
        #     print(f"[WIRE] L{i}.cross_attn: biaser={type(ca.biaser).__name__ if ca.biaser is not None else None} "
        #           f"| expect_bias={getattr(ca, 'expect_bias', None)}")

        # ---- Lock initial runtime (true lock: restored each forward) ----
        try:
            # if len(self.decoder.layers) >= 1:
            #     self.decoder.layers[0].self_attn.lock_runtime_controls()
            # if len(self.decoder.layers) >= 2:
            #     self.decoder.layers[1].self_attn.lock_runtime_controls()
            pass
        except Exception as e:
            print(f"[ASC runtime lock] failed: {e}")

        # ---- Probes (store handles for clean detach) ----
        # ---- Probes (SAFE: do not register probes as submodules) ----
        # try:
        #     # keep strong references without making them child modules
        #     self._probe_refs = []  # list of (probe_callable, handle)

        #     def _attach_probe_safe(mha, name: str, every: int = 50):
        #         # Create the probe instance
        #         probe = AttnProbe(name, every=every)
        #         # Register its __call__ as a forward hook (callable)
        #         handle = mha.register_forward_hook(probe)
        #         # Keep refs in a plain Python list so it won't be treated as a child module
        #         self._probe_refs.append((probe, handle))

        #     # AFTER: cfg.model.get("probe_every", 0) > 0 일 때만
        #     probe_every = getattr(self.cfg, "probe_every", 50)
        #     if probe_every and len(self.decoder.layers) >= 1:
        #         _attach_probe_safe(self.decoder.layers[0].self_attn, "decoder.self_attn.layer0", every=probe_every)
        #     if probe_every and len(self.decoder.layers) >= 2:
        #         _attach_probe_safe(self.decoder.layers[1].self_attn, "decoder.self_attn.layer1", every=probe_every)
        # except Exception as e:
        #     print(f"[Probe] attach failed: {e}")

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.inference_mode(False)
    def forward(self, src: torch.Tensor, tgt_inp: torch.Tensor, return_attn: bool = False):
        memory, src_pad_mask = self.encoder(src)
        logits = self.decoder(tgt_inp, memory, src_pad_mask)

        if return_attn:
            attn_maps = []
            for layer in self.decoder.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "last_attn"):
                    attn_maps.append(layer.self_attn.last_attn)
            return logits, attn_maps

        return logits

    @torch.no_grad()
    def greedy_decode(
        self, src: torch.Tensor, bos_id: int, eos_id: int, max_len: int
    ) -> torch.Tensor:
        device = src.device
        memory, src_pad_mask = self.encoder(src)
        B = src.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(ys, memory, src_pad_mask)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_id], dim=1)
            if (next_id == eos_id).all():
                break
        return ys


# ============================================================
# Training helpers (Label Smoothing & Cache Flush)
# ============================================================

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, smoothing: float, ignore_index: int):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.conf = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab = vocab_size
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, T, V)
        target: (B, T)
        """
        B, T, V = logits.shape
        logits = logits.view(B * T, V)
        target = target.view(B * T)

        with torch.no_grad():
            true_dist = torch.full_like(logits, self.smoothing / (V - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.conf)
            true_dist[target == self.ignore_index] = 0.0

        log_probs = F.log_softmax(logits.float(), dim=-1)
        loss = -(true_dist * log_probs).sum(dim=1)
        loss = loss[target != self.ignore_index].mean()
        return loss


def flush_attn_caches(mha: MultiHeadAttention):
    """
    Clear attention snapshots (useful right after A/B toggles).
    """
    for name in ("attn_bias", "attn_pre", "attn_logits", "attn_probs",
                 "attn_pre_masked", "attn_post_masked", "last_attn", "probe_bias_snapshot"):
        if hasattr(mha, name):
            setattr(mha, name, None)


# ============================================================
# Tiny smoke test
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = TransformerConfig(
        src_vocab_size=1000, tgt_vocab_size=1000, d_model=256,
        n_heads=8, n_layers_enc=3, n_layers_dec=3, d_ff=1024,
        dropout=0.1, pad_id=0, use_ascender=True,
        asc_bias_enc=False, asc_bias_dec_self=True, asc_bias_dec_cross=False
    )
    model = Transformer(cfg)

    B, S, T = 4, 17, 13
    src = torch.randint(1, cfg.src_vocab_size, (B, S))
    src[:, -1] = cfg.pad_id
    tgt_inp = torch.randint(1, cfg.tgt_vocab_size, (B, T))
    tgt_out = torch.randint(1, cfg.tgt_vocab_size, (B, T))

    crit = LabelSmoothingLoss(cfg.tgt_vocab_size, smoothing=0.05, ignore_index=cfg.pad_id)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = NoamLR(opt, d_model=cfg.d_model, warmup_steps=4000)

    logits = model(src, tgt_inp)
    loss = crit(logits, tgt_out)
    loss.backward()

    # quick sanity for asc params
    b = model.decoder.layers[0].biaser_self
    if b is not None and hasattr(b, "gamma_log"):
        g_eff = torch.exp(b.gamma_log.detach()).clamp(max=b.cfg.gamma_cap)
        g_std = float((g_eff.std() if g_eff.ndim > 0 else torch.tensor(0.)).item())
        gate_eff = None
        if getattr(b, "gate_param", None) is not None:
            gr = torch.sigmoid(b.gate_param.detach())
            g_raw = b.cfg.gate_floor + (1.0 - b.cfg.gate_floor) * gr
            g_raw = torch.minimum(g_raw, torch.as_tensor(float(b.cfg.gate_ceiling), device=gr.device))
            gate_eff = float(g_raw.mean().item())
        print(f"[ASC headσ] γ.std={g_std:.3f} | gate.mean={gate_eff if gate_eff is not None else 'None'}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    print("OK — forward/backward step works. Loss:", float(loss))
