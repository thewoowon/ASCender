from __future__ import annotations
import math
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Tuple
from src.models.ascender_bias import AscenderBias, AscenderBiasConfig
from src.hooks.attn_probe import AttnProbe

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities: Masks & Schedules
# -----------------------------

def attach_probes(model, layers=(0,1)):
    handles = []
    for li in layers:
        if li < len(model.decoder.layers):
            mha = model.decoder.layers[li].self_attn
            if not hasattr(mha, "_probe_handle") or mha._probe_handle is None:
                mha.probe = AttnProbe(f"decoder.self_attn.layer{li}")
                mha._probe_handle = mha.register_forward_hook(mha.probe)
                handles.append(mha._probe_handle)
    model._probe_handles = handles

def detach_probes(model, layers=(0,1)):
    for li in layers:
        if li < len(model.decoder.layers):
            mha = model.decoder.layers[li].self_attn
            if hasattr(mha, "_probe_handle") and mha._probe_handle is not None:
                mha._probe_handle.remove()
                mha._probe_handle = None
                if hasattr(mha, "probe"):
                    mha.probe = None


def make_padding_mask(seq: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Create a padding mask from a (B, S) integer tensor.
    Returns bool mask of shape (B, 1, 1, S) for broadcasting in attention.
      True  = mask (ignore)
      False = keep
    """
    assert seq.dim() == 2, f"Expected (B, S), got {seq.shape}"
    mask = (seq == pad_id).unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
    return mask  # bool


def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    Causal (look-ahead) mask for decoder self-attention.
    Shape: (1, 1, T, T), True where future positions should be masked (j>i).
    """
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Vaswani Noam LR schedule:
      lr = d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    """
    def __init__(self, optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


# -----------------------------
# Embeddings & Positional Enc.
# -----------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (non-learnable).
    Adds PE to token embeddings.
    """
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


# -----------------------------
# Core: Attention & FFN
# -----------------------------

class MultiHeadAttention(nn.Module):
    """
    Pre-LN friendly standard MHA (additive bias before softmax).
    Mask semantics:
      - attn_mask: (B,1,T,S) bool, True=mask
      - attn_bias: (B,H,T,S) additive logit bias (optional)
    Extras:
      - biaser: Optional[AscenderBias]
      - std_match_ratio: float   # bias 표준화 후 target 스케일 비율 r (레이어별 조정)
      - attn_temperature: float  # softmax 온도 τ (scores/=τ). 포화 완화용. 기본 1.0
      - sparsify_k_frac: float   # 마지막 축 S 기준 상위 k%만 bias 적용(0~1). 기본 0(해제)
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

        # ASCender
        self.biaser: Optional[AscenderBias] = None
        self.std_match_ratio: float = 1.0
        self.attn_temperature: float = 1.0
        self.sparsify_k_frac: float = 0.0  # 0이면 해제

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,S,dh)

    @staticmethod
    def _expand_valid(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        # mask: (B,1,T,S) -> broadcast to like:(B,H,T,S)
        return mask.expand(like.size(0), 1, like.size(2), like.size(3)).expand_as(like)

    def _masked_std_scores(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], per_head: bool) -> torch.Tensor:
        """유효 위치(~mask)만으로 scores(QK/√d)의 std 추정."""
        x = x.float().detach()
        if attn_mask is None:
            return x.std(dim=(0,2,3)).clamp_min(1e-6) if per_head else x.std().clamp_min(1e-6)
        vexp = self._expand_valid(~attn_mask, x).float()
        if per_head:
            num = vexp.sum(dim=(0,2,3)).clamp_min(1.0)              # (H,)
            mu  = (x*vexp).sum(dim=(0,2,3)) / num                   # (H,)
            var = (((x - mu.view(1,-1,1,1))**2)*vexp).sum(dim=(0,2,3)) / num
            return var.sqrt().clamp_min(1e-6)                       # (H,)
        else:
            num = vexp.sum().clamp_min(1.0)
            mu  = (x*vexp).sum() / num
            var = (((x - mu)**2)*vexp).sum() / num
            return var.sqrt().clamp_min(1e-6)

    def _masked_std_bias(self, b: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """유효 위치만으로 bias std 추정: (B,H,1,1)"""
        b = b.float()
        if attn_mask is None:
            return b.std(dim=(-2,-1), keepdim=True).clamp_min(1e-6)
        vexp = self._expand_valid(~attn_mask, b).float()
        num  = vexp.sum(dim=(-2,-1), keepdim=True).clamp_min(1.0)
        mu   = (b*vexp).sum(dim=(-2,-1), keepdim=True) / num
        var  = (((b - mu)**2)*vexp).sum(dim=(-2,-1), keepdim=True) / num
        return var.sqrt().clamp_min(1e-6)

    @staticmethod
    def _sparsify_last_dim(bias: torch.Tensor, k_frac: float, use_abs: bool = True) -> torch.Tensor:
        if not (0.0 < k_frac < 1.0):
            return bias
        B,H,T,S = bias.shape
        k = max(1, int(S * k_frac))
        sel = bias.abs() if use_abs else bias
        topv, topi = torch.topk(sel, k, dim=-1)
        mask = torch.zeros_like(bias, dtype=torch.bool).scatter_(-1, topi, True)
        return torch.where(mask, bias, torch.zeros_like(bias))

    def forward(
        self,
        q: torch.Tensor,  # (B,T,d_model)
        k: torch.Tensor,  # (B,S,d_model)
        v: torch.Tensor,  # (B,S,d_model)
        attn_mask: Optional[torch.Tensor] = None,   # (B,1,T,S) True=mask
        attn_bias: Optional[torch.Tensor] = None,   # (B,H,T,S)
        *,
        pre_q: Optional[torch.Tensor] = None,
        pre_k: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q.size(); S = k.size(1)

        # projections
        qh = self._shape(self.q_proj(q))   # (B,H,T,dh)
        kh = self._shape(self.k_proj(k))   # (B,H,S,dh)
        vh = self._shape(self.v_proj(v))   # (B,H,S,dh)

        # logits
        scores = torch.matmul(qh, kh.transpose(-2,-1)) / math.sqrt(self.d_head)  # (B,H,T,S)

        # 온도 τ로 포화 완화 (선택, 기본 1.0 → 영향 없음)
        tau = float(getattr(self, "attn_temperature", 1.0))
        if tau != 1.0:
            scores = scores / tau

        scores = torch.nan_to_num(scores, nan=0.0, posinf=80.0, neginf=-80.0)
        scores = scores.clamp(-80, 80)     # <-- 클램프를 먼저

        self.attn_pre = scores.detach()

        # scores std (유효 위치만)
        per_head_mode = (self.biaser is not None) and (
            getattr(self.biaser.cfg, "per_head_scale", False) or getattr(self.biaser.cfg, "per_head_gate", False)
        )
        scores_std = self._masked_std_scores(scores, attn_mask, per_head=per_head_mode)  # scalar or (H,)

        # --- bias 생성/주입 ---
        runtime_bias = None
        if self.biaser is not None:
            _pre_q = pre_q if pre_q is not None else q
            _pre_k = pre_k if pre_k is not None else k
            runtime_bias = self.biaser(qh, kh, pre_q=_pre_q, pre_k=_pre_k, scores_std=scores_std)
        elif attn_bias is not None:
            runtime_bias = attn_bias
        else:
            # 기대되는 자리(expect_bias=True)에서만 1회 경고
            if getattr(self, "expect_bias", False):
                if not hasattr(self, "_warned_no_bias") or not self._warned_no_bias:
                    print("[MHA] No bias injected: biaser=None and attn_bias=None")
                    self._warned_no_bias = True

        if runtime_bias is not None:
            runtime_bias = torch.nan_to_num(runtime_bias, nan=0.0, posinf=80.0, neginf=-80.0)
            assert runtime_bias.shape == scores.shape, f"bias {runtime_bias.shape} vs scores {scores.shape}"

            # (1) 마스크 영역 0 처리(더하기 전)
            if attn_mask is not None:
                runtime_bias = runtime_bias.masked_fill(attn_mask, 0.0)

            # (2) (선택) sparsify로 집중력↑/노이즈↓
            k_frac = float(getattr(self, "sparsify_k_frac", 0.0))
            if k_frac > 0.0:
                runtime_bias = self._sparsify_last_dim(runtime_bias, k_frac=k_frac)

            # (3) 표준화 + 목표 스케일 매칭 (유효 위치 기준)
            #     ★ no_grad 금지: 통계만 detach, 변환은 미분 가능하게 유지
            if isinstance(scores_std, torch.Tensor):
                t_std = scores_std.view(1, -1, 1, 1)
            else:
                t_std = torch.tensor(scores_std, device=scores.device, dtype=scores.dtype).view(1,1,1,1)
            t_std = t_std.detach().clamp_min(1e-6)

            b_std = self._masked_std_bias(runtime_bias, attn_mask).detach().clamp_min(1e-6)  # (B,H,1,1)
            if getattr(self.biaser.cfg, "std_batch_mean", True):
                b_std = b_std.mean(dim=0, keepdim=True)  # (1,H,1,1)

            r = float(getattr(self, "std_match_ratio", 1.0))

            runtime_bias = (runtime_bias / b_std) * (t_std * r)

            self.attn_bias = runtime_bias
            self.probe_bias_snapshot = runtime_bias.detach()  # 로그용 복사

            # --- Probe용: 동일 마스크 기준 스냅샷 (pre: bias 전, post: bias 후)
            pre_for_probe = scores 
            post_for_probe = scores + runtime_bias

            if attn_mask is not None:
                pre_for_probe  = pre_for_probe.masked_fill(attn_mask, float("-inf"))
                post_for_probe = post_for_probe.masked_fill(attn_mask, float("-inf"))

            self.attn_pre_masked  = pre_for_probe.detach()
            self.attn_post_masked = post_for_probe.detach()

            # 실제 적용
            scores = scores + runtime_bias
        else:
            self.attn_bias = None
            if attn_mask is not None:
                pre_masked = scores.masked_fill(attn_mask, float("-inf"))
            else:
                pre_masked = scores
            self.attn_pre_masked  = pre_masked.detach()
            self.attn_post_masked = pre_masked.detach()

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        # 안정화 + softmax
        self.attn_logits = scores.detach()

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        self.attn_probs = attn.detach()

        # 출력 병합
        out = torch.matmul(attn, vh).transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)

        self.last_attn = attn.detach()
        return out, attn

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


# -----------------------------
# Encoder / Decoder Layers
# -----------------------------

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, biaser: Optional[nn.Module] = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.biaser: Optional[AscenderBias] = biaser  # AscenderBias per-layer (can be None)
        # 내부 경로로 단일화: MHA가 직접 bias 생성/적용
        self.self_attn.biaser = self.biaser

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.ln1(x)
        # 내부 biaser가 있으면 MHA가 softmax 직전에 직접 생성/적용
        attn_out, _ = self.self_attn(h, h, h, attn_mask=src_mask, pre_q=h, pre_k=h)
        x = x + self.dropout1(attn_out)

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

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.biaser_self: Optional[AscenderBias] = biaser_self
        self.biaser_cross: Optional[AscenderBias] = biaser_cross

        # 내부 경로로 단일화
        self.self_attn.biaser = self.biaser_self
        self.cross_attn.biaser = self.biaser_cross

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self-attention (causal + padding on target)
        h = self.ln1(x)
        sa_out, _ = self.self_attn(h, h, h, attn_mask=tgt_mask, pre_q=h, pre_k=h)
        x = x + self.dropout1(sa_out)

        # Cross-attention (pad-masked on source)
        h2 = self.ln2(x)
        ca_out, _ = self.cross_attn(h2, memory, memory, attn_mask=memory_mask, pre_q=h2, pre_k=memory)
        x = x + self.dropout2(ca_out)

        h3 = self.ln3(x)
        x = x + self.dropout3(self.ffn(h3))
        return x


# -----------------------------
# Stacks
# -----------------------------

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
        Returns:
          memory: (B, S, d_model)
          src_mask: (B,1,1,S) for cross-attn consumers (compact)
        """
        x = self.dropout(self.pos_enc(self.tok_emb(src)))
        src_pad_mask = make_padding_mask(src, self.pad_id)    # (B,1,1,S)
        src_self_mask = src_pad_mask.expand(-1, 1, src.size(1), -1)  # (B,1,S,S)
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

    def forward(
        self,
        tgt: torch.Tensor,           # (B, T)
        memory: torch.Tensor,        # (B, S, d_model)
        src_pad_mask: torch.Tensor,  # (B,1,1,S)
    ) -> torch.Tensor:
        B, T = tgt.size()
        x = self.dropout(self.pos_enc(self.tok_emb(tgt)))
        device = tgt.device
        causal = make_causal_mask(T, device)               # (1,1,T,T)
        tgt_pad = make_padding_mask(tgt, self.pad_id)      # (B,1,1,T)
        tgt_mask = (causal | tgt_pad.expand(-1, 1, T, -1)) # (B,1,T,T) True=mask
        memory_mask = src_pad_mask.expand(B, 1, T, -1)     # (B,1,T,S)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)

        x = self.ln(x)
        logits = self.proj(x)
        return logits


# -----------------------------
# Full Model
# -----------------------------

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
    use_ascender: bool = False
    asc_bias_enc: bool = True        # bias in encoder self-attn
    asc_bias_dec_self: bool = True   # bias in decoder self-attn
    asc_bias_dec_cross: bool = True  # bias in decoder cross-attn
    asc_cfg: AscenderBiasConfig = field(default_factory=AscenderBiasConfig)

class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        # --- Encoder ---
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

        # --- Decoder ---
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

        if len(self.decoder.layers) >= 1:
            self.decoder.layers[0].self_attn.std_match_ratio = 1.8
        if len(self.decoder.layers) >= 2:
            self.decoder.layers[1].self_attn.std_match_ratio = 1.2

        try:
            self.decoder.layers[0].self_attn.attn_temperature = 2.0
            self.decoder.layers[0].self_attn.sparsify_k_frac = 0.20
            self.decoder.layers[1].self_attn.attn_temperature = 1.5
            self.decoder.layers[1].self_attn.sparsify_k_frac = 0.0
        except Exception:
            pass

        # === ASCender attachment policy ===
        if cfg.use_ascender:
            print(f"[Init] ASCender ON (additive). Attach policy: decoder self-attn first 2 layers only.")
        else:
            print(f"[Init] ASCender OFF (baseline).")

        # Encoder biaser (기본 OFF 권장)
        for i, layer in enumerate(self.encoder.layers):
            layer.biaser = AscenderBias(cfg.asc_cfg) if (cfg.use_ascender and cfg.asc_bias_enc) else None
            # 내부 경로로 장착
            layer.self_attn.biaser = layer.biaser
            if layer.biaser is not None:
                print(f"[Encoder] Layer {i} — biaser attached")

        # Decoder biasers — self(0~1)만 ON, cross OFF 권장
        for i, layer in enumerate(self.decoder.layers):
            if cfg.use_ascender and cfg.asc_bias_dec_self and (i < 2):
                layer.biaser_self = AscenderBias(cfg.asc_cfg)
                layer.self_attn.biaser = layer.biaser_self
                print(f"[Decoder] Layer {i} — self-attn biaser attached")
            else:
                layer.biaser_self = None
                layer.self_attn.biaser = None

            if cfg.use_ascender and cfg.asc_bias_dec_cross:
                # cross-attn은 과거제한이 없으므로 past_only=False 권장
                cross_cfg = dataclasses.replace(cfg.asc_cfg, past_only=False)
                layer.biaser_cross = AscenderBias(cross_cfg)
                layer.cross_attn.biaser = layer.biaser_cross
                print(f"[Decoder] Layer {i} — cross-attn biaser attached (past_only=False)")
            else:
                layer.biaser_cross = None
                layer.cross_attn.biaser = None

        # ---- (NEW) MHA 태깅: role / expect_bias ----
        def _tag(mha: MultiHeadAttention, role: str, expect_bias: bool):
            setattr(mha, "role", role)
            setattr(mha, "expect_bias", bool(expect_bias))
            # 최초 한 번만 경고 찍도록 내부 플래그 초기화(있으면 사용됨)
            if not hasattr(mha, "_warned_no_bias"):
                mha._warned_no_bias = False

        # Encoder MHA 태깅
        for i, layer in enumerate(self.encoder.layers):
            _tag(layer.self_attn, role=f"enc.self.L{i}",
                 expect_bias=(self.cfg.use_ascender and self.cfg.asc_bias_enc))

        # Decoder MHA 태깅
        for i, layer in enumerate(self.decoder.layers):
            _tag(layer.self_attn, role=f"dec.self.L{i}",
                 expect_bias=(self.cfg.use_ascender and self.cfg.asc_bias_dec_self and i < 2))
            _tag(layer.cross_attn, role=f"dec.cross.L{i}",
                 expect_bias=(self.cfg.use_ascender and self.cfg.asc_bias_dec_cross))

        # ---- (NEW) 배선 확정 프린트 ----
        for i in range(len(self.decoder.layers)):
            sa = self.decoder.layers[i].self_attn
            ca = self.decoder.layers[i].cross_attn
            print(f"[WIRE] L{i}.self_attn: biaser={type(sa.biaser).__name__ if sa.biaser is not None else None} "
                  f"| expect_bias={getattr(sa, 'expect_bias', None)} "
                  f"| r={getattr(sa, 'std_match_ratio', None)} "
                  f"| tau={getattr(sa, 'attn_temperature', None)} "
                  f"| topk={getattr(sa, 'sparsify_k_frac', None)}")
            print(f"[WIRE] L{i}.cross_attn: biaser={type(ca.biaser).__name__ if ca.biaser is not None else None} "
                  f"| expect_bias={getattr(ca, 'expect_bias', None)}")

        try:
            # decoder self-attn L0
            if len(self.decoder.layers) >= 1:
                mha0 = self.decoder.layers[0].self_attn
                mha0.probe = AttnProbe("decoder.self_attn.layer0", every=50)
                mha0.register_forward_hook(mha0.probe)
            # decoder self-attn L1
            if len(self.decoder.layers) >= 2:
                mha1 = self.decoder.layers[1].self_attn
                mha1.probe = AttnProbe("decoder.self_attn.layer1", every=50)
                mha1.register_forward_hook(mha1.probe)
        except Exception as e:
            print(f"[Probe] attach failed: {e}")

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


# -----------------------------
# Training helpers
# -----------------------------

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


# -----------------------------
# Tiny smoke test (run directly)
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = TransformerConfig(src_vocab_size=1000, tgt_vocab_size=1000, d_model=256,
                            n_heads=8, n_layers_enc=3, n_layers_dec=3, d_ff=1024,
                            dropout=0.1, pad_id=0, use_ascender=True)
    model = Transformer(cfg)

    B, S, T = 4, 17, 13
    src = torch.randint(1, cfg.src_vocab_size, (B, S))
    src[:, -1] = cfg.pad_id  # add some pads
    tgt_inp = torch.randint(1, cfg.tgt_vocab_size, (B, T))
    tgt_out = torch.randint(1, cfg.tgt_vocab_size, (B, T))

    crit = LabelSmoothingLoss(cfg.tgt_vocab_size, smoothing=0.05, ignore_index=cfg.pad_id)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = NoamLR(opt, d_model=cfg.d_model, warmup_steps=4000)

    logits = model(src, tgt_inp)        # (B,T,V)
    loss = crit(logits, tgt_out)
    loss.backward()

    # grad 체크: 새(γ_log) / 옛(γ) 모두 호환
    b = model.decoder.layers[0].biaser_self

    if b is not None and hasattr(b, "gamma_log"):
        g_eff_h = torch.exp(b.gamma_log.detach()).clamp(max=b.cfg.gamma_cap)  # (H,) or scalar
        if g_eff_h.ndim > 0:
            g_std = float(g_eff_h.std().item())
        else:
            g_std = 0.0
        if getattr(b, "gate_param", None) is not None:
            gr = torch.sigmoid(b.gate_param.detach())
            g_raw = b.cfg.gate_floor + (1.0 - b.cfg.gate_floor) * gr
            g_raw = torch.minimum(g_raw, torch.as_tensor(float(b.cfg.gate_ceiling), device=gr.device))
            ggate_std = float(g_raw.std().item()) if g_raw.ndim > 0 else 0.0
        else:
            ggate_std = 0.0
        print(f"[ASC headσ] γ.std={g_std:.3f} | gate.std={ggate_std:.3f}")

    if b is not None:
        grad_gamma_log = getattr(b, "gamma_log", None)
        grad_gamma = getattr(b, "gamma", None)
        if grad_gamma_log is not None and grad_gamma_log.grad is not None:
            print("[Grad] d(log_gamma) mean =", float(grad_gamma_log.grad.mean()))
        elif grad_gamma is not None and grad_gamma.grad is not None:
            print("[Grad] d(gamma) mean =", float(grad_gamma.grad.mean()))
        else:
            print("[Grad] gamma grad = None")

        if hasattr(b, "ema_ratio"):
            gamma_eff = b.gamma_effective if hasattr(b, "gamma_effective") else (
                float((b.gamma.mean() if hasattr(b, "gamma") else torch.tensor(0.0)).item())
            )
            gate_eff = b.gate_effective if hasattr(b, "gate_effective") else (
                float(torch.sigmoid(b.gate_param).mean().item()) if hasattr(b, "gate_param") and b.gate_param is not None else None
            )
            print(f"[ASC] ratio(ema)={float(b.ema_ratio):.3f} | γ={gamma_eff:.3f} | gate={('None' if gate_eff is None else f'{gate_eff:.3f}')}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    print("OK — forward/backward step works. Loss:", float(loss))
