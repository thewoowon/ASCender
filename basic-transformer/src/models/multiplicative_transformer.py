from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple
from src.models.ascender_bias import AscenderBias, AscenderBiasConfig

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities: Masks & Schedules
# -----------------------------

def make_padding_mask(seq: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Create a padding mask from a (B, S) integer tensor.
    Returns.bool mask of shape (B, 1, 1, S) for broadcasting in attention.
      True  = mask (ignore)
      False = keep
    """
    assert seq.dim() == 2, f"Expected (B, S), got {seq.shape}"
    mask = (seq == pad_id).unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
    return mask  # bool


def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    Causal (look-ahead) mask for decoder self-attention.
    Shape: (1, 1, T, T), True where future positions should be masked.
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
    Pre-LN friendly MHA.
    Mask semantics:
      - attn_mask: (B, 1, T, S) bool, True to mask.
      - attn_bias: (B, H, T, S)  **multiplicative scaling** term (after softmax).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # NEW: learnable multiplicative bias strength (β)
        self.bias_scale = nn.Parameter(torch.tensor(0.5))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, d_model) -> (B, H, S, d_head)
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,  # (B,T,d_model)
        k: torch.Tensor,  # (B,S,d_model)
        v: torch.Tensor,  # (B,S,d_model)
        attn_mask: Optional[torch.Tensor] = None,  # (B,1,T,S) True=mask
        attn_bias: Optional[torch.Tensor] = None,  # (B,H,T,S) multiplicative source
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q.size(); S = k.size(1)
        qh = self._shape(self.q_proj(q))   # (B,H,T,dh)
        kh = self._shape(self.k_proj(k))   # (B,H,S,dh)
        vh = self._shape(self.v_proj(v))   # (B,H,S,dh)

        # scaled dot-product logits
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,T,S)
        scores = scores.clamp(-80, 80)  # 안정성 (MPS 포함)

        # padding/causal mask (True=mask -> -inf)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        # base attention
        attn = F.softmax(scores, dim=-1)

        # NEW: multiplicative bias injection AFTER softmax
        if attn_bias is not None:
            # 안전성: 모양 체크
            if attn_bias.shape != attn.shape:
                raise ValueError(f"attn_bias shape {attn_bias.shape} must match attn {attn.shape}")
            # [-1,1]로 제한된 스케일 → (1 + β * tanh(bias))
            scale = 1.0 + self.bias_scale * torch.tanh(attn_bias)
            # 음수 방지 & 재정규화
            attn = attn * torch.clamp(scale, min=1e-9)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)

        attn = self.dropout(attn)
        out = torch.matmul(attn, vh).transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)
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
        self.biaser = biaser

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Pre-LN
        h = self.ln1(x)
        attn_bias = None
        if self.biaser is not None:
            attn_bias = self.biaser(
                qh=self.self_attn._shape(self.self_attn.q_proj(h)),
                kh=self.self_attn._shape(self.self_attn.k_proj(h)),
                pre_q=h, pre_k=h
            )

        attn_out, _ = self.self_attn(h, h, h, attn_mask=src_mask, attn_bias=attn_bias)
        x = x + self.dropout1(attn_out)

        h = self.ln2(x)
        x = x + self.dropout2(self.ffn(h))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, biaser_self: Optional[nn.Module]=None,
                 biaser_cross: Optional[nn.Module]=None):
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

        self.biaser_self = biaser_self
        self.biaser_cross = biaser_cross

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        memory_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self-attention (causal + padding on target)
        h = self.ln1(x)
        self_bias = None
        if self.biaser_self is not None:
            self_bias = self.biaser_self(
                qh=self.self_attn._shape(self.self_attn.q_proj(h)),
                kh=self.self_attn._shape(self.self_attn.k_proj(h)),
                pre_q=h, pre_k=h
            )
        sa_out, _ = self.self_attn(h, h, h, attn_mask=tgt_mask, attn_bias=self_bias)
        x = x + self.dropout1(sa_out)

        # Cross-attention (pad-masked on source)
        h2 = self.ln2(x)
        cross_bias = None
        if self.biaser_cross is not None:
            cross_bias = self.biaser_cross(
                qh=self.cross_attn._shape(self.cross_attn.q_proj(h2)),
                kh=self.cross_attn._shape(self.cross_attn.k_proj(memory)),
                pre_q=h2, pre_k=memory
            )
        ca_out, _ = self.cross_attn(h2, memory, memory, attn_mask=memory_mask, attn_bias=cross_bias)
        x = x + self.dropout2(ca_out)

        h3 = self.ln3(x)
        x = x + self.dropout3(self.ffn(h3))
        return x


# -----------------------------
# Stacks
# -----------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 dropout: float, pad_id: int, max_len: int = 5000):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.pad_id = pad_id

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(self.pos_enc(self.tok_emb(src)))
        src_mask = make_padding_mask(src, self.pad_id)  # (B,1,1,S)
        src_self_mask = src_mask.expand(-1, 1, src.size(1), -1)  # (B,1,S,S)
        for layer in self.layers:
            x = layer(x, src_self_mask)
        x = self.ln(x)
        return x, src_mask  # (B,S,d), (B,1,1,S)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 dropout: float, pad_id: int, max_len: int = 5000, tie_embeddings: bool = True):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
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
    asc_bias_enc: bool = True
    asc_bias_dec_self: bool = True
    asc_bias_dec_cross: bool = True
    asc_cfg: AscenderBiasConfig = field(default_factory=AscenderBiasConfig)


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        enc_biaser = AscenderBias(cfg.asc_cfg) if (cfg.use_ascender and cfg.asc_bias_enc) else None
        dec_self_biaser = AscenderBias(cfg.asc_cfg) if (cfg.use_ascender and cfg.asc_bias_dec_self) else None
        dec_cross_biaser = AscenderBias(cfg.asc_cfg) if (cfg.use_ascender and cfg.asc_bias_dec_cross) else None

        print(f"[Init] use_ascender={cfg.use_ascender}")
        if cfg.use_ascender:
            print(f"  • Encoder biaser: {'ON' if enc_biaser else 'OFF'}")
            print(f"  • Decoder self-attn biaser: {'ON' if dec_self_biaser else 'OFF'}")
            print(f"  • Decoder cross-attn biaser: {'ON' if dec_cross_biaser else 'OFF'}")

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
        for i, layer in enumerate(self.encoder.layers):
            layer.biaser = enc_biaser
            if cfg.use_ascender and enc_biaser:
                print(f"[Encoder] Layer {i} — biaser attached")

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
        for i, layer in enumerate(self.decoder.layers):
            layer.biaser_self = dec_self_biaser
            layer.biaser_cross = dec_cross_biaser
            if cfg.use_ascender:
                if dec_self_biaser:
                    print(f"[Decoder] Layer {i} — self-attn biaser attached")
                if dec_cross_biaser:
                    print(f"[Decoder] Layer {i} — cross-attn biaser attached")

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.inference_mode(False)
    def forward(self, src: torch.Tensor, tgt_inp: torch.Tensor) -> torch.Tensor:
        memory, src_pad_mask = self.encoder(src)
        logits = self.decoder(tgt_inp, memory, src_pad_mask)

        if self.cfg.use_ascender:
            print(f"[Forward] Applied Ascender bias — src_len={src.size(1)}, tgt_len={tgt_inp.size(1)}")
            first_layer = self.decoder.layers[0]
            if first_layer.biaser_self is not None:
                h = first_layer.ln1(
                    tgt_inp.new_zeros((1, tgt_inp.size(1), self.cfg.d_model), dtype=torch.float32)
                )
                qh = first_layer.self_attn._shape(first_layer.self_attn.q_proj(h))
                kh = first_layer.self_attn._shape(first_layer.self_attn.k_proj(h))
                bias_sample = first_layer.biaser_self(qh, kh, pre_q=h, pre_k=h)  # (1,H,T,T)
                bmean = bias_sample.mean().detach().item()
                bstd = float(bias_sample.std())
                bmin = float(bias_sample.min())
                bmax = float(bias_sample.max())
                print(f"[Debug] Decoder[0].self_bias stats — mean={bmean:.4f}, std={bstd:.4f}, "
                      f"min={bmin:.4f}, max={bmax:.4f}")

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
                            dropout=0.1, pad_id=0)
    model = Transformer(cfg)

    B, S, T = 4, 17, 13
    src = torch.randint(1, cfg.src_vocab_size, (B, S))
    src[:, -1] = cfg.pad_id
    tgt_inp = torch.randint(1, cfg.tgt_vocab_size, (B, T))
    tgt_out = torch.randint(1, cfg.tgt_vocab_size, (B, T))

    crit = LabelSmoothingLoss(cfg.tgt_vocab_size, smoothing=0.1, ignore_index=cfg.pad_id)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = NoamLR(opt, d_model=cfg.d_model, warmup_steps=4000)

    logits = model(src, tgt_inp)
    loss = crit(logits, tgt_out)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    print("OK — forward/backward step works. Loss:", float(loss))
