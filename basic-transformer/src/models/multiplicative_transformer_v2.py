from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# ASCender v2 Imports
# ==========================================
from src.models.ascender_bias_v2 import AscenderBiasV2, AscenderBiasV2Config

# ==========================================
# Utilities: Masks & Scheduler
# ==========================================

def make_padding_mask(seq: torch.Tensor, pad_id: int) -> torch.Tensor:
    mask = (seq == pad_id).unsqueeze(1).unsqueeze(2)
    return mask  # (B,1,1,S) bool

def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

def beta_warmup(step: int, warmup_steps: int, max_beta: float) -> float:
    if warmup_steps <= 0:
        return max_beta
    return max_beta * min(1.0, step / float(warmup_steps))

# ==========================================
# LR Scheduler
# ==========================================

class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1):
        self.d_model = d_model
        self.warmup = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]

# ==========================================
# Embedding + Positional Encoding
# ==========================================

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

# ==========================================
# ASCender v2 MultiHeadAttention
# ==========================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 max_beta: float = 0.6, beta_warmup_steps: int = 4000):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # --- ASCender Bias ---
        cfg = AscenderBiasV2Config()
        self.bias_mod = AscenderBiasV2(cfg, d_model=self.d_head)

        self.max_beta = max_beta
        self.beta_warmup_steps = beta_warmup_steps
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q.size()
        qh = self._shape(self.q_proj(q))
        kh = self._shape(self.k_proj(k))
        vh = self._shape(self.v_proj(v))

        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        beta = beta_warmup(int(self._step.item()), self.beta_warmup_steps, self.max_beta)
        if beta > 0:
            Bmat = torch.tanh(self.bias_mod(qh, kh))
            W = (1.0 + beta * Bmat).clamp(min=1e-3)
            attn = attn * W
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)

        attn = self.dropout(attn)
        out = torch.matmul(attn, vh).transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)
        self._step += 1
        self.last_attn = attn.detach()
        return out, attn

# ==========================================
# FFN + Layers
# ==========================================

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.self_attn(h, h, h, attn_mask=src_mask)
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ffn(self.ln2(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
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
    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor], mem_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.ln1(x)
        sa_out, _ = self.self_attn(h, h, h, attn_mask=tgt_mask)
        x = x + self.dropout1(sa_out)
        h2 = self.ln2(x)
        ca_out, _ = self.cross_attn(h2, memory, memory, attn_mask=mem_mask)
        x = x + self.dropout2(ca_out)
        x = x + self.dropout3(self.ffn(self.ln3(x)))
        return x

# ==========================================
# Encoder / Decoder
# ==========================================

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 dropout: float, pad_id: int, max_len: int = 5000):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.pad_id = pad_id
    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(self.pos_enc(self.tok_emb(src)))
        src_mask = make_padding_mask(src, self.pad_id)
        full_mask = src_mask.expand(-1, 1, src.size(1), -1)
        for layer in self.layers:
            x = layer(x, full_mask)
        x = self.ln(x)
        return x, src_mask

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 dropout: float, pad_id: int, max_len: int = 5000, tie_embeddings: bool = True):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
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
        causal = make_causal_mask(T, device)
        tgt_pad = make_padding_mask(tgt, self.pad_id)
        tgt_mask = (causal | tgt_pad.expand(-1, 1, T, -1))
        mem_mask = src_pad_mask.expand(B, 1, T, -1)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, mem_mask)
        x = self.ln(x)
        return self.proj(x)

# ==========================================
# Full Transformer (ASCender v2)
# ==========================================

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

class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.encoder = Encoder(cfg.src_vocab_size, cfg.d_model, cfg.n_layers_enc,
                               cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.pad_id, cfg.max_len)
        self.decoder = Decoder(cfg.tgt_vocab_size, cfg.d_model, cfg.n_layers_dec,
                               cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.pad_id, cfg.max_len,
                               tie_embeddings=cfg.tie_embeddings)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, src: torch.Tensor, tgt_inp: torch.Tensor) -> torch.Tensor:
        memory, src_pad_mask = self.encoder(src)
        logits = self.decoder(tgt_inp, memory, src_pad_mask)
        return logits
