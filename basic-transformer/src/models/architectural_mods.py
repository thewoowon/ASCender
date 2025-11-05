# architectural_mods.py
"""
Architectural modifications to make ASCender bias more impactful.
These are FUNDAMENTAL changes that go beyond hyperparameter tuning.

Integration options:
1. Residual Bias Path - bias has its own residual stream
2. Gated Bias Integration - learnable gates control bias influence
3. Multi-scale Bias - different components at different scales
4. Bias-conditioned Value - bias affects value projection too
5. Hierarchical Bias - coarse-to-fine bias refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ============================================================
# Option 1: Residual Bias Path
# ============================================================

class MultiHeadAttentionWithResidualBias(nn.Module):
    """
    Instead of adding bias to scores, create a parallel bias-driven attention path
    and mix outputs with a learnable gate.

    Flow:
      1. Normal attention: softmax(Q·K / sqrt(d)) @ V = out_normal
      2. Biased attention: softmax(Q·K / sqrt(d) + BIAS) @ V = out_biased
      3. Final: α * out_normal + (1-α) * out_biased, where α is learned per-head

    Advantage: Bias can't completely overwhelm learned patterns; always blends
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
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

        # Learnable mixing weight per head
        self.alpha_logit = nn.Parameter(torch.zeros(n_heads))  # logit space

        self.biaser = None  # Will be attached externally

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q.size()

        qh = self._shape(self.q_proj(q))
        kh = self._shape(self.k_proj(k))
        vh = self._shape(self.v_proj(v))

        # Base scores
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_head)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))

        # Path 1: Normal attention
        attn_normal = F.softmax(scores, dim=-1)
        attn_normal = self.dropout(attn_normal)
        out_normal = torch.matmul(attn_normal, vh)

        # Path 2: Biased attention
        if self.biaser is not None:
            bias = self.biaser(qh, kh, pre_q=q, pre_k=k)
            scores_biased = scores + bias
            if attn_mask is not None:
                scores_biased = scores_biased.masked_fill(attn_mask, float('-inf'))
            attn_biased = F.softmax(scores_biased, dim=-1)
            attn_biased = self.dropout(attn_biased)
            out_biased = torch.matmul(attn_biased, vh)

            # Learnable mixing (per head)
            alpha = torch.sigmoid(self.alpha_logit).view(1, -1, 1, 1)
            out = alpha * out_normal + (1 - alpha) * out_biased
            attn = alpha * attn_normal + (1 - alpha) * attn_biased
        else:
            out = out_normal
            attn = attn_normal

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)

        return out, attn


# ============================================================
# Option 2: Gated Bias Integration
# ============================================================

class GatedBiasAttention(nn.Module):
    """
    Use a learned gate to control bias influence per-query position.

    Gate is computed from query features: g = σ(W_g @ q)
    Then: final_bias = g * structural_bias

    Advantage: Model learns when to trust bias vs ignore it
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
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

        # Gate network: query -> per-position gate
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, n_heads),
            nn.Sigmoid()
        )

        self.biaser = None

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q.size()

        qh = self._shape(self.q_proj(q))
        kh = self._shape(self.k_proj(k))
        vh = self._shape(self.v_proj(v))

        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Compute per-query-position gate
        gate = self.gate_net(q)  # (B, T, H)
        gate = gate.transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)

        if self.biaser is not None:
            bias = self.biaser(qh, kh, pre_q=q, pre_k=k)
            # Apply gate: each query position has its own trust level
            bias = bias * gate
            scores = scores + bias

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, vh)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)

        return out, attn


# ============================================================
# Option 3: Multi-Scale Bias
# ============================================================

class MultiScaleBiasAttention(nn.Module):
    """
    Apply different bias components at different attention scales.

    Idea:
    - Local bias (cohesion): Strong at nearby positions
    - Global bias (alignment): Affects all positions
    - Mid-range bias: Affects intermediate distances

    Each scale has independent strength control.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
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

        # Scale-specific mixing weights (learned)
        self.local_weight = nn.Parameter(torch.tensor(0.5))
        self.mid_weight = nn.Parameter(torch.tensor(0.3))
        self.global_weight = nn.Parameter(torch.tensor(0.2))

        self.biaser_local = None   # short-range (σ=2)
        self.biaser_mid = None     # mid-range (σ=8)
        self.biaser_global = None  # long-range (alignment-based)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q.size()

        qh = self._shape(self.q_proj(q))
        kh = self._shape(self.k_proj(k))
        vh = self._shape(self.v_proj(v))

        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Accumulate multi-scale biases
        total_bias = torch.zeros_like(scores)

        if self.biaser_local is not None:
            bias_local = self.biaser_local(qh, kh, pre_q=q, pre_k=k)
            total_bias = total_bias + self.local_weight * bias_local

        if self.biaser_mid is not None:
            bias_mid = self.biaser_mid(qh, kh, pre_q=q, pre_k=k)
            total_bias = total_bias + self.mid_weight * bias_mid

        if self.biaser_global is not None:
            bias_global = self.biaser_global(qh, kh, pre_q=q, pre_k=k)
            total_bias = total_bias + self.global_weight * bias_global

        scores = scores + total_bias

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, vh)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)

        return out, attn


# ============================================================
# Option 4: Bias-Conditioned Value
# ============================================================

class BiasConditionedValueAttention(nn.Module):
    """
    Let bias affect not just WHERE to attend (scores), but also WHAT to retrieve (values).

    Idea:
    - Compute bias B from structure
    - Modulate value projection: V' = V * (1 + ε * tanh(B_aggregated))

    This creates a two-way influence:
    1. Bias guides attention distribution
    2. Bias modulates retrieved features
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
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

        # Value modulation strength (learned)
        self.v_mod_strength = nn.Parameter(torch.tensor(0.1))

        self.biaser = None

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q.size()
        S = k.size(1)

        qh = self._shape(self.q_proj(q))
        kh = self._shape(self.k_proj(k))
        vh = self._shape(self.v_proj(v))

        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_head)

        if self.biaser is not None:
            bias = self.biaser(qh, kh, pre_q=q, pre_k=k)  # (B, H, T, S)

            # Apply to scores
            scores = scores + bias

            # Aggregate bias per key position: mean over query dimension
            bias_key = bias.mean(dim=2)  # (B, H, S)

            # Modulate values based on structural importance
            v_mod = 1.0 + self.v_mod_strength * torch.tanh(bias_key).unsqueeze(-1)
            vh = vh * v_mod  # (B, H, S, dh) * (B, H, S, 1)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, vh)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)

        return out, attn


# ============================================================
# Option 5: Hierarchical Bias (Coarse-to-Fine)
# ============================================================

class HierarchicalBiasAttention(nn.Module):
    """
    Apply bias in stages:
    1. Coarse bias (low-resolution) shapes rough attention pattern
    2. Fine bias (high-resolution) refines based on coarse output

    Think of it like:
    - Coarse: "attend to nearby tokens" (positional)
    - Fine: "among nearby, pick most semantically relevant" (content)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
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

        # Two-stage bias strength
        self.coarse_strength = nn.Parameter(torch.tensor(0.5))
        self.fine_strength = nn.Parameter(torch.tensor(0.3))

        self.biaser_coarse = None  # Positional only (cohesion/separation)
        self.biaser_fine = None    # Content-based (alignment)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = q.size()

        qh = self._shape(self.q_proj(q))
        kh = self._shape(self.k_proj(k))
        vh = self._shape(self.v_proj(v))

        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Stage 1: Coarse bias (positional)
        if self.biaser_coarse is not None:
            bias_coarse = self.biaser_coarse(qh, kh, pre_q=q, pre_k=k)
            scores_coarse = scores + self.coarse_strength * bias_coarse
        else:
            scores_coarse = scores

        # Compute coarse attention (don't apply dropout yet)
        if attn_mask is not None:
            scores_coarse_masked = scores_coarse.masked_fill(attn_mask, float('-inf'))
        else:
            scores_coarse_masked = scores_coarse
        attn_coarse = F.softmax(scores_coarse_masked, dim=-1)

        # Stage 2: Fine bias (content-based, informed by coarse attention)
        if self.biaser_fine is not None:
            bias_fine = self.biaser_fine(qh, kh, pre_q=q, pre_k=k)
            # Weight fine bias by coarse attention (focus refinement where coarse is already attending)
            bias_fine = bias_fine * attn_coarse.detach()
            scores_fine = scores_coarse + self.fine_strength * bias_fine
        else:
            scores_fine = scores_coarse

        # Final attention
        if attn_mask is not None:
            scores_fine = scores_fine.masked_fill(attn_mask, float('-inf'))

        attn = F.softmax(scores_fine, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, vh)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)

        return out, attn
