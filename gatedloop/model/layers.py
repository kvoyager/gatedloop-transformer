"""
layers.py
─────────────────────────────────────────────────────────────────────────────
The primitive building blocks: RMSNorm, SwiGLU, rotary embeddings, and causal
self-attention. Nothing here knows about looping — that lives in ``block.py``.

Choices, and why
----------------
  • RMSNorm (no mean-centering, no bias) — lighter and numerically steadier than
    LayerNorm, and it matters more here than in a plain stack: the looped model
    re-applies the SAME attention weights several times per layer, so any
    instability compounds across iterations. Computed in fp32 and cast back, so
    it is safe under bf16 autocast.
  • RoPE on Q/K — relative positions with no learned position table to spend the
    parameter budget on. Implemented in-repo (~30 lines) rather than pulled from
    torchtune: one less dependency, and the cos/sin tables are plain
    non-persistent buffers so checkpoints stay portable across ``max_seq_len``.
  • QK-norm — RMSNorm on each head's Q and K before attention. Small LMs are
    prone to attention-logit blow-ups; this is a cheap, well-tested stabilizer,
    and again it is worth more under the reused attention of the looped variant.
  • SwiGLU FFN — the standard modern gated FFN; better quality per parameter
    than a GELU-MLP at this scale. Three matrices (gate, up, down), so
    ``ff_dim ≈ 8/3·d_model`` matches a vanilla 4·d_model GELU-MLP on parameters.

Attention is causal (decoder-only LM) over packed, unpadded pretraining blocks,
so ``is_causal=True`` is the whole mask story — no key-padding mask is needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root-mean-square layer norm. fp32 internals, original dtype out."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * xf).to(dt)


def make_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "rms":
        return RMSNorm(dim)
    if norm_type == "layer":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm_type {norm_type!r}; expected 'rms' or 'layer'")


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU feed-forward
# ─────────────────────────────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(d_model, ff_dim, bias=False)
        self.up = nn.Linear(d_model, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Rotary position embeddings
# ─────────────────────────────────────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    """RoPE (Su et al. 2021) in the GPT-NeoX / HF "half-split" layout: the head
    dimension is split in two halves that rotate into each other.

    Tables are precomputed to ``max_seq_len`` and registered NON-persistently,
    so they never enter the state dict and a checkpoint can be reloaded with a
    different ``max_seq_len``. Rotation is done in fp32 and cast back.

    Shapes: input/output ``(B, T, H, Dh)``; ``Dh`` must be even.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                       # (T, Dh/2)
        self.register_buffer("cos_cached", freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, :, None, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if T > self.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds the RoPE horizon max_seq_len="
                f"{self.max_seq_len}; raise model.max_seq_len"
            )
        cos = self.cos_cached[:, :T]                           # (1, T, 1, Dh/2)
        sin = self.sin_cached[:, :T]
        x1, x2 = x.float().chunk(2, dim=-1)
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return out.to(x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE causal self-attention (SDPA → flash kernel when available)
# ─────────────────────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional QK-norm.

    This module is the thing the looped variant re-applies: one call is one
    "attention step", and the layer that owns it decides how many steps to run.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, qk_norm: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qW = nn.Linear(d_model, d_model, bias=False)
        self.kW = nn.Linear(d_model, d_model, bias=False)
        self.vW = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = dropout
        self.proj_drop = nn.Dropout(dropout)
        self.q_norm = RMSNorm(self.d_head) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.d_head) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> tuple(torch.Tensor, torch.Tensor, torch.Tensor):
        B, N, _ = x.shape
        q = self.qW(x).reshape(B, N, self.n_heads, self.d_head)
        k = self.kW(x).reshape(B, N, self.n_heads, self.d_head)
        v = self.vW(x).reshape(B, N, self.n_heads, self.d_head)
        k_out = k
        v_out = v
        # q, k, v = qkv.unbind(dim=2)                # (B, N, H, Dh) each

        q = rope(self.q_norm(q))                   # QK-norm then RoPE
        k = rope(self.k_norm(k))

        q = q.transpose(1, 2)                      # (B, H, N, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, self.n_heads * self.d_head)
        return self.proj_drop(self.proj(out)), k_out, v_out

    def forward_kv(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.qW(x).reshape(B, N, self.n_heads, self.d_head)

        q = rope(self.q_norm(q))                   # QK-norm then RoPE
        k = rope(self.k_norm(k))

        q = q.transpose(1, 2)                      # (B, H, N, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, self.n_heads * self.d_head)
        return self.proj_drop(self.proj(out))
