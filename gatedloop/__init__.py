"""
gatedloop
─────────────────────────────────────────────────────────────────────────────
Decoder-only language-model pretraining on plain text, with two architectures
that differ by exactly one setting:

  • plain  — a standard pre-norm transformer (RoPE, RMSNorm, SwiGLU, QK-norm);
  • looped — the same stack, but every layer re-applies its ATTENTION sub-layer
    R times with shared weights and a learnable gated residual before running
    its single feed-forward. R× the token mixing at unchanged parameter count,
    and R is dialable at test time.

Scope is pretraining only: next-token prediction on a packed text corpus. No
instruction tuning, no task fine-tuning.
"""

from __future__ import annotations

__version__ = "0.1.0"
