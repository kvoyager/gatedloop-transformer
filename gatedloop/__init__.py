"""
gatedloop
─────────────────────────────────────────────────────────────────────────────
Decoder-only language-model pretraining on plain text, with two architectures
that differ by exactly one setting:

  • plain  — a standard pre-norm transformer (RoPE, RMSNorm, SwiGLU, QK-norm);
  • looped — the same stack, but every block is applied R times with shared
    weights: pass 1 is ungated, and passes 2…R scale the attention and
    feed-forward updates by two learned scalar gates. R = 1 is exactly the
    plain model, and R is dialable at test time.

Code for K. Kolokolov, "Loop-Count Schedules in Looped Transformers: An
Empirical Study", arXiv:2611.29704 (2026). https://arxiv.org/abs/2611.29704

Scope is pretraining only: next-token prediction on a packed text corpus. No
instruction tuning, no task fine-tuning.
"""

from __future__ import annotations

__version__ = "0.1.0"
