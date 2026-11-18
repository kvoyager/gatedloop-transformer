"""Model package: RoPE causal attention, the attention-looped decoder block, and
the decoder-only LM that ties them together."""

from __future__ import annotations

from gatedloop.model.lm import LMOutput, ModelConfig, TransformerLM

__all__ = ["TransformerLM", "ModelConfig", "LMOutput"]
