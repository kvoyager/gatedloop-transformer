"""
optim.py
─────────────────────────────────────────────────────────────────────────────
Optimizer + LR-schedule helpers.

build_optimizer
  AdamW with the standard two-group weight-decay split: decay the 2-D weight
  matrices (Linear / attention projections), never the 1-D params (RMSNorm
  gains, biases, the loop gate α, the per-iteration bias rows are 2-D but are
  excluded by name). The tied lm_head shares the embedding tensor, so it is
  de-duplicated by object id and lands in its group exactly once.

cosine_warmup_lambda
  Linear warmup → cosine decay to ``floor`` × peak over ``decay_steps``, then a
  flat floor. The decay horizon is decoupled from ``max_steps`` (the training
  budget) so extending a run to resume longer does not reshape the curve — pin
  ``decay_steps`` to the ORIGINAL ``max_steps`` for a continuous lr at resume.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Parameters that are 2-D but should not be weight-decayed: the token embedding
# table and the loop's per-iteration bias (an additive offset, not a weight).
_NO_DECAY_NAMES = ("embed.weight", "step_embed")


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    decay, no_decay, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        skip = p.dim() < 2 or any(k in name for k in _NO_DECAY_NAMES)
        (no_decay if skip else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # NB: do NOT use fused=True here — fused AdamW unscales grads internally,
    # which Lightning's AMP gradient-clipping path rejects ("does not allow for
    # gradient clipping"). torch already picks the fast foreach kernel on CUDA.
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def cosine_warmup_lambda(warmup_steps: int, decay_steps: int | None, floor: float = 0.1):
    """Return an lr-lambda for ``torch.optim.lr_scheduler.LambdaLR`` (multiplies
    the per-group base lr). ``floor`` is the fraction of peak the cosine decays
    to (and holds at) once ``decay_steps`` is reached."""
    warmup_steps = max(0, int(warmup_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if not decay_steps or decay_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda
