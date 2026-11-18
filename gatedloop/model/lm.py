"""
lm.py
─────────────────────────────────────────────────────────────────────────────
The language model: token embeddings → N decoder blocks → final norm → LM head.

One class covers both families in the repo, selected by ``loop_count``:

  loop_count = 1        a plain pre-norm decoder-only transformer
  loop_count = R > 1    the same stack, but every layer applies its ATTENTION
                        sub-layer R times (shared weights) before its single FFN
                        — or, with ``loop_target="block"``, the whole attn+FFN
                        block R times

Parameters are identical between the two apart from the gate scalars α per layer (and
the optional per-iteration bias), so a matched pair — ``plain_10m`` vs
``looped_10m`` — isolates the effect of the loop.

Test-time compute
-----------------
With ``loop_sample=true`` the training loop draws R ~ U[loop_min, loop_max] each
step, so the model learns to produce a usable distribution across a *range* of
compute budgets. At eval you can then dial R up or down (``--loops 8``) and read
off the compute/quality curve — that is what ``scripts/eval_ppl.py`` sweeps and
what ``val_loop_sweep`` logs during training.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from gatedloop.model.block import DecoderBlock, LoopTrace
from gatedloop.model.layers import RotaryEmbedding, make_norm


@dataclass
class ModelConfig:
    # vocabulary / token ids (injected at runtime from the tokenizer)
    vocab_size: int = 8192
    pad_id: int = 0
    # width / depth
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 4
    ff_dim: int = 1024              # SwiGLU inner dim (≈ 8/3·d_model for param parity)
    max_seq_len: int = 1024         # RoPE horizon; must cover data.seq_len + 1
    dropout: float = 0.0
    # the loop — loop_count == 1 → plain transformer
    loop_count: int = 1             # R used at eval / when not sampling
    loop_min: int = 1               # training: sample R ~ U[loop_min, loop_max]
    loop_max: int = 1
    loop_sample: bool = False
    loop_target: str = "attn"       # "attn": FFN once, after the loop | "block":
                                    # attn+FFN both inside each iteration
    loop_gate: bool = True          # learnable α_attn (and α_ff in block mode) per write
    loop_gate_init: float = 0.1     # initial α (σ(θ) at init) for both gates
    loop_step_embed: bool = False   # per-iteration learnable input bias
    loop_backprop_k: int | None = None          # truncated BPTT depth (None → full)
    loop_layers: Sequence[int] | None = None    # which layers loop (None → all)
    loop_static_kv: bool = False    # reuse iteration-0 K/V across the loop; only Q
                                    # is re-projected each iteration
    # misc
    norm_type: str = "rms"          # rms | layer
    qk_norm: bool = True
    rope_base: float = 10_000.0
    tie_embeddings: bool = True
    init_std: float = 0.02


class LMOutput(NamedTuple):
    logits: torch.Tensor
    loss: torch.Tensor | None
    hidden: torch.Tensor


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg = self._validate(cfg)
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_base)

        loop_layers = set(range(cfg.n_layers) if cfg.loop_layers is None else cfg.loop_layers)
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                ff_dim=cfg.ff_dim,
                dropout=cfg.dropout,
                norm_type=cfg.norm_type,
                qk_norm=cfg.qk_norm,
                # a layer outside loop_layers is built as a plain layer
                loop_max=cfg.loop_max if i in loop_layers else 1,
                gate=cfg.loop_gate,
                gate_init=cfg.loop_gate_init,
                step_embed=cfg.loop_step_embed,
                backprop_k=cfg.loop_backprop_k,
                loop_target=cfg.loop_target,
                static_kv=cfg.loop_static_kv,
            )
            for i in range(cfg.n_layers)
        ])
        self.final_norm = make_norm(cfg.norm_type, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)
        # GPT-2 style depth scaling: shrink every projection that WRITES to the
        # residual stream so the stream's variance does not grow with depth. A
        # looped layer writes loop_count times (attention) + once (FFN) — or
        # loop_count times EACH with loop_target="block" — so the number of
        # residual writes, not the layer count, is the right divisor (a plain
        # model recovers the familiar 1/sqrt(2·n_layers)). The α gate damps
        # the looped writes further; ignoring it only makes this init slightly
        # conservative.
        n_loop = self.n_looping_layers
        attn_writes = n_loop * cfg.loop_count + (cfg.n_layers - n_loop)
        ffn_writes = attn_writes if cfg.loop_target == "block" else cfg.n_layers
        writes = attn_writes + ffn_writes
        scale = 1.0 / math.sqrt(max(1, writes))
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std * scale)

    # ── config sanity ─────────────────────────────────────────────────────────
    @staticmethod
    def _validate(cfg: ModelConfig) -> ModelConfig:
        if cfg.d_model % cfg.n_heads:
            raise ValueError(f"d_model {cfg.d_model} not divisible by n_heads {cfg.n_heads}")
        if cfg.loop_layers is not None:
            bad = [i for i in cfg.loop_layers if not 0 <= i < cfg.n_layers]
            if bad:
                raise ValueError(f"loop_layers {bad} out of range for n_layers={cfg.n_layers}")
        if cfg.loop_min < 1 or cfg.loop_count < 1:
            raise ValueError("loop_min and loop_count must be >= 1")
        if not 0.0 < cfg.loop_gate_init < 1.0:
            raise ValueError(f"loop_gate_init must be in (0, 1), got {cfg.loop_gate_init}")
        if cfg.loop_target not in ("attn", "block"):
            raise ValueError(f"loop_target must be 'attn' or 'block', got {cfg.loop_target!r}")
        # loop_max bounds the sampling range and the per-iteration bias table.
        cfg.loop_max = max(cfg.loop_max, cfg.loop_count, cfg.loop_min)
        return cfg

    # ── init ──────────────────────────────────────────────────────────────────
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    # ── compute-budget resolution ─────────────────────────────────────────────
    def _resolve_loops(self, loops: int | None) -> int:
        """Pick R for this forward pass. NB: the sampling draw is per-process —
        under DDP each rank would draw its own R and average gradients taken
        through *different* computations. Broadcast a shared R (or set
        loop_sample=false) before scaling past one device."""
        if loops is not None:
            return max(1, int(loops))
        if self.training and self.cfg.loop_sample:
            return int(torch.randint(self.cfg.loop_min, self.cfg.loop_max + 1, (1,)).item())
        return self.cfg.loop_count

    # ── backbone ──────────────────────────────────────────────────────────────
    def backbone(self, input_ids: torch.Tensor, loops: int) -> torch.Tensor:
        x = self.drop(self.embed(input_ids))
        for block_idx, block in enumerate(self.blocks):
            x = block(x, self.rope, block_idx, loops)
        return self.final_norm(x)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        loops: int | None = None,
    ) -> LMOutput:
        """``labels`` are UNSHIFTED (same alignment as ``input_ids``); the causal
        shift happens here, so position t is scored against token t+1 and
        positions labelled -100 are ignored."""
        loops = self._resolve_loops(loops)
        hidden = self.backbone(input_ids, loops)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return LMOutput(logits=logits, loss=loss, hidden=hidden)

    # ── sampling (eval only) ──────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int | None = 50,
        eos_id: int | None = None,
        loops: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling; returns prompt + continuation, ``(B, T+T_new)``.

        No KV cache: every step is a full forward over the whole prefix. That is
        O(T²), but it keeps the looped path exactly identical to training (a
        cache for looped attention would have to be keyed by iteration as well
        as by layer), and sampling here is a sanity check, not a serving path.
        ``temperature <= 0`` selects greedy decoding.
        """
        was_training = self.training
        self.eval()
        loops = self._resolve_loops(loops)
        out = input_ids
        done = torch.zeros(out.size(0), dtype=torch.bool, device=out.device)

        for _ in range(max_new_tokens):
            ctx = out[:, -self.cfg.max_seq_len:]
            logits = self.lm_head(self.backbone(ctx, loops)[:, -1])      # (B, V)
            if temperature and temperature > 0:
                logits = logits / temperature
                if top_k:
                    k = min(int(top_k), logits.size(-1))
                    kth = logits.topk(k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                nxt = logits.argmax(dim=-1)
            if eos_id is not None:
                nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
                done = done | (nxt == eos_id)
            out = torch.cat([out, nxt.unsqueeze(1)], dim=1)
            if eos_id is not None and bool(done.all()):
                break

        if was_training:
            self.train()
        return out

    # ── diagnostics ───────────────────────────────────────────────────────────
    @contextmanager
    def trace_loop(self, output_kl: bool = False, targets: torch.Tensor | None = None):
        """Record per-token loop convergence for forwards run inside the block:

            with model.trace_loop() as tr:
                model.backbone(ids, loops=R)
            tr.stacked()["rel"]        # (B, T, L, R)

        See ``LoopTrace``. Diagnostics only — adds two fp32 norms per token per
        iteration; nothing is recorded outside the context. ``output_kl=True``
        also records the successive-output KL through final norm + LM head (two
        fp32 readouts per token per iteration), and with ``targets`` (next-token
        ids aligned with the input, -100 = none) the per-pass target loss."""
        trace = LoopTrace(readout=self._readout_logp if output_kl else None, targets=targets)
        for b in self.blocks:
            if b.loop_enabled:
                b._trace = trace
        try:
            yield trace
        finally:
            for b in self.blocks:
                b._trace = None

    def _readout_logp(self, h: torch.Tensor) -> torch.Tensor:
        """Next-token log-probs of a residual state in fp32 with autocast off —
        the readout the loop trace uses for its output-space metrics."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            logits = F.linear(self.final_norm(h.float()), self.lm_head.weight.float())
        return F.log_softmax(logits, dim=-1)

    def gate_values(self) -> dict[str, float]:
        """``{"alpha_attn_l{i}": …, "alpha_ff_l{i}": …}`` for every gated block."""
        return {f"{name}_l{i}": v
                for i, b in enumerate(self.blocks) for name, v in b.gate_values().items()}

    # ── bookkeeping ───────────────────────────────────────────────────────────
    @property
    def n_looping_layers(self) -> int:
        return sum(1 for b in self.blocks if b.loop_enabled)

    @property
    def is_looped(self) -> bool:
        return self.n_looping_layers > 0

    def effective_depth(self, loops: int | None = None) -> int:
        """Attention applications per forward pass (the FFN runs n_layers times
        with loop_target="attn", or matches this count with "block").
        A plain 4-layer model gives 4; 4 layers at R=4 give 16."""
        r = self.cfg.loop_count if loops is None else int(loops)
        return self.n_looping_layers * r + (self.cfg.n_layers - self.n_looping_layers)

    def num_params(self, trainable_only: bool = True, include_embedding: bool = True) -> int:
        seen, total = set(), 0
        for name, p in self.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if id(p) in seen:                       # tied weight counted once
                continue
            if not include_embedding and ("embed" in name or name.startswith("lm_head")):
                continue
            seen.add(id(p))
            total += p.numel()
        return total
