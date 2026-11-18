"""
block.py
─────────────────────────────────────────────────────────────────────────────
The decoder block — and the loop.

Three variants come out of this one module, selected by two settings — how many
iterations (``loops``) and what one iteration wraps (``loop_target``):

  plain (loops = 1)                     looped-attention (loop_target="attn")
  ─────────────────────                 ─────────────────────────────────────
  x = x + attn(norm1(x))                repeat R times:  x = x + α·attn(norm1(x))
  x = x + ff(norm2(x))                  then once:       x = x + ff(norm2(x))

  looped-block (loop_target="block")
  ──────────────────────────────────
  repeat R times:  x = x + α·attn(norm1(x))
                   x = x + α·ff(norm2(x))

With ``loop_target="attn"`` the feed-forward runs exactly once per layer, at
the end, on whatever state the attention iterations converged to. With
``"block"`` the whole block recurs — the weight-tied block recurrence of
universal-transformer / TinyLlama's experimental ``Block`` — so the FFN is
applied (and paid for) R times as well, its write gated by the same α.

Why loop attention and not the whole block
------------------------------------------
Attention is the token-mixing operator: each extra iteration lets every position
re-read the sequence *through the representation the previous iteration built*,
which is where multi-hop composition ("who does 'she' refer to → what did she
own → …") has to come from. The FFN is a per-position feature map; re-running it
in the same loop mostly re-applies the same pointwise transform. So looping
attention alone buys the interesting part of extra depth — R× the token mixing —
at the cheapest activation cost (attention re-runs are O(R·N²) but the FFN, the
FLOP-heavy half at these widths, stays O(1) per layer). ``loop_target="block"``
exists to measure that argument rather than assume it: same gate, same
step-embed, same truncated BPTT — the only difference is one extra (gated) FFN
application per iteration.

Because the weights are *shared* across iterations, R is a pure compute setting:
parameters are unchanged (a looped model adds one scalar α per layer, plus an
optional per-iteration bias), so ``plain_10m`` vs ``looped_10m`` is a controlled
"does looping help?" ablation at matched parameter count.

Configuration
-------------
  loops (R)         how many loop iterations. Resolved by the LM and passed
                    down, so every layer runs the same R in a given forward.
  loop_target       what one iteration wraps: "attn" (default) re-runs attention
                    only and leaves the single FFN after the loop; "block" pulls
                    the FFN into the loop, one gated attn+FFN pass per iteration.
  gate (α)          learnable ``α = σ(θ)`` per write, ``gate_init`` at init: each
                    iteration is a partial-step update instead of a full-strength
                    write, so R iterations do not blow up the residual stream.
                    Two gates per looping block — ``alpha_attn`` on the attention
                    write and, with ``loop_target="block"``, ``alpha_ff`` on the
                    FFN write (the trailing FFN of ``"attn"`` mode is ungated, so
                    it has no ``alpha_ff``). Enabled only for looping layers — a
                    plain layer keeps the exact standard residual ``x + attn(...)``
                    and stays bit-identical to a vanilla pre-norm transformer.
  step_embed        optional learnable bias added to the attention input at
                    iteration i (zero-init, so it starts as a no-op). Breaks the
                    symmetry between iterations — the layer can tell "first pass"
                    from "third pass" and specialize. Costs R·d_model per layer.
  backprop_k        truncated BPTT: only the last k iterations carry gradients,
                    so R can be raised in the forward pass while the backward
                    graph (and activation memory) stays bounded. The skipped
                    iterations are spliced back in as a constant update rather
                    than detached outright — see ``forward``. With
                    ``static_kv``, iteration 0 always keeps its gradients (it
                    produces the K/V all later iterations reuse), so the graph
                    holds k + 1 iterations instead of k.
  loop_layers       which layers loop at all (default: all of them). Looping only
                    the middle of the stack is a common, cheaper variant.
  static_kv         compute K and V once, on the first iteration, and reuse
                    them for every later one — only the query is re-projected
                    from the updated residual. Cuts the per-iteration cost to
                    one projection plus the attention itself.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gatedloop.model.layers import CausalSelfAttention, RotaryEmbedding, SwiGLU, make_norm


class DecoderBlock(nn.Module):
    """Pre-norm decoder block whose attention sub-layer — or, with
    ``loop_target="block"``, the whole attn+FFN block — may be applied R times.

    ``loop_max`` is the largest R this block will ever be asked to run (it sizes
    the per-iteration bias table). ``loop_max <= 1`` makes the block a plain
    transformer layer: no gate, no step embedding, no loop bookkeeping.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        norm_type: str = "rms",
        qk_norm: bool = True,
        loop_max: int = 1,
        gate: bool = True,
        gate_init: float = 0.1,
        step_embed: bool = False,
        backprop_k: int | None = None,
        loop_target: str = "attn",
        static_kv: bool = False,
    ):
        super().__init__()
        self.norm1 = make_norm(norm_type, d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, qk_norm=qk_norm)
        self.norm2 = make_norm(norm_type, d_model)
        self.ff = SwiGLU(d_model, ff_dim, dropout)

        self.loop_max = max(1, int(loop_max))
        self.loop_enabled = self.loop_max > 1
        # "block" pulls the FFN into the loop. A non-looping layer always keeps
        # the single trailing FFN, whatever the model-level target says — so the
        # plain arm of an ablation is identical under either loop_target.
        self.loop_ff = loop_target == "block" and self.loop_enabled
        self.backprop_k = backprop_k
        # Static K/V: iterations 1..R-1 re-project only Q and attend over the
        # keys/values computed on iteration 0. Meaningless on a plain layer.
        self.static_kv = bool(static_kv) and self.loop_enabled

        # α gates and per-iteration bias exist only on looping layers, so a plain
        # model carries exactly the parameters of a vanilla transformer.
        self.gate = bool(gate) and self.loop_enabled
        if self.gate:
            logit = math.log(gate_init / (1.0 - gate_init))          # σ(logit) == gate_init
            self.alpha_attn_logit = nn.Parameter(torch.tensor([logit]))
            if self.loop_ff:
                self.alpha_ff_logit = nn.Parameter(torch.tensor([logit]))
        # Diagnostics hook: a LoopTrace attached by TransformerLM.trace_loop().
        # Plain attribute (not a buffer), so it never touches the state dict.
        self._trace: LoopTrace | None = None
        if step_embed and self.loop_enabled:
            self.step_embed = nn.Parameter(torch.zeros(self.loop_max, d_model))
        else:
            self.register_parameter("step_embed", None)

    @property
    def alpha_attn(self) -> torch.Tensor | None:
        return torch.sigmoid(self.alpha_attn_logit) if self.gate else None

    @property
    def alpha_ff(self) -> torch.Tensor | None:
        return torch.sigmoid(self.alpha_ff_logit) if (self.gate and self.loop_ff) else None

    def gate_values(self) -> dict[str, float]:
        """Current gate values as plain floats (``{}`` for a plain layer)."""
        out = {}
        if self.alpha_attn is not None:
            out["alpha_attn"] = float(self.alpha_attn.detach())
        if self.alpha_ff is not None:
            out["alpha_ff"] = float(self.alpha_ff.detach())
        return out

    def _loop_iters(self, x: torch.Tensor, rope: RotaryEmbedding, block_idx: int,
                    start: int, stop: int, kv: tuple | None = None,
                    last: bool = True,
                    ) -> tuple[torch.Tensor, tuple]:
        """Loop iterations ``start … stop-1`` (absolute indices, so the
        per-iteration bias is picked correctly even for a partial range).

        Iteration 0 is the full ungated pass that also produces the K/V later
        iterations reuse under ``static_kv``; a partial range starting above 0
        must be handed that ``kv`` from the earlier call. Returns ``(x, (k, v))``.
        ``last`` marks the range that ends the loop: only then is the single
        trailing FFN of ``"attn"`` mode applied, so a truncated-BPTT prefix
        (``last=False``) does not add it a second time."""
        a_attn = self.alpha_attn if self.gate else None
        a_ff = self.alpha_ff if self.gate else None
        trace = self._trace

        def sublayers(x, k, v, i, static_kv):
            h = self.norm1(x)
            if self.step_embed is not None:
                # Clamp so an eval-time R above loop_max still runs (reusing the
                # last iteration's bias) instead of indexing out of range.
                h = h + self.step_embed[min(i, self.loop_max - 1)]
            if static_kv:
                attn_x = self.attn.forward_kv(h, k, v, rope)
            else:
                attn_x, _, _ = self.attn(h, rope)
            x = x + (a_attn * attn_x if a_attn is not None else attn_x)
            if self.loop_ff:
                f = self.ff(self.norm2(x))
                x = x + (a_ff * f if a_ff is not None else f)
            return x

        if start == 0:
            # Iteration 0: always a full (ungated) pass that also produces the K/V
            # later iterations reuse under static_kv.
            if trace is not None:
                trace.start(block_idx, x)
            h = self.norm1(x)
            if self.step_embed is not None:
                h = h + self.step_embed[0]
            attn_x, k, v = self.attn(h, rope)
            x = x + attn_x
            if self.loop_ff:
                x = x + self.ff(self.norm2(x))
            if trace is not None:
                trace.record(block_idx, x)
            first = 1
        else:
            k, v = kv if kv is not None else (None, None)
            if self.static_kv and k is None:
                raise RuntimeError("static_kv: a partial loop range needs the K/V of iteration 0")
            first = start

        for i in range(first, stop):
            x = sublayers(x, k, v, i, self.static_kv)
            if trace is not None:
                trace.record(block_idx, x)

        if last and not self.loop_ff:
            x = x + self.ff(self.norm2(x))  # "attn" / plain: the FFN runs ONCE
        return x, (k, v)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding, block_idx, loops: int = 1) -> torch.Tensor:
        loops = max(1, int(loops)) if self.loop_enabled else 1
        k = self.backprop_k if self.backprop_k is not None else loops
        n_nograd = max(0, loops - int(k))

        if n_nograd and torch.is_grad_enabled():
            # Truncated BPTT. The early iterations run under no_grad (storing no
            # activations) and their NET UPDATE is spliced back in as a constant:
            # values are identical to running them normally, and gradient-wise
            # they act as the identity. That last part matters — detaching x
            # outright (the usual recipe for a single top-level recurrence) would
            # here cut the gradient path to every layer BELOW this one, since the
            # loop sits inside each layer rather than around the whole stack.
            kv, lo = None, 0
            if self.static_kv:
                # Iteration 0 computes the K/V that every later iteration reuses.
                # Run it WITH gradients: under no_grad those K/V would be
                # constants, so kW / vW would get no gradient at all and stay at
                # their initial values. Costs one extra iteration of activations.
                x, kv = self._loop_iters(x, rope, block_idx, 0, 1, last=False)
                lo = 1
            if n_nograd > lo:
                with torch.no_grad():
                    x_ng, kv_ng = self._loop_iters(x, rope, block_idx, lo, n_nograd,
                                                   kv=kv, last=False)
                x = x + (x_ng - x).detach()
                kv = kv if kv is not None else kv_ng
            start = n_nograd
        else:
            start, kv = 0, None

        x, _ = self._loop_iters(x, rope, block_idx, start, loops, kv=kv)
        return x


class LoopTrace:
    """Per-token convergence record of the loop — a diagnostics object attached
    to the looping blocks by ``TransformerLM.trace_loop()``.

    With ``s_0`` the block input and ``s_i`` the residual after iteration ``i``
    (attention write, plus the FFN write in ``"block"`` mode), it records for
    ``i = 1..R``:

        rel[i]           = ‖s_i − s_{i−1}‖ / max(‖s_{i−1}‖, 1e-6)
        one_minus_cos[i] = 1 − cos(s_i, s_{i−1})

    per token, in fp32 regardless of the forward's precision. Only the previous
    state is kept per block, so memory is O(1) in R. NB: iteration 0 writes
    ungated while later ones are scaled by α, so ``rel[1]`` is systematically
    the largest entry.

    Given a ``readout`` (hidden state → next-token log-probs; ``trace_loop``
    passes final norm + LM head), it also records the output-space change

        kl[i]  = KL(p_i ‖ p_{i−1}),   p_i = softmax(readout(s_i))

    and, if ``targets`` (B, T) is set (-100 = no target), the per-pass target
    loss ``nll[i] = −log p_i(target)`` (NaN where there is no target). With
    ``loop_target="block"``, p_R of the last block is the model's actual output;
    for earlier blocks it is a logit-lens readout. With ``"attn"`` every readout
    is taken before the block's single trailing FFN, so none is the exact output. Both are computed in row chunks from the two hidden
    states, so no (B, T, V) tensor is kept between iterations.
    """

    def __init__(self, readout=None, targets: torch.Tensor | None = None,
                 chunk_rows: int = 1024):
        self._prev: dict[int, torch.Tensor] = {}
        self.rel: dict[int, list[torch.Tensor]] = {}
        self.one_minus_cos: dict[int, list[torch.Tensor]] = {}
        self.readout = readout
        self.targets = targets
        self.chunk_rows = int(chunk_rows)
        self.kl: dict[int, list[torch.Tensor]] = {}
        self.nll: dict[int, list[torch.Tensor]] = {}

    @torch.no_grad()
    def start(self, block_idx: int, x: torch.Tensor) -> None:
        self._prev[block_idx] = x.detach().float()
        self.rel[block_idx] = []
        self.one_minus_cos[block_idx] = []
        if self.readout is not None:
            self.kl[block_idx] = []
            if self.targets is not None:
                self.nll[block_idx] = []

    @torch.no_grad()
    def record(self, block_idx: int, x: torch.Tensor) -> None:
        x = x.detach().float()
        prev = self._prev[block_idx]
        self.rel[block_idx].append((x - prev).norm(dim=-1) / prev.norm(dim=-1).clamp_min(1e-6))
        self.one_minus_cos[block_idx].append(1.0 - F.cosine_similarity(x, prev, dim=-1))
        if self.readout is not None:
            self._record_output(block_idx, x, prev)
        self._prev[block_idx] = x

    @torch.no_grad()
    def _record_output(self, block_idx: int, x: torch.Tensor, prev: torch.Tensor) -> None:
        """KL(p_x ‖ p_prev) and, with targets, −log p_x(target), chunked over rows."""
        shape, d = x.shape[:-1], x.shape[-1]
        xs, ps = x.reshape(-1, d), prev.reshape(-1, d)
        tg = None if self.targets is None else self.targets.reshape(-1)
        kl, nll = [], []
        for a in range(0, xs.shape[0], self.chunk_rows):
            b = a + self.chunk_rows
            lp, lq = self.readout(xs[a:b]), self.readout(ps[a:b])     # (rows, V) fp32 log-probs
            kl.append((lp.exp() * (lp - lq)).sum(-1).clamp_min(0.0))
            if tg is not None:
                t = tg[a:b]
                n = -lp.gather(-1, t.clamp_min(0)[:, None]).squeeze(-1)
                nll.append(n.masked_fill(t < 0, float("nan")))
        self.kl[block_idx].append(torch.cat(kl).view(shape))
        if tg is not None:
            self.nll[block_idx].append(torch.cat(nll).view(shape))

    @property
    def layers(self) -> list[int]:
        """Block indices that recorded, in stack order."""
        return sorted(self.rel)

    def stacked(self) -> dict[str, torch.Tensor]:
        """``{"rel": (B, T, L, R), "one_minus_cos": (B, T, L, R)}`` — L looping
        layers in the order of ``self.layers``, R iterations — plus ``"kl"`` and
        ``"nll"`` of the same shape when they were recorded."""
        layers = self.layers
        stores = [("rel", self.rel), ("one_minus_cos", self.one_minus_cos)]
        stores += [(n, st) for n, st in (("kl", self.kl), ("nll", self.nll)) if st]
        return {
            name: torch.stack([torch.stack(store[l], dim=-1) for l in layers], dim=-2)
            for name, store in stores
        }
