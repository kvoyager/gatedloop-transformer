"""
test_model.py
─────────────────────────────────────────────────────────────────────────────
Unit checks for the architecture. Runs standalone (``python tests/test_model.py``)
or under pytest. CPU-only and small — a few seconds.

What is actually being verified:
  • a plain model is parameter-identical to a looped one (minus the α gates), so
    the ablation the configs promise is real;
  • the loop changes the computation (R matters) and only touches attention;
  • causality holds — no position can see the future;
  • RoPE gives genuinely *relative* positions;
  • truncated BPTT keeps gradients finite and bounded to the last k iterations;
  • both variants can overfit a single batch, i.e. the training path is wired up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatedloop.model.layers import RotaryEmbedding
from gatedloop.model.lm import ModelConfig, TransformerLM

torch.manual_seed(0)

BASE = dict(vocab_size=256, d_model=64, n_heads=4, n_layers=3, ff_dim=128,
            max_seq_len=64, dropout=0.0)


def _plain(**kw) -> TransformerLM:
    return TransformerLM(ModelConfig(**{**BASE, **kw}))


def _looped(r: int = 3, **kw) -> TransformerLM:
    return TransformerLM(ModelConfig(**{**BASE, "loop_count": r, "loop_min": r,
                                        "loop_max": r, **kw}))


def _looped_block(r: int = 3, **kw) -> TransformerLM:
    return _looped(r, loop_target="block", **kw)


def test_param_parity():
    """Looping must not add parameters beyond one α scalar per looping layer."""
    p, l = _plain(), _looped()
    assert l.num_params() - p.num_params() == BASE["n_layers"], (
        f"looped adds {l.num_params() - p.num_params()} params, expected "
        f"{BASE['n_layers']} (one α per layer)"
    )
    # ... and the two state dicts hold the same weight tensors, same shapes.
    pk, lk = dict(p.named_parameters()), dict(l.named_parameters())
    extra = set(lk) - set(pk)
    assert all("alpha_attn_logit" in k for k in extra), f"unexpected extra params: {extra}"
    for k in pk:
        assert pk[k].shape == lk[k].shape, f"shape drift on {k}"
    # loop_target="block" adds exactly one more gate (α_ff) per looping layer and
    # nothing else, so attn- and block-mode checkpoints differ only by those keys.
    b = _looped_block()
    bk = dict(b.named_parameters())
    ff_gates = set(bk) - set(lk)
    assert len(ff_gates) == BASE["n_layers"] and all("alpha_ff_logit" in k for k in ff_gates), \
        f"block mode changed the parameter set beyond α_ff: {ff_gates}"
    assert set(lk) <= set(bk) and all(bk[k].shape == lk[k].shape for k in lk)
    print(f"  plain {p.num_params():,} params | looped {l.num_params():,} "
          f"(+{len(extra)} α_attn; block mode +{len(ff_gates)} α_ff) — parity OK")


def test_step_embed_cost():
    """The optional per-iteration bias costs exactly R·d_model per looping layer."""
    a, b = _looped(3), _looped(3, loop_step_embed=True)
    expect = BASE["n_layers"] * 3 * BASE["d_model"]
    assert b.num_params() - a.num_params() == expect
    print(f"  step_embed costs {expect:,} params — OK")


def test_depth_and_loops_matter():
    m = _looped(4)
    assert m.effective_depth() == BASE["n_layers"] * 4
    assert m.effective_depth(1) == BASE["n_layers"]
    ids = torch.randint(0, BASE["vocab_size"], (2, 16))
    m.eval()
    with torch.no_grad():
        o1 = m(ids, loops=1).logits
        o2 = m(ids, loops=4).logits
    assert not torch.allclose(o1, o2), "R had no effect on the output"
    # A plain model must ignore the loop argument entirely.
    p = _plain()
    p.eval()
    with torch.no_grad():
        assert torch.allclose(p(ids, loops=1).logits, p(ids, loops=5).logits)
    print("  R changes looped output, and is inert on the plain model — OK")


def test_loop_layers_subset():
    m = _looped(3, loop_layers=[1])
    assert m.n_looping_layers == 1
    assert m.effective_depth() == 1 * 3 + (BASE["n_layers"] - 1)
    assert m.blocks[0].loop_enabled is False and m.blocks[1].loop_enabled is True
    print("  loop_layers=[1] loops exactly one layer — OK")


def test_causality():
    """Changing token t must not move the logits at any position < t."""
    for m in (_plain(), _looped(3), _looped_block(3)):
        m.eval()
        ids = torch.randint(0, BASE["vocab_size"], (1, 12))
        with torch.no_grad():
            a = m(ids).logits
            ids2 = ids.clone()
            ids2[0, 7] = (ids2[0, 7] + 1) % BASE["vocab_size"]
            b = m(ids2).logits
        assert torch.allclose(a[:, :7], b[:, :7], atol=1e-5), "future leaked into the past"
        assert not torch.allclose(a[:, 7:], b[:, 7:]), "the change had no effect at all"
    print("  causal masking holds for all variants — OK")


def test_rope_is_relative():
    """rot(q,i)·rot(k,j) must depend only on (i-j)."""
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    # The SAME vector placed at every position: then any content dependence is
    # factored out and only the rotation angle difference can matter.
    x = torch.randn(1, 1, 1, 16).expand(1, 32, 1, 16)
    r = rope(x)[0, :, 0]                       # (T, Dh)
    d = (r[5] * r[2]).sum()                    # distance 3
    for i, j in [(10, 7), (20, 17), (31, 28)]:
        assert torch.allclose(d, (r[i] * r[j]).sum(), atol=1e-4), "RoPE is not relative"
    print("  RoPE dot products depend only on relative distance — OK")


def test_truncated_bptt():
    """backprop_k bounds the graph without breaking gradient flow — in both
    loop_target modes."""
    ids = torch.randint(0, BASE["vocab_size"], (2, 16))
    for target in ("attn", "block"):
        full = _looped(4, loop_target=target)
        trunc = _looped(4, loop_target=target, loop_backprop_k=1)
        trunc.load_state_dict(full.state_dict())

        losses, grads = {}, {}
        for name, m in [("full", full), ("trunc", trunc)]:
            m.zero_grad()
            loss = m(ids, labels=ids).loss
            loss.backward()
            losses[name] = loss.item()
            # Regression guard: truncation must not starve the layers BELOW the
            # loop. (Detaching the running state instead of splicing a constant
            # update does exactly that — every parameter in block 0 comes back
            # with grad None.)
            for probe in ["blocks.0.attn.qW.weight", "blocks.0.ff.down.weight",
                          "embed.weight"]:
                g = dict(m.named_parameters())[probe].grad
                assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
                    f"{target}/{name}: no usable gradient on {probe}"
            grads[name] = m.blocks[0].attn.qW.weight.grad.clone()

        assert abs(losses["full"] - losses["trunc"]) < 1e-6, \
            f"{target}: truncation changed the forward value; it must only change the backward"
        assert not torch.allclose(grads["full"], grads["trunc"]), \
            f"{target}: truncated BPTT produced identical gradients — truncation is not active"
    print("  truncated BPTT (attn+block): same forward, bounded backward, "
          "no starved layers — OK")


def test_truncated_bptt_static_kv():
    """With static_kv, iteration 0 computes the K/V every later iteration reuses.
    Under truncation it must stay differentiable, or kW/vW of every looped layer
    get no gradient at all and stay frozen at their initial values."""
    ids = torch.randint(0, BASE["vocab_size"], (2, 16))
    for target in ("attn", "block"):
        full = _looped(4, loop_target=target, loop_static_kv=True)
        trunc = _looped(4, loop_target=target, loop_static_kv=True, loop_backprop_k=1)
        trunc.load_state_dict(full.state_dict())

        losses, qgrads = {}, {}
        for name, m in [("full", full), ("trunc", trunc)]:
            m.zero_grad()
            loss = m(ids, labels=ids).loss
            loss.backward()
            losses[name] = loss.item()
            for i in range(BASE["n_layers"]):
                for w in ("kW", "vW"):
                    g = getattr(m.blocks[i].attn, w).weight.grad
                    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
                        f"{target}/{name}: block {i} attn.{w} got no gradient"
            for probe in ["blocks.0.ff.down.weight", "embed.weight"]:
                g = dict(m.named_parameters())[probe].grad
                assert g is not None and g.abs().sum() > 0, f"{target}/{name}: no gradient on {probe}"
            qgrads[name] = m.blocks[1].attn.qW.weight.grad.clone()

        assert abs(losses["full"] - losses["trunc"]) < 1e-6, \
            f"{target}: truncation changed the forward value"
        assert not torch.allclose(qgrads["full"], qgrads["trunc"]), \
            f"{target}: truncation is not active with static_kv"
    print("  truncated BPTT + static_kv (attn+block): K/V projections keep their gradient — OK")


def test_loop_target_block():
    """loop_target="block" pulls the FFN into the loop — and nothing else."""
    # Same weights, different wiring: the two modes must disagree for R > 1.
    attn_m, block_m = _looped(3), _looped_block(3)
    # attn-mode weights load into block mode; only the block-only α_ff gates
    # are missing (they keep their init).
    missing, unexpected = block_m.load_state_dict(attn_m.state_dict(), strict=False)
    assert not unexpected and all("alpha_ff_logit" in k for k in missing)
    ids = torch.randint(0, BASE["vocab_size"], (2, 16))
    attn_m.eval(), block_m.eval()
    with torch.no_grad():
        assert not torch.allclose(attn_m(ids).logits, block_m(ids).logits), \
            "block mode computed the same function as attn mode"
        # R must matter in block mode too.
        assert not torch.allclose(block_m(ids, loops=1).logits,
                                  block_m(ids, loops=3).logits)
    # Non-looping blocks ignore loop_target: the plain arm of an ablation is the
    # same model under either setting, so comparisons stay controlled.
    p, pb = _plain(), _plain(loop_target="block")
    pb.load_state_dict(p.state_dict())
    assert all(not b.loop_ff for b in pb.blocks)
    p.eval(), pb.eval()
    with torch.no_grad():
        assert torch.equal(p(ids).logits, pb(ids).logits), \
            "loop_target changed a PLAIN model"
    print("  block mode: FFN loops, differs from attn mode, plain arm untouched — OK")


def test_loop_trace():
    """trace_loop() records rel / 1−cos per token with shape (B, T, L, R), finite
    and non-negative, only inside the context, and nothing for a plain model."""
    B, T, R = 2, 16, 5
    ids = torch.randint(0, BASE["vocab_size"], (B, T))
    for m in (_looped(), _looped_block()):
        m.eval()
        with torch.no_grad(), m.trace_loop() as tr:
            m.backbone(ids, loops=R)
        arrs = tr.stacked()
        assert tr.layers == list(range(BASE["n_layers"]))
        for name in ("rel", "one_minus_cos"):
            a = arrs[name]
            assert a.shape == (B, T, BASE["n_layers"], R), f"{name}: {tuple(a.shape)}"
            assert torch.isfinite(a).all() and (a >= -1e-6).all(), f"{name} has bad values"
        # the gated iterations (2..R) move the state less than the ungated first one
        assert arrs["rel"][..., 1:].mean() < arrs["rel"][..., 0].mean()
        assert all(b._trace is None for b in m.blocks), "trace not detached"
        assert m.gate_values() and all(0.0 < v < 1.0 for v in m.gate_values().values())
    plain = _plain()
    with torch.no_grad(), plain.trace_loop() as tr:
        plain.backbone(ids, loops=1)
    assert tr.layers == [] and plain.gate_values() == {}
    print(f"  trace shapes (B,T,L,R)=({B},{T},{BASE['n_layers']},{R}) — OK")


def test_loop_trace_output_kl():
    """output_kl adds kl / nll of shape (B, T, L, R); KL is finite and >= 0, and
    in block mode the last block's last-pass readout IS the model output (its nll
    equals the per-token cross-entropy of forward()); in attn mode the trailing
    FFN runs after the last traced pass, so it is not. Off by default."""
    B, T, R = 2, 16, 4
    ids = torch.randint(0, BASE["vocab_size"], (B, T))
    targets = torch.cat([ids[:, 1:], torch.full((B, 1), -100)], dim=1)
    for m in (_looped(), _looped_block(loop_static_kv=True)):
        m.eval()
        with torch.no_grad(), m.trace_loop() as tr:
            m.backbone(ids, loops=R)
        assert set(tr.stacked()) == {"rel", "one_minus_cos"}, "output KL recorded by default"
        with torch.no_grad(), m.trace_loop(output_kl=True, targets=targets) as tr:
            logits = m(ids, loops=R).logits
        arrs = tr.stacked()
        for name in ("kl", "nll"):
            assert arrs[name].shape == (B, T, BASE["n_layers"], R), f"{name}: {tuple(arrs[name].shape)}"
        assert torch.isfinite(arrs["kl"]).all() and (arrs["kl"] >= 0).all()
        assert torch.isnan(arrs["nll"][:, -1]).all() and torch.isfinite(arrs["nll"][:, :-1]).all()
        ce = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1),
            reduction="none").view(B, T - 1)
        exact = torch.allclose(arrs["nll"][:, :-1, -1, -1], ce, atol=1e-4)
        assert exact == m.blocks[-1].loop_ff, f"last pass = output: {exact}, block mode: {m.blocks[-1].loop_ff}"
    print(f"  kl / nll shapes (B,T,L,R)=({B},{T},{BASE['n_layers']},{R}), last pass = output (block mode) — OK")


def test_gate_init():
    """loop_gate_init sets both gates' starting value exactly; out-of-range is rejected."""
    for init in (0.1, 0.5, 0.9):
        m = _looped_block(2, loop_gate_init=init)
        vals = m.gate_values()
        assert len(vals) == 2 * BASE["n_layers"] and all(abs(v - init) < 1e-6 for v in vals.values()), vals
    try:
        _looped(2, loop_gate_init=1.0)
        raise AssertionError("loop_gate_init=1.0 should be rejected")
    except ValueError:
        pass
    print("  gate init 0.1 / 0.5 / 0.9 exact, 1.0 rejected — OK")


def test_generate_shapes():
    m = _looped(2)
    prompt = torch.randint(0, BASE["vocab_size"], (3, 5))
    out = m.generate(prompt, max_new_tokens=7, temperature=0.9, top_k=10)
    assert out.shape == (3, 12), out.shape
    greedy = m.generate(prompt, max_new_tokens=7, temperature=0.0)
    assert greedy.shape == (3, 12)
    assert torch.equal(out[:, :5], prompt), "generate must return the prompt then the sample"
    print("  generate returns prompt+continuation for both sampling modes — OK")


def test_overfit_single_batch():
    """Both variants must drive the loss on one fixed batch toward zero."""
    ids = torch.randint(0, BASE["vocab_size"], (2, 24))
    for name, m in [("plain", _plain()), ("looped", _looped(3)),
                    ("looped_block", _looped_block(3))]:
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        m.train()
        first = last = None
        for i in range(120):
            loss = m(ids, labels=ids).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = loss.item() if i == 0 else first
            last = loss.item()
        assert last < 0.5 * first, f"{name}: loss {first:.3f} → {last:.3f}, not learning"
        print(f"  {name}: overfit one batch {first:.3f} → {last:.3f} — OK")


def main():
    for fn in [test_param_parity, test_step_embed_cost, test_depth_and_loops_matter,
               test_loop_layers_subset, test_causality, test_rope_is_relative,
               test_truncated_bptt, test_truncated_bptt_static_kv, test_loop_target_block, test_loop_trace,
               test_loop_trace_output_kl, test_gate_init,
               test_generate_shapes,
               test_overfit_single_batch]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n✅ all model tests passed")


if __name__ == "__main__":
    main()
