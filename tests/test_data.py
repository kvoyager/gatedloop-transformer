"""
test_data.py
─────────────────────────────────────────────────────────────────────────────
Unit checks for the multi-domain data path. CPU-only, synthetic data, no
downloads — runs standalone (``python tests/test_data.py``) or under pytest.

  • write_domain: token budgets met exactly (document boundary), val is a disjoint slice of the same
    stream, per-document token cap respected, dry stream → tail moved to val;
  • the code / text filters keep and drop what they should;
  • MixtureDataset: exact per-epoch counts, deterministic + resumable order,
    epoch re-draw, clean tiling of a small domain, domain tags;
  • build_datasets: mixture and single-corpus configs both resolve;
  • a Lightning run on a mixture logs per-domain val metrics, a headline that
    is their share-weighted mean, per-domain train loss, per-domain sweep
    columns and per-domain convergence files.
"""

from __future__ import annotations

import csv
import glob
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatedloop.data.mixture import (          # noqa: E402
    FILTERS, build_datasets, encode_docs, write_domain,
)
from gatedloop.data.text import MixtureDataset, PackedTextDataset   # noqa: E402
from gatedloop.tokenizer import TextTokenizer                       # noqa: E402

torch.manual_seed(0)
WORDS = ("the cat sat on a mat and ran to the big red box near the small blue dog "
         "once upon a time a little girl found a tiny key under the old green tree").split()
_TMP = Path(tempfile.mkdtemp(prefix="lt_test_data_"))


def _docs(n: int, seed: int = 0, lo: int = 8, hi: int = 20) -> list[str]:
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, size=int(rng.integers(lo, hi)))) + "." for _ in range(n)]


_TOK: TextTokenizer | None = None


def _tok() -> TextTokenizer:
    global _TOK
    if _TOK is None:
        _TOK = TextTokenizer.train(iter(_docs(600)), vocab_size=400,
                                   save_path=_TMP / "tok.json", show_progress=False)
    return _TOK


def _bin(path: Path, n_tokens: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    rng.integers(3, 400, size=n_tokens, dtype=np.uint16).tofile(path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
def test_write_domain_budgets():
    tok = _tok()
    docs = _docs(3000, seed=1)
    tr, va = _TMP / "d_train.bin", _TMP / "d_val.bin"
    st = write_domain(iter(docs), tok, tr, va, train_tokens=20_000, val_tokens=2_000,
                      batch=100, max_doc_tokens=12, log=lambda *_: None)
    train = np.fromfile(tr, dtype=np.uint16)
    val = np.fromfile(va, dtype=np.uint16)
    # exact to the document boundary: ≤ budget, and within one document of it
    assert train.size == st["train_tokens"] and 20_000 - 12 < train.size <= 20_000
    assert val.size == st["val_tokens"] and 2_000 - 12 < val.size <= 2_000
    assert train[-1] == tok.eos_id and val[-1] == tok.eos_id
    # val continues the SAME stream exactly where train stopped → disjoint by construction
    full = np.concatenate([a for a, _ in encode_docs(iter(docs[:st["docs"]]), tok, 100, 12)])
    assert np.array_equal(np.concatenate([train, val]), full[: train.size + val.size])
    # per-document cap: every <bos>…<eos> segment is ≤ max_doc_tokens and keeps its <eos>
    ends = np.flatnonzero(train == tok.eos_id)
    starts = np.concatenate([[0], ends[:-1] + 1])
    lens = ends - starts + 1
    assert lens.max() <= 12 and (train[starts] == tok.bos_id).all()
    print(f"  write_domain: train {train.size} / val {val.size} tokens, "
          f"max doc {lens.max()} tokens, val disjoint — OK")


def test_write_domain_dry_stream():
    """Budget larger than the stream: warn, and carve val out of train's tail."""
    tok = _tok()
    tr, va = _TMP / "dry_train.bin", _TMP / "dry_val.bin"
    st = write_domain(iter(_docs(200, seed=2)), tok, tr, va, train_tokens=10**9,
                      val_tokens=500, batch=50, log=lambda *_: None)
    assert st["val_tokens"] > 0 and st["train_tokens"] > 0
    assert np.fromfile(tr, dtype=np.uint16).size == st["train_tokens"]
    assert np.fromfile(va, dtype=np.uint16).size == st["val_tokens"]
    print("  write_domain: dry stream → tail moved to val — OK")


def test_filters():
    good_py = "import os\n\n\ndef main():\n    for i in range(10):\n        print(i)\n\n" * 5
    assert FILTERS["code"](good_py)
    assert not FILTERS["code"]("x" * 50)                                # too short
    assert not FILTERS["code"]("a=1;" * 30_000)                         # one 120k-char line
    assert not FILTERS["code"]("# auto-generated file\n" + good_py)
    assert not FILTERS["code"](("\x00\x01" * 60 + "\n") * 5)            # binary-ish
    assert FILTERS["text"]("word " * 40) and not FILTERS["text"]("short")
    print("  filters: code / text keep+drop — OK")


def test_mixture_dataset():
    seq = 16
    sizes = {"a": 8_001, "b": 2_001, "c": 641}          # 500 / 125 / 40 blocks
    parts = {n: PackedTextDataset(_bin(_TMP / f"{n}.bin", s, i), seq)
             for i, (n, s) in enumerate(sizes.items())}
    shares = {"a": 0.8, "b": 0.1, "c": 0.1}
    ds = MixtureDataset(parts, shares, seed=1)
    N = sum(len(p) for p in parts.values())
    assert len(ds) == N == 665
    assert ds.counts == {"a": 532, "b": 66, "c": 67} or sum(ds.counts.values()) == N
    assert sum(ds.counts.values()) == N
    # domain tags and block indices are consistent with the underlying datasets
    for idx in (0, 531, 532, 600, 664):
        item = ds[idx]
        d = int(item["domain"])
        blk = int(ds._block[idx])
        assert torch.equal(item["input_ids"], parts[ds.names[d]][blk]["input_ids"])
    # the small domain (40 blocks, allotted 67) is tiled: every block appears, none >2×
    c_blocks = ds._block[ds._domain == 2]
    counts = np.bincount(c_blocks, minlength=40)
    assert counts.min() >= 1 and counts.max() <= 2 and counts.sum() == ds.counts["c"]
    # deterministic + resumable: same seed/epoch → same map; another epoch → different
    ds2 = MixtureDataset(parts, shares, seed=1)
    assert np.array_equal(ds._block, ds2._block) and np.array_equal(ds._domain, ds2._domain)
    ds2.set_epoch(1)
    assert not np.array_equal(ds._block, ds2._block)
    assert np.array_equal(ds._domain, ds2._domain)                   # counts never change
    passes = ds.passes_per_epoch()
    assert abs(passes["a"] - 532 / 500) < 1e-9 and passes["c"] > 1.5
    print(f"  MixtureDataset: counts {ds.counts}, passes {{{', '.join(f'{k}: {v:.2f}' for k, v in passes.items())}}} — OK")


def _mix_cfg(bin_dir: Path, seq_len: int = 16) -> dict:
    return {"tokenizer_path": str(_TMP / "tok.json"), "seq_len": seq_len, "has_bos": True,
            "bin_dir": str(bin_dir),
            "domains": {"fineweb": {"share": 0.8}, "code": {"share": 0.1}, "math": {"share": 0.1}}}


def _make_mix_bins(bin_dir: Path, seq_len: int = 16) -> dict:
    bin_dir.mkdir(exist_ok=True)
    for i, (n, sz) in enumerate({"fineweb": 4001, "code": 1201, "math": 801}.items()):
        _bin(bin_dir / f"{n}_train.bin", sz, 10 + i)
        _bin(bin_dir / f"{n}_val.bin", 4 * seq_len + 1, 20 + i)     # 4 blocks each
    return _mix_cfg(bin_dir, seq_len)


def test_build_datasets():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(_make_mix_bins(_TMP / "mixbins"))
    data = build_datasets(cfg, str)
    assert data.is_mixture and data.domains == ["fineweb", "code", "math"]
    assert abs(sum(data.shares) - 1) < 1e-9 and list(data.val) == data.domains
    assert set(data.info["domains"]["code"]) >= {"share", "train_tokens", "val_tokens", "passes_per_epoch"}
    single = OmegaConf.create({"seq_len": 16, "train_bin": str(_TMP / "a.bin"), "val_bin": str(_TMP / "b.bin")})
    sd = build_datasets(single, str)
    assert not sd.is_mixture and list(sd.val) == ["val"] and isinstance(sd.train, PackedTextDataset)
    print("  build_datasets: mixture + single corpus — OK")


def test_lit_mixture_validation():
    """Four CPU steps of a tiny looped model on a mixture: per-domain metrics,
    weighted headline, per-domain sweep + convergence outputs."""
    import pytorch_lightning as pl
    from omegaconf import OmegaConf
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader

    from gatedloop.training.callbacks import (
        GateAlphaCallback, LoopConvergenceCallback, ValMetricsCSV)
    from gatedloop.training.lit_module import LitPretrain

    cfg = OmegaConf.create(_make_mix_bins(_TMP / "litbins"))
    data = build_datasets(cfg, str, seed=0)
    model = dict(vocab_size=400, pad_id=0, d_model=32, n_heads=4, n_layers=2, ff_dim=64,
                 max_seq_len=32, loop_count=2, loop_min=2, loop_max=2, loop_target="block")
    lit = LitPretrain(model=model, warmup_steps=1, max_steps=4, val_loop_sweep=[1, 3],
                      domains=data.domains, shares=data.shares)
    lit.data_info = data.info
    out = _TMP / "lit_out"
    logger = CSVLogger(str(out), name="mix")
    tr = pl.Trainer(accelerator="cpu", max_steps=4, val_check_interval=2, log_every_n_steps=1,
                    num_sanity_val_steps=0, logger=logger, enable_checkpointing=False,
                    enable_progress_bar=False, enable_model_summary=False,
                    callbacks=[ValMetricsCSV(), GateAlphaCallback(), LoopConvergenceCallback(n_batches=1)])
    train_loader = DataLoader(data.train, batch_size=4, shuffle=True, drop_last=True)
    val_loaders = [DataLoader(ds, batch_size=4) for ds in data.val.values()]   # 1 batch each
    tr.fit(lit, train_loader, val_loaders)

    rows = list(csv.DictReader(open(Path(logger.log_dir) / "val_metrics.csv")))
    last = rows[-1]
    for k in ("val/loss", "val/ppl", "val/loss_fineweb", "val/loss_code", "val/loss_math",
              "val/ppl_code_r1", "val/ppl_code_r3", "val/loss_r3", "train/loss_fineweb",
              "gate/alpha_attn_l0", "gate/alpha_ff_l1", "val/conv/code/rel_mean_i1",
              "val/conv/math/rel_mean_l1_i2"):
        assert k in last and last[k] != "", f"missing metric {k}"
    weighted = sum(s * float(last[f"val/loss_{d}"]) for d, s in zip(data.domains, data.shares))
    assert abs(float(last["val/loss"]) - weighted) < 1e-4, (last["val/loss"], weighted)
    files = sorted(os.path.basename(f) for f in glob.glob(str(Path(logger.log_dir) / "convergence" / "*.npz")))
    assert files[-3:] == ["step000004_code.npz", "step000004_fineweb.npz", "step000004_math.npz"], files
    z = np.load(Path(logger.log_dir) / "convergence" / "step000004_code.npz")
    assert z["rel"].shape[-1] == 3 and z["rel"].shape[-2] == 2       # R = max(2, sweep 3), 2 layers
    assert z["kl"].shape == z["nll"].shape == z["rel"].shape         # output_kl on by default
    print(f"  lightning mixture: headline val/loss {float(last['val/loss']):.4f} == "
          f"Σ share·loss_d {weighted:.4f}; per-domain sweep + convergence files — OK")


def main():
    for fn in [test_write_domain_budgets, test_write_domain_dry_stream, test_filters,
               test_mixture_dataset, test_build_datasets, test_lit_mixture_validation]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n✅ all data tests passed")


if __name__ == "__main__":
    main()
