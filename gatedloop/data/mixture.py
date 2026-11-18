"""
mixture.py
─────────────────────────────────────────────────────────────────────────────
A multi-domain pretraining corpus: N sources, each tokenized into its OWN pair
of packed ``.bin`` files, mixed at train time by token share.

One YAML (``configs/data/mix_*.yaml``) is the single source of truth for the
mixture. Its ``domains`` block names each domain, its share, and where the
source lives on the HuggingFace Hub; the same block tells the training run
where that domain's bins are. So the prep scripts, the tokenizer trainer, the
training run and ``scripts/eval_ppl.py`` all read one file and cannot disagree
about what "code" means or how much of it there is.

Why per-domain bins instead of one pre-mixed file:
  • per-domain validation for free — ``val/ppl_code_r{R}`` next to
    ``val/ppl_fineweb_r{R}`` is the experiment's headline result;
  • the mix ratio is a training-time setting, not baked in at tokenization;
  • eval can report the per-domain table at every R from the same files.

Shares are defined in TOKENS: at prep time a domain gets ``share × total``
tokens, and at train time ``MixtureDataset`` draws ``share`` of every epoch's
blocks from it (blocks are equal-sized, so block share == token share).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
from omegaconf import DictConfig, OmegaConf

from gatedloop.data.text import MixtureDataset, PackedTextDataset
from gatedloop.tokenizer import TextTokenizer

DTYPE = np.uint16


def resolve_config_path(path: str | Path, root: Path) -> Path:
    """A config path given on the command line: absolute, relative to the repo
    root, or relative to the package (so ``configs/data/x.yaml`` finds
    ``gatedloop/configs/data/x.yaml``, also in an installed copy)."""
    p = Path(path)
    pkg = Path(__file__).resolve().parents[1]
    cands = [p] if p.is_absolute() else [root / p, root / "gatedloop" / p, pkg / p, pkg.parent / p, p]
    for cand in cands:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"config not found: {path} (tried repo root and gatedloop/)")


# ─────────────────────────────────────────────────────────────────────────────
# Spec
# ─────────────────────────────────────────────────────────────────────────────
class Domain:
    """One row of the ``domains`` block, with bin paths resolved."""

    def __init__(self, name: str, cfg, bin_dir: Path | None, resolve: Callable[[str], str]):
        self.name = name
        self.share = float(cfg.get("share", 0.0))
        # source (prep-time only)
        self.dataset = cfg.get("dataset")
        self.config = cfg.get("config")
        self.split = cfg.get("split", "train")
        self.column = cfg.get("column", "text")
        self.filter = cfg.get("filter", "text")
        # bins (train-time). Explicit paths win; else <bin_dir>/<name>_{train,val}.bin
        train_bin = cfg.get("train_bin")
        val_bin = cfg.get("val_bin")
        if train_bin is None:
            if bin_dir is None:
                raise ValueError(f"domain {name!r}: set bin_dir or train_bin")
            train_bin = str(bin_dir / f"{name}_train.bin")
        if val_bin is None and bin_dir is not None:
            val_bin = str(bin_dir / f"{name}_val.bin")
        self.train_bin = resolve(train_bin)
        self.val_bin = resolve(val_bin) if val_bin else None

    def __repr__(self) -> str:
        return f"Domain({self.name}, share={self.share:.2f}, {self.dataset})"


def load_domains(data_cfg: DictConfig | dict | str | Path,
                 resolve: Callable[[str], str] = str) -> list[Domain]:
    """Parse the ``domains`` block of a data config (path, dict or DictConfig).
    Shares are normalized to sum to 1. Returns ``[]`` for a single-corpus
    config (one without ``domains``)."""
    if isinstance(data_cfg, (str, Path)):
        data_cfg = OmegaConf.load(str(data_cfg))
    elif isinstance(data_cfg, dict):
        data_cfg = OmegaConf.create(data_cfg)
    dom_cfg = OmegaConf.select(data_cfg, "domains", default=None)
    if not dom_cfg:
        return []
    bin_dir = OmegaConf.select(data_cfg, "bin_dir", default=None)
    bin_dir = Path(resolve(bin_dir)) if bin_dir else None
    domains = [Domain(name, cfg, bin_dir, resolve) for name, cfg in dom_cfg.items()]
    total = sum(d.share for d in domains)
    if total <= 0:
        raise ValueError("mixture shares must be positive")
    for d in domains:
        d.share /= total
    return domains


# ─────────────────────────────────────────────────────────────────────────────
# Document filters (prep-time)
# ─────────────────────────────────────────────────────────────────────────────
def _keep_text(text: str) -> bool:
    return len(text.strip()) >= 100


def _keep_code(text: str) -> bool:
    """Drop the usual junk in scraped source files: huge or minified files,
    generated code, binary-ish content."""
    n = len(text)
    if n < 100 or n > 100_000:
        return False
    head = text[:500].lower()
    if "auto-generated" in head or "autogenerated" in head or "do not edit" in head:
        return False
    if max((len(line) for line in text.split("\n")), default=0) > 1000:
        return False
    probe = text[:20_000]
    if sum(c.isalnum() for c in probe) / max(1, len(probe)) < 0.25:
        return False
    return True


FILTERS: dict[str, Callable[[str], bool]] = {
    "none": lambda t: bool(t.strip()),
    "text": _keep_text,
    "code": _keep_code,
}


# ─────────────────────────────────────────────────────────────────────────────
# Streaming document sources
# ─────────────────────────────────────────────────────────────────────────────
def iter_hub_docs(domain: Domain, limit: int = 0, streaming: bool = True) -> Iterator[str]:
    """Stream the domain's Hub dataset, applying its filter. ``limit`` caps
    the number of documents YIELDED (0 = unlimited)."""
    from datasets import load_dataset

    keep = FILTERS[domain.filter]
    ds = load_dataset(domain.dataset, domain.config, split=domain.split, streaming=streaming)
    n = 0
    for ex in ds:
        text = ex.get(domain.column) or ""
        if not keep(text):
            continue
        yield text
        n += 1
        if limit and n >= limit:
            break


def iter_mixture_sample(domains: list[Domain], n_docs: int,
                        source: Callable[[Domain, int], Iterable[str]] = iter_hub_docs,
                        ) -> Iterator[str]:
    """A proportional sample of documents for training the tokenizer:
    ``round(share × n_docs)`` documents from each domain, in domain order."""
    for d in domains:
        n = max(1, round(d.share * n_docs))
        print(f"  [{d.name}] {n} documents for the tokenizer sample")
        yield from source(d, n)


# ─────────────────────────────────────────────────────────────────────────────
# Encoding + writing one domain
# ─────────────────────────────────────────────────────────────────────────────
def encode_docs(docs: Iterable[str], tok: TextTokenizer, batch: int = 1000,
                max_doc_tokens: int = 0) -> Iterator[tuple[np.ndarray, int]]:
    """Yield ``(uint16 array, n_docs)`` per encode batch. Each document is
    ``<bos> … <eos>``; one longer than ``max_doc_tokens`` is cut to that length
    (keeping its ``<eos>``) so a single giant file cannot dominate a stretch."""
    buf: list[str] = []

    def flush():
        arrs = []
        for ids in tok.encode_batch(buf):
            if max_doc_tokens and len(ids) > max_doc_tokens:
                ids = ids[:max_doc_tokens - 1] + [tok.eos_id]
            arrs.append(np.asarray(ids, dtype=DTYPE))
        out = (np.concatenate(arrs), len(buf))
        buf.clear()
        return out

    for text in docs:
        buf.append(text)
        if len(buf) >= batch:
            yield flush()
    if buf:
        yield flush()


def write_domain(docs: Iterable[str], tok: TextTokenizer, train_path: Path, val_path: Path,
                 train_tokens: int, val_tokens: int, batch: int = 1000,
                 max_doc_tokens: int = 0, log=print) -> dict:
    """Stream ``docs`` into ``train_path`` until ``train_tokens`` are written,
    then into ``val_path`` until ``val_tokens`` — so val is a disjoint slice of
    the same stream. Budgets are met exactly up to the nearest document
    boundary: an encode batch that would overshoot is cut at its last ``<eos>``
    inside the budget and the remainder carries over to the next file. If the
    stream runs dry before val has anything, the tail of train is moved into val."""
    train_path = Path(train_path)
    val_path = Path(val_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eos = tok.eos_id
    n_train = n_val = n_docs = 0
    done = {"train": False, "val": False}       # budget met (to a document boundary)
    f_train = open(train_path, "wb")
    f_val = open(val_path, "wb")
    try:
        for arr, n in encode_docs(docs, tok, batch, max_doc_tokens):
            n_docs += n
            while arr.size:
                if not done["train"]:
                    which, f, remaining = "train", f_train, train_tokens - n_train
                elif not done["val"]:
                    which, f, remaining = "val", f_val, val_tokens - n_val
                else:
                    break
                if arr.size > remaining:
                    # cut after the last document that ends inside the budget; the
                    # rest of the batch carries over to the next file
                    ends = np.flatnonzero(arr[:remaining] == eos)
                    cut = int(ends[-1]) + 1 if ends.size else 0
                    piece, arr = arr[:cut], arr[cut:]
                    done[which] = True
                else:
                    piece, arr = arr, arr[:0]
                f.write(piece.tobytes())
                if which == "train":
                    n_train += piece.size
                    done["train"] |= n_train >= train_tokens
                else:
                    n_val += piece.size
                    done["val"] |= n_val >= val_tokens
            if n_docs % 50_000 < n:
                log(f"    … {n_docs} docs  train {n_train/1e6:.1f}M  val {n_val/1e6:.2f}M tokens")
            if done["train"] and done["val"]:
                break
    finally:
        f_train.close()
        f_val.close()

    if not done["train"]:
        log(f"    ! stream ended early: {n_train/1e6:.1f}M of {train_tokens/1e6:.1f}M train tokens")
    if n_val == 0 and n_train > 0:
        n_move = min(val_tokens, n_train // 10)
        n_train, n_val = _move_tail(train_path, val_path, n_move)
        log(f"    ! no val docs left in the stream — moved the last {n_val/1e6:.2f}M train tokens to val")
    return {"docs": n_docs, "train_tokens": n_train, "val_tokens": n_val}


def _move_tail(train_path: Path, val_path: Path, n_move: int) -> tuple[int, int]:
    data = np.memmap(train_path, dtype=DTYPE, mode="r")
    n = data.size
    n_move = max(1, min(n_move, n - 1))
    tail = np.array(data[n - n_move:])
    if getattr(data, "_mmap", None) is not None:
        data._mmap.close()
    del data
    tail.tofile(val_path)
    with open(train_path, "r+b") as f:
        f.truncate((n - n_move) * np.dtype(DTYPE).itemsize)
    return n - n_move, n_move


def tokenizer_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def write_manifest(path: Path, tokenizer_path: str, tok: TextTokenizer,
                   domains: list[Domain], stats: dict[str, dict], extra: dict | None = None):
    manifest = {
        "tokenizer": {"path": str(tokenizer_path), "vocab_size": tok.vocab_size,
                      "sha256_16": tokenizer_fingerprint(tokenizer_path)},
        "domains": {
            d.name: {"share": d.share, "dataset": d.dataset, "config": d.config,
                     "split": d.split, "column": d.column, "filter": d.filter,
                     "train_bin": d.train_bin, "val_bin": d.val_bin, **stats.get(d.name, {})}
            for d in domains
        },
        **(extra or {}),
    }
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Train-time: datasets from a data config
# ─────────────────────────────────────────────────────────────────────────────
class DataBundle:
    """What ``build_datasets`` hands the train script.

    ``train``    the training Dataset (Packed or Mixture)
    ``val``      ordered ``{name: PackedTextDataset}`` — one entry ("val") for a
                 single corpus, one per domain for a mixture; empty if no val
    ``domains``  domain names in order, or ``None`` for a single corpus
    ``shares``   matching shares (``None`` for a single corpus)
    ``info``     per-domain token counts etc. for the banner
    """

    def __init__(self, train, val: dict, domains, shares, info: dict):
        self.train, self.val, self.domains, self.shares, self.info = train, val, domains, shares, info

    @property
    def is_mixture(self) -> bool:
        return self.domains is not None


def build_datasets(data_cfg: DictConfig, resolve: Callable[[str], str], seed: int = 0,
                   require_val: bool = False) -> DataBundle:
    seq_len = int(data_cfg.seq_len)
    domains = load_domains(data_cfg, resolve)

    if not domains:                                   # single corpus (the old path)
        train = PackedTextDataset(resolve(data_cfg.train_bin), seq_len)
        val_bin = resolve(OmegaConf.select(data_cfg, "val_bin", default=None))
        val = {}
        if val_bin and Path(val_bin).exists():
            val["val"] = PackedTextDataset(val_bin, seq_len)
        elif require_val:
            raise FileNotFoundError(f"val bin not found: {val_bin}")
        info = {"train_tokens": train.n_tokens,
                "val_tokens": val["val"].n_tokens if val else 0}
        return DataBundle(train, val, None, None, info)

    parts = {d.name: PackedTextDataset(d.train_bin, seq_len) for d in domains}
    shares = {d.name: d.share for d in domains}
    train = MixtureDataset(parts, shares, seed=seed)
    val = {}
    for d in domains:
        if d.val_bin and Path(d.val_bin).exists():
            val[d.name] = PackedTextDataset(d.val_bin, seq_len)
        elif require_val:
            raise FileNotFoundError(f"[{d.name}] val bin not found: {d.val_bin}")
    if val and len(val) != len(domains):
        missing = [d.name for d in domains if d.name not in val]
        raise FileNotFoundError(f"mixture: val bins missing for {missing} — every domain "
                                f"needs one (or none) so the weighted val/loss is well-defined")
    info = {
        "domains": {
            d.name: {"share": d.share, "train_tokens": parts[d.name].n_tokens,
                     "train_blocks": len(parts[d.name]),
                     "val_tokens": val[d.name].n_tokens if d.name in val else 0,
                     "epoch_blocks": train.counts[d.name],
                     "passes_per_epoch": train.passes_per_epoch()[d.name]}
            for d in domains
        },
        "train_tokens": sum(p.n_tokens for p in parts.values()),
        "epoch_blocks": len(train),
    }
    return DataBundle(train, val, [d.name for d in domains], [d.share for d in domains], info)
