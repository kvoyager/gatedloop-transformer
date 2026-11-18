"""
prepare_data.py
─────────────────────────────────────────────────────────────────────────────
Tokenize a text corpus once into flat uint16 ``.bin`` files for packed training.

  python scripts/prepare_data.py                                  # TinyStories
  python scripts/prepare_data.py --limit 50000                    # quick subset
  python scripts/prepare_data.py --dataset wikitext --config wikitext-103-raw-v1
  python scripts/prepare_data.py --files "data/corpus/*.txt" --val-frac 0.01

Writes ``artifacts/train.bin`` and ``artifacts/val.bin``. Each document is
wrapped in ``<bos> … <eos>`` so packed blocks carry document boundaries; tokens
are streamed to disk in chunks, so peak memory stays flat regardless of corpus
size. Requires the tokenizer (scripts/train_tokenizer.py) to exist first.

If the dataset has no validation split (or you pass ``--val-frac``), the tail of
the tokenized training stream is split off as validation — a held-out slice of
the same distribution, which is what a pretraining val/ppl is for.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatedloop.tokenizer import TextTokenizer

DTYPE = np.uint16


def _iter_files(pattern: str, limit: int):
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise SystemExit(f"no files matched {pattern!r}")
    for i, p in enumerate(paths):
        if limit and i >= limit:
            break
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        if text.strip():
            yield text


def _iter_hf(dataset: str, config: str | None, split: str, column: str,
             limit: int, streaming: bool):
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=streaming)
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        t = ex.get(column, "")
        if t and t.strip():
            yield t


def _encode_stream(docs, tok: TextTokenizer, batch: int):
    """Yield uint16 arrays of encoded documents, ``batch`` documents at a time."""
    buf: list[str] = []
    n_docs = 0
    for text in docs:
        buf.append(text)
        n_docs += 1
        if len(buf) >= batch:
            yield np.concatenate([np.asarray(ids, dtype=DTYPE)
                                  for ids in tok.encode_batch(buf)])
            buf.clear()
            if n_docs % 100_000 < batch:
                print(f"  … {n_docs} docs")
    if buf:
        yield np.concatenate([np.asarray(ids, dtype=DTYPE) for ids in tok.encode_batch(buf)])


def _write(docs, tok, out_path: Path, batch: int) -> int:
    n_tokens = 0
    with open(out_path, "wb") as f:
        for arr in _encode_stream(docs, tok, batch):
            f.write(arr.tobytes())
            n_tokens += arr.size
    return n_tokens


def _split_tail(bin_path: Path, val_path: Path, val_frac: float) -> tuple[int, int]:
    """Move the last ``val_frac`` of a written .bin into ``val_path``."""
    data = np.memmap(bin_path, dtype=DTYPE, mode="r")
    n = data.size
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    tail = np.array(data[n_train:])          # copy out (val is a small fraction)
    # Windows refuses to truncate a file while it is still mapped, so close the
    # mapping explicitly rather than trusting refcount timing.
    if getattr(data, "_mmap", None) is not None:
        data._mmap.close()
    del data
    tail.tofile(val_path)
    with open(bin_path, "r+b") as f:         # truncate the train file in place
        f.truncate(n_train * np.dtype(DTYPE).itemsize)
    return n_train, n_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, default="artifacts/tokenizer.json")
    ap.add_argument("--out-dir", type=str, default="artifacts")
    ap.add_argument("--prefix", type=str, default="", help="prefix for the .bin filenames")
    # source
    ap.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    ap.add_argument("--config", type=str, default=None, help="HF dataset config name")
    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--val-split", type=str, default="validation")
    ap.add_argument("--text-column", type=str, default="text")
    ap.add_argument("--files", type=str, default=None,
                    help="glob of local .txt files instead of an HF dataset")
    ap.add_argument("--val-frac", type=float, default=None,
                    help="hold out this fraction of the tokenized train stream as val "
                         "(default: only when no val split exists; local files use 0.005)")
    ap.add_argument("--batch", type=int, default=1000, help="documents per encode batch")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--streaming", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    tok = TextTokenizer.load(root / args.tokenizer)
    if tok.vocab_size > 65536:
        raise SystemExit(f"vocab_size {tok.vocab_size} does not fit uint16")
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_bin = out_dir / f"{args.prefix}train.bin"
    val_bin = out_dir / f"{args.prefix}val.bin"

    # ── train split ───────────────────────────────────────────────────────────
    if args.files:
        docs = _iter_files(args.files, args.limit)
        val_frac = 0.005 if args.val_frac is None else args.val_frac
    else:
        print(f"tokenizing {args.dataset} split={args.train_split} "
              f"(streaming={args.streaming})…")
        docs = _iter_hf(args.dataset, args.config, args.train_split,
                        args.text_column, args.limit, args.streaming)
        val_frac = args.val_frac
    n_train = _write(docs, tok, train_bin, args.batch)
    print(f"  wrote {train_bin}  ({n_train/1e6:.1f}M tokens, uint16)")

    # ── val split: the dataset's own, or a tail of the train stream ───────────
    made_val = False
    if not args.files and val_frac is None:
        try:
            print(f"tokenizing {args.dataset} split={args.val_split}…")
            docs = _iter_hf(args.dataset, args.config, args.val_split,
                            args.text_column, args.limit, args.streaming)
            n_val = _write(docs, tok, val_bin, args.batch)
            print(f"  wrote {val_bin}  ({n_val/1e6:.1f}M tokens, uint16)")
            made_val = n_val > 0
        except Exception as e:                       # no such split → fall through
            print(f"  no usable '{args.val_split}' split ({type(e).__name__}: {e});"
                  f" holding out a tail of train instead")
            val_frac = 0.005

    if not made_val and val_frac:
        n_tr, n_val = _split_tail(train_bin, val_bin, val_frac)
        print(f"  split tail: train {n_tr/1e6:.1f}M tokens → {train_bin}, "
              f"val {n_val/1e6:.1f}M tokens → {val_bin}")


if __name__ == "__main__":
    main()
