"""
train_tokenizer.py
─────────────────────────────────────────────────────────────────────────────
Train the byte-level BPE and save it to artifacts/. Run this once, before
prepare_data.py.

  python scripts/train_tokenizer.py                                   # TinyStories, 8192
  python scripts/train_tokenizer.py --vocab-size 16384
  python scripts/train_tokenizer.py --dataset wikitext --config wikitext-103-raw-v1
  python scripts/train_tokenizer.py --files "data/*.txt"
  python scripts/train_tokenizer.py --limit 200000 --streaming          # quick pass
  python scripts/train_tokenizer.py --mix gatedloop/configs/data/mix_fwe_code_math.yaml
                                     # mixture: proportional sample of every domain,
                                     # vocab / output path from the config's prep block

Vocabulary size is the main lever on the embedding table (vocab × d_model, tied),
which is a third of a 10M model's parameters — keep it small for small models.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root on sys.path

from gatedloop.tokenizer import TextTokenizer


def _iter_files(pattern: str, limit: int):
    """Yield documents from local text files (one document per file)."""
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

    print(f"loading {dataset}"
          f"{'/' + config if config else ''} (split={split}, streaming={streaming})…")
    ds = load_dataset(dataset, config, split=split, streaming=streaming)
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        t = ex.get(column, "")
        if t and t.strip():
            yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--out", type=str, default="artifacts/tokenizer.json")
    ap.add_argument("--min-frequency", type=int, default=2)
    # source: either an HF dataset or local files
    ap.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    ap.add_argument("--config", type=str, default=None, help="HF dataset config name")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--text-column", type=str, default="text")
    ap.add_argument("--files", type=str, default=None,
                    help="glob of local .txt files to train on instead of an HF dataset")
    ap.add_argument("--limit", type=int, default=0, help="cap #documents (0 = all)")
    ap.add_argument("--streaming", action="store_true",
                    help="stream the dataset instead of downloading it (pairs with --limit)")
    ap.add_argument("--mix", type=str, default=None,
                    help="data config with a domains: block → train on a share-proportional "
                         "sample (prep.tokenizer_docs documents, or --limit); --vocab-size and "
                         "--out default to prep.vocab_size and tokenizer_path")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.mix:
        from omegaconf import OmegaConf
        from gatedloop.data.mixture import (
            iter_mixture_sample, load_domains, resolve_config_path)

        cfg = OmegaConf.load(str(resolve_config_path(args.mix, root)))
        prep = OmegaConf.select(cfg, "prep", default=None) or {}
        domains = load_domains(cfg, lambda q: str(Path(q) if Path(q).is_absolute() else root / q))
        if not domains:
            raise SystemExit(f"{args.mix} has no domains: block")
        if "--vocab-size" not in sys.argv[1:] and prep.get("vocab_size"):
            args.vocab_size = int(prep["vocab_size"])
        if "--out" not in sys.argv[1:]:
            args.out = str(cfg.tokenizer_path)
        n_docs = args.limit or int(prep.get("tokenizer_docs", 300_000))
        print(f"mixture tokenizer sample: {n_docs} documents across "
              f"{', '.join(f'{d.name} {d.share:.0%}' for d in domains)}")
        docs = iter_mixture_sample(domains, n_docs)
        out = Path(args.out)
        out = out if out.is_absolute() else root / out
        print(f"training BPE vocab_size={args.vocab_size} → {out}")
        tok = TextTokenizer.train(docs, vocab_size=args.vocab_size, save_path=out,
                                  min_frequency=args.min_frequency)
        print(f"done. vocab_size={tok.vocab_size}  "
              f"pad={tok.pad_id} bos={tok.bos_id} eos={tok.eos_id}")
        return

    if args.vocab_size > 65536:
        raise SystemExit("vocab_size > 65536 does not fit the uint16 .bin format "
                         "used by prepare_data.py")

    docs = (_iter_files(args.files, args.limit) if args.files else
            _iter_hf(args.dataset, args.config, args.split, args.text_column,
                     args.limit, args.streaming))

    out = Path(__file__).resolve().parents[1] / args.out
    print(f"training BPE vocab_size={args.vocab_size} → {out}")
    tok = TextTokenizer.train(docs, vocab_size=args.vocab_size, save_path=out,
                              min_frequency=args.min_frequency)
    print(f"done. vocab_size={tok.vocab_size}  "
          f"pad={tok.pad_id} bos={tok.bos_id} eos={tok.eos_id}")


if __name__ == "__main__":
    main()
