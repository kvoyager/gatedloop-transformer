"""
prepare_mix.py
─────────────────────────────────────────────────────────────────────────────
Tokenize a multi-domain mixture into per-domain packed ``.bin`` pairs.

  python scripts/prepare_mix.py --data gatedloop/configs/data/mix_fwe_code_math.yaml
  python scripts/prepare_mix.py --data ... --total-tokens 5e7          # pipeline check
  python scripts/prepare_mix.py --data ... --domains code,math         # redo two domains

Reads the ``domains`` block of the data config (source dataset, share, filter)
and its ``prep`` block (token budgets), streams each source from the Hub, and
writes ``<bin_dir>/<domain>_train.bin`` + ``<domain>_val.bin`` plus a
``manifest.json`` with token counts and the tokenizer fingerprint. Domain d
gets ``share_d × total_tokens`` training tokens and ``val_tokens`` validation
tokens, the latter a disjoint slice of the same stream.

Requires the mixture tokenizer first:
  python scripts/train_tokenizer.py --mix gatedloop/configs/data/mix_fwe_code_math.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatedloop.data.mixture import (
    iter_hub_docs, load_domains, resolve_config_path, write_domain, write_manifest,
)
from gatedloop.tokenizer import TextTokenizer

ROOT = Path(__file__).resolve().parents[1]


def _abs(p: str | None) -> str | None:
    if p is None:
        return None
    q = Path(p)
    return str(q if q.is_absolute() else ROOT / q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="data config with a domains: block")
    ap.add_argument("--total-tokens", type=float, default=None,
                    help="override prep.total_tokens (train tokens across ALL domains)")
    ap.add_argument("--val-tokens", type=float, default=None, help="override prep.val_tokens (per domain)")
    ap.add_argument("--domains", type=str, default=None, help="comma-separated subset to (re)build")
    ap.add_argument("--batch", type=int, default=1000, help="documents per encode batch")
    ap.add_argument("--no-streaming", action="store_true", help="download datasets fully first")
    args = ap.parse_args()

    cfg = OmegaConf.load(str(resolve_config_path(args.data, ROOT)))
    domains = load_domains(cfg, _abs)
    if not domains:
        raise SystemExit(f"{args.data} has no domains: block — use scripts/prepare_data.py for a single corpus")
    prep = OmegaConf.select(cfg, "prep", default=None) or {}
    total = int(args.total_tokens or prep.get("total_tokens", 0))
    val_tokens = int(args.val_tokens or prep.get("val_tokens", 1_000_000))
    max_doc = int(prep.get("max_doc_tokens", 0))
    if total <= 0:
        raise SystemExit("set prep.total_tokens in the config or pass --total-tokens")

    tok_path = _abs(cfg.tokenizer_path)
    tok = TextTokenizer.load(tok_path)
    if tok.vocab_size > 65536:
        raise SystemExit(f"vocab_size {tok.vocab_size} does not fit uint16")

    selected = set(args.domains.split(",")) if args.domains else {d.name for d in domains}
    unknown = selected - {d.name for d in domains}
    if unknown:
        raise SystemExit(f"unknown domains {sorted(unknown)}; config has {[d.name for d in domains]}")

    bin_dir = Path(domains[0].train_bin).parent
    bin_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bin_dir / "manifest.json"
    stats: dict[str, dict] = {}
    if manifest_path.exists() and args.domains:           # keep stats of untouched domains
        try:
            old = OmegaConf.to_container(OmegaConf.load(manifest_path))
            stats = {k: {kk: v[kk] for kk in ("docs", "train_tokens", "val_tokens") if kk in v}
                     for k, v in old.get("domains", {}).items()}
        except Exception:
            pass

    print(f"mixture: {total/1e9:.2f}B train tokens total, {val_tokens/1e6:.1f}M val per domain, "
          f"vocab {tok.vocab_size}, max_doc_tokens {max_doc or 'unbounded'}")
    for d in domains:
        if d.name not in selected:
            continue
        budget = int(round(d.share * total))
        print(f"\n[{d.name}] share {d.share:.2f} → {budget/1e6:.1f}M train tokens  "
              f"({d.dataset}{'/' + d.config if d.config else ''}, split={d.split}, filter={d.filter})")
        t0 = time.perf_counter()
        docs = iter_hub_docs(d, streaming=not args.no_streaming)
        st = write_domain(docs, tok, d.train_bin, d.val_bin, budget, val_tokens,
                          batch=args.batch, max_doc_tokens=max_doc)
        st["seconds"] = round(time.perf_counter() - t0, 1)
        stats[d.name] = st
        print(f"  wrote {d.train_bin}  ({st['train_tokens']/1e6:.1f}M tokens)\n"
              f"  wrote {d.val_bin}  ({st['val_tokens']/1e6:.2f}M tokens)   "
              f"[{st['docs']} docs, {st['seconds']}s]")
        write_manifest(manifest_path, tok_path, tok, domains, stats,
                       extra={"total_tokens": total, "val_tokens": val_tokens,
                              "max_doc_tokens": max_doc, "seq_len": int(cfg.seq_len)})
    print(f"\nmanifest → {manifest_path}")


if __name__ == "__main__":
    main()
