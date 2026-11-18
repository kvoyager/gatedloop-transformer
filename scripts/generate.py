"""
generate.py
─────────────────────────────────────────────────────────────────────────────
Sample text from a checkpoint — the qualitative counterpart to eval_ppl.py.

  python scripts/generate.py --ckpt runs/plain_10m/version_0/checkpoints/best.ckpt
  python scripts/generate.py --ckpt ... --prompt "Once upon a time" -n 4
  python scripts/generate.py --ckpt ... --loops 8            # more attention per layer
  python scripts/generate.py --ckpt ... --temperature 0      # greedy

``--loops`` only does something for a looped checkpoint: it re-runs the same
weights with a different number of attention iterations per layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatedloop.checkpoint import load_model
from gatedloop.tokenizer import TextTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, default="artifacts/tokenizer.json")
    ap.add_argument("--prompt", type=str, default="", help="empty → sample from <bos>")
    ap.add_argument("-n", "--n-samples", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8, help="<=0 → greedy")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--loops", type=int, default=None,
                    help="attention iterations per layer (default: the model's own)")
    ap.add_argument("--no-bos", action="store_true",
                    help="don't prefix the prompt with <bos> — use this when the corpus "
                         "the checkpoint was trained on did not mark document starts "
                         "(data.has_bos: false), so <bos> is an untrained token")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    tok_path = Path(args.tokenizer)
    tok = TextTokenizer.load(tok_path if tok_path.is_absolute() else root / tok_path)
    model, _ = load_model(args.ckpt, map_location="cpu")
    model.to(args.device)

    ids = ([] if args.no_bos else [tok.bos_id]) + (tok.encode(args.prompt) if args.prompt else [])
    if not ids:                                  # no prompt and no <bos>
        ids = [tok.eos_id]                       # …start from a document boundary
    prompt = torch.tensor([ids] * args.n_samples, dtype=torch.long, device=args.device)

    loops = args.loops if args.loops is not None else model.cfg.loop_count
    print(f"\n[{'LOOPED' if model.is_looped else 'PLAIN'} model, "
          f"{model.num_params()/1e6:.2f}M params, R={loops}, "
          f"{model.effective_depth(loops)} attention applications]\n")

    out = model.generate(
        prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None, eos_id=tok.eos_id, loops=loops,
    )
    for i in range(out.size(0)):
        text = tok.decode(out[i, len(ids):].tolist())
        print(f"── sample {i + 1} " + "─" * 50)
        print(args.prompt + text)
        print()


if __name__ == "__main__":
    main()
