"""
eval_ppl.py
─────────────────────────────────────────────────────────────────────────────
Held-out perplexity of a checkpoint — and, for a looped model, the test-time
compute curve: the same weights evaluated at several attention-loop counts.

  python scripts/eval_ppl.py --ckpt runs/looped_10m/version_0/checkpoints/best.ckpt
  python scripts/eval_ppl.py --ckpt ... --loops 1,2,4,6,8 --max-batches 100
  python scripts/eval_ppl.py --ckpt ... --bin artifacts/val.bin --seq-len 512
  python scripts/eval_ppl.py --ckpt ... --data gatedloop/configs/data/mix_fwe_code_math.yaml --loops 1,2,3,4,6
                                        # mixture: one row per domain per R, plus the
                                        # share-weighted "mix" row

Reading the table: a looped model trained at R=4 should be worse at R=1 (it is
being run below the compute it was trained for) and roughly flat or slowly
improving above 4 — the point where it stops improving is the honest end of the
"think longer" story. A plain model has a single row.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatedloop.checkpoint import load_model
from gatedloop.data.mixture import load_domains, resolve_config_path
from gatedloop.data.text import PackedTextDataset

ROOT = Path(__file__).resolve().parents[1]


def _abs(p: str | None) -> str | None:
    if p is None:
        return None
    q = Path(p)
    return str(q if q.is_absolute() else ROOT / q)


@torch.no_grad()
def evaluate(model, loader, loops: int, device: str, amp_dtype) -> tuple[float, float, float]:
    """Return (mean loss over tokens, token accuracy, seconds)."""
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    t0 = time.perf_counter()
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits = model(ids, loops=loops).logits
        logits = logits[:, :-1].float()
        targets = ids[:, 1:]
        n = targets.numel()
        total_loss += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
        ).item()
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total_tokens += n
    return (total_loss / max(1, total_tokens),
            total_correct / max(1, total_tokens),
            time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--bin", type=str, default="artifacts/val.bin")
    ap.add_argument("--data", type=str, default=None,
                    help="data config with a domains: block → evaluate every domain's val bin "
                         "(overrides --bin; seq_len defaults to the config's)")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="default: the seq_len the checkpoint was trained at, if recorded")
    ap.add_argument("--loops", type=str, default=None,
                    help="comma-separated loop counts to sweep (default: the model's own)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = the whole file")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    model, hparams = load_model(args.ckpt, map_location="cpu")
    model.to(args.device)

    # ── what to evaluate: one bin, or every domain of a mixture ──────────────
    sets: list[tuple[str, PackedTextDataset, float]] = []      # (name, dataset, share)
    seq_len = args.seq_len
    if args.data:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(str(resolve_config_path(args.data, ROOT)))
        seq_len = seq_len or int(cfg.seq_len)
        domains = load_domains(cfg, _abs)
        if not domains:
            sets.append(("val", PackedTextDataset(_abs(cfg.val_bin), seq_len), 1.0))
        for d in domains:
            sets.append((d.name, PackedTextDataset(d.val_bin, seq_len), d.share))
    else:
        seq_len = seq_len or model.cfg.max_seq_len - 1
        sets.append(("val", PackedTextDataset(_abs(args.bin), seq_len), 1.0))

    def make_loader(ds):
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if args.max_batches:
            from itertools import islice

            class _Limited:
                def __init__(self, dl, n): self.dl, self.n = dl, n
                def __iter__(self): return islice(iter(self.dl), self.n)

            loader = _Limited(loader, args.max_batches)
        return loader

    loaders = [(name, make_loader(ds), share, ds) for name, ds, share in sets]
    multi = len(loaders) > 1

    loops_list = ([int(x) for x in args.loops.split(",")] if args.loops
                  else [model.cfg.loop_count])
    if not model.is_looped and len(loops_list) > 1:
        print("[note] this checkpoint is a PLAIN model — every loop count is the same run")
        loops_list = [1]

    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]
    print(f"\ncheckpoint : {args.ckpt}")
    for name, _, share, ds in loaders:
        print(f"data       : {name:<10} {ds.bin_path}  ({ds.n_tokens/1e6:.1f}M tokens, "
              f"seq_len={seq_len}" + (f", share={share:.2f}" if multi else "") + ")")
    print(f"model      : {'LOOPED' if model.is_looped else 'PLAIN'}  "
          f"{model.num_params()/1e6:.2f}M params  trained at R={model.cfg.loop_count}\n")
    print(f"{'R':>3} {'domain':<10} {'attn depth':>11} {'loss':>8} {'ppl':>10} {'tok_acc':>8} {'sec':>7}")
    print("─" * 63)
    for r in loops_list:
        w_loss = w_acc = 0.0
        for name, loader, share, _ in loaders:
            loss, acc, secs = evaluate(model, loader, r, args.device, amp)
            w_loss += share * loss
            w_acc += share * acc
            print(f"{r:>3} {name:<10} {model.effective_depth(r):>11} {loss:>8.4f} "
                  f"{math.exp(min(loss, 20)):>10.2f} {acc:>8.4f} {secs:>7.1f}")
        if multi:
            print(f"{r:>3} {'mix':<10} {model.effective_depth(r):>11} {w_loss:>8.4f} "
                  f"{math.exp(min(w_loss, 20)):>10.2f} {w_acc:>8.4f} {'':>7}   (share-weighted)")
            print("─" * 63)
    print()


if __name__ == "__main__":
    main()
