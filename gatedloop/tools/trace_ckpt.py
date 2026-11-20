"""
gatedloop/tools/trace_ckpt.py
─────────────────────────────────────────────────────────────────────────────
Offline convergence trace for an existing checkpoint: the same arrays the
training-time callback saves (rel, one_minus_cos, and — new — kl, nll), so runs
that were trained before `val_convergence.output_kl` existed get the
successive-output KL without retraining.

  python -m gatedloop.tools.trace_ckpt --ckpt runs_ladder/looped_6/version_0/checkpoints/best.ckpt \
      --data gatedloop/configs/data/mix_fwe_code_math.yaml --loops 8 \
      --n-batches 2 --batch-size 8 --out runs_ladder/looped_6/version_0/convergence_kl

Writes <out>/step<global_step>_<domain>.npz with keys
  rel, one_minus_cos, kl, nll : float32 (N, T, L, R)
  input_ids (N, T) int32, valid (N, T) bool, layers (L,), step (), loops ()
exactly like the callback, so the existing analysis code reads both.
`--n-batches 2 --batch-size 8` at seq_len 1024 reproduces the callback's fixed
16k-token batch per domain (the first two validation batches, unshuffled).

Relative paths (--data and the bin paths inside it) resolve like training's:
against the repo root in a checkout, else the launch directory or GATEDLOOP_ROOT
(see gatedloop/paths.py). --ckpt and --out are taken as given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gatedloop import paths
from gatedloop.checkpoint import load_model
from gatedloop.data.mixture import load_domains, resolve_config_path
from gatedloop.data.text import PackedTextDataset

_abs = paths.resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="gatedloop/configs/data/mix_fwe_code_math.yaml")
    ap.add_argument("--loops", type=int, default=8, help="passes to trace (R at eval)")
    ap.add_argument("--n-batches", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--out", default=None, help="default: <run dir>/convergence_kl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    model, hparams = load_model(args.ckpt, map_location="cpu")
    model.to(args.device).eval()
    if not model.is_looped:
        sys.exit("plain checkpoint — nothing to trace")
    step = int(torch.load(args.ckpt, map_location="cpu", weights_only=False).get("global_step", 0))

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(str(resolve_config_path(args.data, paths.project_root())))
    seq_len = args.seq_len or int(cfg.seq_len)
    domains = load_domains(cfg, _abs)
    sets = [(d.name, PackedTextDataset(d.val_bin, seq_len)) for d in domains] or \
           [("val", PackedTextDataset(_abs(cfg.val_bin), seq_len))]

    out = Path(args.out) if args.out else Path(args.ckpt).resolve().parents[1] / "convergence_kl"
    out.mkdir(parents=True, exist_ok=True)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]
    dev = args.device.split(":")[0]

    for name, ds in sets:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        bufs, layers = [], None
        for bi, batch in enumerate(loader):
            if bi >= args.n_batches:
                break
            ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device) if "labels" in batch else ids.clone()
            # next-token targets aligned with the input (labels are unshifted), as in the callback
            targets = F.pad(labels[:, 1:], (0, 1), value=-100)
            with torch.no_grad(), \
                 torch.autocast(device_type=dev, dtype=amp, enabled=amp is not None), \
                 model.trace_loop(output_kl=True, targets=targets) as tr:
                model.backbone(ids, args.loops)
            arrs = tr.stacked()                                   # (B, T, L, R) fp32
            layers = tr.layers
            valid = (labels != -100)
            bufs.append(dict(**{k: a.float().cpu().numpy() for k, a in arrs.items()},
                             input_ids=ids.cpu().numpy().astype(np.int32),
                             valid=valid.cpu().numpy()))
        path = out / f"step{step:06d}_{name}.npz"
        np.savez(path,
                 **{k: np.concatenate([b[k] for b in bufs], axis=0) for k in bufs[0]},
                 layers=np.asarray(layers, dtype=np.int32),
                 step=np.int64(step), loops=np.int64(args.loops))
        kl = np.concatenate([b["kl"] for b in bufs]); v = np.concatenate([b["valid"] for b in bufs])
        med = [float(np.median(kl[:, :, :, i][v])) for i in range(kl.shape[-1])]
        print(f"{name:8s} -> {path}   median KL per pass: " + " ".join(f"{m:.2e}" for m in med))


if __name__ == "__main__":
    main()
