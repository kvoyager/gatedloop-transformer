"""
gatedloop/tools/trace_norms.py
─────────────────────────────────────────────────────────────────────────────
Residual-norm trace for an existing checkpoint: how the length of the residual
stream changes from one loop pass to the next, alongside the direction change
and relative step that the convergence callback already records.

With s_0 the block input and s_i the residual after pass i, it records per token

    norm[i]       = ‖s_i‖                      for i = 0..R      (B, T, L, R+1)
    norm_ratio[i] = ‖s_i‖ / ‖s_{i-1}‖          for i = 1..R      (B, T, L, R)
    rel[i], one_minus_cos[i]                    as the callback   (B, T, L, R)

and prints the block-mean of per-block medians of norm_ratio per pass, i.e. the
same aggregation as Table 3 of the paper, plus ‖s_R‖/‖s_0‖.

  python -m gatedloop.tools.trace_norms \
      --ckpt runs_ladder/looped_6/version_0/checkpoints/step060000.safetensors \
      --from-npz runs_ladder/looped_6/version_0/convergence/step060000_fineweb.npz \
      --loops 6 --out runs_ladder/analysis/norms

--from-npz reuses the exact input_ids of an existing convergence trace, so the
numbers are computed on the same tokens as the paper's δ and rel rows; without
it the first --n-batches validation batches of --data are used, as trace_ckpt
does. Writes <out>/<run>_step<step>_<domain>_norms.npz and a .md table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gatedloop import paths
from gatedloop.checkpoint import load_model
from gatedloop.model.block import LoopTrace

_abs = paths.resolve


class NormTrace(LoopTrace):
    """LoopTrace that also keeps ‖s_i‖ for i = 0..R per block."""

    def __init__(self):
        super().__init__(readout=None, targets=None)
        self.norm: dict[int, list[torch.Tensor]] = {}

    @torch.no_grad()
    def start(self, block_idx: int, x: torch.Tensor) -> None:
        super().start(block_idx, x)
        self.norm[block_idx] = [x.detach().float().norm(dim=-1)]

    @torch.no_grad()
    def record(self, block_idx: int, x: torch.Tensor) -> None:
        super().record(block_idx, x)
        self.norm[block_idx].append(x.detach().float().norm(dim=-1))

    def stacked(self) -> dict[str, torch.Tensor]:
        out = super().stacked()                      # rel, one_minus_cos: (B, T, L, R)
        layers = self.layers
        norm = torch.stack([torch.stack(self.norm[l], dim=-1) for l in layers], dim=-2)  # (B, T, L, R+1)
        out["norm"] = norm
        out["norm_ratio"] = norm[..., 1:] / norm[..., :-1].clamp_min(1e-6)
        return out


def _batches_from_npz(path: str, batch_size: int):
    d = np.load(path)
    ids = torch.from_numpy(d["input_ids"].astype(np.int64))
    for a in range(0, ids.shape[0], batch_size):
        yield ids[a:a + batch_size]


def _batches_from_data(data_cfg: str, seq_len: int | None, n_batches: int, batch_size: int):
    from omegaconf import OmegaConf
    from gatedloop.data.mixture import load_domains, resolve_config_path
    from gatedloop.data.text import PackedTextDataset
    cfg = OmegaConf.load(str(resolve_config_path(data_cfg, paths.project_root())))
    seq_len = seq_len or int(cfg.seq_len)
    domains = load_domains(cfg, _abs)
    sets = [(d.name, PackedTextDataset(d.val_bin, seq_len)) for d in domains] or \
           [("val", PackedTextDataset(_abs(cfg.val_bin), seq_len))]
    for name, ds in sets:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        out = []
        for bi, batch in enumerate(loader):
            if bi >= n_batches:
                break
            out.append(batch["input_ids"])
        yield name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".safetensors (with .json beside it) or Lightning .ckpt")
    ap.add_argument("--from-npz", default=None,
                    help="existing convergence .npz whose input_ids to reuse (one domain)")
    ap.add_argument("--data", default="gatedloop/configs/data/mix_fwe_code_math.yaml")
    ap.add_argument("--loops", type=int, default=8, help="passes to trace (R at eval)")
    ap.add_argument("--n-batches", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--out", default="runs_ladder/analysis/norms")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", default="fp32", choices=["bf16", "fp16", "fp32"],
                    help="fp32 by default: norms are the quantity of interest")
    args = ap.parse_args()

    model, hparams = load_model(args.ckpt, map_location="cpu")
    model.to(args.device).eval()
    if not model.is_looped:
        sys.exit("plain checkpoint — nothing to trace")
    if args.ckpt.endswith(".safetensors"):
        step = int(json.loads(Path(args.ckpt).with_suffix(".json").read_text())["global_step"])
    else:
        step = int(torch.load(args.ckpt, map_location="cpu", weights_only=False).get("global_step", 0))
    run = Path(args.ckpt).resolve().parents[2].name            # runs_ladder/<run>/version_x/checkpoints
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]
    dev = args.device.split(":")[0]

    if args.from_npz:
        domain = Path(args.from_npz).stem.split("_", 1)[1]     # step060000_fineweb -> fineweb
        sets = [(domain, list(_batches_from_npz(args.from_npz, args.batch_size)))]
    else:
        sets = list(_batches_from_data(args.data, args.seq_len, args.n_batches, args.batch_size))

    out = Path(_abs(args.out))
    out.mkdir(parents=True, exist_ok=True)
    for name, batches in sets:
        bufs, layers = [], None
        for ids in batches:
            ids = ids.to(args.device)
            with torch.no_grad(), \
                 torch.autocast(device_type=dev, dtype=amp, enabled=amp is not None):
                trace = NormTrace()
                for b in model.blocks:
                    if b.loop_enabled:
                        b._trace = trace
                try:
                    model.backbone(ids, args.loops)
                finally:
                    for b in model.blocks:
                        b._trace = None
            arrs = trace.stacked()
            layers = trace.layers
            bufs.append({k: a.float().cpu().numpy() for k, a in arrs.items()})
            bufs[-1]["input_ids"] = ids.cpu().numpy().astype(np.int32)
        arr = {k: np.concatenate([b[k] for b in bufs], axis=0) for k in bufs[0]}
        path = out / f"{run}_step{step:06d}_{name}_norms.npz"
        np.savez(path, **arr, layers=np.asarray(layers, dtype=np.int32),
                 step=np.int64(step), loops=np.int64(args.loops))

        # ── summary: median over tokens per block, then mean over blocks ────────
        ratio = arr["norm_ratio"]                                 # (N, T, L, R)
        norm = arr["norm"]                                        # (N, T, L, R+1)
        L, R = ratio.shape[2], ratio.shape[3]
        med_ratio = np.median(ratio.reshape(-1, L, R), axis=0)    # (L, R)
        med_total = np.median((norm[..., -1] / norm[..., 0]).reshape(-1, L), axis=0)  # (L,)
        med_rel = np.median(arr["rel"].reshape(-1, L, R), axis=0)
        med_oc = np.median(arr["one_minus_cos"].reshape(-1, L, R), axis=0)
        lines = [f"# {run}, step {step}, {name}, {norm.shape[0]} x {norm.shape[1]} tokens, R={R}", "",
                 "Median over tokens of ‖s_i‖/‖s_{i-1}‖ per block and pass "
                 "(pass 1 is the ungated first pass); last column ‖s_R‖/‖s_0‖.", "",
                 "| block | " + " | ".join(f"p{i}" for i in range(1, R + 1)) + " | s_R/s_0 |",
                 "| --- | " + " | ".join("---" for _ in range(R)) + " | --- |"]
        for li, l in enumerate(layers):
            lines.append(f"| {l} | " + " | ".join(f"{v:.3f}" for v in med_ratio[li]) + f" | {med_total[li]:.2f} |")
        lines.append("| block mean | " + " | ".join(f"{v:.3f}" for v in med_ratio.mean(0)) + f" | {med_total.mean():.2f} |")
        lines += ["", "Block-mean medians of rel and δ on the same tokens (should match the paper's Table 3 for passes 2..R):", "",
                  "| metric | " + " | ".join(f"p{i}" for i in range(1, R + 1)) + " |",
                  "| --- | " + " | ".join("---" for _ in range(R)) + " |",
                  "| rel | " + " | ".join(f"{v:.4f}" for v in med_rel.mean(0)) + " |",
                  "| δ | " + " | ".join(f"{v:.4f}" for v in med_oc.mean(0)) + " |"]
        md = "\n".join(lines) + "\n"
        (out / f"{run}_step{step:06d}_{name}_norms.md").write_text(md, encoding="utf-8")
        print(md)
        print(f"-> {path}")


if __name__ == "__main__":
    main()
