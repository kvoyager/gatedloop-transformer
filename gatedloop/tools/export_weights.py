"""
gatedloop/tools/export_weights.py
─────────────────────────────────────────────────────────────────────────────
Strip Lightning checkpoints down to the model weights for publishing.

A training ``.ckpt`` is a pickle holding the weights plus the optimizer state
(~1.5× the weights), callback/loop state, the git info and the zipped source
tree (``code_bytes``). For release only the weights and the config are needed,
and a pickle is the wrong container anyway — loading one runs code. For every
``<stem>.ckpt`` this writes, next to it:

  <stem>.safetensors   the model's state_dict (``model.`` / ``_orig_mod.``
                       prefixes removed; with tied embeddings ``lm_head.weight``
                       is omitted — it is ``embed.weight``)
  <stem>.json          {"model": ModelConfig fields, "training": the other
                       hyper-parameters, "global_step", "epoch", "source"}

Each export is verified tensor-by-tensor against the checkpoint.
``gatedloop.checkpoint.load_model`` reads the ``.safetensors`` directly.

  python -m gatedloop.tools.export_weights runs_ladder/looped_6/version_0/checkpoints/step060000.ckpt
  python -m gatedloop.tools.export_weights --list runs_ladder/analysis/hf_upload_list.txt

With ``--list``, every ``.ckpt`` line is exported and a copy of the list with
those lines replaced by the two new files is written as ``<list>_weights.txt``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from gatedloop.checkpoint import clean_state_dict

FORMAT = "gatedloop-weights/1"


def export(ckpt_path: Path, overwrite: bool = False) -> tuple[Path, Path]:
    st_path, cfg_path = ckpt_path.with_suffix(".safetensors"), ckpt_path.with_suffix(".json")
    if not overwrite and st_path.exists() and cfg_path.exists():
        print(f"[skip] {st_path} exists")
        return st_path, cfg_path

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    model_cfg = dict(hparams.pop("model", {}))
    if not model_cfg:
        raise ValueError(f"{ckpt_path} has no hyper_parameters['model']")

    sd = clean_state_dict(ckpt["state_dict"])
    if model_cfg.get("tie_embeddings", True):
        sd.pop("lm_head.weight", None)
    sd = {k: v.contiguous() for k, v in sd.items()}

    meta = {
        "format": FORMAT,
        "model": model_cfg,
        "training": hparams,
        "global_step": int(ckpt.get("global_step", 0)),
        "epoch": int(ckpt.get("epoch", 0)),
        "source": ckpt_path.name,
        "pytorch_lightning_version": ckpt.get("pytorch-lightning_version"),
    }
    save_file(sd, str(st_path), metadata={"format": "pt", "gatedloop": FORMAT})
    cfg_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    back = load_file(str(st_path))
    if back.keys() != sd.keys() or any(not torch.equal(back[k], sd[k]) for k in sd):
        raise RuntimeError(f"verification failed for {st_path}")
    mb = st_path.stat().st_size / 1e6
    print(f"[ok]   {st_path}  {len(sd)} tensors, {mb:.0f} MB "
          f"(ckpt {ckpt_path.stat().st_size / 1e6:.0f} MB)")
    return st_path, cfg_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("ckpts", nargs="*", type=Path)
    ap.add_argument("--list", type=Path, help="upload list: one path per line")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not args.ckpts and not args.list:
        ap.error("give checkpoint paths or --list")

    for c in args.ckpts:
        export(c, args.overwrite)

    if args.list:
        out_lines = []
        for line in args.list.read_text(encoding="utf-8").splitlines():
            p = line.strip()
            if p.lower().endswith(".ckpt"):
                st, cfg = export(Path(p), args.overwrite)
                base = p[: -len(".ckpt")]
                out_lines += [base + ".safetensors", base + ".json"]
            elif p:
                out_lines.append(p)
        out = args.list.with_name(args.list.stem + "_weights.txt")
        out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"[list] {out}  ({len(out_lines)} entries)")


if __name__ == "__main__":
    sys.exit(main())
