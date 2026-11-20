"""
checkpoint.py
─────────────────────────────────────────────────────────────────────────────
Rebuild a bare :class:`TransformerLM` from a Lightning checkpoint.

The checkpoint stores the model config under ``hyper_parameters["model"]``, so
nothing about the architecture has to be repeated on the command line — the
eval and sampling scripts just point at a ``.ckpt``. Two prefixes are stripped
on the way in: Lightning's ``model.`` (the LightningModule attribute) and
``_orig_mod.`` (inserted by ``torch.compile``).

``overrides`` lets you change config fields that are safe to change at load time
— in practice ``loop_count`` (run the same weights at a different R) and
``max_seq_len`` (the RoPE tables are non-persistent buffers, rebuilt on
construction, so a longer horizon costs nothing but extrapolation risk).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from gatedloop.model.lm import ModelConfig, TransformerLM


def clean_state_dict(state_dict: dict) -> dict:
    """Lightning/compile key names → bare TransformerLM key names."""
    clean = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "")
        # Pre-two-gate checkpoints carried a single per-block α.
        k = k.replace(".alpha_logit", ".alpha_attn_logit")
        if k.startswith("model."):
            k = k[len("model."):]
        clean[k] = v
    return clean


def _read_safetensors(path: Path, map_location) -> tuple[dict, dict]:
    """Weights + hparams of an export from gatedloop.tools.export_weights."""
    from safetensors.torch import load_file

    cfg_path = path.with_suffix(".json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found — it holds the model config for {path.name}")
    meta = json.loads(cfg_path.read_text(encoding="utf-8"))
    device = str(map_location) if map_location is not None else "cpu"
    hparams = {**meta.get("training", {}), "model": meta["model"]}
    return load_file(str(path), device=device), hparams


def load_model(
    ckpt_path: str | Path,
    map_location: str | torch.device = "cpu",
    overrides: dict | None = None,
) -> tuple[TransformerLM, dict]:
    """Return ``(model, hparams)`` with weights loaded and the model in eval mode.

    ``ckpt_path`` is a Lightning ``.ckpt`` or a ``.safetensors`` weights file
    (with its ``.json`` config beside it)."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if ckpt_path.suffix == ".safetensors":
        state_dict, hparams = _read_safetensors(ckpt_path, map_location)
    else:
        # Local, self-produced checkpoints; weights_only=False is needed because
        # we embed the code snapshot / git info alongside the tensors.
        ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
        state_dict, hparams = ckpt["state_dict"], dict(ckpt.get("hyper_parameters", {}))

    model_cfg = dict(hparams.get("model", {}))
    if not model_cfg:
        raise ValueError(
            f"{ckpt_path} has no hyper_parameters['model'] — was it produced by "
            f"gatedloop.train?"
        )
    model_cfg.update(overrides or {})
    model = TransformerLM(ModelConfig(**model_cfg))

    clean = clean_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if model.cfg.tie_embeddings:
        # exports omit lm_head.weight: it is the same tensor as embed.weight
        missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        print(f"[checkpoint] missing={list(missing)} unexpected={list(unexpected)}")
    model.eval()
    return model, hparams
