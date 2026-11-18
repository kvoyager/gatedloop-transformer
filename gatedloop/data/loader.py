"""
loader.py
─────────────────────────────────────────────────────────────────────────────
Resolve DataLoader kwargs for the train vs val role from one dataloader config.

Validation runs only every ``val_check_interval`` steps, so it rarely needs the
train loader's worker count — and persistent val workers would otherwise sit
idle (eating memory) between validations. Fields under ``dataloader.val.*``
override the train values for the VAL loader only; anything left ``null``
inherits the train value. The ``num_workers == 0`` constraints
(``persistent_workers`` off, ``prefetch_factor`` None) are enforced here so the
caller can flip val workers to 0 without tripping a DataLoader error.
"""

from __future__ import annotations

from omegaconf import OmegaConf


def make_loader_kwargs(dl_cfg, val: bool) -> dict:
    """Return DataLoader kwargs (batch_size / num_workers / pin_memory /
    persistent_workers / prefetch_factor) for the train (``val=False``) or
    validation (``val=True``) loader."""

    def pick(field: str, default=None):
        # For the val loader a non-null dataloader.val.<field> wins; otherwise
        # fall back to the shared train-side value.
        if val:
            v = OmegaConf.select(dl_cfg, f"val.{field}", default=None)
            if v is not None:
                return v
        return OmegaConf.select(dl_cfg, field, default=default)

    num_workers = int(pick("num_workers", 0))
    return {
        "batch_size": int(pick("batch_size")),
        "num_workers": num_workers,
        "pin_memory": bool(pick("pin_memory", True)),
        # both are invalid/ignored when num_workers == 0
        "persistent_workers": bool(pick("persistent_workers", False)) and num_workers > 0,
        "prefetch_factor": int(pick("prefetch_factor", 2)) if num_workers > 0 else None,
    }
