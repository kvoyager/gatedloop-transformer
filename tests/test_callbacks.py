"""
test_callbacks.py
─────────────────────────────────────────────────────────────────────────────
Run-control checks: a PAUSE file holds the training loop until it is removed,
a STOP file ends the run cleanly (save-and-stop), and neither leaves the
metrics or the throughput window in a bad state. CPU, synthetic data, ~10 s.

  python tests/test_callbacks.py
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatedloop.training.callbacks import ThroughputCallback, TrainControlCallback   # noqa: E402
from gatedloop.training.lit_module import LitPretrain                                # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="lt_test_cb_"))
MODEL = dict(vocab_size=64, pad_id=0, d_model=32, n_heads=4, n_layers=2, ff_dim=64, max_seq_len=32)


class RandDS(Dataset):
    def __init__(self, n=64, T=17, V=64):
        self.x = torch.randint(0, V, (n, T))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"input_ids": self.x[i], "labels": self.x[i].clone()}


class StepClock(pl.Callback):
    """Wall-clock stamp of every step, so a pause shows up as one long gap."""

    def __init__(self):
        self.stamps: list[tuple[int, float]] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.stamps.append((trainer.global_step, time.perf_counter()))


def _trainer(name: str, max_steps: int, extra: list[pl.Callback]) -> pl.Trainer:
    return pl.Trainer(accelerator="cpu", max_steps=max_steps, logger=CSVLogger(str(_TMP), name=name),
                      enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, num_sanity_val_steps=0, log_every_n_steps=1,
                      limit_val_batches=0, callbacks=[ThroughputCallback(window=2),
                                                      TrainControlCallback(poll_seconds=0.05), *extra])


def _run_dir(trainer) -> Path:
    return Path(trainer.logger.log_dir)


def test_pause_file_holds_and_resumes():
    clock = StepClock()
    ctl_holder = {}
    trainer = _trainer("pause", max_steps=12, extra=[clock])
    lit = LitPretrain(model=MODEL, warmup_steps=1, max_steps=12)

    class Pauser(pl.Callback):
        """At step 4 drop a PAUSE file; a side thread removes it 1.2 s later."""
        def on_train_batch_end(self, tr, pm, outputs, batch, batch_idx):
            if tr.global_step == 4 and "armed" not in ctl_holder:
                ctl_holder["armed"] = True
                f = _run_dir(tr) / "PAUSE"
                f.touch()
                threading.Timer(1.2, f.unlink).start()
    trainer.callbacks.insert(0, Pauser())          # runs BEFORE TrainControlCallback's hook

    trainer.fit(lit, DataLoader(RandDS(), batch_size=4))
    assert trainer.global_step == 12, "run did not complete after resume"
    gaps = {s: t1 - t0 for (s, t0), (_, t1) in zip(clock.stamps, clock.stamps[1:])}
    # the stamp is taken after the batch, so the hold shows as the gap ending at step 4
    paused_at = max(gaps, key=gaps.get)
    assert paused_at in (3, 4) and gaps[paused_at] >= 1.0, f"expected a ≥1 s hold around step 4, got {gaps}"
    assert all(g < 0.5 for s, g in gaps.items() if s != paused_at), "other steps should not stall"
    assert not (_run_dir(trainer) / "PAUSE").exists()
    ctl = next(c for c in trainer.callbacks if isinstance(c, TrainControlCallback))
    assert ctl._paused_total >= 1.0 and not ctl._stop_requested
    print(f"  PAUSE file: held {gaps[paused_at]:.2f}s around step 4, resumed, finished all 12 steps — OK")


def test_stop_file_ends_run():
    trainer = _trainer("stop", max_steps=50, extra=[])
    lit = LitPretrain(model=MODEL, warmup_steps=1, max_steps=50)

    class Stopper(pl.Callback):
        def on_train_batch_end(self, tr, pm, outputs, batch, batch_idx):
            if tr.global_step == 5:
                (_run_dir(tr) / "STOP").touch()
    trainer.callbacks.insert(0, Stopper())

    trainer.fit(lit, DataLoader(RandDS(), batch_size=4))
    assert 5 <= trainer.global_step <= 6, f"stopped at step {trainer.global_step}, expected 5–6"
    assert not (_run_dir(trainer) / "STOP").exists(), "STOP file should be consumed"
    ctl = next(c for c in trainer.callbacks if isinstance(c, TrainControlCallback))
    assert ctl._stop_requested and ctl._reason == "STOP file"
    print(f"  STOP file: run ended cleanly at step {trainer.global_step} — OK")


def test_throughput_window_reset():
    tp = ThroughputCallback(window=5)
    tp._t0 = time.perf_counter() - 100.0
    tp._tokens_window = 12345
    tp.reset_window()
    assert tp._tokens_window == 0 and time.perf_counter() - tp._t0 < 1.0
    print("  ThroughputCallback.reset_window — OK")


def main():
    for fn in [test_pause_file_holds_and_resumes, test_stop_file_ends_run, test_throughput_window_reset]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n✅ all callback tests passed")


if __name__ == "__main__":
    main()
