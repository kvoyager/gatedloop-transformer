"""
callbacks.py
─────────────────────────────────────────────────────────────────────────────
Lightning callbacks for the pretraining run.

  ParamCountCallback   model-size / loop-topology banner at fit start
  ThroughputCallback   tokens/s and cumulative tokens seen
  SampleTextCallback   periodic sampled continuation (the qualitative sanity check)
  CodeSnapshot         zip source + git info at fit start; embed in checkpoints
  ValMetricsCSV        one row per validation → val_metrics.csv
  EpochSummary         one-line train/val summary at each validation / epoch
  LRRichProgressBar    RichProgressBar that renders lr in scientific notation
  TrainControlCallback pause/resume ('p'/'r' or a PAUSE file) and save-and-stop
                       ('s' / Ctrl-C / a STOP file)
  GateAlphaCallback    learned loop gates α_attn / α_ff per block → gate/*
  LoopConvergenceCallback per-iteration relative step, 1−cos and output KL at validation
                       → val/conv/* scalars + convergence/step*.npz arrays
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import RichProgressBar

log = logging.getLogger(__name__)

# Repo root = parent of the package (this file is <pkg>/training/callbacks.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Model-size / loop-topology banner
# ─────────────────────────────────────────────────────────────────────────────
class ParamCountCallback(pl.Callback):
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        m = pl_module.model
        cfg = m.cfg
        total = m.num_params(include_embedding=True)
        nonemb = m.num_params(include_embedding=False)
        if m.is_looped:
            which = ("all" if cfg.loop_layers is None
                     else ",".join(str(i) for i in cfg.loop_layers))
            sampled = (f"train R ~ U[{cfg.loop_min},{cfg.loop_max}]"
                       if cfg.loop_sample else "fixed R")
            target = ("whole-block loop (attn+FFN)" if cfg.loop_target == "block"
                      else "attention-only loop")
            n_ffn = (m.effective_depth() if cfg.loop_target == "block"
                     else cfg.n_layers)
            loop_lines = (
                f"│ variant               : LOOPED — {target}\n"
                f"│ loop iterations (R)   : {cfg.loop_count} at eval  ({sampled})\n"
                f"│ looped layers         : {which}   gates α_attn/α_ff: {cfg.loop_gate} (init {cfg.loop_gate_init})  "
                f"step-embed: {cfg.loop_step_embed}\n"
                f"│ truncated BPTT (k)    : {cfg.loop_backprop_k}   static K/V: {cfg.loop_static_kv}\n"
                f"│ attn applications     : {m.effective_depth()}  "
                f"(FFN applications: {n_ffn})\n"
            )
        else:
            loop_lines = (
                f"│ variant               : PLAIN transformer (no loop)\n"
                f"│ attn applications     : {m.effective_depth()}  "
                f"(FFN applications: {cfg.n_layers})\n"
            )
        print(
            "\n┌─ gatedloop-transformer ──────────────────────────────────────\n"
            f"{loop_lines}"
            f"│ params (total)        : {total/1e6:7.2f} M\n"
            f"│ params (non-embedding): {nonemb/1e6:7.2f} M\n"
            f"│ d_model / heads / ff  : {cfg.d_model} / {cfg.n_heads} / {cfg.ff_dim}\n"
            f"│ layers                : {cfg.n_layers}\n"
            f"│ vocab / max_seq_len   : {cfg.vocab_size} / {cfg.max_seq_len}\n"
            "└───────────────────────────────────────────────────────────\n"
        )
        self._print_mixture(getattr(pl_module, "data_info", None) or {})

    @staticmethod
    def _print_mixture(info: dict) -> None:
        """The mixture table: what each domain contributes and how many times
        the run will traverse it (>1 = the domain repeats within the budget)."""
        doms = info.get("domains")
        if not doms:
            return
        tokens_per_step, max_steps = info.get("tokens_per_step"), info.get("max_steps")
        budget = tokens_per_step * max_steps if tokens_per_step and max_steps else None
        lines = ["┌─ data mixture ────────────────────────────────────────────",
                 "│ domain      share   train tokens   val tokens   passes in run"]
        for name, d in doms.items():
            passes = (budget * d["share"] / max(1, d["train_tokens"])) if budget else float("nan")
            lines.append(f"│ {name:<10} {d['share']:6.2f} {d['train_tokens']/1e6:11.1f}M "
                         f"{d['val_tokens']/1e6:10.2f}M {passes:12.2f}")
        if budget:
            lines.append(f"│ run budget            : {budget/1e9:.2f}B tokens "
                         f"({max_steps} steps × {tokens_per_step} tokens)")
        lines.append("└───────────────────────────────────────────────────────────")
        print("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Throughput
# ─────────────────────────────────────────────────────────────────────────────
class ThroughputCallback(pl.Callback):
    """Log tokens/s (windowed) and total tokens seen. The token count is the
    honest x-axis for comparing a looped run against a plain one — they differ
    in step time, not in tokens per step."""

    def __init__(self, window: int = 50):
        super().__init__()
        self.window = int(window)
        self._t0: float | None = None
        self._tokens_window = 0
        self._tokens_total = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = int(batch["input_ids"].numel())
        self._tokens_total += n
        self._tokens_window += n
        if self._t0 is None:
            self._t0 = time.perf_counter()
            self._tokens_window = 0
            return
        if (batch_idx + 1) % self.window:
            return
        dt = time.perf_counter() - self._t0
        if dt > 0:
            pl_module.log("train/tok_per_s", self._tokens_window / dt,
                          on_step=True, on_epoch=False, prog_bar=True)
        pl_module.log("train/tokens_M", self._tokens_total / 1e6,
                      on_step=True, on_epoch=False)
        if torch.cuda.is_available() and pl_module.device.type == "cuda":
            # peak allocated since process start — includes validation sweeps
            pl_module.log("train/peak_mem_gib", torch.cuda.max_memory_allocated() / 2**30,
                          on_step=True, on_epoch=False)
        self._t0 = time.perf_counter()
        self._tokens_window = 0

    def reset_window(self) -> None:
        """Restart the rate window — after a pause or a validation pass, so
        time spent not training is not counted against throughput."""
        self._t0 = time.perf_counter()
        self._tokens_window = 0

    def on_validation_end(self, trainer, pl_module):
        if self._t0 is not None:
            self.reset_window()

    def state_dict(self) -> dict:
        return {"tokens_total": self._tokens_total}

    def load_state_dict(self, state_dict: dict) -> None:
        self._tokens_total = int(state_dict.get("tokens_total", 0))


# ─────────────────────────────────────────────────────────────────────────────
# Periodic sample generation
# ─────────────────────────────────────────────────────────────────────────────
class SampleTextCallback(pl.Callback):
    """Sample a continuation every ``every_n_steps`` as a qualitative check —
    watching text go from noise to grammatical English is the fastest read on
    whether pretraining is working."""

    def __init__(self, tokenizer, every_n_steps: int = 2000, max_new_tokens: int = 96,
                 prompt: str | None = None, temperature: float = 0.8, top_k: int = 50,
                 loops: int | None = None, add_bos: bool = True):
        self.tok = tokenizer
        self.every = int(every_n_steps)
        self.max_new_tokens = int(max_new_tokens)
        self.prompt = prompt
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.loops = loops
        # Only prompt with <bos> if the corpus actually marks document starts
        # with it (data.has_bos) — otherwise its embedding never sees a gradient
        # and sampling from it just feeds the model an untrained token.
        self.add_bos = bool(add_bos)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if self.every <= 0 or step == 0 or step % self.every != 0:
            return
        ids = ([self.tok.bos_id] if self.add_bos else []) + \
              (self.tok.encode(self.prompt) if self.prompt else [])
        if not ids:                                  # no prompt and no <bos>
            ids = [self.tok.eos_id]                  # …start from a doc boundary
        prompt = torch.tensor([ids], dtype=torch.long, device=pl_module.device)
        was_training = pl_module.model.training
        gen = pl_module.model.generate(
            prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature,
            top_k=self.top_k, eos_id=self.tok.eos_id, loops=self.loops,
        )
        text = self.tok.decode(gen[0, len(ids):].tolist())
        if was_training:
            pl_module.model.train()
        print(f"\n[sample @ step {step}] {(self.prompt or '') + text!r}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Code snapshot + git info
# ─────────────────────────────────────────────────────────────────────────────
_INCLUDE_EXT = {".py", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".json",
                ".md", ".txt", ".sh"}
_EXCLUDE_DIRS = {".git", ".claude", "__pycache__", ".venv", "venv",
                 "artifacts", "artifacts_smoke", "data", "runs", "runs_smoke",
                 "outputs", "multirun", "lightning_logs",
                 ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints",
                 ".idea", ".vscode", "node_modules"}


def _snapshot_code_bytes(source_dir: Path) -> bytes:
    """Zip only source/config files under ``source_dir`` into memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIRS]
            for f in files:
                if Path(f).suffix.lower() not in _INCLUDE_EXT:
                    continue
                full = Path(root) / f
                zf.write(full, full.relative_to(source_dir))
    return buf.getvalue()


def _git_info(repo_dir: Path) -> dict | None:
    """Best-effort git commit/dirty/diff. Returns None if not a repo or git is
    unavailable (this repo need not be version-controlled)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        info = {"commit": commit, "dirty": bool(status)}
        if status:
            info["diff"] = subprocess.check_output(
                ["git", "diff", "HEAD", "--text"], cwd=repo_dir, stderr=subprocess.DEVNULL
            ).decode("utf-8", errors="replace")
        return info
    except Exception:
        return None


class CodeSnapshot(pl.Callback):
    """Zip the source tree + capture git info at fit start, write them to the run
    dir, and embed them in every checkpoint for exact reproducibility."""

    def __init__(self, source_dir: str | Path | None = None,
                 out_dir: str | Path | None = None, embed_in_ckpt: bool = True):
        self.source_dir = Path(source_dir) if source_dir else _REPO_ROOT
        self.out_dir = out_dir
        self.embed_in_ckpt = embed_in_ckpt
        self._git: dict | None = None
        self._code_bytes: bytes | None = None

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        out = Path(self.out_dir or (trainer.logger.log_dir if trainer.logger else None)
                   or trainer.default_root_dir)
        out.mkdir(parents=True, exist_ok=True)

        self._code_bytes = _snapshot_code_bytes(self.source_dir)
        self._git = _git_info(self.source_dir)

        (out / "code_snapshot.zip").write_bytes(self._code_bytes)
        if self._git is not None:
            meta = {k: v for k, v in self._git.items() if k != "diff"}
            (out / "git_info.json").write_text(json.dumps(meta, indent=2))
            if "diff" in self._git:
                (out / "uncommitted.diff").write_text(self._git["diff"], encoding="utf-8")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not self.embed_in_ckpt:
            return
        if self._git is not None:
            checkpoint["git"] = self._git
        if self._code_bytes is not None:
            checkpoint["code_bytes"] = self._code_bytes


# ─────────────────────────────────────────────────────────────────────────────
# Validation-metrics CSV
# ─────────────────────────────────────────────────────────────────────────────
class ValMetricsCSV(pl.Callback):
    """Write one row per validation epoch to val_metrics.csv (a val-only digest
    alongside Lightning's full metrics.csv). With ``val_loop_sweep`` set, each
    row carries the whole test-time-compute curve — ready to plot."""

    def __init__(self, filename: str = "val_metrics.csv", out_dir: str | Path | None = None):
        self.filename = filename
        self.out_dir = out_dir
        self._path: Path | None = None
        self._fieldnames: list[str] | None = None

    def _resolve_path(self, trainer) -> Path:
        base = Path(self.out_dir
                    or (trainer.logger.log_dir if trainer.logger else None)
                    or trainer.default_root_dir)
        base.mkdir(parents=True, exist_ok=True)
        return base / self.filename

    # on_validation_END, not _epoch_end: the LightningModule logs the mixture's
    # share-weighted headline in ITS on_validation_epoch_end, which runs after
    # the callbacks' — by on_validation_end every val metric is reduced.
    def on_validation_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        # A metric logged on_step AND on_epoch yields <name>, <name>_step,
        # <name>_epoch; <name>_step duplicates the bare key, so drop it.
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()
                   if not k.endswith("_step")}
        if not metrics:
            return
        row = {"epoch": trainer.current_epoch, "step": trainer.global_step, **metrics}

        if self._path is None:
            self._path = self._resolve_path(trainer)
            self._fieldnames = list(row.keys())
            with open(self._path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fieldnames).writeheader()

        new_keys = [k for k in row if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            self._rewrite_with_new_header(row)
            return
        with open(self._path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fieldnames).writerow(row)

    def _rewrite_with_new_header(self, latest_row: dict):
        with open(self._path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append(latest_row)
        with open(self._path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in self._fieldnames})


# ─────────────────────────────────────────────────────────────────────────────
# One-line epoch/validation summary
# ─────────────────────────────────────────────────────────────────────────────
class EpochSummary(pl.Callback):
    """One-line train/val summary at each validation (and at epoch end if no
    validation closed the window). Train metrics are averaged over the steps
    since the last validation — the bare ``callback_metrics`` key is the latest
    *batch* value, which is noisy and the source of apparent loss "spikes" when
    read directly. Val metrics are already epoch-averaged, so read straight."""

    def __init__(self):
        super().__init__()
        self._sum: dict[str, float] = defaultdict(float)
        self._cnt: dict[str, int] = defaultdict(int)

    @staticmethod
    def _is_metric(k: str, prefix: str) -> bool:
        return k.startswith(prefix) and not k.endswith(("_step", "_epoch"))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.sanity_checking:
            return
        for k, v in trainer.callback_metrics.items():
            if self._is_metric(k, "train/") and v is not None:
                self._sum[k] += float(v)
                self._cnt[k] += 1

    def _emit(self, prefix: str, trainer) -> None:
        m = {k: self._sum[k] / self._cnt[k] for k in self._sum if self._cnt[k] > 0}
        for k, v in trainer.callback_metrics.items():
            if self._is_metric(k, "val/") and v is not None:
                m[k] = float(v)
        parts = [f"step={trainer.global_step}"] + [f"{k}={v:.4f}" for k, v in sorted(m.items())]
        sys.stderr.write(prefix + " | ".join(parts) + "\n")
        sys.stderr.flush()

    def _reset(self):
        self._sum.clear()
        self._cnt.clear()

    def on_validation_end(self, trainer, pl_module):        # after the module's epoch-end logs
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        self._emit("[val]   ", trainer)
        self._reset()

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        if not self._cnt:        # window already consumed by an end-of-epoch validation
            return
        self._emit(f"[epoch {trainer.current_epoch}] ", trainer)
        self._reset()


# ─────────────────────────────────────────────────────────────────────────────
# Rich progress bar with scientific-notation lr
# ─────────────────────────────────────────────────────────────────────────────
class LRRichProgressBar(RichProgressBar):
    """RichProgressBar's metrics column rounds floats to 3 dp, so a small lr like
    1e-4 renders as '0.000'. Pre-format lr as a string to keep it readable."""

    def get_metrics(self, trainer, pl_module) -> dict:
        items = super().get_metrics(trainer, pl_module)
        lr = items.get("lr")
        if lr is not None:
            try:
                items["lr"] = f"{float(lr):.2e}"
            except (TypeError, ValueError):
                pass
        return items


# ─────────────────────────────────────────────────────────────────────────────
# Graceful keyboard / Ctrl-C stop
# ─────────────────────────────────────────────────────────────────────────────
class TrainControlCallback(pl.Callback):
    """Pause, resume and stop a run without killing it.

    Console keys (Windows console, the process that owns the terminal):
        p          pause / resume (toggle)
        r          resume
        s, Ctrl-C  finish the current step, save last.ckpt, exit cleanly
                   (a second Ctrl-C chains to the previous handler: hard exit)
    Files in the run directory (work from any terminal, or for a detached run):
        <run dir>/PAUSE   paused for as long as the file exists — delete to resume
        <run dir>/STOP    save-and-stop; the file is removed once seen

    A pause takes effect at the end of the current batch: the loop then sleeps,
    the GPU idles, nothing is logged or written, and optimizer / dataloader
    state stay exactly where they are. On resume the throughput window is
    reset so the idle time does not show up as a dip in ``train/tok_per_s``.
    ``train/paused_min`` records cumulative paused minutes.
    """

    KEYS_STOP = (b"s", b"S")
    KEYS_PAUSE = (b"p", b"P")
    KEYS_RESUME = (b"r", b"R")

    def __init__(self, poll_seconds: float = 0.5):
        super().__init__()
        self.poll_seconds = float(poll_seconds)
        self._stop_requested = False
        self._reason: str | None = None
        self._pause_key = False                 # toggled from the key thread
        self._paused_total = 0.0
        self._run_dir: Path | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev_sigint = None

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def on_train_start(self, trainer, pl_module):
        if not getattr(trainer, "is_global_zero", True):
            return
        self._run_dir = Path(trainer.logger.log_dir) if trainer.logger else None
        try:
            self._prev_sigint = signal.signal(signal.SIGINT, self._sigint_handler)
        except (ValueError, OSError):
            self._prev_sigint = None
        if os.name == "nt":
            self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="kbd-control")
            self._thread.start()
        where = f"  |  files: {self._run_dir / 'PAUSE'} / STOP" if self._run_dir else ""
        log.info("[TrainControlCallback] keys: p pause/resume, r resume, s / Ctrl-C save+stop%s", where)

    def on_train_end(self, trainer, pl_module):
        self._stop_event.set()
        if self._prev_sigint is not None:
            try:
                signal.signal(signal.SIGINT, self._prev_sigint)
            except (ValueError, OSError):
                pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._check_stop_file()
        if self._should_pause() and not self._stop_requested:
            self._hold(trainer, pl_module)
        if self._stop_requested and not trainer.should_stop:
            log.info("[TrainControlCallback] stop requested (%s) — finishing step, "
                     "saving last.ckpt and exiting", self._reason)
            trainer.should_stop = True

    # ── pause ─────────────────────────────────────────────────────────────────
    def _pause_file(self) -> Path | None:
        return self._run_dir / "PAUSE" if self._run_dir else None

    def _should_pause(self) -> bool:
        f = self._pause_file()
        return self._pause_key or (f is not None and f.exists())

    def _check_stop_file(self) -> None:
        if self._run_dir is None:
            return
        f = self._run_dir / "STOP"
        if f.exists():
            try:
                f.unlink()
            except OSError:
                pass
            self._request_stop("STOP file")

    def _hold(self, trainer, pl_module) -> None:
        t0 = time.perf_counter()
        how = "key" if self._pause_key else "PAUSE file"
        log.info("[TrainControlCallback] PAUSED at step %d (%s) — 'p'/'r' or delete %s to resume; "
                 "'s'/Ctrl-C to save and stop", trainer.global_step, how, self._pause_file())
        sys.stderr.write(f"\n[paused] step {trainer.global_step} — p/r resumes, s stops\n")
        sys.stderr.flush()
        while self._should_pause() and not self._stop_requested:
            time.sleep(self.poll_seconds)
            self._check_stop_file()
        dt = time.perf_counter() - t0
        self._paused_total += dt
        log.info("[TrainControlCallback] resumed after %.1f min (total paused %.1f min)",
                 dt / 60, self._paused_total / 60)
        for cb in trainer.callbacks:
            if isinstance(cb, ThroughputCallback):
                cb.reset_window()
        pl_module.log("train/paused_min", self._paused_total / 60, on_step=True, on_epoch=False)

    # ── stop ──────────────────────────────────────────────────────────────────
    def _request_stop(self, reason: str) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        self._reason = reason
        self._pause_key = False

    def _sigint_handler(self, signum, frame):
        if self._stop_requested:
            if callable(self._prev_sigint):
                self._prev_sigint(signum, frame)
            else:
                raise KeyboardInterrupt
            return
        self._request_stop("ctrl-c")

    def _poll_loop(self) -> None:
        try:
            import msvcrt  # type: ignore
        except ImportError:
            return
        while not self._stop_event.is_set():
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in self.KEYS_STOP:
                        self._request_stop(f"key {ch.decode(errors='replace')!r}")
                    elif ch in self.KEYS_PAUSE:
                        self._pause_key = not self._pause_key
                    elif ch in self.KEYS_RESUME:
                        self._pause_key = False
            except Exception:
                return
            self._stop_event.wait(0.1)


GracefulStopCallback = TrainControlCallback        # former name


# ─────────────────────────────────────────────────────────────────────────────
# Loop diagnostics: learned gates, per-iteration convergence
# ─────────────────────────────────────────────────────────────────────────────
def _raw_model(pl_module):
    """The TransformerLM behind the LightningModule (unwrapping torch.compile)."""
    m = pl_module.model
    return getattr(m, "_orig_mod", m)


class GateAlphaCallback(pl.Callback):
    """Log the learned loop gates of every gated block as ``gate/alpha_attn_l{ℓ}``
    and (block mode) ``gate/alpha_ff_l{ℓ}``.

    Logged on_step during training — Lightning writes them to TensorBoard every
    ``log_every_n_steps`` — so the chart of α vs training step comes for free,
    and ``val_metrics.csv`` picks up the value current at each validation. A
    plain model has no gates and logs nothing.
    """

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gates = _raw_model(pl_module).gate_values()
        for name, value in gates.items():
            pl_module.log(f"gate/{name}", value, on_step=True, on_epoch=False)


class LoopConvergenceCallback(pl.Callback):
    """Per-token convergence of the loop, measured on the first ``n_batches``
    validation batches with ``TransformerLM.trace_loop()``.

    For s_0 the block input and s_i the residual after iteration i (i = 1..R):

        rel[i]           = ‖s_i − s_{i−1}‖ / ‖s_{i−1}‖
        one_minus_cos[i] = 1 − cos(s_i, s_{i−1})
        kl[i]            = KL(p_i ‖ p_{i−1}),  p_i = softmax(lm_head(final_norm(s_i)))
        nll[i]           = −log p_i(next token)                     (npz only)

    ``kl`` / ``nll`` (``output_kl=True``) are exact model outputs for the last
    block in block mode and logit-lens readouts otherwise — see ``LoopTrace``.

    Scalars (mean over supervised tokens; ``_max`` = max over tokens and layers):
        val/conv/rel_mean_i{i}, val/conv/rel_max_i{i}, val/conv/cos_i{i},
        val/conv/kl_i{i}                                                   (layer-averaged)
        val/conv/rel_mean_l{ℓ}_i{i}, val/conv/cos_l{ℓ}_i{i}, val/conv/kl_l{ℓ}_i{i}
                                                                           (per looping block)

    Full arrays, one file per validation: ``<log_dir>/convergence/step{step}.npz``
    with ``rel`` / ``one_minus_cos`` / ``kl`` / ``nll`` as float32 (N, T, L, R)
    (``nll`` is NaN at the last position, which has no next token),
    ``input_ids`` (N, T), ``valid`` (N, T) bool, ``layers`` (L,), ``step``, ``loops``.

    With a mixture (one val loader per domain) everything is tagged by domain:
    scalars under ``val/conv/<domain>/…`` and files ``step{step}_<domain>.npz``,
    so convergence on code can be compared with convergence on prose directly.

    ``loops`` (R) defaults to the larger of ``loop_count`` and the
    ``val_loop_sweep`` maximum, so the curve extends past the training budget.
    Runs as a separate no-grad forward, so the headline val/loss is untouched.
    """

    def __init__(self, n_batches: int = 2, loops: int | None = None,
                 out_dir: str | Path | None = None, subdir: str = "convergence",
                 output_kl: bool = True):
        super().__init__()
        self.n_batches = int(n_batches)
        self.loops = int(loops) if loops else None
        self.output_kl = bool(output_kl)
        self.out_dir = out_dir
        self.subdir = subdir
        self._buf: dict[str | None, list[dict]] = {}
        self._layers: list[int] = []
        self._loops_used: int | None = None

    def _resolve_loops(self, pl_module) -> int:
        if self.loops is not None:
            return self.loops
        cfg = _raw_model(pl_module).cfg
        sweep = getattr(pl_module, "val_loop_sweep", None) or []
        return max([int(cfg.loop_count), *[int(r) for r in sweep]])

    def _out_path(self, trainer, tag: str | None) -> Path:
        base = Path(self.out_dir
                    or (trainer.logger.log_dir if trainer.logger else None)
                    or trainer.default_root_dir) / self.subdir
        base.mkdir(parents=True, exist_ok=True)
        return base / (f"step{trainer.global_step:06d}_{tag}.npz" if tag
                       else f"step{trainer.global_step:06d}.npz")

    @staticmethod
    def _tag(pl_module, dataloader_idx: int) -> str | None:
        domains = getattr(pl_module, "domains", None)
        return domains[dataloader_idx] if domains else None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                                dataloader_idx=0):
        if trainer.sanity_checking or batch_idx >= self.n_batches:
            return
        model = _raw_model(pl_module)
        if not model.is_looped:
            return
        loops = self._resolve_loops(pl_module)
        ids, labels = batch["input_ids"], batch["labels"]
        # next-token targets aligned with the input (labels are unshifted)
        targets = F.pad(labels[:, 1:], (0, 1), value=-100) if self.output_kl else None
        with torch.no_grad(), trainer.precision_plugin.forward_context(), \
                model.trace_loop(output_kl=self.output_kl, targets=targets) as tr:
            model.backbone(ids, loops)
        arrs = tr.stacked()                                  # (B, T, L, R) fp32
        rel, omc = arrs["rel"], arrs["one_minus_cos"]
        kl = arrs.get("kl")
        valid = labels != -100                               # (B, T)
        n_valid = int(valid.sum())
        if n_valid == 0:
            return
        w = valid.to(rel.dtype)[..., None, None]
        rel_li = (rel * w).sum((0, 1)) / n_valid             # (L, R) mean over tokens
        cos_li = (omc * w).sum((0, 1)) / n_valid
        kl_li = (kl * w).sum((0, 1)) / n_valid if kl is not None else None
        rel_max = rel.masked_fill(~valid[..., None, None], 0.0).amax((0, 1, 2))   # (R,)
        layers = tr.layers
        R = rel.shape[-1]
        tag = self._tag(pl_module, dataloader_idx)
        pre = f"val/conv/{tag}/" if tag else "val/conv/"
        log = lambda k, v, **kw: pl_module.log(pre + k, v, on_step=False, on_epoch=True,
                                                batch_size=n_valid, add_dataloader_idx=False, **kw)
        for i in range(R):
            log(f"rel_mean_i{i + 1}", rel_li[:, i].mean())
            log(f"cos_i{i + 1}", cos_li[:, i].mean())
            log(f"rel_max_i{i + 1}", rel_max[i], reduce_fx="max")
            if kl_li is not None:
                log(f"kl_i{i + 1}", kl_li[:, i].mean())
            for j, l in enumerate(layers):
                log(f"rel_mean_l{l}_i{i + 1}", rel_li[j, i])
                log(f"cos_l{l}_i{i + 1}", cos_li[j, i])
                if kl_li is not None:
                    log(f"kl_l{l}_i{i + 1}", kl_li[j, i])

        self._layers, self._loops_used = layers, loops
        self._buf.setdefault(tag, []).append(dict(
            **{k: a.cpu().numpy() for k, a in arrs.items()},
            input_ids=ids.cpu().numpy().astype(np.int32), valid=valid.cpu().numpy(),
        ))

    def on_validation_epoch_end(self, trainer, pl_module):
        bufs, self._buf = self._buf, {}
        if not bufs or trainer.sanity_checking or not trainer.is_global_zero:
            return
        for tag, buf in bufs.items():
            path = self._out_path(trainer, tag)
            np.savez(
                path,
                **{k: np.concatenate([b[k] for b in buf], axis=0) for k in buf[0]},
                layers=np.asarray(self._layers, dtype=np.int32),
                step=np.int64(trainer.global_step),
                loops=np.int64(self._loops_used),
            )
            log.info(f"[LoopConvergenceCallback] wrote {path}")
