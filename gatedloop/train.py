"""
train.py
─────────────────────────────────────────────────────────────────────────────
Next-token pretraining on a packed text corpus. The only training entry point —
this repo does pretraining and nothing else.

  python -m gatedloop.train                          # plain_10m (default)
  python -m gatedloop.train model=looped_10m         # attention-looped
  python -m gatedloop.train model=looped_10m model.loop_count=6
  python -m gatedloop.train --config-name train_looped_10m
  python -m gatedloop.train resume=runs/pretrain/version_0/checkpoints/last.ckpt

Prerequisites (run once):
  python scripts/train_tokenizer.py
  python scripts/prepare_data.py
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Expandable segments curb CUDA memory fragmentation, which the looped model
# (activation memory paid R times over) is prone to. PyTorch supports them on
# Linux only; on Windows and macOS the setting is ignored with a warning, which
# is filtered below.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging
import sys
import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from gatedloop import paths
from gatedloop.data.loader import make_loader_kwargs
from gatedloop.data.mixture import build_datasets
from gatedloop.tokenizer import TextTokenizer
from gatedloop.training.callbacks import (
    CodeSnapshot, EpochSummary, GateAlphaCallback, LoopConvergenceCallback,
    LRRichProgressBar, ParamCountCallback, SampleTextCallback, ThroughputCallback,
    TrainControlCallback, ValMetricsCSV,
)
from gatedloop.training.lit_module import LitPretrain
from gatedloop.training.logfile import ConsoleLog

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*expandable_segments not supported.*")
torch.set_float32_matmul_precision("high")

# Relative paths in the config (artifacts/, runs/) are anchored at the repo root
# in a source checkout, or at the launch directory for an installed copy — see
# paths.py. Never at site-packages, and not at wherever Hydra moves the cwd.
_abs = paths.resolve


def _run_name(cfg: DictConfig) -> str:
    """``run_name`` from the config; ``null`` → the ``model=`` group choice
    (so the ladder lands in runs/<model>/), falling back to "pretrain"."""
    name = OmegaConf.select(cfg, "run_name", default=None)
    if name:
        return str(name)
    try:
        from hydra.core.hydra_config import HydraConfig
        choice = HydraConfig.get().runtime.choices.get("model")
        if choice:
            return str(choice)
    except ValueError:                      # not running under hydra.main
        pass
    return "pretrain"


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    out_dir = Path(_abs(cfg.out_dir))
    run_name = _run_name(cfg)
    tb = TensorBoardLogger(save_dir=str(out_dir), name=run_name)
    csv = CSVLogger(save_dir=str(out_dir), name=run_name, version=tb.version)

    # Mirror everything printed during the run into <run dir>/train.log. The
    # progress bar keeps the raw stdout (see logfile.py) so the file stays
    # readable; set log_file=null to skip the tee entirely.
    log_name = OmegaConf.select(cfg, "log_file", default="train.log")
    if not log_name:
        return _fit(cfg, tb, csv, sys.stdout)
    log_path = Path(tb.log_dir) / str(log_name)
    with ConsoleLog(log_path) as console:
        print(f"[log] console output → {log_path}")
        return _fit(cfg, tb, csv, console.original_stdout)


def _fit(cfg: DictConfig, tb: TensorBoardLogger, csv: CSVLogger, bar_stream) -> None:
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed, workers=True)

    tok = TextTokenizer.load(_abs(cfg.data.tokenizer_path))

    # A block is seq_len+1 tokens (seq_len supervised targets after the shift),
    # so the RoPE horizon must cover one more position than seq_len.
    if cfg.data.seq_len + 1 > cfg.model.max_seq_len:
        raise ValueError(
            f"data.seq_len+1 ({cfg.data.seq_len + 1}) exceeds model.max_seq_len "
            f"({cfg.model.max_seq_len}) — raise model.max_seq_len or lower data.seq_len"
        )

    # Single corpus (train_bin/val_bin) or a mixture (domains: …) — see data/mixture.py.
    data = build_datasets(cfg.data, _abs, seed=int(cfg.seed))
    if data.is_mixture:
        print(f"[data] mixture of {len(data.domains)} domains, "
              f"{data.info['train_tokens']/1e6:.1f}M train tokens / {data.info['epoch_blocks']} blocks per epoch")
        for name, d in data.info["domains"].items():
            print(f"       {name:<10} share {d['share']:.2f}  train {d['train_tokens']/1e6:8.1f}M tokens"
                  f"  val {d['val_tokens']/1e6:5.2f}M  ({d['passes_per_epoch']:.2f} passes/epoch)")
    else:
        print(f"[data] train {data.info['train_tokens']/1e6:.1f}M tokens / {len(data.train)} blocks"
              + (f" | val {data.info['val_tokens']/1e6:.1f}M tokens" if data.val else " | no val split"))
    # For the banner: tokens per optimizer step → effective passes over each domain.
    data.info["tokens_per_step"] = (int(cfg.dataloader.batch_size)
                                    * int(cfg.trainer.accumulate_grad_batches) * int(cfg.data.seq_len))
    data.info["max_steps"] = int(cfg.trainer.max_steps)

    train_loader = DataLoader(
        data.train, shuffle=True, drop_last=True,
        **make_loader_kwargs(cfg.dataloader, val=False),
    )
    val_loaders = [DataLoader(ds, shuffle=False, **make_loader_kwargs(cfg.dataloader, val=True))
                   for ds in data.val.values()]
    # A mixture validates on one loader per domain (Lightning passes dataloader_idx).
    val_loader = (val_loaders if data.is_mixture else val_loaders[0]) if val_loaders else None

    # Architecture config + runtime token ids → the ModelConfig dict.
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.update(vocab_size=tok.vocab_size, pad_id=tok.pad_id)

    model = LitPretrain(
        model=model_cfg,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        beta1=cfg.optimizer.beta1,
        beta2=cfg.optimizer.beta2,
        warmup_steps=cfg.optimizer.warmup_steps,
        max_steps=cfg.trainer.max_steps,
        lr_decay_steps=OmegaConf.select(cfg, "optimizer.lr_decay_steps", default=None),
        lr_floor=cfg.optimizer.lr_floor,
        val_loop_sweep=OmegaConf.select(cfg, "val_loop_sweep", default=None),
        domains=data.domains,
        shares=data.shares,
    )
    model.data_info = data.info          # read by the ParamCountCallback banner
    if OmegaConf.select(cfg, "trainer.compile", default=False):
        # Sampled R changes the graph, so expect one compile per distinct R.
        model.model = torch.compile(model.model)

    ckpt_dir = os.path.join(tb.log_dir, "checkpoints")
    callbacks = [
        ParamCountCallback(),
        ThroughputCallback(),
        CodeSnapshot(),
        ModelCheckpoint(dirpath=ckpt_dir, filename="step{step:06d}", monitor="step",
                        mode="max", save_top_k=3, save_last=True,
                        every_n_train_steps=cfg.trainer.val_check_interval,
                        auto_insert_metric_name=False),
        LearningRateMonitor(logging_interval="step"),
        ValMetricsCSV(),
        GateAlphaCallback(),
        EpochSummary(),
        LRRichProgressBar(leave=True, console_kwargs={"file": bar_stream}),
        TrainControlCallback(),
    ]
    if val_loader is not None:
        callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, filename="best",
                                         monitor="val/loss", mode="min", save_top_k=1))
        # Per-iteration loop convergence on a few val batches (looped models only;
        # the callback is a no-op for a plain model).
        conv_n = OmegaConf.select(cfg, "val_convergence.n_batches", default=0) or 0
        if conv_n > 0:
            callbacks.append(LoopConvergenceCallback(
                n_batches=conv_n,
                loops=OmegaConf.select(cfg, "val_convergence.loops", default=None),
                output_kl=OmegaConf.select(cfg, "val_convergence.output_kl", default=True),
            ))
    # null / 0 / absent all disable the sample callback.
    sample_every = OmegaConf.select(cfg, "sample_every", default=None)
    if sample_every and sample_every > 0:
        callbacks.append(SampleTextCallback(
            tok, every_n_steps=sample_every,
            prompt=OmegaConf.select(cfg, "sample_prompt", default=None),
            add_bos=OmegaConf.select(cfg, "data.has_bos", default=True),
        ))

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_steps=cfg.trainer.max_steps,
        # Lightning counts val_check_interval in dataloader BATCHES; the config
        # means optimizer steps, so scale by the accumulation factor.
        val_check_interval=(int(cfg.trainer.val_check_interval)
                            * int(cfg.trainer.accumulate_grad_batches) if val_loader else None),
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        num_sanity_val_steps=OmegaConf.select(cfg, "trainer.num_sanity_val_steps", default=2),
        callbacks=callbacks,
        logger=[tb, csv],
    )
    # weights_only=False: resuming our own checkpoints, which carry hparams and
    # callback state beyond bare tensors (rejected by torch ≥ 2.6's default).
    trainer.fit(model, train_loader, val_loader,
                ckpt_path=_abs(OmegaConf.select(cfg, "resume", default=None)),
                weights_only=False)


if __name__ == "__main__":
    main()
