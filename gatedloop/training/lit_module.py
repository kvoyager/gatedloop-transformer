"""
lit_module.py
─────────────────────────────────────────────────────────────────────────────
The pretraining LightningModule: next-token prediction on packed text.

Nothing here is architecture-specific — plain and looped models take the exact
same step — except a few small things that make the loop and the data legible:

  • ``train/loops`` logs the R actually used at each step. With
    ``loop_sample=true`` that is a fresh draw from U[loop_min, loop_max], so the
    curve should look like noise around the mean; it is the cheapest check that
    sampling is live.
  • ``val_loop_sweep`` re-runs each validation batch at several R values and
    logs ``val/ppl_r{R}``. That is the test-time-compute curve, measured during
    training instead of after it: if looping is doing anything, ppl should fall
    as R rises, and it tells you where the returns flatten. Plain models skip
    the sweep (every R is the same forward).
  • With a MIXTURE (``domains`` set) validation gets one DataLoader per domain
    and every val metric appears per domain — ``val/ppl_code``,
    ``val/ppl_code_r{R}`` — next to the headline, which is the share-weighted
    combination of the per-domain means. ``train/loss_<domain>`` is logged
    every ``log_every_n_steps`` from the domain tag each batch item carries.

The headline ``val/loss`` / ``val/ppl`` always use the model's configured
``loop_count`` so checkpoint selection compares like with like across runs.
"""

from __future__ import annotations

from typing import Any, Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from gatedloop.model.lm import ModelConfig, TransformerLM
from gatedloop.optim import build_optimizer, cosine_warmup_lambda


class LitPretrain(pl.LightningModule):
    def __init__(
        self,
        model: dict,                        # ModelConfig fields (resolved by the train script)
        lr: float = 6e-4,
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        warmup_steps: int = 1000,
        max_steps: int = 30000,
        lr_decay_steps: int | None = None,  # None → follow max_steps
        lr_floor: float = 0.1,
        val_loop_sweep: Sequence[int] | None = None,
        domains: Sequence[str] | None = None,   # mixture: domain names, in val-loader order
        shares: Sequence[float] | None = None,  # mixture: matching token shares
    ):
        super().__init__()
        # Coerce to plain Python BEFORE save_hyperparameters() snapshots the init
        # args: an OmegaConf ListConfig here would be pickled into every
        # checkpoint and rejected by torch.load's weights_only default
        # (PyTorch ≥ 2.6) when the run is resumed.
        val_loop_sweep = [int(r) for r in (val_loop_sweep or [])]
        domains = [str(d) for d in (domains or [])] or None
        shares = [float(s) for s in (shares or [])] or None
        if domains and (not shares or len(shares) != len(domains)):
            raise ValueError("domains and shares must have the same length")
        self.save_hyperparameters()
        self.model = TransformerLM(ModelConfig(**model))
        self.val_loop_sweep = val_loop_sweep
        self.domains = domains
        self.shares = shares
        # mixture: per-(suffix, domain) running sums for the weighted headline
        self._val_acc: dict[tuple[str, int], list[float]] = {}

    @property
    def is_mixture(self) -> bool:
        return bool(self.domains)

    def forward(self, input_ids, labels=None, loops=None):
        return self.model(input_ids, labels=labels, loops=loops)

    # ── logging helpers ───────────────────────────────────────────────────────
    def _log_set(self, stage: str, suffix: str, loss, ppl, acc, weight: int,
                 on_step: bool, prog_bar: bool) -> None:
        kw = dict(on_step=on_step, on_epoch=True, batch_size=weight, add_dataloader_idx=False)
        self.log(f"{stage}/loss{suffix}", loss, prog_bar=prog_bar, **kw)
        self.log(f"{stage}/ppl{suffix}", ppl, prog_bar=prog_bar, **kw)
        self.log(f"{stage}/token_acc{suffix}", acc, **kw)

    def _accumulate_val(self, suffix: str, idx: int, loss, ppl, acc, bs: int) -> None:
        """Lightning cannot reduce ONE key across several val dataloaders, so the
        mixture headline (Σ_d share_d · mean_d) is summed here per (suffix,
        domain) and logged once in ``on_validation_epoch_end``."""
        a = self._val_acc.setdefault((suffix, idx), [0.0, 0.0, 0.0, 0])
        a[0] += float(loss) * bs
        a[1] += float(ppl) * bs
        a[2] += float(acc) * bs
        a[3] += bs

    def on_validation_start(self) -> None:
        self._val_acc = {}

    def on_validation_epoch_end(self) -> None:
        if not self.is_mixture or not self._val_acc:
            return
        for suffix in sorted({k[0] for k in self._val_acc}):
            loss = ppl = acc = 0.0
            for i, share in enumerate(self.shares):
                a = self._val_acc.get((suffix, i))
                if a and a[3]:
                    loss += share * a[0] / a[3]
                    ppl += share * a[1] / a[3]
                    acc += share * a[2] / a[3]
            kw = dict(on_step=False, on_epoch=True, prog_bar=not suffix)
            self.log(f"val/loss{suffix}", loss, **kw)
            self.log(f"val/ppl{suffix}", ppl, **kw)
            self.log(f"val/token_acc{suffix}", acc, on_step=False, on_epoch=True)
        self._val_acc = {}

    # ── shared step ───────────────────────────────────────────────────────────
    def _shared_step(self, batch: dict, stage: str, loops: int | None = None,
                     suffix: str = "", dataloader_idx: int = 0) -> torch.Tensor:
        loops = self.model._resolve_loops(loops)
        out = self.model(batch["input_ids"], labels=batch["labels"], loops=loops)
        loss = out.loss

        with torch.no_grad():
            shift_labels = batch["labels"][:, 1:]
            pred = out.logits[:, :-1].argmax(dim=-1)
            valid = shift_labels != -100
            acc = ((pred[valid] == shift_labels[valid]).float().mean()
                   if valid.any() else torch.zeros((), device=loss.device))
            ppl = torch.exp(loss.detach().clamp(max=20.0))

        bs = batch["input_ids"].size(0)
        on_step = stage == "train"
        if stage == "val" and self.is_mixture:
            name = self.domains[dataloader_idx]
            self._log_set(stage, f"_{name}{suffix}", loss, ppl, acc, bs, on_step=False, prog_bar=False)
            self._accumulate_val(suffix, dataloader_idx, loss, ppl, acc, bs)
        else:
            self._log_set(stage, suffix, loss, ppl, acc, bs, on_step=on_step, prog_bar=not suffix)
        if not suffix and dataloader_idx == 0:
            self.log(f"{stage}/loops", float(loops), on_step=on_step,
                     on_epoch=not on_step, batch_size=bs, add_dataloader_idx=False)
        if stage == "train" and not suffix and self.is_mixture and "domain" in batch:
            self._log_train_domains(out.logits, batch, bs)
        return loss

    @torch.no_grad()
    def _log_train_domains(self, logits, batch, bs: int) -> None:
        """Per-domain training loss from the domain tag on each item — only on
        logging steps, since it re-runs the cross-entropy over the batch."""
        every = max(1, int(getattr(self.trainer, "log_every_n_steps", 50) or 50))
        if self.global_step % every:
            return
        shift_logits, shift_labels = logits[:, :-1], batch["labels"][:, 1:]
        dom = batch["domain"].view(-1)
        for i, name in enumerate(self.domains):
            m = dom == i
            if not m.any():
                continue
            lg, lb = shift_logits[m], shift_labels[m]
            l = F.cross_entropy(lg.reshape(-1, lg.size(-1)), lb.reshape(-1), ignore_index=-100)
            self.log(f"train/loss_{name}", l, on_step=True, on_epoch=False,
                     batch_size=int(m.sum()), add_dataloader_idx=False)

    def training_step(self, batch, batch_idx):
        if self.trainer is not None and self.trainer.optimizers:
            self.log("lr", self.trainer.optimizers[0].param_groups[-1]["lr"],
                     prog_bar=True, on_step=True, on_epoch=False)
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        loss = self._shared_step(batch, "val", dataloader_idx=dataloader_idx)
        # Test-time-compute sweep: the same batch at other loop counts.
        if self.model.is_looped:
            for r in self.val_loop_sweep:
                self._shared_step(batch, "val", loops=r, suffix=f"_r{r}",
                                  dataloader_idx=dataloader_idx)
        return loss

    def on_train_epoch_start(self) -> None:
        # A MixtureDataset re-draws its per-domain permutations per epoch.
        dl = self.trainer.train_dataloader
        ds = getattr(dl, "dataset", None)
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(self.current_epoch)

    # ── optimizer / schedule ──────────────────────────────────────────────────
    def configure_optimizers(self) -> Any:
        opt = build_optimizer(
            self.model, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        decay = self.hparams.lr_decay_steps or self.hparams.max_steps
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, cosine_warmup_lambda(self.hparams.warmup_steps, decay, self.hparams.lr_floor)
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_before_optimizer_step(self, optimizer):
        # Lightning already clipped by then; this only *measures* the norm.
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float("inf"))
        self.log("train/grad_norm", norm, on_step=True, on_epoch=False)
