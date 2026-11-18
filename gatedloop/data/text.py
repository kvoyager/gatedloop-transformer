"""
text.py
─────────────────────────────────────────────────────────────────────────────
Packed next-token dataset over a pre-tokenized ``.bin`` corpus.

``scripts/prepare_data.py`` tokenizes the corpus once (documents wrapped in
``<bos> … <eos>``) into a flat ``uint16`` array on disk. This class memmaps that
array and serves contiguous blocks — the nanoGPT layout: cheap random access,
trivial multi-worker sharing (the OS page cache *is* the shared buffer), and no
re-tokenization per epoch.

Each item is ``{"input_ids": (seq_len+1,), "labels": (seq_len+1,)}`` with
``labels == input_ids``; the model applies the causal shift internally, so one
block yields exactly ``seq_len`` supervised predictions and no target is wasted
at the block boundary. Blocks are non-overlapping in their *targets*: block i
starts at ``i·seq_len`` and reads one extra token, so consecutive blocks share
only that boundary token as context.

Note the implication for the model config: attention sees ``seq_len + 1``
positions, so ``model.max_seq_len`` must be ≥ ``data.seq_len + 1`` (the train
script checks this).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTextDataset(Dataset):
    def __init__(self, bin_path: str | Path, seq_len: int, dtype=np.uint16):
        self.bin_path = str(bin_path)
        self.seq_len = int(seq_len)
        self.dtype = dtype
        if not Path(self.bin_path).exists():
            raise FileNotFoundError(
                f"{self.bin_path} not found — run scripts/prepare_data.py first"
            )
        # Memmapped lazily per worker in __getitem__ so the handle is not pickled
        # across the fork/spawn to DataLoader workers.
        self._data: np.memmap | None = None
        self.n_tokens = Path(self.bin_path).stat().st_size // np.dtype(dtype).itemsize
        # Need seq_len+1 tokens from offset i·seq_len for the last block too.
        self._n_blocks = max(0, (self.n_tokens - 1) // self.seq_len)
        if self._n_blocks == 0:
            raise ValueError(
                f"{self.bin_path} holds {self.n_tokens} tokens — too few for "
                f"seq_len={self.seq_len}"
            )

    def _ensure(self) -> np.memmap:
        if self._data is None:
            self._data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        return self._data

    def __len__(self) -> int:
        return self._n_blocks

    def __getitem__(self, idx: int) -> dict:
        data = self._ensure()
        i = idx * self.seq_len
        block = np.asarray(data[i : i + self.seq_len + 1], dtype=np.int64)
        ids = torch.from_numpy(block)
        return {"input_ids": ids, "labels": ids.clone()}


class MixtureDataset(Dataset):
    """Several ``PackedTextDataset``s mixed by share, with exact per-epoch
    counts and a deterministic, resumable order.

    One (virtual) epoch has ``N`` blocks — by default the total block count of
    all domains — of which domain ``d`` contributes ``round(share_d · N)``.
    Those are drawn from a seeded permutation of the domain's blocks, tiled if
    the domain is smaller than its allotment (a small domain therefore repeats
    cleanly rather than being sampled with replacement). The DataLoader's own
    ``shuffle=True`` interleaves the domains; with ``shuffle=False`` they come
    out in domain order.

    ``set_epoch(e)`` re-draws the permutations so successive epochs differ;
    the train module calls it at each epoch start. NB: with persistent
    DataLoader workers the workers keep their epoch-0 copy — the loader still
    reshuffles the order, so the only effect is that the same tiling repeats.

    Items carry ``"domain"`` (int index into ``self.names``) alongside
    ``input_ids`` / ``labels`` so the training step can log per-domain loss.
    """

    def __init__(self, datasets: dict[str, PackedTextDataset], shares: dict[str, float],
                 seed: int = 0, epoch_len: int | None = None):
        if not datasets:
            raise ValueError("MixtureDataset needs at least one domain")
        self.names = list(datasets)
        self.datasets = [datasets[n] for n in self.names]
        total = float(sum(shares[n] for n in self.names))
        self.shares = [float(shares[n]) / total for n in self.names]
        self.seed = int(seed)
        n = int(epoch_len or sum(len(d) for d in self.datasets))
        self._counts = self._exact_counts(self.shares, n)
        self.epoch = -1
        self.set_epoch(0)

    @staticmethod
    def _exact_counts(shares, n: int) -> list[int]:
        """Largest-remainder rounding so the counts sum to exactly ``n``."""
        raw = [s * n for s in shares]
        counts = [int(r) for r in raw]
        for i in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[: n - sum(counts)]:
            counts[i] += 1
        return counts

    @property
    def counts(self) -> dict[str, int]:
        return dict(zip(self.names, self._counts))

    def passes_per_epoch(self) -> dict[str, float]:
        """How many times each domain is traversed per epoch (>1 = repeats)."""
        return {n: c / len(d) for n, c, d in zip(self.names, self._counts, self.datasets)}

    def set_epoch(self, epoch: int) -> None:
        if epoch == self.epoch:
            return
        self.epoch = int(epoch)
        dom, blk = [], []
        for i, (ds, count) in enumerate(zip(self.datasets, self._counts)):
            rng = np.random.default_rng([self.seed, self.epoch, i])
            n_ds = len(ds)
            reps = -(-count // n_ds)                          # ceil
            perm = np.concatenate([rng.permutation(n_ds) for _ in range(reps)])[:count]
            dom.append(np.full(count, i, dtype=np.int8))
            blk.append(perm.astype(np.int64))
        self._domain = np.concatenate(dom)
        self._block = np.concatenate(blk)

    def __len__(self) -> int:
        return int(self._domain.size)

    def __getitem__(self, idx: int) -> dict:
        d = int(self._domain[idx])
        item = self.datasets[d][int(self._block[idx])]
        item["domain"] = torch.tensor(d, dtype=torch.int64)
        return item
