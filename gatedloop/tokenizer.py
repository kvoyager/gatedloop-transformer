"""
tokenizer.py
─────────────────────────────────────────────────────────────────────────────
A small byte-level BPE for text pretraining.

Byte-level BPE (HuggingFace ``tokenizers``) never produces an ``<unk>``: every
byte is in the alphabet, so any input round-trips. Only three special tokens are
reserved — ``<pad>``, ``<bos>``, ``<eos>`` — and they are reserved FIRST so they
take stable low ids (0, 1, 2) across retrainings; ids are nonetheless always
resolved by name (``token_to_id``), never hard-coded.

Vocabulary size is the main lever on parameter count at this scale: with tied
embeddings the table is ``vocab_size × d_model``, which for a 10M model is a
third of the budget. 8192 is a reasonable default for a small/simple corpus
(TinyStories); raise it to 16–32k for broader text, and re-check that the
embedding table still fits the parameter budget you want.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

PAD = "<pad>"
BOS = "<bos>"            # start of a document
EOS = "<eos>"            # end of a document

SPECIAL_TOKENS: list[str] = [PAD, BOS, EOS]


class TextTokenizer:
    """Thin wrapper over a HuggingFace ``Tokenizer`` with the special ids
    resolved by name and a batch encoder used by the corpus preparation script."""

    def __init__(self, tk):
        self._tk = tk
        self._ids: dict[str, int] = {}
        for t in SPECIAL_TOKENS:
            i = tk.token_to_id(t)
            if i is None:
                raise ValueError(f"special token {t!r} missing from tokenizer vocab")
            self._ids[t] = i

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def train(
        cls,
        text_iter: Iterable[str],
        vocab_size: int,
        save_path: str | Path,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> "TextTokenizer":
        """Train a byte-level BPE on ``text_iter`` and save it to ``save_path``."""
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tk = Tokenizer(models.BPE(unk_token=None))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,          # reserved first → stable low ids
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=show_progress,
        )
        tk.train_from_iterator(text_iter, trainer=trainer)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tk.save(str(save_path))
        return cls(tk)

    @classmethod
    def load(cls, path: str | Path) -> "TextTokenizer":
        from tokenizers import Tokenizer

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"tokenizer not found at {p}\n"
                f"Train it first (once):  python scripts/train_tokenizer.py\n"
                f"then build the data:    python scripts/prepare_data.py"
            )
        return cls(Tokenizer.from_file(str(p)))

    # ── special ids ──────────────────────────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._ids[PAD]

    @property
    def bos_id(self) -> int:
        return self._ids[BOS]

    @property
    def eos_id(self) -> int:
        return self._ids[EOS]

    def id_of(self, token: str) -> int:
        return self._ids[token]

    # ── text ─────────────────────────────────────────────────────────────────
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._tk.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(
        self, texts: Sequence[str], add_bos: bool = True, add_eos: bool = True
    ) -> list[list[int]]:
        """Fast multi-text encode (used by the corpus tokenizer). Wraps each
        document in ``<bos> … <eos>`` by default so packed blocks carry document
        boundaries the model can learn to respect."""
        encs = self._tk.encode_batch(list(texts))
        out = []
        for e in encs:
            ids = e.ids
            if add_bos:
                ids = [self.bos_id] + ids
            if add_eos:
                ids = ids + [self.eos_id]
            out.append(ids)
        return out

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        return self._tk.decode(list(ids), skip_special_tokens=skip_special)
