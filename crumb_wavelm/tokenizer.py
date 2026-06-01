"""Tiny tokenizers for Wave-Field LLM.

Two zero-dep tokenizers are provided:

* ``ByteTokenizer``  — UTF-8 byte-level. 256 vocab. Lossless on any text.
* ``CharTokenizer``  — corpus-derived character vocab. Smaller, faster
  for toy training runs.

Both expose the same minimal interface used by ``train.py`` and
``sample.py``:

    enc.encode(text: str) -> list[int]
    enc.decode(ids: list[int]) -> str
    enc.vocab_size  -> int
    enc.eos_id      -> int | None

Neither is BPE: for a research-scale repro you'd want a real tokenizer
(GPT-2 BPE works fine), but for the toy/small configs that ship with
this package, byte- or char-level is the most reliable choice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


# ── Byte-level ───────────────────────────────────────────────────────


class ByteTokenizer:
    """UTF-8 byte tokenizer. Vocab is fixed at 256."""

    vocab_size = 256
    eos_id: int | None = None  # No reserved EOS in raw bytes.

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Iterable[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "byte"}))

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        meta = json.loads(Path(path).read_text())
        assert meta.get("type") == "byte"
        return cls()


# ── Character-level ──────────────────────────────────────────────────


class CharTokenizer:
    """Corpus-derived character vocab + special tokens (PAD, EOS)."""

    def __init__(self, chars: list[str], specials: list[str] | None = None):
        specials = specials or ["<pad>", "<eos>"]
        # Specials get the lowest ids so they're easy to spot in logits.
        self.itos: list[str] = list(specials) + chars
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.eos_id = self.stoi["<eos>"]
        self.vocab_size = len(self.itos)

    @classmethod
    def fit(cls, corpus: str, specials: list[str] | None = None) -> "CharTokenizer":
        chars = sorted(set(corpus))
        return cls(chars, specials=specials)

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        unk_id = self.stoi.get("?", 0)  # no real UNK; fall back to '?' or pad.
        for ch in text:
            out.append(self.stoi.get(ch, unk_id))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(
            self.itos[i] if 0 <= i < len(self.itos) and not self.itos[i].startswith("<")
            else ""
            for i in ids
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"type": "char", "itos": self.itos}, ensure_ascii=False)
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        meta = json.loads(Path(path).read_text())
        assert meta.get("type") == "char"
        itos = meta["itos"]
        # Reconstruct: split specials from regular chars.
        specials = [c for c in itos if c.startswith("<")]
        chars = [c for c in itos if not c.startswith("<")]
        return cls(chars, specials=specials)
