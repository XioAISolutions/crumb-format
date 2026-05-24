"""Corpus loaders for Wave-Field LM training.

Three sources, in order of preference for a given run:

1. ``--data <path>`` — a plain text file or directory of ``*.crumb`` files.
2. The repo's own ``examples/`` directory of crumbs (always available).
3. A tiny synthetic corpus (constant), used as a smoke-test fallback so
   ``crumb wave train --config tiny`` works even on an empty machine.

We deliberately do not depend on the HuggingFace datasets library —
it's a heavy install for a small toy. WikiText-2 reproduction lives in
``scripts/wikitext_train.py`` (out of repo) and isn't required for the
package to function.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor

from .crumb_adapter import iter_crumb_files


SYNTHETIC_CORPUS = """\
The wave field carries information from one token to another by exchanging
energy through a continuous medium. Each head learns a damping, a frequency,
and a phase; together they shape the kernel that propagates across the field.
Tokens scatter their state, the kernel convolves, and they gather back.

CRUMB documents are structured handoffs: goal, context, constraints. A wave
field model can read them as long as their sections are bounded by section
separators, so cross-section leakage is gated by the field itself.

The wave equation says that disturbance at a point creates oscillation at
neighbouring points after a delay set by the propagation speed. In our
discrete model the delay is the FFT-induced phase shift, and the speed
is folded into the per-head omega.
""" * 12


# ── Loaders ──────────────────────────────────────────────────────────


def load_text_corpus(path: str | Path | None) -> str:
    """Return a single concatenated string from path / examples / synthetic.

    Resolution order:
        * If ``path`` is a file: read it.
        * If ``path`` is a directory: concatenate all ``*.txt`` and
          ``*.crumb`` files under it.
        * If ``path`` is None: walk ``examples/`` in the repo.
        * If that yields nothing: return ``SYNTHETIC_CORPUS``.
    """
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
        if p.is_dir():
            chunks = []
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix in (".txt", ".crumb", ".md"):
                    chunks.append(f.read_text(encoding="utf-8", errors="replace"))
            if chunks:
                return "\n\n".join(chunks)
            return SYNTHETIC_CORPUS

    # Default: project examples/.
    repo_root = Path(__file__).resolve().parent.parent
    examples = repo_root / "examples"
    chunks = []
    if examples.exists():
        for f in iter_crumb_files(examples):
            chunks.append(f.read_text(encoding="utf-8", errors="replace"))
    if chunks:
        return "\n\n".join(chunks)
    return SYNTHETIC_CORPUS


# ── Batching ─────────────────────────────────────────────────────────


@dataclass
class TokenStream:
    """In-memory token stream with random-window batching."""

    ids: Tensor  # [T] long
    block_size: int

    def __len__(self) -> int:
        return max(0, self.ids.numel() - self.block_size - 1)

    def batch(self, batch_size: int, rng: random.Random | None = None) -> tuple[Tensor, Tensor]:
        """Return (x, y) with shape ``[batch_size, block_size]``."""
        rng = rng or random
        hi = self.ids.numel() - self.block_size - 1
        idxs = [rng.randint(0, hi) for _ in range(batch_size)]
        x = torch.stack([self.ids[i : i + self.block_size] for i in idxs])
        y = torch.stack([self.ids[i + 1 : i + 1 + self.block_size] for i in idxs])
        return x, y


def make_token_stream(text: str, tokenizer, block_size: int) -> TokenStream:
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return TokenStream(ids=ids, block_size=block_size)


def train_eval_split(text: str, eval_frac: float = 0.05) -> tuple[str, str]:
    """Hold out the tail of ``text`` as eval."""
    n = len(text)
    split = max(1, int(n * (1.0 - eval_frac)))
    return text[:split], text[split:]
