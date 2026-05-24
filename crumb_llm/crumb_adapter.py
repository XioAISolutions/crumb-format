"""Crumb-aware structural priors for the Wave-Field LM.

CRUMB documents have a rich, machine-readable structure that maps
naturally onto knobs the Wave-Field block already exposes:

================================  =======================================
CRUMB feature                     Wave-Field knob it informs
================================  =======================================
Section boundary ``[name]``       Zeroes scatter weight at the boundary,
                                   so cross-section information has to
                                   propagate via the field rather than
                                   leak through the adjacent token.
``@priority: N`` (1..10)          Per-token scatter amplitude — higher
                                   priority deposits a louder signal,
                                   damped by the same field dynamics.
``fold:NAME/summary`` body        Per-token damping prior — summary
                                   tokens are tagged "long-range" so the
                                   adapter reduces α (longer decay).
``fold:NAME/full`` body           Per-token damping prior — full bodies
                                   get the default α (local detail).
``refs=sha256:…``                 Reserved for future ref-cache reuse
                                   (analogous to KV-cache reuse). The
                                   adapter currently records ref ids
                                   but does not load cached field state.
================================  =======================================

The output of ``CrumbPriorBuilder.build`` is a dict of tensors that can
be passed straight into ``WaveFieldLM.forward(...)``:

    {
        "input_ids":       LongTensor [1, N]
        "scatter_weights": FloatTensor [1, N]
        "section_ids":     LongTensor [1, N]   # debug/analysis
    }

The Wave-Field LM gracefully ignores any prior that isn't passed; this
adapter is opt-in and never required.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

# Reuse the canonical parser from cli/crumb.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "cli"))
from crumb import parse_crumb  # type: ignore  # noqa: E402

from .tokenizer import ByteTokenizer  # noqa: E402


_FOLD_RE = re.compile(r"^fold:([a-zA-Z0-9_-]+)/(summary|full)$")
_PRIORITY_RE = re.compile(r"^@priority:\s*(\d+)$")
_TYPE_RE = re.compile(r"^@type:\s*(.+)$")

# Section-boundary marker token (rendered as the literal byte sequence
# "\x1f" in the input stream — Unit Separator, never appears in
# real crumb text). The model learns to treat it as a soft barrier.
SECTION_SEP = "\x1f"


@dataclass
class CrumbPriors:
    input_ids: Tensor              # [1, N] long
    scatter_weights: Tensor        # [1, N] float
    section_ids: Tensor            # [1, N] long — section index per tok
    ref_digests: list[str]         # for future ref-cache reuse


class CrumbPriorBuilder:
    """Convert a parsed crumb into tensors for ``WaveFieldLM.forward``.

    The text fed to the model is the rendered crumb body (sections only,
    no headers) interleaved with ``SECTION_SEP`` between sections. The
    scatter-weight ramp uses @priority annotations to scale per-token
    amplitude; tokens in summary fold sections get a gentle long-range
    boost (1.5x); tokens at section boundaries get a damping spike
    (weight 0.0, so they don't leak across sections).
    """

    def __init__(self, tokenizer=None, default_priority: int = 5):
        self.tok = tokenizer or ByteTokenizer()
        self.default_priority = default_priority

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _section_priority(body: list[str], default: int) -> int:
        for line in body[:2]:
            m = _PRIORITY_RE.match(line.strip())
            if m:
                return int(m.group(1))
        return default

    @staticmethod
    def _section_amplitude(name: str, priority: int) -> float:
        """Map (section-name, priority) → per-token scatter amplitude."""
        amp = priority / 5.0  # 1..10 → 0.2..2.0, centred at 5→1.0
        m = _FOLD_RE.match(name)
        if m and m.group(2) == "summary":
            amp *= 1.5     # summaries propagate further
        return amp

    # ── Main entry ───────────────────────────────────────────────────

    def build(self, crumb_text: str) -> CrumbPriors:
        parsed = parse_crumb(crumb_text)
        headers: dict[str, str] = parsed["headers"]
        sections: dict[str, list[str]] = parsed["sections"]

        ids: list[int] = []
        weights: list[float] = []
        section_ids: list[int] = []
        ref_digests: list[str] = self._extract_refs(headers, sections)

        for section_idx, (name, body) in enumerate(sections.items()):
            # Skip refs section in the input stream; it's metadata.
            if name == "refs":
                continue
            priority = self._section_priority(body, self.default_priority)
            amp = self._section_amplitude(name, priority)
            # Section-header text itself, encoded as the literal section
            # name so the model can attend to it.
            header_text = f"[{name}]\n"
            for b in self.tok.encode(header_text):
                ids.append(b)
                weights.append(amp)
                section_ids.append(section_idx)
            # Body lines.
            for line in body:
                if line.strip().startswith("@"):
                    continue
                line_text = line + "\n"
                for b in self.tok.encode(line_text):
                    ids.append(b)
                    weights.append(amp)
                    section_ids.append(section_idx)
            # Section separator: zero amplitude, distinct token.
            for b in self.tok.encode(SECTION_SEP):
                ids.append(b)
                weights.append(0.0)
                section_ids.append(section_idx)

        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        scatter_weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)
        section_ids_t = torch.tensor(section_ids, dtype=torch.long).unsqueeze(0)

        return CrumbPriors(
            input_ids=input_ids,
            scatter_weights=scatter_weights,
            section_ids=section_ids_t,
            ref_digests=ref_digests,
        )

    @staticmethod
    def _extract_refs(headers: dict, sections: dict) -> list[str]:
        out: list[str] = []
        for ref in (headers.get("refs") or "").split(","):
            ref = ref.strip()
            if ref.startswith("sha256:"):
                out.append(ref)
        for line in sections.get("refs", []):
            for tok in line.split():
                if tok.startswith("sha256:"):
                    out.append(tok)
        return out


def iter_crumb_files(root: Path) -> Iterable[Path]:
    """Walk a directory and yield every ``*.crumb`` file."""
    for p in Path(root).rglob("*.crumb"):
        if p.is_file():
            yield p
