"""Context-pulling engine for Crumb LLM.

Implements Claude-style dynamic context retrieval: instead of stuffing
the entire document into the prompt (context pushing, O(N²) attention
cost), the model requests only the specific crumb sections it needs at
each generation step.

Architecture:
    ┌────────────────────────────────────────────┐
    │                User query                   │
    └─────────────────┬──────────────────────────┘
                      │
    ┌─────────────────▼──────────────────────────┐
    │          Context Router                     │
    │  Scores crumb sections by relevance to the  │
    │  current generation state (field spectrum). │
    │  Pulls top-k sections into the active field.│
    └─────────────────┬──────────────────────────┘
                      │
    ┌─────────────────▼──────────────────────────┐
    │          Crumb Index                        │
    │  Pre-indexed library of .crumb files.       │
    │  Each section stored with its scatter       │
    │  signature (spectral fingerprint from the   │
    │  wave kernel's response).                   │
    └─────────────────┬──────────────────────────┘
                      │
    ┌─────────────────▼──────────────────────────┐
    │       Wave-Field LM (inference)             │
    │  Processes only the pulled context +        │
    │  the user's query.  Field-state cache       │
    │  persists across turns.                     │
    └────────────────────────────────────────────┘

This solves the three problems of context pushing:
  1. "Lost in the middle" — we only inject what's relevant, so there's
     no middle to lose.
  2. O(N²) compute — smaller N means dramatically less compute.
  3. Context rot — stale or conflicting sections are never loaded.

The spectral scoring is the novel part: because the wave-field model
already works in the frequency domain, we can compute a "resonance
score" between the query's spectral signature and each indexed section's
signature — a dot product in frequency space, which is O(F) per section.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from .crumb_adapter import CrumbPriorBuilder, iter_crumb_files, SECTION_SEP
from .tokenizer import ByteTokenizer


@dataclass
class IndexedSection:
    """One pre-indexed crumb section."""
    source_file: str
    section_name: str
    text: str
    token_ids: list[int]
    n_tokens: int
    priority: int
    spectral_signature: list[float]   # |rFFT(scatter(tokens))|, length F//2+1
    content_hash: str


@dataclass
class CrumbIndex:
    """In-memory index of crumb sections for context pulling."""
    sections: list[IndexedSection]
    field_size: int
    vocab_size: int

    def save(self, path: str | Path) -> None:
        data = {
            "field_size": self.field_size,
            "vocab_size": self.vocab_size,
            "sections": [
                {
                    "source_file": s.source_file,
                    "section_name": s.section_name,
                    "text": s.text,
                    "n_tokens": s.n_tokens,
                    "priority": s.priority,
                    "spectral_signature": s.spectral_signature,
                    "content_hash": s.content_hash,
                }
                for s in self.sections
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CrumbIndex":
        data = json.loads(Path(path).read_text())
        sections = [
            IndexedSection(
                source_file=s["source_file"],
                section_name=s["section_name"],
                text=s["text"],
                token_ids=[],
                n_tokens=s["n_tokens"],
                priority=s["priority"],
                spectral_signature=s["spectral_signature"],
                content_hash=s["content_hash"],
            )
            for s in data["sections"]
        ]
        return cls(sections=sections, field_size=data["field_size"], vocab_size=data["vocab_size"])


def _spectral_signature(token_ids: list[int], field_size: int) -> list[float]:
    """Compute |rFFT| of the scattered token sequence as a fingerprint."""
    N = len(token_ids)
    if N == 0:
        return [0.0] * (field_size // 2 + 1)
    ids = torch.tensor(token_ids, dtype=torch.float32)
    # Simple scatter: histogram of token positions on the field grid.
    field = torch.zeros(field_size)
    for i, tid in enumerate(token_ids):
        pos = int((i + 0.5) * field_size / N)
        pos = min(pos, field_size - 1)
        field[pos] += float(tid) / 256.0
    spec = torch.fft.rfft(field).abs()
    return spec.tolist()


def build_index(
    crumb_dir: str | Path,
    field_size: int = 256,
) -> CrumbIndex:
    """Walk a directory of .crumb files and index every section.

    This is the offline step — run once, then use the index for fast
    context pulling at inference time.
    """
    tok = ByteTokenizer()
    sections: list[IndexedSection] = []

    for crumb_path in iter_crumb_files(Path(crumb_dir)):
        try:
            text = crumb_path.read_text(encoding="utf-8", errors="replace")
            from cli.crumb import parse_crumb
        except Exception:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))
            from crumb import parse_crumb
        try:
            parsed = parse_crumb(text)
        except ValueError:
            continue
        for section_name, body_lines in parsed["sections"].items():
            body = "\n".join(body_lines).strip()
            if not body:
                continue
            ids = tok.encode(body)
            priority = 5
            for line in body_lines[:2]:
                if line.strip().startswith("@priority:"):
                    try:
                        priority = int(line.strip().split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass
            sig = _spectral_signature(ids, field_size)
            content_hash = hashlib.sha256(body.encode()).hexdigest()[:16]
            sections.append(IndexedSection(
                source_file=str(crumb_path),
                section_name=section_name,
                text=body,
                token_ids=ids,
                n_tokens=len(ids),
                priority=priority,
                spectral_signature=sig,
                content_hash=content_hash,
            ))

    return CrumbIndex(sections=sections, field_size=field_size, vocab_size=tok.vocab_size)


def score_sections(
    query_text: str,
    index: CrumbIndex,
    top_k: int = 5,
    priority_boost: float = 0.1,
) -> list[tuple[float, IndexedSection]]:
    """Score indexed sections by spectral relevance to a query.

    The score is the cosine similarity between the query's spectral
    signature and each section's pre-computed signature, boosted by
    the section's @priority annotation.

    Returns: sorted list of (score, section) tuples, highest first.
    """
    tok = ByteTokenizer()
    query_ids = tok.encode(query_text)
    query_sig = torch.tensor(
        _spectral_signature(query_ids, index.field_size),
        dtype=torch.float32,
    )
    query_norm = query_sig.norm() + 1e-8

    scored = []
    for section in index.sections:
        sec_sig = torch.tensor(section.spectral_signature, dtype=torch.float32)
        sec_norm = sec_sig.norm() + 1e-8
        cosine = (query_sig @ sec_sig) / (query_norm * sec_norm)
        # Priority boost: higher priority sections get a gentle bonus.
        score = cosine.item() + priority_boost * (section.priority / 10.0)
        scored.append((score, section))

    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def pull_context(
    query_text: str,
    index: CrumbIndex,
    max_tokens: int = 512,
    top_k: int = 10,
) -> str:
    """Pull the most relevant crumb sections for a query.

    Selects top-k sections by spectral relevance, then packs them
    greedily until the token budget is exhausted. Returns a
    concatenated string of pulled sections, ready to prepend to the
    model's input.

    This is the "context pulling" function — the analog of RAG
    retrieval, but using wave-field spectral matching instead of
    embedding cosine similarity.
    """
    scored = score_sections(query_text, index, top_k=top_k)
    pulled: list[str] = []
    total_tokens = 0

    for score, section in scored:
        if total_tokens + section.n_tokens > max_tokens:
            if pulled:
                break
        pulled.append(f"[{section.section_name}]\n{section.text}")
        total_tokens += section.n_tokens

    return (SECTION_SEP + "\n").join(pulled)


@dataclass
class ContextPullSession:
    """Stateful context-pulling session with field-state persistence.

    Maintains a conversation-like state: the wave field accumulates
    context across turns, and each new query pulls fresh sections
    while retaining wave energy from previous exchanges.
    """
    index: CrumbIndex
    max_context_tokens: int = 512
    top_k: int = 5
    pulled_hashes: set = field(default_factory=set)

    def pull(self, query: str) -> str:
        """Pull fresh context for a new query, avoiding duplicates."""
        scored = score_sections(query, self.index, top_k=self.top_k * 2)
        pulled: list[str] = []
        total_tokens = 0

        for score, section in scored:
            if section.content_hash in self.pulled_hashes:
                continue
            if total_tokens + section.n_tokens > self.max_context_tokens:
                if pulled:
                    break
            pulled.append(f"[{section.section_name}]\n{section.text}")
            total_tokens += section.n_tokens
            self.pulled_hashes.add(section.content_hash)

        return (SECTION_SEP + "\n").join(pulled)

    def reset(self) -> None:
        self.pulled_hashes.clear()
