"""Honest compression measurement — the ``crumb measure`` command.

``crumb bench`` grades one crumb on self-invented axes (keyword density,
conciseness) and prints a letter grade. That is a vanity metric: it can only
be read by someone who already believes the format works.

Anyone actually evaluating a context format asks two questions:

  1. **How many tokens did this save?** — answered against a real tokenizer
     when ``tiktoken`` is installed, and against an explicitly-labelled
     character heuristic when it is not. The report always names which one
     produced the number, because a compression ratio computed from
     ``len(text) // 4`` is not a measurement.
  2. **What did it lose?** — answered by extracting the load-bearing tokens
     from the source (file paths, identifiers, numbers, URLs, error codes,
     quoted strings) and checking how many survive into the compressed form,
     broken down by category so you can see *what kind* of information the
     compressor drops.

Retention is a lexical proxy, not semantic fidelity, and the report says so.
A proxy you can reproduce beats a grade you cannot.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple


DEFAULT_ENCODING = "o200k_base"


# ── tokenisation ─────────────────────────────────────────────────────

def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> Tuple[int, str]:
    """Return ``(token_count, method)``.

    ``method`` is part of the result on purpose — every number this module
    prints carries the provenance of how it was counted.
    """
    try:
        import tiktoken  # type: ignore
    except ImportError:
        return len(text) // 4, "heuristic:chars/4"
    try:
        enc = tiktoken.get_encoding(encoding)
    except Exception:  # unknown encoding name, or no network for the BPE file
        return len(text) // 4, "heuristic:chars/4"
    return len(enc.encode(text)), f"tiktoken:{encoding}"


def tokenizer_is_exact(method: str) -> bool:
    return method.startswith("tiktoken:")


# ── load-bearing token extraction ────────────────────────────────────

FACT_PATTERNS: Dict[str, re.Pattern] = {
    # Ordered most-specific first; each token is claimed by one category only.
    "urls": re.compile(r"https?://[^\s<>\"')]+"),
    "paths": re.compile(
        r"\b[\w.-]+/[\w./-]+\.[A-Za-z]{1,6}\b"
        r"|\b[\w-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sql|ya?ml|json|toml|md|sh|css|html|ini|cfg)\b"
    ),
    "errors": re.compile(r"\b(?:[45]\d{2}|E[A-Z]{3,}|[A-Z][a-zA-Z]*(?:Error|Exception))\b"),
    "identifiers": re.compile(
        r"\b(?:[a-z][a-z0-9]*_[a-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b"
    ),
    # A bare one- or two-digit integer is almost always incidental prose
    # ("step 3", "the 2 cases"). Require a unit, a decimal, or 3+ digits so
    # the denominator counts facts a handoff would actually miss.
    "numbers": re.compile(
        r"\b\d+(?:\.\d+)?\s?(?:%|ms|kb|mb|gb|tb|px|req/s)\b"
        r"|\b\d+\.\d+\b"
        r"|\b\d{3,}\b",
        re.IGNORECASE,
    ),
    "quoted": re.compile(r"[\"'`]([^\"'`\n]{3,60})[\"'`]"),
}

# Words that match a pattern but carry no handoff signal.
_STOP_FACTS = {
    "1", "0", "2", "3", "http", "https", "readme.md", "index.js",
}


def extract_facts(text: str) -> Dict[str, set]:
    """Bucket the load-bearing tokens of ``text`` by category.

    A token is claimed by the first category that matches it, so the buckets
    partition the facts rather than double-counting a path as an identifier.
    """
    claimed: set = set()
    buckets: Dict[str, set] = {}
    for name, pattern in FACT_PATTERNS.items():
        found = set()
        for match in pattern.finditer(text):
            token = (match.group(1) if match.groups() else match.group(0)).strip()
            # Trailing sentence punctuation is never part of the fact. Left in,
            # a URL ending a sentence can never match the compressed form.
            token = token.rstrip(".,;:!?")
            key = token.lower()
            if not key or key in _STOP_FACTS or key in claimed:
                continue
            claimed.add(key)
            found.add(key)
        buckets[name] = found
    return buckets


def _normalise_haystack(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


# ── report ───────────────────────────────────────────────────────────

@dataclass
class CategoryResult:
    total: int = 0
    retained: int = 0
    lost: List[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return (self.retained / self.total) if self.total else 1.0


@dataclass
class MeasureResult:
    source_tokens: int = 0
    crumb_tokens: int = 0
    tokenizer: str = ""
    exact: bool = False
    categories: Dict[str, CategoryResult] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        return self.source_tokens / max(self.crumb_tokens, 1)

    @property
    def saved_pct(self) -> float:
        if not self.source_tokens:
            return 0.0
        return (self.source_tokens - self.crumb_tokens) / self.source_tokens * 100

    @property
    def facts_total(self) -> int:
        return sum(c.total for c in self.categories.values())

    @property
    def facts_retained(self) -> int:
        return sum(c.retained for c in self.categories.values())

    @property
    def retention(self) -> float:
        return (self.facts_retained / self.facts_total) if self.facts_total else 1.0

    def to_dict(self) -> Dict:
        return {
            "tokens": {
                "source": self.source_tokens,
                "crumb": self.crumb_tokens,
                "saved_pct": round(self.saved_pct, 1),
                "ratio": round(self.ratio, 2),
                "tokenizer": self.tokenizer,
                "exact": self.exact,
            },
            "retention": {
                "overall": round(self.retention, 4),
                "facts_total": self.facts_total,
                "facts_retained": self.facts_retained,
                "by_category": {
                    name: {
                        "total": c.total,
                        "retained": c.retained,
                        "rate": round(c.rate, 4),
                        "lost": sorted(c.lost)[:10],
                    }
                    for name, c in self.categories.items()
                },
            },
            "caveat": (
                "Retention is a lexical proxy: it checks whether load-bearing "
                "tokens from the source survive into the compressed form. It "
                "does not measure semantic fidelity or downstream task success."
            ),
        }


def measure(source_text: str, crumb_text: str, encoding: str = DEFAULT_ENCODING) -> MeasureResult:
    """Compare a source transcript against its compressed CRUMB."""
    source_tokens, method = count_tokens(source_text, encoding)
    crumb_tokens, _ = count_tokens(crumb_text, encoding)

    haystack = _normalise_haystack(crumb_text)
    categories: Dict[str, CategoryResult] = {}
    for name, facts in extract_facts(source_text).items():
        result = CategoryResult(total=len(facts))
        for fact in facts:
            if fact in haystack:
                result.retained += 1
            else:
                result.lost.append(fact)
        categories[name] = result

    return MeasureResult(
        source_tokens=source_tokens,
        crumb_tokens=crumb_tokens,
        tokenizer=method,
        exact=tokenizer_is_exact(method),
        categories=categories,
    )


def format_report(result: MeasureResult, *, source_label: str, crumb_label: str) -> str:
    """Human-readable report. Every number states how it was produced."""
    lines = [
        f"CRUMB measure — {crumb_label} vs {source_label}",
        "=" * 58,
        f"  Tokenizer:         {result.tokenizer}"
        + ("" if result.exact else "  (approximate — pip install tiktoken for exact counts)"),
        f"  Source:            {result.source_tokens:,} tokens",
        f"  CRUMB:             {result.crumb_tokens:,} tokens",
        f"  Saved:             {result.saved_pct:.1f}%  ({result.ratio:.1f}x smaller)",
        "-" * 58,
        f"  Fact retention:    {result.retention * 100:.1f}%  "
        f"({result.facts_retained}/{result.facts_total} load-bearing tokens kept)",
    ]
    for name, cat in result.categories.items():
        if not cat.total:
            continue
        lost = f"  lost: {', '.join(sorted(cat.lost)[:4])}" if cat.lost else ""
        lines.append(f"    {name:<14} {cat.rate * 100:5.1f}%  ({cat.retained}/{cat.total}){lost}")
    lines += [
        "=" * 58,
        "  Retention is a lexical proxy for fidelity, not a task-success",
        "  measure. It answers 'what got dropped', not 'does the next model",
        "  still succeed'. Treat it as a regression guard, not a benchmark.",
    ]
    return "\n".join(lines)


def format_json(result: MeasureResult, *, source_label: str, crumb_label: str) -> str:
    payload = {"source": source_label, "crumb": crumb_label}
    payload.update(result.to_dict())
    return json.dumps(payload, indent=2, sort_keys=False)


def check_thresholds(
    result: MeasureResult,
    *,
    min_saved_pct: float | None = None,
    min_retention: float | None = None,
) -> List[str]:
    """Return the list of threshold failures — empty means the gate passes."""
    failures: List[str] = []
    if min_saved_pct is not None and result.saved_pct < min_saved_pct:
        failures.append(
            f"token savings {result.saved_pct:.1f}% below required {min_saved_pct:.1f}%"
        )
    if min_retention is not None and result.retention * 100 < min_retention:
        failures.append(
            f"fact retention {result.retention * 100:.1f}% below required {min_retention:.1f}%"
        )
    return failures
