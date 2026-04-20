"""Vowel-strip compression — Layer 4 of the MeTalk pipeline.

Strips interior vowels (a/e/i/o/u, case-insensitive) from words long enough
to remain recognizable from their consonant skeleton. Designed to slot in
*after* MeTalk's dictionary and grammar passes so that the most ambiguous
tech terms have already been canonicalized.

Two modes:
  - Deterministic (this module):  rule-based, no model, no network.
  - Adaptive (`adaptive_strip_text`): tries the strip on each token and
    keeps the change only if a per-line embedding cosine drift stays below
    a threshold. Requires sentence-transformers (the [embeddings] extra).

Encoded text carries a `vs=N` header where N is the minimum word length
that was eligible for stripping. `decode()` is best-effort — vowel removal
is intrinsically lossy.

Usage:
    from cli.vowelstrip import strip_text, encode_crumb, drift_stats

    slim = strip_text("authentication middleware function")
    # → "athntctn mddlwr fnctn"

    encoded = encode_crumb(crumb_text, min_length=4)
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

VOWELS = frozenset("aeiouAEIOU")

DEFAULT_MIN_LENGTH = 4

# Words we never strip even if long enough — collisions hurt comprehension
# more than the saved tokens are worth. Kept tiny on purpose; MeTalk's
# dictionary already covers most ambiguous tech terms.
PROTECTED_WORDS = frozenset({
    "true", "false", "none", "null", "undefined",
    "read", "lead", "feed", "need", "seed",   # confusable consonant skeletons
    "rate", "rote", "rite", "rude",
    "code", "node", "mode",
    "user", "users",
})

# Section names are short already and serve as parser anchors — never strip.
# The encoder skips lines wrapped in [ ... ] regardless, but the constant is
# documented here for readers.
_SECTION_LINE = re.compile(r"^\s*\[[^\]]+\]\s*$")
_FENCE_LINE = re.compile(r"^\s*```")
_TYPED_BLOCK = re.compile(r"^\s*@type\s*:\s*(code|diff|json|yaml)\b", re.IGNORECASE)
_WORD_RUN = re.compile(r"[A-Za-z]+")
# Whitespace-delimited token splitter that preserves separators so we can
# rejoin without normalizing spacing.
_WHITESPACE_SPLIT = re.compile(r"(\s+)")
# A token is treated as opaque (skip vowel strip entirely) if it contains
# any of these — covers URLs, file paths, identifiers, versions, emails.
_OPAQUE_CHARS = frozenset("/:_.@\\=#'")
# Punctuation safe to peel off the ends of a token before deciding whether
# the remaining core is opaque. Anything outside this set (slashes, @, etc.)
# is treated as part of the token's identity — peeling it would corrupt
# URLs, paths, handles, and identifiers.
_SENTENCE_PUNCT = frozenset('.,;:!?"\'()[]{}<>`')


def _should_strip(word: str, min_length: int) -> bool:
    if len(word) < min_length:
        return False
    if word.isupper():           # acronym
        return False
    if word.lower() in PROTECTED_WORDS:
        return False
    return True


def strip_word(word: str, min_length: int = DEFAULT_MIN_LENGTH) -> str:
    """Strip interior vowels from a single alphabetic word.

    First letter is preserved (keeps casing and word boundary recognition).
    Trailing 's' on plurals is preserved so that singular/plural pairs do
    not collapse onto the same skeleton (e.g. "user"/"users").
    """
    if not word.isalpha():
        return word
    if not _should_strip(word, min_length):
        return word

    first = word[0]
    body = word[1:]
    keep_trailing_s = body.endswith(("s", "S")) and len(body) > 1
    core = body[:-1] if keep_trailing_s else body

    stripped_core = "".join(ch for ch in core if ch not in VOWELS)
    result = first + stripped_core + (body[-1] if keep_trailing_s else "")

    # If everything collapsed to just the first letter, keep one vowel for
    # readability. e.g. "aeiou" → "a" would be useless; emit "ae" instead.
    if len(result) <= 1 and len(word) >= min_length:
        return word[0] + word[1]
    return result


def _is_opaque_token(tok: str) -> bool:
    """Tokens with structural punctuation pass through untouched.

    URLs, snake_case identifiers, file paths, version strings, emails — any
    token containing /, :, _, ., @, \\, =, or # is left alone in full because
    splitting it into alpha runs would corrupt its semantics.
    """
    return any(c in _OPAQUE_CHARS for c in tok)


def _strip_token(tok: str, min_length: int) -> str:
    """Process one whitespace-delimited token: peel surrounding punctuation,
    skip if the core looks opaque, otherwise vowel-strip alpha runs."""
    leading: list[str] = []
    while tok and tok[0] in _SENTENCE_PUNCT:
        leading.append(tok[0])
        tok = tok[1:]
    trailing: list[str] = []
    while tok and tok[-1] in _SENTENCE_PUNCT:
        trailing.append(tok[-1])
        tok = tok[:-1]
    if not tok:
        return "".join(leading) + "".join(reversed(trailing))
    if _is_opaque_token(tok):
        core = tok
    else:
        core = _WORD_RUN.sub(lambda m: strip_word(m.group(0), min_length), tok)
    return "".join(leading) + core + "".join(reversed(trailing))


def strip_line(line: str, min_length: int = DEFAULT_MIN_LENGTH) -> str:
    """Strip vowels from every alphabetic word in a single line.

    Trailing/leading punctuation is preserved, opaque tokens (URLs,
    snake_case, file paths, contractions) pass through intact.
    """
    parts = _WHITESPACE_SPLIT.split(line)
    out: list[str] = []
    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue
        out.append(_strip_token(part, min_length))
    return "".join(out)


def strip_text(text: str, min_length: int = DEFAULT_MIN_LENGTH) -> str:
    """Strip vowels from raw prose text. Operates line by line."""
    return "\n".join(strip_line(line, min_length) for line in text.split("\n"))


def _iter_body_lines(lines: list[str], sep_idx: int) -> Iterable[tuple[int, str]]:
    """Yield (index, line) pairs for body content (after the `---` separator)."""
    for i in range(sep_idx + 1, len(lines)):
        yield i, lines[i]


def encode_crumb(text: str, min_length: int = DEFAULT_MIN_LENGTH,
                 transform: Callable[[str], str] | None = None) -> str:
    """Vowel-strip a CRUMB body, leaving headers/sections/code blocks intact.

    `transform` lets callers swap in `adaptive_strip_text` (or any other
    line transformer) without duplicating the structural skip logic.
    """
    if transform is None:
        transform = lambda line: strip_line(line, min_length)

    lines = text.rstrip("\n").split("\n")
    sep_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "---":
            sep_idx = i
            break

    if sep_idx is None:
        return text  # not a structured crumb — leave alone

    in_fence = False
    in_typed_code = False
    out: list[str] = list(lines[: sep_idx + 1])

    for _, line in _iter_body_lines(lines, sep_idx):
        # Section header — emit as-is, also resets typed-code state.
        if _SECTION_LINE.match(line):
            in_typed_code = False
            out.append(line)
            continue

        # Fenced code block toggling.
        if _FENCE_LINE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue

        # v1.2 typed content annotation — switch off stripping for this section.
        m = _TYPED_BLOCK.match(line)
        if m:
            kind = m.group(1).lower()
            in_typed_code = kind in {"code", "diff", "json", "yaml"}
            out.append(line)
            continue

        # END CRUMB / EC sentinels — pass through.
        if line.strip() in {"END CRUMB", "EC"}:
            out.append(line)
            continue

        if in_fence or in_typed_code:
            out.append(line)
            continue

        out.append(transform(line))

    encoded = "\n".join(out) + "\n"
    return _inject_vs_header(encoded, min_length)


def _inject_vs_header(encoded: str, min_length: int) -> str:
    """Insert `vs=N` into the header block if not already present."""
    lines = encoded.split("\n")
    sep_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "---":
            sep_idx = i
            break
    if sep_idx is None:
        return encoded
    for i in range(1, sep_idx):
        if lines[i].strip().startswith("vs="):
            lines[i] = f"vs={min_length}"
            return "\n".join(lines)
    lines.insert(sep_idx, f"vs={min_length}")
    return "\n".join(lines)


# ── Adaptive mode (optional, requires sentence-transformers) ─────────

def _load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-import sentence-transformers. Returns None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        return None
    return SentenceTransformer(model_name)


def _cosine(a, b) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def adaptive_strip_text(text: str, *, threshold: float = 0.85,
                        min_length: int = DEFAULT_MIN_LENGTH,
                        embedder=None) -> str:
    """Strip vowels per line only if cosine drift stays below the threshold.

    Falls back to deterministic strip if no embedder is available — callers
    that need a guarantee should check `_load_embedder()` themselves.
    """
    if embedder is None:
        embedder = _load_embedder()
    if embedder is None:
        return strip_text(text, min_length)

    out: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            out.append(line)
            continue
        candidate = strip_line(line, min_length)
        if candidate == line:
            out.append(line)
            continue
        try:
            emb_orig, emb_new = embedder.encode([line, candidate])
            sim = _cosine(emb_orig, emb_new)
        except Exception:
            out.append(line)
            continue
        out.append(candidate if sim >= threshold else line)
    return "\n".join(out)


# ── Stats helpers ────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def drift_stats(original: str, encoded: str) -> dict:
    """Lexical drift metrics — char count, token estimate, vowel ratio."""
    orig_tokens = _estimate_tokens(original)
    enc_tokens = _estimate_tokens(encoded)
    orig_vowels = sum(1 for c in original if c in VOWELS)
    enc_vowels = sum(1 for c in encoded if c in VOWELS)
    return {
        "original_chars": len(original),
        "encoded_chars": len(encoded),
        "original_tokens": orig_tokens,
        "encoded_tokens": enc_tokens,
        "saved_tokens": orig_tokens - enc_tokens,
        "pct_saved": round(((orig_tokens - enc_tokens) / orig_tokens) * 100, 1)
            if orig_tokens else 0.0,
        "vowels_removed": orig_vowels - enc_vowels,
        "vowel_retention_pct": round((enc_vowels / orig_vowels) * 100, 1)
            if orig_vowels else 0.0,
    }
