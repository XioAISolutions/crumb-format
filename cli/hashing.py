"""Content-addressed refs and seen-set registry for CRUMBs.

Gives every CRUMB a canonical `sha256:<hex>` digest so a sender can elide
content a receiver already holds — the handoff analog of KV-cache reuse.

Canonical form for hashing:
  - parse and re-render the crumb (normalizes whitespace and section order)
  - drop volatile headers (``id``, ``dream_pass``, ``dream_sessions``) and
    the ``refs`` list itself so that a crumb's identity is its body, not
    the network of pointers around it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

from cli import crumb


VOLATILE_HEADERS = frozenset({"id", "dream_pass", "dream_sessions", "refs"})
SEEN_FILE_ENV = "CRUMB_SEEN_FILE"
DEFAULT_SEEN_FILE = Path.home() / ".crumb" / "seen"


def canonical_form(text: str) -> str:
    """Return a normalized text form suitable for hashing."""
    parsed = crumb.parse_crumb(text)
    headers = {
        key: value
        for key, value in parsed["headers"].items()
        if key not in VOLATILE_HEADERS
    }
    sections = {name: list(body) for name, body in parsed["sections"].items()}
    return crumb.render_crumb(headers, sections)


def content_hash(text: str) -> str:
    """Return ``sha256:<hex>`` for a crumb's canonical form."""
    canonical = canonical_form(text)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def short_hash(digest: str, length: int = 16) -> str:
    """Shorten a ``sha256:<hex>`` string for display/refs."""
    if not digest.startswith("sha256:"):
        raise ValueError("expected sha256:<hex> digest")
    hex_part = digest.split(":", 1)[1]
    if len(hex_part) < length:
        return digest
    return f"sha256:{hex_part[:length]}"


def seen_file_path(override: str | os.PathLike | None = None) -> Path:
    if override:
        return Path(override).expanduser()
    env = os.environ.get(SEEN_FILE_ENV)
    if env:
        return Path(env).expanduser()
    return DEFAULT_SEEN_FILE


def load_seen(path: str | os.PathLike | None = None) -> set[str]:
    file_path = seen_file_path(path)
    if not file_path.exists():
        return set()
    entries: set[str] = set()
    for raw in file_path.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        entries.add(entry)
    return entries


def save_seen(entries: Iterable[str], path: str | os.PathLike | None = None) -> Path:
    file_path = seen_file_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = sorted({item.strip() for item in entries if item.strip()})
    file_path.write_text("\n".join(normalized) + ("\n" if normalized else ""), encoding="utf-8")
    return file_path


def add_seen(digests: Iterable[str], path: str | os.PathLike | None = None) -> set[str]:
    existing = load_seen(path)
    for digest in digests:
        entry = digest.strip()
        if not entry:
            continue
        if not entry.startswith("sha256:"):
            raise ValueError(f"seen entries must be sha256:<hex>, got {entry!r}")
        existing.add(entry)
    save_seen(existing, path)
    return existing


def remove_seen(digests: Iterable[str], path: str | os.PathLike | None = None) -> set[str]:
    existing = load_seen(path)
    for digest in digests:
        existing.discard(digest.strip())
    save_seen(existing, path)
    return existing


def clear_seen(path: str | os.PathLike | None = None) -> Path:
    file_path = seen_file_path(path)
    if file_path.exists():
        file_path.write_text("", encoding="utf-8")
    return file_path


def digest_matches_set(digest: str, seen: set[str]) -> bool:
    """Return True if ``digest`` matches any entry in ``seen`` by sha256 prefix.

    Either side may be a prefix of the other so short-form digests can
    satisfy long-form entries and vice versa.
    """
    if not digest.startswith("sha256:"):
        return False
    if digest in seen:
        return True
    hex_part = digest.split(":", 1)[1]
    if not hex_part:
        return False
    for entry in seen:
        if not entry.startswith("sha256:"):
            continue
        entry_hex = entry.split(":", 1)[1]
        if not entry_hex:
            continue
        if hex_part.startswith(entry_hex) or entry_hex.startswith(hex_part):
            return True
    return False


def is_seen(digest: str, path: str | os.PathLike | None = None) -> bool:
    return digest_matches_set(digest, load_seen(path))
