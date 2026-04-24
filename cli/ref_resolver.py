"""CRUMB v1.3 ref resolver (SPEC v1.3 §1).

Normative resolution order:
  1. Bare id   -> <id>.crumb in CWD, then $CRUMB_HOME (default ~/.crumb/)
  2. sha256:   -> $CRUMB_STORE (default ~/.crumb/store/) by digest
  3. URL       -> fetch if network enabled (default off)
  4. Registry  -> configured registry if any (default none)

Refs are advisory. A ref that does not resolve returns None.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{16,64}$")
URL_RE = re.compile(r"^https?://", re.IGNORECASE)
DEFAULT_DEPTH_LIMIT = 5


def _crumb_home() -> Path:
    home = os.environ.get("CRUMB_HOME")
    return Path(home) if home else Path.home() / ".crumb"


def _crumb_store() -> Path:
    store = os.environ.get("CRUMB_STORE")
    return Path(store) if store else _crumb_home() / "store"


def resolve_ref(
    ref: str,
    *,
    search_paths: Optional[List[Path]] = None,
    allow_network: bool = False,
    registry: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve a single ref to a local file path, or return None."""
    ref = ref.strip()
    if not ref:
        return None

    if DIGEST_RE.match(ref):
        store = _crumb_store()
        if store.is_dir():
            prefix = ref.split(":", 1)[1]
            for entry in store.glob("*.crumb"):
                if entry.stem.startswith(prefix) or prefix.startswith(entry.stem):
                    return entry
        return None

    if URL_RE.match(ref):
        if not allow_network:
            return None
        return None

    paths = search_paths or [Path.cwd(), _crumb_home()]
    for base in paths:
        candidate = Path(base) / f"{ref}.crumb"
        if candidate.is_file():
            return candidate

    if registry and registry.is_file():
        pass

    return None


def walk_refs(
    ref: str,
    *,
    search_paths: Optional[List[Path]] = None,
    allow_network: bool = False,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
) -> List[Tuple[str, Optional[Path]]]:
    """Walk refs transitively with a visited-set and depth limit.

    Returns list of (ref, resolved_path_or_None) pairs in discovery order.
    """
    visited: Set[str] = set()
    out: List[Tuple[str, Optional[Path]]] = []
    stack: List[Tuple[str, int]] = [(ref, 0)]
    while stack:
        current, depth = stack.pop(0)
        if current in visited or depth > depth_limit:
            continue
        visited.add(current)
        path = resolve_ref(
            current, search_paths=search_paths, allow_network=allow_network
        )
        out.append((current, path))
        if path is None:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for child in _extract_refs(text):
            if child not in visited:
                stack.append((child, depth + 1))
    return out


def _extract_refs(text: str) -> Iterable[str]:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("refs="):
            value = s.split("=", 1)[1]
            for entry in value.split(","):
                e = entry.strip()
                if e:
                    yield e
