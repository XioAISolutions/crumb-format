"""Read CRUMB packs (directories of CRUMB files produced by Crumb-Bob).

A "pack" is just a directory containing one or more ``.crumb`` files, plus an
optional ``manifest.json`` / ``pack.json`` describing the session. CrumbLLM
reads packs but has zero knowledge of how Crumb-Bob captures or names them —
no Bob-specific logic lives here.
"""

from __future__ import annotations

import json
from pathlib import Path

from crumb_llm.crumb.loader import load_crumb
from crumb_llm.models import CrumbPack

_MANIFEST_NAMES = ("manifest.json", "pack.json", "crumb-pack.json")


def _read_manifest(root: Path) -> dict | None:
    for name in _MANIFEST_NAMES:
        candidate = root / name
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return None
    return None


def read_pack(pack_dir: str | Path) -> CrumbPack:
    """Load every ``.crumb`` file under ``pack_dir`` into a :class:`CrumbPack`.

    Files that fail to parse are skipped silently here; surfacing per-file
    validation belongs to crumb-format's linter, not CrumbLLM.
    """
    root = Path(pack_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"not a pack directory: {root}")

    docs = []
    for crumb_path in sorted(root.rglob("*.crumb")):
        try:
            docs.append(load_crumb(crumb_path))
        except Exception:
            # Malformed members are skipped; validation is crumb-format's job.
            continue

    return CrumbPack(root=str(root), docs=docs, manifest=_read_manifest(root))
