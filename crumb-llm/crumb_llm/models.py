"""Core data models for CrumbLLM.

These are intentionally lightweight, dependency-free dataclasses. They model
*what CrumbLLM reads* (CRUMB docs and packs) and *what CrumbLLM produces*
(analysis results with quality warnings).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Matches path-like tokens such as ``src/app/main.py``, ``./lib/x.ts``,
# ``a/b/c.json``. Used by the hallucination check to compare model output
# against the file paths that actually appear in the source CRUMB.
_PATH_RE = re.compile(
    r"(?:\.{0,2}/)?(?:[\w.\-]+/)+[\w.\-]+\.[A-Za-z0-9]{1,8}"
)


def extract_file_paths(text: str) -> set[str]:
    """Extract file-path-like tokens from arbitrary text.

    This is deliberately conservative: it only matches things that look like
    ``dir/file.ext`` so the hallucination check has low false-positive noise.
    """
    found: set[str] = set()
    for match in _PATH_RE.findall(text or ""):
        cleaned = match.strip("`\"'()[],")
        if cleaned:
            found.add(cleaned)
    return found


@dataclass
class CrumbDoc:
    """A single parsed CRUMB file.

    ``headers`` and ``sections`` come straight from the crumb-format parser.
    We never re-derive them here.
    """

    path: str | None
    headers: dict[str, str]
    sections: dict[str, list[str]]
    raw_text: str

    @property
    def kind(self) -> str:
        return self.headers.get("kind", "unknown")

    @property
    def title(self) -> str:
        return self.headers.get("title", "") or self.headers.get("project", "")

    def section_text(self, name: str) -> str:
        return "\n".join(self.sections.get(name, [])).strip()

    def as_text(self) -> str:
        """Return the raw CRUMB text, used as model context."""
        return self.raw_text

    def file_paths(self) -> set[str]:
        """Every file-path-like token referenced anywhere in this CRUMB."""
        return extract_file_paths(self.raw_text)


@dataclass
class CrumbPack:
    """A CRUMB pack: a directory of CRUMB files plus an optional manifest.

    Packs are produced by Crumb-Bob (IBM Bob session capture). CrumbLLM only
    *reads* them — it has no knowledge of how they are generated.
    """

    root: str
    docs: list[CrumbDoc] = field(default_factory=list)
    manifest: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.docs)

    def file_paths(self) -> set[str]:
        paths: set[str] = set()
        for doc in self.docs:
            paths |= doc.file_paths()
        return paths

    def as_text(self, max_docs: int | None = None) -> str:
        """Concatenate the pack's CRUMB files into a single context blob."""
        parts: list[str] = []
        if self.manifest:
            parts.append("# PACK MANIFEST\n" + _safe_json(self.manifest))
        docs = self.docs if max_docs is None else self.docs[:max_docs]
        for doc in docs:
            name = Path(doc.path).name if doc.path else "(memory)"
            parts.append(f"# CRUMB: {name}\n{doc.raw_text}")
        return "\n\n".join(parts)


@dataclass
class ProviderConfig:
    """Resolved provider settings.

    Secrets are NEVER stored here when persisted to disk. API keys are always
    read from the environment at call time (see ``config.py``).
    """

    provider: str  # openai | anthropic | ollama | lmstudio | scratch | mock
    model: str = ""
    base_url: str = ""
    backend: str = ""  # for "local": ollama | lmstudio

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "backend": self.backend,
        }


@dataclass
class AnalysisResult:
    """The output of any CrumbLLM analysis task.

    ``warnings`` is the contract that enforces "never silently trust model
    output". Quality gates append to it; callers and the CLI surface it.
    """

    task: str
    text: str = ""
    data: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    source_paths: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no quality gate raised a warning."""
        return not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "text": self.text,
            "data": self.data,
            "warnings": self.warnings,
            "provider": self.provider,
            "model": self.model,
            "source_paths": self.source_paths,
        }


def _safe_json(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)
