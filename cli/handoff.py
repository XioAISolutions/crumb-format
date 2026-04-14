"""Multi-agent handoff helpers for CRUMB.

This module is the glue layer for building agent pipelines on top of CRUMB.
It does **not** add new headers or new section types — it only composes
``v=1.1`` crumbs using the upstream-pointer convention documented in
``docs/AGENT_HANDOFFS.md``.

Public API (stable at 0.6.0):

    emit_task(title, goal, context, constraints, source, ...)   -> dict
    emit_mem(title, consolidated, source, ...)                  -> dict
    walk_chain(crumbs_by_id, leaf_id, namespace=...)            -> list[dict]
    validate_chain(crumbs_by_id, leaf_id, namespace=...)        -> None

Every returned dict is shaped like ``parse_crumb`` output:
``{"id": ..., "text": ..., "headers": {...}, "sections": {...}}`` — so callers
can render the crumb with ``crumb.render_crumb`` or ``parse_crumb`` round-trips.
"""

from __future__ import annotations

import uuid
from typing import Iterable, Sequence

from cli import crumb as crumb_mod
from cli.extensions import append_extension

__all__ = [
    "emit_task",
    "emit_mem",
    "walk_chain",
    "validate_chain",
    "new_id",
    "ChainError",
]


class ChainError(ValueError):
    """Raised when a chain of crumbs does not form a valid upstream pointer graph."""


def new_id(prefix: str = "crumb") -> str:
    """Return a short, human-readable id suitable for an upstream pointer."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _coerce_lines(value: str | Iterable[str]) -> list[str]:
    """Normalize body input to the list-of-lines shape the renderer expects."""
    if isinstance(value, str):
        return value.splitlines() or [""]
    return [str(line) for line in value]


def _base_headers(
    *,
    kind: str,
    title: str,
    source: str,
    crumb_id: str | None,
    project: str | None,
    tags: Sequence[str] | None,
    extra_headers: dict[str, str] | None,
) -> dict[str, str]:
    headers: dict[str, str] = {
        "v": "1.1",
        "kind": kind,
        "title": title,
        "source": source,
    }
    headers["id"] = crumb_id or new_id(kind)
    if project:
        headers["project"] = project
    if tags:
        headers["tags"] = ", ".join(t.strip() for t in tags if t.strip())
    if extra_headers:
        for key, value in extra_headers.items():
            headers[key] = value
    return headers


def _apply_upstream(
    headers: dict[str, str],
    *,
    upstream: str | None,
    namespace: str,
) -> None:
    """Inject the ``ext.<namespace>.upstream=`` pointer header + extension name."""
    if not upstream:
        return
    ext_name = f"ext.{namespace}.upstream.v1"
    append_extension(headers, ext_name)
    headers[f"ext.{namespace}.upstream"] = upstream


def _apply_chain(
    headers: dict[str, str],
    *,
    chain: Sequence[str] | None,
    namespace: str,
) -> None:
    """Inject the optional ``ext.<namespace>.chain=`` roster header."""
    if not chain:
        return
    ext_name = f"ext.{namespace}.chain.v1"
    append_extension(headers, ext_name)
    headers[f"ext.{namespace}.chain"] = ",".join(chain)


def _materialize(headers: dict[str, str], sections: dict[str, list[str]]) -> dict:
    """Render the crumb, parse it back to validate, return {id, text, headers, sections}."""
    text = crumb_mod.render_crumb(headers, sections)
    parsed = crumb_mod.parse_crumb(text)
    return {
        "id": parsed["headers"].get("id", ""),
        "text": text,
        "headers": parsed["headers"],
        "sections": parsed["sections"],
    }


# ---------------------------------------------------------------------------
# emit_task / emit_mem
# ---------------------------------------------------------------------------


def emit_task(
    *,
    title: str,
    goal: str | Iterable[str],
    context: str | Iterable[str],
    constraints: str | Iterable[str],
    source: str,
    crumb_id: str | None = None,
    project: str | None = None,
    tags: Sequence[str] | None = None,
    upstream: str | None = None,
    chain: Sequence[str] | None = None,
    namespace: str = "crumb",
    extra_headers: dict[str, str] | None = None,
    extra_sections: dict[str, str | Iterable[str]] | None = None,
) -> dict:
    """Build a ``kind=task`` handoff crumb.

    ``upstream`` is the ``id=`` of the crumb this one replies to (``None`` for
    the root). ``namespace`` controls the extension prefix so a pipeline can
    distinguish its own chain from a nested debate.
    """
    headers = _base_headers(
        kind="task",
        title=title,
        source=source,
        crumb_id=crumb_id,
        project=project,
        tags=tags,
        extra_headers=extra_headers,
    )
    _apply_upstream(headers, upstream=upstream, namespace=namespace)
    _apply_chain(headers, chain=chain, namespace=namespace)

    sections: dict[str, list[str]] = {
        "goal": _coerce_lines(goal),
        "context": _coerce_lines(context),
        "constraints": _coerce_lines(constraints),
    }
    if extra_sections:
        for name, body in extra_sections.items():
            sections[name] = _coerce_lines(body)

    return _materialize(headers, sections)


def emit_mem(
    *,
    title: str,
    consolidated: str | Iterable[str],
    source: str,
    crumb_id: str | None = None,
    project: str | None = None,
    tags: Sequence[str] | None = None,
    upstream: str | None = None,
    chain: Sequence[str] | None = None,
    namespace: str = "crumb",
    extra_headers: dict[str, str] | None = None,
    extra_sections: dict[str, str | Iterable[str]] | None = None,
) -> dict:
    """Build a ``kind=mem`` output/answer crumb.

    Use for durable output, retrieval results, and reducer syntheses — anything
    that is data, not a new request.
    """
    headers = _base_headers(
        kind="mem",
        title=title,
        source=source,
        crumb_id=crumb_id,
        project=project,
        tags=tags,
        extra_headers=extra_headers,
    )
    _apply_upstream(headers, upstream=upstream, namespace=namespace)
    _apply_chain(headers, chain=chain, namespace=namespace)

    sections: dict[str, list[str]] = {
        "consolidated": _coerce_lines(consolidated),
    }
    if extra_sections:
        for name, body in extra_sections.items():
            sections[name] = _coerce_lines(body)

    return _materialize(headers, sections)


# ---------------------------------------------------------------------------
# Chain walking and validation
# ---------------------------------------------------------------------------


def walk_chain(
    crumbs_by_id: dict[str, dict],
    leaf_id: str,
    *,
    namespace: str = "crumb",
) -> list[dict]:
    """Walk upstream pointers from ``leaf_id`` back to the root.

    Returns the chain in **root-first** order. Each entry is an
    ``emit_task`` / ``emit_mem`` style dict (or anything with a matching shape
    — i.e. ``{"headers": {...}}``).
    """
    pointer_key = f"ext.{namespace}.upstream"
    chain: list[dict] = []
    current_id: str | None = leaf_id
    seen: set[str] = set()

    while current_id is not None:
        if current_id in seen:
            raise ChainError(f"upstream cycle detected at id={current_id!r}")
        seen.add(current_id)
        if current_id not in crumbs_by_id:
            raise ChainError(f"missing crumb in chain: id={current_id!r}")
        node = crumbs_by_id[current_id]
        chain.append(node)
        current_id = node["headers"].get(pointer_key)

    chain.reverse()
    return chain


def validate_chain(
    crumbs_by_id: dict[str, dict],
    leaf_id: str,
    *,
    namespace: str = "crumb",
    expected_final_kind: str | None = None,
) -> list[dict]:
    """Walk the chain and enforce protocol rules.

    Protocol rules enforced:
      1. Every upstream pointer resolves to a crumb in ``crumbs_by_id``.
      2. No upstream cycles.
      3. Non-root crumbs MUST carry the ``ext.<namespace>.upstream.v1``
         declaration in their ``extensions=`` header.
      4. If ``expected_final_kind`` is given, the leaf's kind MUST match.

    Returns the validated chain (root-first). Raises ``ChainError`` on failure.
    """
    chain = walk_chain(crumbs_by_id, leaf_id, namespace=namespace)

    expected_ext = f"ext.{namespace}.upstream.v1"
    for node in chain[1:]:  # every non-root
        extensions = node["headers"].get("extensions", "")
        declared = {item.strip() for item in extensions.split(",") if item.strip()}
        if expected_ext not in declared:
            raise ChainError(
                f"crumb id={node['headers'].get('id')!r} has an upstream "
                f"pointer but does not declare {expected_ext!r} in extensions="
            )

    if expected_final_kind is not None:
        leaf_kind = chain[-1]["headers"].get("kind")
        if leaf_kind != expected_final_kind:
            raise ChainError(
                f"chain leaf kind={leaf_kind!r}, expected {expected_final_kind!r}"
            )

    return chain
