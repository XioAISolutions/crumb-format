"""Delta crumbs — carry only what changed between two crumbs.

A delta crumb is analogous to a residual after PolarQuant's rotation:
instead of resending the full body each session, we carry only the
additions, removals, and modifications relative to a base crumb identified
by its content-addressed hash.

Wire format
-----------
::

    BEGIN CRUMB
    v=1.2
    kind=delta
    base=sha256:abc123...
    source=...
    ---
    [changes]
    - +[context] added line
    - -[context] removed line
    - ~[constraints] new value :: replaces :: old value

Each entry starts with ``+`` (added), ``-`` (removed), or ``~`` (changed).
``~`` entries use ``new :: replaces :: old`` so the delta is reversible.
"""

from __future__ import annotations

import difflib
import importlib
from dataclasses import dataclass
from typing import Dict, List, Tuple


CHANGED_SEP = " :: replaces :: "

# Headers that never travel in a delta — they are computed or refer to the base.
EXCLUDED_HEADER_DIFF = frozenset({"v", "kind", "base", "target", "id"})


def _crumb():
    return importlib.import_module("cli.crumb")


@dataclass(frozen=True)
class Change:
    op: str  # "+", "-", or "~"
    section: str
    text: str
    previous: str | None = None  # only set for op="~"


def _section_entries(sections: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Return a copy of sections with blank lines preserved by index."""
    return {name: list(body) for name, body in sections.items()}


def _meaningful_lines(body: List[str]) -> List[str]:
    return [line for line in body if line.strip()]


def compute_changes(
    base_sections: Dict[str, List[str]],
    target_sections: Dict[str, List[str]],
    base_headers: Dict[str, str] | None = None,
    target_headers: Dict[str, str] | None = None,
) -> List[Change]:
    """Compute a flat change list from base → target.

    Uses difflib opcodes over each section so the diff preserves position.
    A 1:1 ``replace`` opcode becomes a ``~`` entry carrying both sides;
    multi-line replacements expand into `-` followed by `+` entries.
    Header differences are encoded as operations on the pseudo-section
    ``@headers`` (entries formatted as ``key=value``).
    """
    crumb = _crumb()
    changes: List[Change] = []

    if base_headers is not None or target_headers is not None:
        base_headers = base_headers or {}
        target_headers = target_headers or {}
        keys = [
            key for key in dict.fromkeys(list(base_headers) + list(target_headers))
            if key not in EXCLUDED_HEADER_DIFF
        ]
        for key in keys:
            old = base_headers.get(key)
            new = target_headers.get(key)
            if old == new:
                continue
            if old is None:
                changes.append(Change("+", crumb.DELTA_HEADERS_SECTION, f"{key}={new}"))
            elif new is None:
                changes.append(Change("-", crumb.DELTA_HEADERS_SECTION, f"{key}={old}"))
            else:
                changes.append(
                    Change(
                        "~",
                        crumb.DELTA_HEADERS_SECTION,
                        f"{key}={new}",
                        previous=f"{key}={old}",
                    )
                )

    all_sections = list(dict.fromkeys(list(base_sections) + list(target_sections)))
    for name in all_sections:
        base = _meaningful_lines(base_sections.get(name, []))
        target = _meaningful_lines(target_sections.get(name, []))
        matcher = difflib.SequenceMatcher(a=base, b=target, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            removed_slice = base[i1:i2]
            added_slice = target[j1:j2]
            if tag == "replace" and len(removed_slice) == len(added_slice):
                for rem, add in zip(removed_slice, added_slice):
                    changes.append(Change("~", name, add, previous=rem))
                continue
            for rem in removed_slice:
                changes.append(Change("-", name, rem))
            for add in added_slice:
                changes.append(Change("+", name, add))
    return changes


def _bullet_prefix(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("- "):
        return None
    body = stripped[2:]
    head, sep, _ = body.partition(":")
    if sep:
        return head.strip().lower()
    return None


def changes_to_lines(changes: List[Change]) -> List[str]:
    lines: List[str] = []
    for change in changes:
        body = change.text.strip()
        if change.op == "~" and change.previous is not None:
            prev = change.previous.strip()
            lines.append(f"- ~[{change.section}] {body}{CHANGED_SEP}{prev}")
        else:
            lines.append(f"- {change.op}[{change.section}] {body}")
    return lines


def parse_changes(body: List[str]) -> List[Change]:
    crumb = _crumb()
    result: List[Change] = []
    for raw in body:
        if not raw.strip() or raw.strip().startswith("@"):
            continue
        match = crumb.DELTA_CHANGE_RE.match(raw)
        if not match:
            raise ValueError(f"malformed delta change line: {raw!r}")
        op, section, remainder = match.group(1), match.group(2).lower(), match.group(3).strip()
        if op == "~":
            if CHANGED_SEP not in remainder:
                raise ValueError(
                    f"~[{section}] change entry missing '{CHANGED_SEP.strip()}' separator"
                )
            new_text, _, prev_text = remainder.partition(CHANGED_SEP)
            result.append(Change("~", section, new_text.strip(), previous=prev_text.strip()))
        else:
            result.append(Change(op, section, remainder))
    return result


def build_delta_crumb(
    base_text: str,
    target_text: str,
    source: str = "crumb.delta",
    title: str | None = None,
) -> str:
    """Return a kind=delta crumb encoding base → target."""
    crumb = _crumb()
    hashing = importlib.import_module("cli.hashing")
    base_parsed = crumb.parse_crumb(base_text)
    target_parsed = crumb.parse_crumb(target_text)
    changes = compute_changes(
        base_parsed["sections"],
        target_parsed["sections"],
        base_headers=base_parsed["headers"],
        target_headers=target_parsed["headers"],
    )
    if not changes:
        raise ValueError("base and target crumbs have no content differences")

    base_digest = hashing.content_hash(base_text)
    target_digest = hashing.content_hash(target_text)
    headers = {
        "v": "1.2",
        "kind": "delta",
        "source": source,
        "base": base_digest,
        "target": target_digest,
        "title": title or f"delta {base_digest[:14]}… → {target_digest[:14]}…",
    }
    sections = {"changes": changes_to_lines(changes) + [""]}
    return crumb.render_crumb(headers, sections)


def apply_delta(base_text: str, delta_text: str, verify: bool = True) -> str:
    """Apply a delta crumb to a base crumb, returning the reconstructed target."""
    crumb = _crumb()
    hashing = importlib.import_module("cli.hashing")
    base_parsed = crumb.parse_crumb(base_text)
    delta_parsed = crumb.parse_crumb(delta_text)
    if delta_parsed["headers"].get("kind") != "delta":
        raise ValueError("delta file is not kind=delta")

    if verify:
        base_digest = hashing.content_hash(base_text)
        declared = delta_parsed["headers"].get("base", "")
        if not _digests_match(base_digest, declared):
            raise ValueError(
                f"base digest mismatch: delta expects {declared}, actual {base_digest}"
            )

    changes = parse_changes(delta_parsed["sections"].get("changes", []))
    sections = _section_entries(base_parsed["sections"])
    rebuilt_headers = dict(base_parsed["headers"])

    for change in changes:
        if change.section == crumb.DELTA_HEADERS_SECTION:
            _apply_header_change(rebuilt_headers, change)
            continue
        body = sections.setdefault(change.section, [])
        if change.op == "+":
            _append_line(body, change.text)
        elif change.op == "-":
            _remove_line(body, change.text)
        elif change.op == "~":
            assert change.previous is not None
            _replace_line(body, change.previous, change.text)

    for name in list(sections):
        body = sections[name]
        if not any(line.strip() for line in body):
            sections.pop(name)

    rebuilt = crumb.render_crumb(rebuilt_headers, sections)
    crumb.parse_crumb(rebuilt)

    if verify:
        expected = delta_parsed["headers"].get("target", "")
        if expected:
            actual = hashing.content_hash(rebuilt)
            if not _digests_match(expected, actual):
                raise ValueError(
                    f"reconstructed target digest {actual} does not match delta target {expected}"
                )
    return rebuilt


def _digests_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if not (a.startswith("sha256:") and b.startswith("sha256:")):
        return a == b
    hex_a = a.split(":", 1)[1]
    hex_b = b.split(":", 1)[1]
    shortest = min(len(hex_a), len(hex_b))
    return hex_a[:shortest] == hex_b[:shortest]


def _append_line(body: List[str], line: str) -> None:
    candidate = line.strip()
    if not candidate:
        return
    if any(existing.strip() == candidate for existing in body):
        return
    while body and not body[-1].strip():
        body.pop()
    body.append(candidate)
    body.append("")


def _remove_line(body: List[str], line: str) -> None:
    target = line.strip()
    if not target:
        return
    for idx, existing in enumerate(body):
        if existing.strip() == target:
            body.pop(idx)
            return


def _replace_line(body: List[str], previous: str, new_line: str) -> None:
    prev = previous.strip()
    replacement = new_line.strip()
    for idx, existing in enumerate(body):
        if existing.strip() == prev:
            body[idx] = replacement
            return
    _append_line(body, new_line)


def _apply_header_change(headers: Dict[str, str], change: Change) -> None:
    text = change.text
    if change.op == "-":
        key = _split_header_entry(text)[0]
        headers.pop(key, None)
        return
    key, value = _split_header_entry(text)
    headers[key] = value


def _split_header_entry(entry: str) -> Tuple[str, str]:
    if "=" not in entry:
        raise ValueError(f"header change entry missing '=': {entry!r}")
    key, _, value = entry.partition("=")
    return key.strip(), value.strip()
