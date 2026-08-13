"""The CRUMB reference parser — the normative implementation of the wire format.

This package exists because CRUMB was three formats pretending to be one. The
spec called itself a standard while three repositories each carried a private
parser with no shared code and no conformance tests between them:

* ``crumb-format`` emitted ``v=1.4``
* ``CrumbContext`` hand-wrote ``BEGIN CRUMB\nv=1.3`` as a string literal
* ``CrumbLLM`` forked a 334-line copy of the validation logic, and ran a CI job
  asserting it could *not* import ``crumb-format``
* ``CrumbLLM``'s EJA schemas required ``crumb_version: "1.5"`` — a version no
  spec defines

Nothing detected the drift, because nothing compared them. A format whose own
ecosystem cannot agree on the current version number is not a standard; it is a
convention with variants.

``crumb_core`` is that single source of truth. It is deliberately small and
dependency-free — stdlib only, no CLI, no I/O — so that any project can depend
on it without inheriting the ~46-command CLI, the MCP servers, or a PyTorch
language model. ``cli.crumb`` re-exports every name here, so existing callers
of ``from cli.crumb import parse_crumb`` keep working unchanged.

Everything in this module is normative with respect to SPEC.md. Changing
behaviour here changes the format for every consumer, so changes belong with a
spec amendment and a conformance-suite case, not with a caller's convenience.
"""

from __future__ import annotations

import re
from typing import Dict, List

__all__ = [
    "CONTENT_REF_RE",
    "DELTA_CHANGE_RE",
    "DELTA_HEADERS_SECTION",
    "FOLD_SECTION_RE",
    "HANDOFF_ID_RE",
    "REQUIRED_HEADERS",
    "REQUIRED_SECTIONS",
    "SUPPORTED_VERSIONS",
    "WORKFLOW_LINE_RE",
    "parse_crumb",
    "render_crumb",
]

# The wire-format version this parser writes. Consumers that need to emit a
# CRUMB must use this rather than hardcoding a version string — hardcoding is
# how CrumbContext ended up emitting v1.3 against a v1.4 spec.
WIRE_VERSION = "1.4"


REQUIRED_HEADERS = ["v", "kind", "source"]
REQUIRED_SECTIONS = {
    "task": ["goal", "context", "constraints"],
    "mem": ["consolidated"],
    "map": ["project", "modules"],
    "log": ["entries"],
    "todo": ["tasks"],
    "passport": ["identity", "permissions"],
    "audit": ["goal", "actions", "verdict"],
    "wake": ["identity"],
    "delta": ["changes"],
    "agent": ["identity"],
}

SUPPORTED_VERSIONS = {"1.1", "1.2", "1.3", "1.4"}

FOLD_SECTION_RE = re.compile(r"^fold:([^/]+)/(summary|full)$")
CONTENT_REF_RE = re.compile(r"^sha256:[0-9a-f]{16,64}$")
DELTA_CHANGE_RE = re.compile(r"^\s*-\s*([+\-~])\[(@?[a-z0-9_:/-]+)\]\s*(.*)$", re.IGNORECASE)
DELTA_HEADERS_SECTION = "@headers"
HANDOFF_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
WORKFLOW_LINE_RE = re.compile(r"^\s*-?\s*(\d+)[.)]\s*(.+)$")


def parse_crumb(text: str) -> Dict[str, object]:
    """Parse a .crumb file and return headers and sections.

    Raises ValueError on any structural problem.
    """
    lines = [line.rstrip("\n") for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or lines[0] != "BEGIN CRUMB":
        raise ValueError("missing BEGIN CRUMB marker")
    if lines[-1] != "END CRUMB":
        raise ValueError("missing END CRUMB marker")
    try:
        sep_index = lines.index("---")
    except ValueError:
        raise ValueError("missing header separator ---")
    headers: Dict[str, str] = {}
    for line in lines[1:sep_index]:
        if not line.strip():
            continue
        if "=" not in line:
            raise ValueError(f"invalid header line: {line!r}")
        key, value = line.split("=", 1)
        headers[key.strip()] = value.strip()
    for key in REQUIRED_HEADERS:
        if key not in headers:
            raise ValueError(f"missing required header: {key}")
    if headers["v"] not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported version: {headers['v']}")
    kind = headers["kind"]
    if kind not in REQUIRED_SECTIONS:
        valid = ", ".join(sorted(REQUIRED_SECTIONS.keys()))
        raise ValueError(f"unknown kind: {kind!r}. valid: {valid}")
    sections: Dict[str, List[str]] = {}
    current_section: str | None = None
    for line in lines[sep_index + 1 : -1]:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip().lower()
            sections.setdefault(current_section, [])
            continue
        if current_section is None:
            if stripped:
                raise ValueError("body content found before first section")
            continue
        sections[current_section].append(line)
    for section in REQUIRED_SECTIONS[kind]:
        fold_summary = f"fold:{section}/summary"
        fold_full = f"fold:{section}/full"
        if section in sections:
            if not any(item.strip() for item in sections[section]):
                raise ValueError(f"section [{section}] is empty")
        elif fold_summary in sections or fold_full in sections:
            for variant_name in (fold_summary, fold_full):
                if variant_name in sections and not any(
                    item.strip() for item in sections[variant_name]
                ):
                    raise ValueError(f"section [{variant_name}] is empty")
        else:
            raise ValueError(
                f"missing required section for kind={kind}: [{section}]"
            )

    _validate_v12_additive(headers, sections)
    _validate_v13_additive(headers, sections)

    return {"headers": headers, "sections": sections}


def _validate_v12_additive(
    headers: Dict[str, str], sections: Dict[str, List[str]]
) -> None:
    """Additive v1.2 validation. Does not reject v1.1 files."""

    if "refs" in headers:
        refs_value = headers["refs"].strip()
        if not refs_value:
            raise ValueError("refs header must not be empty when present")
        for ref in (r.strip() for r in refs_value.split(",")):
            if not ref:
                raise ValueError("refs header contains an empty entry")
            if ref.startswith("sha256:") and not CONTENT_REF_RE.match(ref):
                raise ValueError(
                    f"refs entry {ref!r} has a malformed sha256: digest"
                )

    if "refs" in sections and not any(line.strip() for line in sections["refs"]):
        raise ValueError("[refs] section is empty; omit it instead")

    if "handoff" in sections and not any(
        line.strip() for line in sections["handoff"]
    ):
        raise ValueError("[handoff] section is empty; omit it instead")

    fold_pairs: Dict[str, set] = {}
    for section_name in sections:
        match = FOLD_SECTION_RE.match(section_name)
        if not match:
            continue
        fold_name, variant = match.group(1), match.group(2)
        fold_pairs.setdefault(fold_name, set()).add(variant)

    for fold_name, variants in fold_pairs.items():
        if "full" in variants and "summary" not in variants:
            raise ValueError(
                f"fold:{fold_name} declares /full without a paired /summary"
            )

    for section_name, body in sections.items():
        meaningful = [line for line in body if line.strip()]
        for idx, line in enumerate(meaningful[:2]):
            stripped = line.strip()
            if stripped.startswith("@type:") and idx == 0:
                content_type = stripped.split(":", 1)[1].strip()
                if not content_type:
                    raise ValueError(
                        f"@type annotation has empty value in [{section_name}]"
                    )
            if stripped.startswith("@priority:"):
                raw = stripped.split(":", 1)[1].strip()
                if not raw:
                    raise ValueError(
                        f"@priority annotation has empty value in [{section_name}]"
                    )
                try:
                    score = int(raw)
                except ValueError:
                    raise ValueError(
                        f"@priority value in [{section_name}] must be an integer 1-10"
                    )
                if not 1 <= score <= 10:
                    raise ValueError(
                        f"@priority value in [{section_name}] must be between 1 and 10"
                    )

    if headers.get("kind") == "delta":
        if "base" not in headers or not headers["base"].strip():
            raise ValueError("kind=delta requires a 'base' header identifying the parent crumb")
        changes = [line for line in sections.get("changes", []) if line.strip()]
        if not changes:
            raise ValueError("kind=delta requires at least one entry in [changes]")
        for line in changes:
            stripped = line.strip()
            if stripped.startswith("@"):
                continue
            if not DELTA_CHANGE_RE.match(line):
                raise ValueError(
                    f"malformed [changes] entry: {stripped!r} "
                    "(expected '- +[section] text', '- -[section] text', or '- ~[section] text')"
                )


def _parse_kv_line(line: str) -> Dict[str, str]:
    """Parse 'key=value  key=value' style trailing annotations on a bullet line."""
    tokens: Dict[str, str] = {}
    body = line.strip()
    if body.startswith("- "):
        body = body[2:]
    elif body.startswith("-"):
        body = body[1:]
    for match in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)", body):
        tokens[match.group(1)] = match.group(2)
    return tokens


def _validate_v13_additive(
    headers: Dict[str, str], sections: Dict[str, List[str]]
) -> None:
    """Additive v1.3 validation. Does not reject v1.1 or v1.2 files."""
    if "fold_priority" in headers:
        value = headers["fold_priority"].strip()
        if not value:
            raise ValueError("fold_priority header must not be empty when present")
        for name in (n.strip() for n in value.split(",")):
            if not name:
                raise ValueError("fold_priority contains an empty entry")
            if not re.match(r"^[a-zA-Z0-9_-]+$", name):
                raise ValueError(f"fold_priority entry {name!r} has invalid characters")

    if "handoff" in sections:
        step_ids: Dict[str, int] = {}
        deps: Dict[str, List[str]] = {}
        position = 0
        for line in sections["handoff"]:
            stripped = line.strip()
            if not stripped or stripped.startswith("- [x]"):
                continue
            if not stripped.startswith("-"):
                continue
            position += 1
            tokens = _parse_kv_line(stripped)
            step_id = tokens.get("id", str(position))
            if not HANDOFF_ID_RE.match(step_id):
                raise ValueError(
                    f"[handoff] id={step_id!r} must match [a-zA-Z0-9_-]+"
                )
            if step_id in step_ids:
                raise ValueError(f"[handoff] duplicate id={step_id!r}")
            step_ids[step_id] = position
            after = tokens.get("after", "")
            if after:
                deps[step_id] = [
                    d.strip() for d in after.split(",") if d.strip()
                ]
        for step_id, refs in deps.items():
            for ref in refs:
                if ref not in step_ids:
                    raise ValueError(
                        f"[handoff] id={step_id!r} has unknown after= dependency {ref!r}"
                    )
        _detect_dep_cycle(deps, label="[handoff]")

    if "workflow" in sections:
        step_ids: Dict[str, int] = {}
        deps: Dict[str, List[str]] = {}
        for line in sections["workflow"]:
            stripped = line.strip()
            if not stripped:
                continue
            match = WORKFLOW_LINE_RE.match(stripped)
            if not match:
                if stripped.startswith("-"):
                    continue
                raise ValueError(
                    f"[workflow] line must be numbered (e.g. '1. reproduce_bug'): {stripped!r}"
                )
            num, rest = match.group(1), match.group(2)
            tokens = _parse_kv_line("- " + rest)
            step_id = tokens.get("id", num)
            if not HANDOFF_ID_RE.match(step_id):
                raise ValueError(
                    f"[workflow] id={step_id!r} must match [a-zA-Z0-9_-]+"
                )
            if step_id in step_ids:
                raise ValueError(f"[workflow] duplicate id={step_id!r}")
            step_ids[step_id] = int(num)
            depends = tokens.get("depends_on", "")
            if depends:
                deps[step_id] = [
                    d.strip() for d in depends.split(",") if d.strip()
                ]
        for step_id, refs in deps.items():
            for ref in refs:
                if ref not in step_ids:
                    raise ValueError(
                        f"[workflow] id={step_id!r} has unknown depends_on {ref!r}"
                    )
        _detect_dep_cycle(deps, label="[workflow]")

    if "script" in sections:
        meaningful = [line for line in sections["script"] if line.strip()]
        if meaningful and not meaningful[0].strip().startswith("@type:"):
            raise ValueError("[script] section must begin with @type: <lang>")

    if "checks" in sections:
        for line in sections["checks"]:
            stripped = line.strip()
            if not stripped or not stripped.startswith("-"):
                continue
            body = stripped[1:].strip()
            if "::" not in body:
                raise ValueError(
                    f"[checks] line must use 'name :: status' format: {stripped!r}"
                )


def _detect_dep_cycle(deps: Dict[str, List[str]], label: str) -> None:
    """Raise if deps contains a cycle. Uses DFS coloring."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {k: WHITE for k in deps}

    def visit(node: str) -> None:
        if color.get(node, WHITE) == GRAY:
            raise ValueError(f"{label} dependency cycle through {node!r}")
        if color.get(node, WHITE) == BLACK:
            return
        color[node] = GRAY
        for child in deps.get(node, []):
            if child in deps:
                visit(child)
        color[node] = BLACK

    for node in list(deps):
        if color[node] == WHITE:
            visit(node)


def render_crumb(headers: Dict[str, str], sections: Dict[str, List[str]]) -> str:
    """Render headers and sections back into a .crumb file string."""
    lines = ["BEGIN CRUMB"]
    for key, value in headers.items():
        lines.append(f"{key}={value}")
    lines.append("---")
    for name, body in sections.items():
        lines.append(f"[{name}]")
        lines.extend(body)
        if body and body[-1].strip():
            lines.append("")
    lines.append("END CRUMB")
    return "\n".join(lines) + "\n"

