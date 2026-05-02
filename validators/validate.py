#!/usr/bin/env python3
from __future__ import annotations

import glob
import pathlib
import re
import sys
from typing import Dict, List

REQUIRED_HEADERS = ["v", "kind", "source"]
REQUIRED_SECTIONS = {
    "task": ["goal", "context", "constraints"],
    "mem": ["consolidated"],
    "map": ["project", "modules"],
    "log": ["entries"],
    "todo": ["tasks"],
    "wake": ["identity"],
    "delta": ["changes"],
    "agent": ["identity"],
}
SUPPORTED_VERSIONS = {"1.1", "1.2", "1.3", "1.4"}
FOLD_SECTION_RE = re.compile(r"^fold:([^/]+)/(summary|full)$")
CONTENT_REF_RE = re.compile(r"^sha256:[0-9a-f]{16,64}$")
DELTA_CHANGE_RE = re.compile(r"^\s*-\s*([+\-~])\[(@?[a-z0-9_:/-]+)\]\s*(.*)$", re.IGNORECASE)
HANDOFF_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
WORKFLOW_LINE_RE = re.compile(r"^\s*-?\s*(\d+)[.)]\s*(.+)$")
KV_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)")

class ValidationError(Exception):
    pass

def parse_crumb(text: str) -> Dict[str, object]:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or lines[0] != "BEGIN CRUMB":
        raise ValidationError("missing BEGIN CRUMB marker")
    if lines[-1] != "END CRUMB":
        raise ValidationError("missing END CRUMB marker")
    try:
        sep_index = lines.index("---")
    except ValueError as exc:
        raise ValidationError("missing header separator ---") from exc
    headers: Dict[str, str] = {}
    for line in lines[1:sep_index]:
        if not line.strip():
            continue
        if "=" not in line:
            raise ValidationError(f"invalid header line: {line!r}")
        key, value = line.split("=", 1)
        headers[key.strip()] = value.strip()
    for key in REQUIRED_HEADERS:
        if key not in headers:
            raise ValidationError(f"missing required header: {key}")
    if headers["v"] not in SUPPORTED_VERSIONS:
        raise ValidationError(f"unsupported version: {headers['v']}")
    kind = headers["kind"]
    if kind not in REQUIRED_SECTIONS:
        valid = ", ".join(sorted(REQUIRED_SECTIONS.keys()))
        raise ValidationError(f"unknown kind: {kind!r}. valid: {valid}")
    sections: Dict[str, List[str]] = {}
    current_section: str | None = None
    for line in lines[sep_index + 1 : -1]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip().lower()
            sections.setdefault(current_section, [])
            continue
        if current_section is None:
            raise ValidationError("body content found before first section")
        sections[current_section].append(stripped)
    for section in REQUIRED_SECTIONS[kind]:
        fold_summary = f"fold:{section}/summary"
        fold_full = f"fold:{section}/full"
        if section in sections:
            if not any(item.strip() for item in sections[section]):
                raise ValidationError(f"section is empty: {section}")
        elif fold_summary in sections or fold_full in sections:
            for variant in (fold_summary, fold_full):
                if variant in sections and not any(
                    item.strip() for item in sections[variant]
                ):
                    raise ValidationError(f"section is empty: {variant}")
        else:
            raise ValidationError(
                f"missing required section for kind={kind}: {section}"
            )
    _validate_v12_additive(headers, sections)
    _validate_v13_additive(headers, sections)
    return {"headers": headers, "sections": sections}


def _validate_v12_additive(
    headers: Dict[str, str], sections: Dict[str, List[str]]
) -> None:
    """Additive v1.2 checks. Never rejects v1.1 files."""
    if "refs" in headers:
        refs_value = headers["refs"].strip()
        if not refs_value:
            raise ValidationError("refs header must not be empty when present")
        for ref in (r.strip() for r in refs_value.split(",")):
            if not ref:
                raise ValidationError("refs header contains an empty entry")
            if ref.startswith("sha256:") and not CONTENT_REF_RE.match(ref):
                raise ValidationError(
                    f"refs entry {ref!r} has a malformed sha256: digest"
                )
    if "refs" in sections and not any(line.strip() for line in sections["refs"]):
        raise ValidationError("[refs] section is empty; omit it instead")
    if "handoff" in sections and not any(
        line.strip() for line in sections["handoff"]
    ):
        raise ValidationError("[handoff] section is empty; omit it instead")
    fold_pairs: Dict[str, set] = {}
    for name in sections:
        match = FOLD_SECTION_RE.match(name)
        if not match:
            continue
        fold_name, variant = match.group(1), match.group(2)
        fold_pairs.setdefault(fold_name, set()).add(variant)
    for fold_name, variants in fold_pairs.items():
        if "full" in variants and "summary" not in variants:
            raise ValidationError(
                f"fold:{fold_name} declares /full without a paired /summary"
            )
    for name, body in sections.items():
        meaningful = [line for line in body if line.strip()]
        for idx, line in enumerate(meaningful[:2]):
            stripped = line.strip()
            if stripped.startswith("@type:") and idx == 0:
                type_value = stripped.split(":", 1)[1].strip()
                if not type_value:
                    raise ValidationError(
                        f"@type annotation has empty value in [{name}]"
                    )
            if stripped.startswith("@priority:"):
                raw = stripped.split(":", 1)[1].strip()
                if not raw:
                    raise ValidationError(
                        f"@priority annotation has empty value in [{name}]"
                    )
                try:
                    score = int(raw)
                except ValueError as exc:
                    raise ValidationError(
                        f"@priority value in [{name}] must be an integer 1-10"
                    ) from exc
                if not 1 <= score <= 10:
                    raise ValidationError(
                        f"@priority value in [{name}] must be between 1 and 10"
                    )
    if headers.get("kind") == "delta":
        if "base" not in headers or not headers["base"].strip():
            raise ValidationError(
                "kind=delta requires a 'base' header identifying the parent crumb"
            )
        changes = [line for line in sections.get("changes", []) if line.strip()]
        if not changes:
            raise ValidationError("kind=delta requires at least one entry in [changes]")
        for line in changes:
            stripped = line.strip()
            if stripped.startswith("@"):
                continue
            if not DELTA_CHANGE_RE.match(line):
                raise ValidationError(
                    f"malformed [changes] entry: {stripped!r}"
                )


def _parse_kv_line(line: str) -> Dict[str, str]:
    body = line.strip()
    if body.startswith("- "):
        body = body[2:]
    elif body.startswith("-"):
        body = body[1:]
    return {m.group(1): m.group(2) for m in KV_RE.finditer(body)}


def _detect_dep_cycle(deps: Dict[str, List[str]], label: str) -> None:
    color: Dict[str, int] = {k: 0 for k in deps}

    def visit(node: str) -> None:
        c = color.get(node, 0)
        if c == 1:
            raise ValidationError(f"{label} dependency cycle through {node!r}")
        if c == 2:
            return
        color[node] = 1
        for child in deps.get(node, []):
            if child in deps:
                visit(child)
        color[node] = 2

    for node in list(deps):
        if color.get(node, 0) == 0:
            visit(node)


def _validate_v13_additive(
    headers: Dict[str, str], sections: Dict[str, List[str]]
) -> None:
    if "fold_priority" in headers:
        value = headers["fold_priority"].strip()
        if not value:
            raise ValidationError("fold_priority header must not be empty when present")
        for name in (n.strip() for n in value.split(",")):
            if not name:
                raise ValidationError("fold_priority contains an empty entry")
            if not re.match(r"^[a-zA-Z0-9_-]+$", name):
                raise ValidationError(f"fold_priority entry {name!r} has invalid characters")
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
                raise ValidationError(f"[handoff] id={step_id!r} must match [a-zA-Z0-9_-]+")
            if step_id in step_ids:
                raise ValidationError(f"[handoff] duplicate id={step_id!r}")
            step_ids[step_id] = position
            after = tokens.get("after", "")
            if after:
                deps[step_id] = [d.strip() for d in after.split(",") if d.strip()]
        for step_id, refs in deps.items():
            for ref in refs:
                if ref not in step_ids:
                    raise ValidationError(
                        f"[handoff] id={step_id!r} has unknown after= dependency {ref!r}"
                    )
        _detect_dep_cycle(deps, label="[handoff]")
    if "workflow" in sections:
        step_ids = {}
        deps = {}
        for line in sections["workflow"]:
            stripped = line.strip()
            if not stripped:
                continue
            match = WORKFLOW_LINE_RE.match(stripped)
            if not match:
                if stripped.startswith("-"):
                    continue
                raise ValidationError(
                    f"[workflow] line must be numbered: {stripped!r}"
                )
            num, rest = match.group(1), match.group(2)
            tokens = _parse_kv_line("- " + rest)
            step_id = tokens.get("id", num)
            if not HANDOFF_ID_RE.match(step_id):
                raise ValidationError(f"[workflow] id={step_id!r} must match [a-zA-Z0-9_-]+")
            if step_id in step_ids:
                raise ValidationError(f"[workflow] duplicate id={step_id!r}")
            step_ids[step_id] = int(num)
            depends = tokens.get("depends_on", "")
            if depends:
                deps[step_id] = [d.strip() for d in depends.split(",") if d.strip()]
        for step_id, refs in deps.items():
            for ref in refs:
                if ref not in step_ids:
                    raise ValidationError(
                        f"[workflow] id={step_id!r} has unknown depends_on {ref!r}"
                    )
        _detect_dep_cycle(deps, label="[workflow]")
    if "script" in sections:
        meaningful = [line for line in sections["script"] if line.strip()]
        if meaningful and not meaningful[0].strip().startswith("@type:"):
            raise ValidationError("[script] section must begin with @type: <lang>")
    if "checks" in sections:
        for line in sections["checks"]:
            stripped = line.strip()
            if not stripped or not stripped.startswith("-"):
                continue
            body = stripped[1:].strip()
            if "::" not in body:
                raise ValidationError(
                    f"[checks] line must use 'name :: status' format: {stripped!r}"
                )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: validate.py <file|glob|dir> [...]", file=sys.stderr)
        sys.exit(2)
    exit_code = 0
    paths: list[str] = []
    for arg in sys.argv[1:]:
        if any(ch in arg for ch in "*?[]"):
            matches = sorted(glob.glob(arg))
            paths.extend(matches or [arg])
            continue
        candidate = pathlib.Path(arg)
        if candidate.is_dir():
            paths.extend(str(path) for path in sorted(candidate.rglob("*.crumb")))
            continue
        paths.append(arg)
    for arg in paths:
        path = pathlib.Path(arg)
        try:
            parsed = parse_crumb(path.read_text(encoding="utf-8"))
            print(f"OK  {path}  kind={parsed['headers']['kind']}")
        except Exception as exc:
            print(f"ERR {path}  {exc}", file=sys.stderr)
            exit_code = 1
    sys.exit(exit_code)
