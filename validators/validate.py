#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
from typing import Dict, List

REQUIRED_HEADERS = ["v", "kind", "source"]
REQUIRED_SECTIONS = {
    "task": ["goal", "context", "constraints"],
    "mem": ["consolidated"],
    "map": ["project", "modules"],
}

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
    if headers["v"] != "1.1":
        raise ValidationError("unsupported version")
    kind = headers["kind"]
    if kind not in REQUIRED_SECTIONS:
        raise ValidationError(f"unknown kind: {kind}")
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
        if section not in sections:
            raise ValidationError(f"missing required section for kind={kind}: {section}")
        if not any(item.strip() for item in sections[section]):
            raise ValidationError(f"section is empty: {section}")
    return {"headers": headers, "sections": sections}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: validate.py <file> [<file> ...]", file=sys.stderr)
        sys.exit(2)
    exit_code = 0
    for arg in sys.argv[1:]:
        path = pathlib.Path(arg)
        try:
            parsed = parse_crumb(path.read_text(encoding="utf-8"))
            print(f"OK  {path}  kind={parsed['headers']['kind']}")
        except Exception as exc:
            print(f"ERR {path}  {exc}", file=sys.stderr)
            exit_code = 1
    sys.exit(exit_code)
