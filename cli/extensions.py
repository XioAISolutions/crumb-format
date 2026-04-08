"""CRUMB extension model helpers.

The core v1.1 parser stays permissive. This module documents and normalizes the
optional extension surface so new tooling can add metadata without breaking old
readers.
"""

from __future__ import annotations

import re
from typing import Iterable

SPEC_VERSION = "1.1"
SPEC_URL = "https://github.com/XioAISolutions/crumb-format/blob/main/SPEC.md"

CORE_OPTIONAL_HEADERS = {
    "title",
    "dream_pass",
    "dream_sessions",
    "max_index_tokens",
    "max_total_tokens",
    "id",
    "project",
    "env",
    "tags",
    "url",
    "extensions",
}

KNOWN_HEADERS = {"v", "kind", "source"} | CORE_OPTIONAL_HEADERS
HEADER_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
EXTENSION_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+){1,}$")
NAMESPACED_HEADER_PATTERN = re.compile(r"^(?:x-[a-z0-9][a-z0-9._-]*|ext\.[a-z0-9][a-z0-9._-]*)$")


def parse_extensions(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def format_extensions(values: Iterable[str]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        value = raw.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ", ".join(ordered)


def is_known_header(key: str) -> bool:
    return key in KNOWN_HEADERS


def is_valid_header_key(key: str) -> bool:
    return bool(HEADER_KEY_PATTERN.match(key))


def is_namespaced_extension_name(name: str) -> bool:
    return bool(EXTENSION_NAME_PATTERN.match(name))


def is_namespaced_header(key: str) -> bool:
    return bool(NAMESPACED_HEADER_PATTERN.match(key))


def append_extension(headers: dict[str, str], extension_name: str) -> None:
    current = parse_extensions(headers.get("extensions"))
    current.append(extension_name)
    headers["extensions"] = format_extensions(current)
