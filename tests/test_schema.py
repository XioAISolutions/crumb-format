"""Tests for the JSON Schema contract introduced in 0.7.0.

`schemas/crumb.schema.json` formally describes the shape returned by
`parse_crumb(text)`. This test pins:

1. The schema file is itself valid JSON with the expected top-level shape.
2. Every fixture in `fixtures/valid/` parses to something that matches the
   schema (required keys, value types, allowed `kind`, allowed `v`).

We avoid depending on the `jsonschema` library to keep the runtime package
zero-dependency — instead, we implement the narrow subset of Draft 2020-12
semantics that our schema actually uses (type, required, enum, array items).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from cli import crumb

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "crumb.schema.json"
VALID_FIXTURES = sorted(glob.glob(str(REPO_ROOT / "fixtures" / "valid" / "*.crumb")))
EXTENSION_FIXTURES = sorted(
    glob.glob(str(REPO_ROOT / "fixtures" / "extensions" / "*.crumb"))
)


# ---------------------------------------------------------------------------
# Schema-file integrity
# ---------------------------------------------------------------------------

def test_schema_file_is_valid_json():
    data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert data["$schema"].startswith("https://json-schema.org/")
    assert data["title"].startswith("CRUMB")
    assert data["type"] == "object"


def test_schema_requires_headers_and_sections():
    data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert set(data["required"]) == {"headers", "sections"}


def test_schema_declares_all_six_kinds():
    data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    kinds = data["properties"]["headers"]["properties"]["kind"]["enum"]
    assert set(kinds) == {"task", "mem", "map", "log", "todo", "wake"}


def test_schema_pins_version_1_1():
    data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    versions = data["properties"]["headers"]["properties"]["v"]["enum"]
    assert versions == ["1.1"]


# ---------------------------------------------------------------------------
# Parsed-output conformance — every valid fixture matches the schema shape
# ---------------------------------------------------------------------------

def _validate_parsed(parsed, schema) -> list[str]:
    """Minimal structural validator — returns list of errors (empty = valid)."""
    errors: list[str] = []

    # Top-level
    if not isinstance(parsed, dict):
        return ["root is not an object"]
    for key in schema["required"]:
        if key not in parsed:
            errors.append(f"missing required key: {key}")

    headers = parsed.get("headers", {})
    sections = parsed.get("sections", {})

    # headers: required keys + enum checks
    header_schema = schema["properties"]["headers"]
    for key in header_schema["required"]:
        if key not in headers:
            errors.append(f"headers missing required key: {key}")
    for key, value in headers.items():
        if not isinstance(value, str):
            errors.append(f"header {key!r} is not a string (got {type(value).__name__})")
    kind_enum = header_schema["properties"]["kind"]["enum"]
    if "kind" in headers and headers["kind"] not in kind_enum:
        errors.append(f"kind={headers['kind']!r} not in enum {kind_enum}")
    v_enum = header_schema["properties"]["v"]["enum"]
    if "v" in headers and headers["v"] not in v_enum:
        errors.append(f"v={headers['v']!r} not in enum {v_enum}")

    # sections: dict of string-lists
    if not isinstance(sections, dict):
        errors.append("sections is not an object")
    else:
        for name, body in sections.items():
            if not isinstance(body, list):
                errors.append(f"section {name!r} body is not an array")
                continue
            for i, line in enumerate(body):
                if not isinstance(line, str):
                    errors.append(f"section {name!r}[{i}] is not a string")

    return errors


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("fixture_path", VALID_FIXTURES + EXTENSION_FIXTURES)
def test_valid_fixtures_match_schema(fixture_path, schema):
    text = Path(fixture_path).read_text(encoding="utf-8")
    parsed = crumb.parse_crumb(text)
    errors = _validate_parsed(parsed, schema)
    assert not errors, f"{fixture_path} did not match schema: {errors}"


def test_unknown_kind_fails_schema(schema):
    """Synthetic parsed doc with an off-spec `kind` is rejected by the schema."""
    synthetic = {
        "headers": {"v": "1.1", "kind": "banana", "source": "a"},
        "sections": {"goal": ["x"]},
    }
    errors = _validate_parsed(synthetic, schema)
    assert any("kind" in e for e in errors)


def test_missing_required_header_fails_schema(schema):
    synthetic = {
        "headers": {"v": "1.1", "kind": "task"},  # no source
        "sections": {"goal": ["x"], "context": ["- y"], "constraints": ["- z"]},
    }
    errors = _validate_parsed(synthetic, schema)
    assert any("source" in e for e in errors)
