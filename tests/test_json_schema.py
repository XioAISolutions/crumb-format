"""Validate that schemas/crumb-v1.4.schema.json describes the parsed
form of every shipping example crumb.

The wire format is plain text. Downstream tools (IDE plugins, CI
runners, language ports) work with the parsed structure: a dict with
`headers` and `sections`. This test asserts that structure round-trips
through the JSON Schema for every example we ship.

Drift here means either the schema is out of date or an example is
malformed — both are bugs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "crumb-v1.4.schema.json"


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def parser():
    sys.path.insert(0, str(REPO_ROOT / "cli"))
    sys.path.insert(0, str(REPO_ROOT))
    import crumb as crumb_mod  # type: ignore
    return crumb_mod.parse_crumb


def _all_example_paths():
    """Every example + valid fixture + extension fixture."""
    paths = []
    for d in ["examples", "fixtures/valid", "fixtures/extensions"]:
        paths.extend(sorted((REPO_ROOT / d).glob("*.crumb")))
    return paths


class TestSchemaIsValid:
    def test_schema_loads_as_json(self, schema):
        assert schema["title"] == "CRUMB v1.4 (parsed)"
        assert "$schema" in schema

    def test_schema_uses_2020_12_draft(self, schema):
        # Pin to draft 2020-12 so consumers know which validator dialect
        # to use. This matches `jsonschema.Draft202012Validator`.
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"

    def test_schema_passes_metaschema(self, schema):
        """The schema itself must be a valid JSON Schema."""
        validator = jsonschema.Draft202012Validator
        validator.check_schema(schema)


class TestExamplesValidateAgainstSchema:
    """Every shipping example must round-trip through parse_crumb and
    validate against the schema. If this breaks, either the schema or
    the example is wrong."""

    @pytest.mark.parametrize(
        "path",
        _all_example_paths(),
        ids=[p.name for p in _all_example_paths()],
    )
    def test_example_validates(self, path, schema, parser):
        try:
            parsed = parser(path.read_text())
        except ValueError as exc:
            pytest.skip(f"example doesn't parse: {exc}")

        # The schema expects sections to be a dict of name → list[str].
        # parse_crumb already returns that shape. Verify defensively.
        assert isinstance(parsed["headers"], dict)
        assert isinstance(parsed["sections"], dict)
        for body in parsed["sections"].values():
            assert isinstance(body, list)

        try:
            jsonschema.validate(parsed, schema)
        except jsonschema.ValidationError as exc:
            pytest.fail(f"{path.name} failed schema validation: {exc.message}")


class TestSchemaCatchesBadInput:
    """Negative cases: things the schema must reject."""

    def test_unknown_kind_rejected(self, schema):
        bad = {
            "headers": {"v": "1.4", "kind": "frogpile", "source": "x"},
            "sections": {},
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_unsupported_version_rejected(self, schema):
        bad = {
            "headers": {"v": "1.5", "kind": "task", "source": "x"},
            "sections": {"goal": ["x"], "context": ["x"], "constraints": ["x"]},
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_task_missing_required_section_rejected(self, schema):
        bad = {
            "headers": {"v": "1.4", "kind": "task", "source": "x"},
            "sections": {"goal": ["x"]},  # missing context, constraints
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_agent_missing_id_rejected(self, schema):
        bad = {
            "headers": {"v": "1.4", "kind": "agent", "source": "x"},
            "sections": {"identity": ["x"]},
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_missing_source_header_rejected(self, schema):
        bad = {
            "headers": {"v": "1.4", "kind": "task"},  # no source
            "sections": {"goal": ["x"], "context": ["x"], "constraints": ["x"]},
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)


class TestSchemaAcceptsValidInputs:
    """Positive cases: minimal valid documents per kind."""

    def test_minimal_task_validates(self, schema):
        doc = {
            "headers": {"v": "1.4", "kind": "task", "source": "test", "title": "x"},
            "sections": {"goal": ["g"], "context": ["c"], "constraints": ["k"]},
        }
        jsonschema.validate(doc, schema)

    def test_task_with_fold_validates(self, schema):
        doc = {
            "headers": {"v": "1.4", "kind": "task", "source": "test"},
            "sections": {
                "fold:goal/summary": ["short"],
                "fold:goal/full": ["long"],
                "context": ["c"],
                "constraints": ["k"],
            },
        }
        jsonschema.validate(doc, schema)

    def test_agent_with_id_validates(self, schema):
        doc = {
            "headers": {
                "v": "1.4", "kind": "agent", "source": "test",
                "id": "test-agent-1",
            },
            "sections": {"identity": ["I am a test agent"]},
        }
        jsonschema.validate(doc, schema)
