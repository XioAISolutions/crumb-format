"""Tests for the v1.4 normative wire-format additions (1.0.0 release).

Confirms:
- v=1.4 is accepted by both Python and Node validators
- v=1.4 templates emit cleanly
- Backward compat: v=1.1, v=1.2, v=1.3 all still validate
- The SPEC.md amendments are documented (sanity grep)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "cli"))
sys.path.insert(0, str(REPO_ROOT))

from cli import crumb as crumb_cli  # noqa: E402


V14_MINIMAL = """\
BEGIN CRUMB
v=1.4
kind=mem
title=v1.4 minimal
source=test
---
[consolidated]
- v1.4 is accepted
END CRUMB
"""

V13_STILL_VALID = """\
BEGIN CRUMB
v=1.3
kind=mem
title=v1.3 still validates
source=test
---
[consolidated]
- backward compat preserved
END CRUMB
"""


# ── Python validator ──────────────────────────────────────────────────


class TestV14PythonValidator:
    def test_v14_accepted(self):
        parsed = crumb_cli.parse_crumb(V14_MINIMAL)
        assert parsed["headers"]["v"] == "1.4"

    def test_v15_rejected(self):
        # Sanity: future versions still rejected. Confirms the
        # whitelist is real, not just a string-acceptance-anywhere.
        bad = V14_MINIMAL.replace("v=1.4", "v=1.5")
        with pytest.raises(ValueError, match="unsupported version"):
            crumb_cli.parse_crumb(bad)

    @pytest.mark.parametrize("v", ["1.1", "1.2", "1.3", "1.4"])
    def test_all_supported_versions_validate(self, v):
        text = V14_MINIMAL.replace("v=1.4", f"v={v}")
        parsed = crumb_cli.parse_crumb(text)
        assert parsed["headers"]["v"] == v


# ── Node validator (cross-language parity) ─────────────────────────────


NODE = shutil.which("node")
VALIDATOR_JS = REPO_ROOT / "validators" / "validate.js"


class TestV14NodeValidator:
    @pytest.fixture(scope="class")
    def needs_node(self):
        if NODE is None:
            pytest.skip("node not on PATH; JS validator parity tests skipped")

    def _run(self, text: str) -> int:
        proc = subprocess.run(
            [NODE, "-e", f"require({str(VALIDATOR_JS)!r}).parseCrumb({text!r});"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode

    def test_v14_accepted(self, needs_node, tmp_path):
        path = tmp_path / "x.crumb"
        path.write_text(V14_MINIMAL, encoding="utf-8")
        proc = subprocess.run(
            [NODE, str(VALIDATOR_JS), str(path)],
            capture_output=True, text=True, timeout=10,
        )
        assert proc.returncode == 0, f"v=1.4 rejected by node: {proc.stderr}"
        assert "OK" in proc.stdout

    def test_v15_rejected(self, needs_node, tmp_path):
        path = tmp_path / "x.crumb"
        path.write_text(V14_MINIMAL.replace("v=1.4", "v=1.5"), encoding="utf-8")
        proc = subprocess.run(
            [NODE, str(VALIDATOR_JS), str(path)],
            capture_output=True, text=True, timeout=10,
        )
        assert proc.returncode != 0
        assert "unsupported version" in proc.stderr

    def test_v13_still_validates_in_node(self, needs_node, tmp_path):
        path = tmp_path / "x.crumb"
        path.write_text(V13_STILL_VALID, encoding="utf-8")
        proc = subprocess.run(
            [NODE, str(VALIDATOR_JS), str(path)],
            capture_output=True, text=True, timeout=10,
        )
        assert proc.returncode == 0


# ── Templates emit v=1.4 ───────────────────────────────────────────────


class TestV14Templates:
    @pytest.mark.parametrize("kind", ["task", "mem", "map", "log", "todo", "agent"])
    def test_default_emission_is_v14(self, kind, tmp_path):
        out = tmp_path / f"{kind}.crumb"
        argv = ["new", kind, "--title", "x", "--source", "t", "-o", str(out)]
        if kind == "task":
            argv.extend(["--goal", "g", "--context", "c", "--constraints", "x"])
        elif kind == "mem":
            argv.extend(["--entries", "fact"])
        elif kind == "map":
            argv.extend(["--description", "p"])
        elif kind == "log":
            argv.extend(["--entries", "evt"])
        elif kind == "todo":
            argv.extend(["--entries", "thing"])
        elif kind == "agent":
            argv.extend(["--agent-id", "a", "--identity", "i"])
        crumb_cli.main(argv)
        text = out.read_text(encoding="utf-8")
        assert text.startswith("BEGIN CRUMB\nv=1.4\n"), (
            f"{kind} template did not emit v=1.4 by default: {text[:60]!r}"
        )


# ── SPEC.md amendments ────────────────────────────────────────────────


class TestSpecAmendments:
    @pytest.fixture(scope="class")
    def spec_text(self):
        return (REPO_ROOT / "SPEC.md").read_text(encoding="utf-8")

    def test_title_is_v14(self, spec_text):
        assert spec_text.startswith("# .crumb Specification (v1.4)")

    def test_section_11_4_present(self, spec_text):
        assert "### 11.4 Normative `deadline=` format (v1.4)" in spec_text
        assert "ISO-8601" in spec_text

    def test_section_21_1_1_present(self, spec_text):
        assert "#### 21.1.1 Typed thresholds (v1.4 normative)" in spec_text
        assert "Sender consistency rule" in spec_text

    def test_section_21_1_2_present(self, spec_text):
        assert "#### 21.1.2 Canonical failure-mode names" in spec_text
        # All 10 canonical names appear in the spec table.
        for name in ("hallucinated_tool_call", "refusal_loop",
                     "tool_error_unhandled", "semantic_drift",
                     "token_budget_exceeded", "invalid_handoff_target",
                     "circular_reference", "truncated_output",
                     "prompt_injection_suspected", "unauthorized_tool_call"):
            assert name in spec_text, f"canonical name {name!r} missing"

    def test_canonical_names_match_runtime(self, spec_text):
        # The runtime list (cli/failure_modes.py) and the spec list
        # must stay in sync. The runtime list is the authority for
        # `crumb lint --check-failure-modes`.
        from cli.failure_modes import CANONICAL_NAMES
        for name in CANONICAL_NAMES:
            assert name in spec_text, (
                f"runtime name {name!r} not in SPEC.md — they have drifted"
            )
