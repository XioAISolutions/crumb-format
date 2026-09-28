"""Run the conformance suite against the reference parser and the JS validator.

The suite is the artifact that makes "standard" a defensible word. Before it
existed, CRUMB had three independent parsers across three repositories, no
shared code, and no test comparing them — which is how ``crumb-format`` came to
emit ``v=1.4`` while ``CrumbContext`` hand-wrote ``v=1.3`` and ``CrumbLLM``'s
schemas demanded a ``v=1.5`` that no spec defines.

Only the accept/reject decision is normative. Rejection *messages* are
deliberately unconstrained so implementations can phrase errors for their own
users without breaking conformance.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import crumb_core  # noqa: E402

SUITE_DIR = REPO_ROOT / "conformance"
MANIFEST = json.loads((SUITE_DIR / "manifest.json").read_text(encoding="utf-8"))
CASES = MANIFEST["cases"]

ACCEPT = [c for c in CASES if c["expect"] == "accept"]
REJECT = [c for c in CASES if c["expect"] == "reject"]


def _read(case: dict) -> str:
    return (SUITE_DIR / case["file"]).read_text(encoding="utf-8")


def _ids(cases: list[dict]) -> list[str]:
    return [c["id"] for c in cases]


# ── the suite itself ─────────────────────────────────────────────────

def test_suite_declares_the_current_wire_version():
    assert MANIFEST["wire_version"] == crumb_core.WIRE_VERSION


def test_suite_has_both_polarities():
    """A suite of only-valid cases certifies nothing about rejection."""
    assert len(ACCEPT) >= 10
    assert len(REJECT) >= 10


def test_every_manifest_case_file_exists():
    missing = [c["id"] for c in CASES if not (SUITE_DIR / c["file"]).is_file()]
    assert not missing, f"manifest references missing files: {missing}"


def test_case_ids_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


def test_every_case_cites_a_spec_section_and_a_reason():
    for case in CASES:
        assert case.get("spec"), f"{case['id']} cites no SPEC section"
        assert len(case.get("reason", "")) > 20, f"{case['id']} has no substantive reason"


# ── the reference parser ─────────────────────────────────────────────

@pytest.mark.parametrize("case", ACCEPT, ids=_ids(ACCEPT))
def test_reference_parser_accepts(case):
    parsed = crumb_core.parse_crumb(_read(case))
    assert parsed["headers"]["kind"]


@pytest.mark.parametrize("case", REJECT, ids=_ids(REJECT))
def test_reference_parser_rejects(case):
    with pytest.raises(ValueError):
        crumb_core.parse_crumb(_read(case))


# ── the second implementation ────────────────────────────────────────
# A conformance suite that only one implementation runs is a regression test.
# Running the Node validator over the same cases is what makes it interop.

JS_VALIDATOR = REPO_ROOT / "validators" / "validate.js"
node_available = shutil.which("node") is not None


def _run_node(case: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", str(JS_VALIDATOR), str(SUITE_DIR / case["file"])],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


@pytest.mark.skipif(not node_available, reason="node not installed")
@pytest.mark.parametrize("case", ACCEPT, ids=_ids(ACCEPT))
def test_js_validator_accepts(case):
    result = _run_node(case)
    assert result.returncode == 0, f"{case['id']}: {result.stdout}{result.stderr}"


@pytest.mark.skipif(not node_available, reason="node not installed")
@pytest.mark.parametrize("case", REJECT, ids=_ids(REJECT))
def test_js_validator_rejects(case):
    result = _run_node(case)
    assert result.returncode != 0, (
        f"{case['id']} was accepted by the JS validator but must be rejected: "
        f"{case['reason']}"
    )


# ── round-tripping ───────────────────────────────────────────────────

@pytest.mark.parametrize("case", ACCEPT, ids=_ids(ACCEPT))
def test_parse_render_parse_is_stable(case):
    """render(parse(x)) must itself parse, and to the same content."""
    first = crumb_core.parse_crumb(_read(case))
    rendered = crumb_core.render_crumb(first["headers"], first["sections"])
    second = crumb_core.parse_crumb(rendered)
    assert second["headers"] == first["headers"]
    assert set(second["sections"]) == set(first["sections"])
