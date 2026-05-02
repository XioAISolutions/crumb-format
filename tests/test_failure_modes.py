"""Tests for cli/failure_modes and `crumb lint --check-failure-modes`.

Covers:
- canonical-name lookup
- ad-hoc → canonical suggestion heuristics (10+ patterns)
- check_failure_mode_lines walks the [checks] section correctly
- end-to-end CLI integration (flag off silent, flag on emits findings)
- the canonical list in code matches the doc (sanity check)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "cli"))
sys.path.insert(0, str(REPO_ROOT))

from cli import crumb as crumb_cli  # noqa: E402
from cli.failure_modes import (  # noqa: E402
    CANONICAL_NAMES,
    CANONICAL_SET,
    check_failure_mode_lines,
    is_canonical,
    suggest_canonical,
)


# ── canonical lookup ───────────────────────────────────────────────────


class TestCanonicalLookup:
    def test_known_names_recognized(self):
        assert is_canonical("hallucinated_tool_call")
        assert is_canonical("refusal_loop")
        assert is_canonical("unauthorized_tool_call")

    def test_unknown_names_not_recognized(self):
        assert not is_canonical("totally_made_up")
        assert not is_canonical("")
        assert not is_canonical("HALLUCINATED_TOOL_CALL")  # case-sensitive

    def test_canonical_list_matches_doc(self):
        # Sanity: 10 names per the v1.4 draft.
        assert len(CANONICAL_NAMES) == 10
        # And the doc lists each by name — string-grep the doc to confirm.
        doc_path = REPO_ROOT / "docs" / "v1.4" / "agent-failure-modes.md"
        doc_text = doc_path.read_text(encoding="utf-8")
        for name in CANONICAL_NAMES:
            assert f"`{name}`" in doc_text, f"{name} missing from doc"


# ── suggestion heuristics ──────────────────────────────────────────────


class TestSuggestCanonical:
    @pytest.mark.parametrize("ad_hoc,expected", [
        ("hallucination",                  "hallucinated_tool_call"),
        ("Hallucinated-Tool-Call",         "hallucinated_tool_call"),
        ("madeUpTool",                     "hallucinated_tool_call"),
        ("nonexistent_tool",               "hallucinated_tool_call"),
        ("refusal",                        "refusal_loop"),
        ("model_refused",                  "refusal_loop"),
        ("tool error",                     "tool_error_unhandled"),
        ("toolFail",                       "tool_error_unhandled"),
        ("off-topic",                      "semantic_drift"),
        ("contextLost",                    "semantic_drift"),
        ("budget-exceeded",                "token_budget_exceeded"),
        ("oversize",                       "token_budget_exceeded"),
        ("invalid_handoff",                "invalid_handoff_target"),
        ("cycle",                          "circular_reference"),
        ("circular_ref",                   "circular_reference"),
        ("truncated",                      "truncated_output"),
        ("cutoff",                         "truncated_output"),
        ("prompt_injection",               "prompt_injection_suspected"),
        ("jailbreak",                      "prompt_injection_suspected"),
        ("unauthorized",                   "unauthorized_tool_call"),
        ("policy_violation",               "unauthorized_tool_call"),
    ])
    def test_known_patterns_map(self, ad_hoc, expected):
        assert suggest_canonical(ad_hoc) == expected

    def test_canonical_returns_none(self):
        # Already-canonical names should NOT trigger a suggestion.
        for name in CANONICAL_NAMES:
            assert suggest_canonical(name) is None

    def test_unrecognized_returns_none(self):
        # Don't false-correct names that don't match any pattern.
        assert suggest_canonical("totally_made_up") is None
        assert suggest_canonical("xyz123") is None
        assert suggest_canonical("") is None


# ── check_failure_mode_lines ───────────────────────────────────────────


class TestCheckLines:
    def test_canonical_emits_info(self):
        lines = ["- hallucinated_tool_call :: detected count=1"]
        findings = list(check_failure_mode_lines(lines))
        assert len(findings) == 1
        assert findings[0].code == "canonical_failure_mode"
        assert findings[0].name == "hallucinated_tool_call"

    def test_ad_hoc_with_match_emits_suggestion(self):
        lines = ["- hallucination :: detected"]
        findings = list(check_failure_mode_lines(lines))
        assert len(findings) == 1
        assert findings[0].code == "ad_hoc_with_suggestion"
        assert "hallucinated_tool_call" in findings[0].message

    def test_ad_hoc_without_match_silent(self):
        lines = ["- coverage_baseline :: pass value=87"]
        assert list(check_failure_mode_lines(lines)) == []

    def test_skips_non_check_lines(self):
        lines = [
            "@type: text/plain",         # type annotation
            "- hallucinated_tool_call :: detected",
            "",                           # blank
            "free-form prose",            # not a check line
            "- refusal_loop :: detected count=2",
        ]
        findings = list(check_failure_mode_lines(lines))
        assert [f.name for f in findings] == ["hallucinated_tool_call", "refusal_loop"]

    def test_line_numbers_track_position(self):
        lines = [
            "- coverage :: pass",
            "- hallucinated_tool_call :: detected",
        ]
        findings = list(check_failure_mode_lines(lines))
        # First finding is on the second line (line 2).
        assert findings[0].line_no == 2


# ── lint integration ──────────────────────────────────────────────────


CRUMB_WITH_CHECKS = """\
BEGIN CRUMB
v=1.3
kind=task
title=test
source=test
---
[goal]
ship

[context]
- nothing

[constraints]
- nothing

[checks]
- coverage :: pass value=87 threshold=85
- hallucinated_tool_call :: detected count=1
- refusal :: detected count=2
- random_unmapped_check :: pass
END CRUMB
"""


class TestLintFailureModesFlag:
    def _run(self, tmp_path, *extra_args, capsys=None):
        path = tmp_path / "x.crumb"
        path.write_text(CRUMB_WITH_CHECKS, encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            crumb_cli.main(["lint", str(path), *extra_args])
        return exc.value.code

    def test_without_flag_silent(self, tmp_path, capsys):
        rc = self._run(tmp_path, capsys=capsys)
        assert rc == 0
        captured = capsys.readouterr()
        # No failure-mode findings should have been emitted.
        assert "canonical_failure_mode" not in (captured.out + captured.err)
        assert "ad_hoc_with_suggestion" not in (captured.out + captured.err)

    def test_with_flag_emits_findings(self, tmp_path, capsys):
        rc = self._run(tmp_path, "--check-failure-modes", capsys=capsys)
        assert rc == 0
        all_output = capsys.readouterr().out + capsys.readouterr().err
        # Re-read to reset capsys (tested empirically above)
        path = tmp_path / "x.crumb"
        with pytest.raises(SystemExit):
            crumb_cli.main(["lint", str(path), "--check-failure-modes"])
        all_output = capsys.readouterr().out
        # The canonical name `hallucinated_tool_call` should be flagged INFO
        assert "canonical_failure_mode" in all_output
        # `refusal` should suggest `refusal_loop`
        assert "ad_hoc_with_suggestion" in all_output
        assert "refusal_loop" in all_output
        # `random_unmapped_check` should NOT trigger anything
        assert "random_unmapped_check" not in all_output or "ad_hoc" not in all_output
