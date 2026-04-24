"""Tests for cli/guardrails.py — [guardrails] → AgentAuth translator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))
import guardrails


SAMPLE_LINES = [
    "- type=tool      deny=shell-exec        why=security boundary",
    "- type=verify    require=tests          why=regression prevention",
    "- type=approval  action=merge  who=human  why=schema change",
    "- type=scope     max=files=5            why=token budget",
    "- prose line with no structured keys",
    "",
]


class TestParseGuardrailLine:
    def test_parses_key_value_bullet(self):
        parsed = guardrails.parse_guardrail_line(SAMPLE_LINES[0])
        assert parsed == {"type": "tool", "deny": "shell-exec", "why": "security"}
        # only the first token of a why= with spaces lands (matches the regex by design)

    def test_prose_bullet_returns_empty_dict(self):
        parsed = guardrails.parse_guardrail_line(SAMPLE_LINES[4])
        assert parsed == {}

    def test_blank_line_returns_none(self):
        assert guardrails.parse_guardrail_line("") is None
        assert guardrails.parse_guardrail_line("   ") is None

    def test_non_bullet_returns_none(self):
        assert guardrails.parse_guardrail_line("not a bullet") is None


class TestTranslateGuardrails:
    def test_buckets_are_grouped_by_action(self):
        buckets = guardrails.translate_guardrails(SAMPLE_LINES)
        assert any("shell-exec" == e.get("deny") for e in buckets["deny"])
        assert any("tests" == e.get("require") for e in buckets["require"])
        assert any(e.get("action") == "merge" for e in buckets["approval"])
        assert any(e.get("max") == "files=5" for e in buckets["scope"])

    def test_prose_line_goes_to_skipped(self):
        buckets = guardrails.translate_guardrails([SAMPLE_LINES[4]])
        assert len(buckets["skipped"]) == 1
        assert buckets["skipped"][0]["_reason"] == "no key=value pairs"

    def test_unknown_type_with_no_actionable_key_skipped(self):
        buckets = guardrails.translate_guardrails(["- type=fancy  comment=hello"])
        assert buckets["skipped"]
        assert "fancy" in buckets["skipped"][0]["_reason"]


class TestApplyGuardrailsToPolicy:
    def test_dry_run_without_policy(self):
        summary = guardrails.apply_guardrails_to_policy(
            SAMPLE_LINES, agent_name="agent-x"
        )
        assert summary["agent_name"] == "agent-x"
        assert "shell-exec" in summary["tools_denied"]
        assert "tests" in summary["tools_required"]
        assert summary["approvals"][0]["action"] == "merge"
        assert summary["scope"][0]["max"] == "files=5"
        assert summary["applied"] is False

    def test_applied_flag_set_when_policy_passed(self):
        calls = []

        class FakePolicy:
            def set_policy(self, **kwargs):
                calls.append(kwargs)
                return kwargs

        summary = guardrails.apply_guardrails_to_policy(
            SAMPLE_LINES, agent_name="agent-x", policy=FakePolicy()
        )
        assert summary["applied"] is True
        assert calls, "policy.set_policy should have been invoked"
        call = calls[0]
        assert call["agent_name"] == "agent-x"
        assert "shell-exec" in call["tools_denied"]
        assert "tests" in call["tools_allowed"]

    def test_no_deny_or_require_skips_policy_call(self):
        """If lines have no deny= or require=, don't bother AgentAuth."""
        calls = []

        class FakePolicy:
            def set_policy(self, **kwargs):
                calls.append(kwargs)

        summary = guardrails.apply_guardrails_to_policy(
            ["- type=approval  action=merge  who=human"],
            agent_name="agent-x",
            policy=FakePolicy(),
        )
        assert summary["applied"] is False
        assert calls == []

    def test_empty_input_returns_empty_summary(self):
        summary = guardrails.apply_guardrails_to_policy([], agent_name="agent-x")
        assert summary["tools_denied"] == []
        assert summary["tools_required"] == []
        assert summary["skipped"] == []
