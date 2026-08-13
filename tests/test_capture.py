"""Tests for `crumb capture` and the Claude Code transcript reader.

`crumb it` only works if someone remembers to say it. A session-end hook
captures the handoff whether or not anyone thinks to ask — which is the
difference between a format people try once and one they keep using, and it is
the thing the leading competitor in this space wins on.

A hook must also never break the session it is attached to, so the failure
behaviour is tested as carefully as the happy path.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cli import transcripts  # noqa: E402

CLI = [sys.executable, str(REPO_ROOT / "crumb_cli.py")]


def _record(role: str, content, **extra) -> dict:
    record = {
        "type": role,
        "sessionId": "abc123",
        "uuid": f"u-{role}",
        "cwd": "/work",
        "gitBranch": "main",
        "message": {"role": role, "content": content},
    }
    record.update(extra)
    return record


@pytest.fixture
def transcript(tmp_path: Path) -> Path:
    records = [
        _record("user", "Fix the 503 in src/api/handler.py. Do not change the public API."),
        _record("assistant", [
            {"type": "thinking", "thinking": "a long private reasoning trace"},
            {"type": "text", "text": "Found it. The cause is a stale connection pool."},
            {"type": "tool_use", "name": "read_file", "input": {"path": "src/api/handler.py"}},
        ]),
        _record("user", [{"type": "tool_result", "content": "def handler(): ..."}]),
        _record("assistant", [{"type": "text",
                               "text": "Decision: recycle the pool on 503. The next step is to add "
                                       "a regression test in tests/test_handler.py."}]),
        # Bookkeeping records that are not conversation.
        {"type": "queue-operation", "sessionId": "abc123"},
        {"type": "last-prompt", "sessionId": "abc123"},
        # A subagent sidechain, which must not be spliced into the main thread.
        _record("assistant", [{"type": "text", "text": "sidechain internal chatter"}],
                isSidechain=True),
    ]
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


# ── the transcript reader ────────────────────────────────────────────

def test_claude_code_transcript_is_detected(transcript: Path):
    fmt, _ = transcripts.load_messages(transcript.read_text(encoding="utf-8"))
    assert fmt == "claude-code-transcript"


def test_only_conversation_records_are_read(transcript: Path):
    """queue-operation, last-prompt and friends are bookkeeping, not turns."""
    _, messages = transcripts.load_messages(transcript.read_text(encoding="utf-8"))
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]


def test_sidechains_are_excluded(transcript: Path):
    """A handoff describes the main thread, not a subagent's internal turns."""
    _, messages = transcripts.load_messages(transcript.read_text(encoding="utf-8"))
    assert not any("sidechain" in m.text for m in messages)


def test_thinking_is_dropped_and_tool_calls_are_kept(transcript: Path):
    fmt, messages = transcripts.load_messages(transcript.read_text(encoding="utf-8"))
    crumb = transcripts.to_crumb(messages, fmt=fmt)
    assert "private reasoning trace" not in crumb
    assert "read_file(path=src/api/handler.py)" in crumb


def test_a_single_message_jsonl_is_a_valid_transcript(tmp_path: Path):
    """One line of JSONL is also valid JSON, so it arrives as a bare dict."""
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps({"role": "user", "content": "just one turn here"}), encoding="utf-8")
    fmt, messages = transcripts.load_messages(path.read_text(encoding="utf-8"))
    assert len(messages) == 1


# ── extraction quality on long transcripts ───────────────────────────

def test_markdown_headings_are_not_extracted_as_content():
    """'## Remaining queue' matches the open-thread keywords but says nothing."""
    parts = transcripts.sentences("## Remaining queue\nThe next step is to ship it.")
    assert not any(p.startswith("#") for p in parts)
    assert any("next step" in p for p in parts)


def test_narration_is_not_mistaken_for_decisions():
    assert transcripts._is_noise("Let me check the logs for the error.")
    assert transcripts._is_noise("I'll start by reading the config.")
    assert not transcripts._is_noise("We're going with a signed cookie instead.")


def test_serialized_payloads_are_not_extracted_as_prose():
    blob = '{"author":"github-actions[bot]","comment":"## Results","pr":"owner/repo#1"}'
    assert transcripts._is_noise(blob)
    assert not transcripts._is_noise("The build id is abcdef1234567890 and it passed.")


def test_budgets_grow_with_transcript_length_and_are_bounded():
    assert transcripts._budget(40, 8, 24) == 8       # short session keeps the base
    assert transcripts._budget(1000, 8, 24) == 24    # long session is capped, not unbounded
    assert transcripts._budget(400, 12, 60) > 12     # and grows in between


def test_recency_wins_when_a_section_overflows():
    """In a long session the latest decisions supersede the early ones."""
    items = [f"decision number {i} was taken" for i in range(20)]
    kept = transcripts._dedupe(items, 3)
    assert kept == items[-3:]
    assert transcripts._dedupe(items, 3, prefer="head") == items[:3]


# ── the capture command ──────────────────────────────────────────────

def _capture(payload: dict, cwd: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        CLI + ["capture", *args], input=json.dumps(payload),
        capture_output=True, text=True, cwd=cwd,
    )


def test_capture_writes_a_valid_crumb_from_a_hook_payload(transcript: Path, tmp_path: Path):
    result = _capture({"session_id": "abc123def456", "transcript_path": str(transcript),
                       "hook_event_name": "SessionEnd"}, tmp_path)
    assert result.returncode == 0, result.stderr

    written = list((tmp_path / ".crumb").glob("session-*.crumb"))
    assert len(written) == 1
    import crumb_core
    parsed = crumb_core.parse_crumb(written[0].read_text(encoding="utf-8"))
    assert parsed["headers"]["kind"] == "task"
    assert parsed["headers"]["v"] == crumb_core.WIRE_VERSION


def test_capture_reports_what_it_cost(transcript: Path, tmp_path: Path):
    result = _capture({"session_id": "s", "transcript_path": str(transcript)}, tmp_path)
    assert "fact retention" in result.stderr
    assert "smaller" in result.stderr


def test_capture_accepts_an_explicit_transcript_without_stdin(transcript: Path, tmp_path: Path):
    result = subprocess.run(
        CLI + ["capture", "--transcript", str(transcript), "-o", "-"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("BEGIN CRUMB")


# ── a hook must not break the session it hangs off ───────────────────

def test_unreadable_transcript_exits_zero_by_default(tmp_path: Path):
    """Exiting non-zero here would surface an error at the end of every session."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("[]", encoding="utf-8")
    result = _capture({"session_id": "s", "transcript_path": str(empty)}, tmp_path)
    assert result.returncode == 0
    assert "skipped" in result.stderr


def test_strict_turns_an_unreadable_transcript_into_a_failure(tmp_path: Path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("[]", encoding="utf-8")
    result = _capture({"session_id": "s", "transcript_path": str(empty)}, tmp_path, "--strict")
    assert result.returncode == 1


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"session_id": "s"}, "no 'transcript_path'"),
        ({"session_id": "s", "transcript_path": "/does/not/exist.jsonl"}, "not found"),
    ],
)
def test_malformed_payloads_fail_loudly_with_a_usable_message(payload, expected, tmp_path):
    result = _capture(payload, tmp_path)
    assert result.returncode == 2
    assert expected in result.stderr


def test_non_json_stdin_is_reported_as_such(tmp_path: Path):
    result = subprocess.run(
        CLI + ["capture"], input="not json at all",
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "not JSON" in result.stderr


def test_no_stdin_and_no_transcript_explains_both_options(tmp_path: Path):
    result = subprocess.run(
        CLI + ["capture"], input="", capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "--transcript" in result.stderr
