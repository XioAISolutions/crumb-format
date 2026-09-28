"""Tests for `crumb from-messages` — real conversation ingest.

Covers the four provider shapes, the JSONL variant, and the extraction
behaviours that make the output a handoff rather than a transcript summary.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cli import crumb as crumb_cli  # noqa: E402
from cli import transcripts  # noqa: E402


# ── fixtures ─────────────────────────────────────────────────────────

CONVERSATION = [
    ("user", "The cart drops on refresh at /checkout/payment. You must not change "
             "the public CartService API, and don't touch the webhook handler."),
    ("assistant", "Sure, happy to help. We're going with sessionStorage as the cause. "
                  "Decision: move hydration into middleware.ts. The next step is to add "
                  "a regression test in tests/checkout_refresh.spec.ts covering refresh."),
    ("user", "Good. Make sure we keep the cart_v2 cookie name. Write the fix."),
]


def _openai(tmp_path: Path) -> Path:
    payload = {"messages": [{"role": r, "content": c} for r, c in CONVERSATION]}
    path = tmp_path / "openai.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _anthropic(tmp_path: Path) -> Path:
    messages = []
    for role, content in CONVERSATION:
        blocks = [{"type": "text", "text": content}]
        if role == "assistant":
            blocks.insert(0, {"type": "thinking", "thinking": "long private reasoning trace"})
            blocks.append({"type": "tool_use", "name": "read_file",
                           "input": {"path": "src/checkout/CartService.ts"}})
        messages.append({"role": role, "content": blocks})
    path = tmp_path / "anthropic.json"
    path.write_text(json.dumps({"messages": messages}), encoding="utf-8")
    return path


def _chatgpt_export(tmp_path: Path) -> Path:
    mapping = {
        f"node-{i}": {
            "message": {
                "create_time": 1700000000 + i,
                "author": {"role": role},
                "content": {"parts": [content]},
            }
        }
        for i, (role, content) in enumerate(CONVERSATION)
    }
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps([{"title": "Checkout", "mapping": mapping}]), encoding="utf-8")
    return path


def _claude_export(tmp_path: Path) -> Path:
    payload = [{
        "name": "Checkout",
        "chat_messages": [
            {"sender": "human" if r == "user" else "assistant", "text": c}
            for r, c in CONVERSATION
        ],
    }]
    path = tmp_path / "claude_export.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


BUILDERS = {
    "openai-messages": _openai,
    "anthropic-messages": _anthropic,
    "chatgpt-export": _chatgpt_export,
    "claude-export": _claude_export,
}


# ── detection and parsing ────────────────────────────────────────────

@pytest.mark.parametrize("expected_fmt", sorted(BUILDERS))
def test_each_format_is_detected_and_parsed(tmp_path, expected_fmt):
    path = BUILDERS[expected_fmt](tmp_path)
    fmt, messages = transcripts.load_messages(path.read_text(encoding="utf-8"))
    assert fmt == expected_fmt
    assert [m.role for m in messages][:3] == ["user", "assistant", "user"]


def test_jsonl_of_bare_messages(tmp_path):
    path = tmp_path / "messages.jsonl"
    path.write_text(
        "\n".join(json.dumps({"role": r, "content": c}) for r, c in CONVERSATION),
        encoding="utf-8",
    )
    fmt, messages = transcripts.load_messages(path.read_text(encoding="utf-8"))
    assert fmt == "openai-messages"
    assert len(messages) == 3


def test_openai_response_envelope_with_choices():
    payload = {"choices": [{"message": {"role": "assistant", "content": "We decided to ship it."}}]}
    fmt, messages = transcripts.load_messages(json.dumps(payload))
    assert fmt == "openai-messages"
    assert messages[0].role == "assistant"


def test_cli_format_list_matches_module():
    assert crumb_cli._TRANSCRIPT_FORMATS == transcripts.FORMATS


@pytest.mark.parametrize("bad", ["", "not json at all", json.dumps({"foo": "bar"})])
def test_unusable_input_raises(bad):
    with pytest.raises(transcripts.TranscriptError):
        transcripts.load_messages(bad)


# ── extraction behaviour ─────────────────────────────────────────────

def _crumb_for(tmp_path, fmt="openai-messages") -> str:
    path = BUILDERS[fmt](tmp_path)
    detected, messages = transcripts.load_messages(path.read_text(encoding="utf-8"))
    return transcripts.to_crumb(messages, fmt=detected)


def test_output_is_a_valid_v14_task_crumb(tmp_path):
    text = _crumb_for(tmp_path)
    parsed = crumb_cli.parse_crumb(text)
    assert parsed["headers"]["v"] == "1.4"
    assert parsed["headers"]["kind"] == "task"
    assert not set(crumb_cli.REQUIRED_SECTIONS["task"]) - set(parsed["sections"])


def test_thinking_blocks_are_dropped(tmp_path):
    text = _crumb_for(tmp_path, "anthropic-messages")
    assert "private reasoning trace" not in text


def test_tool_calls_are_captured_with_their_paths(tmp_path):
    text = _crumb_for(tmp_path, "anthropic-messages")
    assert "read_file(path=src/checkout/CartService.ts)" in text
    # Paths mentioned only inside tool arguments still reach [context].
    assert "src/checkout/CartService.ts" in text


def test_constraints_are_extracted_from_user_turns(tmp_path):
    sections = crumb_cli.parse_crumb(_crumb_for(tmp_path))["sections"]
    body = " ".join(sections["constraints"])
    assert "must not change the public CartService API" in body
    assert "cart_v2" in body


def test_open_threads_become_handoff_items(tmp_path):
    sections = crumb_cli.parse_crumb(_crumb_for(tmp_path))["sections"]
    assert "handoff" in sections
    assert any("checkout_refresh.spec.ts" in line for line in sections["handoff"])


def test_dotted_filenames_survive_sentence_splitting():
    """A naive split on '.' truncates exactly the filenames a handoff needs."""
    parts = transcripts.sentences(
        "Add a test in tests/checkout_refresh.spec.ts now. Then ship v1.4 of the parser."
    )
    assert parts[0].endswith("tests/checkout_refresh.spec.ts now.")
    assert "v1.4" in parts[1]


def test_goal_skips_bare_acknowledgements(tmp_path):
    sections = crumb_cli.parse_crumb(_crumb_for(tmp_path))["sections"]
    goal = " ".join(sections["goal"]).strip()
    assert goal.startswith("Make sure we keep the cart_v2 cookie name")


def test_overlapping_decisions_are_collapsed(tmp_path):
    sections = crumb_cli.parse_crumb(_crumb_for(tmp_path))["sections"]
    decisions = [l for l in sections["context"] if l.strip().startswith("- ") or l.startswith("  -")]
    normalised = [l.strip(" -") for l in decisions]
    for a in normalised:
        assert sum(1 for b in normalised if a and a in b) == 1, f"duplicate prefix: {a}"


def test_all_formats_yield_the_same_signal(tmp_path):
    """The reader is a shim; extraction must not depend on the provider shape."""
    goals = set()
    for fmt in ("openai-messages", "chatgpt-export", "claude-export"):
        sections = crumb_cli.parse_crumb(_crumb_for(tmp_path, fmt))["sections"]
        goals.add(" ".join(sections["goal"]).strip())
    assert len(goals) == 1


# ── CLI wiring ───────────────────────────────────────────────────────

def test_cli_end_to_end(tmp_path):
    src = _openai(tmp_path)
    out = tmp_path / "handoff.crumb"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "crumb_cli.py"), "from-messages",
         "-i", str(src), "-o", str(out), "--stats"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "Fact retention" in result.stdout
    validated = subprocess.run(
        [sys.executable, str(REPO_ROOT / "crumb_cli.py"), "validate", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert validated.returncode == 0, validated.stdout + validated.stderr


def test_cli_reports_unreadable_input(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "crumb_cli.py"), "from-messages", "-i", str(bad)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 1
    assert "error:" in result.stderr
