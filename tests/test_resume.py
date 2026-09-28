"""Tests for `crumb resume` — the SessionStart half of the handoff loop.

`crumb capture` writes a handoff when a session ends. Without `resume`, picking
it up is a manual copy-paste, and the step people forget is the step that
decides whether a tool gets used twice.

Two properties here are load-bearing and easy to break by accident:

1. **stdout is model context.** Claude Code adds a SessionStart hook's stdout to
   the conversation. A stray `print()` anywhere in this path is not a cosmetic
   bug — it is text injected into the user's session as though they had written
   it. Every diagnostic must go to stderr.
2. **`clear` must stay excluded.** The user asked for a clean slate; quietly
   restoring the old context overrides an explicit instruction.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI = REPO_ROOT / "crumb_cli.py"

CRUMB = """BEGIN CRUMB
v=1.4
kind=task
source=transcript.claude-code
---
[goal]
Fix the 502 on /api/ledger after the 1200ms timeout
[context]
Root cause was connection pool exhaustion
[constraints]
- Do not change the public API
END CRUMB
"""


def run_resume(cwd: Path, payload: dict | str | None, *args: str):
    stdin = "" if payload is None else (
        payload if isinstance(payload, str) else json.dumps(payload)
    )
    return subprocess.run(
        [sys.executable, str(CLI), "resume", *args],
        input=stdin, capture_output=True, text=True, cwd=cwd,
    )


def make_project(tmp_path: Path, *, age_hours: float = 0.0, name: str = "session-a.crumb") -> Path:
    crumb_dir = tmp_path / ".crumb"
    crumb_dir.mkdir(parents=True, exist_ok=True)
    target = crumb_dir / name
    target.write_text(CRUMB, encoding="utf-8")
    if age_hours:
        old = time.time() - age_hours * 3600
        os.utime(target, (old, old))
    return target


def injected_context(result) -> str:
    payload = json.loads(result.stdout)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    return payload["hookSpecificOutput"]["additionalContext"]


# ── the hook contract ────────────────────────────────────────────────

def test_startup_injects_the_latest_crumb_as_hook_json(tmp_path):
    make_project(tmp_path)
    result = run_resume(tmp_path, {"source": "startup", "hook_event_name": "SessionStart"})
    assert result.returncode == 0, result.stderr
    context = injected_context(result)
    assert "BEGIN CRUMB" in context
    assert "/api/ledger" in context


def test_the_newest_crumb_wins(tmp_path):
    make_project(tmp_path, name="older.crumb", age_hours=5)
    make_project(tmp_path, name="newest.crumb")
    result = run_resume(tmp_path, {"source": "startup"})
    assert "newest.crumb" in result.stderr


def test_injected_context_warns_against_trusting_it_blindly(tmp_path):
    """A handoff describes a tree that may have moved since it was written."""
    make_project(tmp_path)
    context = injected_context(run_resume(tmp_path, {"source": "startup"}))
    assert "verify" in context.lower()
    assert "background" in context.lower()


def test_context_states_the_age_of_the_handoff(tmp_path):
    make_project(tmp_path, age_hours=30)
    context = injected_context(run_resume(tmp_path, {"source": "startup"}))
    assert "30 hours ago" in context


# ── which sources get a handoff ──────────────────────────────────────

def test_clear_never_injects(tmp_path):
    """`/clear` is an instruction. Restoring the context anyway overrides it."""
    make_project(tmp_path)
    result = run_resume(tmp_path, {"source": "clear"})
    assert result.returncode == 0
    assert result.stdout == "", "injected context into a session the user just cleared"


def test_resume_does_not_duplicate_restored_history(tmp_path):
    make_project(tmp_path)
    assert run_resume(tmp_path, {"source": "resume"}).stdout == ""


def test_compact_does_not_inject_by_default(tmp_path):
    """Compaction fires mid-session, but the newest crumb is from the last
    session that *ended* — the current one has no crumb yet. Injecting it
    would restore a different session's detail at the exact moment the model
    is most disoriented. `--on startup,compact` remains available."""
    make_project(tmp_path)
    assert run_resume(tmp_path, {"source": "compact"}).stdout == ""
    assert "BEGIN CRUMB" in injected_context(
        run_resume(tmp_path, {"source": "compact"}, "--on", "startup,compact")
    )


def test_sources_are_configurable(tmp_path):
    make_project(tmp_path)
    assert run_resume(tmp_path, {"source": "clear"}, "--on", "clear,startup").stdout != ""


# ── stdout is model context; nothing else may reach it ───────────────

def test_nothing_reaches_stdout_when_there_is_nothing_to_inject(tmp_path):
    """Every skip path must leave stdout completely empty.

    Anything printed here is added verbatim to the user's conversation. A
    friendly "no crumbs found" message on stdout would appear to the model as
    though the user had typed it.
    """
    (tmp_path / ".crumb").mkdir()
    cases = [
        ({"source": "startup"}, "empty .crumb dir"),
        ({"source": "clear"}, "excluded source"),
        ({"source": "resume"}, "excluded source"),
        ("not json at all", "malformed payload"),
        (None, "no payload"),
    ]
    for payload, label in cases:
        result = run_resume(tmp_path, payload)
        assert result.stdout == "", f"{label} wrote to stdout: {result.stdout!r}"
        assert result.returncode == 0, f"{label} exited {result.returncode}"


def test_diagnostics_go_to_stderr(tmp_path):
    make_project(tmp_path)
    result = run_resume(tmp_path, {"source": "startup"})
    assert "crumb resume:" in result.stderr
    assert "crumb resume:" not in result.stdout


def test_stdout_is_exactly_one_json_object(tmp_path):
    """Trailing chatter would corrupt the hook payload."""
    make_project(tmp_path)
    result = run_resume(tmp_path, {"source": "startup"})
    json.loads(result.stdout)  # raises if anything else was printed


# ── failure modes must never break the session ───────────────────────

def test_a_malformed_payload_never_fails_the_session(tmp_path):
    make_project(tmp_path)
    assert run_resume(tmp_path, "{not json", ).returncode == 0


def test_strict_surfaces_what_the_default_swallows(tmp_path):
    make_project(tmp_path)
    assert run_resume(tmp_path, "{not json", "--strict").returncode != 0


def test_missing_crumb_dir_is_not_an_error(tmp_path):
    result = run_resume(tmp_path, {"source": "startup"})
    assert result.returncode == 0
    assert result.stdout == ""


def test_stale_crumbs_are_not_injected(tmp_path):
    """A three-week-old handoff for unrelated work is noise, not context."""
    make_project(tmp_path, age_hours=24 * 21)
    result = run_resume(tmp_path, {"source": "startup"})
    assert result.stdout == ""
    assert "max-age" in result.stderr


def test_the_age_limit_can_be_disabled(tmp_path):
    make_project(tmp_path, age_hours=24 * 21)
    assert run_resume(tmp_path, {"source": "startup"}, "--max-age-hours", "0").stdout != ""


def test_an_empty_crumb_file_is_skipped(tmp_path):
    crumb_dir = tmp_path / ".crumb"
    crumb_dir.mkdir()
    (crumb_dir / "empty.crumb").write_text("   \n", encoding="utf-8")
    assert run_resume(tmp_path, {"source": "startup"}).stdout == ""


# ── age comes from the content, not the filesystem ───────────────────
# `git checkout` stamps every file with the checkout time. A handoff committed
# to a repository months ago therefore arrives on a fresh clone looking newly
# written — so an mtime-based staleness gate is blind to exactly the crumbs
# most likely to be stale.

def write_crumb(tmp_path: Path, *, captured: str | None, mtime_hours: float = 0.0) -> Path:
    crumb_dir = tmp_path / ".crumb"
    crumb_dir.mkdir(parents=True, exist_ok=True)
    text = CRUMB
    if captured is not None:
        text = text.replace("---", f"captured={captured}\n---", 1)
    target = crumb_dir / "session-a.crumb"
    target.write_text(text, encoding="utf-8")
    if mtime_hours:
        old = time.time() - mtime_hours * 3600
        os.utime(target, (old, old))
    return target


def _iso_hours_ago(hours: float) -> str:
    import datetime

    stamp = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_a_freshly_checked_out_but_old_handoff_is_still_stale(tmp_path):
    """The bug this fixes: mtime says 'now', the content says three weeks."""
    write_crumb(tmp_path, captured=_iso_hours_ago(24 * 21), mtime_hours=0.0)
    result = run_resume(tmp_path, {"source": "startup"})
    assert result.stdout == "", "injected a three-week-old handoff that had just been checked out"
    assert "max-age" in result.stderr


def test_a_recent_handoff_with_an_old_mtime_is_still_injected(tmp_path):
    """The converse: a rebase or archive extract can backdate a current file."""
    write_crumb(tmp_path, captured=_iso_hours_ago(2), mtime_hours=24 * 21)
    assert "BEGIN CRUMB" in injected_context(run_resume(tmp_path, {"source": "startup"}))


def test_the_age_basis_is_reported(tmp_path):
    """Which clock was trusted is part of the answer, not an implementation detail."""
    write_crumb(tmp_path, captured=_iso_hours_ago(1))
    assert "by captured header" in run_resume(tmp_path, {"source": "startup"}).stderr

    write_crumb(tmp_path, captured=None)
    assert "by file mtime" in run_resume(tmp_path, {"source": "startup"}).stderr


def test_an_unparseable_captured_header_falls_back_to_mtime(tmp_path):
    """A malformed stamp must not make resume refuse to work."""
    write_crumb(tmp_path, captured="not-a-date")
    result = run_resume(tmp_path, {"source": "startup"})
    assert "BEGIN CRUMB" in injected_context(result)
    assert "by file mtime" in result.stderr


def test_capture_records_when_the_handoff_was_made(tmp_path):
    """Without this the fix has nothing to read."""
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("\n".join(json.dumps(m) for m in [
        {"type": "user", "message": {"role": "user", "content": "the /api/ledger route 502s"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Traced it to pool exhaustion in src/db/pool.ts at 1200ms."}]}},
    ]), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CLI), "capture", "--transcript", str(transcript), "--dir", ".crumb"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    written = next((tmp_path / ".crumb").glob("*.crumb")).read_text(encoding="utf-8")
    import re

    assert re.search(r"^captured=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", written, re.MULTILINE)


# ── a session receives the handoff at most once ──────────────────────
# Not hypothetical: install.sh and the Claude Code plugin each register this
# hook, and a user with both gets two firings on the same SessionStart.

def test_the_same_session_is_never_injected_twice(tmp_path):
    make_project(tmp_path)
    payload = {"source": "startup", "session_id": "sess-aaaa"}
    first = run_resume(tmp_path, payload)
    assert "BEGIN CRUMB" in injected_context(first)

    second = run_resume(tmp_path, payload)
    assert second.stdout == "", "second firing for the same session injected again"
    assert "not injecting twice" in second.stderr
    assert second.returncode == 0


def test_a_new_session_injects_even_after_a_previous_one_did(tmp_path):
    make_project(tmp_path)
    run_resume(tmp_path, {"source": "startup", "session_id": "sess-aaaa"})
    result = run_resume(tmp_path, {"source": "startup", "session_id": "sess-bbbb"})
    assert "BEGIN CRUMB" in injected_context(result)


def test_a_payload_without_a_session_id_still_injects(tmp_path):
    """The dedupe guard keys on session_id; its absence must not disable resume."""
    make_project(tmp_path)
    assert "BEGIN CRUMB" in injected_context(run_resume(tmp_path, {"source": "startup"}))


# ── round trip with capture ──────────────────────────────────────────

def test_capture_then_resume_round_trips(tmp_path):
    """The loop that makes the format usable without anyone remembering it."""
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("\n".join(json.dumps(m) for m in [
        {"type": "user", "message": {"role": "user", "content": "the /api/ledger route 502s"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Traced it to pool exhaustion in src/db/pool.ts at 1200ms."}]}},
    ]), encoding="utf-8")

    captured = subprocess.run(
        [sys.executable, str(CLI), "capture", "--transcript", str(transcript), "--dir", ".crumb"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert captured.returncode == 0, captured.stderr

    context = injected_context(run_resume(tmp_path, {"source": "startup"}))
    assert "BEGIN CRUMB" in context
    assert "ledger" in context
