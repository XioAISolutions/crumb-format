"""Tests for cli/crumb.py — parsing, validation, CLI commands, and edge cases."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add cli/ to path so we can import crumb directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))
import crumb


# ── Fixtures ─────────────────────────────────────────────────────────

VALID_TASK = """\
BEGIN CRUMB
v=1.1
kind=task
title=Fix login bug
source=cursor.agent
---
[goal]
Fix the redirect loop.

[context]
- App uses JWT auth

[constraints]
- Don't change login UI
END CRUMB
"""

VALID_MEM = """\
BEGIN CRUMB
v=1.1
kind=mem
title=Prefs
source=human.notes
---
[consolidated]
- Prefers TypeScript
- No ORMs
END CRUMB
"""

VALID_MAP = """\
BEGIN CRUMB
v=1.1
kind=map
title=My project
source=human.notes
project=myapp
---
[project]
A REST API.

[modules]
- src/routes
- src/db
END CRUMB
"""


# ── parse_crumb ──────────────────────────────────────────────────────

class TestParseCrumb:
    def test_valid_task(self):
        result = crumb.parse_crumb(VALID_TASK)
        assert result["headers"]["kind"] == "task"
        assert result["headers"]["title"] == "Fix login bug"
        assert "goal" in result["sections"]
        assert "context" in result["sections"]
        assert "constraints" in result["sections"]

    def test_valid_mem(self):
        result = crumb.parse_crumb(VALID_MEM)
        assert result["headers"]["kind"] == "mem"
        assert "consolidated" in result["sections"]

    def test_valid_map(self):
        result = crumb.parse_crumb(VALID_MAP)
        assert result["headers"]["kind"] == "map"
        assert "project" in result["sections"]
        assert "modules" in result["sections"]

    def test_missing_begin_marker(self):
        with pytest.raises(ValueError, match="missing BEGIN CRUMB"):
            crumb.parse_crumb("v=1.1\nkind=task\n---\n[goal]\nDo stuff\nEND CRUMB")

    def test_missing_end_marker(self):
        with pytest.raises(ValueError, match="missing END CRUMB"):
            crumb.parse_crumb("BEGIN CRUMB\nv=1.1\nkind=task\nsource=test\n---\n[goal]\nDo stuff")

    def test_missing_separator(self):
        with pytest.raises(ValueError, match="missing header separator"):
            crumb.parse_crumb("BEGIN CRUMB\nv=1.1\nkind=task\nsource=test\n[goal]\nDo stuff\nEND CRUMB")

    def test_missing_required_header(self):
        with pytest.raises(ValueError, match="missing required header: source"):
            crumb.parse_crumb("BEGIN CRUMB\nv=1.1\nkind=task\n---\n[goal]\nDo\n[context]\nC\n[constraints]\nX\nEND CRUMB")

    def test_wrong_version(self):
        with pytest.raises(ValueError, match="unsupported version"):
            crumb.parse_crumb("BEGIN CRUMB\nv=2.0\nkind=task\nsource=test\n---\n[goal]\nDo\nEND CRUMB")

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="unknown kind"):
            crumb.parse_crumb("BEGIN CRUMB\nv=1.1\nkind=recipe\nsource=test\n---\n[goal]\nDo\nEND CRUMB")

    def test_missing_required_section(self):
        with pytest.raises(ValueError, match="missing required section"):
            crumb.parse_crumb("BEGIN CRUMB\nv=1.1\nkind=task\nsource=test\n---\n[goal]\nDo\nEND CRUMB")

    def test_empty_section(self):
        with pytest.raises(ValueError, match="section.*is empty"):
            crumb.parse_crumb("BEGIN CRUMB\nv=1.1\nkind=task\nsource=test\n---\n[goal]\n\n[context]\nC\n[constraints]\nX\nEND CRUMB")

    def test_leading_trailing_whitespace(self):
        padded = "\n\n" + VALID_TASK + "\n\n"
        result = crumb.parse_crumb(padded)
        assert result["headers"]["kind"] == "task"

    def test_windows_line_endings(self):
        text = VALID_TASK.replace("\n", "\r\n")
        result = crumb.parse_crumb(text)
        assert result["headers"]["kind"] == "task"

    def test_unknown_headers_ignored(self):
        text = VALID_TASK.replace("source=cursor.agent", "source=cursor.agent\ncustom_field=hello")
        result = crumb.parse_crumb(text)
        assert result["headers"]["custom_field"] == "hello"

    def test_unknown_sections_ignored(self):
        text = VALID_TASK.replace("END CRUMB", "[notes]\nSome notes\nEND CRUMB")
        result = crumb.parse_crumb(text)
        assert "notes" in result["sections"]


# ── render_crumb ─────────────────────────────────────────────────────

class TestRenderCrumb:
    def test_round_trip(self):
        parsed = crumb.parse_crumb(VALID_MEM)
        rendered = crumb.render_crumb(parsed["headers"], parsed["sections"])
        reparsed = crumb.parse_crumb(rendered)
        assert reparsed["headers"] == parsed["headers"]
        assert set(reparsed["sections"].keys()) == set(parsed["sections"].keys())


# ── normalize_entry ──────────────────────────────────────────────────

class TestNormalizeEntry:
    def test_strips_bullet(self):
        assert crumb.normalize_entry("- Prefers TypeScript") == "prefers typescript"

    def test_strips_double_bullet(self):
        assert crumb.normalize_entry("- - Prefers TypeScript") == "prefers typescript"

    def test_strips_asterisk(self):
        assert crumb.normalize_entry("* Prefers TypeScript") == "prefers typescript"

    def test_normalizes_whitespace(self):
        assert crumb.normalize_entry("-  Prefers   TypeScript  ") == "prefers typescript"

    def test_case_insensitive(self):
        assert crumb.normalize_entry("PREFERS TYPESCRIPT") == crumb.normalize_entry("prefers typescript")

    def test_empty_string(self):
        assert crumb.normalize_entry("") == ""


# ── estimate_tokens ──────────────────────────────────────────────────

class TestEstimateTokens:
    def test_short_text(self):
        assert crumb.estimate_tokens("hello world") > 0

    def test_empty(self):
        assert crumb.estimate_tokens("") == 0

    def test_proportional(self):
        short = crumb.estimate_tokens("hello")
        long = crumb.estimate_tokens("hello " * 100)
        assert long > short


# ── cmd_new (via main) ───────────────────────────────────────────────

class TestCmdNew:
    def test_new_task(self, capsys):
        crumb.main(["new", "task", "--title", "Test", "--source", "test", "--goal", "Do something"])
        output = capsys.readouterr().out
        assert "BEGIN CRUMB" in output
        assert "kind=task" in output
        assert "Do something" in output

    def test_new_mem(self, capsys):
        crumb.main(["new", "mem", "--title", "Prefs", "--source", "test", "--entries", "TypeScript", "No ORMs"])
        output = capsys.readouterr().out
        assert "kind=mem" in output
        assert "- TypeScript" in output
        assert "- No ORMs" in output

    def test_new_map(self, capsys):
        crumb.main(["new", "map", "--title", "API", "--source", "test", "--project", "myapp", "--modules", "src/", "tests/"])
        output = capsys.readouterr().out
        assert "kind=map" in output
        assert "project=myapp" in output
        assert "- src/" in output


# ── cmd_validate ─────────────────────────────────────────────────────

class TestCmdValidate:
    def test_validate_valid(self, tmp_path):
        f = tmp_path / "test.crumb"
        f.write_text(VALID_TASK)
        with pytest.raises(SystemExit) as exc:
            crumb.main(["validate", str(f)])
        assert exc.value.code == 0

    def test_validate_invalid(self, tmp_path):
        f = tmp_path / "bad.crumb"
        f.write_text("not a crumb")
        with pytest.raises(SystemExit) as exc:
            crumb.main(["validate", str(f)])
        assert exc.value.code == 1


# ── cmd_inspect ──────────────────────────────────────────────────────

class TestCmdInspect:
    def test_inspect(self, tmp_path, capsys):
        f = tmp_path / "test.crumb"
        f.write_text(VALID_TASK)
        crumb.main(["inspect", str(f)])
        output = capsys.readouterr().out
        assert "kind = task" in output
        assert "[goal]" in output

    def test_inspect_headers_only(self, tmp_path, capsys):
        f = tmp_path / "test.crumb"
        f.write_text(VALID_TASK)
        crumb.main(["inspect", str(f), "--headers-only"])
        output = capsys.readouterr().out
        assert "kind = task" in output
        assert "Fix the redirect loop" not in output


# ── cmd_append ───────────────────────────────────────────────────────

class TestCmdAppend:
    def test_append_creates_raw_section(self, tmp_path):
        f = tmp_path / "mem.crumb"
        f.write_text(VALID_MEM)
        crumb.main(["append", str(f), "New preference", "Another pref"])
        text = f.read_text()
        assert "[raw]" in text
        assert "New preference" in text
        assert "Another pref" in text

    def test_append_rejects_non_mem(self, tmp_path):
        f = tmp_path / "task.crumb"
        f.write_text(VALID_TASK)
        with pytest.raises(SystemExit) as exc:
            crumb.main(["append", str(f), "Some entry"])
        assert exc.value.code == 1


# ── cmd_dream ────────────────────────────────────────────────────────

class TestCmdDream:
    def test_dream_merges_raw(self, tmp_path):
        f = tmp_path / "mem.crumb"
        f.write_text(VALID_MEM)
        crumb.main(["append", str(f), "Likes Rust"])
        crumb.main(["dream", str(f)])
        text = f.read_text()
        assert "[raw]" not in text
        assert "Likes Rust" in text
        assert "dream_pass=" in text
        assert "dream_sessions=" in text

    def test_dream_deduplicates(self, tmp_path):
        f = tmp_path / "mem.crumb"
        f.write_text(VALID_MEM)
        crumb.main(["append", str(f), "Prefers TypeScript", "prefers typescript"])
        crumb.main(["dream", str(f)])
        text = f.read_text()
        # Should only have one instance after dedup
        consolidated = text.split("[consolidated]")[1].split("[dream]")[0]
        ts_count = consolidated.lower().count("prefers typescript")
        assert ts_count == 1

    def test_dream_dry_run_no_write(self, tmp_path, capsys):
        f = tmp_path / "mem.crumb"
        f.write_text(VALID_MEM)
        original = f.read_text()
        crumb.main(["dream", str(f), "--dry-run"])
        assert f.read_text() == original
        output = capsys.readouterr().out
        assert "BEGIN CRUMB" in output


# ── cmd_search ───────────────────────────────────────────────────────

class TestCmdSearch:
    def test_search_finds_match(self, tmp_path, capsys):
        f = tmp_path / "test.crumb"
        f.write_text(VALID_TASK)
        crumb.main(["search", "JWT auth", "--dir", str(tmp_path)])
        output = capsys.readouterr().out
        assert "test.crumb" in output

    def test_search_no_match(self, tmp_path, capsys):
        f = tmp_path / "test.crumb"
        f.write_text(VALID_TASK)
        crumb.main(["search", "kubernetes helm", "--dir", str(tmp_path)])
        output = capsys.readouterr().out
        assert "No matches" in output


# ── cmd_merge ────────────────────────────────────────────────────────

class TestCmdMerge:
    def test_merge_two_mems(self, tmp_path, capsys):
        f1 = tmp_path / "a.crumb"
        f2 = tmp_path / "b.crumb"
        f1.write_text(VALID_MEM)
        f2.write_text(VALID_MEM.replace("Prefers TypeScript", "Likes Rust").replace("No ORMs", "Uses Postgres"))
        crumb.main(["merge", str(f1), str(f2)])
        output = capsys.readouterr().out
        parsed = crumb.parse_crumb(output)
        entries = [l.strip() for l in parsed["sections"]["consolidated"] if l.strip()]
        assert len(entries) == 4  # TypeScript, No ORMs, Rust, Postgres (no dupes)

    def test_merge_deduplicates(self, tmp_path, capsys):
        f1 = tmp_path / "a.crumb"
        f2 = tmp_path / "b.crumb"
        f1.write_text(VALID_MEM)
        f2.write_text(VALID_MEM)  # identical
        crumb.main(["merge", str(f1), str(f2)])
        output = capsys.readouterr().out
        parsed = crumb.parse_crumb(output)
        entries = [l.strip() for l in parsed["sections"]["consolidated"] if l.strip()]
        assert len(entries) == 2  # deduplicated


# ── cmd_init ─────────────────────────────────────────────────────────

class TestCmdInit:
    def test_init_creates_map(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "README.md").touch()
        crumb.main(["init", "--dir", str(tmp_path), "--project", "testapp"])
        map_path = tmp_path / "crumbs" / "map.crumb"
        assert map_path.exists()
        parsed = crumb.parse_crumb(map_path.read_text())
        assert parsed["headers"]["kind"] == "map"
        assert parsed["headers"]["project"] == "testapp"

    def test_init_claude_md(self, tmp_path):
        (tmp_path / "src").mkdir()
        crumb.main(["init", "--dir", str(tmp_path), "--claude-md"])
        claude_md = tmp_path / "CLAUDE.md"
        assert claude_md.exists()
        assert "CRUMB" in claude_md.read_text()


# ── Validate all repo examples ───────────────────────────────────────

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("example", sorted(EXAMPLES_DIR.glob("*.crumb")))
def test_example_validates(example):
    text = example.read_text(encoding="utf-8")
    result = crumb.parse_crumb(text)
    assert "headers" in result
    assert "sections" in result
