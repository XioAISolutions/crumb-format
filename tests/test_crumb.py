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


# ── cmd_diff ─────────────────────────────────────────────────────────

class TestCmdDiff:
    def test_diff_identical(self, tmp_path, capsys):
        f1 = tmp_path / "a.crumb"
        f2 = tmp_path / "b.crumb"
        f1.write_text(VALID_MEM)
        f2.write_text(VALID_MEM)
        crumb.main(["diff", str(f1), str(f2)])
        output = capsys.readouterr().out
        assert "No differences" in output

    def test_diff_added_entries(self, tmp_path, capsys):
        f1 = tmp_path / "a.crumb"
        f2 = tmp_path / "b.crumb"
        f1.write_text(VALID_MEM)
        f2.write_text(VALID_MEM)
        crumb.main(["append", str(f2), "Likes Rust"])
        crumb.main(["dream", str(f2)])
        crumb.main(["diff", str(f1), str(f2)])
        output = capsys.readouterr().out
        assert "+ " in output
        assert "likes rust" in output.lower()

    def test_diff_removed_entries(self, tmp_path, capsys):
        f1 = tmp_path / "a.crumb"
        f2 = tmp_path / "b.crumb"
        f1.write_text(VALID_MEM)
        # f2 has only one entry instead of two
        f2.write_text(VALID_MEM.replace("- No ORMs\n", ""))
        crumb.main(["diff", str(f1), str(f2)])
        output = capsys.readouterr().out
        assert "- " in output
        assert "change(s)" in output


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


# ── extract_keywords ─────────────────────────────────────────────────

class TestExtractKeywords:
    def test_removes_stopwords(self):
        kw = crumb.extract_keywords("Use the TypeScript compiler for building")
        assert "typescript" in kw
        assert "compiler" in kw
        assert "the" not in kw
        assert "for" not in kw

    def test_empty_string(self):
        assert crumb.extract_keywords("") == set()

    def test_only_stopwords(self):
        assert crumb.extract_keywords("the is a an") == set()

    def test_technical_terms(self):
        kw = crumb.extract_keywords("PostgreSQL v14 with pgvector")
        assert "postgresql" in kw
        assert "pgvector" in kw


# ── score_entry ──────────────────────────────────────────────────────

class TestScoreEntry:
    def test_unique_entry_scores_higher(self):
        entries = ["- Use postgres", "- Use postgres", "- Use redis"]
        kw = {crumb.normalize_entry(e): crumb.extract_keywords(e) for e in entries}
        pg_score = crumb.score_entry(entries[0], entries, kw)
        redis_score = crumb.score_entry(entries[2], entries, kw)
        # "redis" is unique (appears once), "postgres" appears twice
        assert redis_score > pg_score

    def test_technical_entry_gets_bonus(self):
        entries = ["- Use v14 of src/db", "- Keep it simple"]
        kw = {crumb.normalize_entry(e): crumb.extract_keywords(e) for e in entries}
        tech_score = crumb.score_entry(entries[0], entries, kw)
        plain_score = crumb.score_entry(entries[1], entries, kw)
        assert tech_score > plain_score

    def test_empty_entry_scores_zero(self):
        entries = ["", "- Something"]
        kw = {crumb.normalize_entry(e): crumb.extract_keywords(e) for e in entries}
        assert crumb.score_entry("", entries, kw) == 0.0


# ── cmd_compact ──────────────────────────────────────────────────────

class TestCmdCompact:
    def test_compact_strips_optional_headers(self, tmp_path, capsys):
        f = tmp_path / "mem.crumb"
        f.write_text(VALID_MEM)
        out = tmp_path / "compact.crumb"
        crumb.main(["compact", str(f), "-o", str(out)])
        result = crumb.parse_crumb(out.read_text())
        # Should keep v, kind, source, title but not extras
        assert "v" in result["headers"]
        assert "kind" in result["headers"]
        assert "title" in result["headers"]
        assert "source" in result["headers"]

    def test_compact_keeps_required_sections_only(self, tmp_path, capsys):
        # Add an extra section to a mem crumb
        text = VALID_MEM.replace("END CRUMB", "[notes]\nSome notes\nEND CRUMB")
        f = tmp_path / "mem.crumb"
        f.write_text(text)
        out = tmp_path / "compact.crumb"
        crumb.main(["compact", str(f), "-o", str(out)])
        result = crumb.parse_crumb(out.read_text())
        assert "consolidated" in result["sections"]
        assert "notes" not in result["sections"]

    def test_compact_reduces_tokens(self, tmp_path, capsys):
        f = tmp_path / "mem.crumb"
        # Use the dogfood mem which has dream headers and dream section
        dogfood = (Path(__file__).resolve().parent.parent / "crumbs" / "mem.crumb").read_text()
        f.write_text(dogfood)
        out = tmp_path / "compact.crumb"
        crumb.main(["compact", str(f), "-o", str(out)])
        output = capsys.readouterr().out
        assert "reduction" in output
        original_tokens = crumb.estimate_tokens(dogfood)
        compact_tokens = crumb.estimate_tokens(out.read_text())
        assert compact_tokens < original_tokens

    def test_compact_stdout(self, tmp_path, capsys):
        f = tmp_path / "mem.crumb"
        f.write_text(VALID_MEM)
        crumb.main(["compact", str(f)])
        output = capsys.readouterr().out
        assert "BEGIN CRUMB" in output

    def test_compact_map_keeps_project_header(self, tmp_path, capsys):
        f = tmp_path / "map.crumb"
        f.write_text(VALID_MAP)
        out = tmp_path / "compact.crumb"
        crumb.main(["compact", str(f), "-o", str(out)])
        result = crumb.parse_crumb(out.read_text())
        assert result["headers"].get("project") == "myapp"


# ── signal-scored dream pruning ──────────────────────────────────────

class TestDreamSignalPruning:
    def test_dream_keeps_unique_entries_over_duplicates(self, tmp_path):
        """When pruning is needed, unique high-signal entries survive."""
        f = tmp_path / "mem.crumb"
        f.write_text(VALID_MEM)
        # Append several entries including duplicates
        crumb.main(["append", str(f), "Use PostgreSQL v14", "Use PostgreSQL v14", "Keep it simple"])
        crumb.main(["dream", str(f)])
        text = f.read_text()
        parsed = crumb.parse_crumb(text)
        entries = [l.strip() for l in parsed["sections"]["consolidated"] if l.strip()]
        # PostgreSQL should appear only once (deduped), and all unique entries kept
        pg_count = sum(1 for e in entries if "postgresql" in e.lower())
        assert pg_count == 1


# ── Validate all repo examples ───────────────────────────────────────

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
CRUMBS_DIR = Path(__file__).resolve().parent.parent / "crumbs"


@pytest.mark.parametrize("example", sorted(EXAMPLES_DIR.glob("*.crumb")))
def test_example_validates(example):
    text = example.read_text(encoding="utf-8")
    result = crumb.parse_crumb(text)
    assert "headers" in result
    assert "sections" in result


@pytest.mark.parametrize("crumb_file", sorted(CRUMBS_DIR.glob("*.crumb")))
def test_dogfood_crumb_validates(crumb_file):
    text = crumb_file.read_text(encoding="utf-8")
    result = crumb.parse_crumb(text)
    assert "headers" in result
    assert "sections" in result
