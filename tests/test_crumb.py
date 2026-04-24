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

class TestCliVersion:
    def test_version_matches_release(self, capsys):
        with pytest.raises(SystemExit) as exc:
            crumb.main(["--version"])
        assert exc.value.code == 0
        assert capsys.readouterr().out.strip() == "crumb 0.6.0"


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


# ── crumb init (multi-tool seeding) ──────────────────────────────────

class TestCmdInitMultiTool:
    def test_init_cursor_rules(self, tmp_path, capsys):
        crumb.main(["init", "--dir", str(tmp_path), "--cursor-rules"])
        rules_path = tmp_path / ".cursor" / "rules"
        assert rules_path.exists()
        content = rules_path.read_text()
        assert "CRUMB" in content
        assert "crumb it" in content

    def test_init_windsurf_rules(self, tmp_path, capsys):
        crumb.main(["init", "--dir", str(tmp_path), "--windsurf-rules"])
        rules_path = tmp_path / ".windsurfrules"
        assert rules_path.exists()
        content = rules_path.read_text()
        assert "CRUMB" in content
        assert "windsurf.agent" in content

    def test_init_chatgpt_rules(self, tmp_path, capsys):
        crumb.main(["init", "--dir", str(tmp_path), "--chatgpt-rules"])
        output = capsys.readouterr().out
        assert "ChatGPT" in output
        assert "crumb it" in output
        assert "source=chatgpt" in output

    def test_init_all(self, tmp_path, capsys):
        crumb.main(["init", "--dir", str(tmp_path), "--all"])
        assert (tmp_path / "CLAUDE.md").exists()
        assert (tmp_path / ".cursor" / "rules").exists()
        assert (tmp_path / ".windsurfrules").exists()
        output = capsys.readouterr().out
        assert "ChatGPT" in output

    def test_init_idempotent(self, tmp_path, capsys):
        crumb.main(["init", "--dir", str(tmp_path), "--claude-md"])
        capsys.readouterr()  # clear
        crumb.main(["init", "--dir", str(tmp_path), "--claude-md"])
        output = capsys.readouterr().out
        assert "already has CRUMB section" in output

    def test_init_tip_shown(self, tmp_path, capsys):
        crumb.main(["init", "--dir", str(tmp_path)])
        output = capsys.readouterr().out
        assert "--all" in output


# ── MCP server ───────────────────────────────────────────────────────

class TestMCPServer:
    def test_mcp_import(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp"))
        import server as mcp_server
        assert hasattr(mcp_server, 'TOOLS')
        assert len(mcp_server.TOOLS) >= 10

    def test_mcp_tool_call_new(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp"))
        import server as mcp_server
        result = mcp_server.handle_tool_call("crumb_new", {"kind": "task", "title": "Test"})
        assert "BEGIN CRUMB" in result
        assert "kind=task" in result

    def test_mcp_tool_call_template_list(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp"))
        import server as mcp_server
        result = mcp_server.handle_tool_call("crumb_template", {"action": "list"})
        assert "bug-fix" in result

    def test_mcp_tool_call_validate(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp"))
        import server as mcp_server
        f = tmp_path / "task.crumb"
        f.write_text(VALID_TASK)
        result = mcp_server.handle_tool_call("crumb_validate", {"files": [str(f)]})
        assert "OK" in result

    def test_mcp_tool_call_export(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp"))
        import server as mcp_server
        f = tmp_path / "task.crumb"
        f.write_text(VALID_TASK)
        result = mcp_server.handle_tool_call("crumb_export", {"file": str(f), "format": "json"})
        import json
        obj = json.loads(result)
        assert obj["headers"]["kind"] == "task"


# ── kind=log ─────────────────────────────────────────────────────────

VALID_LOG = """\
BEGIN CRUMB
v=1.1
kind=log
title=Debug session
source=cli
---
[entries]
- [2026-03-28T12:00:00Z] Started debugging
END CRUMB
"""

VALID_TODO = """\
BEGIN CRUMB
v=1.1
kind=todo
title=Sprint tasks
source=cli
---
[tasks]
- [ ] Fix auth bug
- [ ] Add tests
- [x] Write spec
END CRUMB
"""


class TestNewKinds:
    def test_parse_log(self):
        result = crumb.parse_crumb(VALID_LOG)
        assert result["headers"]["kind"] == "log"
        assert "entries" in result["sections"]

    def test_parse_todo(self):
        result = crumb.parse_crumb(VALID_TODO)
        assert result["headers"]["kind"] == "todo"
        assert "tasks" in result["sections"]

    def test_new_log(self, tmp_path, capsys):
        out = tmp_path / "session.crumb"
        crumb.main(["new", "log", "-t", "My session", "-s", "cli",
                     "-e", "Started work", "-o", str(out)])
        result = crumb.parse_crumb(out.read_text())
        assert result["headers"]["kind"] == "log"
        entries = [l for l in result["sections"]["entries"] if l.strip()]
        assert any("Started work" in e for e in entries)

    def test_new_todo(self, tmp_path, capsys):
        out = tmp_path / "tasks.crumb"
        crumb.main(["new", "todo", "-t", "My tasks", "-s", "cli",
                     "-e", "Do something", "-o", str(out)])
        result = crumb.parse_crumb(out.read_text())
        assert result["headers"]["kind"] == "todo"
        tasks = [l for l in result["sections"]["tasks"] if l.strip()]
        assert any("Do something" in t for t in tasks)


class TestCmdLog:
    def test_log_creates_file(self, tmp_path, capsys):
        f = tmp_path / "session.crumb"
        crumb.main(["log", str(f), "First entry", "Second entry"])
        assert f.exists()
        result = crumb.parse_crumb(f.read_text())
        assert result["headers"]["kind"] == "log"
        entries = [l for l in result["sections"]["entries"] if l.strip()]
        assert len(entries) == 2

    def test_log_appends(self, tmp_path, capsys):
        f = tmp_path / "session.crumb"
        f.write_text(VALID_LOG)
        crumb.main(["log", str(f), "New entry"])
        result = crumb.parse_crumb(f.read_text())
        entries = [l for l in result["sections"]["entries"] if l.strip()]
        assert len(entries) == 2
        assert any("New entry" in e for e in entries)

    def test_log_rejects_non_log(self, tmp_path, capsys):
        f = tmp_path / "mem.crumb"
        f.write_text(VALID_MEM)
        with pytest.raises(SystemExit) as exc:
            crumb.main(["log", str(f), "Entry"])
        assert exc.value.code == 1


class TestCmdTodo:
    def test_todo_add_creates_file(self, tmp_path, capsys):
        f = tmp_path / "tasks.crumb"
        crumb.main(["todo-add", str(f), "Build feature", "Write tests"])
        result = crumb.parse_crumb(f.read_text())
        tasks = [l.strip() for l in result["sections"]["tasks"] if l.strip()]
        assert sum(1 for t in tasks if t.startswith("- [ ]")) == 2

    def test_todo_done(self, tmp_path, capsys):
        f = tmp_path / "tasks.crumb"
        f.write_text(VALID_TODO)
        crumb.main(["todo-done", str(f), "auth"])
        result = crumb.parse_crumb(f.read_text())
        tasks = [l.strip() for l in result["sections"]["tasks"] if l.strip()]
        # "Fix auth bug" should now be [x]
        auth_tasks = [t for t in tasks if "auth" in t.lower()]
        assert all("[x]" in t for t in auth_tasks)

    def test_todo_done_no_match(self, tmp_path, capsys):
        f = tmp_path / "tasks.crumb"
        f.write_text(VALID_TODO)
        crumb.main(["todo-done", str(f), "nonexistent_xyz"])
        output = capsys.readouterr().out
        assert "No open tasks" in output

    def test_todo_list(self, tmp_path, capsys):
        f = tmp_path / "tasks.crumb"
        f.write_text(VALID_TODO)
        crumb.main(["todo-list", str(f)])
        output = capsys.readouterr().out
        assert "2 open" in output
        assert "1 done" in output

    def test_todo_list_all(self, tmp_path, capsys):
        f = tmp_path / "tasks.crumb"
        f.write_text(VALID_TODO)
        crumb.main(["todo-list", str(f), "--all"])
        output = capsys.readouterr().out
        assert "[x] Write spec" in output

    def test_todo_dream_archives(self, tmp_path, capsys):
        f = tmp_path / "tasks.crumb"
        f.write_text(VALID_TODO)
        crumb.main(["todo-dream", str(f)])
        result = crumb.parse_crumb(f.read_text())
        # [x] Write spec should be in [archived]
        assert "archived" in result["sections"]
        archived = [l.strip() for l in result["sections"]["archived"] if l.strip()]
        assert any("Write spec" in a for a in archived)
        # Open tasks remain in [tasks]
        tasks = [l.strip() for l in result["sections"]["tasks"] if l.strip()]
        assert not any("[x]" in t for t in tasks)

    def test_todo_dream_nothing_to_archive(self, tmp_path, capsys):
        todo = VALID_TODO.replace("- [x] Write spec\n", "")
        f = tmp_path / "tasks.crumb"
        f.write_text(todo)
        crumb.main(["todo-dream", str(f)])
        output = capsys.readouterr().out
        assert "No completed tasks" in output


# ── parse_chat_lines ─────────────────────────────────────────────────

class TestParseChatLines:
    def test_detects_code_blocks(self):
        chat = "user: show me code\n```python\ndef hello():\n    pass\n```\nuser: thanks"
        user, ai, code, decisions = crumb.parse_chat_lines(chat)
        assert len(code) == 1
        assert code[0]['lang'] == 'python'
        assert 'def hello' in code[0]['code']

    def test_extracts_decisions(self):
        chat = "claude: Decided to use PostgreSQL.\nclaude: Going with Redis for caching."
        user, ai, code, decisions = crumb.parse_chat_lines(chat)
        assert len(decisions) == 2
        assert any('PostgreSQL' in d for d in decisions)
        assert any('Redis' in d for d in decisions)

    def test_separates_user_and_ai(self):
        chat = "user: hello\nclaude: hi there\nuser: bye"
        user, ai, code, decisions = crumb.parse_chat_lines(chat)
        assert len(user) == 2
        assert len(ai) == 1

    def test_handles_multiline_code_blocks(self):
        chat = "```js\nconst x = 1;\nconst y = 2;\nreturn x + y;\n```"
        user, ai, code, decisions = crumb.parse_chat_lines(chat)
        assert len(code) == 1
        assert 'const x = 1' in code[0]['code']
        assert code[0]['lang'] == 'js'

    def test_no_decisions_found(self):
        chat = "user: just chatting\nclaude: hello there"
        user, ai, code, decisions = crumb.parse_chat_lines(chat)
        assert decisions == []


class TestFromChatSmart:
    def test_from_chat_extracts_decisions(self, tmp_path, capsys):
        chat_file = tmp_path / "chat.txt"
        chat_file.write_text("claude: Decided to use React.\nclaude: Switched to TypeScript.")
        crumb.main(["from-chat", "-i", str(chat_file)])
        output = capsys.readouterr().out
        assert "Decisions made:" in output

    def test_from_chat_extracts_code_blocks(self, tmp_path, capsys):
        chat_file = tmp_path / "chat.txt"
        chat_file.write_text("user: show code\n```python\ndef foo(): pass\n```\nuser: ok")
        crumb.main(["from-chat", "-i", str(chat_file)])
        output = capsys.readouterr().out
        assert "Code discussed" in output
        assert "(python)" in output

    def test_from_chat_mem_kind(self, tmp_path, capsys):
        chat_file = tmp_path / "chat.txt"
        chat_file.write_text("claude: Decided to use Postgres.\nclaude: Switched to Prisma ORM.")
        crumb.main(["from-chat", "-i", str(chat_file), "--kind", "mem", "--title", "Decisions"])
        output = capsys.readouterr().out
        parsed = crumb.parse_crumb(output)
        assert parsed["headers"]["kind"] == "mem"
        entries = [l.strip() for l in parsed["sections"]["consolidated"] if l.strip()]
        assert len(entries) >= 2

    def test_from_chat_still_works_plain(self, tmp_path, capsys):
        chat_file = tmp_path / "chat.txt"
        chat_file.write_text("user: fix the bug\nclaude: I fixed it")
        crumb.main(["from-chat", "-i", str(chat_file), "--title", "Bug fix"])
        output = capsys.readouterr().out
        assert "BEGIN CRUMB" in output
        assert "kind=task" in output


# ── search modes ─────────────────────────────────────────────────────

class TestSearchModes:
    @pytest.fixture()
    def search_dir(self, tmp_path):
        """Create a directory with several crumb files for search testing."""
        (tmp_path / "auth.crumb").write_text(VALID_TASK)
        (tmp_path / "prefs.crumb").write_text(VALID_MEM)
        (tmp_path / "project.crumb").write_text(VALID_MAP)
        return tmp_path

    def test_keyword_search(self, search_dir, capsys):
        crumb.main(["search", "TypeScript", "--dir", str(search_dir)])
        output = capsys.readouterr().out
        assert "prefs.crumb" in output

    def test_keyword_no_match(self, search_dir, capsys):
        crumb.main(["search", "nonexistent_xyz", "--dir", str(search_dir)])
        output = capsys.readouterr().out
        assert "No matches" in output

    def test_fuzzy_search_finds_approximate(self, search_dir, capsys):
        # "Typscript" (misspelled) should fuzzy-match "TypeScript"
        crumb.main(["search", "Typscript", "--dir", str(search_dir), "--method", "fuzzy"])
        output = capsys.readouterr().out
        assert "prefs.crumb" in output

    def test_ranked_search(self, search_dir, capsys):
        crumb.main(["search", "TypeScript", "--dir", str(search_dir), "--method", "ranked"])
        output = capsys.readouterr().out
        assert "prefs.crumb" in output

    def test_fuzzy_search_no_match(self, search_dir, capsys):
        crumb.main(["search", "zzzzxxxxxqqqq", "--dir", str(search_dir), "--method", "fuzzy"])
        output = capsys.readouterr().out
        assert "No matches" in output


# ── cmd_export ────────────────────────────────────────────────────────

class TestCmdExport:
    def test_export_json(self, tmp_path, capsys):
        f = tmp_path / "task.crumb"
        f.write_text(VALID_TASK)
        crumb.main(["export", str(f), "-f", "json"])
        output = capsys.readouterr().out
        import json
        obj = json.loads(output)
        assert obj["headers"]["kind"] == "task"
        assert "goal" in obj["sections"]

    def test_export_markdown(self, tmp_path, capsys):
        f = tmp_path / "task.crumb"
        f.write_text(VALID_TASK)
        crumb.main(["export", str(f), "-f", "markdown"])
        output = capsys.readouterr().out
        assert "# Fix login bug" in output
        assert "| kind | task |" in output

    def test_export_clipboard(self, tmp_path, capsys):
        f = tmp_path / "task.crumb"
        f.write_text(VALID_TASK)
        crumb.main(["export", str(f), "-f", "clipboard"])
        output = capsys.readouterr().out
        assert "[CRUMB handoff" in output
        assert "Goal:" in output
        assert "github.com/XioAISolutions/crumb-format" in output

    def test_export_to_file(self, tmp_path, capsys):
        f = tmp_path / "mem.crumb"
        f.write_text(VALID_MEM)
        out = tmp_path / "mem.json"
        crumb.main(["export", str(f), "-f", "json", "-o", str(out)])
        import json
        obj = json.loads(out.read_text())
        assert obj["headers"]["kind"] == "mem"

    def test_export_mem_clipboard(self, tmp_path, capsys):
        f = tmp_path / "mem.crumb"
        f.write_text(VALID_MEM)
        crumb.main(["export", str(f), "-f", "clipboard"])
        output = capsys.readouterr().out
        assert "Known facts:" in output

    def test_export_map_clipboard(self, tmp_path, capsys):
        f = tmp_path / "map.crumb"
        f.write_text(VALID_MAP)
        crumb.main(["export", str(f), "-f", "clipboard"])
        output = capsys.readouterr().out
        assert "Key modules:" in output


# ── cmd_import ────────────────────────────────────────────────────────

class TestCmdImport:
    def test_import_json_round_trip(self, tmp_path, capsys):
        f = tmp_path / "task.crumb"
        f.write_text(VALID_TASK)
        # Export to JSON
        j = tmp_path / "task.json"
        crumb.main(["export", str(f), "-f", "json", "-o", str(j)])
        # Import back
        out = tmp_path / "reimported.crumb"
        crumb.main(["import", "--from", "json", "-i", str(j), "-o", str(out)])
        # Validate the result
        result = crumb.parse_crumb(out.read_text())
        assert result["headers"]["kind"] == "task"
        assert "goal" in result["sections"]

    def test_import_markdown_round_trip(self, tmp_path, capsys):
        f = tmp_path / "mem.crumb"
        f.write_text(VALID_MEM)
        # Export to markdown
        md = tmp_path / "mem.md"
        crumb.main(["export", str(f), "-f", "markdown", "-o", str(md)])
        # Import back
        out = tmp_path / "reimported.crumb"
        crumb.main(["import", "--from", "markdown", "-i", str(md), "-o", str(out)])
        result = crumb.parse_crumb(out.read_text())
        assert result["headers"]["kind"] == "mem"
        assert "consolidated" in result["sections"]

    def test_import_json_missing_kind(self, tmp_path, capsys):
        j = tmp_path / "bad.json"
        j.write_text('{"headers": {"v": "1.1"}, "sections": {}}')
        with pytest.raises(SystemExit) as exc:
            crumb.main(["import", "--from", "json", "-i", str(j)])
        assert exc.value.code == 1


# ── hooks ────────────────────────────────────────────────────────────

class TestHooks:
    def test_load_hooks_no_file(self, tmp_path):
        hooks = crumb.load_hooks(str(tmp_path))
        assert hooks == {}

    def test_load_hooks_with_file(self, tmp_path):
        rc = tmp_path / ".crumbrc"
        rc.write_text("[hooks]\npost_dream = echo done\npost_append = echo added\n")
        hooks = crumb.load_hooks(str(tmp_path))
        assert hooks["post_dream"] == "echo done"
        assert hooks["post_append"] == "echo added"

    def test_load_hooks_ignores_comments(self, tmp_path):
        rc = tmp_path / ".crumbrc"
        rc.write_text("[hooks]\n# this is a comment\npost_dream = echo done\n")
        hooks = crumb.load_hooks(str(tmp_path))
        assert len(hooks) == 1

    def test_load_hooks_multiple_sections(self, tmp_path):
        rc = tmp_path / ".crumbrc"
        rc.write_text("[settings]\nfoo = bar\n[hooks]\npost_dream = echo ok\n[other]\nx = y\n")
        hooks = crumb.load_hooks(str(tmp_path))
        assert hooks == {"post_dream": "echo ok"}

    def test_run_hook_not_defined(self):
        assert crumb.run_hook('nonexistent_hook') is True

    def test_cmd_hooks_no_config(self, capsys):
        crumb.main(["hooks", "--dir", "/tmp"])
        output = capsys.readouterr().out
        assert "No hooks configured" in output


# ── templates ────────────────────────────────────────────────────────

class TestTemplates:
    def test_template_list(self, capsys):
        crumb.main(["template", "list"])
        output = capsys.readouterr().out
        assert "bug-fix" in output
        assert "feature" in output
        assert "onboarding" in output

    def test_template_use_builtin(self, capsys):
        crumb.main(["template", "use", "bug-fix"])
        output = capsys.readouterr().out
        assert "BEGIN CRUMB" in output
        assert "kind=task" in output
        # Must be a valid crumb
        crumb.parse_crumb(output)

    def test_template_use_to_file(self, tmp_path, capsys):
        out = tmp_path / "new.crumb"
        crumb.main(["template", "use", "preferences", "-o", str(out)])
        result = crumb.parse_crumb(out.read_text())
        assert result["headers"]["kind"] == "mem"

    def test_template_use_unknown(self):
        with pytest.raises(SystemExit) as exc:
            crumb.main(["template", "use", "nonexistent-template"])
        assert exc.value.code == 1

    def test_template_add_and_use(self, tmp_path, capsys, monkeypatch):
        # Override template dir to tmp
        monkeypatch.setattr(crumb, 'TEMPLATE_DIR', tmp_path / 'templates')
        # Create a valid crumb to use as template
        src = tmp_path / "my.crumb"
        src.write_text(VALID_MEM)
        crumb.main(["template", "add", "my-prefs", str(src)])
        # Now use it
        crumb.main(["template", "use", "my-prefs"])
        output = capsys.readouterr().out
        assert "Prefers TypeScript" in output

    def test_all_builtin_templates_valid(self):
        for name, tmpl in crumb.BUILTIN_TEMPLATES.items():
            parsed = crumb.parse_crumb(tmpl['content'])
            assert parsed['headers']['kind'] == tmpl['kind'], f"Template '{name}' kind mismatch"


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
