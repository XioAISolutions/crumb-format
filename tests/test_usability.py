"""Tests for v0.7 usability + simplicity changes.

Covers: template v=1.3 emission, deprecated alias shims, optimize modes,
unified exit codes, unknown-kind enumeration, and the AgentAuth first-use
notice.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import crumb


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(*args, env=None, cwd=None):
    """Run crumb.py as a subprocess; return (stdout, stderr, returncode)."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "cli" / "crumb.py"), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )
    return proc.stdout, proc.stderr, proc.returncode


class TestTemplatesEmitV13:
    @pytest.mark.parametrize("kind", ["task", "mem", "map", "log", "todo"])
    def test_default_template_emits_v13(self, kind, tmp_path, capsys):
        out_path = tmp_path / f"x.crumb"
        argv = [
            "new", kind, "--title", "smoke", "--source", "test",
            "-o", str(out_path),
        ]
        if kind == "task":
            argv.extend(["--goal", "g", "--context", "c", "--constraints", "x"])
        elif kind == "mem":
            argv.extend(["-e", "fact"])
        elif kind == "map":
            argv.extend(["--description", "p"])
        elif kind == "log":
            argv.extend(["-e", "event"])
        elif kind == "todo":
            argv.extend(["-e", "thing"])
        crumb.main(argv)
        text = out_path.read_text(encoding="utf-8")
        parsed = crumb.parse_crumb(text)
        assert parsed["headers"]["v"] == "1.3", f"{kind} template not bumped"

    def test_agent_template_still_emits_v13(self, tmp_path):
        out_path = tmp_path / "agent.crumb"
        crumb.main([
            "new", "agent",
            "--title", "x", "--source", "test",
            "--agent-id", "a",
            "-o", str(out_path),
        ])
        parsed = crumb.parse_crumb(out_path.read_text(encoding="utf-8"))
        assert parsed["headers"]["v"] == "1.3"


class TestDeprecatedAliases:
    def test_todo_add_alias_prints_hint(self, tmp_path):
        target = tmp_path / "t.crumb"
        out, err, rc = _run_cli(
            "todo-add", str(target), "first", "--title", "x", "--source", "t",
        )
        assert rc == 0
        assert "[deprecated]" in err
        assert "crumb todo add" in err
        assert target.exists()

    def test_todo_add_new_form_silent(self, tmp_path):
        target = tmp_path / "t2.crumb"
        out, err, rc = _run_cli(
            "todo", "add", str(target), "first", "--title", "x", "--source", "t",
        )
        assert rc == 0
        assert "[deprecated]" not in err

    def test_compress_alias_prints_hint(self, tmp_path):
        src = tmp_path / "in.crumb"
        src.write_text(
            "BEGIN CRUMB\nv=1.3\nkind=mem\nsource=t\n---\n"
            "[consolidated]\n- a\nEND CRUMB\n",
            encoding="utf-8",
        )
        out, err, rc = _run_cli("compress", str(src), "-o", "-")
        assert "[deprecated]" in err
        assert "crumb optimize --mode signal" in err

    def test_share_alias_prints_hint(self, tmp_path):
        src = tmp_path / "x.crumb"
        src.write_text(
            "BEGIN CRUMB\nv=1.3\nkind=mem\nsource=t\n---\n"
            "[consolidated]\n- a\nEND CRUMB\n",
            encoding="utf-8",
        )
        # share command requires `gh` or generates a data URI; we just need
        # to confirm the deprecation hint fires before any external dep.
        out, err, rc = _run_cli("share", str(src))
        assert "[deprecated]" in err
        assert "crumb handoff" in err

    def test_dashboard_alias_prints_hint(self, tmp_path):
        out, err, rc = _run_cli("dashboard", "-o", str(tmp_path / "d.html"))
        assert "[deprecated]" in err
        assert "audit export --format html" in err


class TestOptimizeModes:
    @pytest.fixture
    def src(self, tmp_path):
        path = tmp_path / "in.crumb"
        path.write_text(
            "BEGIN CRUMB\nv=1.3\nkind=task\ntitle=t\nsource=t\n---\n"
            "[goal]\ng\n[context]\nc\n[constraints]\nx\nEND CRUMB\n",
            encoding="utf-8",
        )
        return path

    def test_minimal_mode_runs(self, src):
        out, err, rc = _run_cli("optimize", str(src), "--mode", "minimal")
        assert rc == 0
        assert "BEGIN CRUMB" in out

    def test_signal_mode_runs(self, src):
        out, err, rc = _run_cli("optimize", str(src), "--mode", "signal")
        assert rc == 0
        assert "BEGIN CRUMB" in out

    def test_budget_mode_dry_run(self, src):
        out, err, rc = _run_cli(
            "optimize", str(src), "--mode", "budget", "--budget", "200", "--dry-run",
        )
        assert rc == 0
        assert "squeeze:" in out

    def test_budget_mode_requires_budget(self, src):
        out, err, rc = _run_cli("optimize", str(src), "--mode", "budget")
        assert rc != 0
        assert "--budget" in err


class TestExitCodesUniform:
    def test_validate_exits_2_on_parse_error(self, tmp_path):
        bad = tmp_path / "bad.crumb"
        bad.write_text("not a crumb\n", encoding="utf-8")
        out, err, rc = _run_cli("validate", str(bad))
        assert rc == 2

    def test_validate_exits_0_on_valid(self, tmp_path):
        good = tmp_path / "good.crumb"
        good.write_text(
            "BEGIN CRUMB\nv=1.3\nkind=mem\nsource=t\n---\n"
            "[consolidated]\n- x\nEND CRUMB\n",
            encoding="utf-8",
        )
        out, err, rc = _run_cli("validate", str(good))
        assert rc == 0


class TestUnknownKindEnumeration:
    def test_unknown_kind_lists_valid_kinds(self):
        bad = (
            "BEGIN CRUMB\nv=1.3\nkind=frogpile\nsource=t\n---\n"
            "[goal]\ng\nEND CRUMB\n"
        )
        with pytest.raises(ValueError) as exc:
            crumb.parse_crumb(bad)
        msg = str(exc.value)
        assert "unknown kind" in msg
        assert "frogpile" in msg
        # at least three known kinds appear in the enumeration
        for k in ("task", "mem", "agent"):
            assert k in msg


class TestAgentAuthFirstUseNotice:
    def test_first_use_prints_to_stderr_and_creates_dir(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from agentauth.store import PassportStore
        captured = capsys.readouterr()  # clear any prior
        PassportStore()
        captured = capsys.readouterr()
        assert "AgentAuth storage initialized" in captured.err
        assert (tmp_path / ".crumb-auth").is_dir()
        # second instantiation should be silent
        PassportStore()
        captured2 = capsys.readouterr()
        assert "AgentAuth storage initialized" not in captured2.err

    def test_crumb_quiet_suppresses_notice(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CRUMB_QUIET", "1")
        from agentauth.store import PassportStore
        capsys.readouterr()
        PassportStore()
        captured = capsys.readouterr()
        assert "AgentAuth storage initialized" not in captured.err


class TestGroupedHelp:
    def test_help_contains_grouped_index(self):
        out, err, rc = _run_cli("--help")
        assert rc == 0
        # one group label per concern from build_parser()
        for label in ("Create:", "Inspect:", "Edit:", "Optimize:",
                      "Handoff:", "Memory:", "Format:", "Governance:"):
            assert label in out
