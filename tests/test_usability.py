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


class TestRemovedAliases:
    """v0.11: the v0.7 deprecation aliases are gone for real now.

    Each removed name should now be rejected by argparse as an unknown
    subcommand (exit 2). The replacement command (e.g. `crumb todo add`)
    should continue to work.
    """

    @pytest.mark.parametrize("removed", [
        "todo-add", "todo-done", "todo-list", "todo-dream",
        "compact", "compress", "squeeze",
        "share", "dashboard",
    ])
    def test_removed_alias_rejected(self, removed):
        out, err, rc = _run_cli(removed, "--help")
        # argparse exits 2 on invalid subcommand
        assert rc == 2, f"{removed!r} should be unknown but exited {rc}"
        assert "invalid choice" in err.lower() or removed in err

    def test_canonical_replacements_still_work(self, tmp_path):
        # The canonical names that replaced the deprecated aliases.
        target = tmp_path / "t.crumb"
        out, err, rc = _run_cli(
            "todo", "add", str(target), "first", "--title", "x", "--source", "t",
        )
        assert rc == 0
        assert target.exists()
        # Plain `crumb optimize --mode minimal/signal/budget` is the
        # surviving path for the old compact/compress/squeeze trio
        # (smoked elsewhere in TestOptimizeModes).


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


class TestMempalaceBridgeShim:
    """v0.11: cli/mempalace_bridge.py was renamed to cli/memory_bridge.py.
    The old import path stays as a wildcard re-export shim for one
    release. Codex caught a P1 in the first draft of this shim where
    it explicitly imported names that don't exist on the new module
    (cmd_mempalace, adapters) and crashed on import.
    """

    def test_shim_imports_without_error(self):
        # The literal regression: `import cli.mempalace_bridge` must not
        # raise. (Codex finding P1 — broke outright in the first draft.)
        import importlib
        import cli.mempalace_bridge
        importlib.reload(cli.mempalace_bridge)

    def test_shim_reexports_public_surface(self):
        import cli.mempalace_bridge as old
        import cli.memory_bridge as new
        # Each public name reachable through the new module must be
        # reachable through the shim, AND they must be the same object
        # (the shim isn't holding stale copies).
        for name in ("MempalaceAdapter", "BridgeAdapter", "ADAPTERS",
                     "get_adapter", "run_bridge_export", "run_bridge_import"):
            assert hasattr(new, name), f"{name} missing from new module"
            assert hasattr(old, name), f"{name} not re-exported by shim"
            assert getattr(old, name) is getattr(new, name), (
                f"{name} on shim is a different object than on the new module"
            )


class TestFromHaloHidden:
    """v0.11: `from-halo` is a hidden alias of `from-otel`. Codex P2:
    `help=argparse.SUPPRESS` doesn't drop the entry from the
    subparsers listing — it just renders the literal "==SUPPRESS=="
    text. The fix removes the entry from `sub._choices_actions`.
    """

    def test_from_halo_not_in_default_help(self):
        out, err, rc = _run_cli("--help")
        assert rc == 0
        assert "from-halo" not in out
        assert "SUPPRESS" not in out

    def test_from_halo_not_in_help_all(self):
        out, err, rc = _run_cli("--help-all")
        assert rc == 0
        assert "from-halo" not in out
        assert "SUPPRESS" not in out

    def test_from_halo_still_parses(self, tmp_path):
        # The alias is hidden but functional — scripts that pinned
        # `crumb from-halo` continue to work.
        from pathlib import Path
        fixture = (Path(__file__).resolve().parent / "fixtures" / "halo-traces.jsonl")
        out, err, rc = _run_cli("from-halo", str(fixture))
        assert rc == 0
        assert "BEGIN CRUMB" in out
        assert "kind=log" in out


class TestHelpAllScopedToTopLevel:
    """v0.11 P2: `--help-all` interception must only fire as a top-level
    flag, not as a positional inside subcommand args. Codex caught the
    first version doing `if "--help-all" in argv` unconditionally, which
    hijacked any invocation with the literal token anywhere.
    """

    def test_help_all_as_literal_task_text(self, tmp_path):
        # `crumb todo add` with --help-all as part of a task name should
        # add the task verbatim, not print help.
        target = tmp_path / "t.crumb"
        out, err, rc = _run_cli(
            "todo", "add", str(target), "do --help-all stuff",
            "--title", "x", "--source", "t",
        )
        assert rc == 0
        assert target.exists()
        body = target.read_text(encoding="utf-8")
        assert "do --help-all stuff" in body

    def test_help_all_as_literal_in_goal(self, tmp_path):
        # `crumb new task --goal "explain --help-all"` should NOT trigger
        # the top-level help-all path.
        out_path = tmp_path / "g.crumb"
        out, err, rc = _run_cli(
            "new", "task", "--title", "x", "--source", "s",
            "--goal", "explain --help-all",
            "--context", "c", "--constraints", "x",
            "-o", str(out_path),
        )
        assert rc == 0
        assert out_path.exists()
        assert "explain --help-all" in out_path.read_text(encoding="utf-8")

    def test_help_all_after_double_dash(self, tmp_path):
        # `crumb todo add file -- --help-all task1` should treat
        # --help-all as a positional task name, not trigger help.
        target = tmp_path / "t.crumb"
        out, err, rc = _run_cli(
            "todo", "add", str(target), "--title", "x", "--source", "t",
            "--", "--help-all",
        )
        assert rc == 0
        assert target.exists()


class TestGroupedHelp:
    def test_help_shows_core_commands(self):
        # v0.11: default --help shows only the core 5 commands plus a
        # pointer to --help-all. The full grouped index moved behind
        # --help-all.
        out, err, rc = _run_cli("--help")
        assert rc == 0
        assert "Core commands" in out
        for cmd in ("new", "validate", "handoff", "receive", "lint"):
            assert cmd in out
        assert "--help-all" in out

    def test_help_does_not_dump_all_subcommands(self):
        # Codex P2: argparse's auto-rendered subcommand table under
        # "positional arguments:" was leaking the full ~40-command
        # list into --help, defeating the two-tier flow. We clear
        # sub._choices_actions to suppress that table.
        out, err, rc = _run_cli("--help")
        assert rc == 0
        # The subcommand table renders each command on its own indented
        # line. Look for that signature on non-core names that don't
        # also appear in our description prose.
        import re
        for non_core in ("palace", "guardrails", "passport",
                         "metalk", "from-chat", "from-otel",
                         "audit", "policy", "comply", "webhook",
                         "delta", "apply", "seen", "hash", "resolve"):
            # Pattern: 2+ spaces, name, then spaces or newline (the
            # argparse table format). Doesn't match prose like "format
            # bridges" or "audit trail".
            pattern = r"^ {2,}" + re.escape(non_core) + r"(?:\s|$)"
            assert not re.search(pattern, out, re.MULTILINE), (
                f"non-core command {non_core!r} leaked into --help table"
            )
        # And the output should be short — sanity-check ~40 lines or
        # less so we don't silently regrow.
        assert out.count("\n") < 40, (
            f"--help is {out.count(chr(10))} lines; should be ~30 or less"
        )

    def test_help_all_contains_grouped_index(self):
        out, err, rc = _run_cli("--help-all")
        assert rc == 0
        # one group label per concern from build_parser()
        for label in ("Create:", "Inspect:", "Edit:", "Optimize:",
                      "Handoff:", "Memory:", "Format:", "Governance:"):
            assert label in out
