"""Tests for the user-friendly surface: `crumb hello`, `crumb doctor`,
and the FriendlyArgumentParser hint that fires on missing-arg errors.

These commands are the first surface a brand-new user touches, so we test
them as black-box subprocesses to mirror real usage.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


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


class TestHello:
    def test_hello_runs(self):
        out, err, rc = _run_cli("hello", "--no-clipboard")
        assert rc == 0, f"stderr={err!r}"
        assert "Welcome to CRUMB" in out
        assert "BEGIN CRUMB" in out
        assert "END CRUMB" in out

    def test_hello_emits_valid_crumb(self):
        """The sample crumb that hello prints must validate."""
        out, _, rc = _run_cli("hello", "--no-clipboard")
        assert rc == 0
        # Extract the indented crumb block from hello's output
        lines = out.splitlines()
        body = []
        in_block = False
        for line in lines:
            if line.startswith("  BEGIN CRUMB"):
                in_block = True
            if in_block:
                body.append(line[2:] if line.startswith("  ") else line)
            if line.startswith("  END CRUMB"):
                break
        assert body, "no crumb block found in hello output"

        # Round-trip through the actual parser
        sys.path.insert(0, str(REPO_ROOT / "cli"))
        sys.path.insert(0, str(REPO_ROOT))
        import crumb as crumb_mod  # type: ignore
        parsed = crumb_mod.parse_crumb("\n".join(body))
        assert parsed["headers"]["kind"] == "task"
        assert parsed["headers"]["v"] in {"1.1", "1.2", "1.3"}

    def test_hello_lists_next_steps(self):
        out, _, rc = _run_cli("hello", "--no-clipboard")
        assert rc == 0
        assert "Next steps:" in out
        assert "crumb new task" in out
        assert "crumb doctor" in out
        assert "crumb --help-all" in out

    def test_hello_no_clipboard_flag_skips_clipboard(self):
        out, _, rc = _run_cli("hello", "--no-clipboard")
        assert rc == 0
        # Either the clipboard tool is missing or we skipped — never the
        # success message under --no-clipboard.
        assert "Copied to clipboard" not in out


class TestDoctor:
    def test_doctor_runs(self):
        out, _, rc = _run_cli("doctor")
        assert rc == 0
        assert "crumb doctor" in out

    def test_doctor_reports_runtime(self):
        out, _, rc = _run_cli("doctor")
        assert rc == 0
        assert "Runtime:" in out
        assert f"Python {sys.version_info.major}.{sys.version_info.minor}" in out

    def test_doctor_reports_validators(self):
        out, _, rc = _run_cli("doctor")
        assert rc == 0
        assert "Python validator" in out
        assert "Node validator" in out

    def test_doctor_reports_clipboard(self):
        out, _, rc = _run_cli("doctor")
        assert rc == 0
        assert "Clipboard:" in out

    def test_doctor_reports_optional_integrations(self):
        out, _, rc = _run_cli("doctor")
        assert rc == 0
        assert "Optional integrations:" in out
        assert "Palace" in out
        assert "Claude Code" in out

    def test_doctor_always_exits_zero(self):
        """doctor is informational; even with all warnings it must exit 0."""
        out, _, rc = _run_cli("doctor")
        assert rc == 0


class TestFriendlyErrors:
    """The _FriendlyArgumentParser hook appends a one-line hint to
    argparse's terse error messages so a confused user has somewhere
    to go next."""

    def test_missing_required_arg_shows_hint(self):
        _, err, rc = _run_cli("validate")
        assert rc == 2
        assert "the following arguments are required" in err
        assert "crumb hello" in err
        assert "--help" in err

    def test_invalid_choice_shows_hint(self):
        _, err, rc = _run_cli("new", "frogpile")
        assert rc == 2
        assert "invalid choice" in err
        assert "crumb hello" in err

    def test_top_level_no_command_shows_help_zero(self):
        """Bare `crumb` should show the core help, not error out."""
        out, _, rc = _run_cli()
        assert rc == 0
        assert "hello" in out
        assert "doctor" in out


class TestHelpMentionsNewCommands:
    def test_default_help_mentions_hello(self):
        out, _, rc = _run_cli("--help")
        assert rc == 0
        assert "hello" in out
        assert "doctor" in out

    def test_help_all_mentions_hello_in_setup_group(self):
        out, _, rc = _run_cli("--help-all")
        assert rc == 0
        assert "hello" in out
        assert "doctor" in out
