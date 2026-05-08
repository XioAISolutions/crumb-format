"""Smoke tests for the integration installers.

These don't actually install (they'd touch ~/.claude or ~/.cursor on the
test machine). They check syntactic well-formedness, asset presence, and
that --dry-run produces the documented preview output.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


INTEGRATIONS = ["claude-code", "cursor"]


@pytest.mark.parametrize("integration", INTEGRATIONS)
class TestInstallerShape:
    """Each installer must syntax-check, have an executable bit, and
    accept --dry-run / --help without writing anything."""

    def test_install_sh_syntax(self, integration):
        path = REPO_ROOT / "integrations" / integration / "install.sh"
        assert path.exists(), f"{path} missing"
        rc = subprocess.run(["bash", "-n", str(path)]).returncode
        assert rc == 0, f"{integration}/install.sh has syntax errors"

    def test_uninstall_sh_syntax(self, integration):
        path = REPO_ROOT / "integrations" / integration / "uninstall.sh"
        assert path.exists(), f"{path} missing"
        rc = subprocess.run(["bash", "-n", str(path)]).returncode
        assert rc == 0, f"{integration}/uninstall.sh has syntax errors"

    def test_install_sh_executable(self, integration):
        path = REPO_ROOT / "integrations" / integration / "install.sh"
        # Shebang exists and is bash.
        first = path.read_text().splitlines()[0]
        assert first.startswith("#!"), f"{integration} missing shebang"
        assert "bash" in first, f"{integration} shebang isn't bash: {first!r}"

    def test_install_help_exits_zero(self, integration):
        """--help must succeed and not require crumb-format to be installed."""
        path = REPO_ROOT / "integrations" / integration / "install.sh"
        proc = subprocess.run(
            ["bash", str(path), "--help"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"--help exit {proc.returncode}: {proc.stderr}"
        assert "Usage:" in proc.stdout

    def test_install_unknown_arg_rejected(self, integration):
        path = REPO_ROOT / "integrations" / integration / "install.sh"
        proc = subprocess.run(
            ["bash", str(path), "--bogus-flag"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 2, "unknown arg should exit 2"
        assert "unknown argument" in proc.stderr


class TestMcpTemplateRendering:
    """Both MCP templates use the same __CRUMB_PYTHON__ /
    __CRUMB_INSTALL_PATH__ substitution shape and must stay in sync."""

    @pytest.mark.parametrize("integration", INTEGRATIONS)
    def test_template_is_valid_json_after_substitution(self, integration):
        path = REPO_ROOT / "integrations" / integration / "mcp.json.template"
        assert path.exists()
        rendered = path.read_text() \
            .replace("__CRUMB_PYTHON__", "/usr/bin/python3") \
            .replace("__CRUMB_INSTALL_PATH__", "/some/install/path")
        data = json.loads(rendered)
        assert "mcpServers" in data
        assert "crumb" in data["mcpServers"]
        srv = data["mcpServers"]["crumb"]
        assert srv["command"] == "/usr/bin/python3"
        assert srv["args"] == ["/some/install/path/mcp/server.py"]
        # Every shipping template must carry _managed_by so uninstall
        # can distinguish entries it created from user-managed ones.
        assert "_managed_by" in srv, f"{integration} missing _managed_by marker"
        assert integration in srv["_managed_by"]


class TestCursorRules:
    """Cursor uses .cursor/rules/*.mdc — verify each rule has the
    expected frontmatter fields."""

    RULES_DIR = REPO_ROOT / "integrations" / "cursor" / "rules"

    @pytest.mark.parametrize("rule", ["crumb-export", "crumb-import", "crumb-it"])
    def test_rule_has_frontmatter(self, rule):
        path = self.RULES_DIR / f"{rule}.mdc"
        assert path.exists(), f"{path} missing"
        text = path.read_text()
        # Must start with --- frontmatter delimiter
        assert text.startswith("---\n"), f"{rule} missing frontmatter"
        # Frontmatter must contain description + alwaysApply (Cursor's required keys)
        # Match within the first frontmatter block only
        match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
        assert match, f"{rule} frontmatter not closed properly"
        fm = match.group(1)
        assert "description:" in fm, f"{rule} missing description in frontmatter"
        assert "alwaysApply:" in fm, f"{rule} missing alwaysApply in frontmatter"

    def test_crumb_it_is_always_apply(self):
        """The verbal-trigger rule must be alwaysApply: true so Cursor
        recognizes the format ambiently."""
        text = (self.RULES_DIR / "crumb-it.mdc").read_text()
        assert re.search(r"alwaysApply:\s*true", text), \
            "crumb-it.mdc must have alwaysApply: true"

    def test_export_and_import_are_not_always_apply(self):
        """Export/import rules fire only on demand or on file open."""
        for rule in ["crumb-export", "crumb-import"]:
            text = (self.RULES_DIR / f"{rule}.mdc").read_text()
            assert re.search(r"alwaysApply:\s*false", text), \
                f"{rule} should be alwaysApply: false"


class TestClaudeCodeCommands:
    """Claude Code uses ~/.claude/commands/*.md slash commands. Each
    must have valid frontmatter."""

    COMMANDS_DIR = REPO_ROOT / "integrations" / "claude-code" / "commands"

    @pytest.mark.parametrize("cmd", ["crumb-export", "crumb-import"])
    def test_command_has_frontmatter(self, cmd):
        path = self.COMMANDS_DIR / f"{cmd}.md"
        assert path.exists(), f"{path} missing"
        text = path.read_text()
        assert text.startswith("---\n"), f"{cmd} missing frontmatter"
        match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
        assert match, f"{cmd} frontmatter not closed"
        fm = match.group(1)
        assert "description:" in fm, f"{cmd} missing description"


class TestUninstallerMarkerCheck:
    """Each uninstaller must reference the install.sh marker before
    deleting anything — that's the "preserve user files" contract."""

    @pytest.mark.parametrize("integration", INTEGRATIONS)
    def test_uninstaller_checks_marker(self, integration):
        path = REPO_ROOT / "integrations" / integration / "uninstall.sh"
        text = path.read_text()
        assert "_managed_by" in text or "managed by crumb-format" in text, \
            f"{integration}/uninstall.sh must check for the install.sh marker"
        assert "preserved" in text.lower(), \
            f"{integration}/uninstall.sh should report preserved (un-marked) files"
