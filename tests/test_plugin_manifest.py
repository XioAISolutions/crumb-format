"""The Claude Code plugin manifests must stay installable.

This is a **second distribution channel**, and the reason it matters is
specific: plugins install from GitHub, not PyPI. While the published wheel was
four months stale, this path would still have delivered a working CRUMB — no
`pip install`, no index, just the repository at a tag.

That independence is also the risk. Nothing about `pip` breaking would tell
anyone this path had rotted, so the invariants are asserted here instead:
the manifests parse, every path they name exists, the executables the hooks
invoke are real, and the version does not drift from the package.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = REPO_ROOT / ".claude-plugin"


def load(name: str) -> dict:
    return json.loads((PLUGIN_DIR / name).read_text(encoding="utf-8"))


def resolve(reference: str) -> Path:
    """Plugin paths are relative to the plugin root and start with './'."""
    assert reference.startswith("./"), (
        f"plugin path {reference!r} must be relative to the plugin root and start with './'"
    )
    return REPO_ROOT / reference[2:]


# ── the manifests themselves ─────────────────────────────────────────

def test_every_manifest_is_valid_json():
    for name in ("plugin.json", "hooks.json", "mcp.json", "marketplace.json"):
        load(name)


def test_plugin_declares_the_required_name():
    """`name` is the only required field, and it is immutable once published."""
    assert load("plugin.json")["name"] == "crumb-format"


def test_declared_component_paths_all_exist():
    """A manifest pointing at a moved directory installs and then does nothing."""
    manifest = load("plugin.json")
    missing = [
        f"{key} -> {manifest[key]}"
        for key in ("commands", "mcpServers", "hooks")
        if key in manifest and not resolve(manifest[key]).exists()
    ]
    assert not missing, f"plugin.json names paths that do not exist: {missing}"


# ── what the hooks actually run ──────────────────────────────────────

def test_hooks_cover_both_halves_of_the_handoff_loop():
    """Capture without resume is the manual-paste problem all over again."""
    hooks = load("hooks.json")["hooks"]
    assert "SessionEnd" in hooks, "no SessionEnd hook: sessions would leave no handoff"
    assert "SessionStart" in hooks, "no SessionStart hook: the handoff would need pasting"

    commands = [
        entry["command"]
        for event in hooks.values()
        for group in event
        for entry in group["hooks"]
    ]
    assert any("capture" in c for c in commands)
    assert any("resume" in c for c in commands)


def test_hook_commands_reference_the_plugin_root():
    """A bare relative path resolves against the user's cwd, not the plugin."""
    hooks = load("hooks.json")["hooks"]
    for event, groups in hooks.items():
        for group in groups:
            for entry in group["hooks"]:
                assert "${CLAUDE_PLUGIN_ROOT}" in entry["command"], (
                    f"{event} hook does not use ${{CLAUDE_PLUGIN_ROOT}}: {entry['command']}"
                )


def test_the_scripts_the_plugin_invokes_exist():
    referenced = [
        load("hooks.json")["hooks"][event][0]["hooks"][0]["command"]
        for event in ("SessionEnd", "SessionStart")
    ]
    referenced.append(" ".join(load("mcp.json")["mcpServers"]["crumb"]["args"]))

    for command in referenced:
        # Everything after the substituted plugin root is a repo-relative path.
        _, _, tail = command.partition("${CLAUDE_PLUGIN_ROOT}/")
        script = tail.split('"')[0].split()[0]
        assert (REPO_ROOT / script).is_file(), f"plugin invokes a missing script: {script}"


def test_the_cli_runs_from_a_bare_checkout():
    """The whole point: no install step, so the entry point must work as-is.

    If this ever needs `pip install` first, the plugin channel inherits the
    same failure the PyPI channel already had.
    """
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "crumb_cli.py"), "--version"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "crumb" in result.stdout


# ── the marketplace entry ────────────────────────────────────────────

def test_marketplace_points_at_this_repository():
    entry = load("marketplace.json")["plugins"]["crumb-format"]
    assert entry["source"] == "github"
    assert entry["owner"] == "XioAISolutions"
    assert entry["repo"] == "crumb-format"


def test_plugin_and_marketplace_versions_track_the_package():
    """Two more places a version can go stale. release_check.py enforces it;
    this asserts the wiring is present so a rename cannot quietly drop it."""
    import re

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE).group(1)

    assert load("plugin.json")["version"] == version
    assert load("marketplace.json")["plugins"]["crumb-format"]["version"] == version
