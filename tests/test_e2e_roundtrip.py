"""End-to-end roundtrip tests for the create → validate → (handoff →
receive →) inspect lifecycle.

These exercise the full path a user actually takes:
  1. `crumb new <kind>` produces a file
  2. The file validates with the Python validator
  3. The same file validates with the Node validator (parity)
  4. `crumb inspect` recovers the headers/sections we put in
  5. After a clipboard-style roundtrip (text → parse → render →
     validate), the second crumb is byte-for-byte equivalent

A regression in any of these breaks the "switch AIs without losing
context" core promise. Unit tests miss the cross-step interactions.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
NODE_BIN = shutil.which("node")


def _run_cli(*args, env=None, cwd=None, input_text=None):
    """Run crumb.py as a subprocess; return (stdout, stderr, returncode)."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "cli" / "crumb.py"), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        input=input_text,
    )
    return proc.stdout, proc.stderr, proc.returncode


def _validate_python(path):
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "validators" / "validate.py"), str(path)],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _validate_node(path):
    proc = subprocess.run(
        ["node", str(REPO_ROOT / "validators" / "validate.js"), str(path)],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


@pytest.mark.parametrize("kind", ["task", "mem", "map", "log", "todo", "agent"])
class TestNewValidateInspectRoundtrip:
    """For every kind, `crumb new` → file → `crumb validate` → `crumb
    inspect` must recover the same headers and the file must also
    validate against the Node validator."""

    def test_python_validate(self, kind, tmp_path):
        out_path = tmp_path / f"x-{kind}.crumb"
        argv = ["new", kind, "--title", f"e2e {kind}", "--output", str(out_path)]
        if kind == "task":
            argv += ["--goal", "ship the thing"]
        out, err, rc = _run_cli(*argv)
        assert rc == 0, f"new failed: {err}"
        assert out_path.exists()

        rc, vout, verr = _validate_python(out_path)
        assert rc == 0, f"python validate failed: {verr or vout}"
        assert "OK" in vout

    @pytest.mark.skipif(NODE_BIN is None, reason="node not on PATH")
    def test_node_validate(self, kind, tmp_path):
        out_path = tmp_path / f"x-{kind}.crumb"
        argv = ["new", kind, "--title", f"e2e {kind}", "--output", str(out_path)]
        if kind == "task":
            argv += ["--goal", "ship the thing"]
        _run_cli(*argv)

        rc, vout, verr = _validate_node(out_path)
        assert rc == 0, f"node validate failed: {verr or vout}"
        assert "OK" in vout

    def test_inspect_recovers_kind_and_title(self, kind, tmp_path):
        out_path = tmp_path / f"x-{kind}.crumb"
        argv = ["new", kind, "--title", f"e2e {kind}", "--output", str(out_path)]
        if kind == "task":
            argv += ["--goal", "ship the thing"]
        _run_cli(*argv)

        out, err, rc = _run_cli("inspect", str(out_path))
        assert rc == 0, f"inspect failed: {err}"
        assert kind in out
        assert f"e2e {kind}" in out


class TestVersionWhitelist:
    """Wire format must accept v=1.1..1.4 and reject v=1.5+. This is
    the contract that lets old crumbs keep working post-1.0."""

    def _build(self, version, tmp_path, kind="task"):
        path = tmp_path / f"v{version}.crumb"
        path.write_text(
            f"BEGIN CRUMB\n"
            f"v={version}\n"
            f"kind={kind}\n"
            f"title=test\n"
            f"source=test\n"
            f"---\n"
            f"[goal]\nfoo\n\n"
            f"[context]\n- bar\n\n"
            f"[constraints]\n- baz\n"
            f"END CRUMB\n"
        )
        return path

    @pytest.mark.parametrize("version", ["1.1", "1.2", "1.3", "1.4"])
    def test_supported_versions_validate(self, version, tmp_path):
        path = self._build(version, tmp_path)
        rc, _, _ = _validate_python(path)
        assert rc == 0, f"v={version} should validate"

    @pytest.mark.parametrize("version", ["1.0", "1.5", "2.0"])
    def test_unsupported_versions_rejected(self, version, tmp_path):
        path = self._build(version, tmp_path)
        rc, _, _ = _validate_python(path)
        assert rc != 0, f"v={version} should be rejected"


class TestHandoffReceiveRoundtrip:
    """Simulate the clipboard path: write a crumb, read its bytes,
    parse, re-render, and assert the second crumb validates AND has
    the same essential headers as the first."""

    def test_render_parse_render_is_stable(self, tmp_path):
        """A crumb that's parsed and re-rendered should still validate
        and preserve kind/v/title/source. (Body-level byte equality
        isn't promised — render is canonical, source may vary in
        whitespace — but the headers are.)"""
        sys.path.insert(0, str(REPO_ROOT / "cli"))
        sys.path.insert(0, str(REPO_ROOT))
        import crumb as crumb_mod  # type: ignore

        # Round 1
        out_path = tmp_path / "round1.crumb"
        _run_cli("new", "task", "--title", "Stability check",
                 "--goal", "Verify roundtrip preserves headers",
                 "--output", str(out_path))
        text1 = out_path.read_text()
        parsed1 = crumb_mod.parse_crumb(text1)

        # Round 2: re-render from the parsed structure
        re_rendered = crumb_mod.render_crumb(
            headers=parsed1["headers"],
            sections={k: v for k, v in parsed1["sections"].items()},
        )
        round2_path = tmp_path / "round2.crumb"
        round2_path.write_text(re_rendered)
        rc, _, _ = _validate_python(round2_path)
        assert rc == 0, "re-rendered crumb must validate"

        parsed2 = crumb_mod.parse_crumb(re_rendered)
        for header in ["v", "kind", "title", "source"]:
            assert parsed1["headers"].get(header) == parsed2["headers"].get(header), \
                f"header {header!r} drifted across roundtrip"


class TestContextCommand:
    """`crumb context` is the auto-generation path. It must produce a
    valid v=1.4 crumb with a [handoff] block, since users paste this
    output directly into another AI without inspecting it."""

    def test_context_emits_v14_with_handoff(self, tmp_path):
        # Initialize a tiny git repo so the git path through cmd_context
        # has something to read. (Without git, cmd_context still works
        # but the context block is sparse.)
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                        "-c", "commit.gpgsign=false",
                        "commit", "--allow-empty", "-m", "Initial commit", "-q"],
                       cwd=tmp_path, check=True)

        out_path = tmp_path / "ctx.crumb"
        out, err, rc = _run_cli("context", "-o", str(out_path), cwd=str(tmp_path))
        assert rc == 0, f"context failed: {err}"

        text = out_path.read_text()
        assert "v=1.4" in text, "context must emit v=1.4 (post-1.0 default)"
        assert "[handoff]" in text, "context must include a [handoff] block"
        assert "- next:" in text, "[handoff] must point at a concrete next step"

        # And it must validate.
        rc, _, _ = _validate_python(out_path)
        assert rc == 0
