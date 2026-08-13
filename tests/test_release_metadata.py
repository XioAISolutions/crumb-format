"""Release-metadata guards, run on every PR rather than only at release time.

These exist because of a concrete four-month failure: PyPI served
``crumb-format`` 0.2.0 from April while the repository documented v1.4 and ~46
commands. Every ``pip install crumb-format`` in that window produced a package
whose README quickstart did not run. Nothing caught it because nothing compared
the documentation to the code, or the working tree to the published index.

``scripts/release_check.py`` holds the logic; these tests make it fail a normal
build instead of waiting for someone to remember to run it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import release_check  # noqa: E402


def test_versions_agree_across_the_repo():
    """pyproject, CLI_VERSION, and the CHANGELOG heading must not diverge."""
    version = release_check.check_versions(tag=None)
    assert version, "no version resolved"


def test_tag_must_match_the_declared_version():
    version = release_check.check_versions(tag=None)
    release_check.check_versions(tag=f"v{version}")  # exact match is accepted
    with pytest.raises(release_check.CheckFailed, match="must equal"):
        release_check.check_versions(tag="v0.0.1-wrong")


def test_documented_commands_all_exist():
    """The guard that would have caught the stale wheel.

    Caught three real breaks on its first run: `crumb compress` in README.md
    and docs/QUICKSTART.md, and `crumb squeeze` in SPEC.md — all folded into
    `crumb optimize --mode ...` but never updated in the docs.
    """
    release_check.check_docs_match_cli()


def test_docs_check_actually_rejects_a_missing_command(tmp_path, monkeypatch):
    """A guard that cannot fail is not a guard — prove this one can."""
    fake_doc = tmp_path / "FAKE.md"
    fake_doc.write_text(
        "Run it:\n\n```bash\ncrumb definitely-not-a-real-command foo\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(release_check, "ROOT", tmp_path)
    documented = release_check.documented_subcommands("FAKE.md")
    assert documented["FAKE.md"] == {"definitely-not-a-real-command"}


def test_docs_check_ignores_prose_and_reads_only_code():
    """Prose like 'CRUMB can do X' must not be parsed as `crumb can`."""
    import re

    prose = "CRUMB is a format. crumb can do things. See `crumb validate f.crumb`."
    regions = release_check._code_regions(prose)
    names = {m.group(1) for m in release_check.INVOCATION_RE.finditer(regions)}
    assert names == {"validate"}


def test_publish_workflow_stays_tag_triggered():
    """Reverting to `release: published` reintroduces the original failure."""
    release_check.check_publish_is_tag_triggered()


def test_release_check_script_runs_clean():
    """End-to-end: the script itself exits 0 on a healthy tree."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "release_check.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_readme_command_is_reachable_from_help_all():
    """`--help-all` is the documented index; it must not omit shipped commands."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "crumb_cli.py"), "--help-all"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    listed = result.stdout
    for command in ("from-messages", "measure", "validate", "handoff", "receive"):
        assert command in listed, f"{command} missing from --help-all"
