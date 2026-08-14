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


def _publish_workflow() -> dict:
    import yaml

    text = (REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text()
    loaded = yaml.safe_load(text)
    # PyYAML reads the bare key `on` as the boolean True (YAML 1.1).
    loaded["on"] = loaded.get("on", loaded.get(True))
    return loaded


def test_publish_workflow_is_valid_yaml_with_three_entry_points():
    """Tag push, branch push, and dispatch must all reach the publish jobs.

    The tag path alone is not enough: `git push origin v1.2.0` is rejected at
    `git-receive-pack` from this org's automation, and the Actions dispatch API
    answers 403 for the same identity. If a future edit drops the branch
    trigger, releasing goes back to requiring a human at an unrestricted
    terminal — which is exactly how 0.2.0 stayed on PyPI for four months.
    """
    workflow = _publish_workflow()
    triggers = workflow["on"]
    assert "v*.*.*" in triggers["push"]["tags"]
    assert "main" in triggers["push"]["branches"]
    assert "tag" in triggers["workflow_dispatch"]["inputs"]


def test_workflows_running_the_suite_install_its_test_dependencies():
    """Three separate workflows run `pytest tests/`; all three need the deps.

    Adding the yaml-based workflow guards above broke two of them until this
    test pointed at it: they installed `pytest` and nothing else, so a
    test-only import would have failed in CI rather than locally.
    """
    workflows = (REPO_ROOT / ".github" / "workflows").glob("*.yml")
    TEST_ONLY_IMPORTS = {"yaml": "pyyaml"}

    offenders = []
    for path in workflows:
        text = path.read_text()
        if "pytest tests/" not in text:
            continue
        for module, distribution in TEST_ONLY_IMPORTS.items():
            if distribution not in text:
                offenders.append(f"{path.name} runs the suite without installing {distribution}")
    assert not offenders, "\n".join(offenders)


def test_branch_push_cannot_republish_an_existing_version():
    """Every merge to main runs this workflow; only new versions may release.

    Without the tag-existence check, each merge after a release would rebuild
    and re-upload the same version. The gate job answers that question once and
    every downstream job keys off its output.
    """
    workflow = _publish_workflow()
    jobs = workflow["jobs"]

    decide = next(
        step for step in jobs["gate"]["steps"] if step.get("id") == "decide"
    )
    assert "refs/tags/$TAG" in decide["run"], "gate no longer checks for the tag"

    for name in ("build", "release", "publish"):
        assert jobs[name]["if"] == "needs.gate.outputs.release == 'true'", (
            f"job {name!r} does not honour the gate; a merge to main after a "
            "release would rebuild and re-upload the published version"
        )
        assert "gate" in jobs[name]["needs"]


def test_release_check_script_runs_clean():
    """End-to-end: the script itself exits 0 on a healthy tree."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "release_check.py")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# Commands deliberately kept out of the index. `from-halo` is a hidden alias of
# `from-otel`, retained for scripts that already call it.
HIDDEN_COMMANDS = {"from-halo"}


def test_every_shipped_command_appears_in_help_all():
    """`--help-all` is the documented index; it must not omit shipped commands.

    Asserting a hand-written list of a few names is what let `crumb capture`
    ship without appearing in the index at all. This compares the index against
    every subcommand the parser actually registers.
    """
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "crumb_cli.py"), "--help-all"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    listed = result.stdout

    registered = release_check.registered_subcommands() - HIDDEN_COMMANDS
    missing = sorted(name for name in registered if name not in listed)
    assert not missing, (
        f"commands ship but are absent from `crumb --help-all`: {missing}. "
        "Either add them to a group in build_parser's full_help, or add them "
        "to HIDDEN_COMMANDS with a reason."
    )


def test_the_wheel_does_not_ship_the_research_language_model():
    """`pip install crumb-format` must not carry a PyTorch model.

    crumb_wavelm is ~4,600 lines of wave-field research code with nothing to do
    with the wire format. Shipping it inside the format's distribution is a
    liability for anyone evaluating CRUMB as a format, and it inflates the
    install for everyone. The source stays in the repository; only the packaging
    excludes it.
    """
    pyproject = release_check.read("pyproject.toml")
    packages_block = pyproject.split("packages = [", 1)[1].split("]", 1)[0]
    assert "crumb_wavelm" not in packages_block, (
        "crumb_wavelm is back in the wheel's package list — build it as its own "
        "distribution with crumb_wavelm.setup_standalone instead."
    )
