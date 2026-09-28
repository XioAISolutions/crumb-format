#!/usr/bin/env python3
"""Fail fast when the release metadata, the docs, and the CLI disagree.

This exists because of a specific, four-month-long failure: PyPI served
``crumb-format`` 0.2.0 (April) while the repository documented v1.4 and ~46
commands. Every ``pip install crumb-format`` in that window produced a package
where the README's own quickstart — ``crumb hello``, ``crumb doctor`` — did not
exist. Nothing in CI noticed, because nothing compared the docs to the code.

Three classes of drift are checked here:

1. **Version drift** — ``pyproject.toml``, ``cli.crumb.CLI_VERSION``, the
   ``CHANGELOG.md`` heading, and the git tag must all agree.
2. **Documentation drift** — every ``crumb <subcommand>`` shown in the README or
   the quickstart must be a subcommand the CLI actually registers. A README that
   promises commands the package does not ship is the failure above.
3. **Publish-path drift** — the publish workflow must stay tag-triggered. The
   previous workflow fired on ``release: published``, which required a human to
   create a GitHub Release by hand; nobody did, so it never ran once.

Run with ``--tag vX.Y.Z`` in CI to additionally pin the tag, or bare for a
local check. ``--check-pypi`` additionally compares against the published
index, which is the drift detector that would have caught the original failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PYPI_JSON = "https://pypi.org/pypi/crumb-format/json"

# Tokens that follow the word "crumb" in prose or shell examples but are not
# subcommands. Keeping this list explicit (rather than silently skipping
# anything unrecognised) is what makes the docs check meaningful.
NOT_SUBCOMMANDS = {
    "it",  # the "crumb it" verbal trigger
    "install",
    "format",
    "run",
    "--help",
    "--help-all",
    "--version",
    "-h",
}


class CheckFailed(AssertionError):
    """A release precondition is not met."""


def read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise CheckFailed(f"missing required release file: {relative}")
    return path.read_text(encoding="utf-8")


def capture(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise CheckFailed(f"could not find {label}")
    return match.group(1)


# ── the checks ───────────────────────────────────────────────────────

def registered_subcommands() -> set[str]:
    """Every subcommand the CLI actually registers, including hidden ones."""
    from cli import crumb

    parser = crumb.build_parser()
    for action in parser._actions:
        if getattr(action, "choices", None) and hasattr(action, "_name_parser_map"):
            return set(action.choices)
    raise CheckFailed("could not enumerate subcommands from the argument parser")


FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
# Only an actual invocation counts: start of a line in a code block, optionally
# behind a shell prompt. Matching bare prose ("CRUMB can ...") would flag every
# English word that happens to follow the project name.
INVOCATION_RE = re.compile(r"^\s*(?:\$\s*)?crumb\s+([a-z][a-z0-9-]*)", re.MULTILINE)


def _code_regions(text: str) -> str:
    """Everything the reader would plausibly copy and paste."""
    blocks = FENCED_BLOCK_RE.findall(text)
    inline = INLINE_CODE_RE.findall(text)
    return "\n".join(blocks + inline)


def documented_subcommands(*relative_paths: str) -> dict[str, set[str]]:
    """Map each doc file to the ``crumb <subcommand>`` invocations it shows.

    Only code blocks and inline code are scanned. CHANGELOG.md is deliberately
    excluded elsewhere: it is a historical record, and an entry describing a
    command that existed at v0.9 is not a broken promise.
    """
    found: dict[str, set[str]] = {}
    for relative in relative_paths:
        names = {m.group(1) for m in INVOCATION_RE.finditer(_code_regions(read(relative)))}
        found[relative] = names - NOT_SUBCOMMANDS
    return found


def check_versions(tag: str | None) -> str:
    pyproject = read("pyproject.toml")
    cli_source = read("cli/crumb.py")
    changelog = read("CHANGELOG.md")

    project_version = capture(r'^version\s*=\s*"([^"]+)"', pyproject, "pyproject version")
    cli_version = capture(r'^CLI_VERSION\s*=\s*"([^"]+)"', cli_source, "CLI_VERSION")

    versions = {"pyproject.toml": project_version, "cli/crumb.py": cli_version}

    # The plugin manifests are a second distribution channel — installed from
    # GitHub rather than PyPI — and they carry their own version string. That
    # is exactly the shape of the drift that let PyPI serve 0.2.0 for four
    # months, so they are checked here rather than left to rot independently.
    plugin = json.loads(read(".claude-plugin/plugin.json"))
    versions[".claude-plugin/plugin.json"] = plugin.get("version", "")

    marketplace = json.loads(read(".claude-plugin/marketplace.json"))
    entry = marketplace.get("plugins", {}).get("crumb-format", {})
    versions[".claude-plugin/marketplace.json"] = entry.get("version", "")

    if len(set(versions.values())) != 1:
        raise CheckFailed(f"versions disagree: {versions}")

    escaped = project_version.replace(".", r"\.")
    if not re.search(rf"^## v{escaped}(\s|$)", changelog, flags=re.MULTILINE):
        raise CheckFailed(
            f"CHANGELOG.md has no '## v{project_version}' heading — "
            "move the Unreleased section under a version heading before releasing"
        )

    expected_tag = f"v{project_version}"
    if tag and tag != expected_tag:
        raise CheckFailed(f"release tag {tag!r} must equal {expected_tag!r}")

    return project_version


def check_docs_match_cli() -> None:
    """Every command the docs promise must exist. This is the important one."""
    registered = registered_subcommands()
    problems: list[str] = []
    docs = documented_subcommands(
        "README.md", "docs/QUICKSTART.md", "SPEC.md", "docs/PROTOCOL.md", "docs/PACKS.md"
    )
    for relative, names in docs.items():
        missing = sorted(names - registered)
        if missing:
            problems.append(f"{relative} documents commands the CLI does not register: {missing}")
    if problems:
        raise CheckFailed(
            "\n".join(problems)
            + "\n\nEither implement the command, or fix the docs. A README that "
            "promises commands the package does not ship is how crumb-format "
            "shipped a four-month-stale wheel without anyone noticing."
        )


def check_version_is_not_already_released(version: str) -> None:
    """Fail when this version is already tagged at a *different* commit.

    Every other check here verifies internal consistency: pyproject, the CLI,
    the CHANGELOG and the tag all saying the same number. They cannot catch the
    case where all four agree and the number is simply wrong — where a version
    has already been tagged and released, and then eight more feature commits
    landed on main still claiming to be it.

    That is not hypothetical: v1.2.0 was tagged and released, and `crumb
    resume`, the Claude Code plugin, the compression receipt and the `captured=`
    header all merged afterwards under the same version string. Two different
    trees answering to one version is exactly the drift this file exists to
    prevent, arriving somewhere the file did not look.

    Skipped, with a note, when tags are unavailable — a shallow CI checkout has
    none, and refusing to run would make the check useless where it is cheapest
    to run. The publish workflow fetches with ``fetch-depth: 0``, so it is
    enforced at the moment that matters.
    """
    import subprocess

    tag = f"v{version}"
    try:
        tagged = subprocess.run(
            ["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"],
            capture_output=True, text=True, cwd=ROOT,
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        print("  !  git unavailable; skipping the released-version check")
        return

    if head.returncode != 0:
        print("  !  no git history here; skipping the released-version check")
        return
    if tagged.returncode != 0:
        return  # not tagged yet, which is what an unreleased version looks like

    tagged_commit, head_commit = tagged.stdout.strip(), head.stdout.strip()
    if tagged_commit == head_commit:
        return  # this commit *is* the release

    raise CheckFailed(
        f"{tag} is already tagged at {tagged_commit[:12]}, but HEAD is "
        f"{head_commit[:12]}.\n"
        f"      Version {version} has been released with different content. Bump "
        "the version and add a CHANGELOG heading before merging more work, or "
        "two trees will answer to one version."
    )


def check_publish_is_tag_triggered() -> None:
    workflow = read(".github/workflows/publish-pypi.yml")
    if "tags:" not in workflow:
        raise CheckFailed(
            "publish-pypi.yml is not tag-triggered. The previous workflow fired on "
            "'release: published', which needed a hand-created GitHub Release and "
            "therefore never ran. Pushing a tag must be sufficient to publish."
        )


def check_pypi_is_current(local_version: str) -> None:
    """Compare the local version against the published index."""
    try:
        with urllib.request.urlopen(PYPI_JSON, timeout=20) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"  ! could not reach PyPI ({exc}); skipping index comparison")
        return

    published = payload.get("info", {}).get("version", "")
    releases = payload.get("releases", {})
    if published == local_version:
        print(f"  ok  PyPI serves {published}, matching the working tree")
        return
    print(
        f"  !!  DRIFT: PyPI serves {published}, working tree is {local_version}.\n"
        f"      Published versions: {sorted(releases)}\n"
        f"      Anyone running 'pip install crumb-format' gets {published}."
    )
    raise CheckFailed(f"published version {published} != local version {local_version}")


# ── entry point ──────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", help="Release tag to verify, e.g. v1.2.0.")
    parser.add_argument(
        "--check-pypi",
        action="store_true",
        help="Also fail if the published PyPI version differs from the working tree.",
    )
    args = parser.parse_args(argv)

    checks = [
        ("versions agree", lambda: check_versions(args.tag)),
        ("docs match the CLI", check_docs_match_cli),
        ("publish path is tag-triggered", check_publish_is_tag_triggered),
    ]

    version = ""
    failures = 0
    for label, check in checks:
        try:
            result = check()
            if isinstance(result, str):
                version = result
            print(f"  ok  {label}")
        except CheckFailed as exc:
            failures += 1
            print(f"  FAIL {label}:\n       {exc}", file=sys.stderr)

    # Runs after the loop because it needs the version the first check resolved.
    if version:
        try:
            check_version_is_not_already_released(version)
            print("  ok  version is not already released")
        except CheckFailed as exc:
            failures += 1
            print(f"  FAIL version is not already released:\n       {exc}", file=sys.stderr)

    if args.check_pypi and version:
        try:
            check_pypi_is_current(version)
        except CheckFailed as exc:
            failures += 1
            print(f"  FAIL PyPI is current:\n       {exc}", file=sys.stderr)

    if failures:
        print(f"\n{failures} release check(s) failed.", file=sys.stderr)
        return 1
    print(f"\nAll release checks passed for {version or 'the working tree'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
