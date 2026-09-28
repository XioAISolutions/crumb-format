"""Every `.crumb` file in this repository must be a valid CRUMB.

The reference implementation shipping files its own validator rejects is the
sharpest possible version of the drift this project has already had twice. It
was happening: `.github/workflows/auto-crumb-pr.yml` generated `.crumb/latest.crumb`
on every pull request using YAML front matter — `--- task: ...`, a `source:`
line, a closing `---` — with no `BEGIN CRUMB` marker, no `v=`, no `kind=`, and
no `END CRUMB`. It was not a CRUMB, and the workflow committed it anyway,
because nothing validated it.

Files that are *deliberately* invalid are enumerated below rather than matched
by a loose pattern, so a genuinely broken file cannot hide behind a wildcard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

import crumb_core  # noqa: E402

# Directories whose contents exist to be rejected.
DELIBERATELY_INVALID_DIRS = (
    REPO_ROOT / "fixtures" / "invalid",
)
# Conformance cases named reject-* are required to fail; the conformance suite
# asserts that separately.
def _is_deliberately_invalid(path: Path) -> bool:
    if any(str(path).startswith(str(d)) for d in DELIBERATELY_INVALID_DIRS):
        return True
    return path.parent.name == "cases" and path.name.startswith("reject-")


def _tracked_crumbs() -> list[Path]:
    return sorted(
        p for p in REPO_ROOT.rglob("*.crumb")
        if p.is_file() and ".git" not in p.parts and not _is_deliberately_invalid(p)
    )


CRUMBS = _tracked_crumbs()


def test_repository_contains_crumb_files():
    assert CRUMBS, "no .crumb files discovered — the glob is wrong"


@pytest.mark.parametrize("path", CRUMBS, ids=[str(p.relative_to(REPO_ROOT)) for p in CRUMBS])
def test_every_repository_crumb_parses(path: Path):
    crumb_core.parse_crumb(path.read_text(encoding="utf-8"))


def test_the_auto_generated_pr_crumb_is_valid():
    """The specific file that was invalid, called out so the fix cannot regress."""
    generated = REPO_ROOT / ".crumb" / "latest.crumb"
    if not generated.is_file():
        pytest.skip("no generated PR crumb in this checkout")
    parsed = crumb_core.parse_crumb(generated.read_text(encoding="utf-8"))
    assert parsed["headers"]["kind"] == "task"
    assert parsed["headers"]["v"] == crumb_core.WIRE_VERSION


def test_deliberately_invalid_fixtures_are_actually_invalid():
    """If these start parsing, the exclusion list is hiding a real regression."""
    invalid = [
        p for d in DELIBERATELY_INVALID_DIRS if d.is_dir()
        for p in sorted(d.glob("*.crumb"))
    ]
    assert invalid, "no invalid fixtures found — the exclusion list may be stale"
    for path in invalid:
        with pytest.raises(ValueError):
            crumb_core.parse_crumb(path.read_text(encoding="utf-8"))
