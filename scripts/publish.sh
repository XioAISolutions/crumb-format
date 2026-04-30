#!/usr/bin/env bash
# Publish crumb-format to PyPI.
#
# Usage:
#   scripts/publish.sh           # uploads the version currently in pyproject.toml
#   scripts/publish.sh --test    # uploads to TestPyPI first
#   scripts/publish.sh --help    # show this help
#
# Prerequisites (run once on your machine):
#   1. PyPI account: https://pypi.org/account/register/
#   2. API token: https://pypi.org/manage/account/token/
#      Save it into ~/.pypirc:
#        [pypi]
#        username = __token__
#        password = pypi-<token>
#   3. Modern toolchain: pip install -U build twine 'packaging>=24.2'
#
# What this script does:
#   - Cleans dist/ and build/ so we never re-upload stale artifacts.
#   - Builds wheel + sdist via `python3 -m build`.
#   - Verifies metadata with `twine check`.
#   - Confirms the version in pyproject.toml matches the latest CHANGELOG entry.
#   - Uploads to PyPI (or TestPyPI with --test).
set -euo pipefail

cd "$(dirname "$0")/.."

# ── Phase 1: parse arguments ────────────────────────────────────────────
# Resolve --help / unknown flags BEFORE running the test suite or build,
# so a typo (`--tset`) or `--help` never burns a minute on tests + a wheel
# build that the user didn't actually want.
#
# Reject extra args too — `publish.sh --test --help` would otherwise run
# the full --test pipeline and silently ignore --help. Same risk class
# as the `--tset` typo: unexpected flags should fail fast, not get
# absorbed into a real upload.
if (( $# > 1 )); then
    echo "ERROR: too many arguments ($#); expected at most one" >&2
    echo "Usage: $0 [--test|--help]" >&2
    exit 2
fi

case "${1:-}" in
    "")
        MODE=prod
        ;;
    --test)
        MODE=test
        ;;
    -h|--help)
        # Print the leading comment block (lines 2–23) as the help text.
        sed -n '2,23p' "$0"
        exit 0
        ;;
    *)
        echo "ERROR: unknown argument: '$1'" >&2
        echo "Usage: $0 [--test|--help]" >&2
        exit 2
        ;;
esac

# ── Phase 2: prechecks ──────────────────────────────────────────────────
# Read project.version from pyproject.toml without depending on tomllib
# (3.11+) — pyproject declares requires-python = ">=3.10", so tomllib import
# would crash this script on 3.10. The grep/sed below works on any Python
# version (and indeed needs no Python at all to extract the version).
VERSION=$(grep -E '^version\s*=' pyproject.toml | head -1 | sed -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/')
if [[ -z "$VERSION" ]]; then
    echo "ERROR: could not parse 'version' from pyproject.toml" >&2
    exit 1
fi
echo "==> Releasing crumb-format v${VERSION}"

# Sanity: CHANGELOG mentions this version. Match exactly:
#   - escape dots in $VERSION so '0.8.0' isn't a regex wildcard for '0x8x0'
#   - require the heading to end after the version (or be followed by
#     whitespace) so '## v0.8.0-rc1' doesn't satisfy 'VERSION=0.8.0'
ESCAPED_VERSION=${VERSION//./\\.}
if ! grep -qE "^## v${ESCAPED_VERSION}([[:space:]]|$)" CHANGELOG.md; then
    echo "ERROR: CHANGELOG.md is missing a '## v${VERSION}' heading." >&2
    exit 1
fi

# Sanity: tests still pass.
echo "==> Running test suite"
python3 -m pytest tests/ -q

# ── Phase 3: build ──────────────────────────────────────────────────────
echo "==> Building artifacts"
rm -rf dist/ build/ ./*.egg-info
python3 -m build

echo "==> Verifying metadata with twine"
python3 -m twine check dist/*

# ── Phase 4: upload ─────────────────────────────────────────────────────
case "$MODE" in
    prod)
        echo "==> Uploading to PyPI"
        python3 -m twine upload dist/*
        echo "Done. Verify with: pip install crumb-format==${VERSION}"
        ;;
    test)
        echo "==> Uploading to TestPyPI"
        python3 -m twine upload --repository testpypi dist/*
        echo "Done. Verify with: pip install -i https://test.pypi.org/simple/ crumb-format==${VERSION}"
        ;;
esac
