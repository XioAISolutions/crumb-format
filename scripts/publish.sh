#!/usr/bin/env bash
# Publish crumb-format to PyPI.
#
# Usage:
#   scripts/publish.sh           # uploads the version currently in pyproject.toml
#   scripts/publish.sh --test    # uploads to TestPyPI first
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

# Sanity: CHANGELOG mentions this version.
if ! grep -q "^## v${VERSION}" CHANGELOG.md; then
    echo "ERROR: CHANGELOG.md is missing a '## v${VERSION}' heading." >&2
    exit 1
fi

# Sanity: tests still pass.
echo "==> Running test suite"
python3 -m pytest tests/ -q

# Clean and build.
echo "==> Building artifacts"
rm -rf dist/ build/ ./*.egg-info
python3 -m build

# Verify metadata.
echo "==> Verifying metadata with twine"
python3 -m twine check dist/*

# Upload. Strict arg parsing — anything other than "" or "--test" must
# error rather than silently fall through to a production upload (a typo
# like `--tset` is a real risk; treating "unknown == prod" is dangerous).
case "${1:-}" in
    "")
        echo "==> Uploading to PyPI"
        python3 -m twine upload dist/*
        echo "Done. Verify with: pip install crumb-format==${VERSION}"
        ;;
    --test)
        echo "==> Uploading to TestPyPI"
        python3 -m twine upload --repository testpypi dist/*
        echo "Done. Verify with: pip install -i https://test.pypi.org/simple/ crumb-format==${VERSION}"
        ;;
    -h|--help)
        sed -n '2,16p' "$0"
        exit 0
        ;;
    *)
        echo "ERROR: unknown argument: '$1'" >&2
        echo "Usage: $0 [--test|--help]" >&2
        exit 2
        ;;
esac
