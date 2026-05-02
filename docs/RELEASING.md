# Releasing crumb-format

The release path. Run from your local machine (PyPI credentials don't ship in the sandbox).

## One-time setup

```bash
# 1. Make a PyPI account if you don't have one
#    https://pypi.org/account/register/

# 2. Generate an API token (scope: "Entire account" for the first publish)
#    https://pypi.org/manage/account/token/

# 3. Save the token in ~/.pypirc
cat > ~/.pypirc <<'EOF'
[pypi]
username = __token__
password = pypi-<your-token-here>

[testpypi]
username = __token__
password = pypi-<your-testpypi-token-here>  # optional, separate token
EOF
chmod 600 ~/.pypirc

# 4. Modern toolchain (PEP 639 license metadata needs packaging>=24.2)
pip install --upgrade build twine 'packaging>=24.2'
```

## Per-release path

Use the bundled script:

```bash
git checkout main
git pull origin main

# Smoke against TestPyPI first (recommended for any major release)
scripts/publish.sh --test

# Then ship to real PyPI
scripts/publish.sh
```

The script:
- Reads version from `pyproject.toml` (no Python required)
- Confirms `CHANGELOG.md` has a `## v<version>` heading
- Runs the full pytest suite (must be green)
- Cleans `dist/` and `build/`
- Builds wheel + sdist via `python3 -m build`
- Runs `twine check` for metadata sanity
- Uploads via `twine upload`

Reject paths:
- Unknown args fail fast (`scripts/publish.sh --tset` exits 2 immediately)
- `--help` exits without running tests
- Any precheck failure stops before the upload step

## Per-release checklist

| Step | What |
|---|---|
| 1 | All PRs merged into `main` |
| 2 | `cli/crumb.py:CLI_VERSION` and `pyproject.toml:version` match the target version |
| 3 | `tests/test_crumb.py:TestCliVersion` asserts the new version string |
| 4 | `CHANGELOG.md` has a `## vX.Y.Z` heading at the top |
| 5 | `python3 -m pytest tests/ -q` is green |
| 6 | `python3 validators/validate.py examples/*.crumb fixtures/valid/*.crumb fixtures/extensions/*.crumb` is OK on every file |
| 7 | `node validators/validate.js <files>` is OK on every file |
| 8 | `scripts/publish.sh --test` succeeds on TestPyPI |
| 9 | `pip install -i https://test.pypi.org/simple/ crumb-format==X.Y.Z` works in a clean venv |
| 10 | `scripts/publish.sh` succeeds on real PyPI |
| 11 | Tag locally: `git tag -a vX.Y.Z -m "vX.Y.Z" <merge-commit>` |
| 12 | (Optional) `git push origin vX.Y.Z` — note: tag pushes have historically 403'd in this repo's sandbox; the merge commit named `Release X.Y.Z` is the canonical release marker |

## Version policy

Per the 1.0.0 release commitment:

- **Wire format**: stable post-1.0. New SPEC additions must be backward-compatible (additive headers and sections per §8). Anything breaking waits for v2.0.
- **CLI surface**: evolves under semver minor bumps. Deprecation aliases get a 4-release announce window (v0.7's removal landed in v0.11).
- **Validators**: stay in lockstep with the SPEC. `SUPPORTED_VERSIONS` adds new wire-format versions; never removes.

## Troubleshooting

**`twine check` fails with `unrecognized field 'license-expression'`** — your `packaging` is too old. `pip install --upgrade 'packaging>=24.2'`.

**Upload returns `403 Forbidden`** — your API token is wrong, expired, or scoped to the wrong project.

**Upload returns `400 File already exists`** — the version is already on PyPI; you can't re-upload an existing version. Bump the version (even a patch like `1.0.0.post1`) and retry.

**TestPyPI upload succeeds but `pip install -i ...` fails to find the package** — TestPyPI sometimes lags by a minute on indexing. Wait and retry.

## Provenance

Five releases shipped this session:
- v0.7.0 — usability and simplicity pass
- v0.8.0 — guardrails bridge, MCP v1.3 surface, CI bench fix
- v0.9.0 — v1.4 deadlines impl + lint --check-deadlines
- v0.10.0 — lint --check-failure-modes + JS deadline parser
- v0.11.0 — simplification + neutral naming
- v1.0.0 — v1.4 wire format normative + stable spec

PyPI publish for any of them follows this same recipe.
