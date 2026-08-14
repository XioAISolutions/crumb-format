# Releasing crumb-format

**Merging a version bump to `main` is the release.** Everything else is
automation. Land a PR that sets the new version in `pyproject.toml`,
`cli/crumb.py`, and `CHANGELOG.md`, and the merge publishes it.

There is no separate ceremony, and nothing to remember to do afterwards. The
publish workflow reads the version from `pyproject.toml`, checks whether
`v<version>` has ever been tagged, and releases it if not — creating the tag,
the GitHub Release, and the PyPI upload in one run. Merges that do not change
the version see an existing tag and stop at the gate.

## The other two entry points

Both still work and both are honoured even when the tag already exists, since
each is a deliberate act:

```bash
# A tag push, if your credentials can create tags on this repo:
git checkout main && git pull origin main
python scripts/release_check.py          # versions, docs, publish path
git tag -a v1.2.0 -m "v1.2.0"
git push origin v1.2.0

# Or dispatch it by hand from the Actions tab ("Publish to PyPI" → Run
# workflow → tag: v1.2.0), which needs `actions: write`.
```

Any of the three triggers `.github/workflows/publish-pypi.yml`, which verifies the
release metadata, runs the tests, builds the wheel, **installs it into a clean
venv outside the source tree and runs the README's own quickstart against it**,
creates the GitHub Release, and publishes to PyPI via Trusted Publishing (no
API token required). `.github/workflows/verify-pypi.yml` then reinstalls from
the public index on 3.10/3.11/3.12 and re-runs the quickstart.

## Why this document changed

crumb-format served **0.2.0 on PyPI from April to August 2026** while the
repository moved to 1.1.0 and gained ~40 commands. Anyone who ran
`pip install crumb-format` in that window got a package where the README's own
first command did not exist.

Three compounding causes, all now fixed:

1. The publish workflow triggered on `release: published`, requiring a
   hand-created GitHub Release. Between the workflow landing (May 28) and
   August, nobody created one, so it never ran. It is now tag-triggered.
2. This very checklist listed the tag push as **"(Optional)"**, noting that
   tag pushes "have historically 403'd" and declaring the merge commit the
   "canonical release marker." Nothing downstream read merge commits, so that
   marker meant nothing.

   The 403 was real, and it is still real: pushing `v1.2.0` from this org's
   automation is rejected at `git-receive-pack` for any ref outside the
   session's allowed branch, and `POST /actions/workflows/.../dispatches`
   answers `403 Resource not accessible by integration` for the same identity.
   Rather than leaving the release gated on a human at an unrestricted
   terminal — the exact dependency that let 0.2.0 sit on PyPI for four months
   — the merge commit is now genuinely load-bearing: a push to `main` carrying
   an untagged version releases, and the workflow creates the tag itself.
   `tests/test_release_metadata.py` fails if that branch trigger is ever
   removed, or if the gate that prevents republishing is bypassed.
3. Nothing compared the published version to the working tree. A weekly
   `detect-drift` job now does, and `scripts/release_check.py --check-pypi`
   does it on demand.

## Manual fallback

`scripts/publish.sh` still works for a local upload if CI is unavailable, and
needs PyPI credentials in `~/.pypirc`. Prefer the tag path — the manual script
skips the out-of-tree install proof, which is the check that catches a wheel
that cannot run its own documentation.

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
| 8 | `python scripts/release_check.py` exits 0 |
| 9 | Tag: `git tag -a vX.Y.Z -m "vX.Y.Z"` |
| 10 | **`git push origin vX.Y.Z`** — this is the release. Not optional. If it 403s, fix the token or branch protection; skipping it is why PyPI sat four months stale |
| 11 | Watch `publish-pypi.yml` go green; confirm the GitHub Release was created |
| 12 | Confirm `pip install crumb-format==X.Y.Z` resolves from public PyPI, and that `verify-pypi.yml` passed on all three Python versions |

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
