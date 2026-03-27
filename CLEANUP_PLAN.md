# Repo Cleanup Plan

What to delete, what to keep, and why.

## Delete

| File | Reason |
|---|---|
| `DEPLOY_NOW.md` | Internal launch checklist. Not relevant to users. |
| `RELEASE_NOTES_v0.1.0.md` | Use GitHub Releases instead. |
| `ROADMAP.md` | Placeholder. Use Issues/Discussions for roadmap. |
| `CODE_OF_CONDUCT.md` | Boilerplate governance. Premature for a pre-1.0 spec with no external contributors. |
| `CONTRIBUTING.md` | Same. Add back when contributors exist. |
| `SECURITY.md` | A text format has no security vulnerabilities. Signals AI-generated scaffolding. |
| `FAQ.md` | Key content folded into the new README. |
| `posts/` (entire directory) | Marketing drafts (LinkedIn, X, Reddit, HN). Don't belong in the repo. |
| `demo/BEFORE_AFTER_DEMO.md` | The README now serves as the demo. |
| `docs/CRUMB_vs_CLAUDE_MD.md` | Defensive positioning against specific vendors. Ages poorly. |
| `docs/CRUMB_vs_AGENTS_MD.md` | Same. |

## Keep

| File | Reason |
|---|---|
| `README.md` | Rewritten. Developer-first, conversion-oriented. |
| `SPEC.md` | Core specification. The actual product. |
| `DREAMING.md` | Consolidation/dream-pass guidance. Unique and valuable. |
| `TRADEMARK.md` | Necessary for brand protection. |
| `LICENSE` | Required. |
| `cli/crumb.py` | Functional tooling. |
| `validators/` | Python and Node validators. |
| `examples/` | Core to the repo's purpose. |
| `docs/HANDOFF_PATTERNS.md` | Practical, pattern-oriented. |
| `.github/` | CI workflow for example validation. |

## Target structure

```
README.md
SPEC.md
DREAMING.md
CLEANUP_PLAN.md
TRADEMARK.md
LICENSE
cli/
validators/
examples/
docs/          (HANDOFF_PATTERNS.md only)
.github/
```

## Commands

```bash
git rm DEPLOY_NOW.md RELEASE_NOTES_v0.1.0.md ROADMAP.md CODE_OF_CONDUCT.md CONTRIBUTING.md SECURITY.md FAQ.md
git rm demo/BEFORE_AFTER_DEMO.md
git rm docs/CRUMB_vs_CLAUDE_MD.md docs/CRUMB_vs_AGENTS_MD.md
git rm -r posts/
```
