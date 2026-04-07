# CRUMB 0.2.1 Draft Release Notes

## Release summary

This draft proposes **CRUMB 0.2.1** as the next 0.2.x release after `v0.2.0`. The release focuses on making CRUMB materially more useful in everyday handoff workflows by improving local generation, deterministic capture, zero-install prompts, browser capture, editor authoring, and overall documentation quality.

In practical terms, this release turns CRUMB into a more complete toolkit for moving work between AI systems. Users can now generate handoffs locally with Ollama, derive handoffs directly from repository state without any model call, seed multiple AI tools with portable prompt files, capture live chats from major AI interfaces with one click, and author CRUMBs faster inside VS Code.

## Highlights

| Area | What is new | Why it matters |
| --- | --- | --- |
| Local generation | Added local Ollama support for `crumb new` and `crumb compress` | Enables private, offline-friendly CRUMB workflows |
| Deterministic generation | Added `crumb new task --from-diff` and `crumb new map --dir` | Produces handoffs directly from repo state with no model dependency |
| Zero-install prompts | Added ready-made prompt files for ChatGPT, Claude Projects, and Cursor | Makes CRUMB easier to adopt across tools |
| Browser capture | Upgraded the extension with a floating **Copy as CRUMB** flow for ChatGPT, Claude, and Gemini | Makes live handoff capture far faster |
| Editor ergonomics | Added CRUMB v1.1 snippets for all five CRUMB kinds in the VS Code extension | Speeds up hand-written CRUMB authoring |
| Polish | Expanded docs, added regression tests, and removed the legacy conflicting snippet file | Improves clarity and release confidence |

## Changelog

### Added

CRUMB now supports local Ollama-backed generation for both `crumb new` and `crumb compress`, including model selection through `--ollama-model` and graceful failure handling when the local endpoint or model is unavailable.

The CLI now supports deterministic handoff generation through `crumb new task --from-diff`, which builds a task handoff from `git diff HEAD`, and `crumb new map --dir <path>`, which builds a repository map from a directory tree while filtering common junk folders and respecting `.gitignore` patterns.

A new `prompts/` directory now includes ready-to-use zero-install instruction files for ChatGPT custom instructions, Claude Projects, and Cursor rules. Each file teaches the target tool to emit a CRUMB v1.1 handoff when the user types `/crumb`.

The browser extension now supports one-click CRUMB capture on ChatGPT, Claude, and Gemini through a floating **Copy as CRUMB** button that generates a `kind=log` handoff from the most recent visible conversation turns.

The VS Code extension now includes `!crumb-task`, `!crumb-mem`, `!crumb-map`, `!crumb-log`, and `!crumb-todo` snippets for CRUMB v1.1 authoring.

### Changed

The top-level README now documents the local Ollama workflow, deterministic generation, zero-install prompts, browser-extension workflow, and updated VS Code snippet usage.

The browser-extension documentation now reflects the new page-level capture flow, supported sites, and the distinction between one-click chat capture and selection-based capture.

The VS Code extension README now reflects the new `!crumb-*` trigger format and CRUMB v1.1 snippet behavior.

### Removed

The legacy `vscode-extension/snippets/crumb.json` file has been removed so the extension no longer conceptually conflicts with the new CRUMB v1.1 snippet set.

### Quality and verification

Regression coverage has been extended for the new prompt assets, browser-extension configuration, and VS Code snippet wiring. The repository test suite passed with **178 tests green** during final verification.

## Suggested GitHub release text

### CRUMB 0.2.1

CRUMB 0.2.1 makes the handoff workflow much more practical across local, browser, and editor-based usage.

This release adds local Ollama support for generation and compression, deterministic handoff generation from git diffs and directory trees, zero-install prompt packs for ChatGPT, Claude Projects, and Cursor, a one-click browser handoff flow for ChatGPT, Claude, and Gemini, and upgraded VS Code snippets for all five CRUMB kinds.

It also improves documentation, adds asset-level regression coverage, and removes the older conflicting snippet file from the VS Code extension.

## Suggested short changelog entry

`0.2.1` adds local Ollama support, deterministic repo-to-CRUMB generation, zero-install prompt packs, one-click browser capture for major AI chat tools, upgraded VS Code snippets, broader documentation, and additional regression coverage.

## Release readiness checklist

| Item | Status | Notes |
| --- | --- | --- |
| Main branch pushed | Ready | Latest work is on `main` at commit `4443ac4` |
| Test suite | Ready | Final verification passed with `178` tests |
| Package build | Ready | Source distribution and wheel built successfully after installing `build` locally |
| Release notes draft | Ready | This document can be used as the basis for the GitHub release |
| Version bump | Pending | `pyproject.toml` still reports `0.2.0` |
| Git tag | Pending | Existing latest version tag is `v0.2.0`, so the next cut should add a new `v0.2.1` tag |
| Changelog file in repo | Optional | There is currently no dedicated top-level `CHANGELOG.md` |

## Release-cut recommendation

The natural next version appears to be **0.2.1**, because the current package metadata still reports `0.2.0` while the repository has accumulated substantial post-tag improvements intended to refine the existing 0.2 line rather than introduce a new major compatibility boundary.

In short, the project looks **functionally ready** for a `0.2.1` release, but I would still make one small release-management pass before publishing: bump the package version from `0.2.0` to `0.2.1`, optionally add a top-level `CHANGELOG.md`, then create and publish the `v0.2.1` tag.
