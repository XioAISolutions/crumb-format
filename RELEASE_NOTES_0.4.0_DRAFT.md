# CRUMB 0.4.0 — v1.2 wire format, standalone

**Status:** draft — under review. Ships the CRUMB format bump from `v=1.1` to `v=1.2` with four fully additive, backward-compatible primitives.

## Why 0.4.0

0.3.0 was the "protocol-grade tooling" release — deterministic `crumb pack`, MemPalace adapter bridge, `crumb lint` safety scanner, golden fixture suite, extension model — all while keeping the wire format frozen at `v=1.1`.

0.4.0 is the first release that bumps the wire format itself. Every v1.2 addition is optional, purely additive, and a v1.1 parser accepts a v1.2 file by ignoring unknown headers and sections (per `SPEC.md §8`). A v1.2 parser accepts both `v=1.1` and `v=1.2`. No handoff CRUMBs currently in the wild break.

## What ships

### 1. Cross-crumb references (`refs`)

Optional `refs=` header and optional `[refs]` section let a CRUMB point at other CRUMBs by id, turning an isolated handoff into a navigable graph.

```text
refs=mem-prefs-abc123, map-web-app-2026q2
---
[refs]
- mem-prefs-abc123  role=style  why=caller prefers terse commits
- map-web-app-2026q2  role=terrain  note=only [modules] is needed
```

Resolution strategy is intentionally left to implementations for v1.2. See [`docs/v1.2-ref-resolution.md`](docs/v1.2-ref-resolution.md) for the four candidate schemes (bare id, content hash, URL, registry, hybrid).

### 2. Foldable sections

A single logical section can now carry two forms — a short summary and a full body — using namespaced section names, preserving the flat grammar:

```text
[fold:context/summary]
JWT middleware races the cookie parser on refresh.

[fold:context/full]
Full repro + stack trace + 40 lines of investigation...
```

Validators accept a fold pair as a substitute for the plain required section (the "fold-satisfies-required" rule). A `/full` without a paired `/summary` is a validation error.

Consumer selection heuristic (size-greedy vs emitter-weighted vs declaration-order vs receiver-delegated) is left open. See [`docs/v1.2-fold-heuristic.md`](docs/v1.2-fold-heuristic.md).

### 3. `[handoff]` primitive

An optional, explicit "next AI do this" block — distinct from `[goal]` (which says what the work is):

```text
[handoff]
- to=any    do=reproduce the failing test in tests/test_auth.py
- to=any    do=propose a fix without landing it
- to=human  do=approve the fix before merge
- [x] reproduced the bug on main@da5e312
```

Lines starting with `- [x]` are treated as completed context (same convention as `kind=todo`). `to`, `do`, `why`, `deadline`, `ref` are advisory — consumers that don't understand the namespaced form treat the line as a bullet.

### 4. Typed content annotations

A section's first non-blank line MAY start with `@type:` to tag the content type:

```text
[context]
@type: code/typescript
export async function requireAuth(req) { ... }
```

Suggested values: `text/markdown` (default), `code/LANG`, `diff/unified`, `json`, `yaml`, `toml`. Unknown types fall back to plain text.

## Backward compatibility

- v1.1 CRUMBs parse unchanged under v1.2 parsers.
- v1.2 CRUMBs parse under v1.1 parsers by ignoring unknown headers (`refs`) and unknown sections (`[refs]`, `[handoff]`, `[fold:*/*]`). The v1.1 parser's required-section check still passes because either (a) the plain required section is present, or (b) — only when the writer used a fold pair — the v1.1 parser will reject the file with "missing required section." Writers emitting v1.2 folds MUST set `v=1.2`.
- `crumb lint` does not flag v1.2 primitives as unknown extensions.

## What's in this release

- [`SPEC.md`](SPEC.md) — bumped to v1.2, added §§9–12 (refs, folds, handoff, typed content)
- [`cli/crumb.py`](cli/crumb.py) — parser accepts `v ∈ {1.1, 1.2}`, enforces fold-satisfies-required, validates v1.2 primitives
- [`validators/validate.py`](validators/validate.py), [`validators/validate.js`](validators/validate.js) — same, for the reference validators
- [`examples/v12-refs.crumb`](examples/v12-refs.crumb), [`v12-fold.crumb`](examples/v12-fold.crumb), [`v12-handoff.crumb`](examples/v12-handoff.crumb), [`v12-typed-content.crumb`](examples/v12-typed-content.crumb) — one example per v1.2 feature
- [`docs/v1.2-ref-resolution.md`](docs/v1.2-ref-resolution.md), [`docs/v1.2-fold-heuristic.md`](docs/v1.2-fold-heuristic.md) — two open design docs with `TODO(author)` blocks

## All v0.3.0 features still ship

Palace, Reflect, MeTalk, Wake, AgentAuth (passport/policy/audit/webhooks), deterministic `crumb pack`, MemPalace bridge, `crumb lint`, REST/A2A bridges, MCP servers, golden fixture suite, 41+ CLI commands — all unchanged.

## Standalone, no external orchestration ties

0.4.0 continues the explicit decision to stay standalone: no bridge code to Weft/WeaveMind, LangGraph, n8n, or external orchestration runtimes. CRUMB is plain text that travels via copy-paste, git, and clipboard. Interop lives in existing, already-shipped bridges (`openai-threads`, `langchain-memory`, `crewai-task`, `autogen`, `claude-project`, MemPalace).

## Install

```bash
pip install crumb-format==0.4.0
```
