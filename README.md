# CRUMB 🍞

[![Validate examples](https://github.com/XioAISolutions/crumb-format/actions/workflows/validate-examples.yml/badge.svg)](https://github.com/XioAISolutions/crumb-format/actions/workflows/validate-examples.yml)

**CRUMB is the copy-paste AI handoff format.**

> **Switch AIs without losing the plot.**
>
> Pass the crumb. 🍞

CRUMB exists for one common failure: when work moves from one AI tool, session, or teammate to another, the important context gets lost and people have to re-explain everything.

CRUMB gives that handoff a small, structured, plain-text shape that survives ordinary copy-paste.

## Why people care

- You can paste it into another AI without extra tooling
- It is readable by humans and LLMs
- It is small enough to travel through chats, docs, issues, and notes
- It keeps the next step clear instead of dumping raw history

## Quick example

```text
BEGIN CRUMB
v=1.1
kind=task
title=Continue dark mode work
source=chatgpt.chat
max_index_tokens=512
max_total_tokens=2048
---
[goal]
Finish the dark mode feature without changing app navigation.

[context]
- Theme context exists.
- Settings toggle UI is half-done.
- Persistence is not wired yet.

[constraints]
- Keep Expo setup unchanged.
- No new dependencies.
END CRUMB
```

Paste that into another AI and it can continue the work with less re-explaining.

## Core file kinds

- `kind=task` — what to do next
- `kind=mem` — consolidated long-term memory
- `kind=map` — repo or project map

## What CRUMB is

CRUMB is:
- plain UTF-8 text
- readable by humans
- parseable by tiny scripts
- understandable by LLMs without a decoder
- durable under chat, docs, notes, issues, commits, and email copy-paste

CRUMB is not:
- primarily a compression format
- a replacement for MCP, A2A, JSON, or OpenAPI
- a promise that every vendor supports it natively today

## Repository contents

- `SPEC.md` — CRUMB v1.1 spec
- `DREAMING.md` — consolidation guidance
- `FAQ.md` — quick answers to common questions
- `examples/` — ready-to-paste CRUMB files
- `examples/README.md` — how to use the example crumbs
- `docs/HANDOFF_PATTERNS.md` — practical handoff patterns
- `docs/CRUMB_vs_CLAUDE_MD.md` — where CRUMB fits vs `CLAUDE.md`
- `docs/CRUMB_vs_AGENTS_MD.md` — where CRUMB fits vs `AGENTS.md`
- `validators/` — Python and Node validators
- `cli/crumb.py` — tiny CLI for creating and validating `.crumb` files
- `.github/workflows/validate-examples.yml` — CI for example validation

## Start here

- Read [`FAQ.md`](FAQ.md)
- Read [`SPEC.md`](SPEC.md)
- Read [`DREAMING.md`](DREAMING.md)
- Browse [`examples/README.md`](examples/README.md)
- Compare [`CRUMB vs CLAUDE.md`](docs/CRUMB_vs_CLAUDE_MD.md)
- Compare [`CRUMB vs AGENTS.md`](docs/CRUMB_vs_AGENTS_MD.md)

## Quickstart

Validate an example:

```bash
python3 validators/validate.py examples/task-feature-continuation.crumb
node validators/validate.js examples/task-feature-continuation.crumb
```

Generate a task handoff from a chat transcript:

```bash
python3 cli/crumb.py from-chat --input chat.txt --output handoff.crumb
```

## Positioning

- **Category:** AI handoff format
- **Tagline:** Switch AIs without losing the plot.
- **Pitch:** CRUMB is a copy-paste handoff format for moving work between AIs without losing context.

## Status

- Spec status: Draft RFC
- Canonical form: text-first `.crumb`
- Optional binary transport: out of core spec scope for now
- Release notes and launch post drafts are included in the repo

## License

MIT for spec and reference code. See `TRADEMARK.md` for brand guidance.
