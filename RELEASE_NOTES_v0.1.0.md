# CRUMB v0.1.0 🍞

CRUMB is a **copy-paste AI handoff format** for moving work between AIs without losing context.

This first release ships the **text-first** foundation:
- a small human-readable spec
- a dreaming/consolidation guide
- validators
- a tiny CLI
- starter examples

## Why this exists

When work moves from one AI tool, session, or teammate to another, the important context usually gets lost.

People end up:
- pasting giant transcripts
- rewriting the same background
- losing constraints, decisions, and next steps

CRUMB gives that handoff a compact shape that survives ordinary copy-paste.

## What’s in v0.1.0

- `SPEC.md` — CRUMB v1.1 draft spec
- `DREAMING.md` — consolidation guidance for long-term memory
- `validators/validate.py` — Python validator
- `validators/validate.js` — Node validator
- `cli/crumb.py` — tiny CLI for generating task handoffs from chat logs
- `examples/` — starter crumbs for task continuation, bug fixing, memory/preferences, and repo onboarding

## Core idea

CRUMB is:
- plain UTF-8 text
- readable by humans
- understandable by LLMs without a decoder
- small enough to paste into chats, docs, notes, and issues

CRUMB is not:
- primarily a compression format
- a replacement for MCP, A2A, JSON, or OpenAPI
- a claim of native vendor support today

## Core file kinds

- `kind=task` — what to do next
- `kind=mem` — consolidated long-term memory
- `kind=map` — repo or project map

## Recommended first demo

1. Start work in one AI tool
2. Convert the current state into a `.crumb`
3. Paste that `.crumb` into another AI
4. Show that it continues cleanly

Pass the crumb. 🍞
