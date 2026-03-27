# CRUMB

The copy-paste AI handoff format.

---

Ever been deep into a task with one AI, then need to switch to another? You either paste an enormous chat log and hope it picks up the thread, or you start over and re-explain everything from scratch.

CRUMB is a third option. It's a small, structured text block you copy-paste between AI tools. The next AI gets exactly what it needs to continue your work -- the goal, the context, and the constraints -- without the noise.

## Try it right now

Copy this and paste it into any AI:

```text
BEGIN CRUMB
v=1.1
kind=task
title=Fix login redirect bug
source=cursor.agent
project=web-app
---
[goal]
Fix the bug where authenticated users are redirected back to /login after refresh.

[context]
- App uses JWT cookie auth
- Redirect loop happens only on full page refresh
- Middleware reads auth state before cookie parsing is complete

[constraints]
- Do not change the login UI
- Preserve existing cookie names
- Add a regression check before merging
END CRUMB
```

That's it. The next AI knows what to fix, what it can't change, and why.

## Real-world examples

### Debug handoff across tools

You found the bug in Cursor but need Claude to write the regression test. Instead of pasting 200 lines of chat history, you hand off a 15-line crumb. The receiving AI skips straight to writing the test because the goal, root cause, and constraints are already structured.

### Project memory that follows you

Tired of re-explaining your preferences every session? A `mem` crumb captures your working style and survives across tools and sessions:

```text
BEGIN CRUMB
v=1.1
kind=mem
title=Builder preferences
source=human.notes
---
[consolidated]
- Prefers direct technical answers with minimal fluff
- Wants copy-pasteable outputs when possible
- Cares about launch speed more than theoretical purity
- Prefers solutions that survive switching between AI tools
END CRUMB
```

Three kinds: **task** (what to do next), **mem** (long-term memory), **map** (repo or project map). See the [spec](SPEC.md) for details.

## How it compares

| | Paste raw chat | Start over | Use CRUMB |
|---|---|---|---|
| Context preserved | Partial, noisy | None | Structured, high-signal |
| Next AI acts immediately | Unlikely | No | Yes |
| Works across all AI tools | Yes | Yes | Yes |
| Token-efficient | No | Yes (but lossy) | Yes |
| Human-readable | Barely | N/A | Yes |

## Add to your AI workflow

Add this to your AI's custom instructions and it will generate CRUMBs automatically:

```text
When I say "crumb it" or when a task is being handed off, generate a CRUMB
summarizing the current state. Use this format:

BEGIN CRUMB
v=1.1
kind=task
title=<short description>
source=<this tool>
---
[goal]
<what needs to happen next>

[context]
<key facts, decisions, and current state>

[constraints]
<what must not change>
END CRUMB
```

Works in ChatGPT custom instructions, Claude Projects, Cursor rules, or any AI that accepts system prompts.

## What's in this repo

- [`SPEC.md`](SPEC.md) -- the format specification
- [`DREAMING.md`](DREAMING.md) -- how memory consolidation works
- [`examples/`](examples/) -- ready-to-paste `.crumb` files
- [`cli/crumb.py`](cli/crumb.py) -- CLI for creating and validating crumbs
- [`validators/`](validators/) -- Python and Node reference validators
- [`docs/HANDOFF_PATTERNS.md`](docs/HANDOFF_PATTERNS.md) -- practical handoff patterns

## Quickstart

Validate an example:

```bash
python3 validators/validate.py examples/task-bug-fix.crumb
```

Generate a task handoff from a chat transcript:

```bash
python3 cli/crumb.py from-chat --input chat.txt --output handoff.crumb
```

## License

MIT for spec and reference code. See [`TRADEMARK.md`](TRADEMARK.md) for brand guidance.

CRUMB is plain text. It works everywhere text works.
