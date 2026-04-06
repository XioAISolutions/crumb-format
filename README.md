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

## Two real-world scenarios

**Found the bug in Cursor, need Claude to write the test.** Generate a task crumb from Cursor with `crumb it`, paste it into Claude. Claude sees the goal, the surrounding code context, and the constraints -- and writes the test without asking you to re-explain anything.

**Re-explaining your preferences every session?** A mem crumb stores your working style once:

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

Paste it at the start of any session. No more "I like concise answers, don't use emojis, prefer TypeScript..." every time.

Five kinds: `task` (what to do next), `mem` (long-term memory), `map` (repo overview), `log` (session transcript), `todo` (work items).

## How it compares

| | Paste raw chat | Start over | Use CRUMB |
|---|---|---|---|
| Context preserved | Partial, noisy | None | Structured |
| Next AI acts immediately | Unlikely | No | Yes |
| Works across all AI tools | Yes | Yes | Yes |
| Token-efficient | No | Yes (lossy) | Yes |
| Human-readable | Barely | N/A | Yes |

## Add "crumb it" to your AI

Add this to your AI's custom instructions and it generates CRUMBs on command:

```text
When I say "crumb it", generate a CRUMB summarizing the current state.

For tasks: kind=task with [goal], [context], [constraints]
For memory: kind=mem with [consolidated]
For repos: kind=map with [project], [modules]

Format: BEGIN CRUMB / v=1.1 / headers / --- / sections / END CRUMB
```

Works in ChatGPT custom instructions, Claude Projects, Cursor rules, or any AI with system prompts.

## Install

```bash
pip install crumb-format
```

## New in 0.2.0: MeTalk

MeTalk is the caveman-compression layer for CRUMBs. It shrinks handoff text
before you paste it into the next AI.

```bash
# default level 2
crumb metalk examples/task-bug-fix.crumb

# aggressive mode
crumb metalk examples/task-bug-fix.crumb --level 3

# add MeTalk as Stage 3 after TurboQuant compression
crumb compress examples/task-bug-fix.crumb --metalk --metalk-level 2
```

## Quick start

```bash
# create a task crumb
crumb new task --title "Fix auth" --goal "Fix token refresh race condition"

# validate
crumb validate examples/*.crumb

# append observations to memory, then consolidate
crumb append prefs.crumb "Switched to Neovim" "Dropped Redux"
crumb dream prefs.crumb

# search across crumbs (keyword, fuzzy, or TF-IDF ranked)
crumb search "auth JWT" --dir ./crumbs/

# seed all your AI tools at once
crumb init --all

# compress a crumb for cross-AI handoff
crumb metalk examples/task-bug-fix.crumb --level 2
```

Full command reference: `crumb --help` (23 commands including MeTalk compression, export, import, templates, todos, watch mode, and more).

## Integrations

**MCP Server** -- native tool integration with Claude Desktop, Cursor, Claude Code:
```bash
claude mcp add crumb python3 /path/to/crumb-format/mcp/server.py
```
See [`mcp/README.md`](mcp/README.md) for setup.

**Pre-commit hook** -- validate `.crumb` files on every commit:
```yaml
repos:
  - repo: https://github.com/XioAISolutions/crumb-format
    rev: main
    hooks:
      - id: validate-crumbs
```

**ClawHub skill** -- install as an OpenClaw agent skill. See [`clawhub-skill/`](clawhub-skill/).

## What's in this repo

- [`SPEC.md`](SPEC.md) -- the format specification
- [`DREAMING.md`](DREAMING.md) -- how memory consolidation works
- [`examples/`](examples/) -- ready-to-paste `.crumb` files
- [`cli/crumb.py`](cli/crumb.py) -- full CLI (23 commands)
- [`cli/metalk.py`](cli/metalk.py) -- MeTalk caveman compression layer
- [`validators/`](validators/) -- Python and Node reference validators
- [`docs/HANDOFF_PATTERNS.md`](docs/HANDOFF_PATTERNS.md) -- practical handoff patterns

## License

MIT. See [`TRADEMARK.md`](TRADEMARK.md) for brand guidance.

CRUMB is plain text. It works everywhere text works.
