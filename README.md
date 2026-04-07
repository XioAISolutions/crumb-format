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

## Local and zero-install workflows

CRUMB now supports three ways to generate useful handoffs without depending on a paid hosted model.

| Workflow | What it does | Example |
| --- | --- | --- |
| Local Ollama generation | Uses a local model for `crumb new` and `crumb compress` | `crumb new task --ollama --title "Fix auth" --goal "Stabilize refresh flow"` |
| Deterministic repo capture | Builds CRUMBs directly from your git diff or directory tree | `crumb new task --from-diff` |
| Zero-install prompts | Lets ChatGPT, Claude, or Cursor emit CRUMBs from a custom instruction file | `prompts/chatgpt_custom_instructions.md` |

If you have [Ollama](https://ollama.com/) running locally, CRUMB can generate and compress handoffs privately with `--ollama` (alias `--use-local`). It checks `http://localhost:11434/api/generate`, defaults to the `llama3` model, and fails cleanly if the local endpoint is unavailable.

```bash
# generate a task crumb with a local model
crumb new task --ollama --title "Fix auth" --goal "Stabilize token refresh race"

# use a specific local model
crumb new task --ollama --ollama-model phi3 --title "Audit billing retry flow" --goal "Identify failure mode"

# compress an existing crumb locally before handoff
crumb compress examples/task-bug-fix.crumb --ollama
```

For cases where you do not want any model involved, the CLI also supports deterministic handoff generation.

```bash
# turn the current branch diff into a task crumb
crumb new task --from-diff --title "Continue current branch"

# turn a repository tree into a map crumb while respecting .gitignore
crumb new map --dir . --title "Repository map"
```

## Zero-install prompts

If you do not want to install anything into the target AI tool, use the ready-made prompt files in [`prompts/`](prompts/):

| File | Target |
| --- | --- |
| [`prompts/chatgpt_custom_instructions.md`](prompts/chatgpt_custom_instructions.md) | ChatGPT custom instructions |
| [`prompts/claude_projects.md`](prompts/claude_projects.md) | Claude Project instructions |
| [`prompts/cursor_rules.md`](prompts/cursor_rules.md) | Cursor rules |

These prompt files teach the model to respond to `/crumb` by emitting a CRUMB v1.1 handoff inside a fenced code block with `BEGIN CRUMB` and `END CRUMB` markers.

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

**Browser extension** -- the unpacked browser extension can now inject a **Copy as CRUMB** button into ChatGPT, Claude, and Gemini, capture the most recent visible exchanges, and copy a `kind=log` handoff directly to your clipboard. See [`browser-extension/README.md`](browser-extension/README.md) and [`browser-extension/INSTALL.md`](browser-extension/INSTALL.md) for setup.

**VS Code snippets** -- the bundled VS Code extension now includes hand-writing snippets for all five CRUMB kinds. Install the extension from [`vscode-extension/`](vscode-extension/), open a `.crumb` file, then type one of these triggers and press `Tab`:

| Trigger | Expands to |
| --- | --- |
| `!crumb-task` | `kind=task` handoff with `[goal]`, `[context]`, and `[constraints]` |
| `!crumb-mem` | `kind=mem` block with `[consolidated]` |
| `!crumb-map` | `kind=map` block with `[project]` and `[modules]` |
| `!crumb-log` | `kind=log` block with timestamped `[entries]` |
| `!crumb-todo` | `kind=todo` block with `[items]` |

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
- [`prompts/`](prompts/) -- zero-install instruction files for ChatGPT, Claude Projects, and Cursor
- [`cli/crumb.py`](cli/crumb.py) -- full CLI (23 commands)
- [`cli/local_ai.py`](cli/local_ai.py) -- local Ollama generation helpers
- [`browser-extension/`](browser-extension/) -- one-click browser handoff capture for AI chat UIs
- [`vscode-extension/`](vscode-extension/) -- syntax highlighting, commands, and CRUMB v1.1 snippets
- [`cli/metalk.py`](cli/metalk.py) -- MeTalk caveman compression layer
- [`validators/`](validators/) -- Python and Node reference validators
- [`docs/HANDOFF_PATTERNS.md`](docs/HANDOFF_PATTERNS.md) -- practical handoff patterns

## License

MIT. See [`TRADEMARK.md`](TRADEMARK.md) for brand guidance.

CRUMB is plain text. It works everywhere text works.
