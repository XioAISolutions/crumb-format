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

## Three kinds

Pick the kind that matches what you're handing off:

| Kind | Use when | Required sections |
|---|---|---|
| `task` | You know the next action | `[goal]` `[context]` `[constraints]` |
| `mem` | Preferences that survive across sessions | `[consolidated]` |
| `map` | An AI needs to understand a codebase | `[project]` `[modules]` |

### task -- what to do next

The example above is a task crumb. Use it for bug fixes, feature continuations, code reviews, or any handoff where the next step is clear.

### mem -- long-term memory

Capture your working style, project conventions, or decisions that should persist across sessions and tools:

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

Paste this at the start of any AI session. No more "I like concise answers, don't use emojis, prefer TypeScript..." every time.

### map -- repo or project structure

Onboard a new AI to your codebase in seconds instead of letting it guess the architecture:

```text
BEGIN CRUMB
v=1.1
kind=map
title=CRUMB repo onboarding
source=human.notes
project=crumb-format
---
[project]
CRUMB is a text-first AI handoff format for moving work between tools without losing context.

[modules]
- SPEC.md: core format specification
- DREAMING.md: memory consolidation guidance
- validators/: reference validators in Python and Node
- cli/: tiny helper CLI
- examples/: handoff examples to copy and adapt

[invariants]
- The canonical form is plain text .crumb
- Unknown headers and sections should be ignored, not rejected
- The format should stay small enough to paste into ordinary chats
END CRUMB
```

## Templates

Blank templates you can copy and fill in:

<details>
<summary><b>task template</b></summary>

```text
BEGIN CRUMB
v=1.1
kind=task
title=
source=
---
[goal]


[context]


[constraints]

END CRUMB
```

</details>

<details>
<summary><b>mem template</b></summary>

```text
BEGIN CRUMB
v=1.1
kind=mem
title=
source=
---
[consolidated]

END CRUMB
```

</details>

<details>
<summary><b>map template</b></summary>

```text
BEGIN CRUMB
v=1.1
kind=map
title=
source=
project=
---
[project]


[modules]

END CRUMB
```

</details>

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
When I say "crumb it", generate a CRUMB summarizing the current state.

For tasks and handoffs, use kind=task:
  BEGIN CRUMB
  v=1.1
  kind=task
  title=<short description>
  source=<this tool>
  ---
  [goal]       <what needs to happen next>
  [context]    <key facts, decisions, current state>
  [constraints] <what must not change>
  END CRUMB

For preferences and memory, use kind=mem:
  BEGIN CRUMB
  v=1.1
  kind=mem
  title=<topic>
  source=<this tool>
  ---
  [consolidated] <durable facts, preferences, decisions>
  END CRUMB

For repo/project overviews, use kind=map:
  BEGIN CRUMB
  v=1.1
  kind=map
  title=<project name>
  source=<this tool>
  ---
  [project]  <one-line description>
  [modules]  <key files and directories>
  END CRUMB
```

Works in ChatGPT custom instructions, Claude Projects, Cursor rules, or any AI that accepts system prompts.

## CLI and validators

Validate a crumb:

```bash
python3 validators/validate.py examples/task-bug-fix.crumb
# or
node validators/validate.js examples/task-bug-fix.crumb
```

Generate a task crumb from a chat transcript:

```bash
# from a file
python3 cli/crumb.py from-chat --input chat.txt --output handoff.crumb

# from clipboard (macOS)
pbpaste | python3 cli/crumb.py from-chat --title "Continue auth work" --source claude.chat
```

## What's in this repo

- [`SPEC.md`](SPEC.md) -- the format specification
- [`DREAMING.md`](DREAMING.md) -- how memory consolidation works
- [`examples/`](examples/) -- ready-to-paste `.crumb` files for every kind
- [`cli/crumb.py`](cli/crumb.py) -- CLI for creating and validating crumbs
- [`validators/`](validators/) -- Python and Node reference validators
- [`docs/HANDOFF_PATTERNS.md`](docs/HANDOFF_PATTERNS.md) -- practical handoff patterns

## License

MIT for spec and reference code. See [`TRADEMARK.md`](TRADEMARK.md) for brand guidance.

CRUMB is plain text. It works everywhere text works.
