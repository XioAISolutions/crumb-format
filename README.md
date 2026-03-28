# CRUMB

[![Validate and test](https://github.com/XioAISolutions/crumb-format/actions/workflows/validate-examples.yml/badge.svg)](https://github.com/XioAISolutions/crumb-format/actions/workflows/validate-examples.yml)
[![CRUMB ready](https://img.shields.io/badge/CRUMB-ready-orange)](https://github.com/XioAISolutions/crumb-format)

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

### Teach your AI to receive crumbs

The instruction above teaches an AI to *generate* crumbs. Add this to teach it to *act on* crumbs it receives:

```text
When you receive text containing BEGIN CRUMB / END CRUMB markers, treat it
as a structured handoff. Parse the headers and sections, then act on the crumb
directly — do not ask the user to re-explain what is already in the crumb.
```

Both sides of the loop matter. A crumb is only viral if the receiving AI knows what to do with it.

## Install

```bash
pip install crumb-format
```

Or clone and use directly:

```bash
git clone https://github.com/XioAISolutions/crumb-format.git
python3 cli/crumb.py --help
```

## CLI

Create a crumb from the command line:

```bash
# create a task crumb
python3 cli/crumb.py new task \
  --title "Fix auth" \
  --source cursor.agent \
  --goal "Fix the token refresh race condition" \
  --context "App uses JWT" "Refresh token expires silently" \
  --constraints "Don't change login flow"

# create a mem crumb
python3 cli/crumb.py new mem \
  --title "My prefs" \
  --source human.notes \
  --entries "Prefers TypeScript" "No ORMs" "Keep it simple"

# create a map crumb
python3 cli/crumb.py new map \
  --title "My API" \
  --source human.notes \
  --project myapp \
  --description "REST API for task management" \
  --modules "src/routes: endpoints" "src/db: database layer"
```

Convert a chat log into a crumb (auto-extracts decisions and code blocks):

```bash
python3 cli/crumb.py from-chat --input chat.txt --output handoff.crumb

# extract just decisions as a mem crumb
python3 cli/crumb.py from-chat --input chat.txt --kind mem --title "Stack decisions"

# from clipboard (macOS)
pbpaste | python3 cli/crumb.py from-chat --title "Continue auth work" --source claude.chat
```

Validate and inspect:

```bash
# validate (uses full spec parser, not just marker checks)
python3 cli/crumb.py validate examples/*.crumb

# inspect a crumb's structure
python3 cli/crumb.py inspect examples/task-bug-fix.crumb

# headers and section names only
python3 cli/crumb.py inspect examples/task-bug-fix.crumb --headers-only
```

Memory lifecycle (append → dream → search → merge):

```bash
# append raw observations to an existing mem crumb
python3 cli/crumb.py append prefs.crumb "Switched to Neovim" "Dropped Redux for Zustand"

# run a consolidation pass: deduplicate, merge [raw] → [consolidated], prune to budget
python3 cli/crumb.py dream prefs.crumb
python3 cli/crumb.py dream prefs.crumb --dry-run  # preview without writing

# search across all .crumb files (keyword, fuzzy, or ranked)
python3 cli/crumb.py search "auth JWT" --dir ./crumbs/
python3 cli/crumb.py search "authenication" --dir ./crumbs/ --method fuzzy   # typo-tolerant
python3 cli/crumb.py search "auth JWT" --dir ./crumbs/ --method ranked       # TF-IDF scoring

# merge multiple mem crumbs into one
python3 cli/crumb.py merge team/*.crumb --title "Team preferences" -o merged.crumb
```

Initialize CRUMB in any project:

```bash
# generates a map crumb + prints custom instruction snippets
python3 cli/crumb.py init --project myapp --description "REST API for tasks"

# also create/update CLAUDE.md with CRUMB instructions
python3 cli/crumb.py init --project myapp --claude-md
```

Compare and compress:

```bash
# see what changed between dream passes or versions
crumb diff prefs-v1.crumb prefs-v2.crumb

# strip a crumb to minimum viable form (required headers + sections only)
crumb compact handoff.crumb -o slim.crumb
```

Export to other formats:

```bash
# JSON (for APIs, databases, integrations)
crumb export handoff.crumb -f json -o handoff.json

# Markdown (for docs, wikis, PRs)
crumb export handoff.crumb -f markdown

# Clipboard-friendly (for pasting into ChatGPT, Claude, Cursor)
crumb export handoff.crumb -f clipboard
```

Import from other formats:

```bash
# import from JSON
crumb import --from json -i data.json -o handoff.crumb

# import from structured markdown
crumb import --from markdown -i notes.md -o handoff.crumb
```

Templates — start from a proven pattern:

```bash
# see all available templates
crumb template list

# scaffold from a template
crumb template use bug-fix -o fix.crumb
crumb template use feature -o feature.crumb
crumb template use onboarding -o onboard.crumb

# save your own template
crumb template add my-template my-handoff.crumb
```

Automation hooks (`.crumbrc`):

```ini
# .crumbrc — runs shell commands on crumb events
[hooks]
post_dream = git add *.crumb && git commit -m 'dream pass'
post_append = echo "Entry added to $CRUMB_FILE"
```

```bash
# see configured hooks
crumb hooks
```

Node validator also available:

```bash
node validators/validate.js examples/task-bug-fix.crumb
```

## GitHub Action

Add CRUMB validation to any repo's CI:

```yaml
# .github/workflows/validate-crumbs.yml
name: Validate .crumb files
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: XioAISolutions/crumb-format@main
        with:
          path: '**/*.crumb'
```

Add the badge to your README:

```markdown
[![CRUMB ready](https://img.shields.io/badge/CRUMB-ready-orange)](https://github.com/XioAISolutions/crumb-format)
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
