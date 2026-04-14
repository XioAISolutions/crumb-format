---
name: crumb
description: Create, manage, and search structured AI handoff files using the CRUMB format. Includes Palace spatial memory, self-learning gap detection (reflect), session bootstrap (wake), and token compression (metalk).
version: 0.2.0
metadata:
  openclaw:
    requires:
      bins: [python3]
      anyBins: [crumb, python3]
    primaryEnv: ""
    always: false
    skillKey: crumb
    emoji: "\U0001F9C0"
    homepage: https://github.com/XioAISolutions/crumb-format
    os: ["macos", "linux", "windows"]
    install:
      - kind: uv
        package: crumb-format
        bins: [crumb]
---

# CRUMB — Structured AI Handoff Format

You have access to the `crumb` CLI for creating and managing `.crumb` handoff files. CRUMB is a plain-text format for moving context between AI tools without losing information.

## When to use CRUMB

- When the user says **"crumb it"** — generate a crumb summarizing current work
- When the user **pastes a BEGIN CRUMB / END CRUMB block** — parse and act on it
- When the user wants to **hand off work** to another AI tool
- When the user wants to **save preferences** that persist across sessions
- When the user asks to **create, validate, search, or manage** crumb files

## Available commands

### Create crumbs

```bash
# Create a task handoff
crumb new task --title "Fix auth bug" --source "openclaw" \
  --goal "Fix the redirect loop" \
  --context "Uses JWT auth" "Middleware ordering issue" \
  --constraints "Don't change login UI" \
  -o handoff.crumb

# Create a memory crumb
crumb new mem --title "User preferences" --source "openclaw" \
  --entries "Prefers TypeScript" "No ORMs" "Terse responses" \
  -o prefs.crumb

# Create a project map
crumb new map --title "Project map" --source "openclaw" \
  --project myapp --description "REST API for tasks" \
  --modules "src/routes" "src/db" \
  -o map.crumb

# Create a todo tracker
crumb new todo --title "Sprint tasks" --source "openclaw" \
  --entries "Fix auth" "Add tests" \
  -o tasks.crumb
```

### Start from a template

```bash
crumb template list
crumb template use <template-name> -o out.crumb
```

### Validate and inspect

```bash
crumb validate examples/*.crumb
crumb inspect handoff.crumb
crumb inspect handoff.crumb --headers-only
```

### Memory lifecycle

```bash
# Append observations
crumb append prefs.crumb "Switched to Neovim" "Dropped Redux"

# Consolidate: deduplicate, merge raw into consolidated, prune to budget
crumb dream prefs.crumb
crumb dream prefs.crumb --dry-run

# Search across crumb files
crumb search "auth JWT" --dir ./crumbs/
crumb search "auth" --dir ./crumbs/ --method fuzzy    # typo-tolerant
crumb search "auth" --dir ./crumbs/ --method ranked   # TF-IDF scoring

# Merge multiple mem crumbs
crumb merge team/*.crumb --title "Team prefs" -o merged.crumb
```

### Palace (spatial memory)

Persistent knowledge base organized as **wings → halls → rooms** (`.crumb` files on disk).

```bash
# Add a fact/event/discovery/preference/advice to the palace
crumb palace add --wing project:myapp --hall facts --text "Uses JWT auth"
crumb palace add --wing person:alice --hall preferences --text "Prefers TypeScript"

# List wings / halls / rooms
crumb palace list
crumb palace list --wing project:myapp

# Search across the palace
crumb palace search "auth"
crumb palace search "auth" --wing project:myapp

# Render a wing as a wiki-style overview
crumb palace wiki --wing project:myapp
```

### Reflect (self-learning gap detection)

Scores palace health (0-100, grades A-F) and surfaces missing knowledge.

```bash
crumb reflect                      # human-readable report
crumb reflect --crumb               # emit as a mem crumb
crumb reflect --stale-days 60       # flag rooms older than N days
```

### Wake (session bootstrap)

Generate an instant-context crumb for a new AI session from your palace.

```bash
crumb wake                          # emit wake crumb to stdout
crumb wake --reflect                # include [gaps] section
crumb wake -o wake.crumb
```

### Handoff / receive / context (cross-AI interop)

```bash
# Build a handoff from git commits + optional goal
crumb context --commits 10 --goal "Ship auth fix" -o handoff.crumb
crumb context --commits 5 --clipboard            # copy to clipboard
crumb context --commits 5 --metalk --metalk-level 2   # compress

# Wrap/unwrap for different AI tools
crumb handoff task.crumb --target claude -o claude.md
crumb handoff task.crumb --target cursor

# Receive a crumb shared by another AI and stash it in the palace
crumb receive --file incoming.crumb --palace --wing project:myapp --hall facts
```

### MeTalk (token compression)

Caveman-style compression: dictionary substitution + grammar stripping.

```bash
crumb metalk task.crumb                # default level 2 (~40% savings)
crumb metalk task.crumb --level 1      # lossless, reversible
crumb metalk task.crumb --level 3      # aggressive (~50-60% savings)
crumb metalk task.crumb --decode       # reverse Layer 1 substitutions
crumb metalk task.crumb -o slim.crumb
```

### Todo workflow

```bash
crumb todo-add tasks.crumb "Add caching" "Update docs"
crumb todo-done tasks.crumb "caching"
crumb todo-list tasks.crumb
crumb todo-list tasks.crumb --all
crumb todo-dream tasks.crumb   # archive completed tasks
```

### Session logging

```bash
crumb log session.crumb "Found the bug" "Fixed middleware"
```

### Compare and compress

```bash
crumb diff prefs-v1.crumb prefs-v2.crumb
crumb compact handoff.crumb -o slim.crumb
```

### Export and import

```bash
crumb export handoff.crumb -f json -o handoff.json
crumb export handoff.crumb -f markdown
crumb export handoff.crumb -f clipboard

crumb import --from json -i data.json -o handoff.crumb
crumb import --from markdown -i notes.md -o handoff.crumb
```

### Initialize CRUMB in a project

```bash
crumb init --project myapp --description "REST API" --claude-md
```

## CRUMB format reference

There are 6 kinds of crumb:

| Kind | Purpose | Required sections |
|------|---------|-------------------|
| `task` | Hand off work | `[goal]` `[context]` `[constraints]` |
| `mem` | Persistent preferences/knowledge | `[consolidated]` |
| `map` | Codebase overview | `[project]` `[modules]` |
| `log` | Append-only session transcript | `[entries]` |
| `todo` | Track work items | `[tasks]` |
| `wake` | Session bootstrap from palace | `[identity]` |

Every crumb has this structure:

```
BEGIN CRUMB
v=1.1
kind=<task|mem|map|log|todo|wake>
title=<short description>
source=<tool that created it>
---
[section]
- content here
END CRUMB
```

## Behavior rules

1. When asked to "crumb it", generate a crumb that captures the **current goal, context, and constraints** of the conversation. Use `kind=task` unless the user specifies otherwise.
2. When you receive a `BEGIN CRUMB / END CRUMB` block, parse it and **act on it directly** without asking the user to re-explain.
3. When working with crumb files, prefer the CLI commands over manual text editing.
4. Always set `source=openclaw` when generating crumbs.
5. For memory crumbs, use `crumb append` + `crumb dream` rather than rewriting the whole file.
6. When the user has a `.crumbrc` file, hooks will fire automatically after dream and append operations.
