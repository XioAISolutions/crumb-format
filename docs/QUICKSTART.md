# CRUMB Quickstart

Get productive with CRUMB in five minutes.

## Install

```bash
pip install crumb-format
```

## 1. Create your first crumb

```bash
crumb new task --title "Fix auth bug" --goal "Fix token refresh race condition"
```

This prints a valid `.crumb` to stdout. Save it to a file:

```bash
crumb new task --title "Fix auth bug" \
  --goal "Fix token refresh race condition" \
  -o auth-fix.crumb
```

## 2. Validate and inspect

```bash
crumb validate auth-fix.crumb
crumb inspect auth-fix.crumb
```

## 3. Hand off to another AI

Copy the crumb to your clipboard and paste it into any AI tool:

```bash
crumb handoff auth-fix.crumb
```

Or generate a context crumb from your current project state automatically:

```bash
crumb context --goal "Fix the login redirect bug"
crumb context --clipboard   # copies straight to clipboard
```

## 4. Receive a crumb from another AI

If another AI generated a crumb and you copied it:

```bash
crumb receive                           # reads from clipboard
crumb receive --file handoff.crumb      # reads from file
crumb receive --palace --wing myproject  # auto-files to palace memory
```

## 5. Build persistent memory with Palace

Palace gives you hierarchical memory that survives across sessions.

```bash
# Initialize
crumb palace init

# Add observations — hall is auto-classified
crumb palace add "decided to use Postgres" --wing myproject --room db-choice
crumb palace add "prefers concise code style" --wing myproject --room style
crumb palace add "shipped MVP on June 1st" --wing myproject --room launch

# Browse
crumb palace list --wing myproject
crumb palace search "postgres"

# Wake up a new AI session with your full context (~170 tokens)
crumb wake
```

## 6. Track work with todos

```bash
crumb todo-add sprint.crumb "Add rate limiting" "Write payment tests"
crumb todo-list sprint.crumb
crumb todo-done sprint.crumb "rate limiting"
crumb todo-dream sprint.crumb   # archive completed tasks
```

## 7. Compress for token efficiency

```bash
# Two-stage compression (dedup + signal pruning)
crumb compress auth-fix.crumb

# MeTalk caveman compression for AI-to-AI messages
crumb metalk auth-fix.crumb              # level 2 (default, ~40% savings)
crumb metalk auth-fix.crumb --level 3    # aggressive (~50-60% savings)

# Combine both
crumb compress auth-fix.crumb --metalk
```

## 8. Benchmark your crumbs

```bash
crumb bench auth-fix.crumb
```

Shows score, grade, token count, and compression ratio.

## 9. Search across crumbs

```bash
crumb search "auth JWT" --dir ./crumbs/
crumb search "payment" --method ranked    # TF-IDF ranking
```

## 10. Seed your AI tools

Generate CRUMB instructions for every AI tool you use:

```bash
crumb init --all              # seeds Claude, Cursor, Copilot, and more
crumb init --claude-md        # just CLAUDE.md
crumb init --cursor-rules     # just .cursor/rules
```

## Daily workflow

1. **Start of session**: `crumb wake` — paste into your AI for instant context
2. **During work**: `crumb palace add "..." --wing proj --room topic` — capture decisions
3. **Switching tools**: `crumb context --clipboard` — hand off project state
4. **Receiving handoffs**: `crumb receive --palace` — validate and file incoming crumbs
5. **End of session**: `crumb dream prefs.crumb` — consolidate raw observations

## Full reference

```bash
crumb --help        # 41 commands
crumb <cmd> --help  # per-command help
```
