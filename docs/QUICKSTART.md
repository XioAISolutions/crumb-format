# CRUMB Quickstart

Get productive with CRUMB in five minutes.

## Install

```bash
pip install crumb-format
```

## 0. (Optional) One-time setup

Most commands work right after install. The Palace, Wake, and Resolve commands need a one-time directory. Run this only if you want persistent memory:

```bash
crumb palace init        # creates .crumb-palace/ in the current directory
```

`crumb resolve` and `crumb seen` honor `CRUMB_HOME` (default `~/.crumb/`) and `CRUMB_STORE` (default `~/.crumb/store/`); set them only if you want a non-default location.

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
crumb todo add sprint.crumb "Add rate limiting" "Write payment tests"
crumb todo list sprint.crumb
crumb todo done sprint.crumb "rate limiting"
crumb todo dream sprint.crumb   # archive completed tasks
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

## 11. v1.3 — agent personas and dependency-aware handoffs

**Create a reusable agent persona** (v1.3 `kind=agent`):

```bash
crumb new agent \
  --agent-id reviewer-v2 \
  --title "Senior code reviewer" \
  --source human.notes \
  --rules "never approve without tests" "flag O(n^2) in hot paths" \
  --knowledge "expert=python, typescript"
```

Reference it from a task handoff by putting `refs=reviewer-v2` in the task's header. A v1.3 runtime loads the persona first, then processes the task.

**Handoffs with dependencies** — non-linear order without inventing a new section:

```text
[handoff]
- id=repro   to=any    do=reproduce on main
- id=fix     to=any    do=propose a fix              after=repro
- id=test    to=any    do=add regression test        after=fix
- id=review  to=human  do=approve before merge       after=test
```

Parsers detect cycles (`crumb validate` rejects them) and reject unknown deps.

**Resolve a ref** (bare id, sha256, or URL) per SPEC §17:

```bash
crumb resolve reviewer-v2                  # bare id → local directory
crumb resolve sha256:abc123...              # digest → content store
crumb resolve some-id --walk --depth 5      # transitive walk
crumb resolve unknown-id --strict           # exit 1 when unresolved
```

**Lint with reference checks**:

```bash
crumb lint handoff.crumb --check-refs             # warns on unresolved refs
crumb lint handoff.crumb --check-refs --strict    # exit 1 on any warning
```

## Full reference

```bash
crumb --help        # ~46 commands grouped by concern (v0.7.0: grouping; v0.8.0: adds `guardrails`)
crumb <cmd> --help  # per-command help
```
