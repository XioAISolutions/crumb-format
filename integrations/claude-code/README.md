# Claude Code integration

**Native CRUMB handoff for Claude Code in one command.**

This directory ships installable artifacts that drop into a Claude Code user's `~/.claude/` directory. Once installed:

- `/crumb-export <path>` — slash command to export the current session as a CRUMB.
- `/crumb-import <path>` — slash command to load a CRUMB into the current session.
- `crumb_*` MCP tools — Claude can drive exports/validates directly without a slash command.
- `crumb it` verbal trigger — say it in any session, get a paste-able handoff.

No Claude Code source modification needed. Everything works through Claude Code's existing extension points: slash commands, MCP servers, and CLAUDE.md.

## Install

```bash
# From a fresh checkout:
./integrations/claude-code/install.sh

# Or directly:
bash <(curl -fsSL https://raw.githubusercontent.com/XioAISolutions/crumb-format/main/integrations/claude-code/install.sh)
```

The installer:
1. Verifies `crumb-format` is installed (else prints `pip install crumb-format` and exits).
2. Detects `~/.claude/` (the standard Claude Code config location).
3. Copies slash command markdown files to `~/.claude/commands/`.
4. Optionally copies the MCP server config to `~/.claude/.mcp.json` (merged, not overwritten).
5. Optionally appends the `crumb it` verbal-trigger snippet to your project CLAUDE.md.
6. Prints a one-line confirmation.

Idempotent — running it twice does not duplicate anything.

## What gets installed

| Path | What |
|---|---|
| `~/.claude/commands/crumb-export.md` | `/crumb-export` slash command |
| `~/.claude/commands/crumb-import.md` | `/crumb-import` slash command |
| `~/.claude/.mcp.json` (merged) | MCP server entry pointing at `mcp/server.py` from this repo |
| Your project's `CLAUDE.md` (optional, prompted) | Appends the "crumb it" verbal-trigger block |

## Uninstall

```bash
./integrations/claude-code/uninstall.sh
```

Removes only what `install.sh` added. User-saved handoffs are preserved.

## Usage

After install, in any Claude Code session:

```
/crumb-export ~/handoff.crumb
```

Claude reads the conversation context and writes a `kind=task` (or appropriate kind) crumb to disk. Paste-friendly.

```
/crumb-import ~/handoff-from-cursor.crumb
```

Claude parses the bundle, summarizes its contents, and proceeds as if continuing that work.

For ambient handoff:

```
> crumb it
```

If you appended the verbal-trigger block to your project CLAUDE.md, this works in any session.

## Why this works without forking Claude Code

Claude Code already supports:
- **Slash commands** (markdown files in `~/.claude/commands/`)
- **MCP servers** (JSON-RPC tool servers in `~/.claude/.mcp.json` or `.claude/.mcp.json`)
- **CLAUDE.md** instructions (read on every session start)

This integration is a thin overlay on those existing extension points. No code patch to Anthropic's runtime. No coordination needed with Anthropic.

## Status

This is the first integration. Future targets (briefed in `docs/integrations/`):
- [Cursor](../../docs/integrations/cursor.md) — same shape via Cursor's MCP support
- [Aider](../../docs/integrations/aider.md) — Python plugin
- [OpenCode / sst-opencode](../../docs/integrations/opencode.md) — TUI command map

Each integration follows the same three-artifact pattern (slash command, MCP server, optional auto-export hook).
