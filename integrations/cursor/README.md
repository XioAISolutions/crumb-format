# Cursor integration

**Native CRUMB handoff for Cursor in one command.**

This directory ships installable artifacts that drop into a Cursor user's `~/.cursor/` (global) and the current project's `./.cursor/` (per-project). Once installed:

- The `crumb-export` rule activates when you say "crumb it" or "export as CRUMB" in Composer/chat — Cursor produces a `.crumb` handoff file you can paste anywhere.
- The `crumb-import` rule activates whenever a `.crumb` file is opened or pasted — Cursor parses it and continues the work it describes.
- All 24 `crumb_*` MCP tools become available in Composer (validate, lint, search, palace memory, governance).

No fork of Cursor. Everything works through Cursor's existing extension points: rule files (`.cursor/rules/*.mdc`) and MCP servers (`~/.cursor/mcp.json`).

## Install

```bash
# From a fresh checkout:
./integrations/cursor/install.sh

# Or directly:
bash <(curl -fsSL https://raw.githubusercontent.com/XioAISolutions/crumb-format/main/integrations/cursor/install.sh)

# Preview what would change without writing anything:
./integrations/cursor/install.sh --dry-run
```

The installer:
1. Verifies `crumb-format` is installed (else prints `pip install crumb-format` and exits).
2. Registers the CRUMB MCP server in `~/.cursor/mcp.json` (merged, not overwritten).
3. If the current directory has a `.cursor/` folder (or you opt in interactively), copies three rule files into `./.cursor/rules/`.
4. Prints a one-line confirmation.

Idempotent — running it twice does not duplicate anything.

## What gets installed

| Path | What |
|---|---|
| `~/.cursor/mcp.json` (merged) | Global MCP server entry pointing at `mcp/server.py` |
| `./.cursor/rules/crumb-export.mdc` | Rule that fires on "crumb it" / "export as CRUMB" |
| `./.cursor/rules/crumb-import.mdc` | Rule that fires when a `.crumb` file is opened or pasted |
| `./.cursor/rules/crumb-it.mdc` | Always-apply rule: makes Cursor recognize the format ambiently |

## Uninstall

```bash
./integrations/cursor/uninstall.sh
```

Removes only what `install.sh` added (entries marked with `_managed_by`). User-saved rules are preserved.

## Usage

After install, in any Cursor chat or Composer session:

```
crumb it
```

Cursor reads the conversation context and writes a `kind=task` (or appropriate kind) crumb to `./.crumb/handoff.crumb`. Paste-friendly.

When you (or anyone) opens a `.crumb` file in Cursor, the `crumb-import` rule fires automatically. Cursor parses the bundle, summarizes its contents, and proceeds as if continuing that work.

## Differences from Claude Code

- **Rule files vs slash commands.** Cursor uses `.cursor/rules/*.mdc` for both invocation and ambient context, where Claude Code uses `~/.claude/commands/*.md` slash commands. Same content, different surface.
- **Global MCP path.** Cursor's MCP config lives at `~/.cursor/mcp.json`; Claude Code uses `~/.claude/.mcp.json`.
- **Per-project rules.** Cursor rules are project-level by design. The installer prompts before copying rules into the current directory; the global MCP entry works in every project regardless.
- **Background agents** — Cursor's background agents have a separate config path; out of scope for v1.

## Why this works without forking Cursor

Cursor already supports:
- **Rule files** in `.cursor/rules/*.mdc` with frontmatter metadata
- **MCP servers** in `~/.cursor/mcp.json`

This integration is a thin overlay on those existing extension points. No code patch to Cursor. No coordination needed with Anysphere.

## Status

Shipping. Same three-artifact pattern as the Claude Code integration in `../claude-code/`. Future targets briefed in `docs/integrations/`:
- [Aider](../../docs/integrations/aider.md) — Python plugin
- [OpenCode / sst-opencode](../../docs/integrations/opencode.md) — TUI command map
