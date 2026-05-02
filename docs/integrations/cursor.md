# Cursor integration brief

**Status:** Plan only. No code yet. Same shape as the Claude Code integration in `integrations/claude-code/`.

Cursor supports MCP servers, custom rules, and rule files in `.cursor/`. CRUMB integration follows the same three-artifact pattern that Claude Code uses.

## What ships

| Artifact | Cursor surface | Lives at |
|---|---|---|
| `crumb-export` rule | `.cursor/rules/` | `integrations/cursor/rules/crumb-export.md` |
| `crumb-import` rule | `.cursor/rules/` | `integrations/cursor/rules/crumb-import.md` |
| MCP server entry | `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global) | `integrations/cursor/mcp.json.template` |
| `crumb it` verbal trigger snippet | `.cursorrules` or `.cursor/rules/crumb-it.md` | `integrations/cursor/cursorrules-snippet.md` |
| Installer | `install.sh` | `integrations/cursor/install.sh` |

## Differences from Claude Code

- **No slash commands.** Cursor uses rule files instead. The export/import "commands" are rule entries that Cursor Composer/Chat reads.
- **MCP config path differs.** Cursor uses `.cursor/mcp.json` (Claude Code uses `.claude/.mcp.json`). The MCP server payload is identical otherwise.
- **No equivalent of CLAUDE.md per project** — Cursor has `.cursorrules` for that role. Same content, different filename.
- **Background agent runs aren't on this surface.** Cursor's background agents have a separate config path; out of scope for v1.

## Implementation order

1. **Spike**: clone Cursor's published MCP example. Confirm CRUMB's `mcp/server.py` registers cleanly.
2. **Templates**: copy Claude Code's `crumb-export.md` and `crumb-import.md` rule files; rename if Cursor's frontmatter syntax differs.
3. **Installer**: same install.sh shape, different target paths.
4. **README**: same `integrations/cursor/README.md` story as Claude Code.

Estimated total: ~150 LOC + tests + docs. Roughly 2 hours of work once the Cursor MCP spike confirms the format.

## Pitch

"Native CRUMB handoff for Cursor — install with one command, then `crumb it` in any chat to hand work off to Claude Code, ChatGPT, or another AI tool."

## Open questions

- Does Cursor's rule file format match Claude Code's slash-command frontmatter? If so, the rule files are byte-identical. If not, they're 80% the same with a different YAML header.
- Does Cursor surface MCP tool errors visibly in the UI? If not, the user-facing error story for malformed crumbs needs more thought.
