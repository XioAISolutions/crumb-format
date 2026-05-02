# OpenCode integration brief

**Status:** Plan only. OpenCode (sst-opencode) is an open-source TUI coding agent with a plugin system.

## What ships

| Artifact | Where it lives | Notes |
|---|---|---|
| OpenCode plugin: `crumb-export` command | `integrations/opencode/plugin/crumb_export.ts` (or `.js`) | TUI command that produces a CRUMB. |
| OpenCode plugin: `crumb-import` command | `integrations/opencode/plugin/crumb_import.ts` | Loads a CRUMB into context. |
| MCP server entry | `integrations/opencode/mcp.json.template` | If OpenCode's MCP support has stabilized; if not, drop this artifact. |
| Installer | `integrations/opencode/install.sh` | Drops the plugin; merges MCP if applicable. |

## Differences from Claude Code / Cursor

- **TypeScript / JavaScript runtime, not Markdown.** OpenCode's plugin system is closer to a real plugin API — write actual code, not slash-command markdown. This is the most substantial of the three "follow-on" integrations.
- **TUI surface only.** OpenCode is a terminal app. Slash commands map to keystrokes/`/commands`. No browser/Electron extension surface.
- **Native MCP support is recent.** Verify the MCP server protocol version before committing to the MCP path; fall back to direct CLI shellout if needed.

## Pitch

"OpenCode + CRUMB: hand off your TUI session to any AI tool with one keystroke. The first plugin for OpenCode that actually moves work between agents."

## Implementation effort

~400 LOC of TypeScript (two plugin commands + bundle config) + ~80 LOC install.sh + tests. Largest of the three follow-ons because it requires a real plugin build (TS → JS bundle) rather than just shipping markdown templates.

## Sequencing

OpenCode is the most-effort, lowest-prior of the three follow-ons. Recommend doing it after Cursor lands. If OpenCode's plugin API is unstable, defer until they hit 1.0.

## Open questions

- What's OpenCode's plugin distribution story? npm package? Single-file drop-in? Affects the install.sh shape.
- Does OpenCode have a "session" concept analogous to Claude Code's session log? If history is per-buffer or per-window, the export-shape differs.
