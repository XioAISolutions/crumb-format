# Aider integration brief

**Status:** Plan only. Aider is a Python CLI coding agent that maintains its own conversation history per repo.

Aider's extension surface is different from Claude Code or Cursor — it's a Python CLI with custom commands defined via `--commands` files and a plugin hook in newer versions. CRUMB integration is mostly a wrapper plus two new aider-side commands.

## What ships

| Artifact | Where it lives | Notes |
|---|---|---|
| `/crumb-export` aider command | `integrations/aider/commands/crumb_export.py` | Reads aider's `coder.io` history; writes a CRUMB. |
| `/crumb-import` aider command | `integrations/aider/commands/crumb_import.py` | Parses a CRUMB; injects into aider's working context. |
| Settings overlay | `integrations/aider/.aider.conf.yml.template` | Adds the commands dir + a hint about the verbal trigger. |
| Installer | `integrations/aider/install.sh` | Drops the commands and merges the conf overlay. |

## Differences from Claude Code / Cursor

- **No MCP** — Aider doesn't speak MCP. The integration is in-process Python; we import `crumb_format` directly rather than running a JSON-RPC subprocess.
- **History is rich** — Aider tracks full edit history, not just chat turns. The `crumb_export` command can produce a `kind=log` crumb with one bullet per edit, similar to the HALO bridge but for aider's own log.
- **Repo-scoped** — Aider sessions are repo-scoped. The default export path is `<repo>/.aider/handoffs/<timestamp>.crumb`.

## Pitch

"Hand off your aider session to Claude Code or Cursor in one command. Repo state, edit history, and unfinished todos all travel as a CRUMB."

## Implementation effort

~250 LOC of Python (two commands + helper) + ~50 LOC install.sh + tests. Larger than the Claude Code integration because aider's command API is more complex than Claude Code's markdown slash commands, but no fundamentally new infrastructure.

## Open questions

- Aider's command-loading is changing as of recent versions (`aider --commands` vs the older `~/.aider/commands.py`). Pin a minimum aider version.
- Does aider have a clean hook for "session ended"? If yes, an auto-export at session end is straightforward (matches the SessionEnd hook plan in the Claude Code brief).
