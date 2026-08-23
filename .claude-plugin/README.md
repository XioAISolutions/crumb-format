# The CRUMB Claude Code plugin

```
/plugin marketplace add XioAISolutions/crumb-format
/plugin install crumb-format
```

Installs from GitHub, not PyPI. That is the point: the CLI is stdlib-only and
runs straight from this checkout, so there is no build step and no package
index between you and a working install.

| File | What it declares |
| --- | --- |
| `plugin.json` | the manifest — name, version, component paths |
| `hooks.json` | `SessionEnd` → `crumb capture`, `SessionStart` → `crumb resume` |
| `mcp.json` | the CRUMB MCP server |
| `marketplace.json` | makes this repository its own marketplace |

## What the hooks do

`capture` writes a handoff to `.crumb/` when a session ends. `resume` injects
the most recent one when the next session starts — on `startup` only, never on
`clear` (that is the user asking for a clean slate) and never on `compact`
(which fires mid-session, when the newest crumb belongs to a *different*
session).

If you also ran `integrations/claude-code/install.sh`, the hooks are registered
twice — once in `settings.json`, once here. That is safe: `resume` records the
session it injected for and declines to inject into the same session again.

## Known limitation: Windows

Both hook commands and the MCP server invoke `python3`. On macOS and Linux that
is correct. On **native Windows** (outside WSL or a POSIX shell) the executable
is usually `python` or `py -3`, and `python3` may not resolve — in which case
the hooks fail silently and no handoff is written or injected.

This is stated rather than guessed at: no Windows machine was available to test
a fallback, and shipping an untested launcher would be worse than naming the
gap. If you hit this, either run Claude Code under WSL, or edit
`hooks.json` and `mcp.json` in your installed copy to use the interpreter your
system provides. A verified cross-platform launcher is the right fix and is not
yet written.

## Not a spec change

The plugin adds two optional headers to the crumbs it writes — `captured=` and
`measured=`. Both are ignorable: `conformance/cases/accept-unknown-header-ignored.crumb`
pins that parsers must skip headers they do not recognise, so the wire format
stays frozen at v1.4.
