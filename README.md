# CRUMB

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

## Two real-world scenarios

**Found the bug in Cursor, need Claude to write the test.** Generate a task crumb from Cursor with `crumb it`, paste it into Claude. Claude sees the goal, the surrounding code context, and the constraints -- and writes the test without asking you to re-explain anything.

**Re-explaining your preferences every session?** A mem crumb stores your working style once:

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

Paste it at the start of any session. No more "I like concise answers, don't use emojis, prefer TypeScript..." every time.

Five kinds: `task` (what to do next), `mem` (long-term memory), `map` (repo overview), `log` (session transcript), `todo` (work items).

## How it compares

| | Paste raw chat | Start over | Use CRUMB |
|---|---|---|---|
| Context preserved | Partial, noisy | None | Structured |
| Next AI acts immediately | Unlikely | No | Yes |
| Works across all AI tools | Yes | Yes | Yes |
| Token-efficient | No | Yes (lossy) | Yes |
| Human-readable | Barely | N/A | Yes |

## Add "crumb it" to your AI

Add this to your AI's custom instructions and it generates CRUMBs on command:

```text
When I say "crumb it", generate a CRUMB summarizing the current state.

For tasks: kind=task with [goal], [context], [constraints]
For memory: kind=mem with [consolidated]
For repos: kind=map with [project], [modules]

Format: BEGIN CRUMB / v=1.1 / headers / --- / sections / END CRUMB
```

Works in ChatGPT custom instructions, Claude Projects, Cursor rules, or any AI with system prompts.

## Install

```bash
pip install crumb-format
```

## Quick start

```bash
# create a task crumb
crumb new task --title "Fix auth" --goal "Fix token refresh race condition"

# validate
crumb validate examples/*.crumb

# append observations to memory, then consolidate
crumb append prefs.crumb "Switched to Neovim" "Dropped Redux"
crumb dream prefs.crumb

# search across crumbs (keyword, fuzzy, or TF-IDF ranked)
crumb search "auth JWT" --dir ./crumbs/

# seed all your AI tools at once
crumb init --all
```

Full command reference: `crumb --help` (25+ commands including export, import, templates, todos, watch mode, and more).

## AgentAuth — Agent Identity & Governance

Every AI agent in your org gets a passport. Every tool call gets policy-checked. Every action gets an audit trail. One kill switch revokes everything.

```bash
# Register an agent — issues a cryptographic passport
crumb passport register my-claude-agent --framework langchain --owner alice \
  --tools-allowed "read_*" "search" --tools-denied "delete_*" --ttl-days 90

# Check what an agent is allowed to do
crumb policy set my-claude-agent --allow "read_*" "search" --deny "delete_*"
crumb policy test my-claude-agent read_file    # → ALLOW
crumb policy test my-claude-agent delete_user  # → DENY

# Kill switch — instantly revoke all access
crumb passport revoke ap_abc12345

# Audit trail — every action logged with risk scoring
crumb audit export --format json --agent ap_abc12345
crumb audit feed   # live action stream

# Shadow AI scanner — discover unauthorized agents in your project
crumb scan --path . --min-risk medium
```

**Python SDK** for embedding in your own tools:

```python
from agentauth import AgentPassport, ToolPolicy, CredentialBroker, protect

# Register
mgr = AgentPassport()
result = mgr.register("my-agent", framework="crewai", owner="ops-team")

# Policy gate
policy = ToolPolicy()
policy.set_policy("my-agent", tools_denied=["rm_rf", "drop_table"])

# Decorator — enforces policy before any function runs
@protect(agent_id=result["agent_id"], tool="database.query")
def query_database(sql, _agentauth_credential=None):
    return db.execute(sql)
```

## Integrations

**MCP Server** -- native tool integration with Claude Desktop, Cursor, Claude Code:
```bash
# CRUMB tools (create, validate, search, etc.)
claude mcp add crumb python3 /path/to/crumb-format/mcp/server.py

# AgentAuth tools (passport, policy, audit — 13 tools)
claude mcp add agentauth python3 /path/to/crumb-format/mcp/agentauth_server.py
```
See [`mcp/README.md`](mcp/README.md) for setup.

**Pre-commit hook** -- validate `.crumb` files on every commit:
```yaml
repos:
  - repo: https://github.com/XioAISolutions/crumb-format
    rev: main
    hooks:
      - id: validate-crumbs
```

**ClawHub skill** -- install as an OpenClaw agent skill. See [`clawhub-skill/`](clawhub-skill/).

## What's in this repo

- [`SPEC.md`](SPEC.md) -- the format specification
- [`DREAMING.md`](DREAMING.md) -- how memory consolidation works
- [`examples/`](examples/) -- ready-to-paste `.crumb` files
- [`cli/crumb.py`](cli/crumb.py) -- full CLI (25+ commands)
- [`agentauth/`](agentauth/) -- AgentAuth SDK (passport, policy, credentials, audit)
- [`mcp/`](mcp/) -- MCP servers for CRUMB and AgentAuth
- [`validators/`](validators/) -- Python and Node reference validators
- [`docs/HANDOFF_PATTERNS.md`](docs/HANDOFF_PATTERNS.md) -- practical handoff patterns

## License

MIT. See [`TRADEMARK.md`](TRADEMARK.md) for brand guidance.

CRUMB is plain text. It works everywhere text works.
