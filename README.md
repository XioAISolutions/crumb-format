# CRUMB

The copy-paste AI handoff format.

![CRUMB CLI overview](docs/assets/crumb-banner.svg)

---

Ever been deep into a task with one AI, then need to switch to another? You either paste an enormous chat log and hope it picks up the thread, or you start over and re-explain everything from scratch.

CRUMB is a third option. It's a small, structured text block you copy-paste between AI tools. The next AI gets exactly what it needs to continue your work -- the goal, the context, and the constraints -- without the noise.

> **v0.3.0** — Protocol-grade release. Deterministic `crumb pack`, MemPalace bridge adapter, `crumb lint` safety scanner, extension model, golden fixtures. All v0.2 features (Palace, Reflect, MeTalk, REST/A2A, AgentAuth, shadow-AI scanner, 41+ CLI commands) included. `pip install crumb-format`.

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

Full command reference: `crumb --help` (41 commands including export, import, templates, todos, watch mode, compression, agent governance, and more). See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) for a 5-minute walkthrough.

## Palace — Spatial Memory That Stays With You

AI conversations disappear when the session ends. Palace gives you a persistent, hierarchical memory that any AI can read — organized by **wings** (people/projects), **halls** (facts/events/discoveries/preferences/advice), and **rooms** (specific topics). No database, no cloud — just a directory of `.crumb` files that are grep-able, git-able, and diff-able.

```bash
# Initialize a palace
crumb palace init

# File observations (hall auto-classified if omitted)
crumb palace add "decided to use Postgres" --wing orion --room db-choice
crumb palace add "shipped v0.1 yesterday"  --wing orion --room launch
crumb palace add "prefers concise commits" --wing nova  --room style

# List, search, and cross-reference
crumb palace list --wing orion
crumb palace search "postgres"
crumb palace tunnel                  # find cross-wing links
crumb palace stats

# Wake-up: one-shot context for a new session (~170 tokens)
crumb wake                           # identity + top facts + room index
crumb wake --metalk                  # compressed for token density
```

Auto-classification puts each observation in the right hall without you specifying it — "decided X" → `facts`, "shipped X" → `events`, "prefers X" → `preferences`. Use `crumb classify --text "..."` to test it standalone.

```
.crumb-palace/wings/
  orion/
    facts/db-choice.crumb           # kind=mem
    events/launch.crumb
  nova/
    preferences/style.crumb
    facts/db-choice.crumb           # ← same room name → tunnel detected
```

## Reflect — Self-Learning Gap Detection

A filing cabinet stores what you put in it. A second brain tells you what's *missing*. `crumb reflect` analyzes your palace and identifies knowledge gaps, stale rooms, and imbalances — then suggests exactly what to add next.

```bash
# Health check — scored 0-100 with actionable suggestions
crumb reflect

# Output as a crumb for AI consumption
crumb reflect -f crumb

# Include gap awareness in session wake-ups
crumb wake --reflect

# Generate a structured wiki from palace contents
crumb palace wiki
```

Example output:
```
Palace Health: 76/100 (Grade: C)
Wings: 2  Rooms: 6
Hall coverage: advice=1, discoveries=1, events=1, facts=2, preferences=1

Found 3 gap(s):

   !! [MEDIUM] Wing team has only 1 room(s). Sparse knowledge.
      -> Add more observations: crumb palace add "..." --wing team --room <topic>
    ! [LOW] Wing team is missing hall 'preferences' (present in other wings).
      -> Add to fill the gap: crumb palace add "..." --wing team --hall preferences --room <topic>
    ! [LOW] Wing team has no discoveries — nothing learned or realized.
      -> Capture learnings: crumb palace add "realized ..." --wing team --room <insight>
```

Gap types: empty halls, thin wings, stale rooms (configurable threshold), missing cross-wing halls, undocumented preferences, no discoveries.

## MeTalk — Caveman Compression for AI-to-AI

AI-to-AI messages don't need polished English. MeTalk strips articles, abbreviates tech terms, and shortens verbose phrasing so you can pack more context into the same token budget.

```bash
# Default level 2 (dict + grammar strip, ~40% savings)
crumb metalk task.crumb

# Lossless dictionary substitution only (round-trippable)
crumb metalk task.crumb --level 1

# Aggressive condensing (~50-60% savings)
crumb metalk task.crumb --level 3

# Chain with compress for maximum density
crumb compress task.crumb --metalk
```

Output shows live stats: `MeTalk: 127 → 68 tokens (46.5% saved, 1.87x ratio)`.

## Cross-AI Interop

CRUMB speaks every major AI protocol so your context travels freely between tools.

```bash
# REST API (OpenAPI 3.1) — run as a service
python -m api.server                       # see api/README.md

# Agent-to-Agent (A2A) bridge — Google's A2A spec
python -m a2a.server                       # see a2a/README.md

# Convert CRUMB <-> other formats
crumb bridge list                          # supported formats
crumb bridge export task.crumb --to openai-threads
crumb bridge import chat.json --from langchain-memory

# Event webhooks for agent activity
crumb webhook add https://hooks.example.com/agent-events
crumb webhook test https://hooks.example.com/agent-events
```

Formats supported via `bridge`: `openai-threads`, `langchain-memory`, `crewai-task`, `autogen`, `claude-project`.

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
- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) -- 5-minute daily workflow guide
- [`examples/`](examples/) -- ready-to-paste `.crumb` files (task, mem, map, log, todo, wake)
- [`cli/crumb.py`](cli/crumb.py) -- full CLI (41 commands)
- [`cli/reflect.py`](cli/reflect.py) -- self-learning gap detection and knowledge health scoring
- [`cli/palace.py`](cli/palace.py) -- Palace spatial memory (wings/halls/rooms/tunnels)
- [`cli/classify.py`](cli/classify.py) -- rule-based hall classifier
- [`cli/metalk.py`](cli/metalk.py) -- MeTalk caveman compression module
- [`agentauth/`](agentauth/) -- AgentAuth SDK (passport, policy, credentials, audit, webhooks)
- [`mcp/`](mcp/) -- MCP servers for CRUMB and AgentAuth
- [`api/`](api/) -- REST API server with OpenAPI 3.1 spec
- [`a2a/`](a2a/) -- Google A2A protocol bridge (agent card, task handler, server)
- [`validators/`](validators/) -- Python and Node reference validators
- [`tests/`](tests/) -- 291 tests covering the full surface area
- [`docs/HANDOFF_PATTERNS.md`](docs/HANDOFF_PATTERNS.md) -- practical handoff patterns

## License

MIT. See [`TRADEMARK.md`](TRADEMARK.md) for brand guidance.

CRUMB is plain text. It works everywhere text works.
