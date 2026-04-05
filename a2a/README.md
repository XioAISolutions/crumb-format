# CRUMB AgentAuth — A2A Protocol Bridge

This directory implements a bridge between CRUMB AgentAuth and Google's
[Agent-to-Agent (A2A) protocol](https://github.com/google/A2A).  The bridge
makes every AgentAuth capability — passport management, policy enforcement,
audit logging, CRUMB parsing, and shadow-AI scanning — discoverable and
callable by any A2A-compatible agent.

## What is A2A?

A2A is a JSON-based protocol that lets AI agents discover and communicate with
each other over HTTP.  The two core concepts are:

| Concept | Description |
|---------|-------------|
| **Agent Card** | A JSON document served at `/.well-known/agent.json` that describes the agent's name, skills, authentication, and supported I/O modes. Other agents fetch this card to learn what the agent can do. |
| **Task** | A JSON request/response pair sent to `/tasks/send`.  A task targets a specific *skill* by ID, carries an `input` payload, and receives a structured `result` (or `error`) in return. |

## Quick start

```bash
# From the repository root:
python -m a2a.server              # starts on port 8421
python -m a2a.server --port 9000  # custom port
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/.well-known/agent.json` | Returns the A2A agent card |
| `POST` | `/tasks/send` | Accepts an A2A task and routes it to the appropriate AgentAuth skill |
| `GET`  | `/health` | Returns `{"status": "ok"}` |

## Agent card

Fetch the card to discover available skills:

```bash
curl http://localhost:8421/.well-known/agent.json
```

## Sending a task

Every task is a JSON object with three fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Caller-assigned task identifier |
| `skill_id` | string | One of the skill IDs from the agent card |
| `input` | object | Skill-specific input payload |

### Examples

**Register an agent passport**

```bash
curl -X POST http://localhost:8421/tasks/send \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "task-1",
    "skill_id": "passport.register",
    "input": {"name": "my-bot", "framework": "langchain"}
  }'
```

**Verify a passport**

```bash
curl -X POST http://localhost:8421/tasks/send \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "task-2",
    "skill_id": "passport.verify",
    "input": {"agent_id": "ap_abc12345"}
  }'
```

**Parse a CRUMB block**

```bash
curl -X POST http://localhost:8421/tasks/send \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "task-3",
    "skill_id": "crumb.parse",
    "input": {"text": "BEGIN CRUMB\nv=1.0\nkind=note\n---\n## body\n  hello world\nEND CRUMB"}
  }'
```

**Check a tool policy**

```bash
curl -X POST http://localhost:8421/tasks/send \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "task-4",
    "skill_id": "policy.check",
    "input": {"agent_id": "ap_abc12345", "tool": "web_search"}
  }'
```

**Shadow AI scan**

```bash
curl -X POST http://localhost:8421/tasks/send \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "task-5",
    "skill_id": "scan.shadow",
    "input": {"path": "/path/to/project"}
  }'
```

## Available skills

| Skill ID | Description | Required input fields |
|----------|-------------|-----------------------|
| `crumb.parse` | Parse CRUMB text into structured data | `text` |
| `crumb.render` | Render structured data into CRUMB text | `headers`, `sections` |
| `crumb.validate` | Validate a CRUMB text block | `text` |
| `passport.register` | Register an agent and issue a passport | `name` (optional: `framework`, `owner`, `tools_allowed`, `tools_denied`, `data_classes`, `ttl_days`) |
| `passport.verify` | Verify an agent passport | `agent_id` |
| `passport.revoke` | Revoke an agent passport | `agent_id` |
| `policy.check` | Check whether an agent may use a tool | `agent_id`, `tool` (optional: `action`, `data_class`) |
| `audit.log` | Log an agent action to the audit trail | `agent_id`, `tool` (optional: `detail`, `goal`, `allowed`, `reason`) |
| `scan.shadow` | Scan a directory for shadow AI usage | `path` (optional: `min_risk`) |

## Architecture

```
a2a/
  agent_card.py    # Builds the JSON agent card
  task_handler.py  # Routes tasks to AgentAuth subsystems
  server.py        # stdlib HTTP server (no external dependencies)
  README.md        # This file
```

The bridge imports directly from `cli/crumb.py` (for `parse_crumb` /
`render_crumb`) and from the `agentauth/` package (for `AgentPassport`,
`ToolPolicy`, and `AuditLogger`).  No external dependencies are required —
everything uses the Python standard library.
