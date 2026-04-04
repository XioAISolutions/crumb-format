# CRUMB MCP Server

Exposes CRUMB tools to any MCP-compatible client (Claude Desktop, Cursor, Claude Code, etc.).

## Setup

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "crumb": {
      "command": "python3",
      "args": ["/path/to/crumb-format/mcp/server.py"]
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "crumb": {
      "command": "python3",
      "args": ["/path/to/crumb-format/mcp/server.py"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add crumb python3 /path/to/crumb-format/mcp/server.py
```

## Available Tools

| Tool | Description |
|------|-------------|
| `crumb_new` | Create a new crumb (task, mem, map, log, todo) |
| `crumb_validate` | Validate .crumb files |
| `crumb_inspect` | Display a crumb's structure |
| `crumb_append` | Append observations to a mem crumb |
| `crumb_dream` | Run consolidation pass |
| `crumb_search` | Search crumbs (keyword, fuzzy, ranked) |
| `crumb_export` | Export to JSON, markdown, or clipboard format |
| `crumb_template` | List or use crumb templates |
| `crumb_todo_add` | Add tasks to a todo crumb |
| `crumb_todo_done` | Mark tasks done |
| `crumb_log` | Append timestamped entries to a log crumb |

## Zero dependencies

The MCP server uses only Python stdlib. No pip install needed beyond the crumb-format package itself.

---

# AgentAuth MCP Server

Exposes the AgentAuth SDK (passports, policies, credentials, audit) as MCP tools. Lets any MCP-compatible client register agents, enforce tool policies, issue short-lived credentials, and maintain audit trails.

## Setup

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "crumb": {
      "command": "python3",
      "args": ["/path/to/crumb-format/mcp/server.py"]
    },
    "agentauth": {
      "command": "python3",
      "args": ["/path/to/crumb-format/mcp/agentauth_server.py"]
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "crumb": {
      "command": "python3",
      "args": ["/path/to/crumb-format/mcp/server.py"]
    },
    "agentauth": {
      "command": "python3",
      "args": ["/path/to/crumb-format/mcp/agentauth_server.py"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add crumb python3 /path/to/crumb-format/mcp/server.py
claude mcp add agentauth python3 /path/to/crumb-format/mcp/agentauth_server.py
```

## Available AgentAuth Tools

### Passport Management

| Tool | Description |
|------|-------------|
| `passport_register` | Register a new agent passport (name, framework, owner, permissions, TTL) |
| `passport_inspect` | Inspect a passport by agent_id or name |
| `passport_verify` | Verify an agent is valid, not revoked, and not expired |
| `passport_revoke` | Kill switch -- immediately revoke an agent passport |
| `passport_list` | List all registered agents, optionally filtered by status |

### Policy

| Tool | Description |
|------|-------------|
| `policy_set` | Set or update tool policy for an agent (allowed/denied lists, data classes, action limits) |
| `policy_check` | Check if an agent is allowed to use a specific tool |

### Credentials

| Tool | Description |
|------|-------------|
| `credential_issue` | Issue a short-lived HMAC token scoped to agent + tool |
| `credential_validate` | Validate a credential token |

### Audit

| Tool | Description |
|------|-------------|
| `audit_start` | Start an audit session (returns session_id) |
| `audit_log` | Log an action within a session (tool, detail, allowed/denied) |
| `audit_end` | End a session and persist the audit trail as a .crumb file |
| `audit_export` | Export audit evidence (crumb, JSON, or CSV format) |

## Data Storage

All AgentAuth data is stored under `.crumb-auth/` in the working directory:

```
.crumb-auth/
  passports/ap_XXXXXXXX.crumb   # Agent passport files
  policies/agent-name.json       # Tool policy configs
  audit/YYYY-MM-DD/as_XXXXX.crumb  # Audit session trails
  tokens.json                    # Active credential tokens
  revoked.json                   # Revoked agent IDs
```
