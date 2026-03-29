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
