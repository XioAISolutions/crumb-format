#!/usr/bin/env python3
"""MCP server that exposes CRUMB tools to Claude Desktop, Cursor, and other MCP clients.

Run with: python3 mcp/server.py
Configure in claude_desktop_config.json or .cursor/mcp.json
"""

import json
import sys
from pathlib import Path

# Add parent dir so we can import cli/crumb.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))
import crumb


def read_stdin():
    """Read a JSON-RPC message from stdin."""
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)


def write_stdout(msg):
    """Write a JSON-RPC message to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def respond(req_id, result):
    write_stdout({"jsonrpc": "2.0", "id": req_id, "result": result})


def respond_error(req_id, code, message):
    write_stdout({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


TOOLS = [
    {
        "name": "crumb_new",
        "description": "Create a new .crumb file. Kinds: task, mem, map, log, todo.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["task", "mem", "map", "log", "todo"]},
                "title": {"type": "string", "description": "Title for the crumb"},
                "goal": {"type": "string", "description": "Goal text (task only)"},
                "context": {"type": "array", "items": {"type": "string"}, "description": "Context items (task only)"},
                "constraints": {"type": "array", "items": {"type": "string"}, "description": "Constraints (task only)"},
                "entries": {"type": "array", "items": {"type": "string"}, "description": "Entries (mem/log/todo)"},
                "project": {"type": "string", "description": "Project name (map only)"},
                "description": {"type": "string", "description": "Project description (map only)"},
                "modules": {"type": "array", "items": {"type": "string"}, "description": "Module entries (map only)"},
            },
            "required": ["kind"],
        },
    },
    {
        "name": "crumb_validate",
        "description": "Validate one or more .crumb files against the spec.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "files": {"type": "array", "items": {"type": "string"}, "description": "Paths to .crumb files"},
            },
            "required": ["files"],
        },
    },
    {
        "name": "crumb_inspect",
        "description": "Parse and display a .crumb file's structure (headers, sections, line counts).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to .crumb file"},
                "headers_only": {"type": "boolean", "description": "Show only headers and section names"},
            },
            "required": ["file"],
        },
    },
    {
        "name": "crumb_append",
        "description": "Append raw observations to a mem crumb's [raw] section.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to a kind=mem .crumb file"},
                "entries": {"type": "array", "items": {"type": "string"}, "description": "Observations to append"},
            },
            "required": ["file", "entries"],
        },
    },
    {
        "name": "crumb_dream",
        "description": "Run a consolidation pass on a mem crumb: deduplicate, merge raw into consolidated, prune to budget.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to a kind=mem .crumb file"},
                "dry_run": {"type": "boolean", "description": "Preview without writing"},
            },
            "required": ["file"],
        },
    },
    {
        "name": "crumb_search",
        "description": "Search .crumb files by keyword, fuzzy match, or ranked relevance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "dir": {"type": "string", "description": "Directory to search (default: .)"},
                "method": {"type": "string", "enum": ["keyword", "fuzzy", "ranked"], "description": "Search method"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "crumb_export",
        "description": "Export a .crumb file to JSON, markdown, or clipboard-friendly format.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to .crumb file"},
                "format": {"type": "string", "enum": ["json", "markdown", "clipboard"]},
            },
            "required": ["file", "format"],
        },
    },
    {
        "name": "crumb_template",
        "description": "List available templates or generate a crumb from a template.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "use"]},
                "name": {"type": "string", "description": "Template name (for 'use')"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "crumb_todo_add",
        "description": "Add tasks to a todo crumb (creates file if missing).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to todo crumb"},
                "tasks": {"type": "array", "items": {"type": "string"}, "description": "Tasks to add"},
            },
            "required": ["file", "tasks"],
        },
    },
    {
        "name": "crumb_todo_done",
        "description": "Mark tasks as done in a todo crumb by substring match.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to todo crumb"},
                "query": {"type": "string", "description": "Substring to match against open tasks"},
            },
            "required": ["file", "query"],
        },
    },
    {
        "name": "crumb_log",
        "description": "Append timestamped entries to a log crumb (creates file if missing).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to log crumb"},
                "entries": {"type": "array", "items": {"type": "string"}, "description": "Entries to log"},
            },
            "required": ["file", "entries"],
        },
    },
    {
        "name": "crumb_palace_add",
        "description": "File an observation into Palace spatial memory (auto-classifies hall if omitted).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Observation text"},
                "wing": {"type": "string", "description": "Wing name (person/project/topic)"},
                "room": {"type": "string", "description": "Room name (specific topic)"},
                "hall": {"type": "string", "enum": ["facts", "events", "discoveries", "preferences", "advice"],
                         "description": "Hall (auto-classified if omitted)"},
            },
            "required": ["text", "wing", "room"],
        },
    },
    {
        "name": "crumb_palace_list",
        "description": "List rooms in the Palace, optionally filtered by wing and/or hall.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "Filter by wing name"},
                "hall": {"type": "string", "enum": ["facts", "events", "discoveries", "preferences", "advice"]},
            },
        },
    },
    {
        "name": "crumb_palace_search",
        "description": "Search across Palace rooms by keyword.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "wing": {"type": "string", "description": "Restrict to one wing"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "crumb_palace_wiki",
        "description": "Generate a structured knowledge index from Palace contents.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "crumb_reflect",
        "description": "Analyze Palace health and identify knowledge gaps with actionable suggestions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["text", "crumb"], "description": "Output format (default: text)"},
            },
        },
    },
    {
        "name": "crumb_wake",
        "description": "Emit a session wake-up crumb from Palace — identity, top facts, and room index.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_facts": {"type": "integer", "description": "Max facts to include (default: 8)"},
                "reflect": {"type": "boolean", "description": "Include top knowledge gaps"},
                "metalk": {"type": "boolean", "description": "Apply MeTalk compression"},
            },
        },
    },
    {
        "name": "crumb_context",
        "description": "Generate a task crumb from current project state (git, palace facts, todos).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Override the auto-detected goal"},
                "title": {"type": "string", "description": "Override the auto-generated title"},
                "commits": {"type": "integer", "description": "Number of recent commits (default: 5)"},
                "metalk": {"type": "boolean", "description": "Apply MeTalk compression"},
            },
        },
    },
    {
        "name": "crumb_metalk",
        "description": "Apply MeTalk caveman compression to a crumb file (reduce tokens for AI-to-AI).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to .crumb file"},
                "level": {"type": "integer", "enum": [1, 2, 3],
                          "description": "1=dict only, 2=dict+grammar, 3=aggressive (default: 2)"},
                "decode": {"type": "boolean", "description": "Decode MeTalk back to full form"},
            },
            "required": ["file"],
        },
    },
    {
        "name": "crumb_classify",
        "description": "Classify text into a Palace hall (facts/events/discoveries/preferences/advice).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to classify"},
            },
            "required": ["text"],
        },
    },
]


def handle_tool_call(name, args):
    """Execute a CRUMB tool and return the result text."""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        if name == "crumb_new":
            kind = args["kind"]
            cli_args = ["new", kind]
            if args.get("title"):
                cli_args.extend(["--title", args["title"]])
            cli_args.extend(["--source", "mcp"])
            if kind == "task":
                if args.get("goal"):
                    cli_args.extend(["--goal", args["goal"]])
                for c in args.get("context", []):
                    cli_args.extend(["--context", c])
                for c in args.get("constraints", []):
                    cli_args.extend(["-c", c])
            elif kind == "mem":
                for e in args.get("entries", []):
                    cli_args.extend(["-e", e])
            elif kind == "map":
                if args.get("project"):
                    cli_args.extend(["-p", args["project"]])
                if args.get("description"):
                    cli_args.extend(["-d", args["description"]])
                for m in args.get("modules", []):
                    cli_args.extend(["-m", m])
            elif kind in ("log", "todo"):
                for e in args.get("entries", []):
                    cli_args.extend(["-e", e])

            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_validate":
            cli_args = ["validate"] + args["files"]
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                try:
                    crumb.main(cli_args)
                except SystemExit:
                    pass
            return stdout_buf.getvalue() + stderr_buf.getvalue()

        elif name == "crumb_inspect":
            cli_args = ["inspect", args["file"]]
            if args.get("headers_only"):
                cli_args.append("--headers-only")
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_append":
            cli_args = ["append", args["file"]] + args["entries"]
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_dream":
            cli_args = ["dream", args["file"]]
            if args.get("dry_run"):
                cli_args.append("--dry-run")
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_search":
            cli_args = ["search", args["query"]]
            if args.get("dir"):
                cli_args.extend(["--dir", args["dir"]])
            if args.get("method"):
                cli_args.extend(["--method", args["method"]])
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_export":
            cli_args = ["export", args["file"], "-f", args["format"]]
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_template":
            action = args["action"]
            cli_args = ["template", action]
            if action == "use" and args.get("name"):
                cli_args.append(args["name"])
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_todo_add":
            cli_args = ["todo-add", args["file"]] + args["tasks"]
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_todo_done":
            cli_args = ["todo-done", args["file"], args["query"]]
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_log":
            cli_args = ["log", args["file"]] + args["entries"]
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_palace_add":
            cli_args = ["palace", "add", args["text"],
                        "--wing", args["wing"], "--room", args["room"]]
            if args.get("hall"):
                cli_args.extend(["--hall", args["hall"]])
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                try:
                    crumb.main(cli_args)
                except SystemExit:
                    pass
            return stdout_buf.getvalue() + stderr_buf.getvalue()

        elif name == "crumb_palace_list":
            cli_args = ["palace", "list"]
            if args.get("wing"):
                cli_args.extend(["--wing", args["wing"]])
            if args.get("hall"):
                cli_args.extend(["--hall", args["hall"]])
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_palace_search":
            cli_args = ["palace", "search", args["query"]]
            if args.get("wing"):
                cli_args.extend(["--wing", args["wing"]])
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                try:
                    crumb.main(cli_args)
                except SystemExit:
                    pass
            return stdout_buf.getvalue() + stderr_buf.getvalue()

        elif name == "crumb_palace_wiki":
            cli_args = ["palace", "wiki"]
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                try:
                    crumb.main(cli_args)
                except SystemExit:
                    pass
            return stdout_buf.getvalue() + stderr_buf.getvalue()

        elif name == "crumb_reflect":
            cli_args = ["reflect"]
            if args.get("format"):
                cli_args.extend(["-f", args["format"]])
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                try:
                    crumb.main(cli_args)
                except SystemExit:
                    pass
            return stdout_buf.getvalue() + stderr_buf.getvalue()

        elif name == "crumb_wake":
            cli_args = ["wake"]
            if args.get("max_facts"):
                cli_args.extend(["--max-facts", str(args["max_facts"])])
            if args.get("reflect"):
                cli_args.append("--reflect")
            if args.get("metalk"):
                cli_args.append("--metalk")
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_context":
            cli_args = ["context"]
            if args.get("goal"):
                cli_args.extend(["--goal", args["goal"]])
            if args.get("title"):
                cli_args.extend(["--title", args["title"]])
            if args.get("commits"):
                cli_args.extend(["--commits", str(args["commits"])])
            if args.get("metalk"):
                cli_args.append("--metalk")
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_metalk":
            cli_args = ["metalk", args["file"]]
            if args.get("level"):
                cli_args.extend(["--level", str(args["level"])])
            if args.get("decode"):
                cli_args.append("--decode")
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_classify":
            cli_args = ["classify", "--text", args["text"]]
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        else:
            return f"Unknown tool: {name}"

    except SystemExit:
        return stdout_buf.getvalue() + stderr_buf.getvalue()
    except Exception as e:
        return f"Error: {e}"


def main():
    """Run the MCP server using stdio transport."""
    while True:
        msg = read_stdin()
        if msg is None:
            break

        method = msg.get("method", "")
        req_id = msg.get("id")

        if method == "initialize":
            respond(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "crumb-format",
                    "version": "0.2.0",
                },
            })

        elif method == "notifications/initialized":
            pass  # No response needed

        elif method == "tools/list":
            respond(req_id, {"tools": TOOLS})

        elif method == "tools/call":
            tool_name = msg["params"]["name"]
            tool_args = msg["params"].get("arguments", {})
            result = handle_tool_call(tool_name, tool_args)
            respond(req_id, {
                "content": [{"type": "text", "text": result}],
            })

        elif req_id is not None:
            respond_error(req_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()
