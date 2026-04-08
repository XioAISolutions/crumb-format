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
        "name": "crumb_pack",
        "description": "Build a deterministic task, mem, or map context pack from a directory of crumbs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dir": {"type": "string", "description": "Directory containing source .crumb files"},
                "query": {"type": "string", "description": "Desired handoff query"},
                "project": {"type": "string", "description": "Optional project filter"},
                "kind": {"type": "string", "enum": ["task", "mem", "map"]},
                "max_total_tokens": {"type": "integer", "description": "Estimated token budget"},
                "strategy": {"type": "string", "enum": ["keyword", "ranked", "recent", "hybrid"]},
            },
            "required": ["dir", "query", "kind", "max_total_tokens"],
        },
    },
    {
        "name": "crumb_lint",
        "description": "Lint CRUMBs for secrets, oversize logs, suspicious headers, and budget issues.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "files": {"type": "array", "items": {"type": "string"}},
                "secrets": {"type": "boolean"},
                "strict": {"type": "boolean"},
                "max_size": {"type": "integer"},
            },
            "required": ["files"],
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

        elif name == "crumb_pack":
            cli_args = [
                "pack",
                "--dir",
                args["dir"],
                "--query",
                args["query"],
                "--kind",
                args["kind"],
                "--max-total-tokens",
                str(args["max_total_tokens"]),
                "-o",
                "-",
            ]
            if args.get("project"):
                cli_args.extend(["--project", args["project"]])
            if args.get("strategy"):
                cli_args.extend(["--strategy", args["strategy"]])
            with redirect_stdout(stdout_buf):
                crumb.main(cli_args)
            return stdout_buf.getvalue()

        elif name == "crumb_lint":
            cli_args = ["lint"] + args["files"]
            if args.get("secrets"):
                cli_args.append("--secrets")
            if args.get("strict"):
                cli_args.append("--strict")
            if args.get("max_size") is not None:
                cli_args.extend(["--max-size", str(args["max_size"])])
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                try:
                    crumb.main(cli_args)
                except SystemExit:
                    pass
            return stdout_buf.getvalue() + stderr_buf.getvalue()

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
