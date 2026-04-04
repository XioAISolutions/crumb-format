#!/usr/bin/env python3
"""MCP server that exposes AgentAuth tools over JSON-RPC 2.0 stdio.

Run with: python3 mcp/agentauth_server.py
Configure in claude_desktop_config.json or .cursor/mcp.json
"""

import json
import sys
from pathlib import Path

# Add parent dir so we can import agentauth package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentauth.passport import AgentPassport
from agentauth.policy import ToolPolicy
from agentauth.credentials import CredentialBroker
from agentauth.audit import AuditLogger
from agentauth.store import PassportStore


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


# ── Tool definitions ──────────────────────────────────────────────

TOOLS = [
    {
        "name": "passport_register",
        "description": "Register a new agent passport. Returns agent_id, name, and passport file path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Agent name"},
                "framework": {"type": "string", "description": "Agent framework (e.g. langchain, autogen, crewai)"},
                "owner": {"type": "string", "description": "Owner or team responsible for this agent"},
                "tools_allowed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tools this agent is allowed to use (glob patterns ok). Omit for unrestricted.",
                },
                "tools_denied": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tools explicitly denied (glob patterns ok).",
                },
                "ttl_days": {
                    "type": "integer",
                    "description": "Passport validity in days (default 90)",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "passport_inspect",
        "description": "Inspect an agent passport by agent_id or name. Returns parsed passport data.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID (ap_xxx) or agent name"},
            },
            "required": ["agent_id"],
        },
    },
    {
        "name": "passport_verify",
        "description": "Verify that an agent passport is valid and not revoked or expired.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID (ap_xxx) or agent name"},
            },
            "required": ["agent_id"],
        },
    },
    {
        "name": "passport_revoke",
        "description": "Revoke an agent passport (kill switch). The agent will no longer pass verification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID (ap_xxx) to revoke"},
            },
            "required": ["agent_id"],
        },
    },
    {
        "name": "passport_list",
        "description": "List all registered agent passports, optionally filtered by status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "description": "Filter by status: 'active', 'revoked', or 'all' (default 'all')",
                },
            },
        },
    },
    {
        "name": "policy_set",
        "description": "Set or update tool policy for an agent. Controls which tools the agent can use.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "description": "Agent name to set policy for"},
                "tools_allowed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tools the agent is allowed to use (glob patterns ok)",
                },
                "tools_denied": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tools explicitly denied (glob patterns ok)",
                },
                "data_classes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Allowed data classifications (e.g. public, internal, confidential)",
                },
                "max_actions_per_session": {
                    "type": "integer",
                    "description": "Maximum actions allowed per session (default 1000)",
                },
            },
            "required": ["agent_name"],
        },
    },
    {
        "name": "policy_check",
        "description": "Check whether an agent is allowed to use a specific tool under current policy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID (ap_xxx) or agent name"},
                "tool": {"type": "string", "description": "Tool name to check access for"},
            },
            "required": ["agent_id", "tool"],
        },
    },
    {
        "name": "credential_issue",
        "description": "Issue a short-lived credential token scoped to an agent and tool. Requires valid passport and policy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID (ap_xxx)"},
                "tool": {"type": "string", "description": "Tool the credential grants access to"},
                "ttl_seconds": {
                    "type": "integer",
                    "description": "Token lifetime in seconds (default 300)",
                },
            },
            "required": ["agent_id", "tool"],
        },
    },
    {
        "name": "credential_validate",
        "description": "Validate a credential token for a given agent and tool.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "token": {"type": "string", "description": "Credential token to validate"},
                "agent_id": {"type": "string", "description": "Agent ID the token was issued for"},
                "tool": {"type": "string", "description": "Tool the token was issued for"},
            },
            "required": ["token", "agent_id", "tool"],
        },
    },
    {
        "name": "audit_start",
        "description": "Start an audit session for an agent. Returns a session_id to use with audit_log and audit_end.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID (ap_xxx)"},
                "goal": {"type": "string", "description": "Goal or purpose of this session"},
            },
            "required": ["agent_id", "goal"],
        },
    },
    {
        "name": "audit_log",
        "description": "Log an action within an audit session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID from audit_start"},
                "tool": {"type": "string", "description": "Tool that was used"},
                "detail": {"type": "string", "description": "Description of the action"},
                "allowed": {"type": "boolean", "description": "Whether the action was allowed"},
                "reason": {"type": "string", "description": "Reason for the allow/deny decision"},
            },
            "required": ["session_id", "tool", "detail", "allowed"],
        },
    },
    {
        "name": "audit_end",
        "description": "End an audit session and persist the audit trail as a .crumb file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID from audit_start"},
                "status": {
                    "type": "string",
                    "description": "Session outcome: 'completed', 'failed', 'aborted' (default 'completed')",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "audit_export",
        "description": "Export audit evidence for an agent. Returns audit trails in the requested format.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID to export audits for (omit for all agents)"},
                "since": {"type": "string", "description": "Only include audits since this date (YYYY-MM-DD)"},
                "format": {
                    "type": "string",
                    "enum": ["crumb", "json", "csv"],
                    "description": "Export format (default 'crumb')",
                },
            },
        },
    },
]


# ── Shared instances (created once per server lifetime) ───────────

store = PassportStore()
passport = AgentPassport(store)
policy = ToolPolicy(store)
credential_broker = CredentialBroker(store)
audit_logger = AuditLogger(store)


# ── Tool dispatch ─────────────────────────────────────────────────

def handle_tool_call(name, args):
    """Execute an AgentAuth tool and return the result text."""
    try:
        if name == "passport_register":
            result = passport.register(
                name=args["name"],
                framework=args.get("framework", "unknown"),
                owner=args.get("owner", ""),
                tools_allowed=args.get("tools_allowed"),
                tools_denied=args.get("tools_denied"),
                ttl_days=args.get("ttl_days", 90),
            )
            return json.dumps(result, indent=2)

        elif name == "passport_inspect":
            result = passport.inspect(args["agent_id"])
            if result is None:
                return json.dumps({"error": "passport not found", "agent_id": args["agent_id"]})
            return json.dumps(result, indent=2)

        elif name == "passport_verify":
            result = passport.verify(args["agent_id"])
            return json.dumps(result, indent=2)

        elif name == "passport_revoke":
            success = passport.revoke(args["agent_id"])
            if success:
                return json.dumps({"revoked": True, "agent_id": args["agent_id"]})
            return json.dumps({"revoked": False, "agent_id": args["agent_id"], "reason": "not found or already revoked"})

        elif name == "passport_list":
            status_filter = args.get("status_filter", "all")
            results = passport.list_all(status_filter=status_filter)
            return json.dumps(results, indent=2)

        elif name == "policy_set":
            result = policy.set_policy(
                agent_name=args["agent_name"],
                tools_allowed=args.get("tools_allowed"),
                tools_denied=args.get("tools_denied"),
                data_classes=args.get("data_classes"),
                max_actions_per_session=args.get("max_actions_per_session", 1000),
            )
            return json.dumps(result, indent=2)

        elif name == "policy_check":
            result = policy.check(
                agent_id_or_name=args["agent_id"],
                tool=args["tool"],
            )
            return json.dumps(result, indent=2)

        elif name == "credential_issue":
            result = credential_broker.issue(
                agent_id=args["agent_id"],
                tool=args["tool"],
                ttl_seconds=args.get("ttl_seconds", 300),
            )
            return json.dumps(result, indent=2)

        elif name == "credential_validate":
            result = credential_broker.validate(
                token=args["token"],
                agent_id=args["agent_id"],
                tool=args["tool"],
            )
            return json.dumps(result, indent=2)

        elif name == "audit_start":
            session_id = audit_logger.start_session(
                agent_id=args["agent_id"],
                goal=args["goal"],
            )
            return json.dumps({"session_id": session_id, "agent_id": args["agent_id"]})

        elif name == "audit_log":
            audit_logger.log_action(
                session_id=args["session_id"],
                tool=args["tool"],
                detail=args["detail"],
                allowed=args["allowed"],
                reason=args.get("reason", ""),
            )
            return json.dumps({"logged": True, "session_id": args["session_id"]})

        elif name == "audit_end":
            content = audit_logger.end_session(
                session_id=args["session_id"],
                status=args.get("status", "completed"),
            )
            return content

        elif name == "audit_export":
            result = audit_logger.export_evidence(
                agent_id=args.get("agent_id"),
                since=args.get("since"),
                output_format=args.get("format", "crumb"),
            )
            return result if result else "(no audit records found)"

        else:
            return f"Unknown tool: {name}"

    except PermissionError as e:
        return json.dumps({"error": str(e)})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


# ── Main server loop ─────────────────────────────────────────────

def main():
    """Run the AgentAuth MCP server using stdio transport."""
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
                    "name": "crumb-agentauth",
                    "version": "0.1.0",
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
