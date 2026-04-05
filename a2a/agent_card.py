"""A2A Agent Card — generates and serves the CRUMB AgentAuth agent card.

The agent card is a JSON document that describes this agent's capabilities,
skills, and metadata per the Google A2A (Agent-to-Agent) protocol.  It is
served at GET /.well-known/agent.json so that any A2A-compatible agent can
discover and interact with CRUMB AgentAuth services.
"""

import json

DEFAULT_PORT = 8421


def build_agent_card(host: str = "localhost", port: int = DEFAULT_PORT) -> dict:
    """Return the full A2A agent card as a Python dict."""
    return {
        "name": "CRUMB AgentAuth",
        "description": "Agent identity, governance, and handoff format service",
        "version": "0.2.0",
        "url": f"http://{host}:{port}",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
        },
        "skills": [
            {
                "id": "crumb.parse",
                "name": "Parse CRUMB",
                "description": "Parse a CRUMB-formatted text block into structured data",
            },
            {
                "id": "crumb.render",
                "name": "Render CRUMB",
                "description": "Render structured data into CRUMB format",
            },
            {
                "id": "crumb.validate",
                "name": "Validate CRUMB",
                "description": "Validate a CRUMB text block",
            },
            {
                "id": "passport.register",
                "name": "Register Agent",
                "description": "Register a new agent and issue a passport",
            },
            {
                "id": "passport.verify",
                "name": "Verify Agent",
                "description": "Verify an agent passport is valid",
            },
            {
                "id": "passport.revoke",
                "name": "Revoke Agent",
                "description": "Revoke an agent's passport (kill switch)",
            },
            {
                "id": "policy.check",
                "name": "Check Policy",
                "description": "Check if an agent is allowed to use a tool",
            },
            {
                "id": "audit.log",
                "name": "Log Action",
                "description": "Log an agent action to the audit trail",
            },
            {
                "id": "scan.shadow",
                "name": "Shadow AI Scan",
                "description": "Scan a directory for unauthorized AI agents",
            },
        ],
        "authentication": {"schemes": ["none"]},
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
    }


def agent_card_json(host: str = "localhost", port: int = DEFAULT_PORT) -> str:
    """Return the agent card serialised as a JSON string."""
    return json.dumps(build_agent_card(host, port), indent=2)
