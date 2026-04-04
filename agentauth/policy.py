"""ToolPolicy — set, check, and test tool-level permissions for agents."""

import fnmatch
from datetime import datetime, timezone

from .store import PassportStore


class ToolPolicy:
    """Evaluate whether an agent is allowed to use a given tool."""

    def __init__(self, store: PassportStore = None):
        self.store = store or PassportStore()

    # ── set_policy ─────────────────────────────────────────────

    def set_policy(
        self,
        agent_name: str,
        tools_allowed: list = None,
        tools_denied: list = None,
        data_classes: list = None,
        max_actions_per_session: int = 1000,
    ) -> dict:
        policy = {
            "agent_name": agent_name,
            "tools_allowed": tools_allowed or [],
            "tools_denied": tools_denied or [],
            "data_classes": data_classes or [],
            "max_actions_per_session": max_actions_per_session,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        self.store.save_policy(agent_name, policy)
        return policy

    # ── check ──────────────────────────────────────────────────

    def check(
        self,
        agent_id_or_name: str,
        tool: str,
        action: str = None,
        data_class: str = None,
    ) -> dict:
        # Resolve agent name from passport
        from .passport import AgentPassport

        passport = AgentPassport(self.store)
        verification = passport.verify(agent_id_or_name)

        if not verification["valid"]:
            return {
                "allowed": False,
                "reason": verification["reason"],
                "tool": tool,
                "agent_id": agent_id_or_name,
            }

        data = verification["passport"]
        agent_name = data["headers"].get("agent_name", agent_id_or_name)
        agent_id = data["headers"].get("id", agent_id_or_name)

        return self._evaluate(agent_name, agent_id, tool, action, data_class)

    # ── test (dry-run) ─────────────────────────────────────────

    def test(self, agent_name: str, tool: str) -> dict:
        """Evaluate policy without requiring a valid passport (dry-run)."""
        return self._evaluate(agent_name, agent_name, tool)

    # ── internal evaluation ────────────────────────────────────

    def _evaluate(
        self,
        agent_name: str,
        agent_id: str,
        tool: str,
        action: str = None,
        data_class: str = None,
    ) -> dict:
        policy = self.store.load_policy(agent_name)
        if policy is None:
            policy = self.store.load_default_policy()
        if policy is None:
            # No policy at all — allow by default
            return {
                "allowed": True,
                "reason": "no policy defined (default allow)",
                "tool": tool,
                "agent_id": agent_id,
            }

        tools_denied = policy.get("tools_denied", [])
        tools_allowed = policy.get("tools_allowed", [])
        allowed_data = policy.get("data_classes", [])

        # Check denied list (supports fnmatch globs)
        for pattern in tools_denied:
            if fnmatch.fnmatch(tool, pattern):
                return {
                    "allowed": False,
                    "reason": f"tool '{tool}' matches denied pattern '{pattern}'",
                    "tool": tool,
                    "agent_id": agent_id,
                }

        # Check allowed list
        if tools_allowed:
            matched = any(fnmatch.fnmatch(tool, pat) for pat in tools_allowed)
            if not matched:
                return {
                    "allowed": False,
                    "reason": f"tool '{tool}' not in allowed list",
                    "tool": tool,
                    "agent_id": agent_id,
                }

        # Check data class restriction
        if data_class and allowed_data and data_class not in allowed_data:
            return {
                "allowed": False,
                "reason": f"data class '{data_class}' not permitted",
                "tool": tool,
                "agent_id": agent_id,
            }

        return {
            "allowed": True,
            "reason": "policy allows",
            "tool": tool,
            "agent_id": agent_id,
        }
