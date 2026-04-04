"""AuditLogger — session-based audit trail for agent actions."""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cli.crumb import parse_crumb, render_crumb  # noqa: E402

from .store import PassportStore  # noqa: E402


class AuditLogger:
    """Record, export, and review agent audit trails."""

    def __init__(self, store: PassportStore = None):
        self.store = store or PassportStore()
        self._sessions: dict = {}  # session_id -> session state

    # ── start_session ──────────────────────────────────────────

    def start_session(self, agent_id: str, goal: str) -> str:
        session_id = "as_" + hashlib.sha256(
            (agent_id + datetime.now(timezone.utc).isoformat() + os.urandom(8).hex()).encode()
        ).hexdigest()[:8]

        self._sessions[session_id] = {
            "session_id": session_id,
            "agent_id": agent_id,
            "goal": goal,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "actions": [],
        }
        return session_id

    # ── log_action ─────────────────────────────────────────────

    def log_action(
        self,
        session_id: str,
        tool: str,
        detail: str,
        allowed: bool,
        reason: str = "",
    ) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id}")

        verdict = "ALLOW" if allowed else "DENY"
        session["actions"].append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": tool,
                "detail": detail,
                "verdict": verdict,
                "reason": reason,
            }
        )

    # ── end_session ────────────────────────────────────────────

    def end_session(self, session_id: str, status: str = "completed") -> str:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"unknown session: {session_id}")

        end_time = datetime.now(timezone.utc).isoformat()
        actions = session["actions"]

        allow_count = sum(1 for a in actions if a["verdict"] == "ALLOW")
        deny_count = sum(1 for a in actions if a["verdict"] == "DENY")
        total = len(actions)
        risk_score = round(deny_count / total * 100, 1) if total else 0.0

        headers = {
            "v": "1.1",
            "kind": "audit",
            "source": f"agentauth/{session['agent_id']}",
            "id": session_id,
            "agent_id": session["agent_id"],
            "session_start": session["start_time"],
            "session_end": end_time,
        }

        goal_lines = [f"  {session['goal']}"]

        action_lines = []
        for a in actions:
            line = f"  [{a['timestamp']}] {a['verdict']} {a['tool']}: {a['detail']}"
            if a["reason"]:
                line += f" ({a['reason']})"
            action_lines.append(line)
        if not action_lines:
            action_lines = ["  (no actions recorded)"]

        verdict_lines = [
            f"  status: {status}",
            f"  total_actions: {total}",
            f"  allowed: {allow_count}",
            f"  denied: {deny_count}",
            f"  risk_score: {risk_score}%",
        ]

        sections = {
            "goal": goal_lines,
            "actions": action_lines,
            "verdict": verdict_lines,
        }

        content = render_crumb(headers, sections)
        self.store.save_audit(content, session_id)

        # Clean up in-memory state
        del self._sessions[session_id]

        return content

    # ── export_evidence ────────────────────────────────────────

    def export_evidence(
        self,
        agent_id: str = None,
        since: str = None,
        output_format: str = "crumb",
    ) -> str:
        paths = self.store.list_audits(agent_id=agent_id, since=since)
        texts = [p.read_text(encoding="utf-8") for p in paths]

        if output_format == "crumb":
            return "\n".join(texts)

        if output_format == "json":
            records = []
            for text in texts:
                try:
                    data = parse_crumb(text)
                    records.append(data)
                except ValueError:
                    continue
            return json.dumps(records, indent=2)

        if output_format == "csv":
            lines = ["session_id,agent_id,timestamp,verdict,tool,detail"]
            for text in texts:
                try:
                    data = parse_crumb(text)
                except ValueError:
                    continue
                sid = data["headers"].get("id", "")
                aid = data["headers"].get("agent_id", "")
                for action_line in data["sections"].get("actions", []):
                    stripped = action_line.strip()
                    if not stripped or stripped.startswith("("):
                        continue
                    # Parse: [timestamp] VERDICT tool: detail
                    if stripped.startswith("["):
                        try:
                            ts_end = stripped.index("]")
                            ts = stripped[1:ts_end]
                            rest = stripped[ts_end + 2:]
                            parts = rest.split(" ", 1)
                            verdict = parts[0]
                            tool_detail = parts[1] if len(parts) > 1 else ""
                            if ": " in tool_detail:
                                tool, detail = tool_detail.split(": ", 1)
                            else:
                                tool, detail = tool_detail, ""
                            # Escape CSV fields
                            detail = detail.replace('"', '""')
                            lines.append(
                                f'{sid},{aid},{ts},{verdict},{tool},"{detail}"'
                            )
                        except (ValueError, IndexError):
                            continue
            return "\n".join(lines)

        return ""

    # ── feed ───────────────────────────────────────────────────

    def feed(self, agent_id: str = None) -> "list[str]":
        """Load recent audit entries and return formatted action lines."""
        paths = self.store.list_audits(agent_id=agent_id)
        lines = []
        for p in paths:
            try:
                data = parse_crumb(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            sid = data["headers"].get("id", "")
            aid = data["headers"].get("agent_id", "")
            for action_line in data["sections"].get("actions", []):
                stripped = action_line.strip()
                if stripped and not stripped.startswith("("):
                    lines.append(f"[{aid}/{sid}] {stripped}")
        return lines
