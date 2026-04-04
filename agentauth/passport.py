"""AgentPassport — register, inspect, verify, and revoke agent passports."""

import hashlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Import parse_crumb / render_crumb from the CLI module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cli.crumb import parse_crumb, render_crumb  # noqa: E402

from .store import PassportStore  # noqa: E402


class AgentPassport:
    """Manage agent passport lifecycle: register, inspect, verify, revoke."""

    def __init__(self, store: PassportStore = None):
        self.store = store or PassportStore()

    # ── register ───────────────────────────────────────────────

    def register(
        self,
        name: str,
        framework: str = "unknown",
        owner: str = "",
        tools_allowed: list = None,
        tools_denied: list = None,
        data_classes: list = None,
        ttl_days: int = 90,
    ) -> dict:
        now = datetime.now(timezone.utc)
        timestamp = now.isoformat()

        agent_id = "ap_" + hashlib.sha256(
            (name + timestamp).encode()
        ).hexdigest()[:8]

        fingerprint = "sha256:" + hashlib.sha256(
            (name + framework + agent_id).encode()
        ).hexdigest()[:16]

        issued = now.strftime("%Y-%m-%d")
        expires = (now + timedelta(days=ttl_days)).strftime("%Y-%m-%d")

        headers = {
            "v": "1.1",
            "kind": "passport",
            "source": f"agentauth/{name}",
            "id": agent_id,
            "agent_name": name,
            "agent_framework": framework,
            "issued": issued,
            "expires": expires,
            "status": "active",
        }

        identity_lines = [
            f"  name: {name}",
            f"  framework: {framework}",
            f"  owner: {owner}" if owner else f"  owner: (unspecified)",
            f"  fingerprint: {fingerprint}",
        ]

        permissions_lines = [
            f"  tools_allowed: {', '.join(tools_allowed) if tools_allowed else '*'}",
            f"  tools_denied: {', '.join(tools_denied) if tools_denied else 'none'}",
            f"  data_classes: {', '.join(data_classes) if data_classes else 'all'}",
        ]

        sections = {
            "identity": identity_lines,
            "permissions": permissions_lines,
        }

        content = render_crumb(headers, sections)
        path = self.store.save_passport(content, agent_id)

        return {"agent_id": agent_id, "name": name, "passport_path": str(path)}

    # ── inspect ────────────────────────────────────────────────

    def inspect(self, agent_id_or_name: str) -> "dict | None":
        text = self.store.load_passport(agent_id_or_name)
        if text is None:
            return None
        return parse_crumb(text)

    # ── revoke ─────────────────────────────────────────────────

    def revoke(self, agent_id: str) -> bool:
        text = self.store.load_passport(agent_id)
        if text is None:
            return False

        success = self.store.revoke(agent_id)
        if not success:
            return False

        # Update the passport crumb status header
        updated = text.replace("status=active", "status=revoked")
        self.store.save_passport(updated, agent_id)
        return True

    # ── verify ─────────────────────────────────────────────────

    def verify(self, agent_id: str) -> dict:
        text = self.store.load_passport(agent_id)
        if text is None:
            return {"valid": False, "reason": "passport not found", "passport": None}

        try:
            data = parse_crumb(text)
        except ValueError as exc:
            return {"valid": False, "reason": f"parse error: {exc}", "passport": None}

        headers = data["headers"]

        # Check revocation
        effective_id = headers.get("id", agent_id)
        if self.store.is_revoked(effective_id):
            return {"valid": False, "reason": "passport revoked", "passport": data}

        # Check status header
        if headers.get("status") == "revoked":
            return {"valid": False, "reason": "passport revoked", "passport": data}

        # Check expiration
        expires_str = headers.get("expires", "")
        if expires_str:
            try:
                expires_date = datetime.strptime(expires_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                if datetime.now(timezone.utc) > expires_date:
                    return {
                        "valid": False,
                        "reason": "passport expired",
                        "passport": data,
                    }
            except ValueError:
                pass

        return {"valid": True, "reason": "valid", "passport": data}

    # ── list_all ───────────────────────────────────────────────

    def list_all(self, status_filter: str = "all") -> "list[dict]":
        results = []
        for path in self.store.list_passports():
            text = path.read_text(encoding="utf-8")
            try:
                data = parse_crumb(text)
            except ValueError:
                continue
            headers = data["headers"]
            status = headers.get("status", "unknown")
            if status_filter != "all" and status != status_filter:
                continue
            results.append(
                {
                    "agent_id": headers.get("id", path.stem),
                    "name": headers.get("agent_name", ""),
                    "status": status,
                    "issued": headers.get("issued", ""),
                    "expires": headers.get("expires", ""),
                    "path": str(path),
                }
            )
        return results
