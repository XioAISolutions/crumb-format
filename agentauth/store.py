"""PassportStore — file-based storage backend for Agent Passport."""

import json
from datetime import date, datetime
from pathlib import Path


class PassportStore:
    """File-based storage for passports, policies, audits, and revocations.

    Layout:
        .crumb-auth/
          passports/ap_XXXXXXXX.crumb
          policies/agent-name.json
          audit/YYYY-MM-DD/as_XXXXX.crumb
          revoked.json
    """

    def __init__(self, root: str = ".crumb-auth"):
        self.root = Path(root)
        self.passports_dir = self.root / "passports"
        self.policies_dir = self.root / "policies"
        self.audit_dir = self.root / "audit"
        self.revoked_path = self.root / "revoked.json"

        # Create directories if missing
        self.passports_dir.mkdir(parents=True, exist_ok=True)
        self.policies_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    # ── passports ──────────────────────────────────────────────

    def save_passport(self, content: str, agent_id: str) -> Path:
        path = self.passports_dir / f"{agent_id}.crumb"
        path.write_text(content, encoding="utf-8")
        return path

    def load_passport(self, agent_id: str) -> "str | None":
        """Load a passport by agent_id (ap_xxx) or by agent name."""
        # Try direct id match first
        path = self.passports_dir / f"{agent_id}.crumb"
        if path.exists():
            return path.read_text(encoding="utf-8")

        # Search by agent name inside passport files
        for p in self.passports_dir.glob("ap_*.crumb"):
            text = p.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("agent_name=") and stripped.split("=", 1)[1].strip() == agent_id:
                    return text
        return None

    def list_passports(self) -> "list[Path]":
        return sorted(self.passports_dir.glob("ap_*.crumb"))

    # ── revocation ─────────────────────────────────────────────

    def _load_revoked_set(self) -> set:
        if self.revoked_path.exists():
            data = json.loads(self.revoked_path.read_text(encoding="utf-8"))
            return set(data.get("revoked", []))
        return set()

    def _save_revoked_set(self, revoked: set) -> None:
        self.revoked_path.write_text(
            json.dumps({"revoked": sorted(revoked)}, indent=2) + "\n",
            encoding="utf-8",
        )

    def revoke(self, agent_id: str) -> bool:
        revoked = self._load_revoked_set()
        if agent_id in revoked:
            return False
        revoked.add(agent_id)
        self._save_revoked_set(revoked)
        return True

    def is_revoked(self, agent_id: str) -> bool:
        return agent_id in self._load_revoked_set()

    # ── policies ───────────────────────────────────────────────

    def save_policy(self, agent_name: str, policy: dict) -> Path:
        path = self.policies_dir / f"{agent_name}.json"
        path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        return path

    def load_policy(self, agent_name: str) -> "dict | None":
        path = self.policies_dir / f"{agent_name}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def load_default_policy(self) -> "dict | None":
        return self.load_policy("default")

    # ── audit ──────────────────────────────────────────────────

    def save_audit(self, content: str, session_id: str) -> Path:
        today = date.today().isoformat()
        day_dir = self.audit_dir / today
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{session_id}.crumb"
        path.write_text(content, encoding="utf-8")
        return path

    def list_audits(self, agent_id: str = None, since: str = None) -> "list[Path]":
        """List audit files, optionally filtered by agent_id or since date."""
        results = []
        for day_dir in sorted(self.audit_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            dir_name = day_dir.name
            # Filter by date if since is provided
            if since and dir_name < since:
                continue
            for crumb_file in sorted(day_dir.glob("as_*.crumb")):
                if agent_id:
                    text = crumb_file.read_text(encoding="utf-8")
                    if f"agent_id={agent_id}" not in text:
                        continue
                results.append(crumb_file)
        return results
