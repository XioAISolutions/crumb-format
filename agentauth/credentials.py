"""CredentialBroker — issue and validate short-lived tool credentials."""

import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .store import PassportStore


class CredentialBroker:
    """Issue HMAC-based short-lived tokens scoped to agent + tool."""

    def __init__(self, store: PassportStore = None, secret_key: str = None):
        self.store = store or PassportStore()
        self.tokens_path = self.store.root / "tokens.json"
        self.secret_path = self.store.root / ".secret"

        if secret_key:
            self.secret = secret_key.encode()
        elif self.secret_path.exists():
            self.secret = self.secret_path.read_bytes()
        else:
            self.secret = os.urandom(32)
            self.secret_path.write_bytes(self.secret)

    # ── token storage helpers ──────────────────────────────────

    def _load_tokens(self) -> list:
        if self.tokens_path.exists():
            return json.loads(self.tokens_path.read_text(encoding="utf-8"))
        return []

    def _save_tokens(self, tokens: list) -> None:
        self.tokens_path.write_text(
            json.dumps(tokens, indent=2) + "\n", encoding="utf-8"
        )

    # ── issue ──────────────────────────────────────────────────

    def issue(self, agent_id: str, tool: str, ttl_seconds: int = 300) -> dict:
        # Verify passport
        from .passport import AgentPassport

        passport = AgentPassport(self.store)
        verification = passport.verify(agent_id)
        if not verification["valid"]:
            raise PermissionError(
                f"cannot issue credential: {verification['reason']}"
            )

        # Check policy
        from .policy import ToolPolicy

        policy = ToolPolicy(self.store)
        check = policy.check(agent_id, tool)
        if not check["allowed"]:
            raise PermissionError(f"policy denied: {check['reason']}")

        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=ttl_seconds)

        token = hmac.new(
            self.secret,
            f"{agent_id}:{tool}:{expires.isoformat()}".encode(),
            hashlib.sha256,
        ).hexdigest()

        record = {
            "token": token,
            "agent_id": agent_id,
            "tool": tool,
            "issued": now.isoformat(),
            "expires": expires.isoformat(),
            "ttl": ttl_seconds,
        }

        tokens = self._load_tokens()
        tokens.append(record)
        self._save_tokens(tokens)

        return record

    # ── validate ───────────────────────────────────────────────

    def validate(self, token: str, agent_id: str, tool: str) -> dict:
        tokens = self._load_tokens()
        now = datetime.now(timezone.utc)

        for record in tokens:
            if (
                record["token"] == token
                and record["agent_id"] == agent_id
                and record["tool"] == tool
            ):
                expires_dt = datetime.fromisoformat(record["expires"])
                if now > expires_dt:
                    return {
                        "valid": False,
                        "agent_id": agent_id,
                        "tool": tool,
                        "expires": record["expires"],
                    }
                return {
                    "valid": True,
                    "agent_id": agent_id,
                    "tool": tool,
                    "expires": record["expires"],
                }

        return {
            "valid": False,
            "agent_id": agent_id,
            "tool": tool,
            "expires": "",
        }

    # ── revoke_all ─────────────────────────────────────────────

    def revoke_all(self, agent_id: str) -> int:
        tokens = self._load_tokens()
        original_count = len(tokens)
        tokens = [t for t in tokens if t["agent_id"] != agent_id]
        removed = original_count - len(tokens)
        self._save_tokens(tokens)
        return removed
