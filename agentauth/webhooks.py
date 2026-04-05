"""WebhookManager — event subscription and delivery for AgentAuth."""

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .store import PassportStore

logger = logging.getLogger(__name__)

# All recognised event types.
VALID_EVENTS = frozenset(
    [
        "passport.registered",
        "passport.revoked",
        "passport.expired",
        "policy.denied",
        "audit.session_started",
        "audit.session_ended",
        "scan.critical_finding",
    ]
)

# Retry configuration
MAX_RETRIES = 3
BACKOFF_SECONDS = [1, 2, 4]


class WebhookManager:
    """Register, fire, list, and remove webhook subscriptions.

    Subscriptions are persisted to ``.crumb-auth/webhooks.json``.
    Event delivery is non-blocking (threaded) with exponential-backoff retries.
    """

    def __init__(self, store: PassportStore | None = None):
        self.store = store or PassportStore()
        self._path: Path = self.store.root / "webhooks.json"
        self._lock = threading.Lock()

    # ── persistence helpers ────────────────────────────────────

    def _load(self) -> list[dict]:
        if self._path.exists():
            return json.loads(self._path.read_text(encoding="utf-8"))
        return []

    def _save(self, hooks: list[dict]) -> None:
        self._path.write_text(
            json.dumps(hooks, indent=2) + "\n", encoding="utf-8"
        )

    # ── register ──────────────────────────────────────────────

    def register(self, url: str, events: list[str], label: str = "") -> dict:
        """Subscribe *url* to one or more *events*.  Returns the new hook."""
        bad = set(events) - VALID_EVENTS
        if bad:
            raise ValueError(f"unknown event type(s): {', '.join(sorted(bad))}")
        if not events:
            raise ValueError("at least one event type is required")

        hook_id = "wh_" + hashlib.sha256(
            (url + datetime.now(timezone.utc).isoformat() + os.urandom(8).hex()).encode()
        ).hexdigest()[:8]

        hook = {
            "id": hook_id,
            "url": url,
            "events": sorted(events),
            "label": label,
            "created": datetime.now(timezone.utc).isoformat(),
            "active": True,
        }

        with self._lock:
            hooks = self._load()
            hooks.append(hook)
            self._save(hooks)

        logger.info("registered webhook %s -> %s for %s", hook_id, url, events)
        return hook

    # ── list / get ────────────────────────────────────────────

    def list_hooks(self) -> list[dict]:
        """Return all registered webhooks."""
        with self._lock:
            return self._load()

    def get(self, hook_id: str) -> dict | None:
        """Return a single webhook by id, or ``None``."""
        for h in self.list_hooks():
            if h["id"] == hook_id:
                return h
        return None

    # ── remove ────────────────────────────────────────────────

    def remove(self, hook_id: str) -> bool:
        """Remove a webhook.  Returns ``True`` if found and removed."""
        with self._lock:
            hooks = self._load()
            new = [h for h in hooks if h["id"] != hook_id]
            if len(new) == len(hooks):
                return False
            self._save(new)
        logger.info("removed webhook %s", hook_id)
        return True

    # ── fire ──────────────────────────────────────────────────

    def fire(self, event: str, data: dict | None = None) -> None:
        """Dispatch *event* to all matching subscribers (non-blocking)."""
        if event not in VALID_EVENTS:
            raise ValueError(f"unknown event type: {event}")

        payload = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        }

        hooks = self.list_hooks()
        for hook in hooks:
            if not hook.get("active", True):
                continue
            if event in hook["events"]:
                t = threading.Thread(
                    target=self._deliver,
                    args=(hook, payload),
                    daemon=True,
                )
                t.start()

    def _deliver(self, hook: dict, payload: dict) -> bool:
        """POST *payload* to the hook URL with retries.  Returns success."""
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(MAX_RETRIES):
            try:
                req = Request(
                    hook["url"],
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req, timeout=10) as resp:
                    status = resp.status
                if 200 <= status < 300:
                    logger.info(
                        "webhook %s delivered (%s), status=%d",
                        hook["id"],
                        payload["event"],
                        status,
                    )
                    return True
                logger.warning(
                    "webhook %s got status %d (attempt %d/%d)",
                    hook["id"],
                    status,
                    attempt + 1,
                    MAX_RETRIES,
                )
            except (URLError, OSError) as exc:
                logger.warning(
                    "webhook %s delivery failed (attempt %d/%d): %s",
                    hook["id"],
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                )
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_SECONDS[attempt])

        logger.error(
            "webhook %s delivery exhausted retries for event %s",
            hook["id"],
            payload["event"],
        )
        return False

    # ── test ──────────────────────────────────────────────────

    def test(self, hook_id: str) -> dict:
        """Send a test event to the given webhook. Returns delivery result."""
        hook = self.get(hook_id)
        if hook is None:
            return {"success": False, "error": f"webhook not found: {hook_id}"}

        payload = {
            "event": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"message": "This is a test event from AgentAuth."},
        }
        body = json.dumps(payload).encode("utf-8")
        try:
            req = Request(
                hook["url"],
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                return {"success": True, "url": hook["url"], "status_code": resp.status}
        except (URLError, OSError) as exc:
            return {"success": False, "url": hook["url"], "error": str(exc)}

    # ── formatters for common targets ─────────────────────────

    @staticmethod
    def format_slack(event: dict) -> dict:
        """Format *event* payload as a Slack Block Kit message.

        *event* must be the full payload dict (event, timestamp, data).
        """
        title = event.get("event", "unknown.event")
        ts = event.get("timestamp", "")
        data = event.get("data", {})

        detail_lines = [f"*{k}:* {v}" for k, v in data.items()]
        detail_text = "\n".join(detail_lines) if detail_lines else "_no details_"

        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"AgentAuth: {title}",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": detail_text,
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Timestamp: {ts}",
                        }
                    ],
                },
            ]
        }

    @staticmethod
    def format_pagerduty(event: dict) -> dict:
        """Format *event* payload as a PagerDuty Events API v2 trigger."""
        title = event.get("event", "unknown.event")
        data = event.get("data", {})

        severity_map = {
            "passport.revoked": "warning",
            "passport.expired": "warning",
            "policy.denied": "error",
            "scan.critical_finding": "critical",
            "passport.registered": "info",
            "audit.session_started": "info",
            "audit.session_ended": "info",
        }
        severity = severity_map.get(title, "info")

        return {
            "routing_key": "",  # caller must fill in
            "event_action": "trigger",
            "payload": {
                "summary": f"AgentAuth: {title}",
                "source": "crumb-agentauth",
                "severity": severity,
                "timestamp": event.get("timestamp", ""),
                "custom_details": data,
            },
        }

    @staticmethod
    def format_generic(event: dict) -> dict:
        """Return the event payload as-is (plain JSON)."""
        return event
