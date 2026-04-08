from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from platformdirs import user_data_dir
except ImportError:  # pragma: no cover - exercised when optional dep is missing
    user_data_dir = None


def _history_root() -> Path:
    if user_data_dir is not None:
        return Path(user_data_dir("CRUMB Studio", "XIO AI Solutions"))

    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "CRUMB Studio"
    if sys.platform == "win32":
        return home / "AppData" / "Local" / "CRUMB Studio"
    return home / ".local" / "share" / "crumb-studio"


class HistoryStore:
    """Persist recent Studio generations for quick reload and demo flows."""

    def __init__(self, path: Path | None = None, limit: int = 20) -> None:
        self.path = path or (_history_root() / "history.json")
        self.limit = limit

    def _read_items(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        items = payload.get("items", [])
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    def _write_items(self, items: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"items": items[: self.limit]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_items(self) -> list[dict[str, Any]]:
        return self._read_items()

    def add(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        items = [existing for existing in self._read_items() if existing.get("id") != item.get("id")]
        items.insert(0, item)
        self._write_items(items)
        return items[: self.limit]

    def get(self, item_id: str) -> dict[str, Any] | None:
        for item in self._read_items():
            if item.get("id") == item_id:
                return item
        return None

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
