"""Configuration for CrumbLLM.

Hard rule: **no secrets are ever written to config files or SQLite.** API keys
live only in environment variables (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``)
and are read at call time. The on-disk config stores only non-secret routing:
which provider, which model, which base URL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from crumb_llm.models import ProviderConfig

# Keys we refuse to persist even if a caller tries to slip them in.
_SECRET_KEYS = {
    "api_key",
    "openai_api_key",
    "anthropic_api_key",
    "token",
    "secret",
    "password",
    "key",
}

# Environment variable each provider reads its secret from (never stored).
PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def config_path() -> Path:
    """Resolve the config file path, honoring ``CRUMB_LLM_CONFIG`` and XDG."""
    override = os.environ.get("CRUMB_LLM_CONFIG")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "crumb-llm" / "config.json"


def _scrub_secrets(data: dict) -> dict:
    """Drop any key that looks like a secret. Defense in depth."""
    return {k: v for k, v in data.items() if k.lower() not in _SECRET_KEYS}


def load_config() -> ProviderConfig | None:
    """Load the persisted provider config, or None if unset."""
    path = config_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    raw = _scrub_secrets(raw)
    if not raw.get("provider"):
        return None
    return ProviderConfig(
        provider=raw.get("provider", ""),
        model=raw.get("model", ""),
        base_url=raw.get("base_url", ""),
        backend=raw.get("backend", ""),
    )


def save_config(config: ProviderConfig) -> Path:
    """Persist non-secret provider routing to disk and return the path."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _scrub_secrets(config.to_dict())
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Best-effort: keep the file private.
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def status() -> dict:
    """Return a status snapshot for ``crumblm status``.

    Reports whether required secrets are present in the environment WITHOUT
    ever printing their values.
    """
    config = load_config()
    info: dict = {
        "config_path": str(config_path()),
        "configured": config is not None,
        "provider": config.provider if config else None,
        "model": config.model if config else None,
        "base_url": config.base_url if config else None,
        "backend": config.backend if config else None,
    }
    env_key = PROVIDER_ENV.get(config.provider) if config else None
    if env_key:
        info["secret_env_var"] = env_key
        info["secret_present"] = bool(os.environ.get(env_key))
    else:
        info["secret_present"] = "n/a (local/offline provider)"
    return info
