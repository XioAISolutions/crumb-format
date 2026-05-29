"""Provider abstraction.

Every provider exposes a single method:

    generate(prompt, max_tokens, temperature) -> str

Providers are thin: they take a fully-rendered prompt and return raw model
text. All CRUMB awareness, prompt assembly, and quality gating lives in the
engine, not here.
"""

from __future__ import annotations

import abc
import json
import urllib.error
import urllib.request


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce output (missing key, no server)."""


class BaseProvider(abc.ABC):
    """Base class for all CrumbLLM providers."""

    name: str = "base"

    def __init__(self, model: str = "", base_url: str = ""):
        self.model = model
        self.base_url = base_url

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        """Return raw model text for ``prompt``."""
        raise NotImplementedError

    def available(self) -> tuple[bool, str]:
        """Return ``(ready, reason)``. Default: always ready."""
        return True, "ok"


def http_post_json(
    url: str,
    payload: dict,
    headers: dict | None = None,
    timeout: float = 120.0,
) -> dict:
    """Minimal stdlib JSON POST so providers need no third-party HTTP deps."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        body = exc.read().decode("utf-8", "replace")
        raise ProviderError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network
        raise ProviderError(f"could not reach {url}: {exc.reason}") from exc


class MockProvider(BaseProvider):
    """Deterministic offline provider.

    Used as the default when nothing is configured and as the test double. It
    requires no keys and no network, satisfying "the package works without
    cloud keys". It echoes a compact, structured stub derived from the prompt
    so that downstream JSON gates and the CLI have something real to chew on.
    """

    name = "mock"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        wants_json = '"' in prompt and (
            "JSON" in prompt or "json" in prompt
        )
        if wants_json:
            return json.dumps(
                {
                    "summary": "Mock analysis (no live provider configured).",
                    "risks": [],
                    "next_actions": [],
                    "note": "Run `crumblm setup ...` to use a real provider.",
                }
            )
        return (
            "Mock analysis (no live provider configured).\n"
            "Configure a provider with `crumblm setup ...` to get real output."
        )
