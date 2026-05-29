"""Anthropic provider. Reads ``ANTHROPIC_API_KEY`` from the environment."""

from __future__ import annotations

import os

from crumb_llm.providers.base import BaseProvider, ProviderError, http_post_json

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, model: str = "", base_url: str = ""):
        super().__init__(model or DEFAULT_MODEL, base_url or DEFAULT_BASE_URL)

    def available(self) -> tuple[bool, str]:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False, "ANTHROPIC_API_KEY is not set"
        return True, "ok"

    def generate(self, prompt, max_tokens=1024, temperature=0.2) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        url = f"{self.base_url.rstrip('/')}/messages"
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        resp = http_post_json(url, payload, headers)
        try:
            return "".join(
                block.get("text", "")
                for block in resp["content"]
                if block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"unexpected Anthropic response: {resp}") from exc
