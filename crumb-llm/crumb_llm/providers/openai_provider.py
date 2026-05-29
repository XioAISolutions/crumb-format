"""OpenAI provider. Reads ``OPENAI_API_KEY`` from the environment."""

from __future__ import annotations

import os

from crumb_llm.providers.base import BaseProvider, ProviderError, http_post_json

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, model: str = "", base_url: str = ""):
        super().__init__(model or DEFAULT_MODEL, base_url or DEFAULT_BASE_URL)

    def available(self) -> tuple[bool, str]:
        if not os.environ.get("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is not set"
        return True, "ok"

    def generate(self, prompt, max_tokens=1024, temperature=0.2) -> str:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set")
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = http_post_json(url, payload, {"Authorization": f"Bearer {key}"})
        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected OpenAI response: {resp}") from exc
