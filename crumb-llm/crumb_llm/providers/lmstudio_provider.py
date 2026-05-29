"""LM Studio provider — fully local, no cloud key required.

LM Studio exposes an OpenAI-compatible server. Defaults to
``http://localhost:1234/v1``.
"""

from __future__ import annotations

from crumb_llm.providers.base import BaseProvider, ProviderError, http_post_json

DEFAULT_MODEL = "local-model"
DEFAULT_BASE_URL = "http://localhost:1234/v1"


class LMStudioProvider(BaseProvider):
    name = "lmstudio"

    def __init__(self, model: str = "", base_url: str = ""):
        super().__init__(model or DEFAULT_MODEL, base_url or DEFAULT_BASE_URL)

    def generate(self, prompt, max_tokens=1024, temperature=0.2) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # LM Studio ignores auth but the OpenAI client convention sends one.
        resp = http_post_json(url, payload, {"Authorization": "Bearer lm-studio"})
        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected LM Studio response: {resp}") from exc
