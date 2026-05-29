"""Ollama provider — fully local, no cloud key required.

Defaults to the standard Ollama endpoint ``http://localhost:11434/api/generate``.
"""

from __future__ import annotations

from crumb_llm.providers.base import BaseProvider, ProviderError, http_post_json

DEFAULT_MODEL = "llama3.1"
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, model: str = "", base_url: str = ""):
        super().__init__(model or DEFAULT_MODEL, base_url or DEFAULT_BASE_URL)

    def generate(self, prompt, max_tokens=1024, temperature=0.2) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        resp = http_post_json(url, payload)
        text = resp.get("response")
        if text is None:
            raise ProviderError(f"unexpected Ollama response: {resp}")
        return text
