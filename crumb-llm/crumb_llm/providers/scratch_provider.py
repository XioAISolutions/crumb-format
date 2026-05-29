"""Experimental local "from-scratch" provider.

This loads a small model trained by ``crumb_llm.training`` and generates text
locally with no network and no cloud key. It is **experimental** and entirely
optional: it must fail *gracefully* when torch is missing or no model file is
present, never crashing the rest of CrumbLLM.

It deliberately does NOT train anything. Training lives in
``crumb_llm.training`` and is never invoked automatically.
"""

from __future__ import annotations

import os

from crumb_llm.providers.base import BaseProvider, ProviderError

DEFAULT_MODEL_PATH = os.path.expanduser("~/.cache/crumb-llm/scratch-model.pt")


class ScratchProvider(BaseProvider):
    name = "scratch"

    def __init__(self, model: str = "", base_url: str = ""):
        # For scratch, "model" is a path to a checkpoint file.
        super().__init__(model or DEFAULT_MODEL_PATH, base_url)

    def _checkpoint_path(self) -> str:
        return self.model or DEFAULT_MODEL_PATH

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except Exception:
            return False, "torch is not installed (pip install 'crumb-llm[scratch]')"
        path = self._checkpoint_path()
        if not os.path.exists(path):
            return False, f"no scratch model at {path}"
        return True, "ok (experimental)"

    def generate(self, prompt, max_tokens=1024, temperature=0.2) -> str:
        ready, reason = self.available()
        if not ready:
            raise ProviderError(
                f"scratch provider unavailable: {reason}. "
                "The scratch provider is experimental; use a cloud or local "
                "(ollama/lmstudio) provider for production."
            )
        # Lazy import so the dependency stays optional.
        try:
            from crumb_llm.training.train_scratch import load_and_generate
        except Exception as exc:  # pragma: no cover - optional path
            raise ProviderError(f"scratch backend failed to load: {exc}") from exc
        return load_and_generate(
            self._checkpoint_path(),
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
