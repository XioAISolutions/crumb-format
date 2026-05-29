"""Provider registry and factory."""

from __future__ import annotations

from crumb_llm.models import ProviderConfig
from crumb_llm.providers.anthropic_provider import AnthropicProvider
from crumb_llm.providers.base import BaseProvider, MockProvider, ProviderError
from crumb_llm.providers.lmstudio_provider import LMStudioProvider
from crumb_llm.providers.ollama_provider import OllamaProvider
from crumb_llm.providers.openai_provider import OpenAIProvider
from crumb_llm.providers.scratch_provider import ScratchProvider

_REGISTRY: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "lmstudio": LMStudioProvider,
    "scratch": ScratchProvider,
    "mock": MockProvider,
}


def get_provider(config: ProviderConfig | None) -> BaseProvider:
    """Build a provider instance from config.

    Falls back to the :class:`MockProvider` when nothing is configured, so the
    package always works offline without keys.
    """
    if config is None or not config.provider:
        return MockProvider()
    cls = _REGISTRY.get(config.provider)
    if cls is None:
        raise ProviderError(f"unknown provider: {config.provider}")
    return cls(model=config.model, base_url=config.base_url)


__all__ = [
    "AnthropicProvider",
    "BaseProvider",
    "LMStudioProvider",
    "MockProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderError",
    "ScratchProvider",
    "get_provider",
]
