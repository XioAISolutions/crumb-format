"""Crumb LLM — O(N log N) language modeling via wave-equation dynamics.

The package metadata and light utilities are importable without PyTorch so
standalone packaging and ``crumb-llm info`` work on clean machines. Model
symbols are loaded lazily and raise a focused install hint when PyTorch is
missing.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.3.1"

_TORCH_HINT = (
    "crumb_llm requires PyTorch for model operations. Install with:\n"
    "    pip install crumb-llm\n"
    "or, from a crumb-format checkout:\n"
    "    pip install -e '.[llm]'"
)

_EXPORTS: dict[str, tuple[str, str]] = {
    # Core kernels / scatter
    "wave_kernel_time": ("crumb_llm.kernels", "wave_kernel_time"),
    "wave_kernel_freq": ("crumb_llm.kernels", "wave_kernel_freq"),
    "fft_convolve": ("crumb_llm.kernels", "fft_convolve"),
    "scatter_linear": ("crumb_llm.scatter_gather", "scatter_linear"),
    "gather_linear": ("crumb_llm.scatter_gather", "gather_linear"),
    # Layers
    "RMSNorm": ("crumb_llm.layers", "RMSNorm"),
    "SwiGLUFFN": ("crumb_llm.layers", "SwiGLUFFN"),
    "WaveFieldHead": ("crumb_llm.layers", "WaveFieldHead"),
    "WaveFieldBlock": ("crumb_llm.layers", "WaveFieldBlock"),
    # V2 blocks
    "SmoothCausalKernel": ("crumb_llm.v2", "SmoothCausalKernel"),
    "LearnedScatterGather": ("crumb_llm.v2", "LearnedScatterGather"),
    "QueryFrequencyGate": ("crumb_llm.v2", "QueryFrequencyGate"),
    "ShortConvGate": ("crumb_llm.v2", "ShortConvGate"),
    "RotaryFieldEncoding": ("crumb_llm.v2", "RotaryFieldEncoding"),
    "WaveFieldBlockV2": ("crumb_llm.v2", "WaveFieldBlockV2"),
    "WaveFieldBlockV2Config": ("crumb_llm.v2", "WaveFieldBlockV2Config"),
    "AdaptiveKernelHead": ("crumb_llm.v2", "AdaptiveKernelHead"),
    "FieldAttentionGate": ("crumb_llm.v2", "FieldAttentionGate"),
    "ResonanceMemory": ("crumb_llm.v2", "ResonanceMemory"),
    "SpectralGate": ("crumb_llm.v2", "SpectralGate"),
    # Model / generation
    "WaveFieldLM": ("crumb_llm.model", "WaveFieldLM"),
    "WaveFieldConfig": ("crumb_llm.model", "WaveFieldConfig"),
    "FieldStateCache": ("crumb_llm.cache", "FieldStateCache"),
    "generate_cached": ("crumb_llm.cache", "generate_cached"),
    # Hub
    "save_for_hub": ("crumb_llm.hub", "save_for_hub"),
    "load_hub_model": ("crumb_llm.hub", "load_hub_model"),
    # Registry
    "download_model": ("crumb_llm.registry", "download_model"),
    "find_model": ("crumb_llm.registry", "find_model"),
    "list_models": ("crumb_llm.registry", "list_models"),
    "register_local_model": ("crumb_llm.registry", "register_local_model"),
    # Context pulling
    "CrumbIndex": ("crumb_llm.context_pull", "CrumbIndex"),
    "ContextPullSession": ("crumb_llm.context_pull", "ContextPullSession"),
    "pull_context": ("crumb_llm.context_pull", "pull_context"),
    # Quantization
    "quantize_dynamic_linear": ("crumb_llm.quantize", "quantize_dynamic_linear"),
    "model_size_mb": ("crumb_llm.quantize", "model_size_mb"),
    "linear_param_breakdown": ("crumb_llm.quantize", "linear_param_breakdown"),
    # Chat templates
    "apply_chat_template": ("crumb_llm.chat", "apply_chat_template"),
    "stop_tokens_for": ("crumb_llm.chat", "stop_tokens_for"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'crumb_llm' has no attribute {name!r}") from exc

    try:
        value = getattr(import_module(module_name), attr_name)
    except ImportError as exc:
        if exc.name == "torch":
            raise ImportError(_TORCH_HINT) from exc
        raise

    globals()[name] = value
    return value
