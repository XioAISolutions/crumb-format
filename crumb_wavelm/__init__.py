"""Crumb LLM — O(N log N) language modeling via wave-equation dynamics.

Crumb LLM is an experimental open-source architecture that replaces the
traditional O(N²) transformer self-attention with physics-based wave
equations at O(N log N) complexity. It is native to the crumb-format
ecosystem and treats CRUMB document structure as physical priors on a
continuous wave field.

Architecture: each layer performs three steps instead of QKᵀV attention:

    1. Scatter   — tokens deposit their state onto a continuous 1-D field
    2. Convolve  — an FFT-based wave kernel propagates information across
                   the field in O(F log F) time
    3. Gather    — tokens read updated state back from the field at their
                   positions

Each head learns three physics scalars that parameterise the kernel:

    k(t) = exp(-α |t|) · cos(ω t + φ)

with α (damping), ω (frequency), φ (phase). Convolution is performed in
the frequency domain via rfft / irfft.

Advanced physics features:
    - Multi-scale fields: heads can operate at different field resolutions
    - Dispersion: frequency-dependent wave speed via learned dispersion
    - Boundary conditions: periodic (default), absorbing, or reflecting
    - Wave-packet heads: Gabor wavelet kernels for localised attention
    - Interference mixing: multi-head wave superposition with learned coupling

The package metadata and light utilities are importable without PyTorch so
standalone packaging and ``crumb-wavelm info`` work on clean machines. Model
symbols are loaded lazily and raise a focused install hint when PyTorch is
missing.

Crumb-aware extensions (optional, off by default) let crumb section
boundaries, fold priorities, and @priority annotations bias the field
dynamics. See ``crumb_adapter`` and ``docs/crumb-llm-architecture.md``.

Based on Wave-Field LLM (Badaramoni 2026; cf. arXiv:2510.04304
"Wave-PDE Nets") — extended with Crumb-native structural priors
and advanced wave physics.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.3.1"

_TORCH_HINT = (
    "crumb_wavelm requires PyTorch for model operations. Install with:\n"
    "    pip install crumb-wavelm\n"
    "or, from a crumb-format checkout:\n"
    "    python -m crumb_wavelm.setup_standalone --output ./crumb-wavelm-pkg"
)

_EXPORTS: dict[str, tuple[str, str]] = {
    # Core kernels / scatter
    "wave_kernel_time": ("crumb_wavelm.kernels", "wave_kernel_time"),
    "wave_kernel_freq": ("crumb_wavelm.kernels", "wave_kernel_freq"),
    "fft_convolve": ("crumb_wavelm.kernels", "fft_convolve"),
    "scatter_linear": ("crumb_wavelm.scatter_gather", "scatter_linear"),
    "gather_linear": ("crumb_wavelm.scatter_gather", "gather_linear"),
    # Layers
    "RMSNorm": ("crumb_wavelm.layers", "RMSNorm"),
    "SwiGLUFFN": ("crumb_wavelm.layers", "SwiGLUFFN"),
    "WaveFieldHead": ("crumb_wavelm.layers", "WaveFieldHead"),
    "WaveFieldBlock": ("crumb_wavelm.layers", "WaveFieldBlock"),
    # V2 blocks
    "SmoothCausalKernel": ("crumb_wavelm.v2", "SmoothCausalKernel"),
    "LearnedScatterGather": ("crumb_wavelm.v2", "LearnedScatterGather"),
    "QueryFrequencyGate": ("crumb_wavelm.v2", "QueryFrequencyGate"),
    "ShortConvGate": ("crumb_wavelm.v2", "ShortConvGate"),
    "RotaryFieldEncoding": ("crumb_wavelm.v2", "RotaryFieldEncoding"),
    "WaveFieldBlockV2": ("crumb_wavelm.v2", "WaveFieldBlockV2"),
    "WaveFieldBlockV2Config": ("crumb_wavelm.v2", "WaveFieldBlockV2Config"),
    "AdaptiveKernelHead": ("crumb_wavelm.v2", "AdaptiveKernelHead"),
    "FieldAttentionGate": ("crumb_wavelm.v2", "FieldAttentionGate"),
    "ResonanceMemory": ("crumb_wavelm.v2", "ResonanceMemory"),
    "SpectralGate": ("crumb_wavelm.v2", "SpectralGate"),
    # Model / generation
    "WaveFieldLM": ("crumb_wavelm.model", "WaveFieldLM"),
    "WaveFieldConfig": ("crumb_wavelm.model", "WaveFieldConfig"),
    "FieldStateCache": ("crumb_wavelm.cache", "FieldStateCache"),
    "generate_cached": ("crumb_wavelm.cache", "generate_cached"),
    # Hub
    "save_for_hub": ("crumb_wavelm.hub", "save_for_hub"),
    "load_hub_model": ("crumb_wavelm.hub", "load_hub_model"),
    # Registry
    "download_model": ("crumb_wavelm.registry", "download_model"),
    "find_model": ("crumb_wavelm.registry", "find_model"),
    "list_models": ("crumb_wavelm.registry", "list_models"),
    "register_local_model": ("crumb_wavelm.registry", "register_local_model"),
    # Context pulling
    "CrumbIndex": ("crumb_wavelm.context_pull", "CrumbIndex"),
    "ContextPullSession": ("crumb_wavelm.context_pull", "ContextPullSession"),
    "pull_context": ("crumb_wavelm.context_pull", "pull_context"),
    # Quantization
    "quantize_dynamic_linear": ("crumb_wavelm.quantize", "quantize_dynamic_linear"),
    "model_size_mb": ("crumb_wavelm.quantize", "model_size_mb"),
    "linear_param_breakdown": ("crumb_wavelm.quantize", "linear_param_breakdown"),
    # Chat templates
    "apply_chat_template": ("crumb_wavelm.chat", "apply_chat_template"),
    "stop_tokens_for": ("crumb_wavelm.chat", "stop_tokens_for"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'crumb_wavelm' has no attribute {name!r}") from exc

    try:
        value = getattr(import_module(module_name), attr_name)
    except ImportError as exc:
        if exc.name == "torch":
            raise ImportError(_TORCH_HINT) from exc
        raise

    globals()[name] = value
    return value
