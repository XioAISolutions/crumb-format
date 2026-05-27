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

The package is gated behind the ``[llm]`` install extra. Importing
``crumb_llm`` without torch installed raises a clear ImportError
pointing at ``pip install crumb-format[llm]``.

Crumb-aware extensions (optional, off by default) let crumb section
boundaries, fold priorities, and @priority annotations bias the field
dynamics. See ``crumb_adapter`` and ``docs/crumb-llm-architecture.md``.

Based on Wave-Field LLM (Badaramoni 2026; cf. arXiv:2510.04304
"Wave-PDE Nets") — extended with Crumb-native structural priors
and advanced wave physics.
"""

from __future__ import annotations

_TORCH_HINT = (
    "crumb_llm requires PyTorch. Install with:\n"
    "    pip install 'crumb-format[llm]'\n"
    "or directly:\n"
    "    pip install torch numpy"
)

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised by gating
    raise ImportError(_TORCH_HINT) from exc


from .kernels import (  # noqa: E402
    wave_kernel_time,
    wave_kernel_freq,
    fft_convolve,
)
from .scatter_gather import scatter_linear, gather_linear  # noqa: E402
from .layers import (  # noqa: E402
    RMSNorm,
    SwiGLUFFN,
    WaveFieldHead,
    WaveFieldBlock,
)
from .model import WaveFieldLM, WaveFieldConfig  # noqa: E402
from .cache import FieldStateCache, generate_cached  # noqa: E402
from .hub import save_for_hub, load_hub_model  # noqa: E402
from .v2 import (  # noqa: E402
    SmoothCausalKernel,
    LearnedScatterGather,
    QueryFrequencyGate,
    ShortConvGate,
    RotaryFieldEncoding,
    WaveFieldBlockV2,
    WaveFieldBlockV2Config,
    AdaptiveKernelHead,
    FieldAttentionGate,
    ResonanceMemory,
    SpectralGate,
)
from .context_pull import CrumbIndex, ContextPullSession, pull_context  # noqa: E402

__all__ = [
    # Core
    "wave_kernel_time", "wave_kernel_freq", "fft_convolve",
    "scatter_linear", "gather_linear",
    "RMSNorm", "SwiGLUFFN",
    # V1 blocks
    "WaveFieldHead", "WaveFieldBlock",
    # V2 blocks
    "SmoothCausalKernel", "LearnedScatterGather", "QueryFrequencyGate",
    "ShortConvGate", "RotaryFieldEncoding",
    "WaveFieldBlockV2", "WaveFieldBlockV2Config",
    "AdaptiveKernelHead", "FieldAttentionGate", "ResonanceMemory", "SpectralGate",
    # Model
    "WaveFieldLM", "WaveFieldConfig",
    # Cache
    "FieldStateCache", "generate_cached",
    # Hub
    "save_for_hub", "load_hub_model",
    # Context pulling
    "CrumbIndex", "ContextPullSession", "pull_context",
]

__version__ = "0.3.0"
