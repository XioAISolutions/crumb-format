"""Wave-Field LLM — O(N log N) sequence modeling via wave-equation dynamics.

This subpackage is a reference implementation of the Wave-Field LLM
architecture (Badaramoni 2026; cf. arXiv:2510.04304 "Wave-PDE Nets") inside
the crumb-format repo. It replaces O(N^2) self-attention with three steps
per layer:

    1. Scatter   — tokens deposit their state onto a continuous 1-D field
    2. Convolve  — an FFT-based wave kernel propagates information across
                   the field in O(F log F) time
    3. Gather    — tokens read updated state back from the field at their
                   positions

Each head learns three physics scalars:

    k(t) = exp(-α |t|) · cos(ω t + φ)

with α (damping), ω (frequency), φ (phase). Convolution with k is done in
the frequency domain via rfft / irfft.

The package is gated behind the ``[wave]`` install extra. Importing
``wave_field_llm`` without torch installed raises a clear ImportError
pointing at ``pip install crumb-format[wave]``.

Crumb-aware extensions (optional, off by default) let crumb section
boundaries, fold priorities, and @priority annotations bias the field
dynamics. See ``crumb_adapter`` and ``docs/wave-field-llm.md``.
"""

from __future__ import annotations

_TORCH_HINT = (
    "wave_field_llm requires PyTorch. Install with:\n"
    "    pip install 'crumb-format[wave]'\n"
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

__all__ = [
    "wave_kernel_time",
    "wave_kernel_freq",
    "fft_convolve",
    "scatter_linear",
    "gather_linear",
    "RMSNorm",
    "SwiGLUFFN",
    "WaveFieldHead",
    "WaveFieldBlock",
    "WaveFieldLM",
    "WaveFieldConfig",
]

__version__ = "0.1.0"
