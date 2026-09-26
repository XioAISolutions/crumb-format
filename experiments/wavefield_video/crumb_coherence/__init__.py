"""crumb_coherence — a training-free, pipeline-agnostic low-band spectral
coherence plugin for long (2-5 min) segment/AR video.

Scope (be honest, per PLUGIN_SPEC_crumb_coherence.md §0): this fixes
*global / low-band drift* — exposure, color balance, contrast, large-scale
layout — by holding the low-frequency FFT magnitude steady across segments while
leaving low-band *phase* (motion) free. It does NOT fix object-identity drift,
texture crawl, or semantic incoherence; those live in high frequencies and/or
need the generator's own memory.
"""
from .core import (
    CoherenceState,
    SpectralCoherenceEngine,
    StatsEMAEngine,
    StatsState,
    detect_cut,
    ema_update,
    gaussian_band_weight,
    low_band_box,
    phase_band_weight,
    put_low_band,
    rgb_to_ycbcr,
    target_lowband,
    ycbcr_to_rgb,
)
from .adapters import wrap_generator
from .metrics import (
    delta_e_vs_ref,
    highfreq_ssim,
    lowband_trajectory_variance,
    temporal_flicker,
)

__all__ = [
    "SpectralCoherenceEngine", "StatsEMAEngine", "CoherenceState", "StatsState",
    "rgb_to_ycbcr", "ycbcr_to_rgb", "low_band_box", "put_low_band",
    "gaussian_band_weight", "phase_band_weight", "ema_update", "target_lowband",
    "detect_cut",
    "wrap_generator", "lowband_trajectory_variance", "delta_e_vs_ref",
    "highfreq_ssim", "temporal_flicker",
]
