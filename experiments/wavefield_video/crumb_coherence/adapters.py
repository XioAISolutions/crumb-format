"""crumb_coherence.adapters — pipeline-agnostic entry point (§6).

`wrap_generator` decorates any segment generator so every emitted clip is passed
through the coherence engine with threaded state. The generator never learns
which engine (spectral or stats-EMA baseline) is running behind the interface.
"""
from __future__ import annotations

from typing import Callable

import torch


def wrap_generator(gen_segment_fn: Callable[..., torch.Tensor], engine, state):
    """Wrap `gen_segment_fn() -> [T,3,H,W]` with per-segment coherence.

    Parameters
    ----------
    gen_segment_fn : callable returning a segment tensor [T,3,H,W] in [0,1].
    engine         : a SpectralCoherenceEngine or StatsEMAEngine.
    state          : the CoherenceState/StatsState to thread across calls.

    Returns
    -------
    wrapped : callable with the same signature as `gen_segment_fn`. Each call
              generates a segment, runs it through `engine.process_segment`, and
              returns the corrected segment. State is threaded automatically via
              a closure, so successive segments share one rolling anchor.
    """
    box = {"state": state}

    def wrapped(*args, **kwargs):
        seg = gen_segment_fn(*args, **kwargs)
        corrected, box["state"] = engine.process_segment(seg, box["state"])
        return corrected

    wrapped.get_state = lambda: box["state"]   # inspect the threaded anchor
    return wrapped
