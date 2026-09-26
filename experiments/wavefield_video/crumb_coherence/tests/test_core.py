"""Unit tests for crumb_coherence (§6 invariants + spec state-byte claims).

Run:  python -m pytest crumb_coherence/tests -q      (from the wavefield_video dir)
  or: python crumb_coherence/tests/test_core.py       (plain assert runner)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from crumb_coherence import (          # noqa: E402
    SpectralCoherenceEngine, StatsEMAEngine, target_lowband, low_band_box,
)

torch.manual_seed(0)


def _clip(T=6, H=32, W=32):
    return torch.rand(T, 3, H, W)


# --- invariant (i): alpha=0 is a byte-identical no-op ---------------------- #
def test_alpha0_byte_identical_spectral():
    for mode in ("magnitude", "dc_only", "complex"):
        eng = SpectralCoherenceEngine(alpha=0.0, anchor_mode=mode)
        st = eng.init_state(32, 32)
        x = _clip()
        y, _ = eng.process_segment(x, st)
        assert torch.equal(x, y), f"alpha=0 not identity for {mode}"


def test_alpha0_byte_identical_stats():
    eng = StatsEMAEngine(alpha=0.0)
    st = eng.init_state(32, 32)
    x = _clip()
    y, _ = eng.process_segment(x, st)
    assert torch.equal(x, y)


# --- invariant (ii): process_segment == process_frame looped --------------- #
def test_segment_equals_frame_loop():
    for Eng, mode in [(SpectralCoherenceEngine, "magnitude"),
                      (SpectralCoherenceEngine, "dc_only"),
                      (SpectralCoherenceEngine, "complex")]:
        eng = Eng(alpha=0.5, anchor_mode=mode)
        x = _clip()
        # batched
        sb = eng.init_state(32, 32)
        yb, _ = eng.process_segment(x, sb)
        # streamed
        sf = eng.init_state(32, 32)
        ys = []
        for t in range(x.shape[0]):
            yt, sf = eng.process_frame(x[t], sf)
            ys.append(yt)
        ys = torch.stack(ys, 0)
        assert torch.equal(yb, ys), f"segment != frame-loop for {mode}"


def test_segment_equals_frame_loop_stats():
    eng = StatsEMAEngine(alpha=0.5)
    x = _clip()
    sb = eng.init_state(32, 32)
    yb, _ = eng.process_segment(x, sb)
    sf = eng.init_state(32, 32)
    ys = []
    for t in range(x.shape[0]):
        yt, sf = eng.process_frame(x[t], sf)
        ys.append(yt)
    assert torch.equal(yb, torch.stack(ys, 0))


# --- cut guard triggers ---------------------------------------------------- #
def test_cut_guard_hard_reset():
    eng = SpectralCoherenceEngine(alpha=0.5, anchor_mode="magnitude",
                                  reset_on_cut=True, cut_thresh=0.35)
    st = eng.init_state(32, 32)
    # Scene A: a dark, smooth low-frequency ramp (mean ~0.35).
    base = 0.30 + 0.10 * torch.linspace(0, 1, 32).view(1, 1, 32).expand(3, 32, 32).clone()
    for _ in range(4):
        _, st = eng.process_frame(base + 0.01 * torch.rand(3, 32, 32), st)
    n_before = st.n_seen
    assert n_before == 4, "scene A wrongly flagged a cut mid-way"
    # Hard scene cut: a bright flat field (mean 0.9) — DC alone blows the ratio.
    sceneB = torch.full((3, 32, 32), 0.9)
    out, st = eng.process_frame(sceneB, st)
    # On a hard reset the blend is skipped (output == input) and the anchor is
    # reseeded (n_seen back to 1).
    assert torch.equal(out, sceneB), "cut frame should pass through untouched"
    assert st.n_seen == 1, f"anchor not reseeded (n_seen={st.n_seen}, was {n_before})"


def test_no_false_cut_on_smooth_drift():
    eng = SpectralCoherenceEngine(alpha=0.5, cut_thresh=0.35)
    st = eng.init_state(32, 32)
    base = torch.rand(3, 32, 32) * 0.3 + 0.3
    _, st = eng.process_frame(base, st)
    # A tiny exposure nudge must NOT read as a cut -> anchor keeps accumulating.
    _, st = eng.process_frame(base + 0.02, st)
    assert st.n_seen == 2, "smooth drift wrongly flagged as a cut"


# --- state_bytes matches the spec headline numbers ------------------------- #
def test_state_bytes_spec_values():
    eng = SpectralCoherenceEngine(cutoff_frac=0.10, anchor_mode="magnitude")
    b512 = eng.state_bytes(512, 512)
    assert b512 == 3 * 51 * 51 * 8 == 62424, b512      # ~62 KB
    b256 = eng.state_bytes(256, 256)
    assert b256 == 3 * 26 * 26 * 8 == 16224, b256      # ~15 KB

    dc = SpectralCoherenceEngine(anchor_mode="dc_only")
    assert dc.state_bytes(512, 512) == 24, dc.state_bytes(512, 512)  # 24 bytes
    assert dc.state_bytes(256, 256) == 24

    stats = StatsEMAEngine()
    assert stats.state_bytes(512, 512) == 24


# --- target_lowband keeps phase in magnitude mode -------------------------- #
def test_magnitude_preserves_phase():
    x = torch.rand(3, 16, 16)
    X = torch.fft.rfft2(x)
    Xlo = low_band_box(X, 0.25)
    anchor = torch.fft.rfft2(torch.rand(3, 16, 16))
    anchor_lo = low_band_box(anchor, 0.25)
    tgt = target_lowband(Xlo, anchor_lo, "magnitude")
    # phase kept from Xlo, magnitude taken from anchor (dividing by a positive
    # real leaves the angle exact; magnitude matches to the eps guard).
    assert torch.allclose(torch.angle(tgt), torch.angle(Xlo), atol=1e-4)
    assert torch.allclose(tgt.abs(), anchor_lo.abs(), atol=1e-3, rtol=1e-3)


def test_state_bytes_actual_tensor_matches():
    # The claimed byte count equals the real anchor tensor size.
    eng = SpectralCoherenceEngine(cutoff_frac=0.10)
    st = eng.init_state(64, 64)
    real = st.anchor.numel() * st.anchor.element_size()
    assert real == eng.state_bytes(64, 64)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
