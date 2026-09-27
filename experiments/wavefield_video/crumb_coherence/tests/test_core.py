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
    phase_band_weight, estimate_lowband_shift, apply_lowband_shift,
    radial_band_mask, estimate_lowband_shift_phaseplane,
)
from crumb_coherence.core import _box_freqs, _smoothstep   # noqa: E402

torch.manual_seed(0)


def _clip(T=6, H=32, W=32):
    return torch.rand(T, 3, H, W)


# --- invariant (i): alpha=0 is a byte-identical no-op ---------------------- #
def test_alpha0_byte_identical_spectral():
    for mode in ("magnitude", "dc_only", "complex", "complex_mc"):
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
                      (SpectralCoherenceEngine, "complex"),
                      (SpectralCoherenceEngine, "complex_mc")]:
        kw = {"mc_strength": 0.8} if mode == "complex_mc" else {}
        eng = Eng(alpha=0.5, anchor_mode=mode, **kw)
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


# --- invariant (M0.2): phase_anchor default is a byte-identical no-op, and it
#     only ever affects the complex mode ------------------------------------- #
def test_phase_anchor_default_is_noop():
    # phase_anchor defaults to 0.0 => identical output to explicitly passing 0.0,
    # for every mode: the shipped behaviour is untouched.
    x = _clip()
    for mode in ("magnitude", "dc_only", "complex"):
        base = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode)
        expl = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode,
                                       phase_anchor=0.0)
        yb, _ = base.process_segment(x, base.init_state(32, 32))
        ye, _ = expl.process_segment(x, expl.init_state(32, 32))
        assert torch.equal(yb, ye), f"phase_anchor=0 not a no-op for {mode}"


def test_phase_anchor_only_affects_complex():
    # In magnitude / dc_only the flat phase weight must never be applied — those
    # modes are defined to leave phase free / touch only DC.
    x = _clip()
    for mode in ("magnitude", "dc_only"):
        off = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode,
                                      phase_anchor=0.0)
        on = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode,
                                     phase_anchor=1.0)
        yoff, _ = off.process_segment(x, off.init_state(32, 32))
        yon, _ = on.process_segment(x, on.init_state(32, 32))
        assert torch.equal(yoff, yon), f"phase_anchor leaked into {mode}"
    # In complex mode it MUST change the result (it is the whole point).
    off = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex",
                                  phase_anchor=0.0)
    on = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex",
                                 phase_anchor=1.0)
    yoff, _ = off.process_segment(x, off.init_state(32, 32))
    yon, _ = on.process_segment(x, on.init_state(32, 32))
    assert not torch.equal(yoff, yon), "phase_anchor had no effect in complex"


def test_phase_band_weight_shape_and_dc():
    # Flat everywhere, zero at the DC cell (box row kh//2, col 0), so it anchors
    # the position-carrying cells without re-locking the exposure DC.
    wp = phase_band_weight(7, 5)
    assert wp.shape == (7, 5)
    assert wp[7 // 2, 0].item() == 0.0
    others = wp.clone()
    others[7 // 2, 0] = 1.0
    assert torch.all(others == 1.0), "phase weight not flat off-DC"
    # negative phase_anchor is rejected at construction
    try:
        SpectralCoherenceEngine(phase_anchor=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative phase_anchor should raise")


# --- invariant (M0.3): complex_mc with mc_strength=0 == complex ------------- #
def test_mc_strength_zero_equals_complex():
    # mc_strength=0 must be byte-identical to plain complex: the sweep anchor and
    # the do-no-harm guarantee. (New engine name, old behaviour when off.)
    x = _clip()
    cx = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex")
    mc = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                 mc_strength=0.0)
    yc, _ = cx.process_segment(x, cx.init_state(32, 32))
    ym, _ = mc.process_segment(x, mc.init_state(32, 32))
    assert torch.equal(yc, ym), "complex_mc(mc=0) diverged from complex"


def test_mc_strength_only_affects_complex_mc():
    # A non-zero mc_strength must never leak into the other modes.
    x = _clip()
    for mode in ("magnitude", "dc_only", "complex"):
        off = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode)
        on = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode,
                                     mc_strength=1.0)
        yoff, _ = off.process_segment(x, off.init_state(32, 32))
        yon, _ = on.process_segment(x, on.init_state(32, 32))
        assert torch.equal(yoff, yon), f"mc_strength leaked into {mode}"
    # In complex_mc it MUST change the result (it is the whole point).
    off = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                  mc_strength=0.0)
    on = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                 mc_strength=1.0)
    yoff, _ = off.process_segment(x, off.init_state(32, 32))
    yon, _ = on.process_segment(x, on.init_state(32, 32))
    assert not torch.equal(yoff, yon), "mc_strength had no effect in complex_mc"
    # negative mc_strength is rejected at construction
    try:
        SpectralCoherenceEngine(mc_strength=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative mc_strength should raise")


def test_estimate_lowband_shift_recovers_known_shift():
    # A circular roll is an exact spectral shift; phase correlation on the low
    # band must recover it to sub-pixel accuracy.
    H = W = 64
    ys = torch.arange(H).view(H, 1).float()
    xs = torch.arange(W).view(1, W).float()
    blob = torch.exp(-(((ys - 30) ** 2 + (xs - 34) ** 2) / (2 * 9.0 ** 2)))
    base = blob.unsqueeze(0).expand(3, H, W).contiguous()   # [3,H,W]
    sy, sx = 3, -2
    shifted = torch.roll(base, shifts=(sy, sx), dims=(-2, -1))
    a_lo = low_band_box(torch.fft.rfft2(base), 0.25)
    x_lo = low_band_box(torch.fft.rfft2(shifted), 0.25)
    dy, dx = estimate_lowband_shift(x_lo, a_lo, H, W)
    assert abs(dy - sy) < 0.5, f"dy={dy} != {sy}"
    assert abs(dx - sx) < 0.5, f"dx={dx} != {sx}"
    # Empty anchor -> no shift (guard against the cold-start case).
    assert estimate_lowband_shift(x_lo, torch.zeros_like(a_lo), H, W) == (0.0, 0.0)


def test_apply_lowband_shift_is_exact_and_magnitude_preserving():
    # apply_lowband_shift(dy,dx) must equal the low band of a real circular roll
    # (integer shift) and must never change per-cell magnitude (unit ramp).
    H = W = 64
    ys = torch.arange(H).view(H, 1).float()
    xs = torch.arange(W).view(1, W).float()
    blob = torch.exp(-(((ys - 28) ** 2 + (xs - 36) ** 2) / (2 * 8.0 ** 2)))
    base = blob.unsqueeze(0).expand(3, H, W).contiguous()
    sy, sx = 4, 3
    base_lo = low_band_box(torch.fft.rfft2(base), 0.25)
    rolled_lo = low_band_box(torch.fft.rfft2(
        torch.roll(base, shifts=(sy, sx), dims=(-2, -1))), 0.25)
    applied = apply_lowband_shift(base_lo, sy, sx, H, W)
    assert torch.allclose(applied, rolled_lo, atol=1e-4), "shift != real roll"
    assert torch.allclose(applied.abs(), base_lo.abs(), atol=1e-5), \
        "shift changed magnitude"
    # dy=dx=0 is an exact identity (fast path)
    assert torch.equal(apply_lowband_shift(base_lo, 0.0, 0.0, H, W), base_lo)


# --- invariant (M0.4): graded (edge-tapered) shift is unit-modulus and grades - #
def test_apply_lowband_shift_taper_is_unit_modulus_and_grades():
    # The M0.4 taper scales the PHASE per cell (not the amplitude), so every cell
    # stays unit-modulus (exactly magnitude-preserving); where taper==0 it is an
    # exact identity (band edge the SSIM high-pass overlaps is protected), where
    # taper==1 it equals the rigid full shift.
    H = W = 64
    ys = torch.arange(H).view(H, 1).float()
    xs = torch.arange(W).view(1, W).float()
    blob = torch.exp(-(((ys - 30) ** 2 + (xs - 33) ** 2) / (2 * 8.0 ** 2)))
    base = blob.unsqueeze(0).expand(3, H, W).contiguous()
    base_lo = low_band_box(torch.fft.rfft2(base), 0.25)
    kh, kw = base_lo.shape[-2], base_lo.shape[-1]
    sy, sx = 3.0, -2.0
    taper = torch.ones(kh, kw)
    taper[0, :] = 0.0                                   # protect the top edge row
    rigid = apply_lowband_shift(base_lo, sy, sx, H, W)             # taper=None
    graded = apply_lowband_shift(base_lo, sy, sx, H, W, taper=taper)
    # unit-modulus: per-cell magnitude is exactly preserved by the graded ramp.
    assert torch.allclose(graded.abs(), base_lo.abs(), atol=1e-5), \
        "graded shift changed magnitude"
    # taper==0 row is an exact identity (edge protected).
    assert torch.allclose(graded[:, 0, :], base_lo[:, 0, :], atol=1e-6), \
        "taper==0 cells were still shifted"
    # taper==1 rows equal the rigid shift.
    assert torch.allclose(graded[:, 1:, :], rigid[:, 1:, :], atol=1e-6), \
        "taper==1 cells diverged from the rigid shift"
    # and the graded shift is not the rigid one overall (the edge was protected).
    assert not torch.allclose(graded, rigid, atol=1e-6), \
        "taper had no effect"


# --- invariant (M0.4): zero-shift is byte-identical to the non-mc complex path - #
def test_complex_mc_zero_shift_is_complex_identity():
    # A drift-free (static) stream: the bump never wanders, so the estimated shift
    # is zero and complex_mc must be byte-identical to the non-mc `complex` path —
    # the explicit zero-shift-identity invariant, at full mc_strength.
    frame = torch.rand(3, 32, 32)
    clip = frame.unsqueeze(0).expand(6, 3, 32, 32).contiguous()
    cx = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex")
    mc = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                 mc_strength=1.0)
    yc, _ = cx.process_segment(clip, cx.init_state(32, 32))
    ym, _ = mc.process_segment(clip, mc.init_state(32, 32))
    assert torch.equal(yc, ym), "complex_mc not identity on a zero-shift stream"


# --- invariant (M0.4): the deadband snaps a sub-threshold shift to a no-op ---- #
def test_mc_deadband_snaps_to_identity():
    # A huge deadband snaps every shift to an exact no-op, so complex_mc == complex
    # even on a wandering stream; a zero deadband lets the shift through so they
    # differ. This is the mechanism that makes ~zero-wander scenarios do no harm.
    x = _clip()
    cx = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex")
    dead = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                   mc_strength=1.0, mc_deadband=1e6)
    live = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                   mc_strength=1.0, mc_deadband=0.0)
    yc, _ = cx.process_segment(x, cx.init_state(32, 32))
    yd, _ = dead.process_segment(x, dead.init_state(32, 32))
    yl, _ = live.process_segment(x, live.init_state(32, 32))
    assert torch.equal(yc, yd), "large deadband was not a no-op (== complex)"
    assert not torch.equal(yc, yl), "zero deadband did not let the shift through"
    # negative mc_deadband is rejected at construction
    try:
        SpectralCoherenceEngine(mc_deadband=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative mc_deadband should raise")


# --- invariant (M0.6): radial band mask is unity inside, zero outside, radially
#     symmetric, and unit-modulus-safe (it multiplies phase, not amplitude) ---- #
def test_radial_band_mask_hard_inside_outside_and_symmetry():
    # A hard mask must be exactly 1.0 on every box cell with radial freq
    # r < r_inner and exactly 0.0 elsewhere, and depend ONLY on r (radial
    # symmetry: cells at +/- the same ky, same kx are equal).
    H = W = 64
    kh, kw = 12, 7
    r_inner = 0.06
    m = radial_band_mask(kh, kw, H, W, r_inner, edge="hard")
    assert m.shape == (kh, kw)
    ky, kx = _box_freqs(kh, kw, H, W)
    r = torch.sqrt(ky * ky + kx * kx)                    # [kh,kw]
    assert torch.all(m[r < r_inner] == 1.0), "hard mask not unity inside"
    assert torch.all(m[r >= r_inner] == 0.0), "hard mask not zero outside"
    # every value is a clean 0 or 1 (a mask, no partial cells for a hard cut)
    assert torch.all((m == 0.0) | (m == 1.0)), "hard mask has partial cells"
    # radial symmetry across the DC row (box row kh//2): +j and -j rows match.
    dc = kh // 2
    for j in range(1, min(dc, kh - dc - 1) + 1):
        assert torch.equal(m[dc + j], m[dc - j]), f"row {j} not radially symmetric"
    # DC cell (r=0) is always inside; the far corner is always outside.
    assert m[dc, 0].item() == 1.0
    assert m[0, kw - 1].item() == 0.0 or r[0, kw - 1].item() < r_inner


def test_radial_band_mask_flat_top_grades_and_bounds():
    # The flat-top edge is unity in the interior, 0 well outside, and takes
    # intermediate values only in the raised-cosine transition centered on
    # r_inner. Everything stays within [0,1].
    H = W = 64
    kh, kw = 16, 9
    r_inner = 0.08
    m = radial_band_mask(kh, kw, H, W, r_inner, edge="flat_top", width_frac=0.3)
    assert torch.all((m >= 0.0) & (m <= 1.0)), "flat_top mask out of [0,1]"
    ky, kx = _box_freqs(kh, kw, H, W)
    r = torch.sqrt(ky * ky + kx * kx)
    w = 0.3 * r_inner
    r1, r2 = r_inner - 0.5 * w, r_inner + 0.5 * w
    assert torch.all(m[r < r1] == 1.0), "flat_top not unity below transition"
    assert torch.all(m[r >= r2] == 0.0), "flat_top not zero above transition"
    dc = kh // 2
    assert m[dc, 0].item() == 1.0, "flat_top DC not unity"
    # radial symmetry holds for the smooth edge too.
    for j in range(1, min(dc, kh - dc - 1) + 1):
        assert torch.allclose(m[dc + j], m[dc - j]), f"row {j} not symmetric"
    # unknown edge is rejected.
    try:
        radial_band_mask(kh, kw, H, W, r_inner, edge="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown edge should raise")


# --- invariant (M0.6): each new complex_mc kwarg defaults to a byte-identical
#     no-op, and only ever affects complex_mc ---------------------------------- #
def test_mc_band_default_is_noop_and_only_complex_mc():
    # mc_band defaults to 0.0 => identical output to explicitly passing 0.0 in
    # every mode, and a non-zero mc_band never leaks into a non-complex_mc mode.
    x = _clip()
    for mode in ("magnitude", "dc_only", "complex", "complex_mc"):
        kw = {"mc_strength": 0.8} if mode == "complex_mc" else {}
        base = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode, **kw)
        expl = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode,
                                       mc_band=0.0, **kw)
        yb, _ = base.process_segment(x, base.init_state(32, 32))
        ye, _ = expl.process_segment(x, expl.init_state(32, 32))
        assert torch.equal(yb, ye), f"mc_band=0 not a no-op for {mode}"
    for mode in ("magnitude", "dc_only", "complex"):
        off = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode, mc_band=0.0)
        on = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode, mc_band=0.05)
        yoff, _ = off.process_segment(x, off.init_state(32, 32))
        yon, _ = on.process_segment(x, on.init_state(32, 32))
        assert torch.equal(yoff, yon), f"mc_band leaked into {mode}"
    # In complex_mc with a real shift it MUST change the result vs the full-band
    # (mc_band=0) path — it is restricting the correction to the inner band.
    off = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                  mc_strength=1.0, mc_taper=False, mc_band=0.0,
                                  mc_deadband=0.0)
    on = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                 mc_strength=1.0, mc_taper=False, mc_band=0.05,
                                 mc_deadband=0.0)
    yoff, _ = off.process_segment(x, off.init_state(32, 32))
    yon, _ = on.process_segment(x, on.init_state(32, 32))
    assert not torch.equal(yoff, yon), "mc_band had no effect in complex_mc"
    # negative mc_band is rejected at construction.
    try:
        SpectralCoherenceEngine(mc_band=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative mc_band should raise")


def test_mc_est_default_is_noop():
    # mc_est defaults to "corr" => identical to explicitly passing it; and the
    # "phase_plane" alternative changes complex_mc (a different estimator) while
    # never touching the other modes (which ignore mc_est entirely).
    x = _clip()
    base = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                   mc_strength=0.8, mc_deadband=0.0)
    expl = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                   mc_strength=0.8, mc_est="corr", mc_deadband=0.0)
    yb, _ = base.process_segment(x, base.init_state(32, 32))
    ye, _ = expl.process_segment(x, expl.init_state(32, 32))
    assert torch.equal(yb, ye), "mc_est='corr' is not the default behaviour"
    pp = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                 mc_strength=0.8, mc_est="phase_plane",
                                 mc_deadband=0.0)
    yp, _ = pp.process_segment(x, pp.init_state(32, 32))
    assert not torch.equal(yb, yp), "phase_plane estimator had no effect"
    try:
        SpectralCoherenceEngine(mc_est="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown mc_est should raise")


def test_mc_smooth_default_is_noop():
    # mc_smooth defaults to False => byte-identical to explicitly passing False;
    # turning it on runs the alpha-beta filter and changes complex_mc output.
    x = _clip()
    base = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                   mc_strength=0.8, mc_deadband=0.0)
    expl = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                   mc_strength=0.8, mc_smooth=False,
                                   mc_deadband=0.0)
    yb, _ = base.process_segment(x, base.init_state(32, 32))
    ye, _ = expl.process_segment(x, expl.init_state(32, 32))
    assert torch.equal(yb, ye), "mc_smooth=False is not the default behaviour"
    sm = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                 mc_strength=0.8, mc_smooth=True, mc_deadband=0.0)
    ys, _ = sm.process_segment(x, sm.init_state(32, 32))
    assert not torch.equal(yb, ys), "mc_smooth had no effect"


def test_mc_edge_default_is_noop_given_band():
    # mc_edge only matters when mc_band>0; "hard" is the default and "flat_top"
    # changes the correction. With mc_band==0 the edge is irrelevant (no-op).
    x = _clip()
    hard = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                   mc_strength=1.0, mc_band=0.06, mc_deadband=0.0)
    hard2 = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                    mc_strength=1.0, mc_band=0.06, mc_edge="hard",
                                    mc_deadband=0.0)
    yh, _ = hard.process_segment(x, hard.init_state(32, 32))
    yh2, _ = hard2.process_segment(x, hard2.init_state(32, 32))
    assert torch.equal(yh, yh2), "mc_edge='hard' is not the default"
    ft = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                 mc_strength=1.0, mc_band=0.06, mc_edge="flat_top",
                                 mc_deadband=0.0)
    yf, _ = ft.process_segment(x, ft.init_state(32, 32))
    assert not torch.equal(yh, yf), "flat_top edge had no effect"
    # mc_edge is irrelevant when mc_band==0 (no band mask is built).
    e0h = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                  mc_strength=1.0, mc_band=0.0, mc_edge="hard")
    e0f = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                  mc_strength=1.0, mc_band=0.0, mc_edge="flat_top")
    y0h, _ = e0h.process_segment(x, e0h.init_state(32, 32))
    y0f, _ = e0f.process_segment(x, e0f.init_state(32, 32))
    assert torch.equal(y0h, y0f), "mc_edge leaked with mc_band==0"
    try:
        SpectralCoherenceEngine(mc_edge="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown mc_edge should raise")


def test_phaseplane_recovers_known_shift():
    # The weighted phase-plane fit must recover a known small circular-roll
    # translation to sub-pixel accuracy, matching estimate_lowband_shift's sign.
    H = W = 64
    ys = torch.arange(H).view(H, 1).float()
    xs = torch.arange(W).view(1, W).float()
    blob = torch.exp(-(((ys - 32) ** 2 + (xs - 30) ** 2) / (2 * 9.0 ** 2)))
    base = blob.unsqueeze(0).expand(3, H, W).contiguous()
    sy, sx = 2, -1
    shifted = torch.roll(base, shifts=(sy, sx), dims=(-2, -1))
    a_lo = low_band_box(torch.fft.rfft2(base), 0.25)
    x_lo = low_band_box(torch.fft.rfft2(shifted), 0.25)
    dy, dx = estimate_lowband_shift_phaseplane(x_lo, a_lo, H, W)
    assert abs(dy - sy) < 0.5, f"phase-plane dy={dy} != {sy}"
    assert abs(dx - sx) < 0.5, f"phase-plane dx={dx} != {sx}"
    # empty anchor -> no shift (cold-start guard).
    assert estimate_lowband_shift_phaseplane(
        x_lo, torch.zeros_like(a_lo), H, W) == (0.0, 0.0)


# --- invariant (M3): the velocity-coherence gate. ------------------------- #
def test_smoothstep_bounds_and_monotone():
    # 0 below lo, 1 above hi, monotone non-decreasing in between, all in [0,1].
    assert _smoothstep(0.3, 0.7, 0.2) == 0.0
    assert _smoothstep(0.3, 0.7, 0.8) == 1.0
    assert _smoothstep(0.3, 0.7, 0.5) == 0.5           # symmetric midpoint
    prev = -1.0
    for i in range(21):
        x = i / 20.0
        v = _smoothstep(0.3, 0.7, x)
        assert 0.0 <= v <= 1.0
        assert v >= prev - 1e-9, "smoothstep not monotone"
        prev = v
    # degenerate hi<=lo is a hard step at hi
    assert _smoothstep(0.5, 0.5, 0.49) == 0.0
    assert _smoothstep(0.5, 0.5, 0.5) == 1.0


def test_mc_gate_default_is_noop_and_only_complex_mc():
    # mc_gate defaults False => byte-identical to explicitly passing False in
    # every mode; and a True gate never leaks into a non-complex_mc mode.
    x = _clip(T=10)
    for mode in ("magnitude", "dc_only", "complex", "complex_mc"):
        kw = {"mc_strength": 0.8} if mode == "complex_mc" else {}
        base = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode, **kw)
        expl = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode,
                                       mc_gate=False, **kw)
        yb, _ = base.process_segment(x, base.init_state(32, 32))
        ye, _ = expl.process_segment(x, expl.init_state(32, 32))
        assert torch.equal(yb, ye), f"mc_gate=False not a no-op for {mode}"
    for mode in ("magnitude", "dc_only", "complex"):
        off = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode, mc_gate=False)
        on = SpectralCoherenceEngine(alpha=0.9, anchor_mode=mode, mc_gate=True)
        yoff, _ = off.process_segment(x, off.init_state(32, 32))
        yon, _ = on.process_segment(x, on.init_state(32, 32))
        assert torch.equal(yoff, yon), f"mc_gate leaked into {mode}"
    # bad gate params are rejected at construction.
    for bad in (dict(mc_gate_decay=1.0), dict(mc_gate_decay=-0.1),
                dict(mc_gate_lo=-0.1), dict(mc_gate_lo=0.8, mc_gate_hi=0.5)):
        try:
            SpectralCoherenceEngine(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad gate params {bad} should raise")


def _translating_blob_clip(T, H=64, W=64, vx=0.6, vy=0.0, sigma=8.0, x0=None):
    """A single soft Gaussian on a straight constant-velocity path -> a purely
    COHERENT low-band motion (the thing the gate must leave alone)."""
    ys = torch.arange(H).view(1, H, 1).float()
    xs = torch.arange(W).view(1, 1, W).float()
    t = torch.arange(T).float()
    cx = (0.30 * W if x0 is None else x0) + vx * t
    cy = 0.5 * H + vy * t
    blob = torch.exp(-(((xs - cx.view(T, 1, 1)) ** 2)
                       + ((ys - cy.view(T, 1, 1)) ** 2)) / (2 * sigma ** 2))
    return (0.08 + 0.7 * blob).unsqueeze(1).expand(T, 3, H, W).contiguous().clamp(0, 1)


def _oscillating_blob_clip(T, H=64, W=64, amp=10.0, cycles=2.0, sigma=8.0):
    """A single soft Gaussian oscillating about a center -> INCOHERENT wander
    (net displacement stays bounded while path length grows: what the gate must
    keep correcting)."""
    import math
    ys = torch.arange(H).view(1, H, 1).float()
    xs = torch.arange(W).view(1, 1, W).float()
    t = torch.arange(T).float()
    cx = 0.5 * W + amp * torch.sin(2 * math.pi * cycles * t / T)
    cy = torch.full((T,), 0.5 * H)
    blob = torch.exp(-(((xs - cx.view(T, 1, 1)) ** 2)
                       + ((ys - cy.view(T, 1, 1)) ** 2)) / (2 * sigma ** 2))
    return (0.08 + 0.7 * blob).unsqueeze(1).expand(T, 3, H, W).contiguous().clamp(0, 1)


def _dev(a, b):
    return float((a - b).abs().mean().item())


def test_gate_preserves_coherent_translation():
    # On a purely translating blob the gate must recognise coherent motion
    # (g->0) and leave it far closer to the input than the ungated engine does.
    T = 48
    clip = _translating_blob_clip(T)
    cfg = dict(alpha=0.95, rho=0.995, cutoff_frac=0.14, anchor_mode="complex_mc",
               mc_strength=0.5, mc_band=0.03)
    ungated = SpectralCoherenceEngine(**cfg, mc_gate=False)
    gated = SpectralCoherenceEngine(**cfg, mc_gate=True)
    yu, _ = ungated.process_segment(clip, ungated.init_state(64, 64))
    yg, su = gated.process_segment(clip, gated.init_state(64, 64))
    dev_ungated = _dev(clip, yu)
    dev_gated = _dev(clip, yg)
    # the gate must cut the damage to coherent motion by a large margin.
    assert dev_gated < 0.5 * dev_ungated, (
        f"gate did not spare coherent motion: dev_gated={dev_gated:.5f} "
        f"vs dev_ungated={dev_ungated:.5f}")
    # and it must have actually detected coherence (settled gate near 0).
    assert su.gate_g < 0.5, f"gate did not open on coherent motion (g={su.gate_g})"


def test_gate_still_corrects_wander():
    # On an oscillating (returning) blob the gate must STAY engaged (g->1) so the
    # drift is still removed: gated output should track the ungated one, and both
    # should differ clearly from the raw input.
    T = 64
    clip = _oscillating_blob_clip(T)
    cfg = dict(alpha=0.95, rho=0.995, cutoff_frac=0.14, anchor_mode="complex_mc",
               mc_strength=0.5, mc_band=0.03)
    ungated = SpectralCoherenceEngine(**cfg, mc_gate=False)
    gated = SpectralCoherenceEngine(**cfg, mc_gate=True)
    yu, _ = ungated.process_segment(clip, ungated.init_state(64, 64))
    yg, sg = gated.process_segment(clip, gated.init_state(64, 64))
    dev_input = _dev(clip, yg)          # gate still changed the wander
    dev_vs_ungated = _dev(yg, yu)       # and stayed close to the ungated result
    assert dev_input > 1e-3, "gate wrongly passed wander through untouched"
    assert dev_vs_ungated < dev_input, (
        "gated wander drifted away from the ungated correction "
        f"(vs_ungated={dev_vs_ungated:.5f}, vs_input={dev_input:.5f})")
    assert sg.gate_g > 0.5, f"gate opened on wander (g={sg.gate_g}); should stay engaged"


def test_gate_reset_on_cut_clears_velocity_history():
    # A hard cut must drop the gate's velocity integrators so the new scene
    # starts from correction (g=1) and re-learns coherence from scratch.
    eng = SpectralCoherenceEngine(alpha=0.9, anchor_mode="complex_mc",
                                  mc_strength=0.5, mc_band=0.03, mc_gate=True,
                                  reset_on_cut=True, cut_thresh=0.35)
    st = eng.init_state(64, 64)
    clip = _translating_blob_clip(20)
    for t in range(clip.shape[0]):
        _, st = eng.process_frame(clip[t], st)
    assert st.gate_init, "gate never warmed on the moving scene"
    sceneB = torch.full((3, 64, 64), 0.9)                # hard cut -> reset
    _, st = eng.process_frame(sceneB, st)
    assert st.gate_init is False, "cut did not reset the gate history"
    assert st.gate_g == 1.0 and st.gate_sp == 0.0, "gate not cleared on cut"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
