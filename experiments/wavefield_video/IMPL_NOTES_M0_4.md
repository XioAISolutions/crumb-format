# IMPL_NOTES M0.4 — lossless k-space phase-ramp re-centering (2026-09-26)

## Task recap

M0.3 validated the direction (hotspot pos-drift removal 70.2% at mc=0.5, clearing
the D>=60 gate) but hf_ssim collapsed to ~0.93 at every mc>0 (S gate: >=0.98), and
gain_field's complex_mc showed 0.9678 despite ~zero wander. The M0.4 theory: the
re-centering is a lossy spatial resample; replace it with an exact unitary k-space
phase-ramp shift, make zero-shift a byte-exact identity, and stop grading the
hotspot-specific correction in gain_field.

## Finding #1 — the "interpolation" premise is refuted by inspection

`apply_lowband_shift` was ALREADY an exact k-space phase ramp in M0.3:
`Xlo * exp(-i·2π·(ky·dy + kx·dx))`. It never interpolated and never touched
per-cell magnitude (unit-modulus multiply). So the hf_ssim collapse is NOT a
spatial-warp artifact — the theory's stated cause was wrong. The real cause is a
band-overlap coupling between the edited band and the metric:

- `highfreq_ssim` measures `luma - blur(luma)` at `sigma=1.5`. That high-pass
  still retains ~16% of energy at the low-band box's HEIGHT edge (0.0625 cyc/px)
  and ~50% at its WIDTH edge (0.125 cyc/px on the 64-grid, cutoff 0.14). So the
  box EDGE cells are heavily inside the SSIM gate's measurement band.
- Plain `complex` protects those edge cells: the Gaussian blend weight `aw≈0`
  there, so the blend barely touches them (that is why `complex` keeps 0.9957).
- The M0.3 rigid ramp rotated EVERY box cell at full strength, with the LARGEST
  phase rotation at the highest-k edge — exactly the detail-carrying cells the
  gate watches. That is the 0.9957 -> ~0.93 drop.
- The hotspot bump lives at very low k (spatial sigma 8px -> spectral sigma
  ~0.02 cyc/px ~ 2 cells), so re-centering only needs the innermost cells.
  Rotating the edge cells buys ~zero drift correction and costs all the HF.

Conclusion: the fix is not "make the shift more unitary" (it already is) but
"apply the unitary shift only where the position lives, and leave the
detail-carrying / metric-overlapping band edge as an exact identity."

## What changed (crumb_coherence/ only; public API stable)

`core.py`

- `apply_lowband_shift(Xlo, dy, dx, H, W, taper=None)` — new optional `taper`
  ([kh,kw] in [0,1]). It scales the PHASE per cell, `exp(-i·2π·taper·(...))`, NOT
  the amplitude, so every cell stays unit-modulus — the shift is still exactly
  unitary and magnitude-preserving. `taper=1` at low k / `->0` at the band edge
  gives a GRADED shift: full translation on the bump cells, exact identity on the
  detail edge. `taper=None` reproduces the exact rigid M0.3 ramp (so the existing
  exact-round-trip invariant is unchanged).
- `SpectralCoherenceEngine` gains two knobs (defaults chosen so `complex`/
  `magnitude`/`dc_only` remain byte-identical):
  - `mc_taper: bool = True` — apply the graded shift weighted by the Gaussian
    band weight ("re-center only what we anchor"); `False` = the M0.3 rigid ramp.
  - `mc_deadband: float = 1e-2` — a shift below this many pixels (after the
    mc_strength scale) is snapped to an EXACT no-op, so estimator noise at ~zero
    wander yields output byte-identical to the non-mc `complex` path. Negative
    values are rejected at construction.
- `process_frame` complex_mc branch: estimate the shift, scale by `-mc_strength`,
  snap to identity below `mc_deadband`, else apply the graded (or rigid) shift.

`scripts/run_m0.py`

- `complex_mc` is now a HOTSPOT-ONLY graded engine. It is no longer added to the
  gain_field table (Finding #2), so every pre-existing gain_field row is
  byte-identical. New flag `--hotspot-mc-taper / --no-hotspot-mc-taper`
  (default on) for the before/after hf_ssim comparison; threaded into the main
  hotspot row and the mc_strength sweep. All prior flags unchanged.

`tests/test_core.py` — 16 -> 19 invariants. New:

- `test_apply_lowband_shift_taper_is_unit_modulus_and_grades` — the graded ramp
  preserves per-cell magnitude exactly, is an exact identity where `taper==0`
  (edge protected), equals the rigid shift where `taper==1`, and differs from the
  rigid shift overall.
- `test_complex_mc_zero_shift_is_complex_identity` — on a drift-free (static)
  stream the estimated shift is zero, so complex_mc is byte-identical to `complex`
  at full mc_strength (the explicit zero-shift-identity invariant).
- `test_mc_deadband_snaps_to_identity` — a large deadband makes complex_mc a
  no-op (== complex) even on a wandering stream; a zero deadband lets the shift
  through (they differ); negative deadband raises.

## Finding #2 — gain_field must not grade a hotspot-specific correction

gain_field is magnitude drift with no positional wander; running the hotspot
re-centering there measures nothing but its artifacts. M0.4 removes complex_mc
from the gain_field graded table entirely (it must be "a no-op OR not listed as a
graded engine" — this is the second option). For extra safety the engine's own
`mc_deadband` would also snap gain_field's ~zero shift to an exact no-op if it
were run.

## Exact commands (owner-verifiable)

```bash
cd experiments/wavefield_video

# 1. Invariants (expect "All 19 tests passed."):
python crumb_coherence/tests/test_core.py

# 2. gain_field — regression guard. complex_mc is ABSENT; the four pre-existing
#    rows must be byte-identical to M0.3:
python crumb_coherence/scripts/run_m0.py --scenario gain_field --no-video

# 3. hotspot — main table + mc_strength sweep at mc=0.0/0.5/0.7 (graded shift):
python crumb_coherence/scripts/run_m0.py --scenario hotspot --no-video \
    --hotspot-mc-sweep 0.0,0.5,0.7

# 4. before/after HF ablation — same sweep with the M0.3 rigid ramp:
python crumb_coherence/scripts/run_m0.py --scenario hotspot --no-video \
    --no-hotspot-mc-taper --hotspot-mc-sweep 0.0,0.5,0.7
```

## RESULTS — PENDING OWNER RUN

> The `python` step is execution-gated in this session (the owner-run convention
> from IMPL_NOTES_M0_2/M0_3). Tables are left blank rather than fabricated. Run
> the commands above and paste the output; the predictions below are labeled as
> hypotheses to be confirmed or refuted, exactly as in M0.3.

### Invariants

```
[paste: python crumb_coherence/tests/test_core.py]
expected: "All 19 tests passed."
```

### gain_field (regression guard — complex_mc ABSENT; other rows byte-identical)

| engine             | drift%(var) | drift%(ΔE) | hf_ssim | motion% | state | verdict |
| ------------------ | ----------- | ---------- | ------- | ------- | ----- | ------- |
| stats-EMA baseline |             |            |         |         | 24B   |         |
| spectral dc_only   |             |            |         |         | 24B   |         |
| spectral magnitude |             |            |         |         |       |         |
| spectral complex   |             |            |         |         |       |         |

(complex_mc intentionally not listed — Finding #2.)

### hotspot — main table (mc_strength = 0.5, mc_taper on)

| engine                  | drift%(pos) | \|ΔE\|corr | hf_ssim | motion% | state | verdict |
| ----------------------- | ----------- | ---------- | ------- | ------- | ----- | ------- |
| stats-EMA baseline      |             |            |         |         | 24B   |         |
| spectral dc_only        |             |            |         |         | 24B   |         |
| spectral magnitude      |             |            |         |         |       |         |
| spectral complex        |             |            |         |         |       |         |
| **spectral complex_mc** |             |            |         |         |       |         |

### hotspot — mc_strength sweep, GRADED shift (mc_taper on)

| mc_strength | drift%(pos) | hf_ssim | motion% | verdict (D>=60 & S>=0.98 & M>=97) |
| ----------- | ----------- | ------- | ------- | --------------------------------- |
| 0.00        |             |         |         | (== complex baseline)             |
| 0.50        |             |         |         | **target row**                    |
| 0.70        |             |         |         |                                   |

### hotspot — mc_strength sweep, RIGID shift (--no-hotspot-mc-taper, M0.3 ablation)

| mc_strength | drift%(pos) | hf_ssim | motion% | verdict                         |
| ----------- | ----------- | ------- | ------- | ------------------------------- |
| 0.00        |             |         |         |                                 |
| 0.50        |             | ~0.93   |         | (M0.3 reference: 70.2 / 0.9341) |
| 0.70        |             |         |         |                                 |

## Predicted shape (hypothesis, to be confirmed/refuted by the run)

- gain_field: the four pre-existing rows are byte-identical to M0.3 (no code path
  they touch changed); complex_mc is simply gone.
- hotspot GRADED (mc=0.5): hf_ssim should recover well above the ~0.93 rigid
  figure toward `complex`'s 0.9957 — the width edge (which leaks ~50% into the
  SSIM gate) is fully protected by the taper, and the height edge is ~88%
  protected. Whether it clears 0.98 exactly is the open question.
- hotspot GRADED drift%(pos): should stay near the rigid 70.2% because the bump's
  energy is in the innermost cells (taper ~0.5-1.0 there), which still get most of
  the shift; a modest drop is possible. The risk is the taper trimming drift below
  60 while lifting hf_ssim above 0.98 — the two gates trade off along the taper
  width, and the run decides whether both clear at mc=0.5.
- RIGID sweep should reproduce M0.3 (70.2 / 0.9341 at mc=0.5), isolating the
  taper as the sole cause of any hf_ssim change.

## Verdict: PENDING (fill after run).

If the graded shift does not clear D>=60 AND S>=0.98 AND M AND E together at
mc=0.5, say so plainly and record which gate is limiting and the taper-width
trade-off. The mechanism (band-overlap coupling, not interpolation) is
falsifiable via the RIGID-vs-GRADED ablation above.
