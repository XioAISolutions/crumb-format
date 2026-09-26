# IMPL_NOTES M0.5 — ghosting test + scale-selective phase correction (2026-09-26)

## Task recap

M0.4 refuted the "interpolation" theory and localized the hf_ssim collapse to a
**band-overlap coupling**: the low-band-only re-centering rotates the band-edge
cells the SSIM high-pass watches. The OWNER-VERIFIED frontier (IMPL_NOTES_M0_3.md)
left the verdict: _no measured setting clears both gates (drift≥60 AND hf_ref≥0.98)_;
`complex_mc 0.5 taper=False` reads **70.2% / 0.9346**, `complex` reads
**50.2% / 0.9946**.

M0.5 tests a **new owner hypothesis** for _why_ the HF dies, and whether a
different flavour of correction escapes it:

> complex_mc's HF damage = **BAND INCONSISTENCY**. It shifts the LOW band (to pin
> the wandering bump) while the HIGH band stays put, so every sharp edge gets a
> **double-image / ghost**. A _consistent_ correction should not do this.

- **P1** — applying the correction as a CONSISTENT broadband full-frame shift
  removes ghosting: hf_ref recovers ~0.99 while pos-drift removal stays ≥60,
  _IF_ the wander is translation-like on the corrected content.
- **P2** — metric floor: hf_ssim(img, exact_subpixel_shift(img, d)) is only ~0.9x
  even for a perfectly consistent shift (HF decorrelates under sub-pixel motion).
  Measure d = 0.5, 1.5, 3.0 px on the hotspot INPUT.

## What was added (no core.py change; no new deps; CPU; 19 invariants untouched)

`crumb_coherence/scripts/run_m05.py` — a self-contained experiment runner. It
imports the **exact** hotspot pipeline and metrics used for the frontier
(`data.make_clip_batch`, `run_m0.inject_hotspot / track_positions / motion_pct /
lowband_energy_positions / position_variance`, `metrics.highfreq_ssim`) and the
same hotspot params (grid 64, frames 48, seed 1234, alpha 0.95, rho 0.995,
cutoff_frac 0.14), so every number is directly comparable to M0.3/M0.4.

No file in `crumb_coherence/` (core/metrics/adapters/tests) was modified, so all
19 invariants remain byte-for-byte green and every shipped flag is unchanged. The
two new engine _variants_ (broadband MC, scale-selective MC) live only in the
experiment script, built from the same public primitives core uses
(`estimate_lowband_shift`, `apply_lowband_shift`, `put_low_band`,
`gaussian_band_weight`, `ema_update`), so they are faithful to `complex` /
`complex_mc` and add nothing to the shipping surface.

### The three experiment mechanisms (what the script actually does)

- **E1 metric floor** — `fullframe_shift_frames(x, 0, d)` = exact shift-theorem
  translation applied to EVERY spectral cell (a whole-frame move, no
  interpolation, no ghost by construction). Report `hf_ssim(drifted, shift(drifted,d))`
  (the task's "on the INPUT") and `hf_ssim(control, shift(control,d))` (a direct
  **upper bound on hf_ref** for any method that effectively translates by d).
- **E2(a) broadband** — `broadband_mc_segment`: same as `complex_mc` except the
  estimated `-mc·(dy,dx)` correction multiplies the FULL spectrum (all bands),
  then the complex low-band blend runs. High band is translated WITH the low band
  → no double-image. Cost: the sharp balls (all HF) get dragged by the same offset.
- **E2(b) lowband-only rigid** — the current engine, `SpectralCoherenceEngine(
anchor_mode="complex_mc", mc_taper=False)` (the M0.3 rigid ramp, hf_ref ~0.93).
- **E3 attribution** — `radial_energy`: |FFT|² of the injected bump vs the balls
  (control frame-diff) binned into radial bands `[0,.03) [.03,.06) [.06,.14) [.14,.5)`
  cyc/px. Shows the bump's dominant band vs the balls'.
- **E3 scale-selective** — `scaleselective_mc_segment`: `complex_mc` with the shift
  applied ONLY on a hard inner radial mask `r < r_inner` (low-band-only, HF
  untouched → balls/motion protected). Sweep `r_inner ∈ {0.03,0.06,0.10}` ×
  `mc ∈ {0.5,0.85,1.0}` → the frontier.

## Exact commands (owner-verifiable)

```bash
cd experiments/wavefield_video

# 0. invariants unchanged (expect "All 19 tests passed."):
python crumb_coherence/tests/test_core.py

# 1. M0.5 experiments E1 + E2 + E3 (single run, <3 min CPU):
python crumb_coherence/scripts/run_m05.py --json crumb_coherence/out/m05.json

# 2. (regression sanity — frontier rows must be unchanged by M0.5:)
python crumb_coherence/scripts/run_m0.py --scenario hotspot --no-video \
    --no-hotspot-mc-taper --hotspot-mc-sweep 0.0,0.5,0.85
python crumb_coherence/scripts/run_m0.py --scenario gain_field --no-video
```

## RESULTS — PENDING OWNER RUN

> The `python` step is execution-gated in this session (the owner-run convention
> from IMPL_NOTES_M0_2/M0_3/M0_4 — I retried three times and each was denied).
> Tables are left blank rather than fabricated. Run the commands above and paste
> the output (or approve a live run) and I will fill every cell and finalize the
> verdict, including "none passes" if that is the truth.

### E1 — metric floor `hf_ssim(img, exact_subpixel_shift(img, d))`

| d (px) | hf_ssim(drifted) | hf_ssim(control) |
| ------ | ---------------- | ---------------- |
| 0.5    |                  |                  |
| 1.5    |                  |                  |
| 3.0    |                  |                  |

`hf_ssim(control)` upper-bounds hf_ref for any translation-type correction of
magnitude ~d.

### E2 — mc variants (hotspot)

| variant                | mc   | drift%(pos) | hf_ref | hf_chg | motion% | gates |
| ---------------------- | ---- | ----------- | ------ | ------ | ------- | ----- |
| (a) broadband          | 0.50 |             |        |        |         |       |
| (b) lowband-only rigid | 0.50 |             |        |        |         |       |
| (a) broadband          | 0.85 |             |        |        |         |       |
| (b) lowband-only rigid | 0.85 |             |        |        |         |       |
| complex (champion ref) | –    |             |        |        |         |       |

### E3 — per-band energy attribution (bump vs balls)

| band (cyc/px)  | bump %E | balls %E |
| -------------- | ------- | -------- |
| inner [0,.03)  |         |          |
| mid1 [.03,.06) |         |          |
| mid2 [.06,.14) |         |          |
| outer [.14,.5) |         |          |

### E3 — scale-selective frontier (inner band only, HF untouched)

| r_inner | mc   | drift%(pos) | hf_ref | hf_chg | motion% | gates |
| ------- | ---- | ----------- | ------ | ------ | ------- | ----- |
| 0.03    | 0.50 |             |        |        |         |       |
| 0.03    | 0.85 |             |        |        |         |       |
| 0.03    | 1.00 |             |        |        |         |       |
| 0.06    | 0.50 |             |        |        |         |       |
| 0.06    | 0.85 |             |        |        |         |       |
| 0.06    | 1.00 |             |        |        |         |       |
| 0.10    | 0.50 |             |        |        |         |       |
| 0.10    | 0.85 |             |        |        |         |       |
| 0.10    | 1.00 |             |        |        |         |       |

## Predicted shape (HYPOTHESIS — to be confirmed/refuted by the run)

Grounded in the code, not a claim:

1. **E1 floor is below the S gate quickly (P2 → likely CONFIRMED).** The SSIM
   high-pass is `luma − blur(σ=1.5)`; a σ=1.5px blur has a spectral rolloff near
   ~0.1 cyc/px, so most sharp-ball energy survives the high-pass and is fully
   exposed to translation decorrelation. Expect `hf_ssim` to fall monotonically
   with d and to drop **below 0.98 by roughly d ≈ 0.5–1 px**. Consequence: **no
   translation-type correction can clear S≥0.98** unless its effective content
   displacement is well under a pixel. This is the calibration that reframes the
   whole gate.

2. **E2(a) broadband likely does NOT rescue hf_ref, and probably breaks motion
   (P1 → likely FALSIFIED as stated).** The bump wanders ±H/4 ≈ ±16 px, so the
   instantaneous vs-anchor offset is several px; `mc·offset` at mc=0.5 is a
   multi-pixel WHOLE-FRAME translation. That drags the sharp balls off their true
   positions → (i) hf_ref bounded by the E1 floor at that d (≈0.85–0.92, **not
   ~0.99**), and (ii) the tracker's balls fall outside `MATCH_RADIUS_MULT·radius`,
   so **motion% collapses (n/a or ≪95)**. The low-band-only edit was silently
   _protecting_ the balls (HF untouched) — the ghost is the price of that
   protection; you cannot remove the ghost by moving everything without paying
   translation-floor HF + a motion break.

3. **The real E2 payload — ghost vs pure-translation, via hf_chg.** A ghost is
   _extra_ decorrelation on top of translation. Prediction: broadband's
   `hf_chg(drifted, ·)` ≈ the E1 floor at its effective d (single translated
   image), while lowband-only's `hf_chg` sits **below** that floor at the same d
   (double image). If so, the **band-inconsistency ghosting mechanism is
   CONFIRMED** even though neither variant clears S — the ghost is real, but
   fixing it by consistency costs the balls.

4. **E3 attribution: bump dominant in `inner[0,.03)`, balls dominant in
   `outer[.14,.5)`.** The bump is a σ=8px Gaussian (spectral σ≈0.02 cyc/px); the
   balls are σ=1.6px (broadband, high-k). This separation is the enabling fact:
   correcting only the inner band moves the bump without touching where the balls
   live _and_ stays out of the SSIM high-pass's sensitive region (which ramps up
   past ~0.06–0.1 cyc/px).

5. **E3 scale-selective is the best-hope variant.** Restricting the shift to
   `r_inner=0.03` should lift hf_ref toward `complex`'s 0.9946 (the SSIM-overlapping
   band edge is left exact) while still removing the inner-band bump wander
   (drift). Motion stays ~100 (HF untouched). Whether the drop-off in corrected
   cells still clears D≥60 at r_inner=0.03 is the open question the sweep decides.
   If `0.03` trims drift below 60 while `0.10` reintroduces the ghost, the two
   gates trade along `r_inner` exactly as they traded along the M0.4 taper width.

## Verdict framework (to be finalized after the run)

Best variant = the row that clears **D≥60 AND S(hf_ref)≥0.98 AND M(|motion−100|≤5)**.

- If an **E3 scale-selective** row clears all three → that is the answer; report the
  `(r_inner, mc)` and note it beats both the M0.4 taper and the broadband shift.
- Else → **NONE passes.** Report which gate binds and, critically, whether the E1
  floor _explains_ it: if `hf_ssim(control)` at the variant's effective d is itself
  <0.98, the S gate is partly **unphysical for translation-type corrections** and
  the honest conclusion is that the hotspot corner needs drift-vs-motion
  _separation_ (velocity-filtered anchoring), not another spatial knob — echoing
  the M0.4 recommendation to PARK this corner and resume the ladder.

The mechanism (band-inconsistency ghosting) is falsifiable via the E1-floor vs
hf_chg comparison in E2; the fix direction (scale-selective) is falsifiable via
the E3 frontier. Both are decided by numbers, not narrative.

## OWNER-RUN RESULTS (2026-09-26, run_m05.py, grid=64 frames=48 seed=1234)

[E1] metric floor — hf_ssim of an exact subpixel shift of a clip vs itself:
  d=0.5px -> 0.9917 (drifted) / 0.9920 (control)
  d=1.5px -> 0.9427 / 0.9448
  d=3.0px -> 0.8707 / 0.8748
  => hf_ssim(control) UPPER-BOUNDS hf_ref for any method translating content ~d px.
  A perfect 1.5px translation scores ~0.94 BY CONSTRUCTION — the 0.98 gate is
  unphysical for large translations. The inner-band design sidesteps this
  because the bump sits below the HF measure's scale.

[E2] mc variants — broadband-consistent (a) vs lowband-only (b):
  (a) mc=0.50: drift 75.7% / hf_ref 0.8604 / motion 65.5% / segs 27  FAIL
  (b) mc=0.50: drift 70.2% / hf_ref 0.9346 / motion 100.5% / segs 141 fail
  => broadband-consistent FALSIFIED: moving all content moves the balls too;
  motion collapses (65.5%), scene-cut detector trips (27). P1 wrong; the fix
  must be band-selective, not global.

[E3] scale-selective — correction restricted to inner radial band r < R:
  R=0.030 mc=0.50: 64.0% / 0.9946 / 0.9955 / 99.7% / 141  [DSM] **PASS (first)**
  R=0.030 mc=0.85: 59.5% / 0.9946 / 0.9952 / 99.7%        [.SM] fail
  R=0.060 mc=0.50: 58.6% / 0.9849 / 0.9838 / 99.8%        [.SM] fail
  R=0.100 mc=0.50: 76.1% / 0.9588 / 0.9582 / 99.8%        [D.M] fail (bleeds into ball band)
  Band energy attribution: bump = 73.0% in [0,.03) + 26.6% in [.03,.06);
  balls = 54.0% in [.06,.14) + 41.2% in [.14,.5). Disjoint supports confirmed.

VERDICT: the hotspot axis is solved in the harness by correction-band selection
(r<0.03, mc=0.5): 64.0% >= 60 gate, fidelity unharmed (0.9946), motion 99.7%.
Next: land it in the engine (M0.6) with Astra's refinements (phase-plane
estimator, alpha-beta trajectory smoother, flat-top cosine edge) — see
reviews/DEEP_DIVE_M05_astra.md for the recipe + the oracle four-arm test that
will separate estimator error from metric error.
