# IMPL_NOTES M0.3 — motion-compensated phase anchoring (2026-09-26)

## Principle chosen: #1, MOTION-COMPENSATED ANCHORING (not split-memory)

M0.2 falsified "pull phase toward the anchor": it degrades monotonically
(50.2% -> 13.3%) because the EMA anchor _itself lags and wanders_, so a blend
toward it fights the true signal. The fix is not a gentler pull — it is a
different frame of reference.

By the shift theorem, a low-band structure displaced by `p(t)` has coefficients
`Â_k · exp(-i 2π k·p)`: magnitude flat, position living entirely in a phase ramp
that is _linear in k_. So instead of blending phase, we:

1. **Measure** `p(t)` — the sub-pixel offset of the current low band from the
   anchor — by low-pass **phase correlation** (whitened cross-power spectrum,
   inverse-transformed, parabolic sub-pixel peak fit).
2. **Undo** `mc_strength · p(t)` with an **exact frequency-domain shift**
   (`X_lo · exp(-i 2π k·p)`), re-centering the wandering bump onto the anchor's
   position _before_ the (existing complex) blend.

Position drift is removed by construction; the blend that follows only has to
lock magnitude/shape, so it no longer has a positional mismatch to fight — the
exact trap M0.2 hit.

### Deliberate deviation from the task's phrasing (justified)

The task suggested estimating the shift from _successive frames_ ("cross-power
spectrum of successive frames"). I estimate it **against the anchor** (an
absolute offset) instead. Integrating a per-frame velocity re-accumulates
exactly the drift error MC is meant to cancel; measuring the absolute offset vs.
the near-static anchor is self-correcting and needs no integration. This is the
same reasoning that makes the anchor (not the previous frame) the right
reference for the whole plugin.

### Why SSIM and motion% are structurally safe

The engine only ever rewrites the low-band box; `put_low_band` leaves every
high-frequency coefficient untouched. `highfreq_ssim` and the sharp-centroid
multi-blob motion tracker both live in the high band. A low-band re-centering
therefore _cannot_ touch detail or the ball trajectories — it can only move the
smooth wandering bump. (The only coupling is the final `clamp(0,1)` + YCbCr
round-trip, which is why complex already scores 0.9957 rather than 1.0000.)

## What was added (crumb_coherence/ only, numpy/torch, no new deps)

`core.py`

- `estimate_lowband_shift(Xlo, anchor, H, W, exclude_dc=True) -> (dy, dx)` —
  whitened low-pass phase correlation; DC excluded (no positional info);
  returns `(0.0, 0.0)` on a cold anchor.
- `apply_lowband_shift(Xlo, dy, dx, H, W)` — exact shift-theorem translation of
  the low band; unit-modulus ramp, so magnitude and detail are preserved.
- `_box_freqs`, `_parabolic_offset` helpers.
- New anchor mode **`complex_mc`** and constructor knob **`mc_strength`**
  (default `0.0`). `complex_mc` = `complex` target/blend + the MC pre-shift.
  Every existing mode is byte-identical (new name, new knob only).

`scripts/run_m0.py`

- `complex_mc` engine added to both scenario tables (add, don't remove).
- New flags: `--hotspot-mc-strength` (default 0.85) and `--hotspot-mc-sweep`
  (default `0,0.5,0.85,1.0`). All prior flags unchanged.
- A sweep block prints drift%/motion%/ssim across `mc_strength` for the hotspot.

`tests/test_core.py` — invariant count raised (12 -> 16). New/extended:

- `complex_mc` added to the α=0 no-op and segment==frame-loop invariants.
- `test_mc_strength_zero_equals_complex` — `complex_mc(mc=0)` byte-identical to
  `complex` (the do-no-harm guarantee + sweep anchor).
- `test_mc_strength_only_affects_complex_mc` — no leak into other modes; changes
  complex_mc; negative `mc_strength` rejected.
- `test_estimate_lowband_shift_recovers_known_shift` — recovers a known circular
  roll to sub-pixel accuracy; cold anchor -> `(0,0)`.
- `test_apply_lowband_shift_is_exact_and_magnitude_preserving` — equals a real
  roll's low band; never changes per-cell magnitude; `(0,0)` is an exact identity.

## Exact commands

```bash
cd experiments/wavefield_video

# 1. Invariants (must be all-pass):
python crumb_coherence/tests/test_core.py

# 2. Hotspot (positional/phase drift) — main table + mc_strength sweep:
python crumb_coherence/scripts/run_m0.py --scenario hotspot --no-video

# 3. gain_field (magnitude drift) — regression guard, all engines must stay >=85%:
python crumb_coherence/scripts/run_m0.py --scenario gain_field --no-video

# (optional) single-point MC strength override for the main hotspot row:
python crumb_coherence/scripts/run_m0.py --scenario hotspot --no-video \
    --hotspot-mc-strength 1.0
```

## RESULTS — PENDING OWNER RUN

> I was not granted execution approval in this session (the `python` step is
> gated, consistent with the owner-run convention in IMPL_NOTES_M0_2.md /
> M0_RESULTS.md). The tables below are intentionally left blank rather than
> fabricated. Run the three commands above and paste the output; I will fill the
> tables and write the honest verdict (including "<60% if that is the truth").

### Invariants

```
[paste: python crumb_coherence/tests/test_core.py]
expected: "All 16 tests passed."
```

### Scenario B — hotspot (target: beat 50.2%, i.e. >=60%, motion>=97, ssim>=0.98)

| engine                  | drift%(pos) | \|ΔE\|corr | hf_ssim | motion% | state | verdict                               |
| ----------------------- | ----------- | ---------- | ------- | ------- | ----- | ------------------------------------- |
| stats-EMA baseline      |             |            |         |         | 24B   |                                       |
| spectral dc_only        |             |            |         |         | 24B   |                                       |
| spectral magnitude      |             |            |         |         |       |                                       |
| spectral complex        |             |            |         |         |       | (M0.2 champion: 50.2 / 99.7 / 0.9957) |
| **spectral complex_mc** |             |            |         |         |       |                                       |

mc_strength sweep (complex_mc):

| mc_strength | drift%(pos) | \|ΔE\|corr | hf_ssim | motion% | verdict             |
| ----------- | ----------- | ---------- | ------- | ------- | ------------------- |
| 0.00        |             |            |         |         | (== complex, ~50.2) |
| 0.50        |             |            |         |         |                     |
| 0.85        |             |            |         |         |                     |
| 1.00        |             |            |         |         |                     |

### Scenario A — gain_field (regression guard: every engine drift%(var) >= 85%)

| engine                  | drift%(var) | drift%(ΔE) | hf_ssim | motion% | state | verdict |
| ----------------------- | ----------- | ---------- | ------- | ------- | ----- | ------- |
| stats-EMA baseline      |             |            |         |         | 24B   |         |
| spectral dc_only        |             |            |         |         | 24B   |         |
| spectral magnitude      |             |            |         |         |       |         |
| spectral complex        |             |            |         |         |       |         |
| **spectral complex_mc** |             |            |         |         |       |         |

## Predicted shape (hypothesis, NOT a claim — to be confirmed/refuted by the run)

- Hotspot drift%(pos) should rise monotonically with `mc_strength`
  (`0 -> ~50.2`, the complex baseline; `-> 1.0` approaching full re-centering),
  the **opposite** slope to M0.2's phase-anchor knob — that sign flip is the
  whole point of MC vs. blend-toward-a-lagging-target.
- motion% and hf_ssim should stay ≈ their complex values at every `mc_strength`
  (low-band-only edit; high band untouched).
- gain_field should be ≈ unchanged: the drift there is magnitude, and the MC
  shift is a unit-modulus multiply, so drift%(var) is invariant to it.

## Verdict: PENDING (fill after run).

If complex_mc does not clear 60% with motion>=97 and ssim>=0.98, say so plainly
and record the sweep shape; the mechanism is falsifiable exactly like M0.2.


## OWNER-RUN VERDICT (2026-09-26, Zeph, all CPU)

Sixteen invariants pass. MC sweep on hotspot (pos-drift / hf_ssim / motion):
  mc=0.0 -> 50.2% / 0.9957 / 99.7%
  mc=0.5 -> 70.2% / 0.9341 / 100.5%   <- PEAK, D gate clears (>=60)
  mc=0.85 -> 59.4% / 0.9297 / 100.4%
  mc=1.0 -> 31.3% / 0.9296 / 100.5%
Slope is UNIMODAL, not monotonic (prediction said upward; actual peaks at 0.5 then falls).
Default reset 0.85 -> 0.5 (measured peak).

Open defect: hf_ssim ~0.93 at every mc>0 while drift removal works - consistent with a
SPATIAL-WARP/interpolation HF loss, not a physics limit (gain_field complex_mc also shows
0.9678 despite ~zero wander = lossy even at identity shift). Fix direction: exact k-space
phase-ramp shift (unitary, no interpolation) + true no-op at ~zero estimated shift.
-> M0.4 fired on this.

M0.3 verdict: direction VALIDATED (70.2% peak vs 50.2% anchor-only), one gate left (S).

## OWNER-VERIFIED ADDENDUM: M0.4 verdict (2026-09-26, all runs by owner)

K-space phase-ramp hypothesis: FALSIFIED (numerically identical output to the
previous warp: mc=0.5 no-taper reads 70.2% / 0.9341 both before and after).
The HF-SSIM cost is not interpolation; it is intrinsic to the pinning operation.

Metric wiring corrected in run_m0.py: the hotspot S-gate now measures hf_ssim
of corrected vs CLEAN control (truth-alignment), with hf_chg (vs input) kept
as a non-gating diagnostic. Evidence justifying the switch: the honest complex
engine keeps hf_ref = 0.9946 (essentially equal to hf_chg = 0.9957), i.e. a
correct low-band correction does NOT harm HF vs truth; only the mc pinning
degrades it (hf_ref = 0.9346), which the old in-vs-out wiring could not
distinguish from legitimate motion.

Measured frontier (hotspot, pos-drift removal vs hf_ref):
  complex rho=0.995 (default)   50.2%  | 0.9946  <- champion, honest
  rho=0.997                     48.2%  | 0.9945
  rho=0.9990                    45.9%  | 0.9943
  complex_mc 0.5 taper=True     41.6%  | 0.9929
  complex_mc 0.5 taper=False    70.2%  | 0.9346  <- pays real HF fidelity
  complex_mc 0.7 taper=True     37.7%  | 0.9911
  complex_mc 0.85 taper=False   59.4%  | 0.9297

Verdict: NO measured setting clears both gates (drift>=60 AND hf_ref>=0.98).
The mc family buys drift removal with genuine fidelity damage; the taper buys
fidelity by refusing to correct. PARK the hotspot-corner perfection: it needs
drift-vs-motion separation (velocity-filtered anchoring), not knob tuning.
complex at defaults remains the shipping candidate for the phase axis; the
additive/luminance axis (gain_field) passes cleanly at 87-92%.

Recommended next: do NOT spend more rounds on the hotspot corner now; resume
the ladder (long synthetic sequence + in-loop vs post-hoc) where the passing
axis carries product value. Revisit separation idea when an in-loop slot exists.
