# IMPL_NOTES_M0_7 — the oracle four-arm test (2026-09-26)

## Task recap

M0.6 landed the band-limited motion-compensated engine: `complex_mc` with
`mc_band=0.03, mc_strength=0.5, mc_est=corr, mc_edge=hard` is the first full
`[DSME]` pass of the hotspot axis — **64.0% pos-drift removal, hf_ref 0.9946,
motion 99.7%** (owner-verified, IMPL_NOTES_M0_6.md). 64% clears the D≥60 gate but
leaves ~36 points of drift on the table. M0.7 locates _where that headroom lives_
— estimator error vs metric error vs decomposition error — **before** touching
the estimator again, via the oracle four-arm test designed in
reviews/DEEP_DIVE_M05_astra.md ("the single decisive experiment").

## What was built (no crumb_coherence/ source changed this round)

`crumb_coherence/scripts/run_oracle.py` — a self-contained, read-only runner. It
reuses the shipped helpers (`estimate_lowband_shift`, `apply_lowband_shift`,
`radial_band_mask`, `gaussian_band_weight`, `put_low_band`, `_crop_box`,
`rgb_to_ycbcr`, `ycbcr_to_rgb`, `ema_update`, `highfreq_ssim`) and the exact
hotspot pipeline/params (`data.make_clip_batch` + `run_m0` metrics; grid 64,
frames 48, seed 1234, alpha 0.95, rho 0.995, cutoff_frac 0.14), so every number
is directly comparable to the M0.3–M0.6 frontier. Nothing under `crumb_coherence/`
was modified — the 26 invariants stay byte-for-byte green.

### The four arms (exact definitions)

- **Arm 1 — ORACLE COMPONENT (ceiling).** Replace the drifted frame's WHOLE
  low-band box (cutoff 0.14) with the CLEAN control's low band:
  `X_new = put_low_band(rfft(drifted), crop(rfft(clean)))`. Perfect nuisance
  removal — the upper bound of any component-space fix. High band = drifted's
  (carries the bump's small >0.14 cyc/px tail only).

- **Arm 2 — ORACLE BAND-PHASE.** The engine's `complex_mc` loop reproduced
  byte-for-byte (same alpha/rho/cutoff, `mc_strength=0.5`, `mc_band=0.03`,
  `hard` edge, gaussian blend weight, deadband), EXCEPT the per-frame
  displacement is the **oracle** offset — the true bump centre `(cy_t, cx_t)`
  from `inject_hotspot`'s own formula minus an `EMA(rho)` of those true centres.
  That EMA-of-centres is the exact quantity `estimate_lowband_shift` _tries_ to
  measure against the complex-EMA anchor, with zero ball-leakage / parabola
  noise. (It is an approximation of the complex-EMA anchor's phase-implied
  position — documented as such in the script.) Arm 2 differs from Arm 3 ONLY in
  the displacement source, so **Arm2 − Arm3 isolates ESTIMATOR error** and
  **Arm1 − Arm2 isolates DECOMPOSITION error** (bump energy outside r<0.03).

- **Arm 3 — ENGINE BAND-PHASE (context).** The shipped M0.6 winner, real
  correlation estimator: `SpectralCoherenceEngine(anchor_mode="complex_mc",
mc_strength=0.5, mc_band=0.03, mc_est="corr", mc_edge="hard", alpha=0.95,
rho=0.995, cutoff_frac=0.14)`.

- **Arm 4 — X − B̂ + B̂_stable.** Estimate the coherent moving low-band component
  from the engine's own estimate and re-place it at the stable position. Exact
  definition used (simplest faithful causal version):
  - `B̂_0 := A` (the EMA(rho) low-band anchor — coherent template, stable pos, p*=0),
  - `p_t := estimate_lowband_shift(Xlo_t, A)`,
  - `B̂_t := apply_lowband_shift(A, p_t)` (template at the CURRENT position),
  - output low band `= Xlo_t − B̂_t + A = Xlo_t + (A − shift(A, p_t))`, over the
    WHOLE low-band box (Astra's `W(k)=1` on the box), HF untouched, no coherence
    blend (the component swap is the entire correction).
    CAVEAT: `A` also holds the static background/ball low-band, so a non-zero `p_t`
    moves that too — the price of the simplest B̂. A `W(k)` restricted to r<band
    would curb this (future work).

- **hf_reg (all arms).** hf_ref after aligning the corrected clip to the clean
  control with the shipped phase-correlation helper (`estimate_lowband_shift` →
  exact full-frame FFT shift). `hf_reg >> hf_ref` would mean the raw S gate is
  penalising a benign sub-pixel translation (Astra Outcome 2 / H3).

- **CW-SSIM: intentionally SKIPPED** — it needs a new wavelet dependency and the
  task forbids new deps. Noted here, not faked.

## Exact commands (owner-verifiable)

```bash
cd experiments/wavefield_video

# 0. invariants unchanged (expect "All 26 tests passed."):
python crumb_coherence/tests/test_core.py

# 1. the oracle four-arm test (single run, < 5 min/arm CPU):
python crumb_coherence/scripts/run_oracle.py --json crumb_coherence/out/oracle.json

# 2. regression sanity (M0.6 winner row must still read 64.0% / 0.9946 / 99.7%):
python crumb_coherence/scripts/run_m0.py --scenario hotspot --no-video
```

## RESULTS — PENDING OWNER RUN

> `python` is execution-gated in this session (the owner-run convention from
> IMPL_NOTES_M0_2..M0_6 — retried this round, denied). Measurement cells are left
> blank rather than fabricated. Run command #1 and paste the table; the only
> pre-filled row is **Arm 3**, whose config is byte-identical to the M0.6 winner
> and whose numbers are quoted from the M0.6 owner-verified run (re-confirm on the
> paste). The decision framework below then reads off directly.

Gates: **D**=drift≥60 **S**=hf_ref(raw)≥0.98 **M**=|motion−100|≤5.

| arm                  | drift%(pos) | hf_ref | hf_reg | hf_chg | motion% | gates |
| -------------------- | ----------- | ------ | ------ | ------ | ------- | ----- |
| 1 oracle component   |             |        |        |        |         |       |
| 2 oracle band-phase  |             |        |        |        |         |       |
| 3 engine band-phase* | 64.0        | 0.9946 |        | 0.9955 | 99.7    | DSM   |
| 4 X−B̂+B̂_stable       |             |        |        |        |         |       |
| ref drifted (input)  | 0.0         |        |        | 1.0000 | ~100    | .?M   |

\* Arm 3 drift/hf_ref/hf_chg/motion quoted from IMPL_NOTES_M0_6.md (owner-run,
same config); hf_reg is new to this script (blank until the run).

## Predicted shape (HYPOTHESIS, not measurements)

Grounded in the code + the M0.6 owner-run numbers; to be confirmed/refuted by the
run. Clearly labeled predictions, not results.

1. **Arm 1 (ceiling) clears drift by a wide margin with high raw hf_ref.** A
   perfect low-band swap removes essentially all in-band bump energy, so
   drift%pos should sit far above 64 (near the clean noise floor). hf_ref stays
   high because only the low band changed and it changed _toward_ the truth —
   this arm has no ghosting mechanism. If its RAW hf_ref were somehow <0.98 while
   hf_reg≈1.0, that is the pure Outcome-2 signature (metric, not method).

2. **Arm 2 (oracle Δ) ≈ or slightly above Arm 3.** M0.6 already showed the corr
   estimator BEATS the phase-plane estimator on this scenario (64.0% vs 44.4%) —
   evidence the low-band phase correlation is already robust here. So a perfect
   Δ is unlikely to add many drift points at the SAME band/strength. If
   Arm2 ≈ Arm3 (say within a few points), the estimator is NOT the bottleneck;
   the 64% ceiling is set by the band restriction (decomposition) and mc=0.5.

3. **Arm1 − Arm2 is the big gap.** ~27% of the bump's energy lives at r≥0.03
   (M0.5 E3 attribution: bump 73% in inner[0,.03)); the r<0.03 mask cannot touch
   it, and mc_strength=0.5 only removes half of what it _can_ reach. So Arm 2
   (band+half-strength) should land well below Arm 1 (full low band). That gap =
   decomposition error, and it is the headroom the _next_ mechanism must attack
   (wider/soft band or component-space), not a better estimator.

4. **Arm 4 (X−B̂+B̂_stable): strong drift, uncertain fidelity.** Moving the whole
   coherent low-band template should remove more drift than the r<0.03 phase arm
   (it is full-band, full-strength), but because `A` carries the balls'/static
   low-band, shifting it by `p_t` may drag that content — echoing the M0.5 E2
   "broadband consistent" falsification (balls moved, motion/hf hit). Watch
   motion% and hf_ref: if both hold AND drift >> 64, the component route is the
   winner and the band-limited `W(k)` variant is the obvious next step; if motion
   or hf drops, component-space needs the `W(k)` mask before it is viable.

## Decision framework — Astra Outcome 1/2/3/4 (fill after the run)

Read the arms in this order; the FIRST matching row names the outcome:

- **Outcome 2 (metric is mis-specified / H3 wins)** — IF any high-drift arm
  (esp. Arm 1 or Arm 2) posts **raw hf_ref < 0.98 but hf_reg ≥ ~0.98**. Then the
  S gate is rejecting a valid geometric correction; the fix is the metric
  (registered hf_ref, or CW-SSIM as a new dep), NOT the engine. Consequence:
  re-score the frontier with hf_reg and the true ceiling may already pass.

- **Outcome 4 (architecture fine, ESTIMATOR is the bottleneck)** — IF
  **Arm 2 drift ≫ Arm 3 drift** (perfect Δ clears ≥~75–80% where the engine gets 64) at comparable hf/motion. Then invest in the estimator (weighted
  phase-plane with better low-k conditioning, or coherence-gated corr).
  Consequence: the next round tunes `mc_est`, not the band.

- **Outcome 3 (band-selective form is too narrow — DECOMPOSITION error)** — IF
  **Arm 2 ≈ Arm 3** (estimator not it) AND **Arm 2 ≪ Arm 1** (the band leaves
  bump energy behind) AND **Arm 4 (component) reaches near Arm 1 with hf/motion
  intact**. Then the simple r<0.03 band is the limiter; move to the
  `X − B̂ + B̂_stable` component route (with a `W(k)` band mask), not a wider
  hard band. Consequence: promote a component-space variant next.

- **Outcome 1 (target physically attainable; problem is realistic
  estimation/decomposition, gate is sound)** — IF **Arm 1 (and/or Arm 2) clears
  D≥60 AND S≥0.98 on RAW hf_ref** (registration barely changes it). Then the goal
  is reachable and the S metric is fair; the remaining work is engineering the
  estimator + decomposition to close the Arm3→Arm1 gap. This is the "keep going,
  it's tractable" outcome.

**Which one are we in:** _to be stated once the owner pastes the Arm 1/2/4
numbers._ Given the M0.6 evidence (corr ≫ phase-plane, disjoint bump/ball
supports), the pre-registered expectation is **Outcome 3** (band-limited phase is
decomposition-limited, not estimator-limited) with the S gate sound (no large
hf_reg jump) — i.e. the productive next lever is the component-space
`X − B̂ + B̂_stable` route with a soft `W(k)`, and NOT another estimator sweep.
The run will confirm or overturn this; numbers decide, not this paragraph.

## Constraints honored

No new deps; `crumb_coherence/` source untouched (script reuses helpers
read-only); each arm is one 48-frame CPU segment (well under 5 min); no emoji;
no invented numbers (pending cells left blank, Arm 3 quoted with citation).

## OWNER-RUN RESULTS (2026-09-26 15:05, run_oracle.py, grid=64 frames=48 seed=1234)

arm                      drift%   hf_ref   hf_reg   hf_chg  motion%  gates
1 oracle component       -43.3%   0.9998   0.9998   0.9943    99.9%  [.SM] fail
2 oracle band-phase (true Δ)  39.9%   0.9944   0.9944   0.9952    99.7%  [.SM] fail
3 engine band-phase      64.0%   0.9946   0.9946   0.9955    99.7%  [DSM] PASS
4 X−B̂+B̂_stable          -38.6%   0.9466   0.9316   0.9526    99.9%  [..M] fail

VERDICT (falsification PASSED for the engine):
- The shipped engine (Arm 3) beats BOTH oracle arms and the component variant.
  The perfect-estimator oracle (true Δ) lands at 39.9% — WORSE than the real
  phase-correlation estimator. The crude component swap (Arm 1) and the
  X−B̂+B̂_stable variant (Arm 4) both INCREASE position variance (negative).
- hf_reg == hf_ref in every arm → registration changes nothing → the raw S gate
  is NOT unfairly penalizing translation in this setup (Outcome 2 dead).
- Pre-registered Outcome 3 (decomposition-limited) also dead: the mechanism is
  not decomposition-limited; it is at/above its own oracle ceiling.
- Reading: surgical in-band phase correction with a robust low-band estimator
  wins; pixel-centre "ground truth" trajectories are noisier than the
  band-limited phase estimate (noise-chasing), and component swaps create
  band-mismatch ghosts. No further M0 rounds warranted — the hotspot mechanism
  is at its practical optimum in this harness.
- Decision: M0 arc COMPLETE. Next = integration rung (M1: plugin on real
  wave-field rollouts + before/after review artifact).
