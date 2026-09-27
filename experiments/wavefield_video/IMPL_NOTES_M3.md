# IMPL_NOTES_M3 — the GATED engine: velocity-coherence gating (2026-09-27)

## Task recap

M2 (`--legacy` here) CONFIRMED the dangerous hypothesis **H_suppress**: the frozen
`complex_mc` cannot tell a translating subject from wander and destroys legitimate
low-frequency motion — centroid retention 54.7%/70.8% (bar >95%), trajectory
distortion 64.1%/58.9% (bar <5%), HF-SSIM fine (~0.99). M3 must make the
correction SAFE around legitimate motion: apply `complex_mc` only where the
low-band motion looks like INCOHERENT WANDER, and pass COHERENT motion untouched.

## Two constraints that dictate the mechanism (both proven by reading the trap)

1. **Only a near-identity passes the trap.** Trap measure (4) requires the ±12%
   illumination sinusoid (a _legitimate_ low-band **magnitude** change) to survive
   (var ratio in [0.90,1.10]). `complex_mc` freezes position → fails (1),(2).
   `magnitude` mode would _flatten that illumination_ → fails (4). **Neither
   content-blind default passes.** The only thing that passes all four is an
   engine that becomes a **provable no-op on legitimate content**. That is what
   the gate delivers (g→0 ⇒ `Xlo_new == Xlo`), and it is _why a gate is required_,
   not optional — the "ship magnitude as the safe default" fallback does **not**
   clear this trap.
2. **The discriminator is noise- and curvature-limited, not speed-based.** The
   trap object crawls at 0.15 px/frame (≈ the sub-pixel phase-corr noise floor)
   in a straight line; the M0 hotspot _wanders fast_ (~1.5 px/frame) along a
   curve that RETURNS. So the separating signal cannot be speed (a fast pan is
   legit, a slow wander is drift) and cannot be raw per-frame velocity (noise
   swamps the crawl). It must be **directional persistence of the accumulated
   trajectory**: does the low-band structure keep going one way (motion) or
   oscillate and come back (wander)?

## The gate (what it is, why it separates)

`complex_mc` already measures the current low band's sub-pixel offset
`p(t)=(dy,dx)` from the (slow) anchor every frame. The gate adds the missing
signal — is that displacement _directionally persistent_:

- Light-smooth the offset: `p̄(t) = k·p̄(t-1) + (1-k)·p(t)` (k=`mc_gate_pos_decay`,
  ~1-2 frame time-constant). Kills 1-frame estimator noise; leaves wander's
  tens-of-frames oscillation untouched. `v(t) = p̄(t)-p̄(t-1)` = denoised velocity.
- Two leaky integrators (`mu`=`mc_gate_decay`): a **vector** EMA `s(t)` of `v`
  and a **scalar** EMA `sp(t)` of `‖v‖`. Coherence
  **`C(t) = ‖s(t)‖ / (sp(t)+ε)` ∈ [0,1]**.
  - Coherent motion (pan / translation / accelerating subject): every step points
    the same way ⇒ vector and scalar EMAs agree ⇒ **C→1**.
  - Wander: steps cancel because the structure returns (net displacement bounded,
    path length keeps growing) ⇒ **C→0**.
- Gate **`g = 1 − smoothstep(lo, hi, C)`**: full correction on wander (C≤lo ⇒ g=1),
  an **exact identity** on coherent motion (C≥hi ⇒ g=0).
- `g` scales **both** the mc pre-shift _and_ the anchor-blend weight
  (`aw_eff = aw·g`), so at g=0 `Xlo_new == src == Xlo`: the whole low band is
  untouched — position, magnitude AND any legitimate illumination all preserved.

Why the smoothed vector/scalar ratio is the right estimator: the numerator is a
_vector_ mean (per-frame noise cancels), the denominator a mean of _magnitudes_
(noise biases it up). Smoothing the POSITION removes that bias at a timescale too
short to hide real curvature — so a slow straight crawl reads coherent while a
fast Lissajous still reverses and reads incoherent.

## What was built (engine changes are additive and OFF by default)

`crumb_coherence/core.py`

- `_smoothstep(lo,hi,x)`; module GATE comment; `GATE_SPEED_FLOOR`.
- `SpectralCoherenceEngine` gains `mc_gate` (bool, **default False**),
  `mc_gate_lo=0.35`, `mc_gate_hi=0.75`, `mc_gate_decay=0.9`,
  `mc_gate_pos_decay=0.5`, all validated.
- `CoherenceState` gains 8 float/bool gate fields (no tensor state; `state_bytes`
  unchanged — still the anchor only).
- `_velocity_gate(dy,dx,state)` computes `C`/`g`; `process_frame` scales the
  correction by `g` in `complex_mc`; cut/reset clear the gate history.
- **`mc_gate=False` ⇒ g≡1.0 ⇒ every downstream value is byte-identical** to the
  frozen engine (`x*1.0` is exact for finite float32). So all pre-M3 invariants
  hold and the old behaviour is reachable by a flag (acceptance 4).

`crumb_coherence/scripts/run_trap.py` — `--scene {translate|accelerate|curve|
wander}` (translate = verbatim M2 scene; accelerate = 2nd legit class; curve =
gentle 50° arc stress; wander = M0-hotspot Lissajous, DIAGNOSTIC-ONLY), `--legacy`
(drop the gate), `--diag` (per-frame C/g), gate-knob overrides, SHIP_CFG now
carries the gate.

`crumb_coherence/scripts/run_m0.py` — hotspot `complex_mc` runs the SHIP (gated)
config by default (`--hotspot-mc-gate`/`--no-…`), so the removal bar is tested on
what actually ships.

`crumb_coherence/tests/test_core.py` — gate invariants: default no-op &
mode-isolation & param validation; smoothstep bounds/monotonicity;
**gate_preserves_coherent_translation** (gated damage < ½ ungated, settled g<0.5);
**gate_still_corrects_wander** (stays engaged, tracks ungated, g>0.5);
cut clears velocity history.

`run_m3.sh` — one-command acceptance suite + gate-separation sweep.

## Acceptance — RESULTS PENDING OWNER RUN

> `python` is execution-gated in this session (owner-run convention, per
> IMPL_NOTES_M0_2..M2). Nothing below is invented. Run and paste:
>
> ```bash
> cd experiments/wavefield_video
> PY=~/vibevoice-env/bin/python bash run_m3.sh 2>&1 | tee reviews/claude_m3_log.txt
> ```

**(1) TRAP gated — both horizons** (bars: retention>95, distortion<5, hf≥0.98,
var∈[.90,1.10]):

| scene     | frames | retention% | distortion% | HF-SSIM | var ratio | PASS? |
| --------- | ------ | ---------- | ----------- | ------- | --------- | ----- |
| translate | 96     |            |             |         |           |       |
| translate | 192    |            |             |         |           |       |

**(2) 2nd legit scene — no trap overfit** (same bars):

| scene                    | frames | retention% | distortion% | HF-SSIM | var ratio | PASS? |
| ------------------------ | ------ | ---------- | ----------- | ------- | --------- | ----- |
| accelerate               | 96     |            |             |         |           |       |
| accelerate               | 192    |            |             |         |           |       |
| curve (stress, reported) | 96     |            |             |         |           |       |

**(3) Regression:**

| check                           | metric                     | value | bar        | PASS? |
| ------------------------------- | -------------------------- | ----- | ---------- | ----- |
| M0 hotspot, **gated (ship)**    | drift%(pos)                |       | ≥60%       |       |
| M0 hotspot, ungated (reference) | drift%(pos)                |       | (context)  |       |
| M0 gain_field magnitude         | drift%(var) / motion%      |       | ≥60 / ±5   |       |
| test_core.py                    | all tests                  |       | all pass   |       |
| M1 (magnitude, config A)        | low-band var drop / detail |       | ≥50 / ≥.98 |       |

**(4) Legacy (ungated) still FAILS the trap** (for the record):

| scene     | frames | retention% | distortion% | var ratio | (expect FAIL)            |
| --------- | ------ | ---------- | ----------- | --------- | ------------------------ |
| translate | 96     |            |             |           | ~54.7 / 64.1 / 0.53 (M2) |
| translate | 192    |            |             |           | ~70.8 / 58.9 (M2)        |

**Gate separation (section 0 of the run):** settled coherence C for `translate`
(want HIGH, →g=0) vs `wander` (want LOW, →g=1), across `mc_gate_decay`:

| decay | C(translate) settled | C(wander) settled | gap |
| ----- | -------------------- | ----------------- | --- |
| 0.80  |                      |                   |     |
| 0.90  |                      |                   |     |
| 0.95  |                      |                   |     |
| 0.97  |                      |                   |     |

## Pre-registered expectation (HYPOTHESIS, before the run — cannot be retrofitted)

- **Trap translate/accelerate PASS gated.** With the position pre-smoother I
  expect settled C(translate) ≈ 0.85–0.95 (above hi=0.75) ⇒ g→0 within ~1–2
  frames ⇒ the plugin is a near-identity ⇒ all four measures pass at both
  horizons. Accelerate is monotone +x ⇒ C≈1 ⇒ passes at least as easily.
  **Kill:** if settled C(translate) < hi on either horizon, g stays partial,
  motion is damaged, and the trap FAILS — then lower `hi` toward the diagnostic
  gap (or raise `mc_gate_decay`) and re-run; if no (lo,hi,mu) both passes the trap
  and keeps hotspot≥60%, the gate cannot cleanly separate → Plan C below.
- **M0 hotspot stays ≥60% gated, but by a THINNER margin than ungated.** The
  hotspot's first velocity reversal is ~frame 12 of 48; frames before it look
  locally coherent and escape (g→0), so removal drops from the ungated ~90%+
  toward ~65–80%. **Kill:** removal <60% gated ⇒ the fast-Lissajous wander is too
  motion-like early; raise `mc_gate_decay` (longer memory ≈ cumulative net/path,
  which stays low once a return is seen) and re-run, or accept Plan C.
- **`curve` (50° arc) is the BOUNDARY and may be partially gated.** A turning path
  has lower directional persistence than a straight one; I expect C(curve) between
  translate and wander, so retention may land below 95%. Reported honestly, NOT a
  gate on the ship decision — it maps where the gate stops protecting motion.
- **test_core + M1(magnitude) + gain_field PASS unchanged** (gate off for those
  paths ⇒ byte-identical).

## Verdict framework — decide from the numbers, both outcomes prepared

- **GREEN (trap passes both horizons + accelerate + hotspot≥60% + tests + M1):**
  ship gated `complex_mc` as the default. Then bump the plugin version (add
  `__version__` M3), commit, and add a claim-boundary section to THE_POSITION.md.
  Honest claim unlocked: _"corrects incoherent low-band drift while leaving
  coherent camera/subject motion and legitimate exposure change intact"_ — bounded
  to straight/accelerating motion on single-dominant-structure scenes.
- **AMBER (trap passes only after tuning hi/mu):** ship the tuned config; note the
  narrower hotspot margin and the curvature boundary in the envelope.
- **RED / Plan C (no config separates them):** the gate cannot distinguish this
  fast-curved wander from coherent curved motion per-frame. Then the default is
  **`alpha=0` OFF** (apply the engine only where drift is confirmed / on
  known-wander content behind the `mc_gate=False` flag), because — per constraint
  #1 — magnitude mode does NOT pass the trap either. Say so plainly; the "rescue"
  offer then requires human clip triage, not a content-blind pass.

## Honest envelope (what a PASS would let us claim, and NOT)

- **Global, single-signal gate (v1).** C is one scalar per frame from the
  dominant low-band motion. A scene mixing coherent motion + independent wander is
  gated by whichever dominates ⇒ **per-region gating is the next step**, not done.
- **Boundary is curvature, not speed.** Straight/accelerating motion is protected;
  a sharp turn or a _slow_ wander (never reverses within memory) is the grey zone.
  Stated, measured (`curve`), not hidden.
- **Coupling.** In `complex_mc`, g gates position AND magnitude together, so on a
  coherent scene it also (correctly) leaves illumination alone; it cannot
  simultaneously kill positional wander and preserve legit illumination in the
  same frame — that is `magnitude` mode's job on different content.
- **Still synthetic + one real rollout.** A trap PASS is "not falsified here,"
  not "safe on all footage." ENG-1 (1080p) retention bars remain the real-content
  frontier; re-run ENG-1 with the gate is the immediate follow-up.
- No photorealism claims; middleware corrects the generation context, never the
  exported MP4; anchor magnitude / preserve phase unchanged.

## RESULTS — operator acceptance run (2026-09-27 ~01:15 EDT, `run_m3.sh`, vibevoice-env py3.14)

**VERDICT: RED — the gate does not separate.** Per the pre-registered framework:
no `(lo,hi,decay)` config separates legitimate motion from wander; default falls
back to OFF semantics; Plan C activates.

### 0. Gate separation (the disqualifier)
C(translate) ≈ C(wander) at every decay (settled means):

| decay | C translate | C wander | gap |
| ----- | ----------- | -------- | --- |
| 0.80  | 0.946       | 0.944    | 0.002 |
| 0.90  | 0.910       | 0.908    | 0.002 |
| 0.95  | 0.868       | 0.866    | 0.002 |
| 0.97  | 0.836       | 0.835    | 0.001 |

→ ~0.002 margin on a [0,1] scale: no usable window. The discriminator has no information here.

### 1–2. Trap (gated)
| scene      | frames | retention% | distortion% | HF-SSIM | var ratio | verdict |
| ---------- | ------ | ---------- | ----------- | ------- | --------- | ------- |
| translate  | 96     | 118.6      | 6.33        | 1.0000  | 0.987     | FAIL (marginal) |
| translate  | 192    | 306.9      | 57.62       | 0.9967  | 1.238     | FAIL (late misfire: C→0.44–0.59, g→0.87; correction ADDS displacement) |
| accelerate | 96     | 103.3      | 2.13        | 1.0000  | 0.972     | **PASS** |
| accelerate | 192    | 265.8      | 71.64       | 0.9976  | 1.118     | FAIL (same late misfire) |
| curve      | 96     | 119.4      | 7.83        | 0.9999  | 0.984     | FAIL (boundary, as predicted) |

Legacy (ungated) FAILs recorded: 54.7%/64.1% and 70.8%/58.9% — matches M2 exactly.

### 3. Regression — M0 hotspot
- gated (ship): `complex_mc` drift = **−0.2% → FAIL** (gate stays closed on real
  wander: the correction is effectively dead).
- ungated: 64.0% PASS [DSME] (unchanged).
- Band note: at 64×64 with `mc_band=0.03` the low-band box is 1 cell (DC-only) —
  `complex_mc == complex` at this grid (test-grid limitation, not engine).

### 7. Unit tests
7/8 pass; `test_gate_preserves_coherent_translation` FAILS (g=0.76 — "gate did
not open"). The unit test caught the same defect independently of the 192f trap.

### 8. M1 regression: PASS (magnitude mode unaffected by the gate: −85.9% lowband
var, −55.7% centroid var, detail 0.988, motion kept).

### The finding (falsifies H_gate cleanly)
Directional persistence of the smoothed low-band trajectory does not separate
authored motion from wander on these scenes — both read directionally persistent
at matched C, and the late-sequence C dips (curvature/noise decorrelation) make
the gate *engage on legitimate motion*, adding displacement. This corroborates
the third-opinion analysis (Opus consult, 2026-09-27): **statistics alone cannot
separate legitimate unstructured low-band motion from drift; escape requires a
prior — the model's own next-frame prediction (in-loop) or authored motion
(3D-control).**

### Next (per framework — Plan C + D1)
1. **Residual anchoring** (predict-then-correct-the-residual): kill criterion
   pre-registered — trap distortion <5% AND hotspot ≥60% co-pass.
2. Product pivot: triage-form rescue + authored-motion lane as flagship.
3. Gate code stays in-tree, default OFF (already); legacy reachable (already).
