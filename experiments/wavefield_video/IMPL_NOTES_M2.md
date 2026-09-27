# IMPL_NOTES_M2 — legitimate-motion TRAP test (2026-09-26)

## Task recap

Every M0/M1 success contained something we WANTED reduced, so a skeptic can still
say crumb_coherence is "a low-frequency variance suppressor," not "a coherence
engine that distinguishes drift from intended evolution." M2 is Astra's
top-priority falsification (reviews/DEEP_DIVE_M1_assess.md §2, the "M1c
legitimate-motion non-interference" test) that tries to kill the dangerous narrow
hypothesis:

> **H_suppress:** crumb_coherence suppresses low-frequency temporal change whether
> it is pathological (drift) OR semantically correct (intended motion).

If H_suppress is true, the product is a frequency-selective stabilizer, not a
drift/intent discriminator — a much weaker claim.

## What was built (no engine changes; scripts only)

`crumb_coherence/scripts/run_trap.py` — a 64x64 scene with **no injected wander**
(the scene IS the clean ground truth, raw == clean):

- a **large soft Gaussian object** (sigma 10) translating at **0.15 px/frame in x**
  — its spectral energy sits at r ~ 1/(2*pi*10) ~ 0.016 cyc/px, INSIDE the r<0.03
  band `complex_mc` corrects. This is the trap: legitimate low-frequency MOTION
  living exactly where the plugin acts.
- a **slow sinusoidal global illumination** (+/-12%, ~1 cycle over the clip) — an
  intended low-band temporal change.
- **3 small balls** on independent linear trajectories (legit HF motion; also make
  HF-SSIM meaningful — damage to detail would show here).

The frozen SHIPPING config is applied with **no tuning**: `complex_mc`,
`band=0.03`, `strength=0.5`, `corr`, `hard` edge, `alpha=0.95`, `rho=0.995`,
`cutoff=0.14`.

Measures + kill criteria (PASS = all four hold):

| #   | measure                   | definition                                                        | PASS            |
| --- | ------------------------- | ----------------------------------------------------------------- | --------------- |
| 1   | centroid-motion retention | path-length ratio of the low-band energy centroid, plugin / clean | > 95%           |
| 2   | trajectory distortion     | max_t \|\|plugin_c(t) − clean_c(t)\|\| / clean bbox-diagonal      | < 5%            |
| 3   | HF-SSIM vs clean          | highfreq_ssim(clean, plugin)                                      | > 0.98          |
| 4   | low-band variance ratio   | lowband_trajectory_variance(plugin) / (raw)                       | in [0.90, 1.10] |

The centroid is estimated with ONE self-referential estimator (low-band
reconstruction, per-frame spatial-DC strip, positive-lobe centroid) applied
identically to clean and plugin, so estimator bias cancels in the ratio; the
script also prints an estimator-validity line (clean centroid x-range vs the
KNOWN object x-range) so you can confirm the estimator tracks the object.

Naming (per the assessment): the M0.7 arms are the "geometric-displacement
control" and "known-component replacement" (NOT "oracle"). This test uses neither.

## Exact commands (owner-run)

```bash
cd experiments/wavefield_video
PY=~/vibevoice-env/bin/python

$PY crumb_coherence/scripts/run_trap.py --json crumb_coherence/out/trap.json
# optional robustness: vary the horizon / seed
$PY crumb_coherence/scripts/run_trap.py --frames 48
$PY crumb_coherence/scripts/run_trap.py --frames 192 --seed 1
```

## RESULTS — PENDING OWNER RUN

> `python` is execution-gated in this session (owner-run convention from
> IMPL_NOTES_M0_2..M1). No numbers invented; run the command and paste the block.

| measure                       | value | PASS? |
| ----------------------------- | ----- | ----- |
| (1) centroid-motion retention |       |       |
| (2) trajectory distortion     |       |       |
| (3) HF-SSIM vs clean          |       |       |
| (4) low-band var ratio p/raw  |       |       |
| VERDICT (all four)            |       |       |

## Pre-registered expectation (HYPOTHESIS, not measurements)

Grounded in the mechanism, stated before the run so it cannot be retrofitted:

- **I expect the trap to FAIL (H_suppress confirmed), most robustly on measure (4).**
  The engine's core is an EMA anchor blended in at `aw ≈ alpha·gauss ≈ 0.95` near
  the object's frequencies; the output low band is ≈ `0.95·anchor + 0.05·(shifted
current)`. Reducing low-band temporal variance is _precisely_ what that anchor
  does, and here that variance is LEGITIMATE (object translation + illumination),
  so `lowband var ratio` should drop below 0.90. Measure (4) is the cleanest
  indicator.
- **Retention (1) and distortion (2) likely FAIL too, magnitude uncertain.** An EMA
  of a constant-velocity ramp reaches the input's slope only after ~`1/(1−rho)`
  = 200 frames; at the T=96 horizon it is still in transient, so the anchor moves
  slower than the object and lags it → the blended object under-moves and trails
  the clean trajectory. I expect retention materially below 95% and distortion
  above 5%, but the exact margins depend on the transient-vs-horizon ratio (hence
  the `--frames 192` robustness run — a LONGER horizon lets the EMA approach the
  object's velocity, which may _raise_ retention; that itself is diagnostic).
- **HF-SSIM (3) likely PASSES (> 0.98).** The plugin only moves the low band; the
  balls and edges (HF) are untouched. This is exactly why HF fidelity is necessary
  but NOT sufficient — it can read 0.99 while legitimate low-frequency motion is
  being damaged. That is the whole reason this trap exists.

If instead all four PASS, the plugin is smarter than feared — see the verdict.

## Verdict framework — which hypothesis dies

- **If the trap FAILS** (any of retention ≤95%, distortion ≥5%, or var-ratio
  outside [0.90,1.10]; HF is a sanity check, not the crux): **H_suppress is
  CONFIRMED**, and the hypothesis that DIES is the strong product claim —
  _"the shipping complex_mc distinguishes nuisance drift from intended
  low-frequency evolution and is a do-no-harm sidecar for general content."_ What
  survives is the weaker, honest claim: crumb is a **frequency-selective
  low-band stabilizer** that is only safe when the low-band change it sees is
  actually inconsistent with the state model (i.e. real drift). Consequence for
  the next round: `complex_mc` must be **gated** before it touches general content
  — velocity-aware anchoring (don't correct a persistent, self-consistent
  velocity), an explicit drift-vs-motion detector, or restricting the shipped
  default to **magnitude** mode (which leaves phase/motion free and passed the M0
  gain_field motion check). It also means the DSM objective is not yet a proxy for
  "preserve intended motion," and a motion-compensated / tracked-trajectory metric
  must join HF-SSIM (Astra red flag #3).

- **If the trap PASSES** (all four): **H_suppress is FALSIFIED** — the frozen
  complex_mc sees strong legitimate low-frequency motion and leaves it essentially
  alone. The claim strengthens from "interesting spectral filter" toward
  "generator-agnostic coherence sidecar that distinguishes drift from intent," and
  the program can proceed to the real spatial-phase generator test (M1b) with the
  do-no-harm property established. This would be the threshold Astra names at the
  end of §2.

Either outcome is decisive and cheap. Numbers decide, not this paragraph.

## Honest limits

- **Transient regime.** At T=96 the EMA (rho=0.995, time-constant ~200) has not
  reached steady state, so retention is measured mid-transient; the `--frames 192`
  run probes whether a longer horizon changes the verdict.
- **Single synthetic scene / seed** (balls are randomized only in that the seed
  fixes them; the object + illumination are deterministic). One scene is enough to
  falsify a "does no harm" claim (a single clear failure kills it), but not to
  bound the effect across content — treat a PASS as "not falsified here," not
  "proven safe everywhere."
- **Centroid is a low-band proxy**, not a tracked semantic object; it is validated
  against the known object trajectory in the script output, but it is still a
  self-referential estimator, so measure (1)/(2) are proxies for "did the intended
  large-scale motion survive," not ground-truth optical flow.
- 64x64, CPU, < 5 min; no engine changes (crumb_coherence/ frozen).
