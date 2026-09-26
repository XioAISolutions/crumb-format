# RESEARCH SWEEP M0.5: how the field removes wander (2026-09-26)

## The lineage that matches our engine exactly

Wadhwa et al. 2013, "Phase-Based Video Motion Processing" (SIGGRAPH):
motion in a complex steerable pyramid = PHASE VARIATIONS over time; process
them temporally and you can amplify (reveal) OR ATTENUATE (remove) motion -
with NO optical-flow/estimation step. Direct quote from their attenuation
use-case: "attenuated to remove distracting changes." Our plugin IS a
degenerate single-scale version of this framework (k-space cells = the
pyramid coefficients; our rho/EMA = a crude temporal filter).

Recipe (rxian/phase-video docs, faithful reproduction): compute the pyramid,
"perform band-pass temporal filtering on the phase", modify, reconstruct.
Attenuation = multiply the band-passed phase by alpha < 1 (alpha=0 removes
the band's motion entirely).

## Drift-specific lineage (stabilization)

- USTC motion-decomposition stabilization: explicitly eliminates BOTH
  high-frequency (shake) AND low-frequency (drift) components of the
  jitter - i.e., a two-band temporal design: high-pass for shake, low-pass
  subtraction for drift. Matches our wander = a low-frequency phase component.
- Traditional drift-free stabilization = trajectory smoothing on estimated
  paths (L1-optimal, bundled camera paths). Our k-space equivalent: the
  estimated translation path p(t) (phase ramp fit) smoothed/band-split in
  time, correction subtracted consistently across the spectrum.
- Attenuation reference impl: hyue723/Video_Motion_Attenuation (steerable
  pyramid + phase median filtering across neighboring frames).

## Mapping to our engine (design proposal)

1. Estimate the low-band translation path p(t) (phase-correlation/ramp fit)
   - already have this from complex_mc.
2. Instead of PINNING (mc's error), apply TEMPORAL FILTERING to the path:
   subtract only the low-pass (slow, wander) component of p(t); keep its
   faster components. The wander lives in the slow band; legit motion in the
   fast band (assumption to verify by inspecting p(t)'s spectrum!!).
3. Apply the phase correction as a CONSISTENT ramped phase edit across ALL
   spectral cells (or at minimum a whole-band-consistent edit) - the ghost
   failure of lowband-only edits is plausibly a band-inconsistency artifact
   (hypothesis H1, tested in M0.5 E2).

Open checks in flight: M0.5 build (E1 metric floor, E2a broadband-consistent
shift, E3 scale-selective) + Astra consult (DEEP_DIVE_M05_astra.md).

## Meta-lessons from the literature

- Per-SCALE separation (which spatial band the motion lives in) and per-BAND
  temporal separation (slow drift vs fast motion) are orthogonal, both used.
- Phase edits = Fourier-shift-theorem consistent by construction; ghosts
  indicate INCONSISTENT edits (different treatment per band), not phase edits
  per se.
- The field's standard validation = does it LOOK right; our gates (drift,
  ref-SSIM, motion) are stricter - expect to calibrate the gates' meaning for
  translation-type corrections (metric floor measurement E1).
