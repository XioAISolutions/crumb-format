# IMPL_NOTES M0.2 - phase anchoring round (owner-run, 2026-09-26)

## What was built
Additive `phase_anchor` kwarg on SpectralCoherenceEngine + flat phase_band_weight
(zero at DC). Only the hotspot `complex` engine wired to it. +3 invariants (12
total, all passing, incl. "phase_anchor=0 is byte-identical no-op").

## Verification (all run by owner after the build)
- Invariants: 12/12 pass.
- gain_field: no regression - every engine PASS [DSM], magnitude 91.7%/100.6%.
- hotspot sweep (complex engine, pos-drift removal):
    pa=0.0  -> 50.2%  (byte-identical to M0.1, confirms the no-op invariant)
    pa=0.15 -> 44.3%
    pa=0.3  -> 33.1%
    pa=0.6  -> 18.3%
    pa=1.0  -> 13.3%  (was the accidental default; also hf_ssim dropped to 0.9386)

## Verdict: FALSIFIED - flat phase pulling toward the anchor is counterproductive.
Monotonic degradation with strength: the lagging anchor itself wanders, so pulling
phase toward it fights the true motion signal instead of the drift. Consistent with
the DD3 insight "amplitude corrected strongly, phase barely". The kwarg stays (0.0
default, no-op) as scaffolding; the mechanism is rejected.

## Next (M0.3 candidates)
1. Motion-compensated anchoring: estimate the inter-frame low-band shift (phase
   correlation), de-shift the current frame by it (motion), THEN compare to anchor
   so the residual = drift only; correct drift only.
2. Split phase/amplitude memory: stronger (deeper) phase memory than amplitude,
   no new correction term.

Reproduce: python crumb_coherence/scripts/run_m0.py --scenario hotspot --hotspot-phase-anchor <v> --no-video
