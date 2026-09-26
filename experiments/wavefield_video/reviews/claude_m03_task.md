# M0.3 TASK: motion-compensated phase anchoring

Context: read IMPL_NOTES_M0_2.md (the M0.2 falsification) and M0_RESULTS.md.
M0.2 proved: pulling phase toward a lagging anchor degrades monotonically
(50.2% -> 13.3%). The existing complex engine (slow phase EMA) still holds the
best hotspot result at 50.2% pos-drift removal / 99.7% motion / SSIM 0.9957.

Goal: beat 50.2% on hotspot pos-drift removal (target >= 60%) WITHOUT:
  (a) motion% below 97, (b) SSIM below 0.98,
  (c) breaking any of the 12 test_core.py invariants,
  (d) regressing gain_field (all engines keep >= 85% drift removal),
  (e) changing existing engine names' behavior - if you need a new mode, add it
      as a NEW engine name (e.g. complex_mc) so old names stay byte-identical.

Two candidate principles (pick ONE, justify; the first is preferred):
  1. MOTION-COMPENSATED ANCHORING: estimate the inter-frame low-band shift via
     phase correlation (cross-power spectrum of successive frames, sub-pixel via
     parabolic peak fit), de-shift the current low-band by that estimated motion,
     THEN compare the de-shifted frame to the anchor - the residual is drift, not
     motion - and correct ONLY that residual. This separates "balls moving"
     (consistent frame to frame) from "bump wandering" (accumulates vs anchor).
  2. SPLIT MEMORY: keep one decay for amplitude, a deeper one for phase (no new
     correction term), and re-tune rho for the phase path.

Constraints:
  - All changes in crumb_coherence/ only; numpy only; no new deps; keep the
    scenario runner's public flags stable (add, don't remove).
  - Run: test_core.py (12+ invariants), both scenarios. For hotspot, include a
    small sweep over any new knob you add (0, low, mid, high) to show the shape.
  - Write IMPL_NOTES_M0_3.md with: design, full result tables, the exact
    commands, and an honest verdict. If <60% is the truth, say so plainly.
  - Keep CPU runtime under ~3 min per scenario run.
