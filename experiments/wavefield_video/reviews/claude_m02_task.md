# M0.2 TASK: phase-anchoring upgrade (the hotspot gap)

Context: read IMPL_NOTES_M0_ADDENDUM and M0_RESULTS.md in this directory first.
M0 status: scenario A (gain-field) PASSES fully - every engine removes 87-92% of
drift, motion 100%, SSIM >= 0.9987. Scenario B (wandering hotspot = phase drift)
fails the 60% gate: the complex engine is the only one that moves it (50.2%
pos-drift removal; stats baseline ~0%, magnitude negative), motion 99.7%,
SSIM 0.9957.

Goal: lift hotspot pos-drift removal from 50.2% to >= 60% WITHOUT:
  (a) motion% dropping below 97,
  (b) SSIM below 0.98,
  (c) breaking any of the 9 invariants in crumb_coherence/tests/test_core.py,
  (d) regressing scenario A (every engine must keep >= 85% on gain_field).

Directions to consider (implement the ONE you judge most promising, cleanly;
justify the pick in the notes; no kitchen-sink combinations):
  1. Masked phase correction: correct phase only where local coherence loss is
     high (hotspot mask), leave globally-stable regions untouched.
  2. Split amplitude/phase anchoring: separate decay constants (phase slower or
     gated) in the anchor update rule.
  3. Multi-scale anchoring: anchor the hotspot at a coarser cell scale (x4) to
     catch slow wander that single-cell tracking misses.
  4. Sub-cell phase-gradient tracking: anchor the phase of the spatial
     derivative to predict wander direction.

Constraints:
  - All changes inside crumb_coherence/ only. Keep the public API stable
    (engine names + their keyword options). numpy only, no new deps.
  - If you add a new invariant, add it to test_core.py.
  - Run: python crumb_coherence/tests/test_core.py (all 9+ invariants) and both
    scenarios via crumb_coherence/scripts/run_m0.py. Paste the full result
    tables into IMPL_NOTES_M0_2.md with an honest per-engine verdict and the
    exact command used to reproduce each table.
  - Keep total CPU runtime under ~3 minutes per scenario.
  - If the goal is not reached, say so plainly and report the best achieved
    number + what you learned; do not fudge metrics.
