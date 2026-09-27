# M3 — the GATED engine (make coherence correction safe around legitimate motion)

Same session, continuing the crumb_coherence arc. Read IMPL_NOTES_M2.md and reports/ENG1_1080.md
first — both just landed.

## Where we are
- M2 trap FAILED (pre-registered, confirmed on both horizons): the frozen complex_mc destroys
  legitimate low-frequency motion — centroid retention 54.7%/70.8% (need >95%), trajectory
  distortion 64.1%/58.9% (need <5%), HF-SSIM fine (~0.99). The engine cannot tell a translating
  subject from wander.
- ENG-1 (1080p real footage) — its report has the real-resolution numbers; read them.
- This is the ONLY thing between us and a sellable coherence pass. Fix it.

## Goal
Implement the GATE: complex_mc correction must apply only where low-band motion looks like
INCOHERENT WANDER, and pass COHERENT legitimate motion essentially untouched. Frozen-corrected
defaults must pass the trap after this change.

## Candidate mechanisms (pick what the data supports; combine if needed)
1. Velocity-coherence gating: estimate the low-band displacement trajectory; wander has low
   directional autocorrelation / random-walk signature; coherent motion has smooth consistent
   velocity. Gate the correction strength per-frame (or per-region) on that signature.
2. Extend the existing _mc_confidence machinery with a drift-vs-motion classifier.
3. If no gate can separate them cleanly, ship magnitude-mode as the default for general content
   and keep complex_mc behind an explicit flag — and say so in the notes. Honest fallback beats
   an overfit gate.

## Acceptance (all must hold; report numbers)
1. run_trap.py BOTH horizons: centroid-motion retention >= 95%, trajectory distortion < 5%,
   HF-SSIM >= 0.98, low-band var ratio in [0.90, 1.10].
2. Do NOT overfit to the trap scene: add a second legit-motion scene (e.g. accelerating object,
   or a turning/curving path) and require the same retention there.
3. Regression: run_m0.py hotspot removal >= 60% still (this is what the correction exists for);
   test_core.py invariants all pass (update them for the gate); run_m1_integration.py metrics
   preserved (low-band variance reduction >= 50%, detail >= 0.98).
4. The old frozen behavior must remain reachable via an explicit flag (for the record).
5. IMPL_NOTES_M3.md: what the gate is, why it separates the cases, all numbers, and the honest
   envelope of what is now safe to claim.

## Then
If all green: bump the plugin version, commit everything, and update THE_POSITION.md's claim
boundary section (what we can honestly say to a customer now). This is the round that decides
whether the rescue offer can open.
