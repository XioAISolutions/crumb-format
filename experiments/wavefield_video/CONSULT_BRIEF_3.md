# CONSULT BRIEF 3 (2026-09-26): freeze root-cause + PLUGIN path for 2-5 minute videos

## New hard data
- Retune matrix (g32, wave, 2000 steps): ALL variants freeze. K=1 arms: copy_ratio 0.005 (pure copy), tail-MSE 0.016, divergence "passes" — frozen is METRIC-OPTIMAL. Ramp arms: copy_ratio 0.116 (a trace of motion) but tail 0.94 (drift). Objective patches (resid-balanced / motion-weighted) = ZERO effect.
- GEOMETRY CONTROL FAILED: radius 3.2 + speed 2.3 (moving-fraction 0.09 -> 0.55): STILL froze (copy_ratio 0.006, tail 0.0585). Freeze survives scene rescale.
- So the earlier "metric x data-geometry" story is incomplete. New prime suspect: TRAINING BUDGET PER TOKEN at g32 = 17,408 tokens/seq vs g16 = 4,352, SAME 2000 steps => 1/4 the updates per token. g16 (motion OK, copy_ratio 0.72 with same objective) vs g32 (freeze) confounds grid with per-token budget.
- Context: streaming recurrence verified (1,024 frames, flat 0.207GB, ~2,400 fps). Latent 8x8: both arms fade to black. Difficulty ladder (6 rungs) + occlusion memory-probe (5 arms x 5 seeds) built and queued. FreqForcing result: low-freq spectral anchoring fixes AR long-video drift; our state IS k-space.

## Questions
1. FREEZE ROOT-CAUSE: is the budget-per-token confound the most likely driver? Design the decisive experiment suite (we plan: same config, 8,000 steps at g32; what controls make it airtight?). Any other suspect we are missing (e.g., kernel param init at larger N, normalization drift with 4x tokens, LR schedule mismatch)?
2. ARCHITECTURE: rank the next 3 changes with exact specs (from your earlier doc: conditional lambda_t, hybrid gated fusion, self-conditioned rollouts). Which ONE first given the freeze data, and why?
3. PLUGIN PATH (the product ask): we want crumb as a PLUGIN that gives ANY video pipeline 2-5 minute coherence. Our proposal: a 'spectral coherence engine' — rolling wave-state over low-frequency bands; each incoming generated segment gets its low-band content blended toward the anchored state (Gaussian blend, FreqForcing-style, but in plain pixel/FFT space with our recurrence), state = tiny. Tear this apart: correct interface? What must be true for it to work? Better alternative plugin shape? MVP milestones + kill criteria for a 2-5 minute demo."
