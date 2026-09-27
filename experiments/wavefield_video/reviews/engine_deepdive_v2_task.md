# Engine deep dive v2 — how to improve the crumb_coherence engine (and the wider model)

You are the R&D architect for crumb_coherence: middleware that keeps AI-generated video coherent
past the 2–5 minute wall (the regime where frontier generators break). Repo: crumb-format
(experiments/wavefield_video). Read IMPL_NOTES_M0_ADDENDUM.md, reviews/claude_m3_gate_task.md,
reports/ENG1_1080.*, DEEP_DIVE_3_opus.md, and THE_POSITION.md first — they are the campaign's
memory. New since those docs: the 3D-first pilot results (below).

## State (as of 2026-09-27, all measured)
- M0 (synthetic): PASSED broadly. Band-selective complex-mode anchoring (r<0.03 band, magnitude
  anchored, phase preserved): hotspot removal 64.0%, fidelity vs clean 0.9946, tracker motion
  99.7%. rho=0.995 slow anchor is load-bearing: 46%→92% drift removal (rho 0.95→0.995).
- M1 (one real generator, self-referential proxies): low-band magnitude variance down 36–86%,
  centroid drift down 7–56%, detail 0.988–0.99, motion intact.
- M2 trap (legit slow motion): FAILED pre-registered. complex_mc destroys legitimate low-frequency
  motion: centroid retention 54.7%/70.8% (bar 95%), trajectory distortion 64.1%/58.9% (bar <5%).
  HF-SSIM ~0.99 → damage is purely low-band phase. Frozen defaults unsafe for customer footage.
- ENG-1 (real 1080p footage): also missed retention bars; ~64% drift removal intact; one clip
  jitter 2.9x vs raw 1.7x.
- M3 (velocity-coherence GATE): implemented, runs tonight. Acceptance: trap >=95% retention / <5%
  distortion both horizons; second unseen legit-motion scene; no regression (hotspot >=60%, M1
  metrics); old behavior reachable via flag. Result pending — design for both outcomes.
- 3D-first pilot (NEW, big): Blender scene -> depth/edges control passes (fixed clip-level bounds,
  no breathing) -> Wan 2.2 Fun-Control (4-step, 832x480). Controlled runs track geometry ~3x above
  uncontrolled (edge recall 0.648 depth / 0.725 edges vs 0.189 gray; uncontrolled = warp/melt
  failure mode). Placement edits (moved armchair) honored; crossed-comparison collapses. Crumb pass
  ON TOP of a generated clip: low-band trajectory variance -68%, flicker -20%. 60s chain (10
  chunks, 25-frame overlap, stitch ramp, style pass, crumb over full clip) = in flight now.
  Thesis under test: "Geometry remembers WHERE the world is; crumb remembers HOW the video behaves."

## Questions (answer each with: expected gain, cost, kill criterion, and how to measure honestly)
1. Beyond the M3 gate. If it passes as specified — what legitimate-motion classes still break it
   (fast pans? occlusion? parallax? multi-object)? Next robustness step with best data support?
   If it fails — plan C beyond "magnitude default + flag"?
2. The engine's deeper evolution. Rank: hybrid gated fusion (easy motion path off the FFT operator),
   temporal memory / self-conditioned rollouts, learned gating (small CNN/MLP on low-band stats vs
   analytic velocity-coherence), conditional kernel. What has real evidence vs fashion? Minimal
   discriminating experiment for each?
3. The stack play. Is the sellable product "3D-control generation + crumb finishing pass"
   (generator-side geometry + middleware-side coherence), with the in-loop engine as the frontier
   for OTHER generators? What breaks it at 60s+ -> 5-10min? How do overlap-stitch + scene resets +
   crumb scale? The doorway-crossing occlusion test (in flight) is probe 1 — what is probe 2?
4. Proof. Beat rolling stats-EMA baseline and an SSM per-byte claim. Design the honest benchmark:
   protocols, metrics (coherence half-life etc.), what third-party blind test convinces a skeptic,
   what to publish first.
5. Prior art to steal from (2026): FreqForcing / Self-Forcing++ / long-video coherence; anything in
   the last 90 days that changes the roadmap (temporal memory in diffusion, spectral/state-space
   hybrids, test-time training for video). Be specific — name methods, say what to take.

Constraints: middleware not filter (correct the generation context, never the exported MP4); anchor
magnitude, preserve phase; no kernel/speed optimization until the mechanism beats SSM on a
memory-requiring task; every claim needs an honest envelope; no photorealism promises.
Answer as dense bullets, <= 2 pages, no fluff. Do not modify any files.

## Deliverable
A ranked recommendation list with kill criteria (your reply text itself; the operator saves it).
