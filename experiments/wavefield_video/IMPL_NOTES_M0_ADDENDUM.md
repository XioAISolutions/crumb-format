# IMPL_NOTES_M0_ADDENDUM (2026-09-26) - owner-run results, rho fix, hotspot, metric caveats

## 1. rho fix (THE finding)
M0 first run: all engines ~46-51% drift removal (pinned). Root cause: anchor EMA
rho=0.95 lags a ramping drift -> anchor itself tracks ~half the drift -> correction
bounded at ~half. rho=0.99 -> 85-88%; rho=0.995 -> 90-92%; rho=0.999 -> 92-94%.
Defaults patched: core.py rho default 0.95 -> 0.995; run_m0.py --rho default 0.995.

## 2. gain_field scenario, final numbers (rho=0.995, 48f 64x64, peak+centroid metrics)
- removal: magnitude 91.7% var / 87.0 dE; baseline 89.5 / 87.4; complex 89.8 / 86.8
- hf_ssim: 0.9987-1.0000 all engines.
- motion caveat: centroid metric ~89-90% path preserved; peak_path metric 54-74%
  (argmax jumps between balls inflate path loss - metric noisy). Precise motion
  accounting needs a multi-blob tracker (semantic_metrics-style). DONE-pending.
- spectral magnitude edges baseline (91.7 vs 89.5 var) + wins strongly on hotspot.

## 3. hotspot scenario (NEW: mean-preserving wandering low-freq bump)
First run: baseline 6.2% var removal vs spectral magnitude 30.7% vs complex 45.8%
= 7x separation in the designed direction (stats blind, spectral tracks).
Issues: removal < 60 gate; magnitude/complex negative dE (color artifacts on
mean-preserved content); motion metric contaminated by the bump itself.
=> M0.1 task: refine hotspot (stronger anchor weighting / delta-band targeting),
fix dE artifact, tracker-based motion; re-gate.

## 4. Command log (all CPU)
python crumb_coherence/scripts/run_m0.py --frames 48 --grid 64 --no-video [--scenario hotspot] [--rho R]
bash run_m0_smoke.sh (uses system python - run with ~/vibevoice-env/bin/python instead)
