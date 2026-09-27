# ENGINE_D1_SWEEP — residual anchoring: acceptance record (RED / killed)

Task t_3e1d4c9f. Pre-registered gates: trap retention ≥95% + distortion <5% (96 AND 192 frames)
AND hotspot drift ≥60% removal, co-located in ONE config. Source of truth: `out/sweep_d1/`.

## Round 1 — grid wipeout (302 s, 360 runs, box-verified parity)

Grid: window {1,2,4,8,16,32} × strength {0.1,0.25,0.5,1.0} × cutoff {0.0,0.05,0.25,1.0,4.0};
scenes translate:96, translate:192, hotspot:48. Driver: `scripts/run_sweep_d1.py`.

- trap96 distortion < 5%: **0 / 120 configs** (grid minimum: **61.24%**).
- hotspot drift removal: **max 51.2%** (bar 60%); grid range 40–51%.
- **CO-PASS configs: NONE.**

Frontier (from `analyze_sweep.py`): trap96 [61.2, 71.5] ↔ hotspot [0.8, 51.2]; the min-dist corner
(61.24%) is a near-no-op config (w1 s1 c4, f≈0 — corrections almost fully attenuated). Hotspot
leaders (51.0–51.2%) sit at equally attenuated corners (s=0.1/0.05 cutoffs, w1–w32). No region of the
grid trades the two axes usefully.

## Floor controls — why the kill is structural (`out/sweep_d1/controls/`)

| control | trap96 dist | trap192 dist | hotspot drift |
|---|---|---|---|
| residual no-op (s=0.001) | 61.72% | 51.39% | 50.2% |
| D1 grid min→max | 61.24% → 71.5% | 51.1% → 68.6% | 40.2% → 51.2% |
| legacy complex_mc (frozen default) | **6.33%** | **57.62%** | 64.0% (M0) |
| pristine HEAD vs D1, default path | 118.6% / 6.33% — identical (no-op verified) | | |

1. The D1 family's entire reachable set hugs the no-op floor (61.7% on tr96) — the trap metric's
   floor = raw-vs-clean centroid deviation of the illumination-corrupted scene; corrections at every
   usable strength only add damage above it. The <5% bar is unreachable for any correction that
   behaves like this family.
2. On the hotspot, the fixed blend carries ~50% removal by itself; the residual shift adds ≤1pp.
   The legacy 64% lives in the anchor-pull (−0.5·dy toward the slow EMA) — the predictable, slow
   component D1 deliberately refuses to correct.
3. Mechanism inversion: drift here is the *predictable* slow signal; innovation is *estimator noise*
   (2–6px feature-switching residuals on the trap trace). D1 removes the useful pull and redistributes
   noise: no-op 61.7% → D1-full 70.8% → legacy 6.33%, same scene, same metric.

No rounds 2–3: round 1 covers every corner of the knob space; the floor analysis (1–3) shows no grid
can cross a structural floor. **Pre-registered outcome executed: residual mode stays default OFF
(code in-tree behind `mc_residual`), no further estimator/gate refinements on this axis; product =
triage concierge + authored-motion lane, magnitude-mode envelope as the claim boundary.**

## Box (24/7 engine-lab)

- `crumb_coherence/` shipped to `/workspace/slava/exp/wavefield_video/`; md5 parity verified
  (core.py `0d48d08d651bb6cc81b13bee2c256e08` both ends; run_trap/run_m0/run_sweep_d1 match after
  the print-fix re-ship).
- Remote smoke (translate:96, w8 s0.5 c0.25): **identical** to local — 736.2% / 70.81% / 0.9940 /
  0.508, max dev 10.117px (box python: `/workspace/slava/comfy-house/venv/bin/python`, torch 2.6.0+cu124).
- Filler family wired: `gpu_queue/generators/{engine_lab_sweep,engine_lab_eval,engine_lab_corpus}.sh`
  + entries in `gpu_queue/generators.json` (specs under `engine_lab/specs/*.args`, outputs under
  `engine_lab/sweeps|evals|corpus`). Verification job queued via `gpu_queue/pending/` (manual
  injection — the daily filler cap was already saturated when wiring landed).

## Files

- Driver + evidence: `scripts/run_sweep_d1.py`, `out/sweep_d1/sweep_d1_rounds.csv`,
  `out/sweep_d1/manifest_round1.json`, `out/sweep_d1/controls/*.json`
- Analyzer (scratch): `~/.hermes/workspaces/engine_lab/analyze_sweep.py`
- Full writeup: `IMPL_NOTES_M4.md`
