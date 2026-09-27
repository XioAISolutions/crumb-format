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

## Box (24/7 engine-lab) — family deployed + verified in place (2026-09-27)

- `crumb_coherence/` shipped to `/workspace/slava/exp/wavefield_video/`; md5 parity verified
  (core.py `0d48d08d651bb6cc81b13bee2c256e08` both ends; run_trap/run_m0 match).
- Remote smoke (translate:96, w8 s0.5 c0.25): identical to local — 736.2% / 70.81% / 0.9940 /
  0.508, max dev 10.117px (box python: `/workspace/slava/comfy-house/venv/bin/python`, torch 2.6.0+cu124).
- Engine-lab filler family deployed as first-class gpuq v2 generators (not a one-off injection):
  `gpuq/generators/engine_lab_{sweep,eval,corpus}.sh` + registry entries in `gpuq/generators.json`
  (order appended after the two content fillers; enabled, bounded caps). Mac source of truth
  `~/.hermes/scripts/box_gpuq/` — supervisor FILES, status.sh md5 list and box_push manifest all
  carry the family, so drift is auto-repaired every 10 min and it survives box rebuilds
  (registry md5 `6a4e4b69aead7387a05056df40c0860a`).
- Verified in place on the box (2026-09-27 13:16–13:19Z; receipts in
  `/workspace/slava/exp/wavefield_video/engine_lab/verify/`): every generator emits a job file
  (bash -n clean) and every payload ran rc=0 — sweep 6 runs → `sweeps/sweep_d1_rounds.csv` (spec
  archived to `specs/done/`), eval 5 runs → `evals/20260927_131632/`, corpus 8 runs →
  `corpus/20260927_131833/` (4 motion classes × {96,192}). Back-off paths return rc=3 ("no sweep
  spec queued" / "no eval.args staged"), matching the conductor contract checked in gpuq2.py.
- Corpus design fix (found by the in-place run): `--seed` is numerically inert for the tracked
  object (6 seed repeats give identical metrics; only the background balls vary), so the corpus
  diversity axis is motion class × horizon; the seed-based draft is superseded.
- No verification job was left in `pending/`: the queue is saturated for hours and fillers exist
  for idle gaps — the family fires naturally when the queue empties with the other fillers
  unavailable (the earlier "wired + queued" wording in this section was an interim overclaim,
  corrected here after the on-box verification).
- Also fixed while deploying: `run_sweep_d1.py`'s final CSV rewrite wrote only the current round's
  rows, silently dropping prior rounds on multi-round appends (demonstrated: run tA then tB → CSV
  kept tB only). Fix writes the merged set; before/after test green; box copy md5-matched
  (`4da51957a89901fea8b5a00d2d8d8e9c` both ends).

## Files

- Driver + evidence: `scripts/run_sweep_d1.py`, `out/sweep_d1/sweep_d1_rounds.csv`,
  `out/sweep_d1/manifest_round1.json`, `out/sweep_d1/controls/*.json`
- Analyzer (scratch): `~/.hermes/workspaces/engine_lab/analyze_sweep.py`
- Full writeup: `IMPL_NOTES_M4.md`
- Box family sources: `~/.hermes/scripts/box_gpuq/generators/engine_lab_{sweep,eval,corpus}.sh`
  + `generators.json`; receipts on the box: `exp/wavefield_video/engine_lab/verify/`
- Verification harnesses (scratch): `~/.hermes/workspaces/engine_lab/box_verify/`
