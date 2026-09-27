# IMPL_NOTES_M4 — D1 residual anchoring: build + sweep verdict (KILL)

**Scope:** t_3e1d4c9f — implement the pre-registered D1 mechanism (residual anchoring via alpha-beta
constant-velocity prediction; correct only the unpredicted residual, strength scaled by residual
significance), wire it into the harnesses, sweep it against the M3 acceptance gates, and repoint the
box filler family. **Date:** 2026-09-27. **Branch:** crumb-llm-standalone.

## What D1 is (pre-registered)

- Anchor to an alpha-beta trajectory prediction (constant-velocity prior) instead of the raw EMA anchor.
- Correction = −strength · f(|residual|) · residual, where residual = measured shift − predicted shift.
- f(|r|) = smoothstep(0, cutoff, |r|): small innovations are treated as noise and not corrected.
- Constant-velocity motion ⇒ residual ≈ 0 ⇒ motion survives "by construction".
- Kill criteria (pre-registered): trap retention ≥95% + distortion <5% at 96/192 frames AND hotspot
  drift ≥60% removal, co-located in one config, within 3 sweep rounds.

## Build (as landed)

`crumb_coherence/core.py` (md5 `0d48d08d651bb6cc81b13bee2c256e08`):
- New `CoherenceState` fields + kwargs: `mc_residual` (default **False**), `mc_res_window` (8),
  `mc_res_strength` (0.5), `mc_res_cutoff` (0.25); validation `mc_res_strength >= 0`.
- `_residual_shift(dy, dx, state)`: alpha-beta update (α=2/(W+1), β=α²/(1+α)... per impl), innovation
  `r = measured − predicted`, scaling `f = smoothstep(0, cutoff, |r|)`, shift `= −strength·f·r` applied
  through the same band-masked (`mc_band`) phase path as the legacy shift. Confidence-weighted when
  `mc_smooth` conf is available.
- Residual branch in `_measure_update`: when `mc_residual` on, the blend weight is also gated by `f`
  (significance-gated blend), shift = `−mc_res_strength·f·r`; flags off ⇒ byte-identical behavior
  (verified: pristine HEAD vs D1 both give tr96 = 118.6% / 6.33% on the default path).

Harness wiring (`run_trap.py`, `run_m0.py`): additive flags `--residual`, `--res-window`,
`--res-strength`, `--res-cutoff` (`--hotspot-mc-residual`, `--hotspot-only-residual`,
`--hotspot-res-*` for m0), residual row + JSON fields. Sweep driver: `scripts/run_sweep_d1.py`
(resumable, CSV `out/sweep_d1/sweep_d1_rounds.csv`, manifests per round).
Invariants: `engine_lab/smoke_d1.py` — const-velocity residual decays to 3e-05 at t=79; single +0.3px
jump flags f=0.65 (test artifact fixed by stepping frames consecutively). pytest: 3 failures, all
pre-existing on pristine HEAD (float-equality nit etc., not D1).

Harness bugs found and fixed en route: (1) `cfg['mc_res_*']` print KeyError for partial flag sets —
fixed with `.get(..., default)`; (2) earlier over-eager sed fixed with an assertion-guarded rewrite.
Both were display-path only; round-1 numbers were computed with all flags present and are valid. (3) post-deploy: the driver's final CSV rewrite dropped prior rounds on multi-round appends — fixed + before/after tested (round-1 evidence untouched: single-round file).

## Sweep (round 1, 120 configs × 3 scenes = 360 runs, 302s local)

`window ∈ {1,2,4,8,16,32} × strength ∈ {0.1,0.25,0.5,1.0} × cutoff ∈ {0.0,0.05,0.25,1.0,4.0}`;
scenes: translate 96/192 (trap), hotspot 48 (removal). Evidence: `out/sweep_d1/`
(CSV + `manifest_round1.json` + controls in `controls/`).

**Round 1 = decisive wipeout:**
- Trap96 distortion < 5%: **0 / 120 configs** (minimum over the whole grid: 61.24%).
- Hotspot drift removal: **max 51.2%** (bar 60); the whole grid is 40–51%.
- CO-PASS configs: **NONE**.
- Frontier: min d96 (61.2%) sits at near-no-op configs; every usable strength/cutoff region is ≥61%.

**Floor controls (the mechanism-level result):**
| control | trap96 dist | trap192 dist | hotspot drift |
|---|---|---|---|
| no-op residual (s=0.001) | 61.72% | 51.39% | 50.2% |
| D1 grid min / max | 61.24% | 51.1% | 40–51.2% |
| legacy complex_mc (frozen default, no residual) | **6.33%** | **57.62%** | 64.0% (M0 record) |

## Verdict: KILL — pre-registered outcome triggered

1. **No config co-passes; no config comes within an order of magnitude on the trap.** The grid's
   minimum track distortion equals the no-op floor (61.7%): corrections at every usable strength only
   ADD damage to the trap; attenuating them to floor still fails the <5% bar because the floor itself
   is the raw-vs-clean deviation of the illumination-corrupted centroid.
2. **The hotspot removal was never the residual's to give.** At no-op the fixed 0.5-blend gates ≈50%
   removal by itself (magnitude of the blend carries it); the residual shift adds ≤1pp anywhere in the
   grid. The legacy 64% lived in the *anchor-pull* shift (−0.5·dy toward a slow EMA) — exactly the
   predictable, slow component D1 refuses to correct.
3. **The premise inverts the useful correction.** On these traces the drift is the *predictable* slow
   component (illumination sinusoid / anchor-lag ramp) and the innovation is *estimator noise* (feature
   switching, regime jumps, the measured 2–6px residuals). Removing the predictable part and
   distributing the noise makes D1 systematically worse than the legacy pull — demonstrated: no-op
   61.7% → D1-full 70.8% → legacy 6.33% on the same scene.
4. **Anchoring the residual cannot be rescued by knob choice** (window/strength/cutoff swept jointly;
   the family's entire reachable set is [61.2, 71] dist / [40, 51.2] removal). No amount of grid
   refinement moves a structural floor.

**Outcome per pre-registration:** residual mode stays **default OFF** (as landed); the code remains
in-tree behind `mc_residual` for the record; `run_sweep_d1.py` + controls are the audit trail.
No rounds 2–3: the pre-registered rationale is "no config co-passes within 3 rounds"; round 1 was a
total wipeout at every corner AND the mechanism-level floor analysis (2/3 above) shows further grids
cannot cross a structural floor. Product statement stands: **content-blind rescue is not the honest
first sell; triage-form concierge + authored-motion (3D-control) lane remains the flagship**;
magnitude-mode envelope is the claim boundary.

## Box (24/7 engine-lab)

- Code parity verified: `crumb_coherence` shipped to `/workspace/slava/exp/wavefield_video/`,
  md5s identical both ends (core.py `0d48d08d…`; run_trap/run_m0 match), remote smoke run =
  **identical numbers** to local (736.2% / 70.81% / 0.9940 / 0.508; box python =
  `/workspace/slava/comfy-house/venv/bin/python`, torch 2.6.0+cu124).
- Engine-lab filler family deployed + **verified in place** on the box (2026-09-27): all three
  generators emit valid jobs; all three payloads ran rc=0 (receipts under `engine_lab/verify/`);
  registered as first-class gpuq generators with supervisor drift-repair (Mac source
  `~/.hermes/scripts/box_gpuq/`). No job left pending (queue saturated; fillers fire on idle
  gaps). The earlier "wired + queued" wording in this section was corrected after the on-box
  verification — see `reviews/ENGINE_D1_SWEEP.md` §Box for the full record.

## Files

- `crumb_coherence/core.py` (D1, default OFF) • `scripts/run_trap.py`, `scripts/run_m0.py` (wiring + print fix)
- `scripts/run_sweep_d1.py` (driver) • `out/sweep_d1/{sweep_d1_rounds.csv,manifest_round1.json,controls/*}`
- Scratch (not shipped): `~/.hermes/workspaces/engine_lab/` (smoke_d1.py, repro_m0.py, probes, fixer scripts;
  box-verify harnesses under `box_verify/`)
- Box: `~/.hermes/scripts/box_gpuq/` (engine-lab generator family + registry; supervisor-wired) •
  box receipts: `exp/wavefield_video/engine_lab/verify/SUMMARY.md`
