# IMPL NOTES — R15 hypercomplex arms (E1 qssm / E2 qwave / E3 qcolor)

Task: t_7877bc01 "implement + run the top-ranked hypercomplex arms from the parent
study". Authority for design: `reports/HYPERCOMPLEX_STUDY.md` (§4-E1/E2/E3, §5
protocol). This note is the campaign-memory record for the round; the suite results
section is appended as the box run lands.

## What was built (all additive; default paths byte-identical, asserted)

| arm | study | integration | file |
|---|---|---|---|
| E1 `qssm` | §4-E1 | new `--kind qssm` | `ssm_quat.py` (`QuatSSM`) |
| E2 `qwave` | §4-E2 | flag `--q-mix` on `--kind wave` | `wfvideo_quat.py` (`WaveQuatMix`, `QuaternionLinear`) |
| E3 `qcolor` | §4-E3 | flag `--quat-color` on `--kind wave` | `wfvideo_quat.py` (`QuatEmbed`) |

- `train_compare.py`: `--kind qssm` choice; `--q-mix` (BooleanOptionalAction) and
  `--quat-color`; flag-scope + divisibility validation fails fast at argparse;
  result JSONs now record `q_mix` / `quat_color`.
- `wfvideo.py` (`VideoPredictor`): `q_mix`/`quat_color` kwargs, `qssm` dispatch,
  `qssm` added to `_streamable()` (occlusion eval's `streaming` probe is capability
  based, so qssm streams through the O(1) state path).
- `run_occlusion.sh`: arms F_qssm / G_qwave (`--q-mix`) / H_qcolor (`--quat-color`)
  appended (A–E untouched); `CONST_LR=1` env appends `--const-lr` to BASE;
  `RESUME=1` sliced-queue mode (skip finished arms, `--resume` from ckpt, per-arm
  wall cap `SLICE_S`, stops at the boundary with status `SLICED done/units`);
  the legacy `run_longbudget.sh` prelude is now opt-in (`RUN_LONGBUDGET=1`) so a
  suite invocation can never re-fire the historical one-off run.
- Tests: `test_hypercomplex_r15.py` (new), `run_smoke_r15.sh` (new),
  `test_occlusion_runner_r14.py` updated to the 8-arm contract + slice-mode cases.

## Verified invariants (all green locally AND on the box venv, 2026-09-27)

- `qmul`/`qmul_basis` == explicit 4x4 left-multiplication matrices; norm-multiplicative.
- `QuatSSM.state_bytes == SSMLite.state_bytes` (the per-byte premise): checked at
  dim/heads (32,4) and (192,8). qssm state = `[B,HW,nh,dh//2,ds,4]` f32.
- qssm forward (FFT-conv training path) == `step()` recurrence chain:
  maxdiff 2.4e-6 (fp32). fp64: conv == Lmat recurrence 6.8e-7.
- FFT conv == explicit `sum_k K[t-k] ⊗ u_k` from the module kernel: 6e-7–1.2e-6.
- `QuaternionLinear`: exact param formula `nh*slots^2*4 + dim`; scalar-rotation
  identity; per-slot Hamilton product vs dense reference.
- `WaveQuatMix`: pi/po quaternion (1/(4*nh) of Linear params), state bytes equal
  to `WaveMix3D`, forward==step parity at the same level as the wave arm itself
  (1.1e-2 vs wave ref 4.7e-3 at the toy config — the wave family's fp32 level).
- `QuatEmbed`: `9*dim + dim` params; manual tap-by-tap Hamilton conv match.
- Default (flag-off) `VideoPredictor` init is byte-identical: same-seed
  state_dict sha256 equality, `WaveMix3D`/`Conv2d` types unchanged.
- CLI validation cases (flag scope, `--q-mix`+`--fuse`, divisibility) fail fast.
- Full local + box smokes pass: `bash run_smoke_r15.sh` prints `smoke r15 PASS`
  (includes a micro train+occlusion-eval of qssm/q-mix/quat-color and
  `qssm.state_bytes == ssm.state_bytes` on the trained arms' result JSONs).

## Deliberate deviations (documented)

1. **qssm trains via the FFT-conv path, not the study's "direct unrolled recurrence
   over T=17".** Rationale: (a) it mirrors `SSMLite`'s own training path exactly, so
   the per-byte comparison holds at equal resource profile (mem_gb_peak, wall); (b)
   numerically validated equivalent to the unrolled recurrence (fp64 6.8e-7, fp32
   2.4e-6) — the study's own invariant #1; (c) an unrolled recurrence would hold
   ~17x the state tensors per layer in the autograd graph for zero accuracy gain.
   No custom HR-calculus anywhere — plain autograd on real component-axis ops, per
   the study's intent. Streaming rollout still uses the true recurrence (`step()`).
2. `QuatFFN` (optional `--q-ffn`, study §4-E2) is in-tree but **not wired** — the
   arm list does not use it; wiring it would touch `Block`'s FFN construction.
3. `render_rollout.py` kind list not extended for qssm/q-mix (suite eval is in
   `train_compare.py`); eye-candy rendering for the new arms is a follow-up.

## The run (box `root@161.184.224.50:41400`, queue via the conductor)

Recipe per study §5.2 is mechanized: `run_occlusion_job.sh` reads the pre-flight
probe's `copy_ratio` ONCE from `runs_occlusion/result_wave_r15_preflight.json`,
pins it to `runs_occlusion/r15_recipe.txt`, and every slice uses the pinned recipe:

- `copy_ratio < 0.3` (ref: frozen band ~0.02–0.1 vs unfrozen ~0.73) -> escalate
  `CONST_LR=1 STEPS=8000` for **all eight arms** (A–H in one logical suite run);
- otherwise -> `CONST_LR=1 STEPS=2000`.

Queued jobs (pending/, lexicographic order after the pre-existing jobs; conductor
runs one at a time, jobs self-cap at `SLICE_S=4200s` under the conductor's 5400s):

- `gpuq_job_200_r15_boxsmoke.sh` — box-side smoke of the pushed file set.
- `gpuq_job_201a/b/c_r15_preflight.sh` — the 2k `--const-lr` wave probe (3 copies:
  skip-when-done, resume-from-ckpt).
- `gpuq_job_202..461_r15_slice.sh` — 260 identical sliced-suite jobs; the runner is
  self-organizing (skip done arms, resume first incomplete, slice wall), so
  trailing copies no-op once `occlusion_status.txt` reads `DONE`. Queue more copies
  if the matrix outlives them (8 arms x 5 seeds at 8k steps is ~200+ slices).

Ops notes: results in `runs_occlusion/result_<kind><tag>.json` (verdict fields:
`copy_ratio`, `divergence_horizon`, `persistent_state_bytes`, `state_bytes_total`,
`ms_per_frame`, `mem_gb_peak`; read the printed `div_horizon/KB_state`); progress in
`occlusion_progress.txt` / `occlusion_status.txt`; slice/probe logs under
`/workspace/slava/logs/gpuq_job_*_r15_*.log`; keep `STEPS`/`CONST_LR` fixed across
slices of one `OUT` (the pinned recipe file enforces this).

Slicing deviates from "one invocation" (§5.1) only mechanically — the conductor's
5400s cap forces it; same script, same OUT, same seeds, resume-chained, one logical
run. Recorded here per §5.2 ("record the choice and the probe numbers").

## Results (append as they land)

- Commit: `0c87976` (branch `crumb-llm-standalone`, pushed 2026-09-27).
- Deployment: file set md5-verified on the box; jobs queued 2026-09-27 (pending band
  `gpuq_job_200..461`, ordered after the pre-existing P4 jobs 13–19).
- Box smoke (job 200): _pending_
- Pre-flight probe copy_ratio: _pending_ -> recipe: _pending_
- Suite: _pending_ (status file + result JSONs on the box)

## Check commands

```
ssh ... -p 41400 root@161.184.224.50 \
  'tail -6 /workspace/slava/exp/wavefield_video/runs_occlusion/occlusion_progress.txt; \
   cat /workspace/slava/exp/wavefield_video/runs_occlusion/occlusion_status.txt; \
   cat /workspace/slava/exp/wavefield_video/runs_occlusion/r15_recipe.txt 2>/dev/null; \
   ls /workspace/slava/exp/wavefield_video/runs_occlusion/result_*.json 2>/dev/null | wc -l'
```
