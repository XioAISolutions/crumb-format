# R14 — WaveMemory long-memory occlusion probe (the decisive experiment)

R14 builds the one experiment HARDENING_astra.md ("The experiment that matters
most") argues is worth more than fifty bouncing-ball MSE runs: a scene where the
target object disappears for far longer than any local attention window and then
re-emerges on a trajectory fully determined by its _pre-occlusion_ state. Only a
model that carried the hidden object's dynamical state can predict the emergence.
The headline is **divergence-horizon-per-byte-of-persistent-state**. No commits.

The falsifiable claim (HARDENING §"What you're missing"): _a structured recurrent
spectral state retains useful visual dynamics over substantially longer horizons
per byte of persistent state than finite-context attention or a generic SSM._

## Kill criterion (item 5)

If **local+wave (E)** does not beat **local+generic-ssm (D)** on divergence-
horizon-per-persistent-byte, the wave formulation is not pulling its weight — stop
polishing it because it is elegant. Both hybrids carry the _same_ local-attention
window, so the comparison isolates wave-state vs generic-SSM-state.

## The scenario — `data_occlusion.py`

64×64, `nb=8` balls. Ball `TARGET_IDX = 0` is the **target**, drawn in a unique
identity hue (the pure red palette anchor). The other seven roam the full canvas
(always-on motion, so freezing never wins).

- The target is **confined to a chamber rectangle** — it bounces off the chamber
  walls, so it is always inside that region and its physics stay deterministic
  (wall bounces included) behind the occluder.
- The **occluder** is a static rectangle covering the chamber (plus a `3·radius`
  pad). It is composited **opaquely** over the frame only on frame indices
  `[occ_start, occ_end)` (default `[64, 320)`). Because the target never leaves the
  chamber and the occluder covers it, the target is hidden **exactly** on that
  window and visible otherwise — a crisp, checkable invariant.
- **Exit direction at emergence** = `sign` of the target's velocity at `occ_end`,
  a deterministic function of the initial state and the bounces during occlusion.

Same API/RNG discipline as `data.py`: a local CPU `Generator` seeded by `seed`
(never touches global RNG); draw order pos → vel → color-jitter; returns
`[B, T+1, 3, H, W]` in `[0,1]` with optional `return_meta` / `return_moving`. Meta
adds `target_idx`, `occluder` rect, `occ_start/occ_end`, per-frame `vel`, and a
`visible` mask. Occlusion window is overridable via `occ_start`/`occ_end` (used
small in the CPU smoke) and wired from the trainer's `--occ-start/--occ-end`.

At 64×64 the occluder is a proper partial band (`x∈[11,52], y∈[16,47]`); at the
tiny 16×16 smoke grid the pad makes it fill the canvas — fine for plumbing, since
the smoke tests correctness of the pipeline, not trained accuracy.

## The arms — `run_occlusion.sh`

Five arms at an equal ~4M budget (ffn_mult bisected to `--target-params 4000000`
for each, so the two-mixer hybrids are held to the same budget), each over 5
training seeds (25 jobs), 2000 steps, evaluated on a 1024-frame occlusion rollout:

| Arm           | flags                                                       | rollout path       | persistent state              |
| ------------- | ----------------------------------------------------------- | ------------------ | ----------------------------- |
| A local-attn  | `--kind attn`                                               | windowed `m(win)`  | none (keeps a T-frame window) |
| B generic-ssm | `--kind ssm`                                                | streaming `step()` | SSM recurrence                |
| C wave        | `--kind wave --kernel-version dispersion`                   | streaming `step()` | wave recurrence               |
| D local+ssm   | `--kind ssm --fuse local_ssm`                               | streaming (hybrid) | SSM recurrence                |
| E local+wave  | `--kind wave --kernel-version dispersion --fuse local_wave` | streaming (hybrid) | wave recurrence               |

### Gated fusion wrapper — `wfvideo.py` `FusedMix`, `--fuse {none,local_ssm,local_wave}`

`h = g·h_local + (1−g)·h_global`, `g = sigmoid(Linear([h_local; h_global]))`
(HARDENING Rank 2). The local path is **causal** windowed attention (finite
context = T frames); the global path is the O(1)-in-T recurrence (SSM or dispersion
wave). Streaming: the global path advances its recurrence; the local path keeps a
rolling T-frame buffer and recomputes causal attention. **Only the global
recurrence is the persistent state**; the local buffer is bounded `O(T)` and is
reported as `window_bytes`. Because the local attention is causal, ingesting a
T-frame clip one frame at a time reproduces `forward` at the last frame
(`test_fusion_r14.py`, mirroring `sanity_check.py` test 8b).

`--fuse` requires `--kind` to name the matching global operator (`local_ssm↔ssm`,
`local_wave↔wave`); `local_wave` requires `--kernel-version dispersion`. The
hybrids are not wired through the latent path.

## The rollout — why streaming vs windowed matters

`occlusion_rollout_eval` (in `train_compare.py`) is autoregressive with the
**environment occluding observations**: each step the model predicts the next
frame, the prediction is occluded inside the window _before_ being fed back, and we
score the _raw_ prediction's target centroid against the deterministic GT
trajectory. So the pixels fed back carry no target during occlusion — the only way
to track it across the 256-frame gap is persistent state.

- recurrent arms (B/C/D/E) roll out with `stream_step` (persistent state bridges);
- attention (A) rolls out windowed — it physically cannot see past its T-frame
  window (17 ≪ 256), so once the target leaves the window it is gone.

The target centroid uses a chroma weight `relu(frames[:,dom] − luminance)` on the
target's dominant channel, which ignores the gray occluder (R=G=B → 0) and the
other balls' hues. When a model _drops_ the target the weight vanishes and the
centroid collapses to the origin — a large error, exactly as intended.

## Metrics per arm (item 3)

Written to `result_<kind><tag>.json` and the printed table:

- `target_identity_survival` — fraction of frames from emergence on with target
  centroid error `< 2·radius`.
- `exit_direction_accuracy` — at emergence, both velocity-sign axes correct.
- `position_error_at_emergence`, `velocity_error_at_emergence` — grid cells.
- `divergence_horizon` — first frame the median target error exceeds `radius` for
  8 consecutive frames (fixed-before-running thresholds).
- `motion_preservation` — predicted dynamic degree ÷ GT dynamic degree (~1 good, 0
  frozen).
- `rollout_fps`, `ms_per_frame`, `mem_gb_peak` (CUDA).
- **STATE BYTES (the killer metric)** — `persistent_state_bytes` (horizon-
  independent recurrent state; 0 for attention), `window_bytes` (the T-frame window
  a finite-context arm keeps), `state_bytes_total`. The console prints
  `div_horizon/KB_state`.

`--seed` seeds weight init and offsets the per-step training-clip seeds so the
5-seed sweep gets independent runs; the held-out eval seed is fixed (90000) so all
arms/seeds are scored on the _same_ scenes.

## Smoke — `run_smoke_r14.sh` (CPU) + tests

```bash
cd experiments/wavefield_video
PY=/path/to/venv/bin/python bash run_smoke_r14.sh
```

Refuses a nonempty `OUT` (default `smoke_r14`); use a fresh dir to repeat. It runs:

1. `test_occlusion_r14.py` — target hidden **exactly** on `[occ_start, occ_end)`
   and visible otherwise; deterministic/seedable and does not touch global RNG;
   target confined to the chamber and still moving+bouncing while hidden; exit
   direction deterministic and pre-occlusion-determined.
2. `test_fusion_r14.py` — `FusedMix` forward finite; streaming `step()==forward` at
   the last frame for both hybrids; predictor `stream_step==forward`; fuse/kind
   mismatch rejected; state-byte accounting (attn 0, recurrent >0, hybrid ==
   global-only).
3. `sanity_check.py` — regression: the existing wave/attn/ssm paths are unchanged.
4. `test_occlusion_runner_r14.py` — `run_occlusion.sh` dry-run via a fake
   interpreter: five arms × seeds, equal `--target-params 4000000`, correct
   kind/fuse/kernel/seed/occ flags, failure isolation, missing-interpreter status.
5. Micro end-to-end (grid 16, 4 steps) for **all five arms**: trains, runs the
   occlusion eval, and asserts the state-byte asymmetry (attn: 0 persistent + a
   window; recurrent: persistent + no window; hybrid persistent == its global arm).

Standalone (no torch) runner test:

```bash
python3 test_occlusion_runner_r14.py
```

## Full run on the box (item 2)

```bash
cd /workspace/slava/exp/wavefield_video
P=/workspace/slava/comfy-house/venv/bin/python

PY="$P" bash run_smoke_r14.sh                      # validate first
PY="$P" OUT=runs_occlusion STEPS=2000 bash run_occlusion.sh
```

`SEEDS` (default `0 1 2 3 4`), `STEPS`, `OUT`, `PY` are overridable. Output under
`OUT`:

- `occlusion_progress.txt` — UTC start, shell-escaped command, per-arm exit + log.
- `occlusion_status.txt` — `RUNNING`, then `DONE N/N` iff every job succeeded, else
  `FAILED k/N`. A failed job does not skip the rest; the runner exits nonzero.
- `log_occlusion_<kind>_<arm>_s<seed>.txt`, `result_<kind>_<arm>_s<seed>.json`,
  `model_*.pt`, `ckpt_*.pt`.

g64 attention at a 1024-frame windowed rollout is the heaviest arm (re-attends
`17·64·64` tokens every step); `--auto-batch` and `--eval-chunk 1` bound memory,
and `--save-every 500` checkpoints. Reading the results:

```bash
tail -n 30 runs_occlusion/occlusion_progress.txt
cat runs_occlusion/occlusion_status.txt
# Kill-criterion comparison: divergence horizon per KB of persistent state, D vs E.
for f in runs_occlusion/result_*_s0.json; do
  "$P" - "$f" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
kb = max(1, r["persistent_state_bytes"]) / 1024
print(f"{r['kind']:4s} fuse={r['fuse']:11s} "
      f"div_h={r['divergence_horizon']:>5} "
      f"persist_bytes={r['persistent_state_bytes']:>8} "
      f"div_h/KB={r['divergence_horizon']/kb:7.2f} "
      f"exit_acc={r['exit_direction_accuracy']} "
      f"pos_err@emerge={r['position_error_at_emergence']}")
PY
done
```

Average the per-seed JSONs across the five seeds before drawing the verdict.

## Design notes / limitations

- The occluder is _static in shape/position_ but composited only during the window
  ("drawn OVER it" per the brief); this makes "hidden exactly 64–320" an exact,
  testable invariant rather than a fragile geometric coincidence.
- Streaming parity holds at the **last** frame of a T-frame clip (what the
  predictor actually uses); intermediate streamed frames are warmup and are not
  compared to `forward`. This matches the existing `sanity_check.py` contract.
- Hybrid `persistent_state_bytes` counts the global recurrence only; the local
  attention window (`O(T)`, bounded, cannot bridge the gap) is `window_bytes`.
- Training clips (`--frames 17 < occ_start 64`) are all pre-occlusion, so every arm
  learns the _dynamics_ from visible clips; the occlusion appears only in the
  1024-frame eval. Whether a recurrent state actually carries the hidden target is
  the empirical result this apparatus measures — it does not presuppose it.
- Verification status: the smoke was authored but **not executed in the authoring
  session** (Python execution was gated there). Run `run_smoke_r14.sh` before
  trusting any result — it is the correctness gate.

```

```
