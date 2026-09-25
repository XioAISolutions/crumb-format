# R12 — difficulty ladder + semantic object metrics

R12 makes ball count a task parameter and adds detection-aware position metrics
to evaluation and rendering. Dark predictions can score deceptively well under
pixel MSE; they now report missing objects explicitly. No commits.

## Ball count and compatibility

`train_compare.py --n-balls K` passes the count to `data.py` for training,
evaluation, baselines and streaming. The default remains **3**. Existing calls
using `nb` continue to work; default seeded clips retain their previous bytes and
RNG draws. The pairwise collision resolver covers every pair for larger counts.
Result metadata records `n_balls`, `radius` and `speed`, and both render paths use
the saved values to regenerate matching truth. Older result files use the old
defaults when these fields are absent.

The ladder uses **balls** as its data source. “Wave arm” means `--kind wave`, the
predictor architecture; it does not mean the PDE `--data-source waves` generator.
PDE clips have no ball identities, so their semantic metric is `null`.

## Semantic metric

Evaluation and rendering share the palette/centroid detector and matching code.
Three palette anchors separate colors, then spatial peaks identify individual
blobs, including repeated colors at larger counts. Detection requires absolute
density at least 0.25 and local peak prominence at least 0.10; brightness is never
normalized by the frame's maximum. Spatial suppression/support scale with radius,
and connected flat maxima count once. Each predicted frame is scored in pixels
at the native grid, including decoded latent frames. The report contains:

- `expected_n`: number of ground-truth balls.
- `detected_n`: number of detected prediction blobs; it is not forced to equal GT.
- `matched_count`: color-compatible, one-to-one matches within `2 * radius` cells;
  matching maximizes the number of matches before minimizing distance.
- `mean_matched_position_error`: mean distance between matched centers, in grid
  cells; `null` when no objects match.

Read the counts together with position error. A blank prediction has no matched
objects and a `null` position error; it cannot masquerade as perfect tracking.
Misses or extra objects remain visible in the counts even when the surviving
matches have small position errors. Thresholds scale with the ball radius;
position error is in native grid cells, not resized PNG pixels. Comparisons across
grids should account for their different cell sizes and scene geometry.

This is a detector-based measurement, so overlapping same-color balls can merge
and undercount even in a perfect prediction. Boundary-truncated blobs also have a
centroid detection floor. Retaining expected counts and raw per-seed measurements
makes these limits visible.

The training result JSON and `eval_only.py` JSON contain `semantic` for
autoregressive evaluation. Render `metrics.json` contains `metrics.semantic`.
Each semantic block contains the four aggregate values above,
`units: "grid_cells"`, detection settings, `match_tolerance_px`, and a `frames`
list. Frame records retain `samples` with the per-seed counts and error. Count
aggregates average over samples/frames; aggregate position error is weighted by
the number of matched objects. Frame indices start at zero in the scored
prediction sequence. Both console summaries include
`semantic expected=... detected=... matched=... pos_err=...`.

The latent render loop still feeds predicted latents back and decodes only for
scoring/export, as in R11. The training evaluator still uses its pixel-in/pixel-out
`LatentWrapper`; both apply the same detector to decoded pixels. These are
different feedback loops, so their long-horizon results need not be identical.
The checkpoint-only `eval_only.py` utility retains its float32 evaluation behavior.
Trainer `rollout_fps` includes the added CPU semantic detection time and should
not be compared directly with pre-R12 timing. Renderer `model_fps` continues to
exclude metrics, so its timing definition is unchanged.

## Smoke commands

```bash
cd /workspace/slava/exp/wavefield_video
P=/workspace/slava/comfy-house/venv/bin/python
PY="$P" bash run_smoke_r12.sh
```

For a local interpreter:

```bash
cd experiments/wavefield_video
PY=/path/to/venv/bin/python bash run_smoke_r12.sh
```

The smoke refuses a nonempty `OUT` (default `smoke_r12`). Use a fresh directory for
a repeat, preserving the earlier evidence:

```bash
PY="$P" OUT=smoke_r12_repeat bash run_smoke_r12.sh
```

The smoke checks seeded default-path byte identity, larger ball counts and
collisions, synthetic semantic cases with known centers/counts at multiple grids,
the eight-job ladder via a fake interpreter, and the latent decode/render
regression. It proves correctness of the plumbing, not accuracy of a trained
predictor. The standalone runner test needs only Python's standard library:

```bash
python3 test_difficulty_r12.py
```

## Run after the retune

The ordering is **R12 smoke → retune → choose recipe flags → difficulty ladder**.
Retune isolates objective/schedule choices at g32; the ladder then measures how a
fixed selected recipe degrades as the task becomes harder. Each rung starts a
fresh model. These are independent experiments, not a curriculum or checkpoint
continuation.

```bash
cd /workspace/slava/exp/wavefield_video
P=/workspace/slava/comfy-house/venv/bin/python

# First validate R12, then finish the R10 retune using the updated evaluator.
PY="$P" OUT=smoke_r12_before_retune bash run_smoke_r12.sh
mkdir -p runs_deep
bash run_retune.sh
cat runs_deep/retune_progress.txt
cat runs_deep/retune_status.txt
```

Inspect per-arm exits and `runs_deep/result_wave_*.json` before selecting the
recipe. The existing retune runner writes `DONE` after its loop even if an arm
failed. Its progress file and result files establish which arms actually finished.

If retune completed before R12, add the new metric to a separate evaluation file
without retraining or overwriting the original result. For example:

```bash
"$P" eval_only.py runs_deep/model_wave_g32_bal_mw.pt \
  runs_deep/result_wave_g32_bal_mw.json runs_deep/eval_r12_wave_g32_bal_mw.json \
  --eval-seeds 16 --eval-rollout 256 --eval-chunk 4

# The same evaluator accepts latent checkpoints and scores their decoded pixels.
"$P" eval_only.py runs_latent/model_wave_w1lat.pt \
  runs_latent/result_wave_w1lat.json runs_latent/eval_r12_wave_w1lat.json \
  --ae-ckpt ckpts/ae_g32.pt --eval-seeds 16 --eval-rollout 256 --eval-chunk 4
```

`run_difficulty.sh` defaults to retune's g32 K=1 control: dispersion wave, gate,
local fusion, dim 192, 6 layers, 8 heads, 4M target parameters, 17 context frames,
batch 16, causal residual prediction, linear padding, collisions, motion loss,
256-frame evaluation with 16 seeds in chunks of 4, auto-batch, and checkpoint
interval 500. The default budget is 2,000 training steps per arm.

`RECIPE_EXTRA` is appended to that base. Put the winning optional flags here;
the example below demonstrates the combined objective flags and does **not**
claim they won the retune:

```bash
RECIPE_EXTRA='--resid-balanced --motion-weighted' \
  PY="$P" OUT=runs_difficulty STEPS=2000 bash run_difficulty.sh
```

Use an empty value for the unchanged K=1 control:

```bash
RECIPE_EXTRA='' PY="$P" OUT=runs_difficulty_ctrl STEPS=2000 \
  bash run_difficulty.sh
```

If a no-collision arm wins the retune, `--no-collisions` overrides the base's
`--collisions`. For example, use its ramp schedule only if that is the selected
recipe:

```bash
RECIPE_EXTRA='--no-collisions --rollout-ramp' \
  PY="$P" OUT=runs_difficulty_nocoll STEPS=2000 bash run_difficulty.sh
```

`RECIPE_EXTRA` accepts whitespace-separated flags/values, including newlines. It
does not execute shell expressions or expand glob patterns; embedded quotes are
literal. Use `OUT` and `PY` for paths, which may contain spaces. Recipe options
are followed by each rung's task overrides, then the arm, balls data source,
`STEPS`, `OUT`, and tag. Thus retune flags can override base values, while rung
overrides and output routing take precedence. Use a fresh `OUT` for each recipe;
reusing it overwrites that ladder's logs, status and named model/result files.

| Rung | Task overrides after recipe | Arms | Generated rollout frames |
| --- | --- | --- | --- |
| D1 | `--speed 2.30` | wave | 256 |
| D2 | `--n-balls 8` | wave | 256 |
| D3 | `--n-balls 12 --speed 2.30` | wave, attn | 256 |
| D4 | `--radius 0.8` | wave | 256 |
| D5 | `--grid 64 --n-balls 8` | wave, attn | 256 |
| D6 | `--grid 32 --eval-rollout 1024` | wave | 1024 |

There are **eight jobs**, run sequentially in rung order, with the attention
comparison immediately after its wave arm. D6 uses the same 2,000-step training
budget and extends only evaluation. Geometry and rollout settings not overridden
by a rung are inherited from the chosen recipe. GPU OOM recovery is the existing
`--auto-batch` behavior; g64 attention may require a smaller actual batch.

Output convention under `OUT` (default `runs_difficulty`):

- `difficulty_progress.txt`: UTC start, exact shell-escaped command, per-arm exit
  code and log path.
- `difficulty_status.txt`: `RUNNING`, then `DONE` only if every job succeeded;
  otherwise `FAILED N/8`. A failed job does not prevent the remaining jobs from
  being attempted. The script exits nonzero on any failure or interruption.
- `log_difficulty_wave_D3.txt` / `log_difficulty_attn_D3.txt`: per-arm logs.
- `result_wave_D3.json`, `model_wave_D3.pt`, `ckpt_wave_D3.pt`, and analogous
  files for other arms/rungs: trainer outputs.

```bash
tail -n 20 runs_difficulty/difficulty_progress.txt
cat runs_difficulty/difficulty_status.txt
tail -n 30 runs_difficulty/log_difficulty_wave_D3.txt

# Render the same D3 test seed for both completed arms; saved count/geometry apply.
"$P" render_rollout.py --ckpt runs_difficulty/model_wave_D3.pt \
  --kind wave --frames 256 --seed 170001 --side-by-side \
  --out demo_out/r12_wave_D3
"$P" render_rollout.py --ckpt runs_difficulty/model_attn_D3.pt \
  --kind attn --frames 256 --seed 170001 --side-by-side \
  --out demo_out/r12_attn_D3

# Preserve the full D6 stress horizon in the exported demo.
"$P" render_rollout.py --ckpt runs_difficulty/model_wave_D6.pt \
  --kind wave --frames 1024 --seed 170001 --side-by-side \
  --out demo_out/r12_wave_D6
```

## Runner verification

`bash -n run_difficulty.sh` and all **5** `test_difficulty_r12.py` tests passed.
A disposable command-recording interpreter verified all eight jobs, each rung's
flags, multiline recipe extras including `--no-collisions`, argument precedence,
paths containing spaces, per-arm logs and final success status. A failure case
forced D3 attention to exit 7: all eight arms were still attempted, the progress
file retained that exit, final status was `FAILED 1/8`, and the runner exited 1.
Missing interpreters produce a failure status; shell substitutions and globs in
recipe extras remain literal. These tests do not launch training jobs.

Separate pre/post-R12 regression runs verified six-step training against the old
implementation: all **25** saved state tensors were `torch.equal`, and losses plus
eight legacy evaluation/centroid fields matched. A pre/post render comparison
verified identical bytes for all **21** PNG/MP4 files and equality of six original
image metric keys. Semantic fields are additive; timing fields are not expected
to remain identical.

The full R12 smoke passed after the flat-plateau detector fix, including all nine
semantic tests, five runner tests, pixel training/rendering, and latent decode
plus checkpoint-only evaluation. Exact local verification command:

```bash
PY=/Users/slavaz/crumb-format/.venv-standalone/bin/python \
  OUT=/tmp/wavefield-r12-baseline.VDsUw6/smoke_final bash run_smoke_r12.sh
```

Captured log: `/tmp/wavefield-r12-baseline.VDsUw6/smoke_final.log`. Use a different
`OUT` to repeat this completed run.
