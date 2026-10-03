# Supervised self-rollout prototype (opt-in)

This is the training-objective half of the carried-state experiment, not a
quality result or a reproduction of diffusion Self-Forcing++. The ground-truth
prefix guides context, and the true continuation supervises predictions. There
is no frozen video teacher, DMD, preference signal, or noise schedule.
Encoded VAE-shard training is supported, with no diffusion objective or learned
decoder change. Those require separate work, not stronger labels for this code.

References read: `RESEARCH_VIDEO_RECIPES.md` on `zeph/research-notes`
(6f39060a), especially the rollout/self-forcing section; `LONG_HORIZON.md`
8.4 on main and PR #63 (read-only). PR #63's checked revisions ddea031a and
751ea0f0 describe change-based writes as the planned 8.5, not a measured 8.5
verdict. This PR neither imports that branch nor changes its write policy.

## Contract

- `--rollout-k 0` (default, including omitted flag): original teacher-forced
  path, checkpoint/result format, losses, initialization and runner defaults.
- Positive K: of `seq_frames / chunk` chunks, reserve the FINAL K for generated
  inputs. The preceding chunks are real context. At least one context chunk is
  required. K counts chunks, not frames, and does not change clip length.
- Example: `--seq-frames 32 --chunk 8 --rollout-k 2` consumes real frames 0..15;
  the warm-up predicts frame 16. It then consumes its own predictions for input
  frames 16..31, predicting targets 17..32. Context losses already cover targets
  1..16 in dense mode: there is no duplicated/missing boundary target.
- Every generated pixel input is the preceding prediction clamped to [0,1], matching
  pixel streaming evaluation. With `--latents`, feedback is the raw float32
  prediction: VAE latents are signed and have no pixel interval, matching the
  unclamped latent evaluator and streamer. The recurrent state is carried, never reset at a
  rollout boundary. Future truth is used ONLY for loss targets/motion masks.
- `--dense` supervises every position; otherwise only the final prediction of
  each chunk contributes loss. Chunk losses have the SAME mean/scale as before.
- Both feedback and recurrent state have gradients inside a TBPTT group; both
  detach every `--tbptt-chunks` chunks. The live activation graph is bounded by
  G * chunk frames, not the entire rollout. This is not constant-size full BPTT.
  Sequential generation costs more than teacher-forced FFT chunk processing.
- `--grad-ckpt` recomputes generated steps with private state-list containers,
  so replay cannot consume a state list mutated by a later step. Complex wave
  states stay FP32/complex64; no precision or model default is changed.
- Motion loss retains TRUE previous frames and TRUE moving-pixel masks, not
  masks of hallucinated feedback. Its per-micro-batch pooling is unchanged.
- A rollout checkpoint stores `training_objective` in periodic/final checkpoints,
  model exports, and results. Resume rejects changed K, sequence/chunk, TBPTT,
  dense/motion-loss settings, or motion-loss batch partition. Legacy resumes
  remain valid with K=0. Changing teacher forcing to rollout is not a resume;
  this prototype deliberately does not implement a warm-start transfer flag.
- The existing pixel objective remains `supervised_self_rollout_v1` with
  `clamp_0_1`; signed latent rollout records `supervised_self_rollout_v2`,
  representation `vae_latents`, feedback `unclamped_float32`. Existing dataset
  and VAE fingerprints still guard latent resume. Old latent rollout checkpoints
  that used the pixel clamp are rejected; use a fresh output and fresh training.
  Previously completed teacher-forced latent checkpoints are unaffected.

## CPU or single-4090 execution smoke

From `experiments/wavefield_video/`, with torch/numpy/pytest installed:

```sh
PY=/path/to/python OUT=runs_rollout_smoke_01 bash run_rollout_smoke.sh
```

This explicitly runs the new tests plus stateful/stream regressions, kills three
mutants in disposable copies, trains matched K=0/K=2 arms for four steps, checks
all six result/checkpoint/export artifacts, compares actual learned weights,
and loads the rollout export through `long_horizon.py` for 48 generated frames.
The smoke can pass with a frozen model: it proves execution, not coherence.

For the existing GPU queue, stage the complete changed file set and copy
`queue_jobs/rollout/tpl_rollout_smoke.sh` to the next operator-assigned
`gpuq_job_NNN_rollout_smoke.sh` slot. It uses the existing box Python, redirects
to `/workspace/slava/logs/`, requires a CUDA trainer receipt, and makes a fresh
OUT. Do not run beside another GPU owner. Nothing in this PR enqueues a job,
touches the conductor, deploys, or claims a box measurement. RTX 4090 / 24 GB
is the intended target; actual peak VRAM and GPU timing remain to be measured.

A conservative paired preflight (defaults: grid16/dim64/layers2/chunk8, seq32,
K2, batch2/micro1, G1, activation checkpointing, fixed LR 0.002, 20 steps):

```sh
PY=/workspace/slava/comfy-house/venv/bin/python \
  OUT=runs_rollout_preflight_s0 bash run_rollout.sh
```

The runner refuses an existing OUT rather than trusting stale files/sentinels.
It is a bounded fresh-run recipe, NOT a sliced suite. Run the smoke first and
measure wall/VRAM before increasing budgets. Each arm saves every step. If a
longer command is interrupted, replay its recorded `command.txt` arguments with
`--tag _k2 --rollout-k 2 --resume OUT/ckpt_wave_k2.pt` (or the matching K=0
checkpoint), keeping the objective fixed. Finish the other arm separately and
run `verify_rollout_run.py OUT --rollout-k 2 --steps N`. Do not restart the
fresh-OUT wrapper over a partial run or assume it fits the conductor time cap.

## What to measure, and when to stop

`receipt.json` counts actual artifacts, checks finite losses/weights and paired
protocol identity, and reports per arm:

- `copy_ratio`, `identity_survival`, `divergence_horizon` and the saved rollout
  MSE curve: same existing evaluator, same held-out scenes across arms.
- `rollout_mse_over_frozen`: a genuinely frozen LAST CONTEXT FRAME over the
  entire continuation, same seed/geometry/horizon. This diagnostic uses rounded
  model MSE curves. Do NOT use legacy `r_persist_over_model` as a frozen baseline:
  that legacy field refreshes truth every step and is intentionally unchanged.
- `train_sec` and `persistent_state_bytes`. Equal steps/targets are not equal
  wall time. Follow with an equal-resource comparison before mechanism claims.

Hypothesis, not a promise: rollout exposure reduces drift at unseen horizons,
without obtaining lower MSE merely by freezing or darkening. Stop promotion if
rollout conditioning/gradients fail, losses become nonfinite, the model freezes,
or improvements vanish against the fair frozen baseline or equal-resource arm.
The 4/20-step runs are instrumentation smokes, never architecture verdicts.

After preflight, pre-register three seeds (0/1/2), a full-budget fixed-LR
comparison and its equal-wall counterpart. Report semantic survival/horizon,
motion and frozen-baseline curves, not training loss (conditioning differs).
A useful next gate is >=10% longer semantic divergence horizon on at least two
seeds, no identity-survival regression, and no early freeze/flatten; this is a
proposed unrun gate, not a measured outcome. If all arms collapse immediately,
check budget/objective geometry before interpreting the comparison.

The existing streamer accepts exports without changes; `--time-pos none` is
required. For the preflight shape, a later operator-run five-minute screen is:

```sh
python long_horizon.py stream --ckpt runs_rollout_preflight_s0/model_wave_k2.pt \
  --pole-param halflife --grid 16 --frames 8 --dim 64 --layers 2 --heads 4 \
  --time-pos none --batch 1 --stream-frames 7200 --chunk 600 --out stream_k2.json
```

Repeat identically for K=0. Report the FIRST latched collapse flag as the
screening horizon. Even constant state through 7,200 frames is NOT proof of five
minutes of the same content; semantic/decoded-video long_eval and human review
remain required. No such GPU or quality result is included here.

## Verification scope and integration risk

Local CPU acceptance on torch 2.14.0: 60 focused tests + 15 subtests; 3/3 mutants
killed (disabled dispatch, teacher feedback, state reset); paired four-step CLI
smoke produced six checked artifacts and changed 41/41 state-dict tensors; its
48-frame stream had constant carried state and correctly flagged freeze at
absolute frame 4. Uninterrupted vs resumed rollout weights match exactly.

A separate pinned-base probe ran six real CLIs: sparse and dense/motion/checkpoint
recipes against main a8fa84e, omitted flag, and explicit K=0. Model tensors,
optimizer states and non-timing result fields matched bitwise in both recipes.
No CUDA training was executed by this task.

Broader local checks exposed existing failures, reproduced in the unmodified
worktree: wave API parity (`KeyError: radius`; 88 passed/1 failed, 58 subtests)
and release-tag consistency (root suite: 1173 passed/2 failed; v1.3.0 already
points elsewhere). They are separate follow-ups, not fixes hidden in this PR.
Existing GitHub CI is format-focused; it does not discover the new wave tests.
The explicit smoke/test commands above are the evidence for this prototype.

Hotspot: `train_long.py` is also edited by PR #63. The rollout implementation is
isolated in `rollout_training.py`; its small trainer hook/resume metadata still
needs reconciliation when branches meet. Tensor-state wave/SSM paths on main
are tested; future structured change-write states and #63's latent path are not
claimed compatible. Every changed source/doc lives under this experiment tree.

Signed latent feedback repair (2026-10-03): six new behavioral cases first failed
on the old pixel-clamped implementation. With representation-aware feedback,
69 focused rollout/stateful/video-VAE tests plus 15 subtests pass on local CPU;
the three existing dispatch/teacher-leak/state-reset mutants are still killed.
Signed values below zero and above one match an independent recurrent loss and
gradient calculation, and a real synthetic-shard CLI resume matches an
uninterrupted run bitwise. This proves the conditioning and checkpoint contract,
not improvement of a trained video. No new CUDA experiment is included here.
