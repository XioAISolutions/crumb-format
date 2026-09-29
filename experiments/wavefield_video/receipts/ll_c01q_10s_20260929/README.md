# LongLive c01q — first successful generation (2026-09-29)

Job: `gpuq_job_974zzzzzz_c01q_offload.sh` — RUNNING 17:10:02Z → FINISHED **rc=0** 17:13:32Z (210 s).
Config: fp8 W8A8, window 12, sink 8, LENGTHS=10, `LL_OFFLOAD_IDLE=1`, OUT=`runs_longlive_cfg16d`.

## Why it worked now

Idle-offload patch active (pipeline SHA `7373a6fb…`): text encoder + VAE move to CPU right after
prompt encoding (both idle through denoising):

    [mem] pre: alloc=16.66 reserved=23.13
    [mem] te dev: cuda:0
    [mem] post-clear: alloc=16.66 reserved=16.69
    [mem] idle-offload: TE+VAE -> cpu; alloc=4.80 GiB        # freed ~11.9 GiB

Then: KV inference 8 frames/block (`local_attn_size 12`) → 8 blocks × 4 sampling steps → decode.

## Output

`len_10s/rank0-0-0_regular.mp4` — 253 frames, 1280×704, 24 fps, 10.54 s, 27,455,920 bytes
(md5 `3292dc4344e192717bd9357dca9572d0`; provenance + run identity in this dir).

Eval: `long_eval` **NO_FLAGS** (dinov2 diagnostic). 10.5 s is below the evaluator's 30 s window
granularity (`n_samples=22`); semantic quality **unverified** — independent pixel review required.
Contact sheet `contact_9.jpg` (9 samples). Owner screening pending; **not published anywhere**.

## Latent arm (same window)

`runs_latent4_full` — 3000 steps, 3222.8 s, 32.18M params (`ckpt_wave_latent_s0.pt` saved).
Loss tail: 0.0020 / 0.0049 / 0.0057 (steps 2400/2700/3000). `eval_mse 0.4276`,
`eval_mse_over_copylast 0.9685` (c56 300-step smoke: 0.9913) — direction positive, magnitude modest;
long-horizon rollout error still grows (`latent_rollout_steps 114`).

## Next

`c01r` (LENGTHS=30, same fp8/window-12/offload config) queued next; 180 / 300 s rungs after.
`tpl_longlive` un-pause decision after the 30 s rung.
