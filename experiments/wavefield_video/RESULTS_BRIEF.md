# RESULTS BRIEF — crumb-video v2 (2026-09-25)
Arch: wave-field mixer (separable w0 / dispersion w1) vs attention vs S4D-lite SSM.
Harness: residual pred, factorized pos, causal+linear-pad, gate+fuse (wave), param-matched ~1.6M, 3000 steps.

## g16 COMPLETE (eval_mse / copy-last, then rollout fps, mem@256, r_persist)
attn_a1  0.218 | 64 fps | 0.91GB | r 0.033
wave_w1  0.239 | 123 fps | 2.06GB | r 0.022   (dispersion)
wave_w0  0.276 | 238 fps | 0.90GB | r 0.032   (separable)
ssm_s1   0.459 | 354 fps | 0.90GB | r 0.015
one-step baselines: zero 0.0426, copy-last 0.0057, const-vel 0.0044
Rollout@256: attn plateaus ~0.18; w1 ~0.32; w0 ~0.42; ssm ~0.48.

## g32 IN PROGRESS (grid 32, T=17, batch 16)
ssm DONE: 0.99x copy-last (COLLAPSE at 32²), rollout 0.0015->0.07 no plateau, 0.61 st/s.
attn + w1 g32 running now.

## Engineering gaps
- OOM crashes at larger batch/grid; no auto-retry, no resumable checkpoints.
- Dispersion kernel lacks a recurrent step() form (streaming/constant-memory untested).
- Per-pixel tokens; no latent space (minutes-scale = memory wall).

## Ask
1) Rank next upgrades: hardening + quality. 2) Design the streaming experiment (512-1024 frames constant-memory) incl. exact state form + forward()==step() verification. 3) Concrete 48h plan on ONE 4090.
