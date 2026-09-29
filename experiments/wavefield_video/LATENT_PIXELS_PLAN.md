# The latent arm's pixels: texture-lock, and the plan

Evidence (2026-09-29, box runs):

- **c51 stream mechanics are perfect**: 7,200/7,200 latent steps, 57,600 decoded
  frames (40:00 at 24 fps), constant 31.6 MB recurrent state the whole way, zero
  health flags (runs_latent3, `long_horizon.py stream`).
- **The decoded pixels are not**: after the real context, a static fine-texture
  mosaic; identical character at 42 s / 200 s / 400 s / 40 min
  (frames_40min/, contact sheets), while the health stats (mean/std/motion)
  stay "stable".
- **Metrics agree**: `eval_mse_over_copylast` = 1.66 (v3) / 1.45 (v2) — worse
  than copy-last; train loss 0.0012 = the model predicts its own equilibrium.
- **The failure is neither fade nor flatten** — the pre-registered KILL rule
  never fires. The monitor is blind to it. Name it TEXTURE-LOCK and add it to
  the monitor.

Why (ranked):

1. **Capacity**: ~4M params on 48ch x 44x80 latents + a corpus of a few
   thousand latent steps. The MSE optimum of that fit is blur/texture.
2. **Objective**: teacher-forced (K=0) next-latent loss; nothing penalizes
   drift when self-generated samples feed back.
3. **Corpus**: too little long-form content for scene dynamics.

Plan (gated; pre-registered reads before results):

- **P0 (now)**: score existing streams with `long_eval.py` (dinov2) so the
  failure is a number (c55 job queued: the 40-min mp4). Add texture-lock to
  HealthMonitor: high-band spectral energy / frame-novelty floor vs the context
  window (KILL rule extension).
- **P1**: scale — v4 smoke queued: dim 512 / 12 layers / 16 heads (~10x params)
  on the same latents, seq 64 / chunk 16, 300 steps. Gate: fits + loss falls.
  Then the full run at the same scale.
- **P2**: data — encode >= 1-2 h of curated long-form video through LTX
  (target >= 45k latent steps vs ~2-4k today).
- **P3**: rollout loss on latents — the K=2 recipe that showed 0.043 vs 0.85
  MSE plateau at 512 frames on the balls world (runs_rollout_real_20260929T040150Z).
- **P4**: judge by decoded pixels — long_eval coherent-horizon is THE number;
  eye-check every checkpoint before any claim.

Reuse as-is: the O(1) stream harness, the receipts pattern, the gpuq pipeline.
Option: LTX-2 latent space (better reconstruction) on the next re-encode.
