R11 TASK: latent decode-render + pixel-space eval. Read latent_ae.py, render_rollout.py, IMPL_NOTES_R7.md first. No commits.
(1) Extend render_rollout.py: --latent --ae-ckpt PATH args. Run the latent model autoregressively at latent_grid; decode EVERY predicted latent frame through the AE to pixels; compute metrics in PIXEL space vs the pixel GT clip (same seed, make_clip_batch): mse, copy_last_mse (frozen baseline), centroid err, divergence.
(2) Keep the pixel path untouched (default regression must pass). Reuse latent_ae.py's AE for loading ckpts/ae_g32.pt.
(3) run_smoke_r11.sh: end-to-end decode path on a tiny synthetic latent rollout (no trained ckpt needed), plus default-path regression.
(4) IMPL_NOTES_R11.md with exact box commands to render (a) runs_latent/model_wave_w1lat.pt and (b) runs_latent/model_attn_a1lat.pt once it exists, using /workspace/slava/exp/wavefield_video/ckpts/ae_g32.pt, outputting pixel metrics + mp4 into demo_out/.
