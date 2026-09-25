R8 TASK: product demo + spec for brainsnn.com. New files only: render_rollout.py, demo_player.html, PRODUCT_SPEC_brainsnn.md. No commits.
(1) render_rollout.py: load a train_compare checkpoint (runs_*/ckpt or result-matched config), autoregressive rollout N frames, save PNGs + mp4 (ffmpeg; fallback imageio), optional side-by-side vs ground truth; CLI: --ckpt --kind --kernel-version --grid --frames --out.
(2) demo_player.html: self-contained (no CDN): side-by-side video + metrics panel + honest pitch line (long-horizon coherence at constant memory; no photorealism claims).
(3) PRODUCT_SPEC_brainsnn.md: pillar positioning, demo page plan, streaming API sketch, claim boundaries, milestones, user-test questions.
