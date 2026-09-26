# M1 TASK: integration rung — plugin on real wave-field rollouts + before/after artifact

Context: you built M0.7 (oracle test). Owner ran it: the ENGINE arm beats both oracle arms (64.0% vs 39.9% / -43.3%) and the component variant (-38.6%). M0 arc is COMPLETE; the hotspot mechanism is at its practical optimum. Details appended to IMPL_NOTES_M0_7.md.

Goal: the FIRST real integration artifact — the plugin applied to actual model-generated video, with a watchable before/after.

Deliverables:
1. `scripts/m1_integration.py` that:
   - Locates a real rollout sequence to work on. Candidates in this repo: rendered rollouts from the trained checkpoints (see render_rollout.py and any renders/ or runs_*/ dirs; checkpoints: runs_dd3/model_wave_S2.pt is the best model (copy_r 0.915). If no rendered video exists, render one yourself from S2 at 64x64, 48-64 frames, short-behavior regime (the same regime the checkpoints were trained on; use render_rollout.py's existing interfaces).
   - Runs the shipped `crumb_coherence` engine over the rollout frames with the recommended config (complex_mc: band=0.03, strength=0.5, corr, hard edge — read core.py for the exact kwargs; ALSO apply the luminance-axis config if the rollout is color).
   - Computes self-referential metrics (NO clean reference exists for model output — be honest about that): 
     * scene-centroid drift proxy: per-frame energy centroid (or phase-correlation shifts between consecutive frames on the low band), report the low-frequency drift energy before vs after via the plugin's own metrics module if reusable (crumb_coherence/metrics.py), else a small local implementation;
     * cycle/flicker proxy: mean |Δ frame| / temporal-energy above vs below the plugin's band;
     * detail preservation: hf_ssim-style frame-vs-frame consistency check that must NOT collapse (guard against the plugin inventing stillness).
   - Writes a side-by-side MP4: `renders/m1_before_after.mp4` (left = raw rollout, right = plugin output; label text if ffmpeg supports drawtext — local ffmpeg does NOT (no drawtext); if labels impossible, note it and skip labels).
   - Saves before/after PNG pairs for a few frames under `renders/m1_pairs/`.
2. `IMPL_NOTES_M1.md`: what ran, exact commands, honest numbers + caveats (no clean reference; self-referential proxies only; what would falsify "the plugin helps").
3. Keep everything so the owner can watch renders/m1_before_after.mp4 and judge.

Constraints: CPU only (do not touch the box; do not start training); no new dependencies beyond what the repo + local ffmpeg/venv already have (this machine's python = ~/vibevoice-env/bin/python for numpy/torch if needed: /opt/homebrew... check; the repo experiments use it); no changes under crumb_coherence/ (the package is frozen; integration code lives in scripts/); runtime < 10 minutes; execution is owner-run as always (write the script + run instructions; the owner executes).
