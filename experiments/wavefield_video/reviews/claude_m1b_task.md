# M1b TASK: demo v2 — make the improvement LEGIBLE to a human eye

Owner feedback on renders/m1_before_after.mp4: "not sure if that video is good or not." Correct — at 16x16 upscaled, the effect is subtle. Build three demo artifacts that are self-evident, using the SAME frozen engine (no crumb_coherence/ changes; scripts only):

1. `scripts/m1_demo_v2.py` producing:
   a. **Synthetic worst-case, three-panel** (the clearest proof): reuse the hotspot scenario from crumb_coherence/scripts/run_m0.py (or run_m05.py), 64x64, ~96 frames, upscale panels x4 with nearest-neighbor. Compose side-by-side: [DRIFTED input | ENGINE output | CLEAN truth]. Labels via PIL. The wander is 44px^2 variance at 64 grid -> at x4 scale the drift is VISIBLE; the corrected panel should visibly sit like the clean one. Output renders/m1b_three_panel.mp4.
   b. **Blink comparator for the real rollout**: alternate raw / plugin frames every ~0.4s (so the eye catches relative wobble), same crop, big upscale (x8, nearest), frame counter burned in. Output renders/m1b_blink_real.mp4.
   c. **Centroid-trajectory burn-in**: compute per-frame scene-centroid (x,y) for raw vs plugin (real rollout, 256 frames); render a matplotlib figure of both trajectories (2 subplots or xy path) + ALSO a version burned in side-by-side with the video (static graph + moving marker). If matplotlib is unavailable in ~/vibevoice-env, fall back to drawing the polyline with PIL directly (implement a tiny line-drawer). Outputs renders/m1b_trajectory.png + renders/m1b_trajectory.mp4.
2. Suggested order the owner should watch: three-panel -> blink -> trajectory.
3. IMPL_NOTES_M1b.md: commands + what each artifact shows + honest limits (synthetic = controlled drift; real = subtle at g16).
4. Execution stays owner-run; list the exact commands.

Constraints: CPU only; no new deps beyond matplotlib-or-PIL fallback (both likely present; check first, prefer what exists); runtime < 8 min total; artifacts must be >= 30 fps?? no — keep readable: three-panel 12 fps fine.
