# IMPL_NOTES_M1b — demo v2: make the improvement legible (2026-09-26)

## Task recap

Owner feedback on `renders/m1_before_after.mp4` (M1, 16x16 upscaled, magnitude
axis): "not sure if that video is good or not." Correct — at grid 16 the effect
is subtle and a layperson can't tell which side is better. M1b builds three
self-evident artifacts with the SAME frozen engine (nothing under
crumb_coherence/ changes; scripts only).

## What was built

`crumb_coherence/scripts/m1_demo_v2.py` produces three artifacts:

- **(a) three-panel synthetic — the clean proof.** The M0 hotspot at **64x64**,
  ~96 frames, panels upscaled x4 (nearest). Side-by-side
  `[DRIFTED input | ENGINE output | CLEAN truth]`, PIL labels. Here the drift is
  KNOWN and CONTROLLED (~44 px^2 low-band position variance) and the config is the
  M0.6 winner (`complex_mc band=0.03`, **non-degenerate at grid 64**), so the
  corrected panel should visibly sit like the clean one while the drifted panel
  wanders. -> `renders/m1b_three_panel.mp4`
- **(b) blink comparator — real rollout.** The grid-16 wave-model rollout upscaled
  x8, playing forward while the SOURCE toggles raw<->plugin every ~0.4s so the eye
  catches the relative wobble; frame index + source burned in. ->
  `renders/m1b_blink_real.mp4`
- **(c) centroid-trajectory — real rollout.** Per-frame low-band energy centroid of
  raw vs plugin over the full rollout, as a static figure (matplotlib if present,
  else a self-contained PIL line-drawer — the script tries matplotlib and falls
  back automatically) AND a burn-in video (raw rollout beside the graph with
  moving markers). -> `renders/m1b_trajectory.png` + `renders/m1b_trajectory.mp4`

Real-rollout config = `complex_mc band=0.03` (the recommended config). **At grid 16
`mc_band=0.03` is below the fundamental (1/16=0.0625), so it acts as `complex`
low-band anchoring** — the position-stabilizing behaviour a blink / trajectory
reveals. Labels are baked into frames with `PIL.ImageDraw` (local ffmpeg has no
`drawtext`); MP4s are encoded via an ffmpeg rawvideo-stdin pipe (no image-lib MP4
dependency). matplotlib-or-PIL is selected at runtime (no new deps).

## Exact commands (owner-run) + watch order

```bash
cd experiments/wavefield_video
PY=~/vibevoice-env/bin/python

$PY crumb_coherence/scripts/m1_demo_v2.py           # all three artifacts
# or one at a time:
$PY crumb_coherence/scripts/m1_demo_v2.py --only three_panel
$PY crumb_coherence/scripts/m1_demo_v2.py --only blink
$PY crumb_coherence/scripts/m1_demo_v2.py --only trajectory
```

**Watch order:** `renders/m1b_three_panel.mp4` -> `m1b_blink_real.mp4` ->
`m1b_trajectory.mp4` (+ `m1b_trajectory.png`).

## What each artifact shows

- **three-panel** is the artifact to trust: a controlled, non-degenerate 64x64 case
  where the drift is injected and known, so "ENGINE output matches CLEAN truth
  while DRIFTED wanders" is a real, legible proof of the mechanism — the thing M1
  couldn't show at grid 16.
- **blink** surfaces the (subtle) per-frame difference the plugin makes on the REAL
  rollout by toggling source; a blink comparator is designed to reveal small shifts
  the eye misses in a static side-by-side.
- **trajectory** quantifies the real-rollout centroid path (raw vs plugin) so the
  reduction in low-band wander is visible as a tighter path; the script prints the
  centroid path-variance ratio plugin/raw.

## RESULTS — PENDING OWNER RUN

> `python` is execution-gated in this session (owner-run convention). The scripts
> print their own numbers (e.g. the trajectory path-variance ratio and the matplotlib/
> PIL backend used) and write the MP4/PNG artifacts; no numbers are invented here.

## Honest limits

- **Synthetic vs real.** The three-panel is CONTROLLED (injected, known drift on
  a non-degenerate 64x64 grid) — that is why it is the clean proof. The blink and
  trajectory are on the REAL grid-16 rollout, where the effect is genuinely subtle
  and `complex_mc` degenerates to `complex` anchoring (no true band-limited phase
  correction exists at 16x16).
- **Centroid tightening is not proof of "correctness."** On the real rollout the
  plugin's centroid path is tighter, but at grid 16 `complex` anchoring cannot
  distinguish drift from legitimate low-frequency motion, so some of that
  tightening may be motion-damping rather than drift-removal (Astra red flag #2:
  the low-band variance metrics are partly self-referential). The **M2
  legitimate-motion trap** (`run_trap.py`, `IMPL_NOTES_M2.md`) is the test that
  disambiguates this; read the trap verdict alongside these demos before treating
  the real-rollout tightening as a win.
- **Not a product hero shot yet.** Per the assessment, the compelling product demo
  needs footage where the raw side develops an unmistakable long-horizon defect;
  the three-panel is the internal proof, not the marketing clip.

CPU only; runtime < 8 min total; no engine changes.
