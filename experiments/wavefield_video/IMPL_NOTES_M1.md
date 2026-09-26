# IMPL_NOTES_M1 — integration rung: plugin on a real wave-field rollout (2026-09-26)

## Task recap

M0 tuned and froze `crumb_coherence` on SYNTHETIC clips (moving balls + injected
corruptions). M1 is the first artifact of the plugin applied to ACTUAL
model-generated video, with a watchable before/after — and an honest read of
whether it helps when there is NO clean reference to score against.

## What was built (crumb_coherence/ FROZEN — integration code lives in scripts/)

`crumb_coherence/scripts/m1_integration.py` — loads a real wave-model rollout,
runs the shipped engine over it in two configs, computes self-referential
metrics, and writes a side-by-side MP4 + before/after PNG pairs. It only _calls_
the frozen package; nothing under `crumb_coherence/` is modified.

### Frame source (and an important reality check on the checkpoint)

The task named `runs_dd3/model_wave_S2.pt` (best model, copy_r 0.915) rendered at
64x64. **That checkpoint does not exist locally** — `runs_dd3/` was a planned
training run (see `run_freeze_fast.sh`, `IMPL_NOTES_HYBRID.md`) that was never
executed on this machine. The best LOCAL wave checkpoint is
`demo_ckpts/model_wave_w1r4.pt` at **grid 16** (copy_r 0.719), and there is no
64x64 wave checkpoint at all. `render_rollout.py` asserts the trained grid and
does not resize, so 64x64 real model output is not obtainable here.

So M1 uses the real rollout that already exists:
`demo_out/seed_170001_wave/prediction/` — 256 frames of the grid-16 wave-kernel
`VideoPredictor` rolled out autoregressively (windowed mode, seed 170001;
content = moving balls, "wave" = the model's dispersion kernel; metrics.json
confirms grid 16). The script recovers the native 16x16 float frames from the
NEAREST-upscaled display PNGs losslessly (resize back down to the native grid —
a pixel-replicated block collapses to its own value). Default uses the first
`--frames 64`.

The script also accepts `--ckpt` to render a fresh rollout via `render_rollout.py`
first — so the moment S2@64 (or any grid-N wave checkpoint) exists, the SAME
command integrates it: `--ckpt runs_dd3/model_wave_S2.pt --grid 64`.

### Plugin configs run

- **[A] luminance / energy axis** — `anchor_mode="magnitude"`, cutoff 0.10,
  alpha 0.9, rho 0.995. Grid-agnostic; holds low-band energy/contrast steady and
  leaves phase (motion) free. This is the config FEATURED in the before/after MP4.
- **[B] recommended hotspot config** — `anchor_mode="complex_mc"`, mc_strength
  0.5, mc_band 0.03, mc_est "corr", mc_edge "hard", cutoff 0.14, alpha 0.95.

  **Grid-16 caveat (certain, stated up front):** `mc_band` is in cycles/pixel. At
  grid 16 the low-band box is 2x2 and the smallest non-DC frequency is
  1/16 = 0.0625 cyc/px, so the r<0.03 inner-band mask contains ONLY DC. The mask
  therefore zeroes every non-DC phase term and `complex_mc` collapses to plain
  `complex` low-band anchoring — the positional re-centering that won at grid 64
  is inexpressible at grid 16. The script prints this diagnosis
  (`band_note(...)`). This is why the luminance axis, not the hotspot axis, is the
  meaningful integration lever on the currently-available model.

### Self-referential metrics (no clean reference — reused from metrics.py where possible)

- **low-band drift energy** — `lowband_trajectory_variance` (temporal variance of
  the low-band magnitude spectrum) + a local low-band energy-**centroid** variance
  (the scene-centroid drift proxy, same construction run_m0 uses for the hotspot,
  applied to the clip's own low band). Should DROP if the plugin stabilizes.
- **cycle / flicker** — `temporal_flicker` split into a LOW-band and HIGH-band
  reconstruction. The low-band number should drop (drift/flicker removed); the
  HIGH-band number must stay ~unchanged (ratio ~1) or the plugin froze motion.
- **detail preservation** — `highfreq_ssim(raw, plugin)` must stay ~1 (the plugin
  only moves the low band, so HF should be untouched), PLUS a stillness guard
  (`temporal_flicker(plugin)` must be >> 0) so a "win" cannot be the plugin
  inventing a still frame.

### Artifact

- `renders/m1_before_after.mp4` — left = raw rollout, right = plugin[magnitude],
  panels NEAREST-upscaled to ~256px for watchability, encoded via ffmpeg from a
  rawvideo stdin pipe. **Labels ARE present** — baked into the pixels with
  `PIL.ImageDraw` (local ffmpeg has no `drawtext`, so text is drawn in Python
  before encoding rather than skipped).
- `renders/m1_pairs/pair_XXXX.png` — 6 evenly-spaced before/after pair frames.
- `renders/m1_metrics.json` — all numbers, machine-readable.

## Exact commands (owner-run)

```bash
cd experiments/wavefield_video
PY=~/vibevoice-env/bin/python

# default: reuse the existing grid-16 wave rollout, first 64 frames
$PY crumb_coherence/scripts/m1_integration.py

# RECOMMENDED also: the full 256-frame rollout shows the ACCUMULATED drift the
# plugin targets (early frames drift little; late frames are where it shows):
$PY crumb_coherence/scripts/m1_integration.py --frames 256

# when a grid-N wave checkpoint exists (e.g. S2@64), integrate a fresh render:
# $PY crumb_coherence/scripts/m1_integration.py --ckpt runs_dd3/model_wave_S2.pt --grid 64 --frames 64
```

Then watch `renders/m1_before_after.mp4` and judge.

## RESULTS — PENDING OWNER RUN

> `python` is execution-gated in this session (the owner-run convention from
> IMPL_NOTES_M0_2..M0_7 — retried this round, denied). No numbers are invented.
> Run the commands above and paste the two blocks; the table columns below are
> what the script prints. The grid-16 band diagnosis for [B] is a certainty and
> is pre-filled.

Self-referential (raw rollout vs plugin; +% = reduction = good, except the ratio):

| config                   | lowband mag var (raw→plugin, %drop) | centroid var px² (%drop) | low-band flicker (%drop) | high-band flicker ratio (~1=motion kept) | detail hf_ssim(raw,plugin) | plugin flicker (>0?) |
| ------------------------ | ----------------------------------- | ------------------------ | ------------------------ | ---------------------------------------- | -------------------------- | -------------------- |
| A magnitude (featured)   |                                     |                          |                          |                                          |                            |                      |
| B complex_mc (band=0.03) |                                     |                          |                          |                                          |                            |                      |

[B] band diagnosis (certain): low-band box 2x2; cells with r<0.03 = 1 (DC only;
fundamental 1/16 = 0.0625 cyc/px) → **DEGENERATE → complex_mc ≡ complex at grid 16.**

### Pre-registered expectation (HYPOTHESIS, not measurements)

- **[A] magnitude** is the transfer story that should work: an AR rollout tends to
  drift in low-band energy/contrast, which magnitude anchoring holds steady while
  leaving phase (ball motion) free. Expect the low-band magnitude variance and
  low-band flicker to drop, high-band flicker ratio ~1, and detail hf_ssim ~1
  (magnitude never touches HF). If magnitude barely moves the numbers, the honest
  reading is "this rollout's drift isn't in the low-band magnitude" — report it.
- **[B] complex_mc** at grid 16 == complex: it will anchor coarse low-band phase,
  which may reduce centroid drift BUT risks pulling ball motion (watch the
  high-band flicker ratio and detail hf_ssim). Its hotspot-specific value cannot
  appear until grid-64 model output exists.
- The first 64 frames may show only mild drift (early rollout is best-predicted);
  the `--frames 256` run is where accumulated drift — and thus the plugin's
  leverage — should be clearest.

## Honest caveats

- **No clean reference.** Every number is self-referential (raw-vs-plugin or the
  clip's own temporal statistics). None is an accuracy figure; a low-band-variance
  drop means "steadier", not "more correct".
- **Grid 16, not 64.** The only local wave model is 16x16 (copy_r 0.719), not the
  S2@64 the task assumed. The hotspot/complex_mc axis is therefore degenerate here
  (DC-only band); M1 demonstrates end-to-end integration + the luminance axis, and
  the script is ready to integrate S2@64 the instant it is rendered.
- **8-bit recovery.** Native frames are recovered from the display PNGs (already
  uint8); a fresh render (`--ckpt`) would keep float precision but needs the
  checkpoint + config present.
- The engine's shipped cut-detection / anti-collapse guards are active (frozen
  package); no attempt is made to tune them for model output in this round.

## What would falsify "the plugin helps"

On this real rollout, the plugin does NOT help if ANY of:

1. low-band drift energy / centroid var / low-band flicker do NOT drop (no
   stabilization);
2. detail `hf_ssim(raw, plugin)` collapses (< ~0.95 → the plugin damaged detail);
3. high-band flicker ratio → 0, or plugin flicker → 0 (the plugin froze the scene
   into stillness — the failure the anti-collapse guard exists to prevent).

## Next

If [A] shows a clean low-band stabilization with detail + motion intact, that is
the first real "plugin helps a generator" evidence — the integration rung of the
plugin spec is met on available hardware. The 64x64 hotspot axis remains gated on
training S2@64 (owner, on the box — explicitly out of scope for this CPU round).
