# ENG-1 — the 1080p real-footage gate (engineering test for customer safety)

You are continuing the crumb_coherence work (same session). The plugin's science gates pass
(M0: 64.0% drift removal / 0.9946 detail; M1 integration: -86% low-band variance, 0.988 detail,
~0.97 motion). Before we can honestly take customer money for a "coherence pass," one gate is
missing: REAL FOOTAGE AT FULL RESOLUTION.

## Goal
Run the FROZEN default pipeline (do not retune by hand: hotspot-mc-band=0.03, strength=0.5,
phase-anchor=0.0; alpha 0.95, rho 0.995, cutoff 0.14) over two REAL 1080x1920 30fps clips and
produce an honest engineering report. This tests: does the pass harm legitimate motion at real
resolution, and what does it actually do to clean, non-drifted content? (For clean content the
correct behavior is: it should change almost nothing measurable — non-interference.)

## Inputs
- ~/math-reels/dist/e_curve.mp4   (1080x1920, 30fps, ~22.7s)
- ~/math-reels/dist/fib_spiral.mp4 (1080x1920, 30fps, ~22s)
(Copy them into experiments/wavefield_video/data_real/ first.)

## Deliverable
scripts/run_eng1_1080.py  (new script; load via ffmpeg -i clip -f rawvideo pipe; process in
streaming windows of 48 frames using the stateful engine's carried state — do NOT hold all
frames in memory: 1080x1920x3 float32 = ~25MB/frame, 680 frames would be ~17GB. Stream in,
write corrected frames out to a pipe.)

Outputs:
- reports/ENG1_1080.json — metrics per clip:
  - low-band variance ratio (corrected/raw, temporal)
  - temporal centroid stability ratio
  - detail retention (high-freq correlation vs raw, per-frame median)
  - motion magnitude ratio (mean abs frame diff)
  - max single-frame MAE vs neighbors (did we INTRODUCE jitter?)
  - wall-clock runtime + per-frame cost
- renders/eng1_<clip>_sidebyside.mp4 (raw | corrected, each panel 540 wide, ~8s excerpt)
- Compare against a 720p downscale run of the same clip for a cost/quality table.

## Pass criteria (all must hold for a PASS)
1. Detail retention median >= 0.97
2. Motion ratio in [0.90, 1.10]
3. No introduced jitter: max frame-to-frame MAE spike <= 1.5x the clip's own 95th percentile of
   frame-to-frame MAE
4. Runtime <= 30 min per 22s clip on this Mac (CPU) — note the number either way
5. Sanity: for a synthetic DRIFTED clip (reuse the M1 fixture), the pass must still remove drift
   (don't regress the science while chasing real-footage safety)

## Honesty requirements
- If it fails any criterion, report the failure plainly with numbers; do NOT tune parameters to
  pass. A measured "this is why we cannot sell yet" is a valid and useful outcome.
- Log everything to reports/ENG1_1080.md (short, numbers first) and commit.

First: read IMPL_NOTES_M1.md + the crumb_coherence/ scripts to see the existing streaming API.
Then write the script, run it, report. Keep the commit message clear.
