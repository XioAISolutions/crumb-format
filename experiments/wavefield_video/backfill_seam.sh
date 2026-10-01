#!/bin/bash
# backfill_seam: one-shot seam audit over every existing generated mp4 in pr63 run dirs.
# CPU-only (ffmpeg); writes <video>.seam_score.json sidecars. Skips existing sidecars.
cd /workspace/slava/exp/pr63
for mp in runs_probe_*/len_*/*.mp4 runs_seam_*/len_*/*.mp4 runs_longlive*/len_*/*.mp4; do
  [ -f "$mp" ] || continue
  out="${mp%.mp4}.seam_score.json"
  [ -f "$out" ] && continue
  echo "== $mp"
  /usr/bin/python3 seam_score.py "$mp" --json "$out" || echo "FAILED $mp"
done
echo BACKFILL_DONE
