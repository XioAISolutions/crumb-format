#!/bin/bash
# Fetch + score the OPSD-V A/B outputs (original vs opsd configs) and print comparison.
set -euo pipefail
S=/Users/slavaz/.hermes/cache/scratch; cd "$S"
B=root@161.184.224.50; P=41400
for n in orig opsd; do
  H=$S/clip_harvest/opsd_$n; mkdir -p "$H"
  D=/workspace/slava/opsd_out_$n/seed_1
  scp -q -P $P "$B:$D/"'*.mp4' "$H/" 2>/dev/null || D=/workspace/slava/opsd_out_$n
  scp -q -P $P "$B:$D/"'*.mp4' "$H/" 2>/dev/null || { echo "$n: fetch failed"; continue; }
  f=$(ls "$H"/*.mp4 2>/dev/null | head -1)
  [ -n "$f" ] && [ "$f" != "$H/rank0-0-0_regular.mp4" ] && mv "$f" "$H/rank0-0-0_regular.mp4"
  bash "$S/harvest_clip.sh" "opsd_$n" >/dev/null 2>&1 || echo "$n: score failed"
done
echo "=== OPSD A/B (median px/frame @640w) ==="
for n in orig opsd; do
  F="$S/clip_harvest/opsd_$n/flow.json"
  if [ -s "$F" ]; then
    F="$F" "$S/venv_flow/bin/python" -c 'import json,os;d=json.load(open(os.environ["F"]));print("%-6s %5.3f  net %7.1f  rmse %5.1f" % (os.path.basename(os.path.dirname(os.environ["F"]))[5:], d["flow_med_median_px"], d["net_dx_sum_px"], d["rmse_median"]))'
  else
    echo "$n: no flow"
  fi
done
