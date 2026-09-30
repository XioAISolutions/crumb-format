#!/bin/bash
# Fetch + score the six wave-2 probes; print scorecard.
set -euo pipefail
S=/Users/slavaz/.hermes/cache/scratch; cd "$S"
R=/workspace/slava/exp/pr63
NAMES="bf16 seed1 seed2 seed3 fastprompt i2v"
for n in $NAMES; do
  H=$S/clip_harvest/c01v_$n
  [ -d "$H" ] && continue
  bash "$S/fetch_and_score.sh" "c01v_$n" "$R/runs_probe_$n/len_10s" >/dev/null 2>&1 || echo "$n: FETCH/SCORE FAILED"
done
echo "=== WAVE-2 SCORECARD (median px/frame @640w) ==="
for n in $NAMES; do
  f="$S/clip_harvest/c01v_$n/flow.json"
  if [ -s "$f" ]; then
    F="$f" "$S/venv_flow/bin/python" -c 'import json,os;d=json.load(open(os.environ["F"]));print("%-11s %5.3f  net %7.1f  rmse %5.1f" % (os.path.basename(os.path.dirname(os.environ["F"]))[5:], d["flow_med_median_px"], d["net_dx_sum_px"], d["rmse_median"]))'
  else
    echo "$n: no flow"
  fi
done
echo "refs: W12rel 0.404/-24.8 | W32 0.569/-12.6 | forest 0.500/-2.0 | walk target >5"
