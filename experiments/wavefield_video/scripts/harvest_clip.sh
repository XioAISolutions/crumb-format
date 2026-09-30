#!/bin/bash
# Harvest + score a LongLive clip: fetch (optional), flow-check, contact sheet, preview, summary.
# Usage: harvest_clip.sh <name> [remote_len_dir]
#   name: e.g. c01s_180s ; remote_len_dir: e.g. /workspace/slava/exp/pr63/runs_longlive_cfg16i/len_180s
# If remote_len_dir omitted, use an existing rank0-0-0_regular.mp4 in the output dir.
# Outputs: ~/.hermes/cache/scratch/clip_harvest/<name>/{mp4,eval.json,contact_9.jpg,preview.mp4,flow.json}
set -euo pipefail
NAME=$1; RDIR=${2:-}
S=/Users/slavaz/.hermes/cache/scratch
H=$S/clip_harvest/$NAME
mkdir -p "$H"
if [ -n "$RDIR" ]; then
  scp -q -P 41400 root@161.184.224.50:"$RDIR/rank0-0-0_regular.mp4" "$H/"
  scp -q -P 41400 root@161.184.224.50:"$RDIR/rank0-0-0_regular.eval.json" "$H/" 2>/dev/null || true
fi
MP4="$H/rank0-0-0_regular.mp4"
[ -f "$MP4" ] || { echo "NO MP4 at $MP4"; exit 1; }
echo "--- probe"
ffprobe -v error -show_entries format=duration,size:stream=width,height,nb_frames -of default=nw=1 "$MP4"
N=$(ffprobe -v error -show_entries stream=nb_frames -of csv=p=0 "$MP4")
STEP=$(( N / 9 )); [ "$STEP" -lt 1 ] && STEP=1
echo "--- contact sheet (every $STEP frames)"
ffmpeg -y -v error -i "$MP4" -vf "select=not(mod(n\,$STEP)),scale=480:-1,tile=3x3" -frames:v 1 "$H/contact_9.jpg"
echo "--- preview encode"
ffmpeg -y -v error -i "$MP4" -c:v libx264 -crf 26 -preset fast -movflags +faststart "$H/preview.mp4"
echo "--- flow score"
"$S/venv_flow/bin/python" "$S/flow_check.py" "$MP4" | tee "$H/flow.json"
ls -la "$H"
