#!/bin/bash
# Fetch a clip from the box + score it. Usage: fetch_and_score.sh <name> <remote_len_dir>
set -euo pipefail
NAME=$1; RDIR=$2
S=/Users/slavaz/.hermes/cache/scratch
H=$S/clip_harvest/$NAME
mkdir -p "$H"
echo "=== fetch $NAME $(date -u +%FT%TZ)"
scp -q -P 41400 root@161.184.224.50:"$RDIR/rank0-0-0_regular.mp4" "$H/"
scp -q -P 41400 root@161.184.224.50:"$RDIR/rank0-0-0_regular.eval.json" "$H/" 2>/dev/null || true
echo "=== score $NAME"
bash "$S/harvest_clip.sh" "$NAME" 2>&1 | tail -30
echo "=== done $NAME $(date -u +%FT%TZ)"
