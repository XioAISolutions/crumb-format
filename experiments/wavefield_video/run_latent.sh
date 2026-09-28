#!/usr/bin/env bash
# LONG_HORIZON.md phase 2 -- the wave model on a pretrained video VAE's latents.
#
#   1. encode   VIDEOS (a folder of mp4s) -> LATENTS shards with VAE (default LTX:
#               8x time, 32x space, 128 ch; 5 min at 24 fps = 900 latent steps)
#   2. train    train_long.py on SEQ-step latent sequences walked in CHUNK-step
#               chunks with carried state (dense next-latent loss)
#   3. stream   STREAM_STEPS latent steps from real held-out context, decoded chunk
#               by chunk; HealthMonitor judges the DECODED pixels
#
# Pre-registered read (fixed before any result):
#   KILL   fade or flatten fires before latent step 256 (256 x 8 frames / 24 fps
#          ~= 85 s of LTX video) on the held-out stream -> the latent line is not
#          ready for minutes.
#   PASS   no fade/flatten through STREAM_STEPS (default 900 = 5 min through LTX);
#          then judge the decoded frames by eye before any claim.
# Resumable like run_long_horizon.sh: re-run until $OUT/status.txt reads DONE.
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PY=${PY:-/workspace/slava/comfy-house/venv/bin/python}
VIDEOS=${VIDEOS:?set VIDEOS=/path/to/mp4s}
VAE=${VAE:-ltx}; VAE_PATH=${VAE_PATH:-}          # empty -> the preset's HF id
HEIGHT=${HEIGHT:-256}; WIDTH=${WIDTH:-448}; FPS=${FPS:-24}; MAX_FRAMES=${MAX_FRAMES:-513}
OUT=${OUT:-runs_latent}; LATENTS=${LATENTS:-$OUT/latents}
SEQ=${SEQ:-64}; CHUNK=${CHUNK:-16}; TBPTT=${TBPTT:-1}
DIM=${DIM:-256}; LAYERS=${LAYERS:-6}; HEADS=${HEADS:-8}; BATCH=${BATCH:-8}
STEPS=${STEPS:-20000}; SEED=${SEED:-0}; STREAM_STEPS=${STREAM_STEPS:-900}
SLICE_S=${SLICE_S:-4800}
mkdir -p "$OUT"
command -v "$PY" >/dev/null 2>&1 || { echo "Python not executable: $PY" >&2; exit 2; }
TIMEOUT_BIN=$(command -v timeout || true)
vae_args=(--vae "$VAE"); [ -n "$VAE_PATH" ] && vae_args+=(--vae-path "$VAE_PATH")
slice_t0=$(date +%s)
left() { echo $(( SLICE_S - ($(date +%s) - slice_t0) )); }
sliced() { echo "SLICED $1 -- re-queue" | tee -a "$OUT/progress.txt"; echo SLICED > "$OUT/status.txt"; exit 0; }
failed() { echo "FAILED $1" | tee -a "$OUT/progress.txt"; echo "FAILED $1" > "$OUT/status.txt"; exit 1; }
run() {  # run <label> <cmd...> under the remaining slice budget
    local label=$1; shift
    local l; l=$(left); [ "$l" -lt 120 ] && sliced "before $label"
    local rc=0
    ${TIMEOUT_BIN:+$TIMEOUT_BIN "$l"} "$@" >> "$OUT/log_${label}.txt" 2>&1 || rc=$?
    { [ "$rc" = "124" ] || [ "$rc" = "143" ]; } && sliced "$label at budget"
    [ "$rc" = "0" ] || failed "$label exit=$rc (see $OUT/log_${label}.txt)"
}
# Run identity: every stage below skips work whose outputs exist, so an OUT must
# never be reused with different settings (it would report the old run as DONE).
cfg="VIDEOS=$VIDEOS VAE=$VAE VAE_PATH=$VAE_PATH HEIGHT=$HEIGHT WIDTH=$WIDTH FPS=$FPS MAX_FRAMES=$MAX_FRAMES LATENTS=$LATENTS SEQ=$SEQ CHUNK=$CHUNK TBPTT=$TBPTT DIM=$DIM LAYERS=$LAYERS HEADS=$HEADS BATCH=$BATCH STEPS=$STEPS SEED=$SEED STREAM_STEPS=$STREAM_STEPS"
if [ -f "$OUT/run_config.txt" ] && [ "$(cat "$OUT/run_config.txt")" != "$cfg" ]; then
    echo "$OUT holds a run with different settings:" >&2
    diff <(tr ' ' '\n' < "$OUT/run_config.txt") <(tr ' ' '\n' <<< "$cfg") >&2 || true
    echo "use a new OUT (or delete it) instead of mixing runs" >&2
    exit 2
fi
echo "$cfg" > "$OUT/run_config.txt"
echo RUNNING > "$OUT/status.txt"

# 1. encode (skipped once index.json exists; a sliced encode resumes at the next
#    unencoded segment via LATENTS/progress.jsonl; delete LATENTS to re-encode)
if [ ! -f "$LATENTS/index.json" ]; then
    echo "== encode $(date -u +%FT%TZ)" | tee -a "$OUT/progress.txt"
    run encode "$PY" encode_videos.py --videos "$VIDEOS" "${vae_args[@]}" --height "$HEIGHT" \
        --width "$WIDTH" --fps "$FPS" --max-frames "$MAX_FRAMES" --out "$LATENTS"
fi

# 2. train (result + model both present = done; else resume from ckpt)
tag="_latent_s${SEED}"
if ! { [ -f "$OUT/result_wave${tag}.json" ] && [ -f "$OUT/model_wave${tag}.pt" ]; }; then
    resume=(); [ -f "$OUT/ckpt_wave${tag}.pt" ] && resume=(--resume "$OUT/ckpt_wave${tag}.pt")
    echo "== train $(date -u +%FT%TZ) ${resume[*]:-fresh}" | tee -a "$OUT/progress.txt"
    run train "$PY" train_long.py --latents "$LATENTS" --kind wave --pole-param halflife \
        --seq-frames "$SEQ" --chunk "$CHUNK" --tbptt-chunks "$TBPTT" --dense --dim "$DIM" \
        --layers "$LAYERS" --heads "$HEADS" --batch "$BATCH" --steps "$STEPS" --seed "$SEED" \
        --eval-rollout 256 --save-every 500 --save-every-sec 600 ${resume[@]+"${resume[@]}"} --out "$OUT" --tag "$tag"
fi

# 3. decoded long stream
js="$OUT/stream${tag}_${STREAM_STEPS}.json"
if [ ! -f "$js" ]; then
    echo "== stream $(date -u +%FT%TZ)" | tee -a "$OUT/progress.txt"
    run stream "$PY" long_horizon.py stream --pole-param halflife --time-pos none \
        --ckpt "$OUT/model_wave${tag}.pt" --latents "$LATENTS" "${vae_args[@]}" \
        --frames "$CHUNK" --dim "$DIM" --layers "$LAYERS" --heads "$HEADS" \
        --stream-frames "$STREAM_STEPS" --chunk 32 --batch 2 --checkpoint "$OUT/stream${tag}.state" \
        --out "$js.tmp"
    mv "$js.tmp" "$js"
fi
echo DONE > "$OUT/status.txt"
"$PY" -c '
import json, sys
d = json.load(open(sys.argv[1]))
c = d.get("collapse_latent_step") or {}
kill = {k: v for k, v in c.items() if k in ("fade", "flatten")}
print("pre-registered (fade/flatten, KILL if < 256):", kill or "none",
      "->", "KILL" if any(v < 256 for v in kill.values()) else "not killed")
print("other health flags (not part of the rule):",
      {k: v for k, v in c.items() if k not in kill} or "none",
      "| decoded frames:", d["log"][-1]["decoded_frames"])' "$js"
