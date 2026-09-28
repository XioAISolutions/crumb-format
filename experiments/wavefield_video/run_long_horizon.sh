#!/usr/bin/env bash
# LONG_HORIZON.md -- occlusion memory with the gap INSIDE the training window.
#
# run_occlusion.sh trains on 17-frame clips that start at frame 0 while the target
# is hidden on [64, 320): no arm ever sees an occluder during training, so its
# emergence metrics cannot separate the arms. Here every arm trains with the
# target hidden on [T-GAP, T) of its own clip, so frame T -- the loss target -- is
# the first frame after a GAP-frame occlusion and the loss REQUIRES memory. The
# wave/ssm arms train at T=128 (affordable only because the FFT forward is
# O(T log T)); attention keeps the window it can afford (T=32, so its training gap
# is capped at T-8). Eval hides the target on [T+32, T+32+GAP) during rollout.
#
# Arms (grid 32, equal --target-params):
#   W_soft  wave dispersion, softplus poles (v2 default; init half-life ~0.7 frame)
#   W_half  wave dispersion, halflife poles in [2, 4096] frames
#   E_half  local+wave hybrid, halflife poles
#   S_ssm   generic diagonal SSM (same T=128)
#   A_attn  causal attention, T=32 (windowed rollout)
#
# Pre-registered read (fixed before any result):
#   PROVE  W_half or E_half exit-direction accuracy >= 0.8 and beats W_soft and
#          A_attn by >= 0.2 on >= 2/3 seeds  -> scale the gap to 256 then 1024.
#   KILL   halflife arms <= W_soft on exit-direction accuracy on >= 2/3 seeds ->
#          long poles are not what limits memory; stop the pole line.
# Not yet run on GPU: do a preflight first (STEPS=20 SEEDS=0) to size batch/VRAM.
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PY=${PY:-/workspace/slava/comfy-house/venv/bin/python}
OUT=${OUT:-runs_long_horizon}
STEPS=${STEPS:-4000}
SEEDS=${SEEDS:-0 1 2}
T_LONG=${T_LONG:-128}
T_ATTN=${T_ATTN:-32}
GAP=${GAP:-64}
mkdir -p "$OUT"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Python not executable: $PY (override with PY=/path/to/python)" >&2; exit 2
fi

GRID=${GRID:-32}; DIM=${DIM:-128}; LAYERS=${LAYERS:-4}; HEADS=${HEADS:-8}; BATCH=${BATCH:-4}
TARGET_PARAMS=${TARGET_PARAMS:-2000000}; EVAL_ROLLOUT=${EVAL_ROLLOUT:-512}; EVAL_SEEDS=${EVAL_SEEDS:-8}
STREAM_FRAMES=${STREAM_FRAMES:-7200}
BASE=(--data-source occlusion --grid "$GRID" --n-balls 6 --batch "$BATCH"
      --target-params "$TARGET_PARAMS" --causal --residual --linear-pad
      --dim "$DIM" --layers "$LAYERS" --heads "$HEADS" --ckpt
      --motion-loss --rollout-loss 1 --const-lr
      --eval-rollout "$EVAL_ROLLOUT" --eval-seeds "$EVAL_SEEDS" --eval-chunk 1
      --auto-batch --save-every 250)

ARMS=(
    "W_soft|wave|$T_LONG|--kernel-version dispersion --pole-param softplus"
    "W_half|wave|$T_LONG|--kernel-version dispersion --pole-param halflife"
    "E_half|wave|$T_LONG|--kernel-version dispersion --fuse local_wave --pole-param halflife"
    "S_ssm|ssm|$T_LONG|"
    "A_attn|attn|$T_ATTN|"
)

# Sliced + resumable (the box conductor kills jobs at 5400s): finished arms are
# skipped, a live arm resumes from its ckpt, the whole slice is capped at SLICE_S
# wall seconds (arms share the budget), and the script stops at the boundary. Re-queue the same
# command until $OUT/status.txt reads DONE. Keep STEPS fixed across slices.
SLICE_S=${SLICE_S:-4800}
TIMEOUT_BIN=$(command -v timeout || true)
echo RUNNING > "$OUT/status.txt"
slice_t0=$(date +%s)
for seed in $SEEDS; do
    for arm in "${ARMS[@]}"; do
        IFS='|' read -r label kind frames extra <<< "$arm"
        tag="_${label}_s${seed}"
        # Done only when BOTH artifacts exist: train_compare writes the result JSON
        # before model_*.pt, so a slice killed in between must re-run (it resumes
        # from the final ckpt and just re-saves).
        if [ -f "$OUT/result_${kind}${tag}.json" ] && [ -f "$OUT/model_${kind}${tag}.pt" ]; then
            echo "SKIP $tag"; continue
        fi
        extra_args=()
        if [ -n "$extra" ]; then read -r -a extra_args <<< "$extra"; fi
        tgap=$(( GAP < frames - 8 ? GAP : frames - 8 ))
        occ=(--train-occ-start $((frames - tgap)) --train-occ-end "$frames"
             --occ-start $((frames + 32)) --occ-end $((frames + 32 + GAP)))
        resume=()
        if [ -f "$OUT/ckpt_${kind}${tag}.pt" ]; then resume=(--resume "$OUT/ckpt_${kind}${tag}.pt"); fi
        left=$(( SLICE_S - ($(date +%s) - slice_t0) ))
        if [ "$left" -lt 120 ]; then
            echo "SLICED before $tag (budget spent) -- re-queue to continue" | tee -a "$OUT/progress.txt"
            echo "SLICED" > "$OUT/status.txt"; exit 0
        fi
        echo "== $tag $(date -u '+%Y-%m-%dT%H:%M:%SZ') ${resume[*]:-fresh}" | tee -a "$OUT/progress.txt"
        rc=0
        ${TIMEOUT_BIN:+$TIMEOUT_BIN "$left"} "$PY" train_compare.py "${BASE[@]}" "${occ[@]}" \
            ${extra_args[@]+"${extra_args[@]}"} ${resume[@]+"${resume[@]}"} \
            --kind "$kind" --frames "$frames" --seed "$seed" --steps "$STEPS" \
            --out "$OUT" --tag "$tag" >> "$OUT/log${tag}.txt" 2>&1 || rc=$?
        if [ "$rc" = "124" ] || [ "$rc" = "143" ]; then
            echo "SLICED $tag at the ${SLICE_S}s slice budget -- re-queue to continue" | tee -a "$OUT/progress.txt"
            echo "SLICED" > "$OUT/status.txt"
            exit 0
        fi
        if [ "$rc" != "0" ]; then
            # Never mark the suite DONE with an arm missing: slices no-op on DONE.
            # FAILED stops here; fix the cause, then re-queue (finished arms skip).
            echo "FAIL $tag exit=$rc (see $OUT/log${tag}.txt)" | tee -a "$OUT/progress.txt"
            echo "FAILED $tag exit=$rc" > "$OUT/status.txt"
            exit 1
        fi
    done
done

# 5-minute health stream on each trained wave arm (constant state, collapse flags).
# Every wave arm x every configured seed; a missing model here means the suite is
# incomplete, so it is a failure rather than a silent skip.
for seed in $SEEDS; do
  for arm in "${ARMS[@]}"; do
    IFS='|' read -r label kind _ _ <<< "$arm"
    [ "$kind" = "wave" ] || continue
    f="$OUT/model_wave_${label}_s${seed}.pt"
    if [ ! -f "$f" ]; then
        echo "MISSING $f" | tee -a "$OUT/progress.txt"
        echo "FAILED missing $(basename "$f")" > "$OUT/status.txt"; exit 1
    fi
    [ -f "${f%.pt}_stream7200.json" ] && continue
    pp=softplus; [[ "$f" == *_half_* ]] && pp=halflife
    fuse=none; [[ "$(basename "$f")" == model_wave_E_* ]] && fuse=local_wave
    res="$OUT/result_$(basename "${f#*model_}")"; res="${res%.pt}.json"
    ffn=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ffn_mult"])' "$res")
    "$PY" long_horizon.py stream --pole-param "$pp" --ckpt "$f" --grid "$GRID" --frames "$T_LONG" \
        --dim "$DIM" --layers "$LAYERS" --heads "$HEADS" --ffn-mult "$ffn" --fuse "$fuse" \
        --stream-frames "$STREAM_FRAMES" --chunk 600 \
        --out "${f%.pt}_stream7200.json" > "${f%.pt}_stream7200.log" 2>&1 || { echo "STREAM FAIL $f" | tee -a "$OUT/progress.txt"
             echo "FAILED stream $(basename "$f")" > "$OUT/status.txt"; exit 1; }
  done
done
echo DONE > "$OUT/status.txt"
