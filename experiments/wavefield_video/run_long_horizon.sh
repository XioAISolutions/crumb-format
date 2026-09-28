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
#   S_ssm   generic diagonal SSM (same T=128)
#   A_attn  causal attention, T=32 (windowed rollout)
#   E_half  (opt-in, HYBRID=1) local+wave hybrid, halflife poles, at T=T_ATTN:
#           its local path is full causal attention over the window, so at
#           T=128 x grid 32 (131k tokens) it would pay the N^2 bill it exists
#           to avoid. At T_ATTN it tests whether the halflife global state
#           carries memory past a gap longer than any it trained on.
#
# Pre-registered read (fixed before any result):
#   SEQ=1 adds W_half_seq[_cw|_wg] / S_ssm_seq (train_long.py): SEQ_FRAMES=512 sequences
#           in T_LONG chunks with carried state and a SEQ_GAP=256 gap mid-sequence.
#           PROVE (seq): W_half_seq exit-direction accuracy >= 0.8 on >= 2/3 seeds
#           with TBPTT=1 -> constant-memory training bridges gaps > one chunk.
#   PROVE  W_half exit-direction accuracy >= 0.8 and beats W_soft and
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
    "S_ssm|ssm|$T_LONG|"
    "A_attn|attn|$T_ATTN|"
)
if [ "${HYBRID:-0}" = "1" ]; then
    ARMS+=("E_half|wave|$T_ATTN|--kernel-version dispersion --fuse local_wave --pole-param halflife")
fi
# SEQ=1 (LONG_HORIZON.md 8): carried-state arms trained by train_long.py on
# SEQ_FRAMES-long sequences walked in T_LONG chunks, with a SEQ_GAP-frame
# occlusion mid-sequence -- a gap longer than any chunk, so only the carried
# state can bridge it. TBPTT = chunks of gradient through the carried state.
# --dense (every-position loss) is opt-in in train_long.py; used here because
# LONG_HORIZON.md 8.2 measured it better on 2/2 seeds (-3..-4% MSE/copy-last,
# 2x copy-ratio), though below the 10% bar that would make it a default.
SEQ_FRAMES=${SEQ_FRAMES:-512}; SEQ_GAP=${SEQ_GAP:-256}; TBPTT=${TBPTT:-1}
# WRITE (LONG_HORIZON.md 8.4): extra flags for the wave SEQ arm's write path.
# Default --clean-write: 8.4 measured D=128 recall 0.119 -> 0.993 at G=1 with it,
# and the ball scenes are blank-background. WRITE= (empty) = as built;
# WRITE=--write-gate failed 8.4 (0.119) and is kept only for comparison.
WRITE=${WRITE---clean-write}
# SEQ memory: chunk 128 x grid 32 x dim 128 wave training measured (CPU peak,
# per batch element) ~5.8 GB + 3.4 GB/extra layer without checkpointing, i.e.
# ~16 GB/sample at 4 layers -- BATCH=4 cannot fit 24 GB. --grad-ckpt keeps one
# layer's FFT spectra live (~5.9 + 0.9 GB/layer, ~8.6 GB/sample), so micro-
# batches of SEQ_MICRO=2 accumulate to the full BATCH. Lower to 1 on OOM.
SEQ_MICRO=${SEQ_MICRO:-2}
[ "${SEQ:-0}" = "only" ] && SEQ_ONLY=1
# Emergence happens 32 + gap frames into the eval rollout; it must fall inside it,
# or every emergence metric silently comes back null.
need=$(( 32 + GAP + 8 )); { [ "${SEQ:-0}" = "1" ] || [ "${SEQ_ONLY:-0}" = "1" ]; } && need=$(( 32 + (SEQ_GAP > GAP ? SEQ_GAP : GAP) + 8 ))
if [ "$EVAL_ROLLOUT" -lt "$need" ]; then
    echo "EVAL_ROLLOUT=$EVAL_ROLLOUT too short: emergence needs >= $need frames" >&2; exit 2
fi
if [ "${SEQ:-0}" = "only" ]; then ARMS=(); SEQ=1; fi    # SEQ=only: carried-state arms alone
if [ "${SEQ:-0}" = "1" ]; then
    # The write path changes the architecture, so it is part of the arm's identity:
    # W_half_seq_cw (clean write), _wg (gate); plain W_half_seq = as built. An OUT
    # from an older as-built run is never resumed into / mistaken for another.
    wtag=""; case " $WRITE " in *" --clean-write "*) wtag+="_cw";; esac
    case " $WRITE " in *" --write-gate "*) wtag+="_wg";; esac
    ARMS+=("W_half_seq${wtag}|wave|$T_LONG|--pole-param halflife $WRITE|long"
           "S_ssm_seq|ssm|$T_LONG||long")
fi

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
        IFS='|' read -r label kind frames extra trainer <<< "$arm"
        tag="_${label}_s${seed}"
        # Done only when BOTH artifacts exist: train_compare writes the result JSON
        # before model_*.pt, so a slice killed in between must re-run (it resumes
        # from the final ckpt and just re-saves).
        if [ -f "$OUT/result_${kind}${tag}.json" ] && [ -f "$OUT/model_${kind}${tag}.pt" ]; then
            echo "SKIP $tag"; continue
        fi
        extra_args=()
        if [ -n "$extra" ]; then read -r -a extra_args <<< "$extra"; fi
        if [ "$trainer" = "long" ]; then
            # gap centered in the sequence; eval gap of the same length in rollout
            s0=$(( (SEQ_FRAMES - SEQ_GAP) / 2 ))
            occ=(--train-occ-start "$s0" --train-occ-end $((s0 + SEQ_GAP))
                 --occ-start $((frames + 32)) --occ-end $((frames + 32 + SEQ_GAP)))
        else
            tgap=$(( GAP < frames - 8 ? GAP : frames - 8 ))
            occ=(--train-occ-start $((frames - tgap)) --train-occ-end "$frames"
                 --occ-start $((frames + 32)) --occ-end $((frames + 32 + GAP)))
        fi
        resume=()
        if [ -f "$OUT/ckpt_${kind}${tag}.pt" ]; then resume=(--resume "$OUT/ckpt_${kind}${tag}.pt"); fi
        left=$(( SLICE_S - ($(date +%s) - slice_t0) ))
        if [ "$left" -lt 120 ]; then
            echo "SLICED before $tag (budget spent) -- re-queue to continue" | tee -a "$OUT/progress.txt"
            echo "SLICED" > "$OUT/status.txt"; exit 0
        fi
        echo "== $tag $(date -u '+%Y-%m-%dT%H:%M:%SZ') ${resume[*]:-fresh}" | tee -a "$OUT/progress.txt"
        rc=0
        if [ "$trainer" = "long" ]; then
            ${TIMEOUT_BIN:+$TIMEOUT_BIN "$left"} "$PY" train_long.py --data-source occlusion \
                --grid "$GRID" --n-balls 6 --batch "$BATCH" --dim "$DIM" --layers "$LAYERS" \
                --heads "$HEADS" --motion-loss --dense --seq-frames "$SEQ_FRAMES" --chunk "$frames" \
                --grad-ckpt --micro-batch "$SEQ_MICRO" \
                --tbptt-chunks "$TBPTT" "${occ[@]}" --eval-rollout "$EVAL_ROLLOUT" \
                --eval-seeds "$EVAL_SEEDS" --eval-chunk 1 --save-every 250 \
                ${extra_args[@]+"${extra_args[@]}"} ${resume[@]+"${resume[@]}"} \
                --kind "$kind" --seed "$seed" --steps "$STEPS" \
                --out "$OUT" --tag "$tag" >> "$OUT/log${tag}.txt" 2>&1 || rc=$?
        else
            ${TIMEOUT_BIN:+$TIMEOUT_BIN "$left"} "$PY" train_compare.py "${BASE[@]}" "${occ[@]}" \
                ${extra_args[@]+"${extra_args[@]}"} ${resume[@]+"${resume[@]}"} \
                --kind "$kind" --frames "$frames" --seed "$seed" --steps "$STEPS" \
                --out "$OUT" --tag "$tag" >> "$OUT/log${tag}.txt" 2>&1 || rc=$?
        fi
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
    IFS='|' read -r label kind frames _ trainer <<< "$arm"
    [ "$kind" = "wave" ] || continue
    tpos=table; [ "$trainer" = "long" ] && tpos=none
    f="$OUT/model_wave_${label}_s${seed}.pt"
    if [ ! -f "$f" ]; then
        echo "MISSING $f" | tee -a "$OUT/progress.txt"
        echo "FAILED missing $(basename "$f")" > "$OUT/status.txt"; exit 1
    fi
    [ -f "${f%.pt}_stream${STREAM_FRAMES}.json" ] && continue
    pp=softplus; [[ "$f" == *_half_* ]] && pp=halflife
    fuse=none; [[ "$(basename "$f")" == model_wave_E_* ]] && fuse=local_wave
    # Streams share the slice budget too: stop at the boundary rather than let
    # the conductor kill a half-done stream (re-queue resumes at this stream).
    left=$(( SLICE_S - ($(date +%s) - slice_t0) ))
    if [ "$left" -lt 300 ]; then
        echo "SLICED before stream $(basename "$f") (budget spent) -- re-queue" | tee -a "$OUT/progress.txt"
        echo "SLICED" > "$OUT/status.txt"; exit 0
    fi
    # FFN width is read from the checkpoint itself (no rounded ffn_mult).
    rc=0
    ${TIMEOUT_BIN:+$TIMEOUT_BIN "$left"} "$PY" long_horizon.py stream --pole-param "$pp" --ckpt "$f" \
        --grid "$GRID" --frames "$frames" --dim "$DIM" --layers "$LAYERS" --heads "$HEADS" \
        --fuse "$fuse" --time-pos "$tpos" --stream-frames "$STREAM_FRAMES" --chunk 600 \
        --out "${f%.pt}_stream${STREAM_FRAMES}.json" > "${f%.pt}_stream${STREAM_FRAMES}.log" 2>&1 || rc=$?
    if [ "$rc" = "124" ] || [ "$rc" = "143" ]; then
        rm -f "${f%.pt}_stream${STREAM_FRAMES}.json"
        echo "SLICED stream $(basename "$f") at the ${SLICE_S}s budget -- re-queue" | tee -a "$OUT/progress.txt"
        echo "SLICED" > "$OUT/status.txt"; exit 0
    elif [ "$rc" != "0" ]; then
        echo "STREAM FAIL $f exit=$rc" | tee -a "$OUT/progress.txt"
        echo "FAILED stream $(basename "$f")" > "$OUT/status.txt"; exit 1
    fi
  done
done
echo DONE > "$OUT/status.txt"
