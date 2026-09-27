#!/usr/bin/env bash
# Attention-vs-spine difficulty-ladder suite (card t_d77fd4cd, "Use attention
# actively"). Runs the `--kind attn` arm against the wave spine (`--kind wave`)
# at every rung of the difficulty ladder from run_difficulty.sh, at the CURRENT
# unfreeze recipe (--const-lr + 8000 steps; budget and schedule scale together,
# see references/freeze-diagnosis-suite.md), focusing on rung D6 (1024-frame
# horizon, --eval-rollout 1024).
#
# Rungs (identical flags to run_difficulty.sh; rung identity wins over anything
# else on the command line because they are appended last, argparse last-wins):
#   D1  --speed 2.30                     | D4  --radius 0.8
#   D2  --n-balls 8                      | D5  --grid 64 --n-balls 8
#   D3  --n-balls 12 --speed 2.30        | D6  --grid 32 --eval-rollout 1024
#
# Base condition: --speed 2.30 (see IMPL_NOTES_ATTN_LADDER.md, "base condition").
# run_difficulty.sh's base default was 1.15, but every 8k-era run this suite is
# compared against (runs_deep B2, freeze S2, hybrid8 H8*) ran at 2.30; the D1/D3
# speed overrides are kept for lineage and are now no-ops.
#
# D5 is a pair at g64 and is intentionally NOT in the default RUNGS: attention
# at g64 costs ~16x its g32 compute (N^2 over 4x the tokens/window), ~8-26 h to
# train at 8k steps vs ~1.5 h at g32; it runs only under the pre-registered
# condition implemented in attn_ladder_report.py --d5-guard and fired by the
# gpuq_job_5xx_attn_d5_cond.sh band (RUNGS="D5"). Do not hand-run D5 without
# the guard's TRIGGER verdict; see IMPL_NOTES_ATTN_LADDER.md.
#
# Units = RUNGS x arms{attn,wave} x SEEDS, rung-major with attn first so the
# D6 attention half lands first. Seed-outer (all rungs per seed) so a partial
# suite still holds every rung for the seeds it reached.
#
# ENV knobs (same contract as run_occlusion.sh R15 slicing):
#   OUT       output dir (default runs_attn_ladder)
#   STEPS     training steps per arm (default 8000, the unfreeze recipe)
#   CONST_LR  1 = append --const-lr (default 1 here; this suite IS the recipe)
#   RESUME    1 = sliced-queue mode: skip arms with a result JSON, resume live
#             ones from their ckpt, cap each arm at SLICE_S wall seconds and
#             stop at the slice boundary (status "SLICED done/planned").
#             Run the same command again until DONE. Keep STEPS/CONST_LR fixed
#             across slices of one OUT (resume needs the same schedule target).
#   SLICE_S   per-arm wall cap in RESUME mode, seconds (default 4200; the box
#             conductor kills whole jobs at 5400 s)
#   RUNGS     space-separated rung tokens (default "D6 D1 D2 D3 D4")
#   SEEDS     space-separated --seed values (default "0")
#   PY        python with torch (default box venv)
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PY=${PY:-/workspace/slava/comfy-house/venv/bin/python}
OUT=${OUT:-runs_attn_ladder}
STEPS=${STEPS:-8000}
CONST_LR=${CONST_LR:-1}
RESUME=${RESUME:-0}
SLICE_S=${SLICE_S:-4200}
RUNGS=${RUNGS:-"D6 D1 D2 D3 D4"}
SEEDS=${SEEDS:-0}
if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS must be a positive integer" >&2
    exit 2
fi
if [[ ! "$SLICE_S" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLICE_S must be a positive integer" >&2
    exit 2
fi
for rung in $RUNGS; do
    case "$rung" in
        D1|D2|D3|D4|D5|D6) ;;
        *) echo "unknown rung token: $rung (expected D1..D6)" >&2; exit 2 ;;
    esac
done
mkdir -p "$OUT"
PROGRESS="$OUT/attn_ladder_progress.txt"
STATUS="$OUT/attn_ladder_status.txt"
if [ "$RESUME" = "1" ]; then
    printf 'attn-ladder slice %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$PROGRESS"
else
    printf 'attn-ladder start %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$PROGRESS"
fi
printf 'RUNNING\n' > "$STATUS"
finished=0
failures=0
total=0
completed=0
sliced=0
on_exit() {
    local rc=$?
    if (( ! finished )); then
        printf 'FAILED exit=%s\n' "$rc" > "$STATUS"
        printf 'attn-ladder aborted exit=%s %s\n' "$rc" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$PROGRESS"
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Python not executable: $PY (override with PY=/path/to/python)" >&2
    exit 2
fi

# Ladder base = run_difficulty.sh BASE + the freeze-suite unfreeze recipe.
# Kept in an array so paths/arguments are never expanded as shell commands or
# filename patterns.
BASE=(--kernel-version dispersion --gate --local-fuse --speed 2.30
      --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16
      --target-params 4000000 --causal --residual --linear-pad
      --collisions --motion-loss --rollout-loss 1
      --eval-rollout 256 --eval-seeds 16 --eval-chunk 4
      --auto-batch --save-every 500 --telemetry-every 25)
if [ "$CONST_LR" = "1" ]; then BASE+=(--const-lr); fi

# rung token -> rung-specific flags (must match run_difficulty.sh)
rung_args() {
    local rung=$1
    case "$rung" in
        D1) printf '%s\n' "--speed" "2.30" ;;
        D2) printf '%s\n' "--n-balls" "8" ;;
        D3) printf '%s\n' "--n-balls" "12" "--speed" "2.30" ;;
        D4) printf '%s\n' "--radius" "0.8" ;;
        D5) printf '%s\n' "--grid" "64" "--n-balls" "8" ;;
        D6) printf '%s\n' "--grid" "32" "--eval-rollout" "1024" ;;
    esac
}

TIMEOUT_BIN=""
if [ "$RESUME" = "1" ]; then
    TIMEOUT_BIN=$(command -v timeout || command -v gtimeout || true)
    if [ -z "$TIMEOUT_BIN" ]; then
        echo "RESUME=1: no 'timeout' binary found; running uncapped (the slice" \
             "boundary then comes from the parent job's own cap)" >&2
    fi
fi

run() {
    local kind=$1
    shift
    local tag=$1
    shift
    local log_file="$OUT/log_attn_ladder_${kind}${tag}.txt"
    local resume_args=()
    if [ "$RESUME" = "1" ]; then
        if [ -f "$OUT/result_${kind}${tag}.json" ]; then
            total=$((total + 1))
            completed=$((completed + 1))
            printf 'SKIP %s%s result-exists %s\n' "$kind" "$tag" \
                "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$PROGRESS"
            return 0
        fi
        if [ -f "$OUT/ckpt_${kind}${tag}.pt" ]; then
            resume_args=(--resume "$OUT/ckpt_${kind}${tag}.pt")
        fi
    fi
    local rc=0
    local command=("$PY" train_compare.py "${BASE[@]}" "$@"
                   ${resume_args[@]+"${resume_args[@]}"}
                   --kind "$kind" --steps "$STEPS" --out "$OUT" --tag "$tag")
    total=$((total + 1))
    printf '== %s%s %s\n' "$kind" "$tag" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$PROGRESS"
    printf 'command:' >> "$PROGRESS"
    printf ' %q' "${command[@]}" >> "$PROGRESS"
    printf '\n' >> "$PROGRESS"
    if [ "$RESUME" = "1" ] && [ -n "$TIMEOUT_BIN" ]; then
        "$TIMEOUT_BIN" "$SLICE_S" "${command[@]}" > "$log_file" 2>&1 || rc=$?
    else
        if "${command[@]}" > "$log_file" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    fi
    if [ "$RESUME" = "1" ] && { [ "$rc" = "124" ] || [ "$rc" = "143" ]; }; then
        printf '%s%s SLICED wall=%ss %s log=%s\n' "$kind" "$tag" "$SLICE_S" \
            "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$log_file" | tee -a "$PROGRESS"
        return 3
    fi
    if [ "$rc" = "0" ]; then
        completed=$((completed + 1))
    else
        failures=$((failures + 1))
    fi
    printf '%s%s exit=%s %s log=%s\n' "$kind" "$tag" "$rc" \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$log_file" | tee -a "$PROGRESS"
    return 0
}

nseeds=0
for _seed in $SEEDS; do nseeds=$((nseeds + 1)); done
nrungs=0
for _rung in $RUNGS; do nrungs=$((nrungs + 1)); done
units=$(( nrungs * 2 * nseeds ))
for seed in $SEEDS; do
    for rung in $RUNGS; do
        rung_arr=()
        while IFS= read -r token; do rung_arr+=("$token"); done < <(rung_args "$rung")
        if ! run attn "_${rung}" "${rung_arr[@]}" --seed "$seed"; then
            sliced=1
            break
        fi
        if ! run wave "_${rung}" "${rung_arr[@]}" --seed "$seed"; then
            sliced=1
            break
        fi
    done
    if (( sliced )); then break; fi
done

if (( sliced )); then
    printf 'SLICED %s/%s\n' "$completed" "$units" > "$STATUS"
    finished=1
    exit 0
fi
if (( failures )); then
    printf 'FAILED %s/%s\n' "$failures" "$total" > "$STATUS"
    finished=1
    exit 1
fi
printf 'DONE %s/%s\n' "$total" "$total" > "$STATUS"
finished=1
