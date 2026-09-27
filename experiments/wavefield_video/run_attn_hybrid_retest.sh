#!/usr/bin/env bash
# Hybrid attention+wave retest at the current unfreeze recipe (card t_d77fd4cd
# item 2; prior H8wav 0.428 vs plain wave 0.728, IMPL_NOTES_HYBRID.md).
#
# The decisive matched-budget five-way (IMPL_NOTES_HYBRID.md sec 1) on balls at
# g32, ALL ARMS at the same ~4M parameter budget and the same recipe
# (--const-lr + 8000 steps), so the only thing that differs is the temporal
# mechanism:
#
#   local_wave  --kind wave --kernel-version dispersion --fuse local_wave
#               (gated fusion: causal-attention window + structured wave state)
#   local_ssm   --kind ssm  --fuse local_ssm   (same fusion, generic diagonal SSM)
#   wave_only   --kind wave --kernel-version dispersion     (the spine control)
#   attn_only   --kind attn                                 (local-only control)
#   ssm_only    --kind ssm                                  (generic control)
#
# Hybrids run first so the retest read lands before the two extra controls.
# --telemetry-every 25 is in BASE: the mean gate per layer (TELE gate_g=...) is
# the documented kill signal (g -> 0.9+ => the persistent state is dead weight).
#
# ENV knobs (same contract as run_attn_ladder.sh / run_occlusion.sh):
#   OUT, STEPS (default 8000), CONST_LR (default 1), RESUME, SLICE_S, SEEDS, PY
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PY=${PY:-/workspace/slava/comfy-house/venv/bin/python}
OUT=${OUT:-runs_hybrid_retest}
STEPS=${STEPS:-8000}
CONST_LR=${CONST_LR:-1}
RESUME=${RESUME:-0}
SLICE_S=${SLICE_S:-4200}
SEEDS=${SEEDS:-0}
if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS must be a positive integer" >&2
    exit 2
fi
if [[ ! "$SLICE_S" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLICE_S must be a positive integer" >&2
    exit 2
fi
mkdir -p "$OUT"
PROGRESS="$OUT/hybrid_retest_progress.txt"
STATUS="$OUT/hybrid_retest_status.txt"
if [ "$RESUME" = "1" ]; then
    printf 'hybrid-retest slice %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$PROGRESS"
else
    printf 'hybrid-retest start %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$PROGRESS"
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
        printf 'hybrid-retest aborted exit=%s %s\n' "$rc" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$PROGRESS"
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Python not executable: $PY (override with PY=/path/to/python)" >&2
    exit 2
fi

# Base = the hybrid/balls matched-budget recipe (IMPL_NOTES_HYBRID.md sec 1) at
# the unfreeze recipe settings; base data condition --speed 2.30 to match
# H8wav/H8ssm exactly (see IMPL_NOTES_ATTN_LADDER.md). Kept in an array: no
# re-expansion of args.
BASE=(--kernel-version dispersion --gate --local-fuse --speed 2.30
      --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16
      --target-params 4000000 --causal --residual --linear-pad
      --collisions --motion-loss --rollout-loss 1
      --eval-rollout 256 --eval-seeds 16 --eval-chunk 4
      --auto-batch --save-every 500 --telemetry-every 25)
if [ "$CONST_LR" = "1" ]; then BASE+=(--const-lr); fi

# arm label | --kind | arm-specific flags (tag = _<label>)
ARMS=(
    "local_wave|wave|--kernel-version dispersion --fuse local_wave"
    "local_ssm|ssm|--fuse local_ssm"
    "wave_only|wave|--kernel-version dispersion"
    "attn_only|attn|"
    "ssm_only|ssm|"
)

TIMEOUT_BIN=""
if [ "$RESUME" = "1" ]; then
    TIMEOUT_BIN=$(command -v timeout || command -v gtimeout || true)
    if [ -z "$TIMEOUT_BIN" ]; then
        echo "RESUME=1: no 'timeout' binary found; running uncapped (the slice" \
             "boundary then comes from the parent job's own cap)" >&2
    fi
fi

run() {
    local tag=$1 kind=$2
    shift 2
    local log_file="$OUT/log_hybrid_retest_${kind}${tag}.txt"
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
units=$(( ${#ARMS[@]} * nseeds ))
for seed in $SEEDS; do
    for spec in "${ARMS[@]}"; do
        IFS='|' read -r label kind extra <<< "$spec"
        extra_arr=()
        if [[ -n "$extra" ]]; then read -r -a extra_arr <<< "$extra"; fi
        if ! run "_${label}" "$kind" ${extra_arr[@]+"${extra_arr[@]}"} --seed "$seed"; then
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
