#!/usr/bin/env bash
# Run after run_retune.sh; carry its selected objective flags in RECIPE_EXTRA.
# Example: RECIPE_EXTRA='--resid-balanced --motion-weighted' bash run_difficulty.sh
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PY=${PY:-/workspace/slava/comfy-house/venv/bin/python}
OUT=${OUT:-runs_difficulty}
STEPS=${STEPS:-2000}
if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS must be a positive integer" >&2
    exit 2
fi
mkdir -p "$OUT"
PROGRESS="$OUT/difficulty_progress.txt"
STATUS="$OUT/difficulty_status.txt"
printf 'difficulty start %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$PROGRESS"
printf 'RUNNING\n' > "$STATUS"
finished=0
failures=0
on_exit() {
    local rc=$?
    if (( ! finished )); then
        printf 'FAILED exit=%s\n' "$rc" > "$STATUS"
        printf 'difficulty aborted exit=%s %s\n' "$rc" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$PROGRESS"
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Python not executable: $PY (override with PY=/path/to/python)" >&2
    exit 2
fi

# Retune's g32 control recipe, fixed K=1. Keep this in an array so paths and
# arguments are never expanded as shell commands or filename patterns.
BASE=(--kernel-version dispersion --gate --local-fuse
      --dim 192 --layers 6 --heads 8 --grid 32 --frames 17 --batch 16
      --target-params 4000000 --causal --residual --linear-pad
      --collisions --motion-loss --rollout-loss 1
      --eval-rollout 256 --eval-seeds 16 --eval-chunk 4
      --auto-batch --save-every 500)
# RECIPE_EXTRA is whitespace-separated flags/values, including multiline values.
# Quotes, substitutions and glob characters are literal; there is no eval.
recipe_text=${RECIPE_EXTRA:-}
recipe_text=${recipe_text//$'\n'/ }
recipe_extra=()
if [[ -n "$recipe_text" ]]; then
    read -r -a recipe_extra <<< "$recipe_text"
    BASE+=("${recipe_extra[@]}")
fi

run() {
    local tag=$1 kind=$2
    shift 2
    local log_file="$OUT/log_difficulty_${kind}${tag}.txt"
    local rc=0
    # Rung identity and output routing take precedence over recipe extras.
    local command=("$PY" train_compare.py "${BASE[@]}" "$@"
                   --kind "$kind" --data-source balls --steps "$STEPS"
                   --out "$OUT" --tag "$tag")
    printf '== %s%s %s\n' "$kind" "$tag" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$PROGRESS"
    printf 'command:' >> "$PROGRESS"
    printf ' %q' "${command[@]}" >> "$PROGRESS"
    printf '\n' >> "$PROGRESS"
    if "${command[@]}" > "$log_file" 2>&1; then
        rc=0
    else
        rc=$?
        failures=$((failures + 1))
    fi
    printf '%s%s exit=%s %s log=%s\n' "$kind" "$tag" "$rc" \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$log_file" | tee -a "$PROGRESS"
}

run _D1 wave --speed 2.30
run _D2 wave --n-balls 8
run _D3 wave --n-balls 12 --speed 2.30
run _D3 attn --n-balls 12 --speed 2.30
run _D4 wave --radius 0.8
run _D5 wave --grid 64 --n-balls 8
run _D5 attn --grid 64 --n-balls 8
run _D6 wave --grid 32 --eval-rollout 1024

if (( failures )); then
    printf 'FAILED %s/8\n' "$failures" > "$STATUS"
    finished=1
    exit 1
fi
printf 'DONE\n' > "$STATUS"
finished=1
