#!/usr/bin/env bash
# R14 -- the decisive long-memory occlusion benchmark (HARDENING_astra.md
# "The experiment that matters most"). Five arms at an EQUAL ~4M parameter budget,
# each over five training seeds, evaluated with a 1024-frame occlusion rollout:
#
#   A local-attn    --kind attn                                     (windowed rollout)
#   B generic-ssm   --kind ssm                                      (streaming O(1) state)
#   C wave          --kind wave  --kernel-version dispersion        (streaming O(1) state)
#   D local+ssm     --kind ssm   --fuse local_ssm                   (hybrid, streaming global)
#   E local+wave    --kind wave  --kernel-version dispersion --fuse local_wave  (hybrid)
#
# ffn_mult is bisected to --target-params for every arm, so the two-mixer hybrids
# are held to the same budget as the single-mixer arms. Kill criterion (item 5):
# if local+wave (E) does not beat local+generic-ssm (D) on divergence-horizon-per
# -byte-of-persistent-state, the wave formulation is not pulling its weight.
set -euo pipefail

# long-budget control runs first if present
if [ -f run_longbudget.sh ]; then bash run_longbudget.sh; fi
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PY=${PY:-/workspace/slava/comfy-house/venv/bin/python}
OUT=${OUT:-runs_occlusion}
STEPS=${STEPS:-2000}
SEEDS=${SEEDS:-0 1 2 3 4}
if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS must be a positive integer" >&2
    exit 2
fi
mkdir -p "$OUT"
PROGRESS="$OUT/occlusion_progress.txt"
STATUS="$OUT/occlusion_status.txt"
printf 'occlusion start %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$PROGRESS"
printf 'RUNNING\n' > "$STATUS"
finished=0
failures=0
total=0
on_exit() {
    local rc=$?
    if (( ! finished )); then
        printf 'FAILED exit=%s\n' "$rc" > "$STATUS"
        printf 'occlusion aborted exit=%s %s\n' "$rc" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$PROGRESS"
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Python not executable: $PY (override with PY=/path/to/python)" >&2
    exit 2
fi

# Shared occlusion recipe at the fixed ~4M budget. Held identical across arms so
# only the temporal mechanism differs. Kept in an array: paths/args are never
# re-expanded as shell commands or globs.
BASE=(--data-source occlusion --grid 64 --n-balls 8 --frames 17 --batch 8
      --target-params 4000000 --causal --residual --linear-pad
      --dim 192 --layers 6 --heads 8
      --occ-start 64 --occ-end 320 --motion-loss --rollout-loss 1
      --eval-rollout 1024 --eval-seeds 8 --eval-chunk 1
      --auto-batch --save-every 500)

# arm label | --kind | extra arm-specific flags
ARMS=(
    "A_localattn|attn|"
    "B_ssm|ssm|"
    "C_wave|wave|--kernel-version dispersion"
    "D_local_ssm|ssm|--fuse local_ssm"
    "E_local_wave|wave|--kernel-version dispersion --fuse local_wave"
)

run() {
    local tag=$1 kind=$2
    shift 2
    local log_file="$OUT/log_occlusion_${kind}${tag}.txt"
    local rc=0
    local command=("$PY" train_compare.py "${BASE[@]}" "$@"
                   --kind "$kind" --steps "$STEPS" --out "$OUT" --tag "$tag")
    total=$((total + 1))
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

# seed-outer so a partial run still holds every arm for the seeds it reached.
for seed in $SEEDS; do
    for spec in "${ARMS[@]}"; do
        IFS='|' read -r label kind extra <<< "$spec"
        extra_arr=()
        if [[ -n "$extra" ]]; then read -r -a extra_arr <<< "$extra"; fi
        run "_${label}_s${seed}" "$kind" ${extra_arr[@]+"${extra_arr[@]}"} --seed "$seed"
    done
done

if (( failures )); then
    printf 'FAILED %s/%s\n' "$failures" "$total" > "$STATUS"
    finished=1
    exit 1
fi
printf 'DONE %s/%s\n' "$total" "$total" > "$STATUS"
finished=1
