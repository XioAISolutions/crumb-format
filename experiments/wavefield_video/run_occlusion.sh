#!/usr/bin/env bash
# R14/R15 -- the decisive long-memory occlusion benchmark (HARDENING_astra.md
# "The experiment that matters most"). Eight arms at an EQUAL ~4M parameter budget,
# each over five training seeds, evaluated with a 1024-frame occlusion rollout:
#
#   A local-attn    --kind attn                                     (windowed rollout)
#   B generic-ssm   --kind ssm                                      (streaming O(1) state)
#   C wave          --kind wave  --kernel-version dispersion        (streaming O(1) state)
#   D local+ssm     --kind ssm   --fuse local_ssm                   (hybrid, streaming global)
#   E local+wave    --kind wave  --kernel-version dispersion --fuse local_wave  (hybrid)
#   F qssm          --kind qssm                                     (R15 E1: quaternion-state
#                                                                    SSM; state bytes == B)
#   G qwave         --kind wave  --kernel-version dispersion --q-mix        (R15 E2)
#   H qcolor        --kind wave  --kernel-version dispersion --quat-color   (R15 E3)
#
# ffn_mult is bisected to --target-params for every arm, so hybrids and R15 arms
# are held to the same budget as the single-mixer arms. Kill criterion (item 5):
# if local+wave (E) does not beat local+generic-ssm (D) on divergence-horizon-per
# -byte-of-persistent-state, the wave formulation is not pulling its weight;
# the R15 arms extend the same instrument (HYPERCOMPLEX_STUDY.md sec 5).
#
# ENV knobs (R15 queue ops):
#   CONST_LR=1  append --const-lr to every arm (unfreeze-rule escalation, study 5.2)
#   RESUME=1    sliced-queue mode: skip arms with a result JSON, resume live ones
#               from their ckpt, cap each arm at SLICE_S wall seconds and stop the
#               suite at the slice boundary (status "SLICED done/planned"). Run
#               the same command again until DONE. Keep STEPS/CONST_LR fixed
#               across slices of one OUT (resume needs the same schedule target).
#   SLICE_S     per-arm wall cap in RESUME mode, seconds (default 4200; the box
#               conductor kills whole jobs at 5400s)
set -euo pipefail

# long-budget control runs first if present -- legacy one-off hook, opt-in only
# (the historical run is done; a suite invocation must never re-run it)
if [ "${RUN_LONGBUDGET:-0}" = "1" ] && [ -f run_longbudget.sh ]; then bash run_longbudget.sh; fi
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PY=${PY:-/workspace/slava/comfy-house/venv/bin/python}
OUT=${OUT:-runs_occlusion}
STEPS=${STEPS:-2000}
CONST_LR=${CONST_LR:-0}
RESUME=${RESUME:-0}
SLICE_S=${SLICE_S:-4200}
SEEDS=${SEEDS:-0 1 2 3 4}
if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS must be a positive integer" >&2
    exit 2
fi
if [[ ! "$SLICE_S" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLICE_S must be a positive integer" >&2
    exit 2
fi
mkdir -p "$OUT"
PROGRESS="$OUT/occlusion_progress.txt"
STATUS="$OUT/occlusion_status.txt"
if [ "$RESUME" = "1" ]; then
    printf 'occlusion slice %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$PROGRESS"
else
    printf 'occlusion start %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$PROGRESS"
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
if [ "$CONST_LR" = "1" ]; then BASE+=(--const-lr); fi

# arm label | --kind | extra arm-specific flags
ARMS=(
    "A_localattn|attn|"
    "B_ssm|ssm|"
    "C_wave|wave|--kernel-version dispersion"
    "D_local_ssm|ssm|--fuse local_ssm"
    "E_local_wave|wave|--kernel-version dispersion --fuse local_wave"
    "F_qssm|qssm|"
    "G_qwave|wave|--kernel-version dispersion --q-mix"
    "H_qcolor|wave|--kernel-version dispersion --quat-color"
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
    local log_file="$OUT/log_occlusion_${kind}${tag}.txt"
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
    local command=("$PY" train_compare.py "${BASE[@]}" "$@" ${resume_args[@]+"${resume_args[@]}"}
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

# seed-outer so a partial run still holds every arm for the seeds it reached.
nseeds=0
for _seed in $SEEDS; do nseeds=$((nseeds + 1)); done
units=$(( ${#ARMS[@]} * nseeds ))
for seed in $SEEDS; do
    for spec in "${ARMS[@]}"; do
        IFS='|' read -r label kind extra <<< "$spec"
        extra_arr=()
        if [[ -n "$extra" ]]; then read -r -a extra_arr <<< "$extra"; fi
        if ! run "_${label}_s${seed}" "$kind" ${extra_arr[@]+"${extra_arr[@]}"} --seed "$seed"; then
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
