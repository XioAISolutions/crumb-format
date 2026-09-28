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

BASE=(--data-source occlusion --grid 32 --n-balls 6 --batch 4
      --target-params 2000000 --causal --residual --linear-pad
      --dim 128 --layers 4 --heads 8 --ckpt
      --motion-loss --rollout-loss 1 --const-lr
      --eval-rollout 512 --eval-seeds 8 --eval-chunk 1 --auto-batch --save-every 500)

ARMS=(
    "W_soft|wave|$T_LONG|--kernel-version dispersion --pole-param softplus"
    "W_half|wave|$T_LONG|--kernel-version dispersion --pole-param halflife"
    "E_half|wave|$T_LONG|--kernel-version dispersion --fuse local_wave --pole-param halflife"
    "S_ssm|ssm|$T_LONG|"
    "A_attn|attn|$T_ATTN|"
)

for seed in $SEEDS; do
    for arm in "${ARMS[@]}"; do
        IFS='|' read -r label kind frames extra <<< "$arm"
        tag="_${label}_s${seed}"
        if [ -f "$OUT/result_${kind}${tag}.json" ]; then echo "SKIP $tag"; continue; fi
        read -r -a extra_args <<< "$extra"
        tgap=$(( GAP < frames - 8 ? GAP : frames - 8 ))
        occ=(--train-occ-start $((frames - tgap)) --train-occ-end "$frames"
             --occ-start $((frames + 32)) --occ-end $((frames + 32 + GAP)))
        echo "== $tag $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        "$PY" train_compare.py "${BASE[@]}" "${occ[@]}" ${extra_args[@]+"${extra_args[@]}"} \
            --kind "$kind" --frames "$frames" --seed "$seed" --steps "$STEPS" \
            --out "$OUT" --tag "$tag" > "$OUT/log${tag}.txt" 2>&1 || echo "FAIL $tag"
    done
done

# 5-minute health stream on each trained wave arm (constant state, collapse flags).
for f in "$OUT"/model_wave_W_*_s0.pt; do
    [ -f "$f" ] || continue
    pp=softplus; [[ "$f" == *W_half* ]] && pp=halflife
    res="$OUT/result_$(basename "${f#*model_}")"; res="${res%.pt}.json"
    ffn=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ffn_mult"])' "$res")
    "$PY" long_horizon.py stream --pole-param "$pp" --ckpt "$f" --grid 32 --frames "$T_LONG" \
        --dim 128 --layers 4 --heads 8 --ffn-mult "$ffn" --stream-frames 7200 --chunk 600 \
        --out "${f%.pt}_stream7200.json" > "${f%.pt}_stream7200.log" 2>&1 || echo "STREAM FAIL $f"
done
echo DONE > "$OUT/status.txt"
