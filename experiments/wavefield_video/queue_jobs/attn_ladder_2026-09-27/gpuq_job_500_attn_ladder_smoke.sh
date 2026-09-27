#!/bin/bash
# Attention-lane box smoke (card t_d77fd4cd): validates the pushed file set on
# the box before any suite slice -- bash syntax, portable runner contract, and a
# micro GPU train of the attn kind asserting the new recipe keys (const_lr/seed/
# telemetry_every) land in the result JSON. Must print 'smoke attn-ladder PASS'.
LOG=/workspace/slava/logs/gpuq_job_500_attn_ladder_smoke.log
mkdir -p /workspace/slava/logs
exec > "$LOG" 2>&1
echo "=== job500 attn-ladder boxsmoke start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
cd /workspace/slava/exp/wavefield_video || exit 1
PY=/workspace/slava/comfy-house/venv/bin/python
echo "### (1) portable smoke (bash -n + runner contract + report)"
PY_TEST=$PY bash run_smoke_attn_ladder.sh || exit 1
echo "### (2) micro GPU train: attn kind, 2 steps, recipe-key check"
rm -rf runs_attn_smoke
$PY train_compare.py --kind attn --data-source balls --grid 8 --dim 16 --layers 1 \
    --heads 2 --frames 5 --batch 2 --steps 2 --eval-rollout 8 --eval-seeds 2 \
    --eval-chunk 1 --causal --residual --linear-pad --collisions --motion-loss \
    --rollout-loss 1 --const-lr --seed 7 --telemetry-every 1 \
    --out runs_attn_smoke --tag _keycheck > runs_attn_smoke_train.log 2>&1 || exit 1
$PY - <<'PYEOF'
import json
r = json.load(open("runs_attn_smoke/result_attn_keycheck.json"))
assert r["const_lr"] is True, r.get("const_lr")
assert r["seed"] == 7, r.get("seed")
assert r["telemetry_every"] == 1, r.get("telemetry_every")
print("recipe keys OK:", {k: r[k] for k in ("const_lr", "seed", "fp32",
                                            "no_decay_norm_head", "telemetry_every")})
PYEOF
echo "smoke attn-ladder box PASS $(date -u +%FT%TZ)"
exit 0
