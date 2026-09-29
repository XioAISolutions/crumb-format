#!/usr/bin/env bash
# LONGLIVE_4090.md -- minutes of video on one RTX 4090 with LongLive 2.0 (5B, FP8).
#
#   0. setup    (SETUP=1) clone LongLive at a pinned commit, install its deps, and
#               download LongLive-2.0-5B (10 GB) + Wan2.2-TI2V-5B (VAE, T5, base DiT;
#               34 GB). Needs ~60 GB disk; host RAM >= 64 GB recommended (the loader
#               holds the base DiT, the 10 GB checkpoint and the 11 GB T5 at once).
#   1. for each length in LENGTHS (seconds): longlive_long.py generate (latents
#      only, relative RoPE, eager, fp8) -> streaming decode to mp4 -> long_eval.py
#
# Default LENGTHS="10 30 180 300": a 10 s smoke, the 30 s reference clip (what
# LongLive is known to do well), then 3 and 5 minutes.
# Pre-registered read (LONGLIVE_4090.md, fixed before any run):
#   PASS a length   long_eval verdict PASS (every 30 s window: drift_ratio >= 0.9,
#                   luma/contrast/saturation within +-25 %, motion >= 25 % of window 0)
#   The coherent horizon of the longest clip is the number to report, whatever it is.
# Re-run until $OUT/status.txt reads DONE; finished lengths are skipped. LongLive has
# no mid-video resume, so one length's generation is never cut by a slice budget.
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PY=${PY:-python}
LL=${LL:-$HOME/LongLive}
LL_COMMIT=${LL_COMMIT:-6b36d20}
CKPT=${CKPT:-$LL/LongLive-2.0-5B/model_bf16.pt}
PROMPTS=${PROMPTS:-$PWD/prompts_long.txt}
OUT=${OUT:-runs_longlive}
LENGTHS=${LENGTHS:-10 30 180 300}
PRECISION=${PRECISION:-fp8}; WINDOW=${WINDOW:-32}; SINK=${SINK:-8}; SEED=${SEED:-0}
ENCODER=${ENCODER:-dinov2}; DECODE_DEVICE=${DECODE_DEVICE:-cuda}
BASE_CONFIG=${BASE_CONFIG:-}        # empty: LongLive's configs/fp8 (or bf16) inference yaml
base_args=(); [ -n "$BASE_CONFIG" ] && base_args=(--base-config "$BASE_CONFIG")
mkdir -p "$OUT"
command -v "$PY" >/dev/null 2>&1 || { echo "Python not executable: $PY" >&2; exit 2; }
vram_pid=
failed() { [ -n "$vram_pid" ] && kill "$vram_pid" 2>/dev/null; echo "FAILED $1" | tee -a "$OUT/progress.txt"; echo "FAILED $1" > "$OUT/status.txt"; exit 1; }

if [ "${SETUP:-0}" = "1" ]; then
    echo "== setup $(date -u +%FT%TZ)" | tee -a "$OUT/progress.txt"
    if [ ! -d "$LL/.git" ]; then
        git clone --single-branch --branch main https://github.com/NVlabs/LongLive.git "$LL"
    fi
    git -C "$LL" checkout -q "$LL_COMMIT"
    # Torch first, from the wheel index matching the box's CUDA driver, then pinned
    # (-c) so LongLive's requirements cannot pull another build. The default index is
    # cu124: a CUDA 12.2 driver runs 12.x wheels (minor-version compatibility) but
    # not the cu13 builds that an unpinned install picks up.
    TORCH_SPEC=${TORCH_SPEC:-torch==2.6.0}
    TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}
    TORCHAO_SPEC=${TORCHAO_SPEC:-torchao}
    "$PY" -m pip install "$TORCH_SPEC" --index-url "$TORCH_INDEX"
    printf '%s\n' "$TORCH_SPEC" > "$OUT/torch_constraint.txt"
    "$PY" -m pip install -c "$OUT/torch_constraint.txt" -r "$LL/requirements.txt" \
        "$TORCHAO_SPEC" imageio-ffmpeg pyyaml
    # fail fast, before 44 GB of downloads: CUDA must work, and the FP8 path import
    "$PY" - "$PRECISION" <<'PYEOF' || failed "torch/CUDA/torchao check (set TORCH_SPEC, TORCH_INDEX, TORCHAO_SPEC; or PRECISION=bf16 WINDOW=24)"
import sys, torch
assert torch.cuda.is_available(), f"torch {torch.__version__} cannot see the GPU (driver/wheel CUDA mismatch?)"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
if sys.argv[1] == "fp8":
    assert torch.cuda.get_device_capability(0) >= (8, 9), "FP8 needs compute capability 8.9+"
    import torchao
    from torchao.quantization import (  # noqa: F401  (exactly what LongLive's utils/fp8.py imports)
        Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_)
    print("torchao", torchao.__version__)
PYEOF
    "$PY" - "$LL" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download, snapshot_download
ll = sys.argv[1]
snapshot_download("Wan-AI/Wan2.2-TI2V-5B", local_dir=f"{ll}/wan_models/Wan2.2-TI2V-5B")
hf_hub_download("Efficient-Large-Model/LongLive-2.0-5B", "model_bf16.pt", local_dir=f"{ll}/LongLive-2.0-5B")
PYEOF
fi
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT (run with SETUP=1)" >&2; exit 2; }
[ -f "$LL/wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" ] || { echo "no Wan2.2 weights under $LL/wan_models (SETUP=1)" >&2; exit 2; }
[ -f "$PROMPTS" ] || { echo "no prompts file $PROMPTS" >&2; exit 2; }

cfg="LL_COMMIT=$(git -C "$LL" rev-parse --short HEAD 2>/dev/null || echo ?) CKPT=$CKPT PROMPTS=$PROMPTS PRECISION=$PRECISION WINDOW=$WINDOW SINK=$SINK SEED=$SEED ENCODER=$ENCODER BASE_CONFIG=$BASE_CONFIG"
if [ -f "$OUT/run_config.txt" ] && [ "$(cat "$OUT/run_config.txt")" != "$cfg" ]; then
    echo "$OUT holds a run with different settings:" >&2
    diff <(tr ' ' '\n' < "$OUT/run_config.txt") <(tr ' ' '\n' <<< "$cfg") >&2 || true
    echo "use a new OUT instead of mixing runs" >&2
    exit 2
fi
echo "$cfg" > "$OUT/run_config.txt"
echo RUNNING > "$OUT/status.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$OUT/progress.txt" || true

mapfile -t stems < <("$PY" longlive_long.py expected --prompts "$PROMPTS")
[ "${#stems[@]}" -gt 0 ] || { echo "no prompts in $PROMPTS" >&2; exit 2; }
all_evals() {  # all_evals <dir>: every prompt's eval receipt exists
    local s; for s in "${stems[@]}"; do [ -f "$1/$s.eval.json" ] || return 1; done
}
for secs in $LENGTHS; do
    d="$OUT/len_${secs}s"
    mins=$("$PY" -c "print($secs / 60)")
    # no skip before generate: it re-checks the full-content run identity
    # (checkpoint, prompts, VAE) even when every receipt exists, and refuses
    # stale outputs; with matching inputs it only re-hashes and returns
    fresh=1; all_evals "$d" && fresh=0
    [ "$fresh" = 1 ] && echo "== generate ${secs}s $(date -u +%FT%TZ)" | tee -a "$OUT/progress.txt"
    t0=$(date +%s)
    vram_pid=
    if command -v nvidia-smi >/dev/null 2>&1; then     # peak-VRAM receipt
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 \
            > "$OUT/vram_${secs}s.csv" 2>/dev/null & vram_pid=$!
    fi
    # generate skips latents / mp4s already made by this same configuration and
    # refuses an --out that holds a different one (run_identity.json)
    "$PY" longlive_long.py generate --ll-root "$LL" --ckpt "$CKPT" --prompts "$PROMPTS" \
        --minutes "$mins" --out "$d" --precision "$PRECISION" --window "$WINDOW" \
        --sink "$SINK" --seed "$SEED" --decode-device "$DECODE_DEVICE" ${base_args[@]+"${base_args[@]}"} >> "$OUT/log_${secs}s.txt" 2>&1 || failed "generate ${secs}s (see $OUT/log_${secs}s.txt)"
    [ -n "$vram_pid" ] && kill "$vram_pid" 2>/dev/null || true
    vram_pid=
    peak=$( { sort -n "$OUT/vram_${secs}s.csv" 2>/dev/null || true; } | tail -n 1)
    [ "$fresh" = 1 ] && echo "   wall $(( $(date +%s) - t0 )) s for ${secs}s of video; peak VRAM ${peak:-?} MiB" | tee -a "$OUT/progress.txt"
    for s in "${stems[@]}"; do
        v="$d/$s.mp4"
        [ -f "$v" ] || failed "missing $v after generate ${secs}s"
        # --reuse keeps a receipt only if the evaluator source, its arguments and
        # the video bytes all match what produced it; otherwise it recomputes
        "$PY" long_eval.py "$v" --encoder "$ENCODER" --out "$d/$s.eval.json" --reuse \
            >> "$OUT/log_${secs}s.txt" 2>&1 || failed "eval $v"
        tail -n 1 "$OUT/log_${secs}s.txt" | tee -a "$OUT/progress.txt"
    done
done
echo DONE > "$OUT/status.txt"
"$PY" - "$OUT" <<'EOF'
import glob, json, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/len_*/*.eval.json")):
    r = json.load(open(f))
    print(f"{f}: {r['duration_s']:.0f}s  {r['verdict']}  horizon {r['coherent_horizon_s']:.0f}s")
EOF
