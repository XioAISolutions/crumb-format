#!/bin/bash
cd /workspace/slava/exp/wavefield_video
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq 1 720); do tmux has-session -t wfv3 2>/dev/null || break; sleep 60; done
sleep 15; echo "[chain] w1g32 done -> stream"
bash run_stream.sh
echo "[chain] stream done -> bench"
/workspace/slava/comfy-house/venv/bin/python bench_eager_mix.py --dim 128 --layers 4 --grid 16 --frames 16 --batch 2 --steps 64 --iters 6 > log_bench_mix.txt 2>&1
echo "[chain] bench done -> retrains"
bash run_r4_retrains.sh
echo "[chain] ALL DONE"
