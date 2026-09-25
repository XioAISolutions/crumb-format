#!/bin/bash
cd /workspace/slava/exp/wavefield_video
for i in $(seq 1 900); do [ -f runs_r4/r4_status.txt ] && break; sleep 60; done
sleep 20
/workspace/slava/comfy-house/venv/bin/python triton_fused.py > log_triton_selfcheck.txt 2>&1
/workspace/slava/comfy-house/venv/bin/python bench_triton.py > log_triton_bench.txt 2>&1
echo DONE > triton_status.txt
