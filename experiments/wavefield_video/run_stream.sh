#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/slava/exp/wavefield_video
P=/workspace/slava/comfy-house/venv/bin/python
$P train_compare.py --stream-test --kind wave --kernel-version dispersion --gate --local-fuse --dim 128 --layers 4 --heads 8 --grid 16 --frames 16 --target-params 1600000 --stream-frames 1024 --stream-batch 8 --out runs_v2 --tag _streamw > log_stream_wave.txt 2>&1
$P train_compare.py --stream-test --kind attn --dim 128 --layers 4 --heads 8 --grid 16 --frames 16 --target-params 1600000 --stream-frames 1024 --stream-batch 8 --out runs_v2 --tag _streama > log_stream_attn.txt 2>&1
$P train_compare.py --stream-test --kind ssm --dim 128 --layers 4 --heads 8 --grid 16 --frames 16 --target-params 1600000 --stream-frames 1024 --stream-batch 8 --out runs_v2 --tag _streams > log_stream_ssm.txt 2>&1
echo DONE > runs_v2/stream_status.txt
