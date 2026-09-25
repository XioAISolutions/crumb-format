#!/bin/bash
cd /workspace/slava/exp/wavefield_video
P=/workspace/slava/comfy-house/venv/bin/python
mkdir -p runs
echo PHASE1-WAVE; $P train_compare.py --kind wave --steps 1200 --dim 256 --layers 6 --heads 8 --grid 32 --frames 17 --batch 48 --ckpt --out runs > log_wave_p1.txt 2>&1
echo PHASE1-ATTN; $P train_compare.py --kind attn --steps 1200 --dim 256 --layers 6 --heads 8 --grid 32 --frames 17 --batch 48 --ckpt --out runs > log_attn_p1.txt 2>&1
echo PHASE2-WAVE; $P train_compare.py --kind wave --steps 5000 --dim 448 --layers 8 --heads 8 --grid 32 --frames 17 --batch 64 --ckpt --out runs --tag _big > log_wave_p2.txt 2>&1
echo PHASE2-ATTN; $P train_compare.py --kind attn --steps 5000 --dim 448 --layers 8 --heads 8 --grid 32 --frames 17 --batch 64 --ckpt --out runs --tag _big > log_attn_p2.txt 2>&1
echo DONE-ALL > runs/status.txt
