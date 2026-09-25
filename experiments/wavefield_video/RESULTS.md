# Wave-Field 2D smoke test

- equivalence max err: 5.72e-06 (plain), 4.29e-06 (causal)
- scaling (single 64-channel 2D field, CPU):

- 32x32 (N=1024): wave=0.72 ms, attention=5.97
- 64x64 (N=4096): wave=0.84 ms, attention=104.76
- 128x128 (N=16384): wave=3.71 ms, attention=1546.1
- 256x256 (N=65536): wave=13.33 ms, attention=skipped(OOM-risk)

Next: bilinear 2D scatter/gather + a tiny video-prediction
training run (wave-field vs transformer baseline).
