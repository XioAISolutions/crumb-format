# Job scripts (wave-2 + OPSD A/B), 2026-09-29/30

Box-side gpuq job scripts, as deployed to `/workspace/slava/` and copied into
`gpu_queue/pending/` as `gpuq_job_974zzzzzz_<name>.sh`.

**Every job script must pin `PY=/workspace/slava/venvs/longlive/bin/python` and
use the `exec > log` redirect + `/usr/bin/timeout` wrapper** — the dispatcher's
PATH has no plain `python`; scripts without the pin die rc=2 in 0 s.

- `c01v_1_bf16.sh` — bf16-vs-fp8 dynamics control (W24, 10 s)
- `c01v_2..4_seed{1,2,3}.sh` — seed sweep at ladder config (W12 rel-rope)
- `c01v_5_fastprompt.sh` — brisk-travel prompt variant
- `c01w_1_i2v.sh` — I2V anchor-continuation (c01q frame 0), harness `I2V=1`
- `c01x_1_opsdorig.sh` / `c01x_2_opsd.sh` — OPSD-V stack A/B (LongLive-1.3B,
  original vs OPSD post-trained lora; repo `/root/opsd-v`, venv `venvs/opsd`).
  These launchers read their CLI args from `opsd_args_orig.txt` / `opsd_args_opsd.txt`
  (small compressor-safe writes; long single-line exec commands have been
  corrupted by the Hermes compressor — always read back after writing).
- `c01y_1_fast10s_s11.sh` — fast-prompt replication, seed 11, 10 s (no-travel control).
- `c01y_2_fast30s_s0.sh` — fast prompt, seed 0, 30 s (queued as `_s11`, retargeted to
  seed 0 once s11 showed no travel; travel sustained, flow 1.132, net −1.6).
- `c01y_3_fast10s_s2.sh` — fast prompt, seed 2, 10 s (second traveler; 1.006, net +0.6).
- `c01z_1_fast180s_s0.sh` — fast prompt, seed 0, 180 s (first 3-min candidate).
- `c01ab_1_fast300s_s0.sh` — fast prompt, seed 0, **300 s = the 5-min rung**.
- `c01aa_1_opsd479.sh` — OPSD 1.3B length run: `opsd_args_477.txt` (477 = 3×159 and
  (477−1)%4=0; 479 dies on the ×3 assert; 159 latents ≈ 39.5 s video at 16 fps).
- `opsd_args_319.txt` — ~80-s OPSD fallback if 477 still grazes VRAM.
- **OPSD 24 GB surgery** (patchers in `scripts/`): flash-attn wheel into `venvs/opsd`;
  `patch_genswap.py` (generator→CPU before VAE decode); `patch_vidcpu.py` (decoded
  video→CPU before clamp); `patch_catcpu2.py` (vae.py `decode()` CPU-accumulates
  per-frame chunks — the 477-frame `torch.cat` OOM). All AST else-count verified.

