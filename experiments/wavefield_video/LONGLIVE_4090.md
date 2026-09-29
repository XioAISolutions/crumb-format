# Minutes of video on one RTX 4090: the LongLive 2.0 baseline

This is the practical path to a 2–5 minute clip on a single 24 GB card. It uses
no wave code. The wave-memory work (phase 4, LONG_HORIZON.md §11 on the follow-up PR) is judged against this
baseline and has to beat it before it can claim anything.

## Why a wrapper is needed at all

LongLive-2.0-5B (NVlabs, a causal Wan2.2-TI2V-5B at 704×1280, 24 fps) has a
rolling 32-frame KV window with an 8-frame sink, so its compute per block is
constant. Its release `inference.py` still cannot make a 5-minute clip on a
4090. Each blocker below was read in the source at commit `6b36d20`.

| # | Blocker | Where | Fix in `longlive_long.py` |
|---|---|---|---|
| 1 | The absolute temporal RoPE table has 1024 rows (4093 frames, **2.84 min**). Past that, `freqs[start:start+f]` comes up short. | `wan_5b/modules/causal_model.py:1076`, `:477` | Overlay sets `use_relative_rope: true`: the cache holds unrotated keys, and positions are relative to the window. |
| 2 | Decoded frames pile up as float32 on the CPU, then `cat`, rescale and `*255` copies are made: **~73 GiB per copy** at 5 min. | `pipeline/causal_diffusion_inference.py:693,757`, `inference.py:648` | Save latents only (0.57 GiB for 5 min). Then `stream_decode` writes to the mp4 one chunk at a time. |
| 3 | With `vae_type: wan`, `streaming_vae: true` calls `WanVAE_.cached_decode`, which is not defined in `wan_5b/modules/vae2_2.py` (only the LightVAE wrapper has it). | `pipeline/...:675,689` | `stream_decode` is a cached decode for the Wan2.2 VAE. |
| 4 | Estimated VRAM in bf16 is 22.1 GiB (weights 9.3 + KV 9.7 + activations 1.6 + other 1.5), which leaves under 2 GiB of headroom. | — | Default is FP8 weights (TorchAO rowwise; sm_89), estimated at 17.9 GiB. `--window 24` saves another 2.4 GiB. |
| 5 | The README warns that `torch.compile` was validated on a single 8-frame block, and a longer run adds new KV shapes. | README "FP8 PTQ" | Eager is the default; `--compile` opts back in. |

Verified here, on CPU:

- `stream_decode` against LongLive's real `vae2_2.WanVAE_` (small random
  config, 11 latent frames → 41 frames) gives **0 uint8 difference** from
  `WanVAE_.decode` at chunk sizes 1, 3, 4 and 11.
- `tests/test_longlive_long.py` checks the plan, the overlay, and exactness
  against a causal fake VAE. It also checks that a naive per-chunk reset is not
  exact.
- `run_longlive.sh` passes an end-to-end smoke (overlay → generate → decode →
  eval → DONE, with finished lengths skipped on rerun) against a fake LongLive
  root that has LongLive's real VAE code, a small random config, and a stub generator.

## Pre-registered read (fixed before any box run)

`long_eval.py` samples each clip at 2 frames/s, with frozen DINOv2-small
features. It judges every 30 s window against window 0:

- **drift_ratio**(w) = sim(w, 0) / sim(1, 0) ≥ 0.9
- **luma, contrast, saturation** within ±25 % of window 0 (fade, flatten, colour).
  The band is ±25 % of max(window 0, 0.02), so a black or grayscale opening still bounds later windows.
- **motion** ≥ 25 % of window 0 (freeze)

A clip **passes** when no window fails. The start of the first failing window is
the clip's **coherent horizon**. That horizon is reported for 10 s, 30 s, 3 min
and 5 min, whatever it turns out to be.

The thresholds are a first guess, made before any data. If they turn out to be
too loose or too tight, they stay as registered for this run, and a revision
is registered as a new section before any rerun.

**Kill criterion for the wave-memory track (phase 4):** memory only
earns further work if it raises the 5-minute coherent horizon, or re-entry
consistency, over this baseline at equal VRAM.

## Box commands

```bash
cd experiments/wavefield_video
# one-time: clone LongLive@6b36d20, pip install, download 10 GB + 34 GB weights.
# Torch is installed first from TORCH_INDEX (default cu124, for CUDA 12.x drivers;
# an unpinned install pulls cu13 wheels a 12.2 driver cannot run) and pinned with
# TORCH_SPEC (default torch==2.6.0). Setup checks CUDA and the FP8 imports before
# downloading; if torchao does not match that torch, set TORCHAO_SPEC or use
# PRECISION=bf16 WINDOW=24.
SETUP=1 PY=python LL=$HOME/LongLive OUT=runs_longlive bash run_longlive.sh
# later runs: skips finished lengths; status in runs_longlive/status.txt
PY=python LL=$HOME/LongLive OUT=runs_longlive bash run_longlive.sh
# a lower-VRAM variant, if FP8 at window 32 OOMs
WINDOW=24 OUT=runs_longlive_w24 bash run_longlive.sh
```

The wall-time estimates below are guesses: nothing has been measured on a 4090.

- **Setup:** the download (~44 GB) plus pip install.
- **Generation:** LongLive reports 24.8 FPS for the 5B model, presumably on an
  H100 with compile. A 4090 in eager FP8 is plausibly 3–6× slower, about
  4–8 FPS. That puts 5 minutes (7229 frames) at roughly 15–30 min.
- **Decode:** another ~5–10 min at 704×1280.
- **The whole default ladder (10 s, 30 s, 3 min, 5 min):** about 1–1.5 h.

Receipts to send back:

- `runs_longlive/progress.txt`: wall time per length and each verdict line.
- Every `len_*/rank0-0-0_regular.eval.json`.
- Peak VRAM from `nvidia-smi` during the 5 min run.
- The 30 s and 5 min mp4s, so the frames can be checked by eye. A metric pass
  is not a quality claim.

## Run environment notes (2026-09-29)

Three blockers hit on the box during the first fp8 rungs; clear them before any retry:

1. **`flash_attn` must be present or generation dies at the FA2 assert** (`wan_5b/modules/attention.py` — the FA3/FA4/TE branches are default-off). The venv ships without it. Install the prebuilt wheel matching this venv (torch ABI = 0): `flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` from the Dao-AILab releases, `pip install --no-deps` (keep the canonical filename — a renamed copy fails as invalid wheel filename). Smoke a varlen full + windowed call on the GPU before the next retry.
2. **Two cache bugs in the pinned source** (`pipeline/causal_diffusion_inference.py`): unused negative-CFG KV/cross-attn caches allocate with guidance off, and the chunk-reset path touches `crossattn_cache_neg` unguarded. Apply `scripts/longlive_cfg_cache_patch.py` (AST-anchored, fail-closed, exclusive-create) and verify with `scripts/check_longlive_cache.py` plus a SHA readback of the patched file.
3. **Run identity:** after any source/venv change, retry into a fresh `--out` (c01o → `runs_longlive_cfg16b`, c01p → `runs_longlive_cfg16c`).

Status at 2026-09-29 13:35Z: both cache bugs fixed on the box (pipeline SHA `7873c583…`); flash-attn 2.7.4.post1 installed + GPU smoke passed; retry `c01p` (LENGTHS=10, fp8, window 16) queued. Receipts: `receipts/ll_fp8_20260929/`.

**Update (~16:45Z):** `c01p` reached the denoise loop and OOMed there (peak 24.16 of 24.56 GiB — the text encoder and VAE stay resident although both are idle once prompts are encoded). Idle-offload patch: `scripts/longlive_idle_offload_patch.py` (env-guarded `LL_OFFLOAD_IDLE=1`, fail-closed apply, exec-safe). v1's insert split the `use_cfg` if/else — the `else:` reattached to the inserted `if` (env=1 path left `unconditional_dict` unset; compiled and passed naive regressions, caught by region readback before the first run). v2 inserts after the complete if/else and ships `scripts/verify_ll_idle.py` (AST checks incl. else-attachment; env=1 harness pass) — applied on the box, pipeline SHA `7373a6fb…`. Retry `c01q` (fresh `runs_longlive_cfg16d`, window 12 + offload) queued ahead of the v4 latent full run.
