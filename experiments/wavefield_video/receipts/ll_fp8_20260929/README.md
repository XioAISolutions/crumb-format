# LongLive fp8 rung — environment receipts (2026-09-29)

Status: **10 s rung not yet generated.** Three run-blockers were cleared in this pass; the retry (`c01p`) is queued.

## What happened, in order

1. **c01o** (`runs_longlive_cfg16b`, fp8, window 16) — with both cache-guard fixes applied, the run passed cache init (the previous failure point) and died in the denoising loop at `assert FLASH_ATTN_2_AVAILABLE` (`wan_5b/modules/attention.py`; the FA3/FA4/TE branches are default-off). `flash_attn` was absent from the venv. Crop: `c01o_flashattn_assert_crop.txt`.
2. **flash-attn installed** — prebuilt wheel `flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310` (venv torch 2.6.0+cu124, GCC ABI = 0), `pip install --no-deps`. GPU smoke: `fa_smoke_output.txt` — full-causal varlen `max|Δ|` vs SDPA = **0.0**, windowed `(16,16)` runs.
3. **c01p queued** — `c01p_fa2.job.sh` (fresh dir `runs_longlive_cfg16c`, `LENGTHS=10`, fp8, window 16; pending copy sha256 `5870d798…`).

## The cache fixes (shipped earlier in this pass)

- Unused negative-CFG caches (`kv_cache_neg`, `crossattn_cache_neg`) allocated even with guidance off, and the chunk-reset path touched `crossattn_cache_neg` unguarded. Fixed on the exact pinned source via `scripts/longlive_cfg_cache_patch.py` (AST-anchored, fail-closed, exclusive-create). Applied SHA `7873c583…`; CPU regressions pass.
- Effect: the KV-cache-init memory crash no longer reproduces; the failure moved past it to the missing attention kernel (fixed above).

## Companion receipts

- `c56_v4_smoke_result.json` / `c56_train_log.txt` — latent v4 smoke (dim 512, 12L, 16H, 32.2 M params, latents source): fits and trains, 300 steps at 0.933 step/s (322 s wall); `eval_mse_over_copylast` 0.9913 after 300 steps (copy-last parity — expected at smoke length; not a quality claim).
- `c55_eval_tail.txt` — DINOv2 `long_eval` (fixed evaluator) on the 40-min latent clip: `FAIL at 300 s (diagnostic flags: colour); diagnostic horizon 300 s of 2400 s; semantic quality unverified`.

Next: read `runs_longlive_cfg16c` when c01p fires (after the current seq-training slice). If the 10 s rung generates and decodes, extend the length ladder; re-enable the `tpl_longlive` top-up template only after a rung succeeds (currently paused).
