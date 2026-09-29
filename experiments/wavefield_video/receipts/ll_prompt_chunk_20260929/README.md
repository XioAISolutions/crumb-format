# LongLive prompt-encode chunk patch (LL_PROMPT_CHUNK_V1) — c01s/c01t TE-encode OOM fix + re-queue receipts

2026-09-29 ~22:10Z, box side (mission-pump pass). Thread: PR #63 box receipts.

## 1) What failed (harvested this pass)

c01s (180 s) and c01t (300 s) FIRED after the 20:08Z pre-fire fix (20:39:22Z / 20:42:21Z) and
both died in ~2 min, rc=1, BEFORE denoise — inside the text encoder:

- `utils/prompt_conditioning.py:17` encodes ALL block prompts in ONE `text_encoder` call, and the
  T5 attention bias is O(batch): `attn_bias = x.new_zeros(b, n, q, k)` (`wan_5b/modules/t5.py:102`).
- c01s: 136 block-prompts -> tried 4.25 GiB vs 3.43 free (20.19 GiB in use).
- c01t: 226 block-prompts -> tried 7.06 GiB vs 1.45 free (22.16 GiB in use).

Evidence: `c01s_log_180s.crop.txt` / `c01t_log_300s.crop.txt` (prompt-echo lines omitted; full logs
md5 `9d20be0a65da71c8e04331263e39a39f` (@runs_longlive_cfg16f) / `812a31c4f7c35d8d5bd6de388bb19934`
(@runs_longlive_cfg16g) on box). `gpuq_timeline.txt` = queue transitions.

## 2) Fix — `LL_PROMPT_CHUNK_V1`

`/root/LongLive/utils/prompt_conditioning.py`: encode in bounded ordered chunks (slice loop +
`torch.cat(dim=0)`); single upstream call kept when chunk >= count. Env `LL_PROMPT_CHUNK`
(default `"16"`; `0`/negative -> single call). Function-local imports, house patch style.

- md5: before `1ab458b4fa7a7940ca86c8c584120dfb` -> after `8a8fe8a544f5a646e38eed78ba44314f`
  (patched-text sha256 `9c24612de9b1d2e4e84d666724f128e5d131de75a785a5391973005938c4defc`; backup `.before_pchunk`).
- Patcher/verifier/selftest (repo scripts/, box copies `apply_ll_prompt_chunk.py` /
  `verify_ll_prompt_chunk.py` / `ll_prompt_chunk_test/`).
- Evidence: `selftest_box.txt` (16/16 PASS incl. idempotent re-apply, slice-bug variant rejected,
  chunked-vs-single-call equality, reversed-order detection), `smoke_real_torch.txt` (real torch,
  CPU: [16,16,8]/[7x5,5]/[40] call sizes, elementwise equality, order = prompt order),
  `verify_live.txt` (live box file structure check PASS).

## 3) Claim boundaries

- Chunk equivalence is proven logic-level (stdlib mock + real torch on CPU). The chunked encode has
  not yet run on GPU: first end-to-end proof is c01s (next fire; ~35-40 min est) and/or the earlier probes.
- Root cause is proven by the two tracebacks (both die at t5.py:102 in the batched encode). The fix
  is targeted at exactly that allocation; other memory walls later in longer runs are not excluded.

## 4) Re-queue (fire order; first fire after the running seq slice)

1. `c01s` 180 s -> `runs_longlive_cfg16i` (cfg16f keeps the failed attempt)
2. `c01t` 300 s -> `runs_longlive_cfg16j` (cfg16g keeps the failed attempt)
3. `c01u_1..4` probes (see below) -> then the 9k chain.

All six `974`-prefixed pending scripts: md5-verified (pending == `/workspace/slava` sources),
`bash -n` OK, real `/usr/bin/timeout` wrappers (4800 / 5280 / 2400 s), `LL_PROMPT_CHUNK=16` pinned.
Verbatim copies: `job_*.sh` in this dir. md5s: c01s `8791d58571636476ec8e0b668b0ac93c`,
c01t `1ccf46f8f58a555de905f46f9f2ca4d7`, c01u_1 `89f7a5e15c747557cc40cab5382609ba`,
c01u_2 `27795a3756d9f41e00dea49ae6ee5883`, c01u_3 `fec2c94e0b0bc8bc8a2754a0cd8865f1`,
c01u_4 `74111c701e88f59f3d636d660148cdf5`.

## 5) c01u probes (staged ~22:07Z) — pre-fire fixed

Staged four: `c01u_1_prompt` (dynamic prompt, W12, rel-rope on), `c01u_2_w32abs`, `c01u_3_w12abs`,
`c01u_4_w32rel`. As staged they had no `PY` (conductor clean env -> rc=2 at the interpreter gate),
no own log, and a no-op `TIMEOUT=2400`. Fixed additively (configs unchanged): `PY` +
`PYTORCH_CUDA_ALLOC_CONF`, own logs, `/usr/bin/timeout 2400` wrapper, `LL` + `LL_PROMPT_CHUNK`
pinned. Staged originals archived in `gpu_queue/superseded/*.staged2210.sh`. `prompts_dynamic.txt`
copied into `pr63/` (staged copy was missing).

## 6) Rope-flag patch (22:07Z) — re-verified only

`apply_ll_ropeflag.py` (run_longlive.sh + longlive_long.py, backups `.before_ropeflag`): idempotent
re-run OK, `bash -n` + `py_compile` OK. `RELATIVE_ROPE` default `on` == previous hardcoded
`use_relative_rope=True` -> ladder config unchanged vs c01r. W16 rung `cfg16h` still HELD.
