# LongLive c01s — 180 s fp8 clip; prompt-chunk patch GPU-proven (2026-09-29)

**Job.** `gpuq_job_974zzzzzz_c01s_180s.sh` — RUNNING 23:09:30Z → FINISHED **rc=0** 23:31:51Z (dur 1340 s;
runner wall 1330 s for 180 s of video ⇒ ~7.4× realtime). Config: fp8 (W8A8), W12, sink 8, relative-RoPE on,
`LL_OFFLOAD_IDLE=1`, `LL_PROMPT_CHUNK=16`, OUT=`runs_longlive_cfg16i`.

**Claim closed.** Chunk patch `LL_PROMPT_CHUNK_V1` (`/root/LongLive/utils/prompt_conditioning.py`, md5
`8a8fe8a544f5a646e38eed78ba44314f` — verified live on box at receipt time) was previously logic-level proven only;
PR #82 said *"first end-to-end proof = c01s"*. Result: 136 block-prompts encoded end-to-end on GPU, **rc=0,
0 OOM/traceback lines** in the raw run log (md5 `2818de48f7bc884adce34347942962e5`) — vs cfg16f/g dying ~2 min
in at prompt encode (4.25 GiB wanted / 3.43 GiB free).

**Output.** `len_180s/rank0-0-0_regular.mp4` — 4349 frames, 1280×704, 24 fps, **181.2 s**, 474,938,315 B,
sha256 `71de96ed9007fd6037335bbb2c32fd799751a6abf8ac6e6ad6961158dc0751db`. Run identity: ckpt `ec9063a44ea3c91e`,
LongLive `6b36d20…`, `local_attn_size 12`, `use_relative_rope true`. Peak VRAM 24,165 MiB (encode spike;
`[mem] idle-offload: TE+VAE -> cpu; alloc=5.62 GiB` before denoise).

**Eval — no quality claim.** `long_eval` NO_FLAGS; diagnostic horizon 181 s of 181 s (dinov2; all windows pass,
drift ≈ 1.0). Stability ≠ progression. Contact sheet `contact_9.jpg` (Zeph first glance: coherent, photoreal
stills; scene barely advances across the 9 samples — consistent with the known creep signature; **not** the
independent review).

**Progression (flow_check, Farneback 640×352; `flow_c01s_180s.json`; tool `scripts/flow_check.py`, box copy
md5 `d3395c6099b951b073a41e0f0b6b6271`).** Median **0.395 px/frame** (p10 0.221 / p90 0.702); net dx
**−410.7 px** over 181 s; `bound8/intra8` **0.94** (≤1 ⇒ uniform creep, not block jumps); first half 0.376 /
second half 0.410. Comparison — c01q 10 s: 0.404 / −24.8 px; c01r 30 s: 0.370 / −57.8 px.
**Reading: the creep is horizon-stable — ~0.40 px/frame at 10 s, 30 s and 181 s** (≈15× below a real walking
take at this width). The W12 / relative-RoPE suspects from `ll_motion_creep_20260929` are not rescued by longer
horizons; probes c01u_1..4 isolate them. No promotion: flow + owner eyes gate.

**State at receipt.** c01t (300 s) also **DONE rc=0** (00:08:12Z; see `ll_c01t_300s_20260929`). Probes: c01u_1
(prompt-dynamics) **done ~00:12Z** — wall 208 s, peak VRAM 22,101 MiB, NO_FLAGS (10 s); c01u_2..4 next;
9k chain behind them; cfg16h held. Box disk 760 G free; conductor err_streak 0.

**Files.** `eval_180s.json` (md5 `fb3cc196f0bb4dbdd6b0622d6bba1237`), `log_180s.crop.txt` (crop of raw
`2818de48…`; tqdm/progress lines stripped), `vram_180s.csv` (md5 `cc484ed1f8410a1ca16db26209fa81d8`),
`run_config.txt`, `run_identity.json`, `rank0-0-0_regular.mp4.provenance.json`, `job_c01s_180s.sh`,
`contact_9.jpg` (md5 `cb337068fbdb83775dc9ce4eb539c036`), `gpuq_timeline.txt`, `flow_c01s_180s.json`.
Nothing published anywhere; owner screening pending.
