# LongLive c01t — first 5-minute LongLive fp8 clip (301.2 s); rc=0 (2026-09-30)

**Job.** `gpuq_job_974zzzzzz_c01t_300s.sh` — RUNNING 23:32:31Z → FINISHED **rc=0** 2026-09-30T00:08:12Z
(dur 2140 s; runner wall 2125 s for 300 s of video ⇒ ~7.1× realtime). Same config as c01s: fp8 (W8A8), W12,
sink 8, relative-RoPE on, `LL_OFFLOAD_IDLE=1`, `LL_PROMPT_CHUNK=16`, OUT=`runs_longlive_cfg16j`.

**Output.** `len_300s/rank0-0-0_regular.mp4` — 7229 frames, 1280×704, 24 fps, **301.2 s**, 785,993,308 B,
sha256 `350bd473a39da7b2627ec6188b225f2df9d14c6fbbda564243f6884f965ea9ed`. Peak VRAM 24,171 MiB.

**Eval — no quality claim.** `long_eval` NO_FLAGS; diagnostic horizon 301 s of 301 s (dinov2; 11 windows all
pass). Stability ≠ progression. Contact sheet `contact_9.jpg` (Zeph first glance: same coherent street scene as
c01s, near-static across the 9 samples; **not** the independent review).

**Progression (flow_check, Farneback 640×352; `flow_c01t_300s.json`; box copy of `scripts/flow_check.py` md5
`d3395c6099b951b073a41e0f0b6b6271`).** Median **0.385 px/frame** (p10 0.200 / p90 0.686); net dx
**−598.4 px** over 301 s; `bound8/intra8` **0.94** (≤1 ⇒ uniform creep); first half 0.392 / second half 0.379.
Across rungs: c01q 0.404 · c01r 0.370 · c01s 0.395 · **c01t 0.385** px/frame ⇒ **horizon-stable creep ~0.37–0.40
from 10 s to 301 s** — length does not rescue the W12 / rel-RoPE suspects; the probe matrix (c01u_1..4) owns
that call. No promotion: flow + owner eyes gate.

**Ladder context.** All rungs this session rc=0: c01q 10 s → c01r 30 s → c01s 180 s → **c01t 300 s**
(5 min continuous fp8). Next: probes — c01u_1 (prompt-dynamics) done ~00:12Z (wall 208 s, peak VRAM
22,101 MiB, NO_FLAGS); c01u_2..4 in flight; 9k chain resumes behind them; cfg16h (W16 full rung) held.

**Files.** `eval_300s.json` (md5 `314bfcd4c92bc1109cf5dbb239183e7c`), `log_300s.crop.txt` (crop of raw
`f8b12510ddc5bfd2cb92ce9c6a8b4e53`; tqdm/progress lines stripped), `vram_300s.csv`
(md5 `818ff604a791d65f38fcc9d9a1997c25`), `run_config.txt`, `run_identity.json`,
`rank0-0-0_regular.mp4.provenance.json`, `job_c01t_300s.sh`, `contact_9.jpg`
(md5 `9e1293e11ad664b05f6c779b7c3a01be`), `gpuq_timeline.txt`, `flow_c01t_300s.json`.
Nothing published anywhere; owner screening pending.
