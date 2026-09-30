# LongLive c01y_1 — fastprompt replication @ seed 11 (10 s); rc=0 (2026-09-30)

**Job.** `gpuq_job_974zzzzzz_c01y_1_fast10s_s11.sh` — RUNNING 2026-09-30T01:58:26Z → FINISHED **rc=0** 02:02:01Z (dur 214 s).
Config: fp8 (W8A8), W12, sink 8, relative-RoPE on, `LL_OFFLOAD_IDLE=1`, `LL_PROMPT_CHUNK=16`, seed **11**,
OUT=`runs_probe_fast10s_s11`. Same prompt file as c01v_5 (`prompts_street_fast.txt` — the fast-motion prompt that
unlocked camera travel at seed 0). This is the seed-replication test: prompt effect vs seed luck.

**Output.** `len_10s/rank0-0-0_regular.mp4` — 253 frames, 1280×704, 24 fps, 10.54 s, 26,878,778 B,
sha256 `42eae22060e0b3186f462f55f5cb64bfd65346eb4c09584ff36acfc0f5660480` — matches the eval receipt's video
fingerprint `42eae22060e0b318`.

**Eval — no quality claim.** `long_eval` NO_FLAGS; diagnostic horizon 10.5 s of 10.5 s (dinov2). Stability ≠ progression.

**Progression (flow_check, Farneback 640×352).** Median **0.630 px/frame** (p10 0.291 / p90 1.107);
net dx **−12.9 px**; `bound8/intra8` 0.92; halves 0.582 → 0.657. Box re-run: 0.627 (same script; cv2 build diff).

Context (same metric): fastprompt @ **s0 = 1.118** (c01v_5) · ladder @ s0 = 0.40 · ladder seeds 1/2/3 = 0.594 / 0.614 / 0.203.

**Read (Zeph first glance — not the independent review; eyes gate):** forward-walk behavior **replicates** on
seed 11 — contact sheet = coherent continuous walk, no collapse — but magnitude ≈ ½ of s0 ⇒ behavior is
prompt-driven, level is seed-sensitive. Replication series continues: c01y_2 (30 s @ s11), c01y_3 (10 s @ s2) queued.

**Files.** `contact_9.jpg` (md5 `120887e034d4e9a40487cefb9906cdd8`), `flow_c01y_s11.json` (md5 `a3e131cee1d12a1c3387c8e0938b0004`),
`eval_c01y_s11.json` (md5 `904bccc22915f54769bc22d003c63a43`), `job_c01y_1_fast10s_s11.sh` (md5 `ac26677e3901a7a7f6043a6c2bdfa05a`),
`gpuq_timeline.txt`, `run_identity.json`, `rank0-0-0_regular.mp4.provenance.json`.
Video not committed (27 MB; fetch via `clip_harvest`). Nothing published; owner screening pending.
