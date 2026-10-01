# Receipt: W16 A_attn_s1 closure — attn s1 arm complete (8/12); _W_soft_s2 live on 998

Continues receipts/ll_w16_s1harvest_20260930/receipt.md (at its write time: "_A_attn_s1 live
on job 996 ... step ~1.3k/4000").

## What happened
- Job 996 ran _A_attn_s1 fresh from 21:23:49Z and was cleanly SLICED at the 4800 s budget
  (job ended 22:43:49Z, rc=0).
- Job 997 (23:11:35Z) skipped the 7 then-completed legs and resumed _A_attn_s1 23:11:41Z
  from ckpt_attn_A_attn_s1.pt: trained to 4000/4000 and ran its final eval (eval-batch 1,
  eval_chunk 1) — result_attn_A_attn_s1.json + model_attn_A_attn_s1.pt written 00:04:15Z.
  Arm complete (8/12).
- The same slice then opened _W_soft_s2 fresh 00:04:16Z and was cleanly SLICED at the 4800 s
  budget (job ended 00:31:41Z, rc=0; step ~775 logged, ckpt saved 00:26Z).
- Job 998 began 00:32:21Z, skipped the 8 completed legs and resumed _W_soft_s2 00:32:26Z from
  ckpt — live and healthy at write time (step ~825/4000; GPU 100%, 66 C, ~291 W).

## Numbers (seed 1; result_*.json on box)
- A_attn_s1: eval_mse 0.01099 (s0 0.01102), /copy-last 0.64 (s0 0.641), exit_dir 0.0
  (s0 0.375), id_surv 0.188 (s0 0.254), copy_ratio 0.58 (s0 0.605), mean/final tce
  6.122/9.372.
- Baselines s1: zero 0.01764, copy-last 0.01719, const-vel 0.01685.
- Read posture unchanged: no PROVE/KILL mid-suite read (s2 seeds decide both bars;
  pole-line exit_dir s1 so far: W_soft 0.250, W_half 0.125, A_attn 0.0 — noisy, no read).

## Artifacts (box: /workspace/slava/exp/wavefield_video/runs_long_horizon/)
- result_attn_A_attn_s1.json 12801 B md5 0c701f01943a26c744ba8a6797e6ca2b (copy in dir)
- model_attn_A_attn_s1.pt md5 89d214fd9b0b3fa5976410894adae7ac; ckpt_attn_A_attn_s1.pt
  md5 2c526da364a27e0009bfe8de988c07b7
- log_A_attn_s1.txt md5 a822518c949e75e9e3f4c563ae102aab

## Queue / next (at ~00:40Z)
- 998 running (_W_soft_s2 resume; JOB budget ends ~01:52Z) -> 999 next (same v2.3 slice
  template; pre-fire items verified: PY= + expandable_segments + own log redirect +
  SLICE_S=4800 under the conductor cap).
- Then: 9k tpl block (1330/1703 latent no-ops; seq continues) -> 9k_2230 batch (6 LH
  slices + 3 seq) -> 9z seedvr_journey (journey HD for owner pixels review; prior attempt
  SIGKILL'd rc=-9 mid-assembly — no OOM; retry sits last in queue).

## Proof boundary
- Claimed: _A_attn_s1 reached 4000/4000 and its final eval ran; result JSON md5-verified
  (box vs this dir); model/ckpt/log md5s recorded; slice chain read from progress.txt +
  gpuq.log.
- Not claimed: any PROVE/KILL read (suite-end); journey-HD outcome (queued); seq-lane
  closure (separate).
