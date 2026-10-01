# Receipt: W16 s2 harvest — _W_soft_s2 + _W_half_s2 + _S_ssm_s2 closed (11/12); PROVE/KILL read at s2 end

Continues receipts/ll_w16_s1harvest_20260930/receipt.md ("s2 seeds decide both bars;
no early read") and receipts/ll_w16_attns1_20260930/receipt.md (8/12).

## What happened
- Job 999 slice: _W_soft_s2 completed its 4000/4000 + final eval at 02:16Z (result+model
  written; then _W_half_s2 started fresh in the same slice).
- _W_half_s2 completed 4000/4000 + final eval (result+model 08:34Z).
- _S_ssm_s2 completed 4000/4000 + final eval (result+model 09:08Z).
- _A_attn_s2 fresh 09:08Z, sliced at the 4800s budget, resumed 10:06Z, sliced again
  (~11:26Z). Its continuation is already covered by the queued 9k_2230_3/4/5 lh_slice
  jobs — no re-queue action needed.

## Numbers (seed 2; result_*.json copies in this dir, md5-verified vs box)
- W_soft_s2: eval_mse 0.01008, /copy-last 0.598, exit_dir 0.0, id_surv 0.197, pos_err 4.104
- W_half_s2: eval_mse 0.01068, /copy-last 0.634, exit_dir 0.125, id_surv 0.001, pos_err 4.619
- S_ssm_s2:  eval_mse 0.01032, /copy-last 0.613, exit_dir 0.375, id_surv 0.089, pos_err 4.523
- Baselines: zero 0.01730, copy-last 0.01684, const-vel 0.01662 (unchanged).

## Pre-registered read at s2 end (inputs now complete)
- PROVE (W_half exit_dir >= 0.8 AND +0.2 over W_soft & A_attn on >= 2/3 seeds):
  W_half exit_dir s0/s1/s2 = 0.375 / 0.125 / 0.125 — the >= 0.8 clause fails on all
  three seeds, so the AND is false regardless of A_attn_s2. PROVE: NOT MET.
- KILL (halflife <= W_soft exit_dir on >= 2/3 seeds): s0 0.375 vs 0.125 (no),
  s1 0.125 vs 0.250 (yes), s2 0.125 vs 0.0 (no) = 1/3. KILL: NOT TRIGGERED.
- Verdict: the pole line is neither proven nor killed — halflife shows no exit-dir
  advantage over softplus at this scale; exit_dir is seed-noise (0.0-0.5 band) and
  does not support the long-pole hypothesis. No early suite read was made; this read
  waited for s2 W_half + W_soft, both now in. (attn_s2 pending but cannot flip either bar.)

## Artifacts (box: /workspace/slava/exp/wavefield_video/runs_long_horizon/)
- result_wave_W_soft_s2.json md5 402fab02dec026d853803b419c090445 (copy in dir)
- result_wave_W_half_s2.json md5 3b4797a6565bb728b58cb29a43ed244c (copy in dir)
- result_ssm_S_ssm_s2.json  md5 40bbd805e21ee2825561432ecdac83c8 (copy in dir)
- model_wave_W_soft_s2.pt md5 50c7ddc988c921c18bbe0fc20483124d; half 6d1271743478b206a3aca02c6c99d4ad; ssm 7a8cf615d74f2218bb35654ddf584ca8
- logs: log_W_soft_s2.txt 041d06624e32321f3c064b3b73bc36f7; log_W_half_s2.txt b1b9eff1d3bcf7eaa21e280a65028d54; log_S_ssm_s2.txt 045d75cd83e78581498ef29e5d6094fd

## Box health at write (12:15Z)
- GPU 100%, 247W, 64C — healthy. No PAUSE file.
- Live: seq lane _W_half_seq_cw_s1 (resumed 11:28Z, ~step 819+/4000, slices at 4200s).
- Queue: 10 pending — 9k_1030_2 (running), 9k_1130 tpl latent x3, 9k_2230_3-5 lh_slice
  (resume _A_attn_s2), 9k_2230_6-8 seq_cont. Brief's "FAILED encode exit=1" line is the
  stale Sep-28 runs_latent tail (mtime-checked) — not live.

## Proof boundary
- Claimed: three s2 arms reached 4000/4000 with final evals run; result JSONs md5-verified
  (box vs this dir). PROVE/KILL read is arithmetic on those JSONs per LONG_HORIZON.md #7.
- Not claimed: any read on A_attn_s2 (still training); seq-suite s0 results (exp/pr63,
  own harvest pending); owner-side pixel review.
