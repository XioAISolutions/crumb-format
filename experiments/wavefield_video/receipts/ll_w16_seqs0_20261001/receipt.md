# Receipt: W16 seq-suite s0 harvest — _W_half_seq_cw_s0 + _S_ssm_seq_s0 closed

Continues receipts/ll_w16_s2harvest_20261001/receipt.md, which flagged the
seq-suite s0 results as "own harvest pending". Both s0 arms are now in.

## What happened
- _W_half_seq_cw_s0 completed 4000/4000 + final eval (box 2026-09-30 12:09Z).
- _S_ssm_seq_s0 completed 4000/4000 + final eval (box 2026-10-01 05:30Z).
- Both run in SEQ mode (seq_frames=512, eval_rollout=512, persistent state
  64 MiB) — NOT the same eval protocol as the non-seq W16 suite (eval_rollout
  differs, so MSEs are not directly comparable to _W_soft_s2/_W_half_s2).

## Numbers (seed 0; result JSONs md5-verified vs box)
- _W_half_seq_cw_s0: eval_mse 0.002476, /copylast 1.079, exit_dir 0.250,
  id_surv 0.182, r_persist 0.009, pos_err@emg 4.538, train 32194s
- _S_ssm_seq_s0: eval_mse 0.002037, /copylast 0.888, exit_dir 0.375,
  id_surv 0.249, r_persist 0.014, pos_err@emg 4.301, train 8309s
- Read: ssm_seq beats wave_half_seq on eval_mse, exit_dir, id_surv, r_persist;
  both under copy-last baseline MSE but wave_half_seq copies harder
  (copy_ratio 0.946). No early verdict — s1/s2 arms still running.

## Artifacts (box: /workspace/slava/exp/pr63/runs_long_horizon_seq/)
- result_wave_W_half_seq_cw_s0.json md5 69a8fc0f506646c6d441ec8e4da45bb6
- result_ssm_S_ssm_seq_s0.json  md5 a228ec418db78b4f943ea617530b56f9
- log_W_half_seq_cw_s0.txt md5 80536cbe2570748e356280fc51779179
- log_S_ssm_seq_s0.txt      md5 4f40d4bcf2057635d0d29209d0b3a1d3

## Box health at write (16:15Z)
- GPU 100%, 247W, 63C — healthy; no PAUSE.
- Live: _W_half_seq_cw_s1 resumed (step 2800/4000, loss 0.0045), slices at 4200s.
- Queue: 10 pending — _A_attn_s2 resume covered by 9k_2230_3/4/5 (pre-fire
  checked: PY + alloc-conf + log redirect + VRAM gate all present;
  ckpt_attn_A_attn_s2.pt exists on box).

## Proof boundary
- Claimed: two s0 arms at 4000/4000 with final evals; JSONs md5-verified box↔dir.
- Not claimed: s1/s2 arms (running/queued); comparability with the non-seq
  W16 suite; owner-side pixel review.
