# ll_w16_attns2 receipt — 2026-10-02

## Box
- Job: gpuq_job_n1_long_horizon_slice.sh, RUNNING 16:42:44Z → result write 17:21:49Z (resume from ckpt_attn_A_attn_s2.pt).
- Leg: _A_attn_s2 (old suite 12th leg) → old suite 12/12 COMPLETE.

## Result (mse lower=better; r>1 beats persistence)
- eval_mse 0.01088 (s1 0.01099, s0 0.01102) — attn arm best of three seeds
- model/copy-last 0.633 | copy_ratio 0.594 | r persist/model 0.088
- exit_dir_acc 0.25 | id_survival 0.08 | params 1,999,903 (ffn_mult 13.055)
- baselines: zero 0.01764, copy-last 0.01719, const-vel 0.01685

## md5
- result_attn_A_attn_s2.json 9a35ac3576c1542f5bc5c83f3217ea43
- model_attn_A_attn_s2.pt 2cac28a1f0891aaa1094796805ca6dc5 (on box, not pulled)
- log_A_attn_s2.txt (full, box+local) 7d1b6a66ca2afbd0624b56904890985f
- log_A_attn_s2.crop.txt (tail 120 lines) 0866aecc2d469598acae531d1aea9b5d

## OPSD 957 (context, this pass)
- 9z_c01ac_1 retry rc=-9 @14:04:16Z dur=1187s — SIGKILL again (prior 1116s). 4-min kill reproduced.
- Proven max remains 633 frames (2.6 min, out_633). 795 bisect = corridor/owner call.

## Lane state
- Box post-17:35Z: 9k tpl chain no-op cycling (all legs done), GPU idles between keeper restages; EMPTY alerts every 10 min.
- Next work owner-gated: W17 aim / keeper stop / next batch.
