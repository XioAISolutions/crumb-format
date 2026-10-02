# Receipt: W16 seq-suite s1 COMPLETE — _S_ssm_seq_s1 closed (both s1 arms)

Continues the first version of this receipt (@ 1c7c450, wave arm only). _S_ssm_seq_s1 finished on the box at ~01:53Z Oct 2 (the slice job moved on to _W_half_seq_cw_s2 at 01:53:37Z).

## What happened
- _S_ssm_seq_s1 completed 4000/4000 + final eval; result JSON + model_ssm export written on the box.
- Same SEQ protocol as the s0/s1 wave arms (seq_frames=512, eval_rollout=512, TBPTT=1, dense) — apples-to-apples.

## SSM s1 numbers (vs wave s1 / s0)
- eval_mse: 0.002290 vs 0.002763 / 0.002476 — SSM best raw MSE of the three
- eval_mse_over_copylast: 0.9977 vs 1.2038 / 1.0789 — SSM sits at the copy-last baseline
- copy_ratio: 0.033 vs 0.9852 / 0.9459 — SSM does NOT copy-last; different predictor, same MSE
- exit_direction_accuracy: 0.500 vs 0.250 / 0.250 — SSM at chance, wave below chance
- final_target_centroid_err: 4.627 vs 7.656 / 5.538 — SSM best tracking
- position_error_at_emergence: 5.977 vs 6.891 / 4.538
- target_identity_survival 0.209; motion_preservation 0.038
- divergence_horizon 1 / emergence 288 — same profile as wave
- params 690691; persistent_state_bytes 64 MiB; rollout 428.5 fps (2.33 ms/frame)
- Read: no PROVE/KILL movement (0.5 << 0.8) — but SSM is the first arm to abandon copy-last at copylast-equal MSE and beats wave on exit acc + tracking. Seed 2 in flight.

## Artifacts (box /workspace/slava/exp/pr63/runs_long_horizon_seq/)
- result_ssm_S_ssm_seq_s1.json  md5 a7d888052b3291ee4252743d9bcebf0e
- log_S_ssm_seq_s1.txt          md5 aa8f608e522f6163f4c0697259b8f439
- model_ssm_S_ssm_seq_s1.pt on disk (not md5-claimed in this dir)
- wave s1 files (51ca72bf... / 3bf3ad11...) as in the first version

## Box health at write (~02:15Z Oct 2)
- _W_half_seq_cw_s2 running (slice resumed 01:57:50Z, SLICE_S=4200, timeout 4500); S_ssm_seq_s2 fresh next. seq_cont auto-top-up: keeper hourly tpl batch + 2230_6/7/8 staged.
- Old-suite _A_attn_s2 (runs_long_horizon) still SLICED since 10:06Z Oct 1; continuation staged = 9k_2230_3/4/5 lh_slice v2.3, fires after the 0200 tpl batch. ComfyUI unreachable counts as not-busy (v2.2); free-mem gate applies.
- runs_latent "FAILED encode" brief line confirmed stale (progress mtime Sep 28).

## Proof boundary
- Claimed: SSM s1 closed at 4000/4000 + final eval; JSON/log md5-verified box↔dir.
- Not claimed: seed 2 results; stream stage; owner pixel review.
