# Receipt: W16 _W_soft_s1 final-eval CLOSED -- eval-batch-1 fix GPU-proven (job 994)

Closes the proof boundary declared in receipts/ll_w16_evalbatch_20260930/receipt.md
("GPU proof = copy 994 completing _W_soft_s1's eval").

## What happened
- Job 994 (gpuq_job_994_long_horizon_slice.sh, md5 9fcee650..., byte-identical to the
  fix-receipt copy) fired 18:34:26Z. After the SKIP pass it resumed _W_soft_s1 at step
  4000/4000 and ran the previously-OOMing FINAL EVAL with --eval-batch 1: PASSED (~25 s),
  writing result_wave_W_soft_s1.json + model_wave_W_soft_s1.pt at ~18:34:56Z -- arm complete.
- It then started _W_half_s1 fresh, ran crash-free, and was cleanly SLICED at the 4800 s
  slice budget. Job rc=0, dur 4804 s, ended 19:54:31Z.
- Job 995 resumed _W_half_s1 19:55:11Z from ckpt (TELE advancing, GPU 99 %/294 W observed; ~step 2.8k/4000 at write time).

## Numbers (seed 1, W_soft; result json + log RESULT TABLE)
- eval_mse 0.00957 (same arm seed 0: 0.01054); model/copy-last 0.568 (s0: 0.613).
- Baselines: zero 0.01730, copy-last 0.01684, const-vel 0.01662.
- copy_ratio 0.641; r=persist/model 0.025; div_horizon 1/512.
- Emergence: exit_dir_acc 0.25, pos_err 4.742, vel_err 1.001, id_survival 0.011,
  motion_pres 0.053 (mid-suite values; pre-registered PROVE bar reads at suite end).
- eval_chunk 1 + eval_batch 1 (fix live); steps 4000; params 2000015.

## Artifacts (box: /workspace/slava/exp/wavefield_video/runs_long_horizon/)
- result_wave_W_soft_s1.json -- 12530 B, md5 6e4e9c9261f030cab9203c9c61b54678 (copy in this dir; md5-verified).
- log_W_soft_s1.txt -- 50288 B, md5 12f22b28dc17aec1b2b3de55ebb4adce; success block = log_W_soft_s1_success_crop.txt (lines 155-170; md5 0643847d77d2d24300dbe4cdc4b50ac1).
- ckpt_wave_W_soft_s1.pt -- 24090306 B, md5 90538f0d3d4a9f7eb432828d12077225; model_wave_W_soft_s1.pt -- 8024898 B, md5 70d6c0f3e52ce5da1d1aad69d9282848.

## Queue / next (at write time ~20:05Z)
- 995 running (_W_half_s1; expect closure inside the slice, then _S_ssm_s1 starts); 996-999 pending (byte-same template); then tpl 9k (seq_cont/latent lanes). ~5 h staged arm runway; restage more slices next pump if suite unfinished.
- Suite progress: 5/12 legs done (s0 x4 + W_soft_s1); s1 remainder + s2 x4 to go.
- Box tree exp/wavefield_video carries the fix (runner md5 0fdd4a92..., train_compare md5 a5795254...); repo tree files are still pre-fix -- upstreaming to the repo tree is a corridor/owner call.

## Proof boundary
- Claimed: eval-batch-1 removes the final-eval OOM; _W_soft_s1 arm complete (result + model written); fix stays live for the remaining slices. Evidence: 994 rc=0; result/model mtimes ~18:34Z; 995's start skips s1 (no re-attempt); md5-verified artifact copies above.
- Not claimed: any quality verdict on the W16 suite (mid-flight; gates read at suite end).
