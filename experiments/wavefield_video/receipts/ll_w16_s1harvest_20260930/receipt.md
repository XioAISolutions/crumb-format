# Receipt: W16 s1 harvest — _W_half_s1 + _S_ssm_s1 legs closed; slice runway topped

Continues receipts/ll_w16_s1eval_20260930/receipt.md (at its write time: "995 running
(_W_half_s1; expect closure inside the slice, then _S_ssm_s1 starts").

## What happened
- Job 995 (resumed _W_half_s1 ~19:55Z from ckpt, slice cap 4800 s): _W_half_s1 trained to
  4000/4000 and ran its final eval (eval-batch 1) — result_wave_W_half_s1.json +
  model_wave_W_half_s1.pt written 20:41Z. Arm complete.
- Same slice then started _S_ssm_s1 fresh 20:41:22Z: trained 4000/4000 (train_sec 1856.9,
  2.15 s/step telemetry) and completed its final eval by 21:12:31Z — arm complete inside a
  single slice (~31 min total).
- The slice then opened a ~3-min _A_attn_s1 stub (no ckpt saved) and was cleanly SLICED at the
  4800 s budget (~21:15Z). Job 996 began _A_attn_s1 fresh 21:23:49Z — live and healthy at
  write time (step ~1.3k/4000, ckpt saved 21:53Z, GPU 100%).
- Suite: 7/12 legs done (s0 x4; s1: W_soft + W_half + S_ssm); attn_s1 (8th) in progress.

## Numbers (seed 1; result_*.json on box)
- W_half_s1: eval_mse 0.01065 (s0 0.01113), /copy-last 0.632 (s0 0.648), exit_dir 0.125
  (s0 0.375), id_surv 0.032, copy_ratio 0.626.
- S_ssm_s1: eval_mse 0.01061 (s0 0.01109), /copy-last 0.630 (s0 0.648), exit_dir 0.250
  (s0 0.500), id_surv 0.237, copy_ratio 0.609.
- Baselines s1: zero 0.01730, copy-last 0.01684, const-vel 0.01662. All three closed s1 arms
  beat their own s0 eval_mse; exit_dir is noisy across seeds (mid-suite note, no read).

## Pre-registered read (not triggered; reads at suite end)
- PROVE: W_half exit_dir >= 0.8 and +0.2 over W_soft & A_attn on >= 2/3 seeds — no
  (W_half 0.375/0.125 so far; W_soft 0.125/0.250).
- KILL: halflife <= W_soft exit_dir on >= 2/3 seeds — currently 1/2 (s1 yes, s0 no).
  s2 seeds decide both bars; no early read.

## Artifacts (box: /workspace/slava/exp/wavefield_video/runs_long_horizon/)
- result_wave_W_half_s1.json 13011 B md5 2fb64fff6fa29b6340ac6b4c0d9d8b21 (copy in dir)
- result_ssm_S_ssm_s1.json 12781 B md5 c1f7460667ed0e8e07f291007ae08561 (copy in dir)
- model_wave_W_half_s1.pt md5 c50d519fa6352f786fe7856ad7ddb984; ckpt_wave md5 035ea65d794a310a32f3ca1a14e0d0e6
- model_ssm_S_ssm_s1.pt md5 812d236bc31812d86e6308968e7f4f04; ckpt_ssm md5 9caf3854066c579a5ef75e03fb5591c5
- logs: log_W_half_s1.txt md5 4b33fb5869582ad7366b024f9d0a8b21; log_S_ssm_s1.txt md5 f3ba4d4eb0e908076f33480594ee2fbc

## Queue / next (at ~22:25Z)
- After 996: seedvr_journey (pre-fire verified — journey_master.mp4 34.9 MB in inbox, CLI
  present) -> 997-999 (byte-same slices) -> 9k tpl block (4 latent copies no-op: runs_latent2
  + runs_latent3 DONE; seq: _S_ssm_seq_s0 at step 3178/4000 -> 2 slices) -> 9k_2230 batch.
- Topped the runway this pump: +6 LH suite slices (9k_2230_0..5 md5s ce54d089a431dc9548ee18de6fd418ac,
  d378e56915a2d7c0e80ca25903964976, d87b23611c55c885d26c5434140d524b, 3011ab60e47331b2a1a5d94b461e1363,
  5d9ac67025566b4839600a178a255c97, da40d79c5cd75b0aae3c98322ccb5e89) +3 seq slices
  (9k_2230_6..8 md5s 8cfe1e1c602d01500502453340904ce3, ad5b6474e0320fc60747caa6d4814f30,
  46c8a6c600832dae0d1aeb622db8586b); all bash -n clean, mode 755.
- exp/wavefield_video carries the eval-batch fix; repo/pr63 trees unchanged (corridor
  upstreaming call stands).

## Proof boundary
- Claimed: _W_half_s1 and _S_ssm_s1 each reached 4000/4000 and their final evals ran; result
  JSONs md5-verified (box vs this dir). Staged slice copies are byte-same templates + marker
  comment only.
- Not claimed: any PROVE/KILL read on the pole line (s2 pending); eval wins beyond the printed
  numbers; seq-lane results (own closure pending).
