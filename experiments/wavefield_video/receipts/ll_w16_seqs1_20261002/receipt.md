# Receipt: W16 seq-suite s1 harvest — _W_half_seq_cw_s1 closed

Continues receipts/ll_w16_seqs0_20261001/receipt.md (s0 harvest). The s1 wave
arm finished on the box; _S_ssm_seq_s1 is still mid-training (resumed 23:33Z,
4200s slices).

## What happened
- _W_half_seq_cw_s1 completed 4000/4000 + final eval (RESULT line in log).
  result JSON + model_wave export written on the box.
- SEQ mode (seq_frames=512, eval_rollout=512) — same protocol as s0, so the
  s0↔s1 comparison is apples-to-apples.

## Numbers (seed 1 vs seed 0)
- eval_mse: 0.002763 vs 0.002476 (+11.6%)
- eval_mse_over_copylast: 1.2038 vs 1.0789 (s1 above the copy-last baseline)
- copy_ratio: 0.9852 vs 0.9459 (s1 copies harder)
- exit_direction_accuracy: 0.250 vs 0.250
- divergence_horizon: 1 vs 1; emergence_frame: 288 vs 288; div_consec: 8 vs 8
- position_error_at_emergence: 6.891 vs 4.538
- final_target_centroid_err: 7.656 vs 5.538
- Read: s1 seed is clearly worse on every magnitude metric (consistent with
  the known seed-sensitivity pattern); divergence profile identical. Neither
  seed moves the PROVE/KILL gate.

## Artifacts (box: /workspace/slava/exp/pr63/runs_long_horizon_seq/)
- result_wave_W_half_seq_cw_s1.json  md5 51ca72bf31703e2fd4a6feccbb7c6776
- log_W_half_seq_cw_s1.txt          md5 3bf3ad11cadb95886c6a97c482b51509
- model_wave_W_half_seq_cw_s1.pt on disk (not md5-claimed in this dir)

## Box health at write (~00:20Z Oct 2)
- GPU 100%, ~249W — _S_ssm_seq_s1 live (resumed 23:33Z, 4200s slice budget).
- Queue 10 pending, keeper auto-topping-up; no PAUSE. _A_attn_s2 continuation
  covered by 9k_2230_3/4/5 (v2.3 lh_slice, pre-flight verified); seq cont
  2230_6/7/8 re-checked this pass (PY + alloc-conf + timeout 4500 OK).

## Proof boundary
- Claimed: s1 wave arm at 4000/4000 with final eval; JSONs md5-verified
  box↔dir.
- Not claimed: _S_ssm_seq_s1 (running); owner-side pixel review; stream stage.
