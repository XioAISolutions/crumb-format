Objective-fix round (Astra #2 + Opus #4). Read DEEP_DIVE_astra.md sec 1 first.

1) data.py: return an optional 'moving' mask batch (|target-last|>thresh).
2) train_compare.py: --motion-loss = 0.5*mean(E[moving]) + 0.5*mean(E[static]) + 0.25*MSE(pred-last,target-last);
   add copy_ratio = ||pred-last||/||target-last|| to eval + result JSON + table.
3) --rollout-ramp: K 3->8 linearly over the second half of training.
4) run_smoke_v4.sh + IMPL_NOTES_V4.md; CPU smokes prove each flag; no commit.
