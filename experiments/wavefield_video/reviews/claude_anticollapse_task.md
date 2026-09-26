ANTI-COLLAPSE BUILD TASK (Opus). Context: freeze diagnosis suite shows constant-LR, fp32, no-decay all fail to unfreeze g32 at 2k steps; research (RESEARCH_SWEEP_20260926.md) points to: VICReg-style variance-covariance regularizer on frame deltas; drift-perturbed training; discrete history representation. ALSO: DEEP_DIVE_3_opus.md decision tree (hybrid gated fusion = Rank 1).

Deliverables (train_compare.py + wfvideo.py edits, no commits):
1) --var-reg FLOAT (default 0.0=off): variance-matching loss on per-step frame deltas: penalize (std(pred_delta) - std(gt_delta))_+ style hinge, batch-aggregated, added to loss; log its value in step rows.
2) --drift-pert FLOAT (default 0.0=off): drift-perturbed training augmentation: before feeding context, apply random smooth low-frequency gain/bias field (magnitude = FLOAT, same family as crumb_coherence run_m0 inject_drift) to context frames only.
3) --hist-disc {off,bits}: discrete history representation: quantize context frames to N levels (e.g. 16) before feeding model, keep targets continuous.
4) run_anticollapse.sh: 4 arms at g32 2000 steps seed 0, one-knob-each: var-reg 0.5, drift-pert 0.05, hist-disc 16, combo best-two. Telemetry on (--telemetry-every 25).
5) run_anticollapse_smoke.sh: CPU tiny-grid smoke all flags on/off + assert default-path no-op. IMPL_NOTES_ANTICOLLAPSE.md. Append to RESEARCH cross-refs. Fire when trained arms finish on the box; keep commands box-ready (BLOCKING notes if needed).
