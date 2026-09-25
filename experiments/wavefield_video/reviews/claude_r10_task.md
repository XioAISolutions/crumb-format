R10 TASK: scale-aware objective + eval hardening. Read DEEP_DIVE_2_opus.md and DEEP_DIVE_2_astra.md first. Edits: train_compare.py (+data.py if needed). No commits.
(1) --resid-balanced: restrict the 0.25*resid term to moving pixels (Opus leak #1). Default off = unchanged.
(2) --motion-weighted: soft per-pixel weights w = clamp(delta/(delta.mean()+eps), 1, w_max) applied to the motion loss (Astra #1 soft variant). Default off.
(3) --radius / --speed passthrough to the clip generator (defaults = current constants) for the geometry control (Astra exp 2).
(4) Fix eval OOM: chunk rollout-eval seeds into sub-batches (--eval-chunk, default 4). This OOM killed 3 runs tonight.
(5) run_smoke_r10.sh: each flag proven on tiny CPU runs + default-path regression. IMPL_NOTES_R10.md with exact retune-suite commands (B/C/geometry runs).
