R14 TASK (REVISED): WaveMemory memory-probe — long-memory occlusion benchmark. THE decisive experiment (see HARDENING_astra.md section 'The experiment that matters most'). Read data.py, train_compare.py, ssm_lite.py, wfvideo.py, IMPL_NOTES_R10-R12. No commits.

(1) New scenario (data_occlusion.py): 64x64, 8 moving balls + static occluder rectangle. Target ball: visible frames 0-64, occluded 64-320 (occluder drawn OVER it; its physics continues deterministically incl. wall bounces), visible again 320+. Target color = identity. Exit direction at emergence is fully determined by pre-occlusion state. Deterministic, seedable, same API style as data.py.

(2) run_occlusion.sh: 5 arms at equal param budget (~4M): A local-attn, B generic-ssm (ssm_lite), C wave (dispersion), D local+ssm hybrid, E local+wave hybrid. Hybrids: implement minimal gated fusion wrapper h = g*h_local + (1-g)*h_wave with g = sigmoid(linear([h_local; h_wave])) in wfvideo.py (new --fuse local_ssm|local_wave flag). 2000 steps each, 5 seeds, eval-rollout 1024 (streaming for wave/ssm; windowed for attention).

(3) Metrics per arm: target identity survival, exit-direction accuracy at emergence, position/velocity error at emergence, divergence horizon, motion preservation, VRAM, ms/frame, and STATE BYTES (persistent recurrent state size = the killer metric).

(4) run_smoke_r14.sh (CPU, synthetic: occlusion scenario correctness: target hidden exactly 64-320, deterministic; fusion wrapper parity; runner dry-run) + IMPL_NOTES_R14.md with exact box commands.

(5) Kill criterion (from the research): if local+wave does not beat local+generic-ssm on divergence-horizon-per-byte, the wave formulation is not pulling its weight.
