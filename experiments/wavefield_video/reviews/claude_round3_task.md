Implement the streaming + hardening round (keep all existing defaults intact):

(1) RECURRENT STEP for the dispersion wave kernel: add init_state(x_shape) + step(x_t, state) to WaveMix3D's dispersion path implementing the exact
frequency-domain recurrence z_t(kx,ky) = lam(kx,ky)*z_{t-1}(k) + B*x_t(k), out = Re(C*z_t) -> iFFT -> to_out. lam/B/C must reproduce forward() math
exactly. Verification: unrolled step() over T frames == forward() (causal, tol 1e-3); add to the sanity harness.

(2) --stream-test in train_compare.py: 1024-frame streaming rollout using step() (wave) vs windowed rollout (attn/ssm); log peak memory + fps every 128 frames to JSON.
(3) --auto-batch: catch CUDA OOM in train loop -> halve batch, zero grads, retry (max 3); same for eval; empty_cache between eval passes.
(4) --save-every N + --resume PATH: checkpoints {state, opt, step}.
(5) Do not run python (gated); write run_smoke_v3.sh + IMPL_NOTES_V3.md; operator will run.
