R16 TASK: fused single-step Triton streaming kernel (the r5 spec item) — PARKED per Astra hardening research (HARDENING_astra.md): Triton recurrence = rank 9, "only after model survives". Build this only once WaveMemory passes the occlusion memory-probe + hybrid-advantage gates.

When unparked: read triton_fused.py, bench_triton.py, IMPL_NOTES_R6.md, wfvideo.py streaming path.
(1) Fused single streaming step kernel: z_t(k) = lam(k)*z_{t-1}(k) + B*x_t(k) with the in-proj/out-proj + gate fused; complex arithmetic in registers, no intermediate buffer round-trips.
(2) Verify equivalence vs the eager streaming step to 1e-3 on GPU (reuse run_smoke_v3 harness style).
(3) Bench: steps/s + ms/frame at lat 16/32; compare eager/compile/fused. Keep cuFFT where it dominates (chain bench from r6).
(4) run_smoke_r16.sh + IMPL_NOTES_R16.md.
