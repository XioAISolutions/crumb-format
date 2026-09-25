R6 TASK: Triton prototypes. New files only (triton_fused.py, bench_triton.py); do not modify existing modules. No commits.
Context: torch.compile cannot codegen our complex FFT path. Goal: fused Triton kernels, manual real/imag packing (Triton has no complex dtype).
(1) triton_fused.py: kernelA fused_complex_mul (rfft re/im x kernel re/im -> product + optional gate scale, ONE pass).
kernelB fused_step_inner: z = lam*z + b*x over [B,M,H,W] as 2-reals, one pass.
Eager reference impls + self-check kernel-vs-reference (CUDA only; clean exit on CPU). float32, block over flattened spatial.
(2) bench_triton.py: CUDA bench: eager rfft/cplxmul/irfft chain vs fused; step update eager vs fused; grid 16/32 heads 8; write bench_triton.json.
(3) IMPL_NOTES_R6.md: signatures, limits, exact operator run commands (operator has the CUDA box; you cannot run CUDA).
