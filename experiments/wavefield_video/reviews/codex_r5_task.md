R5 TASK (new standalone files only; do not modify existing .py; no git commits).
(1) data_waves.py — 2D PDE field generator, API-compatible with data.py.
[B,T+1,3,H,W] float [0,1]; fields: wave / advection / vortex.
Channels: [field, dt, laplacian] rescaled. Spectral evolution via torch.fft.
Grids 16/32/64, T up to 64, deterministic seeds, moving_mask-compatible.
Analytic test: 2D traveling-wave solution error < 1e-3 vs closed form.
(2) audit_wavefield.sh: py_compile all .py, run smoke v3 + v4, print PASS/FAIL manifest.
(3) proofpack.py + verify_proofpack.py: sha256 manifest + VERIFY.md repro commands.
Deliver: files + IMPL_NOTES_R5.md + tests/test_waves.py. Run your tests yourself.
proofpack: bundle a run dir (result JSON + ckpt + config): manifest.json with sha256 + VERIFY.md with exact repro commands; verify_proofpack re-hashes.
