R7 TASK: LATENT SPACE build — the gate to real pixels. New file latent_ae.py; edit allowed: train_compare.py (--latent flag). No commits.
(1) latent_ae.py: small conv AE for [B,3,H,W] frames. Encoder 3->32->64 stride2 x2; decoder mirrored. encode/decode APIs + train mode: python latent_ae.py --train --data-source balls --grid 32 --steps N --out ckpts/ae.pt; report recon MSE; CPU-quick smoke.
(2) train_compare.py --latent: freeze AE, train kinds on encoded tokens, predict next latent, decode for pixel-space eval (same JSON fields).
(3) run_smoke_r7.sh: AE roundtrip works + tiny latent train runs + balls regression unchanged. IMPL_NOTES_R7.md with operator commands.
