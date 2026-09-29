# LongLive c01r — 30 s rung passed (2026-09-29)

Job `gpuq_job_974zzzzzz_c01r_30s.sh` — RUNNING 19:20:48Z → FINISHED rc=0 19:26:37Z (349 s).
Config: fp8 W8A8, window 12, sink 8, LENGTHS=30, `LL_OFFLOAD_IDLE=1`, OUT=`runs_longlive_cfg16e`.
Idle-offload on the live run: `[mem] idle-offload: TE+VAE -> cpu; alloc=4.90 GiB`.

Output: `len_30s/rank0-0-0_regular.mp4` — 733 frames, 1280×704, 24 fps, 30.54 s, 78,996,132 bytes
(`wall 344 s for 30s of video`).

Eval (`long_eval`, hardened, receipt `24778fdb8f0a6021`): **NO_FLAGS**, diagnostic horizon 30.5 s of 30.5 s.
Two windows, both `fails: []`:
- 0 s: sim 0.979, drift 1.000, motion 0.0860
- 30 s: sim 0.977, drift 0.998, motion 0.0786 (threshold 0.25 × baseline)

Semantic quality unverified — independent pixel review required; owner screening pending; not published
anywhere. Contact sheet `contact_30s.jpg`; full log `log_30s.txt`.
Note: left-edge smear ~5.2–5.4 s reported in first-glance review of the 10 s c01q clip (same opening);
check ~0:05 on screening.

Next: `c01s` (LENGTHS=180, `runs_longlive_cfg16f`) then `c01t` (LENGTHS=300, `runs_longlive_cfg16g`)
queued, same config; est. ~35 min + ~57 min generation once they reach the front.
