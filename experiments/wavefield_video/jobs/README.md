# Job scripts (wave-2 + OPSD A/B), 2026-09-29/30

Box-side gpuq job scripts, as deployed to `/workspace/slava/` and copied into
`gpu_queue/pending/` as `gpuq_job_974zzzzzz_<name>.sh`.

**Every job script must pin `PY=/workspace/slava/venvs/longlive/bin/python` and
use the `exec > log` redirect + `/usr/bin/timeout` wrapper** — the dispatcher's
PATH has no plain `python`; scripts without the pin die rc=2 in 0 s.

- `c01v_1_bf16.sh` — bf16-vs-fp8 dynamics control (W24, 10 s)
- `c01v_2..4_seed{1,2,3}.sh` — seed sweep at ladder config (W12 rel-rope)
- `c01v_5_fastprompt.sh` — brisk-travel prompt variant
- `c01w_1_i2v.sh` — I2V anchor-continuation (c01q frame 0), harness `I2V=1`
- `c01x_1_opsdorig.sh` / `c01x_2_opsd.sh` — OPSD-V stack A/B (LongLive-1.3B,
  original vs OPSD post-trained lora; repo `/root/opsd-v`, venv `venvs/opsd`)
