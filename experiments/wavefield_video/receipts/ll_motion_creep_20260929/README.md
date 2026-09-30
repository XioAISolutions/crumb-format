# Motion-progression check — c01q/c01r creep diagnosis (2026-09-29)

Owner screened the 30 s clip (c01r): reads as **"same scene skipping"**, not a progressing
walk. Confirmed with a new optical-flow metric (`scripts/flow_check.py`, Farneback on
640×352, median over the frame per consecutive pair):

| clip | median flow (px/frame) | net dx over clip | bound8/intra8 | verdict |
|------|------------------------|------------------|----------------|---------|
| c01q 10 s | 0.404 | −24.8 px | 0.91 | creep |
| c01r 30 s | 0.370 | −57.8 px | 0.96 | creep |

A real walking take lands roughly 5–15 px/frame at 640w; ~0.4 px/frame is ~15× below →
the camera barely translates, and micro-jitter reads as skipping. The block-boundary ratio
(8-frame cadence) is ≤1 — it is uniform creep, not block jumps.

**No-claim note:** `long_eval` NO_FLAGS measures stability (drift / colour / a motion floor),
NOT progression — it passed both clips. This receipt adds the missing metric. Promotion of
any config requires flow numbers + owner screening; `long_eval` is secondary.

**Suspects** (diff vs upstream `configs/fp8/inference_fp8.yaml`): `local_attn_size` 12 (ours)
vs 32 (upstream); harness forces `use_relative_rope: true` (upstream: unset → off). The
harness now carries a toggle (`--relative-rope on|off` / `RELATIVE_ROPE`; tool
`scripts/longlive_ropeflag_patch.py`), and probes `c01u_1..4` isolate prompt dynamics /
window / relative-RoPE.
