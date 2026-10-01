# c01ad_1 — OPSD-V 633 frames (2.6 min) — rc=0, travel sustained

OPSD-V (Wan2.1-T2V-1.3B + opsdv LoRA) on the 24 GB 4090, 832×480 @16 fps,
633 output frames = 158.06 s. Longest successful OPSD clip so far; the 957-frame
attempt (≈4 min) died SIGKILL (system RAM) at ~18.6 min.

## Numbers (median px/frame @640w)
| run | len | flow | net px | rmse |
|---|---|---|---|---|
| OPSD orig | 39.5 s | 1.740 | +28.8 | 26.2 |
| OPSD lora | 39.5 s | 1.960 | +13.7 | 28.2 |
| OPSD 477 (c01aa_1) | 119 s | 1.835 | −225.4 | 27.72 |
| **OPSD 633 (c01ad_1)** | **158 s** | **1.836** | **−258.0** | **28.77** |
| OPSD 957 (c01ac_1) | ~239 s | — | — | SIGKILL, no clip |

Flow flat with length (477→633 same median); halves 1.757→1.893 (no decay);
p10/p90 0.987/3.053; bound/intra 0.94 (no chunk-boundary break).

## Contact (first glance — Zeph; independent review required)
3×3 sheet: new street segments in every panel — doorways, window grilles, flower
boxes, shop fronts passing; p9 (curving street with storefronts) ≠ p1 (narrow
alley). No collapse; no visible smearing at sheet scale.

## Provenance
- job `gpuq_job_974zzzzzz_c01ad_1_opsd633.sh`: rc=0, 715 s (03:58:56Z→04:10:51Z)
- `log_c01ad_633.txt` md5 `11b8aef61611275a30c92b05fbfde725` — clean, no traceback/OOM lines
- `args_c01ad_633.txt` — seed_list [1], 633 frames, lmdb pipeline; patch chain live
- `flow_c01ad_633.json`; `contact_9.jpg`
- clip sha256 `c2345885d6eaec4d4f75e1e99ffb5de0fea1c438af9314ce4b7085ec6c0f6431` — matches box copy bit-for-bit (46,660,754 B, 2529 fr)

## Companion (contacts/)
`c01ab_1_fast300s_s0.{jpg,flow.json}` — LongLive 5-min MISSION RUNG contact sheet
(flow 1.165); the merged fastbreak README cited numbers but the sheet existed only
in local scratch.
