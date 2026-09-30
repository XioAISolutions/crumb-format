# OPSD-v OPSD-LoRA arm — c01x_2 first clip (OPSD-LoRA adapter on LongLive base); rc=0 (2026-09-30)

**Job.** `gpuq_job_974zzzzzz_c01x_2_opsd.sh` — RUNNING 02:06:36Z → FINISHED **rc=0** 02:09:54Z (dur 197 s; 3rd attempt).
Args `args_opsd.txt` (md5 `98ed98426532f26e1acfe4f3d0b4a62c`); LoRA = `checkpoints/opsdv_longlive_lora.pt`.

**Output.** `opsd_out_opsd/000000_….mp4` — 633 frames, 832×480, 16 fps, 39.56 s, 11,817,178 B,
sha256 `10925fcbe985a1b413110eaca0535eefbe1141c253fb643aa1d048be09151605`; traceback/OOM grep = 0.

**Progression (flow_check; native 16 fps).** Median **1.96 px/frame** (p10 1.089 / p90 3.300); net dx +13.7 px;
rmse 28.2; `bound8/intra8` 0.94; halves 2.266 → 1.757.

**Pair result (orig ↔ LoRA, same pipeline/seed/159 frames).** orig **1.74** ↔ LoRA **1.96** px/frame;
contacts: both arms = first-person walk with large viewpoint changes (Zeph first glance — not the independent
review). A/B read next.

**Files.** `contact_9.jpg` (md5 `e7d35f3ae1c5582b278ddb45eeb46423`), `flow_c01x_2_opsd.json` (md5 `4c37bff8246b00b36c9368be9cfda296`),
`log_c01x_2.crop.txt` (md5 `899f1f0b8e00f145d3d6e30e3ead5068`; crop of raw `cc6751f74fb68b1df41be5baecfad77d`),
`args_opsd.txt`. Nothing published; owner screening pending.
