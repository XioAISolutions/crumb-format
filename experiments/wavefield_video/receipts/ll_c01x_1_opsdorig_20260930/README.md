# OPSD-v orig arm — c01x_1 first clip (LongLive **orig** config: base + base-LoRA, no OPSD); rc=0 (2026-09-30)

**Job.** `gpuq_job_974zzzzzz_c01x_1_opsdorig.sh` — RUNNING 02:02:41Z → FINISHED **rc=0** 02:05:56Z (dur 194 s; 7th attempt).
`cd /root/opsd-v` → `timeout 5400 venvs/opsd/bin/python inference.py` + args from `opsd_args_orig.txt`
(`configs/inference_longlive_original.yaml … --num_output_frames 159 --seed_list 1 --use_lmdb_pipeline --lmdb_use_relative_sink`).

**Output.** `opsd_out_orig/000000_A_brisk_forward_walk_down_a_narrow_cobblestone_street_in_an_old_European_town_br.mp4`
— 633 frames, 832×480, 16 fps, 39.56 s, 8,881,426 B, sha256 `4fe43cc2a9e8836e1264af706c75bbf5a1e20e9cb1a5fc3468c4335cc8890c8f`.
`grep -icE "traceback|out of memory|error"` on the log = **0**.

**Unblock history (env chain, 6 failed attempts 01:21–01:57Z).** FA2 assert → wheel into `venvs/opsd`; VAE-decode OOM →
memory diet in `causal_inference_lmdb.py` (generator→CPU before decode, `patch_genswap.py`; decoded video→CPU before
clamp, `patch_vidcpu.py`; backups `.before_genswap`/`.before_vidcpu`); final die-point `write_video` → **PyAV missing**
→ `av 17.1.0` installed 01:58Z. Attempt 7 clean — chain closed.

**Progression (flow_check; native 16 fps — not directly comparable to the 24 fps LongLive runs).**
Median **1.74 px/frame** (p10 0.729 / p90 2.994); net dx +28.8 px; rmse 26.2; `bound8/intra8` 1.04.

**Pair.** c01x_2 (OPSD-LoRA arm) re-fired 02:06:36Z with the full diet; `opsd_scorecard.sh` compares the pair when it lands.

**Files.** `contact_9.jpg` (md5 `7a37c7544c47beb46ed3ae65fc4683a4`), `flow_c01x_1_orig.json` (md5 `4d5be00fffd8dfd9c8fdbfba065d473e`),
`log_c01x_1.crop.txt` (md5 `97e2c712fa5bc38197e341ad47e8c5df`; crop of raw md5 `f1d9722c9afdd5f6510e804f751f2326`),
`args_orig.txt` (md5 `2652ca46ac82aa9e6dd52508a894b1b3`). Nothing published; owner screening pending.
