# c01ae — journey-prompt probes: 3 fp8 clips + bf16 30 s OOM (2026-09-30)

LongLive-2.0-5B fp8 (LL_COMMIT 6b36d20, model_bf16.pt), 1280x704 @24 fps, W12/S8,
seed 0, RELATIVE_ROPE=on, dinov2 eval, on the 24 GB 4090, 12:51-13:08Z (974zzzzzz
replay batch). Probes walk-through-town "journey" wordings, compared against the
fast-prompt travel baseline; the fourth probe (bf16, 30 s) hit the card's ceiling.
All three completed runs eval `NO_FLAGS` (diagnostic horizon = full length).

## Numbers (median px/frame @640w)
| run | prompt | len | flow med (p10/p90) | halves | net px | rmse | bound/intra | peak VRAM | wall |
|---|---|---|---|---|---|---|---|---|---|
| c01ae_1 | journey | 30.5 s | 0.853 (0.53/1.34) | 0.823->0.912 | -2.8 | 17.05 | 0.96 | 22101 MiB | 335 s |
| c01ae_2 | journey2 | 10.5 s | 0.869 (0.54/1.42) | 0.813->0.900 | -0.5 | 16.08 | 0.93 | 24165 MiB | 206 s |
| c01ae_3 | nofollow | 10.5 s | 1.142 (0.74/1.82) | 1.158->1.142 | -2.1 | 15.93 | 0.86 | 24163 MiB | 213 s |
| c01ae_4 | journey, bf16 | 30 s | FAILED - TE OOM (see below) | - | - | - | - | - | 125 s |

Reference (fastprompt receipt, same card): fast s0 = 1.118 @10 s / 1.132 @30 s.
Journey wordings land sub-travel at both lengths (0.85-0.87); nofollow is
travel-class (flow >= 1.0, net ~ 0); halves flat (no decay) in all three.

## Contact first-glance (Zeph; independent pixels review required)
- c01ae_1 journey 30 s: persistent arcaded alley across all 9 samples; doorways,
  lamp posts and shopfronts progress; distant tower top-of-frame late (p7); no
  plaza/shop-street transition visible in samples; no walker; gentle motion.
- c01ae_2 journey2 10 s: walker in blue walking ahead at constant distance,
  buildings and doorways gliding past on both sides - follow behavior present,
  steady forward travel.
- c01ae_3 nofollow 10 s: walker in dark jacket ahead through arcaded alley with
  shuttered shopfronts passing; steady follow; landmark change every panel.

## bf16 boundary (c01ae_4, journey 30 s, bf16)
OOM in T5 text-encoder forward: tried 512 MiB; 23.12 GiB in use. fp8 30 s ran at
22101 MiB peak. Plan estimated 16.1 GiB of 23.6 - optimistic for bf16.
=> fp8 remains the >=30 s working point (c01v_1b bf16 failed in the same class).

## Provenance
- jobs gpuq_job_974zzzzzz_z1_c01ae_1..4: rc 0/0/0/1, 340/210/217/125 s, 12:51:17->13:08:12Z (logs_c01ae.txt)
- box runs: exp/pr63/runs_probe_journey30s_s0/len_30s; runs_probe_journey10s_s0/len_10s;
  runs_probe_nofollow10s_s0/len_10s (args_c01ae.txt: scripts, prompts, configs)
- sha256 (bit-for-bit vs box):
  c01ae_1 9c91937d044d094278cab166b35711ef12cbfecec1b39c94243d4aa3647c189c
  c01ae_2 da06efe4435d366d1f9cac51b5034fda1ec1a2fe8121c34f51286dfbb6233bc2
  c01ae_3 6e0a0ab4e22e147640b9c9d127a057d8fdd80e10a1a141303e0448a59ffb2cb7
- eval verdicts NO_FLAGS (contacts/*.eval.json); logs + bf16 crop in this dir
- local harvest: clip_harvest/{c01ae_1_journey30s_s0, c01ae_2_journey10s_s0,
  c01ae_3_nofollow10s_s0}/ (mp4 + preview.mp4 + contact_9.jpg + flow.json)
