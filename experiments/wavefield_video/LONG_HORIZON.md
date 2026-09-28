# LONG_HORIZON — what stands between this repo and 2–5 minute video

_2026-09-28 · branch `claude/crumbllm-video-optimization-w52xup` (off `crumb-llm-standalone`)_

Target: 2–5 min = **2,880–7,200 frames at 24 fps**. The pitch rests on three
receipts; here is what each one actually measured, then the four blockers found
in the code, what was fixed, and the pre-registered next runs.

## 1. What the receipts measure

| Receipt | Where | What it is | What it is not |
|---|---|---|---|
| 417× (3.7 ms vs 1,546 ms @ N=16k) | `kernels2d_smoke.py`, OCEAN.md | one **un-warmed, single-shot** CPU call: FFT conv vs `QKᵀ` then `(QKᵀ)V` with **no softmax**, one 64-channel head, fp32. Re-measured fairly (§5) the gap is ~886×, so the ratio holds on CPU | a model step; a GPU number |
| 1,024 frames at flat 0.207 GB | `train_compare.py --stream-test` | streaming **cost** — the docstring says "Weights need not be trained — this measures streaming cost, not accuracy" | coherent video; with the zero-init residual head an untrained stream is an exact copy of the last frame |
| O(N log N) | FFT mixing | true asymptotically | free: `linear_pad` doubles T, H and W (8× the field) in complex64 — at grid 6, T=256 a CPU training step is ~16 s |

Re-measured fairly in `bench_scaling_honest.py` (warm-up, median of reps,
softmax, and PyTorch SDPA): see §5.

## 2. Blocker A — the ripple forgets in under one frame

`state = fade × state + new_frame` carries the past only while `fade^t` is not ~0.
The v2 pole is `|λ| = exp(-softplus(a0 + a1|k|))` with `a0 = 0.5` →
**|λ| ≈ 0.38, half-life ≈ 0.71 frames** at init.

`python long_horizon.py memory` (dim 128, 4 layers, 8 heads, 3 modes):

| poles | half-life min/median/max (frames) | poles holding ≥1% after 1 s / 10 s / 2 min / 5 min |
|---|---|---|
| softplus (v2 default) | 0.71 / 0.71 / 0.71 | 0/24 · 0/24 · 0/24 · 0/24 |
| **halflife** (new) | 2.3 / 77 / 3,494 | 22/24 · 15/24 · 7/24 · 4/24 |

Training can move `a0`, but reaching a 2,000-frame half-life means driving it to
≈ −8 through a softplus whose slope there is ~3e-4: thousands of steps of
consistently signed gradient (see Blocker B for why that gradient does not exist).

**Fix (`--pole-param halflife`)**: `hl = hl_min·(hl_max/hl_min)^σ(raw)`,
`α_DC = ln2/hl`, plus viscosity `softplus(visc)·|k|` so fine detail still damps
faster than layout. Half-lives are initialized log-uniformly over
`[hl_min, hl_max]` = [2, 4096] frames, and the input is normalized by
`sqrt(1-|λ|²)` (LRU-style) so near-unit poles don't blow up the residual
stream. Default stays `softplus`, so every existing run and test is unchanged.

## 3. Blocker B — gradients never see more than ~0.7 s

Nearly every run trains with `--frames 16/17`. A dependency longer than the
training clip gets no gradient, however long the pole. The occlusion suite
(`run_occlusion.sh`, the "decisive" R14/R15 experiment) makes this concrete:
training clips are frames 0–17 and always start at frame 0, but the target is
hidden on [64, 320). **No arm ever sees the occluder during training**, yet all
arms are scored on tracking through a 256-frame gap. Its wave-vs-SSM kill
criterion cannot come out on architecture.

This is where O(N log N) actually pays: *training* on long clips in parallel.
Streaming at constant memory is common to every recurrent or sliding-window
model.

**Fix**: `--train-occ-start/--train-occ-end` put the gap at `[T−gap, T)` of each
training clip, so the loss target (frame T) is the first frame after the gap
and the loss *requires* memory. `run_long_horizon.sh` trains wave/SSM at T=128
and attention at the T it can afford (32), then evaluates on a post-context gap.

## 4. Blocker C — two correctness bugs (fixed)

1. **Future leakage in the FFT forward for long poles.** `_transfer` is the
   DTFT of the infinite impulse response sampled at L=2T points: a *circular*
   kernel. Frame s>t leaks into output t with weight `λ^(2T−(s−t))`. With
   |λ|=0.38 that is ~1e-7, but for minute-scale poles it is O(1): the model could
   read frames it is asked to predict. The halflife path now uses the exact
   truncated response `(1−λ^T e^{−iωT})/(1−λe^{−iω})`, and
   `test_halflife_forward_is_causal` guards it. The softplus default keeps the v2
   math (leak ≤ |λ|^(T+1)); any run where `a0` trained strongly negative had the
   same channel open.
2. **Hybrid streaming warm-up (`FusedMix.step`)** attended to zero-filled
   window slots during the first T−1 frames, so `stream_step` ≠ `forward` for
   `local_wave` and `local_ssm` (max err 0.014–0.04). `test_fusion_r14.py` passed
   only because the zero-init residual head makes both sides "copy last frame".
   Now it attends only to filled slots (current-frame queries only, which is
   also T× fewer queries). The test randomizes the head, and the error is ~1e-6.

## 5. Evidence from this change (CPU, 4 threads)

### Delayed recall (`recall_probe.py`)

A blob on frame 0, black frames 1…D−1, target = frame 0 again. The only route
is the wave path across D frames. Grid 6, dim 32, 2 layers, 300 steps,
batch 16. recall = 1 − MSE/MSE(forget); chance argmax hit = 1/36 = 0.028.
Pre-registered: "holds D" iff mean recall ≥ 0.5.

| delay D | steps | seeds | softplus recall / argmax hit | halflife recall / argmax hit | verdict |
|---|---|---|---|---|---|
| 16 | 300 | 0, 1 | 0.117 / 0.039 | **0.662 / 1.000** | halflife HOLDS, softplus forgets |
| 64 | 300 | 0, 1 | 0.116 / 0.037 | 0.153 / 0.057 | **both forget** (halflife fails the bar) |
| 128 | 300 | 0 | 0.118 / 0.027 | 0.116 / 0.027 | both forget (budget-limited, see next row) |
| 64 | **1000** | 0 | 0.119 / 0.039 | **0.669 / 1.000** | **halflife HOLDS**, softplus forgets |

Softplus sits at ~0.12 at every delay and budget, which is what predicting
the *average* blob scores; its argmax hit is at chance. It never learns to
remember. Half-life holds at D=16 in 300 steps and at D=64 in 1,000 steps: the
300-step D=64 miss was training budget, not the operator. (An earlier guess,
that viscosity was erasing position detail, is contradicted by this row.)
Longer delays need proportionally more steps. D=128 at 1,000+ steps is the
next CPU row, and the GPU suite in §7 is the real test.

### Honest scaling (`bench_scaling_honest.py`)

One 64-channel head, fp32, CPU (4 threads), warm-up 2, median of 5.
Raw JSON for every table here: `results_long_horizon/`.

| N | wave FFT | orig op (no softmax) | softmax attention | fused SDPA | SDPA ÷ wave |
|---|---|---|---|---|---|
| 1k | 0.18 ms | 2.9 ms | 3.0 ms | 3.1 ms | 17× |
| 4k | 0.26 ms | 37 ms | 52 ms | 49 ms | 189× |
| 16k | 0.97 ms | 552 ms | 837 ms | 858 ms | **886×** |
| 64k | 26.7 ms | skipped (N×N = 17 GB) | skipped | skipped on CPU | — |

So the 417× was *conservative* for this op on CPU: done fairly, the gap at 16k
is ~886×. The caveat is scope, not size: it is one head on CPU. A 4090 runs
the same 16k attention in about a millisecond (estimate), and a model pays it
per head, per layer, per denoising step.

### 5-minute stream, untrained (`long_horizon.py stream --stream-frames 7200`)

Both pole types: state constant at 24.0 MB (batch 2, grid 16), ~50 fps on
CPU. `HealthMonitor` flags **freeze at frame 0**, as it should: an untrained
residual model is copy-last. This is the 1,024-frame receipt extended to 7,200
frames, and it shows why a cost receipt says nothing about coherence.

## 6. What 2–5 minutes costs with a real video VAE (`long_horizon.py budget`)

480×848, 24 fps, dim 512, 12 layers, 3 modes:

| clip | VAE | latent steps | tokens/step | tokens | wave state | full-context KV (bf16) |
|---|---|---|---|---|---|---|
| 2 min | Wan2.1 (4×8×8, patch 2) | 720 | 1,590 | 1.14 M | 74.5 MB | 2.2 GB |
| 2 min | LTX (8×32×32) | 360 | 405 | 146 k | 19.0 MB | 285 MB |
| 5 min | Wan2.1 | 1,800 | 1,590 | 2.86 M | 74.5 MB | 5.5 GB |
| 5 min | LTX | **900** | 405 | 365 k | 19.0 MB | 712 MB |

Two things follow from this table:
- **Memory is not the wall on a 4090.** With an LTX-class VAE, even a
  full-context 5-minute KV cache is under 1 GB. The constant-state advantage is
  real but small in bytes. The N² *compute* over 365 k tokens is the real cost
  (N²/2 pairs × 4·dim × 12 layers ≈ 1.6e15 FLOPs per full causal pass, i.e.
  ~10–20 s on a 4090 at ~80–165 dense TFLOPS, multiplied by denoising steps; an
  estimate, not a measurement). Shipping long-video systems avoid that
  with windowed context, which is also linear.
- **5 minutes through LTX is 900 latent steps**, which is inside the 1,024-step
  horizon already streamed. The step that is missing is a latent space that does
  not collapse: the home-made 16× AE "fades to black" in both arms. Use a
  pretrained video VAE rather than training a new one.

## 7. Next runs, in order, with kill criteria

1. **`run_long_horizon.sh` on the box** (preflight `STEPS=20 SEEDS=0` first; not
   yet run on GPU). PROVE: a halflife arm reaches exit-direction accuracy ≥ 0.8
   and beats softplus and T=32 attention by ≥ 0.2 on ≥ 2/3 seeds → grow the gap
   to 256, then 1,024. KILL: halflife ≤ softplus on ≥ 2/3 seeds → long poles are
   not the limit; stop the pole line.
2. **Re-scope `run_occlusion.sh`**: with default flags its emergence metrics
   measure the training window, not the architecture. Pass `--train-occ-*` or
   stop citing it as decisive.
3. **Latent track on a pretrained VAE** (LTX or Wan2.1, frozen): same
   next-latent objective, same `HealthMonitor` on decoded frames. KILL:
   `fade`/`flatten` fires before 256 latent steps on held-out clips.
4. **Every long render goes through `StreamSession` + `HealthMonitor`.** Report
   the frame of the first collapse flag as the headline number, alongside
   divergence horizon. A stream receipt without trained weights and without
   collapse flags is not a coherence receipt.

## 8. Phase 1 — train it like a language model

Four more limits found in the harness, each capping any result below minutes:

| limit | where | fix (opt-in; defaults byte-identical, verified against `main`) |
|---|---|---|
| every clip starts from a zero state; nothing carries between clips | `WaveMix3D._wave_dispersion` | `VideoPredictor.forward(frames, states=...)` carries each block's recurrent state across T-frame chunks (`_wave_dispersion_stateful`, `SSMLite.forward_stateful`) |
| one supervised frame per T-frame forward | `train_compare.py` training loop | `forward(..., dense=True)`: next-frame targets at all T positions |
| length-T temporal table; `stream_step` clamps it at `pt[T-1]` | `FactorizedPosEmb` | `time_pos="none"`: the recurrence carries order, so train/chunk/stream see identical inputs |
| `--gate` pools over all T frames, so "causal + gate" models read the future (measured leak 0.055) | `_apply_gate` | prefix mean over frames ≤ t when `causal_time` |

Chunked == one long forward == `stream_step` to ~1e-6 (wave softplus, wave
halflife, SSM; `tests/test_stateful.py`). Walking a 128-frame context as 4×32
chunks with full gradient costs 1.15 s/step on CPU vs 1.87 s/step as one
window. With the graph kept only for the last chunk (G=1) it is 0.74 s/step,
and memory is that of one chunk however long the sequence is.

Side finding: the softplus default's circular FFT kernel leaks future frames
into the past by |λ|^(2T−d). Measured at init: 0.7% of a frame's own effect at
T=4, 0.01% at T=8, <1e-6 at T≥16. Runs with `--frames 4–8` carried a small
look-ahead. The stateful path always uses the exact truncated kernel.

### 8.1 Pre-registered: can constant-memory training learn memory past its chunk?

Delayed recall (§5), D=128, chunk 32 (so the blob is 4 chunks back), grid 6,
batch 16, 1,000 steps, seed 0. Arms: {softplus, halflife} × {G=1, G=4}.
G=4 is full backprop through all four chunks (mathematically the single
128-frame window); G=1 keeps a graph for the last chunk only, so the frame-0
write gets no gradient from the frame-128 loss.

- **PROVE (constant memory suffices):** halflife G=1 recall ≥ 0.5.
- **If only halflife G=4 holds:** gradient through the carried state is required
  for writes; training memory grows with G, which becomes the runner's knob.
- **KILL (poles are not enough at this length):** halflife G=4 < 0.5, so D=128
  needs more than 1,000 steps or a different write path.

**Result: KILL at this budget.** Recall / argmax hit (chance ≈ 0.028):

| arm | recall | argmax hit | s/step (CPU, shared) |
|---|---|---|---|
| halflife G=1 | 0.119 | 0.039 | 1.14 |
| halflife G=4 | 0.118 | 0.035 | 0.73 |
| softplus G=1 | 0.120 | 0.031 | 1.32 |
| softplus G=4 | 0.120 | 0.031 | 1.28 |

Every arm sits at the mean-blob floor. Halflife G=4 is full backprop across all
four chunks, mathematically the single 128-frame window, and it fails too. So
at 1,000 steps D=128 is out of reach regardless of carry; D=64 already needed
1,000 steps (§5). Constant-memory training (G=1) is **neither proven nor
disproven** by this run. §8.3 raises the budget.

### 8.2 Pre-registered: dense supervision

Balls, grid 16, T=16, 300 steps, seeds 0/1, `--no-dense` vs `--dense`, equal
steps. PROVE: dense lowers eval MSE / copy-last by ≥ 10% on both seeds;
otherwise it stays opt-in with the numbers reported.

**Result: below the bar, so dense stays opt-in** (`train_long.py --dense`).

| seed | last-only MSE/copy-last | dense MSE/copy-last | Δ | copy-ratio last-only → dense |
|---|---|---|---|---|
| 0 | 1.008 | 0.975 | −3.3% | 0.093 → 0.220 |
| 1 | 1.004 | 0.961 | −4.3% | 0.075 → 0.198 |

Dense is better on both seeds and roughly halves the freeze tendency, and it is
the only arm that beats copy-last. It still misses the pre-registered 10%. All
arms have divergence horizon 1 at 300 steps (under-trained). The SEQ runner
arms pass `--dense` explicitly for the reason above.

### 8.3 Pre-registered follow-up: the same recall at 3,000 steps

D=128, chunk 32, grid 6, batch 16, seed 0, **3,000 steps**; halflife G=4, then
halflife G=1.

- **PROVE (constant memory suffices):** G=1 recall ≥ 0.5.
- **If only G=4 ≥ 0.5:** writes need gradient through the carried state;
  training memory grows with G.
- **If both < 0.5:** at this scale the write path, not the step budget, is the
  limit (next: a learned input gate on `Bin`).

**Result: not run.** The 3,000-step run died on a container restart. It was then
superseded by 8.4, which found that the write path, not the step budget, is the likely limit.

### 8.4 Diagnosis: the memory is there, but drowned (measured before 8.3 ran)

Why does even full backprop fail at D=128? Blob signal vs background in the
first layer's wave state at init (halflife, grid 6; RMS of the state difference
blob-vs-blank, over the RMS of the blank-clip state):

| D | signal/background, as built | no spatial table + no embed bias | + no `pi` bias ("clean write") |
|---|---|---|---|
| 16 | 2.6e-2 | 2.4e-1 | background **exactly 0** |
| 128 | 7.5e-4 | 4.5e-3 | background **exactly 0** |

Every frame, even a blank one, writes the same per-cell constant into the
state: the spatial position embedding plus the embed and input-projection
biases. A shared input projection cannot cancel a per-cell constant, and
long poles integrate it. By D=128 the frame-0 memory is under a background
about 1,000× larger. Real video has the same problem in a stronger form: a
static background is a per-cell constant, and long ripples would integrate
the still scene instead of the events.

Two opt-in fixes (both keep chunk == full == stream exactly):
- `clean_write=True`: no spatial table, no embed bias, no `pi` bias, so a blank
  input writes zero. Structural, but it cannot help with real static content.
- `write_gate=True`: a learned per-token, per-head sigmoid on the wave input.
  The model can learn to write events and skip static content. It is applied
  before the time-invariant recurrence, so the FFT path is unchanged. This is
  the one that could transfer to video.

**Pre-registered (same budget as 8.1: D=128, chunk 32, 1,000 steps, seed 0,
halflife):** arms clean G=4, clean G=1, gate G=4, gate G=1.
- **PROVE (the background was the blocker):** any arm reaches recall ≥ 0.5,
  where 8.1's identical-budget arms sat at 0.118.
- **Constant memory:** a G=1 arm ≥ 0.5 means carried-state training works
  once the write is clean.
- **KILL:** all four arms < 0.5, so the background was not the (only) blocker
  at this budget.

**Result: pending.** Running on CPU (clean G4/G1, gate G4/G1) when this section
merged. The verdict is recorded here either way, in the next PR.

## 9. Phase 2 — a pretrained video VAE under the wave model

The home-made per-frame ConvAE fades to black in both arms and compresses no
time. Production video VAEs compress time as well, so the predictor runs on
latent steps: 5 minutes at 24 fps is 7,193 usable frames (1+8k) → **900 LTX
latent steps**, inside the 1,024-step horizon already streamed.

| piece | file | what it does |
|---|---|---|
| adapter | `video_vae.py` | `VideoVAE("ltx"\|"wan"\|"conv"\|"*-tiny")`: `encode` [B,T,3,H,W]→[B,Tl,C,h,w] (per-channel normalized), `decode`, strides, save/load; `LatentShards` windows over encoded shards with a held-out split |
| offline encode | `encode_videos.py` | mp4 folder → resize/crop → resample fps → 1+t·k segments → `shard_*.pt` + `index.json`; `--synthetic N` writes moving-ball mp4s for smoke |
| training | `train_long.py --latents DIR` | phase 1 carried-state trainer on latent sequences (`VideoPredictor(in_ch=C)`); eval = held-out next-latent MSE vs copy-last + an autoregressive latent rollout curve |
| 5-minute screen | `long_horizon.py stream --latents DIR --vae ltx` | warm on real held-out latents, stream N latent steps, decode chunk by chunk, `HealthMonitor` on **decoded pixels** |
| box runner | `run_latent.sh`, `queue_jobs/latent_2026-09-28/` | encode → train → decoded 900-step stream, sliced/resumable like `run_long_horizon.sh` |

Verified here with tiny random LTX/Wan configs: shapes and strides (LTX 17→3
steps, 32× spatial; Wan 17→5, 8×), save/load round trip, mp4 → shards →
training → decoded stream (`tests/test_video_vae.py`), and a full
`run_latent.sh` pass. **Not verified here:** anything about quality. Hugging
Face is blocked in this container, so real LTX/Wan weights have only run on
the box.

Known limits: each chunk of generated latents is decoded with the previous
latent prepended, and its duplicated first frame is dropped. That keeps the
timeline continuous (N latents give stride·N frames), but the VAE's temporal
context still restarts at every seam, so use a large `--chunk` for the screen.
Collapse positions are reported both in decoded frames (`collapse`) and in
generated latent steps (`collapse_latent_step`). The KILL rule reads the
latter. Shards are loaded fully into memory, which is fine
for hours of 256×448 latents (≈0.1 MB per latent step at fp16) but not for
datasets far larger than that.

Pre-registered box read (`run_latent.sh`): **KILL** if fade or flatten fires
before latent step 256 (~34 s of LTX video) on the held-out stream. **PASS**
if nothing fires through 900 steps (5 minutes); then judge the decoded frames
by eye before any claim.

## Pitch corrections

- "At 16k moments one step takes ~1.5 s": the *ratio* survives a fair re-measure
  (~886× on CPU), but the absolute number is one 64-channel head on CPU, not a
  model step. On a 4090 the same op is ~7e10 FLOPs, about a millisecond with
  fused attention (estimate). Say "one attention op, CPU", not "one step".
- "We streamed 1,024 frames at a flat 0.2 GB": true, with **untrained** weights.
  It shows constant memory, not minutes of coherent video.
- "Veo, Kling, Seedance all pay the N² bill": within a shot, yes. Minute-length
  systems bound cost with windowed or chunked context, which is also linear in
  length. The differentiator worth testing is §3: learning dependencies longer
  than any affordable attention window, because long training clips are cheap.
