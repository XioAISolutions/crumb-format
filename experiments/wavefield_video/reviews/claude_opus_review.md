I have everything I need. Let me note the key numbers from the smoke run, then write the review. The wave model actually did *worse* than attention on the smoke test (eval_mse 0.153 vs 0.092), which is important context — though that's a tiny 8×8/40-step run.

Now I'll give the review directly.

---

# Wave-Field Video Predictor — Expert Review

Read: `wfvideo.py`, `data.py`, `train_compare.py`, `kernels2d_smoke.py`, `run_all.sh`, `RESULTS.md`, and `smoke/*.json`. Below is correctness, scaling, architecture, and the one experiment that decides whether this lives or dies.

The core idea is sound and worth pursuing: replacing O(N²) attention with a **learned separable damped-cosine kernel convolved via FFT** is O(N log N), and the smoke test confirms the asymptotic win (128² grid: 3.7ms vs 1546ms). But there are correctness bugs that make the *current* comparison misleading, and a deeper architectural problem that will cap coherence length regardless of speed.

---

## 1) Correctness / bugs

### 🔴 BLOCKER — The model is non-causal in time, but the task is autoregressive
`WaveMix3D.kernels()` builds a symmetric kernel over `dt = arange(T) - T//2`, i.e. it mixes **past and future frames symmetrically**. During training that's fine (you predict frame T from frames 0..T-1, all context). But at rollout (`train_compare.py:71-75`) you slide a window and feed the model its own predictions. Because every layer's temporal convolution is **circular** (FFT conv wraps) and **bidirectional**, token `t=0` receives content from `t=T-1` via wraparound. The kernel has no notion of "the last frame is the query." Attention has the same non-causality here, so the head-to-head is at least *symmetric* — but for a system whose entire selling point is *minutes-long autoregressive rollout*, shipping a non-causal temporal operator is the central design flaw, not a detail. The `kernels2d_smoke.py` even demonstrates a causal mask (`kt_causal[dt>0]=0`) but `wfvideo.py` never uses it.

`★ Insight ─────────────────────────────────────`
Circular FFT convolution assumes the signal is periodic. For video, frame 0 and frame T-1 are *not* adjacent — wraparound injects the last frame's content into the first. To get linear (non-circular) convolution you must zero-pad to `T_kernel + T_signal - 1` before the FFT, then crop. Right now the "receptive field" silently includes the whole clip in both time directions.
`─────────────────────────────────────────────────`

### 🔴 BUG — Kernel spatial extent is hardcoded to `W`, breaks on non-square grids
`wfvideo.py:50`: `ds = arange(W) - W//2` is used for **both** the y and x kernel axes (`ks` reused at line 59 for `[...,:,None]` and `[...,None,:]`). If `H != W`, the y-kernel has length W but is broadcast against an H-length FFT axis. It happens to work only because every run uses `grid×grid`. Define `dt` from T, `dy` from H, `dx` from W separately.

### 🟠 BUG — `mix_dim()` is dead/fragile abstraction
`Block.__init__` calls `mix_dim(mix)` three times (`wfvideo.py:98-101`). For `WaveMix3D` it reads `pi.out_features`; for `AttnMix` there's no `pi`, so it falls to `po.in_features` = dim. Works, but it's an accident — `AttnMix.qkv` is `dim→3*dim` and `po` is `dim→dim`, so `po.in_features` is right by luck. One refactor away from a silent shape bug. Just pass `dim` explicitly into `Block`.

### 🟠 CORRECTNESS — bf16 autocast around FFT is a no-op / silent upcast
`forward` does `h = h.float()` (line 66) before the FFT, and the eval/train wrap everything in `autocast(bfloat16)`. `torch.fft` doesn't support bf16, so it runs in fp32 regardless — fine numerically, but it means the wave model's "bf16 speed advantage" is partly illusory: the FFT path is always fp32. The attention baseline genuinely runs SDPA in bf16. So the timing comparison is **not** apples-to-apples on dtype. Document or equalize this.

### 🟠 LOGIC — Positional information is weak/asymmetric between arms
- `AttnMix` gets a full learned `pos` embedding of shape `[1, N, dim]` — a huge, dedicated positional table (`wfvideo.py:82`).
- `WaveMix3D` gets positional structure *only* implicitly through the kernel geometry; the token embeddings themselves (`embed = Conv2d`) carry no absolute t/y/x position.

This asymmetry cuts against wave: attention can memorize position, wave cannot represent "which frame am I." For the moving-ball task that partly explains the smoke result (wave 0.153 vs attn 0.092 eval MSE). Add a factorized (t,y,x) positional embedding to the wave arm before declaring it inferior.

### 🟡 The smoke result is being over-read
`RESULTS.md` only reports the *timing* smoke, not the *training* smoke. The training smoke (`smoke/result_*.json`) shows wave **losing** at 8×8/40 steps. That's too small to conclude anything (2 layers, 85k params), but nothing in the repo flags that wave currently loses on quality. Be honest in RESULTS.md.

### 🟡 Minor
- `import math` unused in `wfvideo.py` and `data.py`.
- `st_s` label means steps/sec but reads like seconds; the first-step value (39.68) is compile/warmup-polluted.
- Boundary reflection in `data.py:31-34` can leave a ball exactly on the wall with a half-step; harmless but produces occasional velocity-sign jitter. Not a real bug.
- No fixed eval seed separation guarantee: train uses `seed=1000+step` (up to 1000+5000=6000), eval uses `90000+i`, drift uses `777`. Clean, no leakage. Good.

---

## 2) Memory & compute for single-4090 scaling

Current config: grid 32×32, T=17 → **N = 17,408 tokens**. This is where the architecture's value shows.

**Attention arm** is the binding constraint. SDPA materializes O(N²) work; even with flash-attention kernels the compute is 17k² ≈ 3×10⁸ score-pairs *per head per layer*. At grid 64 (N≈70k) attention is ~16× more expensive and memory-bound; at grid 128 (N≈280k) it's effectively dead on a 4090 for training batch sizes. This is exactly the regime your idea targets.

**Wave arm** memory reality check — it is **not** free:
- The activation `h` is reshaped to `[B, nh, T, H, W, dh]` and FFT'd. `rfftn` allocates a complex output ~half the spatial size but doubles bytes: roughly `B·nh·T·H·(W//2+1)·dh` complex64. At B=64, nh=8, T=17, 32², dh=48 that's ~1.6 GB *per layer* just for the frequency-domain activation, ×(input + kernel-broadcast + output) intermediates. FFT conv trades N² compute for a **3–4× activation-memory multiplier** vs a plain MLP-mixer.
- `torch.utils.checkpoint` (`--ckpt`) is correctly wired (`wfvideo.py:138`) and essential here.
- **The kernel FFT is recomputed every forward** (`wfvideo.py:67`). `self.kernels()` → `rfftn` runs each step even though it only depends on 6·nh parameters. Cache it during eval; during training it must track grads but you can still compute it once per forward instead of relying on it being cheap. Minor at this size, real at grid 128.

**Scaling verdict for "minutes of video":** 1 minute @ 8fps = 480 frames. You cannot hold N = 480·H·W tokens in a dense field on a 4090 — at 32² that's 491k tokens, at 64² it's 2M. The FFT is O(N log N) in *compute* but the **activation memory is O(N)** and will OOM long before attention's compute wall matters at these lengths. So the current dense-spacetime formulation does **not** actually reach minutes; it reaches maybe 2–4 seconds at 32² on a 4090. The path to minutes is not "bigger FFT" — see §3.

`★ Insight ─────────────────────────────────────`
The seductive trap of sub-quadratic mixers: O(N log N) *compute* still carries O(N) *memory* for activations. For long video the memory wall arrives first. Winning "minutes" requires a latent/compressed representation (fewer tokens per frame) or a recurrent state that doesn't grow with T — not just a cheaper full-sequence operator.
`─────────────────────────────────────────────────`

---

## 3) Strongest architecture upgrades for long-context video

Ranked by expected payoff:

1. **Move to a latent space (this is the big one).** Predicting raw pixels with a per-pixel token (`embed = Conv2d(3,dim,3)` at full 32² resolution) is why N explodes. Add a patch/VAE encoder (even a frozen 8× spatial downsample, patch 4×4 → 8×8 latent grid = 64 tokens/frame). Now 480 frames × 64 = 30k tokens is tractable, and the wave FFT gets *longer temporal context* for the same budget. Every frontier long-video system operates in latent space; a raw-pixel field will never reach minutes.

2. **Make time causal + streaming.** Replace the symmetric circular temporal conv with (a) a causal masked kernel (zero `dt>0`), and (b) **linear (zero-padded) convolution** so there's no wraparound. Better: reformulate the damped-cosine temporal kernel as a **linear recurrence**. A damped cosine `e^{-a t}cos(ωt)` is exactly the impulse response of a 2nd-order linear system → it has an O(1)-state recurrent form (this is the SSM/S4 insight). That gives you **constant memory per step at inference** and truly unbounded rollout — the actual route to "minutes." Keep FFT for parallel training, use the recurrence for generation. This is the single most important upgrade and it's *natural* to your kernel family.

3. **Keep spatial as FFT, make temporal an SSM.** Factorize: spatial mixing via the 2D damped-cosine FFT (cheap, N log N, fine to be bidirectional within a frame), temporal via a diagonal-plus-oscillatory SSM. This is essentially a physically-motivated S4ND/S5 and aligns perfectly with the "wave-field" framing (dispersion relation ω(k), damping a(k)).

4. **Learn frequency-dependent damping.** Right now `a_t, w_t` are per-head scalars (6 params/head). The whole expressive power is 6·nh numbers per layer — that's *extremely* parameter-starved vs attention's QKV. Let damping/frequency be a small MLP over head/channel, or make the kernel a sum of K damped cosines (K=4–8 modes) per head. This is where wave will close the quality gap you see in the smoke run.

5. **Add explicit positional conditioning to the wave arm** (factorized t/y/x embeddings) so the fight is fair.

6. **Content-adaptive kernels (the honest weakness).** Attention's power is *input-dependent* routing. A fixed convolution kernel cannot track a ball whose motion depends on the content. Consider gating kernel parameters on a pooled summary of the input (a cheap "hypernetwork" producing `w_t, a_t` per clip). Without some input-dependence, wave will systematically lose on anything with unpredictable motion — which is most real video.

---

## 4) The sharpest next experiment (prove or kill)

**One experiment, designed to falsify the thesis, not flatter it.**

> **Claim under test:** *A wave-field mixer sustains rollout coherence longer than attention at equal quality, as context length grows.*

The current setup can't test this — it trains at T=17 and rolls out 16 steps into a task (bouncing balls) whose *ground truth is deterministic physics the fixed kernel can perfectly encode*. That's rigged in wave's favor on dynamics but rigged against it on adaptivity, and the horizon is too short.

Concrete design:

1. **Fix the two blocker bugs first** (causal linear-conv time, per-axis H/W kernels) and add positional embeddings to wave. Otherwise any result is confounded.
2. **Task with a coherence failure mode:** balls that *occlude and swap depth*, or 2–3 balls with **stochastic velocity kicks** every ~20 frames. Determinism must be partly broken so memorizing one kernel isn't enough.
3. **The measurement that matters — divergence horizon, not next-frame MSE.** Train at T=17. Roll out **256 steps** autoregressively. Plot per-step MSE (and a structural metric — count of correctly-localized ball centroids) for wave vs attention vs a strong linear baseline (a plain SSM/Mamba block, and a ConvLSTM). Report the step index where MSE crosses a fixed threshold. *Minutes of coherence = this curve staying flat.*
4. **Hold quality equal, vary N.** Grids {16, 32, 64}. The thesis lives only if wave's divergence horizon degrades *slower* with N and its wall-clock/memory stays sub-quadratic while attention OOMs. If attention drifts the same or better at N where it still fits, the idea is dead for this regime.
5. **Include the SSM baseline.** This is the real competitor, not full attention. Full attention is a strawman for long context (nobody uses dense attention for minutes). If a vanilla Mamba matches wave's horizon at equal cost, the "wave-field" framing adds nothing. If the physically-structured oscillatory kernel *beats* generic SSM on periodic/wave-like motion, **that** is the publishable result and the reason to keep going.

**Kill criterion:** if, after the bug fixes and a fair positional setup, wave's 256-step divergence horizon is not meaningfully longer than a same-budget SSM baseline on stochastic-motion video, the specific "damped-cosine wave" inductive bias isn't earning its keep — fold back to SSM.

**Prove criterion:** wave holds centroid-accuracy past the horizon where SSM and attention have both diverged, *and* the recurrent (SSM-form) inference runs at constant memory for 480+ steps on the 4090. That combination — long coherence + constant-memory streaming — is exactly the frontier gap (Seedance/Kling/Veo's 10–15s cap) and would justify the whole direction.

---

### Bottom line
The compute asymptotics are real and the smoke test proves them. But (a) the temporal operator is non-causal and circular, which is disqualifying for autoregressive video; (b) the real wall for "minutes" is **activation memory**, not attention's compute, so the dense-pixel formulation can't get there; and (c) the fixed 6-param-per-head kernel is too weak and too content-independent to beat attention on quality yet — the smoke run already shows it losing. The high-value pivot is **latent-space tokens + a causal SSM reformulation of the damped-cosine kernel** (constant-memory streaming rollout), benchmarked against a Mamba baseline on stochastic-motion video by **divergence horizon**, not next-frame MSE. Do that and you either have a frontier-relevant result or a clean kill.

Want me to implement the two blocker fixes (causal linear-conv time axis + per-axis H/W kernels + factorized positional embeddings) so the next comparison is actually valid? Those are ~20 lines and the design choice for the causal kernel (mask-and-pad vs true recurrence) is exactly the kind of decision worth your call.
