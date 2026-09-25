# BrainSNN: inspectable streaming dynamics

Status: R8 product proposal and local demo specification. This document does not establish a deployed page, hosted API, trained capability beyond the attached run, or commercial readiness. Evidence comes from the experiment files in this directory; website deployment has not been checked.

## Positioning and pillars

**Pitch:** “Exploring long-horizon coherence at constant inference-state memory.”

Supporting line: “Inspect a synthetic rollout beside its reference, see where it drifts, and examine the measured cost of keeping it running.” Coherence is a target to demonstrate on declared tasks and horizons. Fixed recurrent state is an architectural property under a fixed configuration; it does not imply accurate predictions indefinitely.

BrainSNN remains an engine product. This experiment is a focused demonstration of temporal state and evaluation, not a replacement identity or a claim that synthetic video prediction delivers general agent intelligence. Initial audience hypotheses are researchers evaluating sequence models, developers of interactive synthetic environments, and engineers comparing inference costs. Demand and willingness to pay remain untested.

| Pillar | User value | Evidence required |
| --- | --- | --- |
| Persistent dynamics | Continue a scene from a short observed context and inspect how motion changes over time. | Open-loop rollout curves, declared horizon and seeds, failure examples, and a frozen-last-frame baseline. |
| Bounded inference state | Keep a declared state size as the generated horizon grows. | Actual inference path, tensor byte counts, steady-state process/device memory, fixed grid/batch/architecture, bounded output queues. |
| Inspectable evidence | Reproduce what the page shows and distinguish measurements from aspirations. | Checkpoint/config/source identity, generator settings, raw metrics, exact render command, and artifact hashes. |

## Current evidence and limits

The local scaffold in [wfvideo.py](wfvideo.py) compares wave, attention and SSM arms. [train_compare.py](train_compare.py) uses a shared predictor scaffold and positional embeddings, offers parameter matching, and evaluates synthetic clips. These are experiment-specific comparisons, not evidence of superiority to all transformers, SSMs or commercial video systems.

- Wave kernels include `separable` and `dispersion`. The recurrent `stream_step` path exists only for wave/dispersion. Its current implementation omits `gate` and `local_fuse`; checkpoints using either must use the fixed-window path or fail an explicit recurrent request. Never silently drop trained operations.
- Recurrent temporal positions clamp to the final learned context position after warm-up. Finite-FFT versus recurrent mixer calculations are approximate; [sanity_check.py](sanity_check.py) includes small-configuration parity checks at `1e-3`, not a proof of equivalence for every trained model or long rollout.
- Current attention and SSM video predictors are evaluated by repeatedly running the last `T` frames. Their context buffers are also bounded when `T` is fixed. This supports a distinction in state representation and recomputation, not “attention memory always grows.”
- [data.py](data.py) supplies synthetic balls. [data_waves.py](data_waves.py) supplies periodic wave/advection fields and a restricted drifting, viscous Taylor–Green vortex family. Its RGB channels encode the field, time derivative and Laplacian, with scales fixed from initial conditions; they are not natural-camera colour. The generator's physics does not certify the learned predictor as a physical simulator.
- The wave clip generator accepts grids 16, 32 or 64 and at most 64 transitions per call. The renderer supports longer references through `spectral_evolve`, starting from the decoded float32 time-zero seed with the original fixed channel scales. This preserves the initial conditions and time index with roundoff differences from the sampler; a checked prefix differed by at most `2.46e-7`. It does not concatenate independently seeded clips.
- Legacy result JSON can omit data-source/field provenance and rounds `ffn_mult`. Pair the exact checkpoint with its result/config and recover tensor dimensions where possible. Missing data-source/field values default to `balls`/`wave` with recorded notes; supply explicit `--data-source`/`--field` when those are not the known training settings. All missing-setting assumptions remain visible. Legacy v1 checkpoints such as `runs_box` are supported through a matching legacy result without v2 positional embeddings, or a config with `architecture: "v1"`; the current synthetic generator may differ from the historical training generator. Do not infer that an old checkpoint was trained on waves because the demo is named “wavefield.”

An example of why the page must show failure: [runs_v2_box/result_wave_w1.json](runs_v2_box/result_wave_w1.json) records next-frame MSE `0.00137` versus copy-last `0.00572`, but a 256-frame evaluation records identity survival `0.083` and divergence horizon `0`. This is a recorded ball experiment, not a new benchmark rerun. Its low next-frame error does not establish long-horizon coherence.

Those legacy rollout results use `m(win)`, not recurrent inference. Equal recorded GPU peaks at 32 and 256 frames do not establish recurrent streaming. Their `r_persist_over_model` denominator is model rollout MSE, while the persistence numerator uses the previous **ground-truth** frame at every step. Label it as a one-step reference with privileged observations; do not present it as a fair open-loop frozen-last-frame baseline. CPU `peak_gb: 0` means GPU memory was not measured, not zero total memory. The `--stream-test` path can run untrained weights and therefore measures execution cost independently of prediction quality.

## What “constant memory” means

Fix batch size `B`, spatial grid, layer count, width, number of modes, numeric types, and execution path. For each dispersion layer, the current complex64 recurrence stores `B × modes × heads × padded_height × padded_width × head_dim` elements. Recurrent tensor bytes are the sum of `numel × element_size` over the actual state tensors. The count does not depend on the generated horizon `N`. Model weights, the current frame and transient workspaces are additional costs.

| Scope | Expected dependence | How to report it |
| --- | --- | --- |
| Recurrent tensor state | Constant in `N`; scales with grid, batch, width, modes and layers. | Exact bytes, dtype, tensor shapes and configuration. |
| Fixed-window context | Constant in `N` at fixed `T`; scales with `T` and model configuration. | Context bytes and forward recomputation cost; do not label it recurrent. |
| Device/process peak | Includes weights, allocator/workspace and other resident data. | Device and measurement method; warm-up separately; null when unavailable. |
| Frames, PNGs, videos, full reference clips and per-frame logs | Grow with `N` unless discarded or streamed to a sink. | Artifact bytes, queue capacity, retention and any buffered reference bytes. |

“Constant inference-state memory” never means constant total storage, zero memory, constant training memory, constant latency, or indefinite retention. The current offline renderer materializes CPU reference frames and retains metric curves, so its host memory grows as `O(N)` even while recurrent state stays fixed. Profile the serving implementation separately from the offline exporter.

## Demo page plan

Keep the main homepage simple: one experiment card and a “Watch the dynamics demo” action. Proposed detail route: `/engine/wavefield`. API material belongs under an engine/developer route. These are planned routes, not claims about the live site.

Preserve BrainSNN's visual language: near-black `#05070b`, cyan `#68eaff`, violet `#947cff`, restrained glow, rounded panels, readable contrast and a system font stack. Use the existing brand rather than introducing a new visual identity.

1. Show the pitch with a persistent “Research demo · synthetic data” label. Explain that playback is a recorded rollout, not live inference.
2. Present synchronized **Prediction** and **Ground truth** videos with visible labels, shared play/pause, scrub, restart and speed controls. Show generated-frame count separately from seed/context length. Display a clear empty or missing-reference state; never substitute a decorative animation for evidence.
3. Place a compact metrics panel beside or below the players. Read values from the selected run, show units and use “Not measured” for unavailable fields. Keep loss curves and methodological caveats expandable.
4. Provide local file inputs for prediction video, optional reference video and `metrics.json`. The self-contained [demo_player.html](demo_player.html) includes its own CSS/JS and no CDN, analytics, external fonts or model runtime. Files stay in the browser; do not upload them. Local metrics are user-supplied evidence, not authenticated provenance.
5. Reveal dataset/field, seed, architecture, kernel, context `T`, generated `N`, grid, inference mode, checkpoint identity and run warnings before offering a comparison. Mismatched video durations or independently chosen files should produce an explicit warning, not a verified-match badge.
6. Offer the reproducibility bundle and limitations beneath the demo. A public launch should include a declared seed set, a typical example and a failure example; selection must not imply that a single attractive clip represents average performance.

Metrics should distinguish inference throughput from encoding/export time and playback FPS. Show per-frame MSE only when an aligned reference exists; calculate it on declared numeric tensors, not a lossy MP4. Use frozen-last-context-frame persistence for an open-loop baseline and state that baseline exactly. Balls may additionally use colour-matched centroid error, identity survival and divergence horizon; those metrics are inapplicable to PDE channels. For fields, add decoded-field error and phase/amplitude drift only after their definitions and normalization are validated. Optional GPU/process measurements must remain null when unavailable. A run label alone must never turn the pitch into a “coherence proven” badge.

Accessibility and empty states are part of the demo: keyboard-operable controls, visible focus, reduced-motion support, no autoplay, readable labels without colour dependence, responsive stacked players, and actionable invalid-file/codec messages. Loading another run clears stale metrics and playback state.

### Local producer-to-player handoff

[render_rollout.py](render_rollout.py) accepts the required CLI surface `--ckpt --kind --kernel-version --grid --frames --out`. Here `--frames N` is the number of **generated** frames, not the checkpoint's trained context length. The default `--mode auto` selects recurrent inference for supported wave/dispersion checkpoints with `gate=false` and `local_fuse=false`, otherwise windowed inference; `--mode recurrent` or `--mode windowed` selects an explicit path. `--config` supplies exact training configuration or matching result JSON. Additional arguments and supported input variants are described by `python3 render_rollout.py --help`.

```sh
python3 render_rollout.py \
  --ckpt smoke_v3/ckpt_wave_ck.pt \
  --kind wave --kernel-version dispersion --grid 8 \
  --frames 256 --side-by-side --out /tmp/brainsnn-rollout-r8
```

This is a smoke-checkpoint rendering example, not a recommended quality showcase or a waves-dataset claim. Use a new output directory. The renderer exports prediction PNGs, `prediction.mp4` and `metrics.json`; `--side-by-side` additionally saves ground-truth/comparison PNG directories and `ground_truth.mp4`/`comparison.mp4`. Dependencies are PyTorch and Pillow, plus ffmpeg or the fallback packages `imageio` and `imageio-ffmpeg`. Load the separate prediction and reference videos into the two player slots; the combined comparison video is a shareable export. A missing encoder is reported explicitly even if PNG export succeeds.

For a published artifact, preserve exact config and render command alongside the checkpoint, source revision, dependency versions, seed/generator settings and metrics definitions. Existing [proofpack.py](proofpack.py) and [verify_proofpack.py](verify_proofpack.py) can bundle completed run artifacts and verify file integrity when explicit configuration and reproduction commands are supplied. Manifest hashes establish byte integrity against the manifest; they do not prove authorship, non-cherry-picking, numerical reproducibility or scientific validity. Do not fabricate a missing training command from incomplete results.

## Proposed streaming API — not implemented

The API would expose an authenticated, bounded inference session for a versioned model. R8 only supplies the local producer/player/spec. No endpoint below is deployed by this work, and example capacities are design defaults to validate.

Use HTTPS for lifecycle operations and a same-service WSS connection for control and binary frames. A model registry must pin checkpoint SHA-256, config, input encoding, supported grids/context length and available inference paths. Initially restrict hosted models to reviewed wave/dispersion checkpoints with `gate=false` and `local_fuse=false`; a separate, explicitly named `fixed_window` mode can serve the other arms. Reject unsupported combinations rather than changing inference semantics.

| Operation | Proposed contract |
| --- | --- |
| `POST /v1/streams` | Authenticate and authorize model; validate limits; reserve state budget; create an idle session. Honour `Idempotency-Key` without allocating duplicate state. |
| `WSS /v1/streams/{id}/events` | Authenticate owner, ingest ordered context frames, generate under client credit, and emit frame/metric/control events. Never put long-lived credentials in URLs. |
| `GET /v1/streams/{id}` | Return lifecycle state, model/config identity, accepted context count, generated/acknowledged counts, bounded queue usage and expiry. |
| `DELETE /v1/streams/{id}` | Idempotently stop work, release state and clear buffered frames. Return a terminal receipt without retaining the media. |

Illustrative create request (a compatible 16×16, eight-context-frame model must actually be registered first):

```json
{
  "model": "wave-dispersion-reviewed-v1",
  "inference_mode": "recurrent",
  "grid": [16, 16],
  "batch": 1,
  "context_frames": 8,
  "max_generated_frames": 1024,
  "input_encoding": "rgb-f32le-0to1",
  "output_encoding": "rgb-u8",
  "max_inflight_frames": 4,
  "idle_ttl_seconds": 60,
  "recording": false
}
```

The response returns `stream_id`, relative socket path, model/config hashes, negotiated limits, exact recurrent-state bytes, context shape, protocol version and an absolute expiry. The model's grid and context length are fixed constraints, not free resizing controls. Version the protocol separately from model revisions; an existing stream never silently changes model.

Frame/control semantics:

- Each input binary frame carries a documented header with protocol version, frame index, shape and payload length; reject nonfinite or out-of-range float values and payload/shape mismatch. Context input indices are `0..T-1`, exactly once and in order. Context ingestion never counts as generated output. Server acknowledges accepted input indices.
- Warm-up ingests the `T` observed frames. Its final prediction is the first generated frame: output index `0`, absolute time index `T`. Emit that prediction before feeding it back. Output index `k` corresponds to absolute time index `T+k`; there is no skipped first prediction. Subsequent predictions use prior model output, with any clamping policy declared in session metadata.
- A `start` event grants initial credit up to the negotiated in-flight bound. Each output has monotonically increasing index, sample-time index, model/config identity and binary payload. An `ack` declares the highest contiguous output index consumed; only advancement of that index replenishes credit. Duplicate acknowledgements are idempotent; future or decreasing acknowledgements are protocol errors.
- Generation pauses when four output frames are outstanding in the example configuration. It also pauses when transport write buffers hit a configured byte limit. No hidden unbounded replay log or background generation is allowed. Client `pause` takes effect at a frame boundary; `resume` requires credit. `cancel`, idle TTL or quota completion releases resources.
- A dropped socket pauses immediately and retains only the bounded state/queue until expiry. An authenticated reconnect may replay only the unacknowledged frames still in that queue. If the state is gone, return `STATE_EXPIRED`; start a new session with context rather than pretending to resume. Durable snapshots/checkpoint downloads are out of scope for the first service.
- Each event states `recurrent_state_bytes`, queue frames/bytes, generated count and inference timing scope. Aggregate metrics can stream in fixed intervals with online summaries; full per-frame logs require an explicit external sink. Separate warm-up time, steady-state inference latency, encoding latency and network delivery. GPU allocated/reserved bytes and host RSS, when available, are separate fields.

Proposed errors include `INVALID_SHAPE`, `CONTEXT_COUNT_MISMATCH`, `OUT_OF_ORDER_INPUT`, `UNSUPPORTED_MODEL_CONFIG`, `MODEL_UNAVAILABLE`, `NUMERICAL_FAILURE`, `QUOTA_EXCEEDED`, `STATE_EXPIRED` and `SLOW_CONSUMER_TIMEOUT`. Use HTTP 400/401/403/404/409/429/503 for lifecycle errors as appropriate; after socket upgrade send a structured terminal error and close. Return retryability and any retry delay explicitly. Never replace failed model output with reference frames or a baseline without changing the visible run identity.

Privacy and resource policy: isolate sessions by authenticated owner; validate byte, shape, frame, concurrency and lifetime limits before allocation; require TLS; default to no recording and no model training from inputs. Raw context, state and unsent output expire on deletion/TTL. Logs should contain counters, latency, coarse errors and run identity rather than frame payloads or credentials. Any future recording option needs explicit consent, retention and deletion semantics. The local HTML has no server session and must retain its browser-only behaviour.

## Claim boundaries

| Acceptable wording now | Boundary or evidence required before stronger wording |
| --- | --- |
| “A research demo of synthetic autoregressive dynamics.” | Always identify balls versus encoded fields, checkpoint and inference mode. |
| “Designed to explore long-horizon coherence at constant inference-state memory.” | Coherence must be measured across held-out seeds/horizons; current artifacts include severe drift. |
| “This configuration has a fixed-size recurrent state.” | Count actual tensors; keep grid/batch/architecture fixed; disclose output/log/reference storage and measured process/device peaks. |
| “This run achieved X on metric Y.” | Attach definition, seed count, horizon, hardware/dtype and reproducible artifacts; do not turn one run into a general advantage. |
| “Ground truth from the specified synthetic generator.” | Generator equations and controlled conditions apply; learned outputs are not certified physically valid. |

Do not claim photorealism, production video generation, physical validity of predictions, general intelligence, autonomous learning, indefinite accuracy, universal model superiority, zero-cost inference, or real-time performance without an explicit measured device/batch/resolution target. A smooth-looking but stationary rollout can be a failure. Stable memory and stable dynamics are independent gates. Scientific simulator replacement, arbitrary uploaded-video support, customer ROI, pricing and availability remain outside the demonstrated scope.

## Evidence-gated milestones

| Milestone | Deliverable and acceptance gate |
| --- | --- |
| M0 — local R8 handoff | Only the three requested new source files. Known-checkpoint export completes; exactly `N` generated PNGs/video frames align with reference time `T..T+N-1`; recurrent versus windowed mode is explicit; metadata/shape conflicts fail clearly. Test ffmpeg and fallback/error behaviour. Player runs without network dependencies, handles empty/invalid files and displays missing metrics honestly. |
| M1 — qualified research demo | Freeze source/config/checkpoint/data provenance; rerun on at least 32 held-out seeds at generated horizons 32, 256 and 1024 where valid reference generation is supported. Declare failure/coherence criteria before evaluation. Publish median and spread, baseline curves, failures and source artifacts. Promote only the horizons that pass; a failed coherence gate still permits an explicitly labelled failure-analysis demo. |
| M2 — memory and path validation | Test supported trained configurations for first-prediction alignment and numerical parity tolerance. At fixed configuration measure recurrent tensor bytes and warm/steady device/process memory across horizons, with outputs drained to a bounded sink. Separately measure fixed-window arms and encoding. Record allocator noise and raw samples; do not use a plot's appearance as the acceptance criterion. |
| M3 — private streaming prototype | Implement the proposed session protocol with agreed limits. Verify bounded queues under slow/disconnected clients, cancellation/expiry cleanup, owner isolation, malformed payload rejection and numerical-failure reporting. Reproduce local first-frame/sequence semantics. No public SLA until load and failure testing exists. |
| M4 — website integration and usefulness test | Integrate the reviewed artifacts into the actual BrainSNN site as a dedicated demo route; verify mobile/desktop/accessibility and asset integrity. Run observed tests with 5–8 target users. Ship any stronger headline only if users understand the evidence and the corresponding scientific gate passes; use interviews to decide whether a hosted API merits further work. |

For M1, balls use predeclared divergence thresholds and identity-survival definitions; fields need a validated phase/amplitude or decoded-field criterion before “coherence” can pass. Match training budget, parameter count, context, seed sets and precision for comparison claims, and report deviations. A throughput result on one machine is labelled with that machine. Milestones are proposed gates, not dates or completed achievements.

## User-test questions and observations

1. After 20 seconds on the page, what do you think this system has demonstrated? Listen for mistaken photorealism, live inference, physics or intelligence claims.
2. Which video is generated, which is reference, and when does prediction start? Ask the user to scrub to the first generated frame and explain the context/generated counts.
3. At what point does this rollout stop being useful, and what change made you decide? Compare their visual threshold with the declared task metric.
4. What does “constant memory” mean here? Ask whether longer saved videos, higher resolution and more concurrent streams would increase resource use.
5. Does a still or slowly drifting output look convincing until you inspect the reference? Test whether the comparison and motion metrics expose copy-last collapse.
6. Which metric, missing condition or failure example would change your judgment? Observe whether users can find the seed, inference mode, normalization and hardware details.
7. Can you load a local run, synchronize playback and identify an unavailable metric without help? Record task completion, mistakes and whether an error message enables recovery.
8. What concrete task would you use this engine for, at which resolution/horizon, and what latency or drift is acceptable? Request a real workflow example rather than a generic expression of interest.
9. Would your integration need offline exports, a local process or a hosted stream? Explore cancellation, slow-consumer behaviour, privacy and reconnect needs before building an SDK.
10. What evidence would justify your next step: reproduce a run, test your own synthetic case, or integrate an API? Record the requested action; interest alone is not demand or willingness to pay.

Success for the first study is comprehension and an identifiable task, not applause for the animation. Record the participant's role, task, observed behaviour and objections; keep private notes separate from any public evidence bundle.
