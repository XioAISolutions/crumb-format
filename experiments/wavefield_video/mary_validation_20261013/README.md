# Mary validation gates

This folder prepares validation only. It changes no existing engine, defaults,
checkpoints, VAE weights, or long-video runner. A bounded offline **CPU encoder
contract** has now run against the existing publisher-pinned LTX cache; no GPU
or learned training campaign has run. Native 720p/1080p owned footage and independent blind pixel review must come
before further engine work. Stable health statistics cannot replace that review.

The receipt at `receipts/20261013-pretrained-ltx-cpu-encoder/receipt.json` verifies
the cached 1,676,798,532-byte weights against the publisher's LFS SHA256 and the
config against its git-blob identity. All 190 loaded state keys/shapes match.
The frozen LTX encoder passed nine prefix/poison comparisons on synthetic
297-frame **32×32** RGB at raw cuts 33/129/289: max past difference
1.3709e-6, with the 289-frame prefix difference exactly zero. The worker took
11.017 seconds with peak sampled RSS about 2.66 GiB under 180-second/8-GiB caps.
This is a narrow actual-pretrained CPU encoder result. **Pretrained decoder
causality/RF/seams, frozen DINOv2 features, native-resolution footage quality,
and the three-seed learned comparison have not passed.** Transformers is absent
in this interpreter; no feature fallback or dependency installation occurred.
`run_cached_ltx_cpu.py` retains publisher-pin and failure/cap evidence without
copying weights. The broader production preflight remains incomplete.

`validation.py` requires an independently pinned local publisher-snapshot
manifest before loading a production `wan` or `ltx` VideoVAE. It rejects `conv`,
`*-tiny`, renamed tiny channel configurations, missing or changed artifacts,
trainable weights and training mode. An approved manifest pins `backend`,
`repository`, immutable 40-hex `revision`,
`source_kind: official_huggingface_snapshot`, and a `files` mapping containing
exact SHA256 values for every `config.json`, weight and shard-index file in the
component directory. The expected repositories come from the existing native
presets. Hashes establish identity/integrity; the independent approved manifest
is the training-provenance trust anchor. A self-written hash is not evidence of
pretraining. No weights are downloaded or installed by this folder.

`probe_raw_future_causality` compares prefix-only encoding with that prefix
inside the full clip, then poisons **raw** suffix pixels with zeros, inversion,
and seeded noise. It checks every past latent at valid raw cut points. Latent
poisoning alone cannot detect an encoder that pooled the future during
preprocessing. Any silent frame trimming or future dependence blocks training.
Production probes cover the actual **289-frame** observed learner context plus
one unseen stride block (293 raw frames for Wan, 297 for LTX), with cuts at
1/33/129/289. A 33-frame short probe cannot certify longer contexts, especially
when temporal tiling or normalization changes after 128 frames. A CPU negative
fixture specifically passes the short check then fails at the real horizon.
The real pretrained encoder must pass; CPU arithmetic fixtures only test this
gate and include an encoder whose full-clip mean must fail.

The old `run_long_horizon.sh` caps attention's training gap at `T-8` while giving
recurrent arms a longer clip. The new `FairProtocol` refuses that comparison.
All registered arms require the same raw context, gap, examples, loss, updates,
held-out source split, VAE fingerprint and **initialization seeds 0/1/2**.
The VAE grid needs 33 cue/context frames plus 256 raw hidden frames: 289 observed
frames, with the first unseen target block starting at raw frame 289. This fits
both stride-4 Wan and stride-8 LTX. The previous 288-frame pixel recipe must not
be silently trimmed to 285/281 frames by the VAE. Bounded attention retains only
32 latent steps but is trained/evaluated on the same full horizon. Stronger
factorized attention and sink/reference comparators, explicit reference/latch
controls, matched write/bias/pole ablations and measured resource costs remain
mandatory before a superiority claim. This folder validates completed run rows;
it does not implement or run those new architectures or their training.

The VAE track uses the same normalized latent context for each native
`VideoPredictor(in_ch=vae.channels)`. The existing `observation_memory_v2`
adapter is deliberately RGB-only; it cannot be renamed into a pretrained
latent adapter. Future target blocks remain in separate loss/scorer code and
never enter context extraction or predictor APIs.

`decoded_health_report` uses native decoded pixels and a separately supplied
fixed observed context. It reports each first fade/flatten/freeze/texture/nonfinite
flag with decoded frame, latent step, seconds and sample. Its optional feature
screen requires a frozen pinned local DINOv2 snapshot, uses the same fixed
context anchor throughout, and permits no automatic pixel/random fallback.
Feature inference resizes to DINOv2's 224-pixel input; native pixel statistics do
not resize. These are heuristic diagnostics with semantic quality unverified.
`pixel_metrics` reports decoded RGB error separately from latent error.

Run the CPU contracts and labelled plumbing smoke with the existing interpreter:

```sh
/Users/slavaz/crumb-format/.venv-standalone/bin/python -B -m unittest -v test_validation
/Users/slavaz/crumb-format/.venv-standalone/bin/python -B run_preflight.py --smoke --out receipts/NEW_CPU_SMOKE
```

For later production validation, use `launch_preflight.sh` from the existing
sole conductor after allocation, disk and source identity are verified. Set the
explicit local artifact paths/pins and owned `PROBE_VIDEO`; the script enables
offline loading, requires CUDA, refuses CPU fallback, and has a hard timeout.
`run_preflight.py` also accepts a raw tensor pack containing **frames only**.
The production video probe reads 293/297 consecutive frames at native resolution with no
spatial resize; dimensions must fit the selected VAE's spatial grid. It writes
causality, decoded reconstruction and fixed-context frozen-feature health to a
fresh directory. Its raw opening context remains unreviewed until independent
pixel review supplies a trust anchor, and its reconstruction is not a learned
forecast or long-rollout quality result. The production preflight also measures
actual decoder receptive field and lookahead latency, then compares two chunk
sizes plus an interrupted/restored stream against one decode. Those seam probes
cap their latent spatial grid to bound cost and use synthetic latents: they do
not replace the native-resolution footage quality gate.

`launch_protocol.json` records the remaining launch and evidence gates. If
artifacts, dependencies or CUDA are absent, the command retains a failure
receipt instead of manufacturing pretrained or GPU evidence.
