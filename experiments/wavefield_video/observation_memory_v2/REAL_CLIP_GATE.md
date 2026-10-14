# Real-clip gate: freeze before viewing results

This is a prepared protocol, not a completed experiment. The observation layer currently has synthetic RGB retrieval evidence only. No native-resolution candidate render or outside-customer acceptance was measured in this task.

## Engineering gate

1. Record independently sourced, owned or explicitly authorized footage, source hashes, native dimensions/frame rate, frame windows and train/test provenance. Different filenames or crops of the same footage are not independent clips. Keep new held-out footage separate from the two-video latent corpus.
2. Freeze the candidate, checkpoint, codec/generative-prior versions, preprocessing and read policy before running the held-out footage. Do not composite the current RGB latch directly: it stores historical appearance and fails the small-occluder control. A visibility-aware consumer is a prerequisite.
3. Poison future raw frames before VAE encoding and verify observed-context latents remain unchanged. Record encoder temporal receptive field, raw-frame gap and latent-step gap separately. Encode only permitted history; do not assume that slicing latents after a whole-clip encode is causal.
4. Give all learned arms the same observed context, prediction horizon, query and independent windows. Retain fixed last-context persistence, reference/latch, sink/reference plus bounded-window attention and a suitable factorized/state-space comparator. Match resource and write variants for any architecture comparison; otherwise report integration results only.
5. Score decoded pixels over the full rollout, not only event matches or one start. Report per-source results, first collapse, motion, detail, freeze/drift, false remembered objects, unknown regions, wall time, peak device memory and actual completed updates. Preserve failed runs and timeouts. A self-evaluator's preferred take is not independent human evidence.
6. Blind the candidate/control order for an independent reviewer. Review complete native 720p/1080p clips rather than a resized grid or selected frames. Check that exposure flicker reduction preserves intended motion and lighting changes.

For a stabilizer task, the existing frozen ENG-1 criteria are a useful starting point, not proof of memory quality: detail correlation median ≥0.97, frame-difference motion ratio in [0.90,1.10], jitter max/p95 ≤1.5 and runtime ≤1,800 s per clip. Its two native-HD clips currently FAIL. Freeze any task-specific additions before evaluating the new candidate, and report original baseline failures alongside candidate results. These ratios alone do not establish object identity, 3D geometry or long coherence.

ENG-1's sources are rendered math reels, despite the report's “real-footage” title. Keep this provenance label; independent natural-scene/client validation remains outstanding.

## Separate 72-hour service test

Freeze one narrow deliverable, eligibility/refusal rules, price, maximum turnaround, comparison procedure and evidence requirements before accepting work. Use three independent authorized client clips and a human review; record refusals, quality failures, actual paid orders and delivered results. Technical success and customer willingness to pay are different outcomes. No payment or demand should be inferred from a free trial, friendly review or draft offer.

No solicitation, pricing change, publication, billing or outside-client processing is authorized by this protocol itself. It provides a concrete reviewable test when the owner chooses to run that separate service challenge.
