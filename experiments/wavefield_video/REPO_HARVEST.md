# REPO HARVEST — what our other repos give the video LLM (2026-09-25)

## gaugegap-foundry — HIGHEST value
1. Reproducibility proofpacks (scripts/generate_reproducibility_proofpack.py): config+artifacts+hashes per run.
   Adopt: every headline claim ships a re-runnable proofpack (streaming 1024f test = first candidate).
2. Claim-boundary audit (scripts/claim_boundary_audit.py) scans docs for overclaims.
   Adopt: enforce on all our RESULTS/DEEP_DIVE/report docs.
3. Known-answer benchmarks (benchmarks/known_answers.json): closed-form ground truth per task.
   Adopt: ball trajectories + wave equation have analytic solutions; formalize as known_answers.json.
4. FlowGap PDE track (Burgers, pressure-Poisson) -> NEXT TASK FAMILY for us: dispersion kernel is
   built for propagating waves; test on true wave/flow fields where the inductive bias must show.
5. Evidence pattern: *.audit.json + *.summary.json beside artifacts -> adopt for runs/.

## the-brain / neural-mirror — artifact discipline
- Training/export/eval/registry/promotion pipeline with honest metadata (trained_fixture_unvalidated, validatedAgainst).
- Adopt: per-run status flags (fixture vs validated) + promotion gates for champion selection.

## Crumb-Bob — 290-test audit culture -> counted smoke suite + per-round audit docs.
## VibeVoice — vibevoice_streaming_processor.py + vllm_plugin = reference architecture for serving our streaming step().
## MiroFish (swarm finance) — not relevant. xio-remotion/math-reels — content pipeline only.

## Integration queue -> r5 spec: (a) wave-equation/PDE task, (b) proofpack + claim-audit harness, (c) Triton fused kernels.
