# Research scope and public claims

Updated October 13, 2026. CRUMB is a structured context handoff format.
Crumb LLM is optional language-model research. The video predictor and video
post-process are separate experiments, with different objectives and evidence.
A timing or quality result from one does not validate the others.

## Claims ledger

| Claim | Evidence and test conditions | Permitted interpretation |
| --- | --- | --- |
| 417× fewer video tokens | [Original and corrected operation timings](../experiments/wavefield_video/LONG_HORIZON.md#1-what-the-receipts-measure) | Retired. The measured quantity was CPU mixer time; VAE token counts were unchanged. |
| 886× operation timing ratio at N=16,384 | [Raw timing receipt](../experiments/wavefield_video/results_long_horizon/bench_scaling_cpu.json): x86_64 CPU, four threads, fp32, one 64-channel head, two warmups, median of five repetitions; 0.968ms FFT versus 857.605ms full SDPA. No trained weights or training-seed comparison. CPU model and RNG seed were not recorded. | One CPU operation against full attention. No GPU, model-step, token-saving or deployed-linear-model advantage follows. |
| Long video at constant state | [7,200-frame stream receipt](../experiments/wavefield_video/results_long_horizon/stream_untrained_halflife.json): untrained weights, grid 16, constant state; freeze flagged at frame zero. [CPU protocol](../experiments/wavefield_video/LONG_HORIZON.md#5-evidence-from-this-change-cpu-4-threads). | Streaming cost only. This is not a coherence result. Initialization seed is not recorded in the stream receipt. |
| Learned long recall | [Clean-write recall](../experiments/wavefield_video/LONG_HORIZON.md#84-diagnosis-the-memory-is-there-but-drowned-measured-before-83-ran): 128-frame blank-background dependency, CPU, grid 6, initialization seed 0, 1,000 updates; clean-write recalls 0.998/0.993. | Narrow procedural recall. Does not establish textured-video memory, three-seed 256-frame recall or geometry. |
| 86–92% drift removal and near-perfect detail/motion | [M0 results](../experiments/wavefield_video/M0_RESULTS.md): fixed analytic engines on one easy synthetic gain scenario, CPU; 24-byte EMA competitive. No learned weights or initialization-seed experiment. Hard phase-drift scenario misses its 60% gate for every engine. Hardware model is not recorded in that report. | Scenario results belong in methods. They are not a customer promise, global quality score or proof the spectral engine beats a simple baseline. |
| Native HD rescue is validated | [Two 1080p gates](../experiments/wavefield_video/reports/ENG1_1080.md): frozen analytic engine versus raw input, two clips, no learned weights. Both fail motion; one fails detail. Exact processor is not recorded. | Failed engineering gate. No successful HD-rescue or restoration claim. |
| The later native 720p gate validates rescue | [October 13 receipt](evidence/native-720-house-20261013.json): one owned generated house clip, 1280×720, 16fps, 121 frames; macOS arm64 CPU, one thread, PyTorch 2.12.0. Frozen analytic operator, no trained weights. Detail 0.991998 passes; motion 0.516330 fails the 0.90 floor. Runtime 5.185s; human review pending. | Failed motion gate. It is not a stranger submission, customer acceptance, payment or improved-video result. |
| 1.5×/2.2× language-model scaling | [Historical CPU table](crumb-llm-architecture.md#5-complexity): dim 64, two layers, four heads, field 8,192; 1.53× at 4K and 2.16× at 8K versus full attention. CPU model, threads, RNG seed and warmup protocol are not recorded. | Retain the qualified timing example. It does not establish video speed, token savings, generation cost or an advantage over windowed/SSM layers. |
| Tiny text perplexity means “10× better” | [Reported text comparison](../CRUMB_LLM.md#1-a-reported-tiny-structured-text-comparison): 231,492 versus 299,040 parameters; wave training 87s versus 41s. The table lacks held-out split and initialization-seed records. | Retired general quality claim. A perplexity ratio is not a task-success ratio. |
| A $99 rescue product has demand | [Pilot decision](../experiments/wavefield_video/THE_POSITION.md) | Proposed price and test. Demand requires independent incoming clips and accepted payment for a useful result. Preparation, owned fixtures and AI reviews do not count as customers. |

## Public wording

CRUMB: “A structured handoff for moving a task between AI tools, with measured
token counts and lexical-retention diagnostics.” Task success is a separate
evaluation, as [the current harness explains](TASK_SUCCESS.md).

Video service evaluation: “A frozen-settings clip rescue pilot. US$99 only
for a useful result the creator accepts. Native HD gates have failed; failed
outputs are ineligible for paid fulfillment.”

Video model research: “Testing whether learned memory improves retrieval and
rollouts against explicit reference memory, frozen frames and relevant linear
baselines.” Multi-minute coherence and architecture superiority remain unproven.

## Before a stronger claim

Record the exact baseline, hardware, dataset/split, independent source count,
initialization seeds, trained status, completed updates and measured cost.
Keep oracle inputs out of inference. Report negative controls, no-event false
presence, actual state-erasure/donor interventions and the first collapse.
Distinguish 256 raw frames from 256 temporally compressed latent steps.

Windowed attention, state-space layers and factorized spatial/temporal
attention are relevant comparisons. A full all-pairs attention timing is a
diagnostic, not the production alternative. Existing `SSMLite` is a generic
diagonal control and must not be described as deployed Mamba.

The rescue service needs a blind native-resolution clip gate and a recorded
customer decision. Research needs the registered 256-frame dependency on
three trained initialization seeds, followed by held-out decoded-video review.
An AI-authored review does not replace an independent human research review.
