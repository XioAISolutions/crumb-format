Deep dive 3

The new data changes the diagnosis materially.

The geometry hypothesis is now weak. You increased moving fraction from 0.09 to 0.55 and g32 still collapsed to near-perfect copying. At the same time, g16 and g32 differ by exactly 4× in tokens per sequence while receiving the same 2,000 optimizer steps. That makes training budget per token the leading confound, but I would not call it proven yet because sequence length/grid size also changes optimization statistics, normalization, kernel discretization, and the effective frequency spectrum simultaneously. 
CONSULT_BRIEF_3

My main recommendation:

Do not modify the architecture yet. First run one tightly controlled scale/budget experiment.

If g32 wakes up when compute/token exposure is equalized, most of the architecture work you were about to do would have been treating an optimization failure as a representational failure.

1. FREEZE ROOT-CAUSE
Current ranking of suspects
1. Insufficient training budget at g32

Most likely.

Your numbers are unusually clean:

g16: 4,352 tokens/sequence
g32: 17,408 tokens/sequence
ratio: 4.0×
both receive: 2,000 optimizer steps
g16 learns motion
g32 converges toward copy/freeze

That is exactly the kind of scaling transition I'd expect when a cheap local optimum becomes learnable before the harder dynamic solution.

The model is not merely "failing."

It has discovered that:

copy previous frame → excellent pixel loss → stable gradients → low short-horizon loss.

Your K=1 result is the giveaway: copy_ratio ≈ 0.005 with tail-MSE 0.016 and your divergence metric still says "pass." In other words, the optimizer is being rewarded for giving up on dynamics. 
CONSULT_BRIEF_3

The interesting possibility is:

g16 had enough optimization density to escape the copy basin before it hardened. g32 did not.

Decisive experiment: do not just run 8,000 steps

8,000-step g32 is necessary.

It is not sufficient to prove budget-per-token causality.

You want a small factorial experiment that separates:

optimizer steps
number of tokens processed
LR schedule
grid-dependent architecture behavior
Experiment A: token-budget equivalence

Keep absolutely everything else identical.

Run	Grid	Steps	Relative tokens seen
A1	16	2,000	1× baseline
A2	32	2,000	4× sequence / same steps
A3	32	4,000	2× more optimization
A4	32	8,000	token-budget matched
A5	16	8,000	overtrained control

Do 5 seeds, not 1.

Record every 100-250 steps:

copy_ratio
motion magnitude
motion direction accuracy
train MSE
rollout MSE
tail MSE
spectral energy per band
gradient norm
parameter update norm
kernel parameter distributions
output variance
prediction-vs-copy loss gap

The most diagnostic plot is:

copy_ratio vs total training tokens processed, not vs optimizer step.

If g16 and g32 curves approximately align when x-axis = tokens seen, you've found your answer.

The stronger control: equal token count per optimizer update

The above still leaves an issue:

g32's individual optimization steps may behave differently because each forward pass contains 4× as many spatial tokens.

So run this too.

Experiment B: equal tokens/update

If memory permits:

g16

batch = B

g32

batch = B/4

or use gradient accumulation so that:

tokens_per_optimizer_update is identical.

Then give both the same number of:

optimizer updates
total tokens
LR schedule position

You want the experiment to answer:

Is failure caused by insufficient total exposure, or does merely increasing N change the optimization landscape?

That distinction matters enormously.

Most important additional control: downsampled g32

This is the experiment I think you're missing.

Experiment C: same 32×32 tensor geometry, 16×16 information content

Take your g16 scene.

Upsample it to 32×32.

Train the g32 model.

Now:

architectural N = 32
token count = g32
task information = essentially g16
motion geometry = equivalent to successful condition

If this still freezes at 2k but wakes at 8k:

training-budget explanation gets much stronger.

If it still freezes at 8k:

something about N=32 itself is breaking learning.

That immediately points you toward:

kernel parameterization
normalization
frequency scaling
receptive-field scaling
initialization.
The other major suspect: your kernel changes meaning with N

This is the strongest architectural confound.

You are using a spectral/wave operator.

Going from 16×16 → 32×32 is not necessarily a neutral "more pixels" operation.

Depending on implementation, your learned kernel parameters may correspond to:

FFT bin index
normalized frequency
pixel-space wavelength
physical coordinate frequency

Those are not interchangeable.

If a learned/configured wavelength was sensible at N=16 but the same raw parameter becomes effectively half-sized or twice-sized relative to the scene at N=32, then your wave mechanism itself changes.

Audit immediately

For every kernel parameter, answer:

Does this parameter represent frequency in cycles/image or cycles/pixel?

You want physical normalization such as:

kx = fftfreq(W, d=1/W)

or an explicitly normalized [-0.5, 0.5] / [-π, π] frequency grid.

Not raw FFT-bin IDs.

Diagnostic

Plot the initialized and learned transfer functions for g16 and g32 against:

normalized spatial frequency, not FFT index.

They should overlap.

If they don't, you've accidentally changed architectures.

Suspect #3: LR schedule is step-based when difficulty is token-based

Suppose you use:

warmup = 100 steps
decay beginning at step X
same peak LR
same total steps

At g32 the model sees four times more token-level prediction tasks in each forward pass, but only a quarter as many parameter updates relative to total prediction positions.

A 200-step warmup therefore represents a radically different amount of data.

For Experiment A, compare:

g32-8k / schedule stretched 4×

Every schedule boundary ×4:

warmup
decay onset
total decay length
g32-8k / original 2k schedule repeated or exhausted

This control tells you whether the issue is:

training duration versus schedule geometry.

My prediction: stretched schedule wins.

Suspect #4: loss normalization is hiding motion gradients

Check this before burning GPU time.

If your loss is:

MSE.mean() over all pixels/tokens,

then moving-object gradients become diluted by static background.

You already tried increasing moving fraction dramatically and it still froze, which weakens this explanation. 
CONSULT_BRIEF_3

But there is another dilution mechanism:

Gradient density

Even if 55% of pixels participate geometrically, the gradient from:

static low-frequency structure
background
slowly varying regions

may dominate the harder velocity-sensitive component.

Instrument:

||grad_dynamic|| / ||grad_static||

or approximate it with two masked backward passes.

That gives you more information than another motion-weighted scalar objective.

Suspect #5: implicit spectral bias toward DC / low frequencies

This one is highly relevant to your architecture.

Neural systems typically learn smooth/low-frequency structure earlier than fine/high-frequency dynamics.

Your model may discover:

background
object appearance
static placement
only later motion

At 2,000 steps g16 reaches stage 4.

g32 reaches stage 2-3.

The result looks like an architectural freeze when it's actually optimization staging.

Measure per-band prediction error through training:

DC
lowest 10%
middle frequencies
upper frequencies

If g32 progressively solves low frequencies while motion-sensitive bands lag, that strongly supports the budget hypothesis.

My decisive suite

Don't run 20 experiments.

Run these six.

Stage 1
g16 / 2k
g32 / 2k
g32 / 8k with 4× stretched LR schedule

5 seeds each.

Stage 2, only if #3 improves substantially
g32 / 8k original schedule
g16 / 8k
upsampled-g16-content → g32 / 8k

That gives you enough information to distinguish almost every major hypothesis.

Decision thresholds

Define success before training.

For g32-8k:

Budget hypothesis strongly supported

If median:

copy_ratio > 0.45
meaningful motion survives >64 frame rollout
tail error improves ≥30% relative to g32-2k
≥4/5 seeds escape freeze
Partial support

copy_ratio 0.15-0.45.

Then optimization contributes, but architecture/objective still needs intervention.

Budget hypothesis substantially falsified

If:

copy_ratio remains <0.1
4/5 or 5/5 seeds freeze
training loss continues falling

after token-equivalent training.

At that point stop spending compute.

Something structural is wrong.

2. ARCHITECTURE: NEXT THREE CHANGES

My ranking has changed because of the freeze data.

#1 conditional λ_t

Do this first, but only after the 8k control.

Why?

Because your failure isn't simply "not enough motion loss."

Your objective patches already had effectively zero effect. 
CONSULT_BRIEF_3

A fixed motion weight says:

motion always matters by the same amount.

But what you actually need is:

increase anti-copy pressure precisely when the model begins collapsing toward identity dynamics.

That is a control-system problem.

Exact conditional λ spec

Let:

c_t = similarity(pred_t, input_last_frame)

Use whichever differentiable proxy tracks your copy_ratio best.

For example:

d_motion = mean(|ŷ_t - x_t|)

Maintain target movement magnitude based on GT:

d_gt = mean(|y_t - x_t|)

Define collapse:

r_t = d_motion / (d_gt + ε)

Then:

λ_t = λ_min + (λ_max - λ_min) * sigmoid((τ - r_t)/T)

Suggested start:

λ_min = 0.1
λ_max = 2.0
τ = 0.35
T = 0.08

Interpretation:

If predicted motion falls under roughly 35% of GT motion, motion-preservation pressure ramps sharply.

Do not use copy_ratio itself if non-differentiable.

Loss:

L = L_pred + λ_t L_motion

but stop-gradient through λ_t.

Otherwise the model can game the controller.

Better version

Condition λ on two variables:

under-motion
current rollout horizon

λ(t,h) = λ_motion(t) × (1 + α h/H)

with:

α = 0.5 initially
cap total λ ≤ 2.5

That gradually increases pressure on motion persistence deeper into rollout.

#2 self-conditioned rollout training

This likely matters much more for your long-video goal than for the immediate 1-step freeze.

Teacher-forced training can produce a model that behaves well on clean prefixes but deteriorates once conditioned on its own predictions. That train/inference mismatch and resulting error accumulation is the classic motivation behind scheduled sampling. 
Google Research

Exact experiment

Don't jump to 32-frame rollouts.

Use curriculum:

Phase A

75% teacher-forced
25% 2-step self-conditioned

Then:

Phase B

50% teacher
25% 2-step
25% 4-step

Then:

Phase C

25% teacher
25% 2-step
25% 4-step
25% 8-step

Detach predicted frames initially:

x_{t+1} = stopgrad(ŷ_t)

Otherwise memory/optimization cost explodes.

Later test gradient-through-rollout for 2-4 steps only.

Loss weighting

Use:

w_h = γ^(h-1)

with γ ≈ 0.85-0.95.

Don't equally punish frame 1 and frame 8 initially.

#3 hybrid gated fusion

This is potentially the strongest eventual architecture.

But it is the worst one to introduce right now because it can mask your scientific result.

You need to know whether the wave operator itself works.

Architecture

Parallel:

z_wave = WaveBlock(x)

z_local = LocalMixer(x)

Then:

g = sigmoid(MLP([x, z_wave, z_local]))

and:

z = g * z_wave + (1-g) * z_local

But I would simplify first.

First implementation

Use one scalar gate per channel/head, not per pixel:

g_h = sigmoid(a_h)

Initialize:

a_h = +1.4

so:

g ≈ 0.80

Meaning ~80% wave / 20% fallback path initially.

Fallback path:

depthwise 3×3 conv
temporal 3×1×1 conv
small channel MLP

Not attention yet.

Why?

Because if you immediately add attention and it wins, you've learned almost nothing about whether wave dynamics work.

Ranking
Conditional λ_t
Self-conditioned rollout
Hybrid gated fusion

But:

Which one first?
None until the g32-8k test.

After that:

If g32 wakes up

Do self-conditioned rollout first.

Freeze was optimization.

Your next real bottleneck becomes autoregressive exposure.

If g32 partially wakes but remains copy-biased

Do conditional λ_t first.

If g32 remains dead

Do hybrid fusion or revisit kernel scaling, not λ tinkering.

At that point the operator is likely missing something fundamental.

3. CRUMB AS A PLUGIN

This direction is much more commercially interesting than "new video foundation model."

But I would change the product definition.

Your current concept:

take output video → FFT → replace/blend low frequencies toward rolling wave state.

is clever, extremely cheap, and probably useful.

But I do not believe that alone can deliver "ANY pipeline gets 2-5 minute coherence."

That's too strong.

What pixel-space spectral anchoring can fix

Likely:

global luminance drift
color-temperature drift
broad layout drift
large-scale shape drift
background tone
low-frequency camera/scene consistency
some temporal flicker

FreqForcing gives real evidence that long-rollout degradation is strongly associated with low-frequency spectral drift, and its training-free spectral anchoring extended a short-horizon model to two-minute generation. 
arXiv
+1

But note what FreqForcing actually anchors.

It does not merely FFT the final RGB video.

It operates on internal attention representations, combining local and anchor branches and injecting the low-frequency residual from anchor attention while preserving local high-frequency behavior. 
Academ.us

That difference is huge.

Pixel FFT cannot reliably preserve semantic identity

Consider a person.

The low-frequency RGB spectrum does not uniquely encode:

face identity
clothing identity
object permanence
which hand holds an object
number of people
scene topology
causal state
temporal intent

Two semantically different frames can have almost identical low-frequency spectra.

So a pure pixel FFT plugin could produce:

visually stable but semantically drifting video.

That is the key risk.

Worse: aggressive spectral anchoring can itself cause freezing

This should jump out given your current model failure.

If you repeatedly pull low-frequency components toward an anchor, you're telling the system:

remain like the past.

Too much λ gives you exactly what you're fighting now:

spectral copy bias.

It can preserve scene structure while quietly suppressing:

camera motion
subject translation
genuine scene transitions
scale changes
lighting changes

Your anchoring strength therefore cannot be constant.

Better product: coherence controller, not coherence filter

I would make Crumb a tiny stateful inference middleware layer.

Conceptually:

Plain text
generator
   ↓
current latent/frame chunk
   ↓
Crumb
 ├── persistent spectral state
 ├── motion state
 ├── semantic anchor state
 └── adaptive correction
   ↓
next-generation context

Not merely:

Plain text
frames → FFT filter → frames
Three-tier interface
Tier 1: RGB-only universal mode

Works with literally anything.

API:

Python
crumb.push(frames)
corrected_frames, state = crumb.correct(frames)

State contains:

low-frequency temporal spectrum
DC/color statistics
coarse optical flow
low-resolution anchor frame
scene-change score

This becomes the universal baseline.

It won't solve identity perfectly.

But it is truly pipeline-agnostic.

Tier 2: latent mode

Much stronger.

API:

Python
crumb.correct_latents(
    z_current,
    z_anchor,
    timestep,
    motion_hint=None
)

This is where I expect the product to become compelling.

Operate on:

VAE latents
intermediate video tokens
transformer hidden states

rather than decoded pixels.

Why?

Because latents preserve more semantic structure while still being small.

Tier 3: attention/KV mode

Strongest, least universal.

For pipelines that expose:

KV cache
attention tensors
temporal context cache

Crumb supplies persistent long-horizon anchor state.

This gets closest to what FreqForcing shows can work well: anchor and local internal representations can be fused spectrally rather than trying to repair final RGB after semantic errors have already occurred. 
Academ.us

This is the plugin shape I would build
Crumb State

Maintain:

1. S_spec

Persistent low-frequency state.

Size:

~32×32×C or smaller.

Update:

S_t = β S_(t-1) + (1-β) F_low(x_t)

But β is adaptive.

2. S_anchor

Sparse clean anchor snapshots.

Not just frame 0.

For example:

startup anchor
periodic "trusted" anchors
scene-change reset
3. S_motion

Coarse motion vector / trajectory statistics.

Without this, your spectral anchor will fight legitimate camera or subject movement.

4. S_sem

Optional small semantic embedding.

Something as cheap as a pooled visual encoder vector could help determine:

is this drift, or is this a legitimate scene change?

Adaptive λ is essential

Don't use:

corrected = current + λ H(current_anchor-current)

with constant λ.

Use something like:

λ = f(drift, motion, scene_change, confidence, horizon)

Qualitatively:

spectral drift ↑ → λ ↑
legitimate optical flow ↑ → λ ↓
scene cut ↑ → λ → 0 and reset anchor
identity confidence ↓ → maybe λ ↑
strong intentional camera movement → directional/phase-aware correction

Otherwise Crumb becomes a sophisticated freeze filter.

Biggest technical issue: phase

Magnitude-only spectral control is not enough.

For moving objects, translation changes Fourier phase dramatically while leaving magnitude related.

If you pull complex low-frequency coefficients directly toward an old anchor, you can fight real translation.

Therefore:

Separate amplitude and phase

Try:

stronger correction on amplitude
weaker correction on phase

Example:

A' = (1-λ_A) A_cur + λ_A A_state

φ' = φ_cur + λ_φ wrap(φ_state - φ_cur)

with:

λ_A ~ 0.2-0.6
λ_φ ~ 0-0.15 initially

Better still:

estimate coarse optical flow / camera transform and warp the state before spectral comparison.

That could be crucial.

Temporal FFT window matters

A full-history FFT is wrong for an evolving scene.

Use rolling windows.

Suggested initial:

spatial state: 32×32 or 64×64 downsampled
temporal history: 16-32 frames
update every generated chunk
retain exponentially decayed global state separately

You effectively want:

short-term temporal spectrum + very-long-term anchor memory.

What must be true for Crumb to work

There are five hypotheses.

H1. Long-video degradation has a measurable low-frequency precursor

Before visible collapse:

D_low(t)

must increase.

If degradation happens first in semantics/high frequencies, your controller gets no useful early warning.

FreqForcing provides evidence for this phenomenon in its target architecture, but not yet across arbitrary video generators. 
arXiv

H2. Correcting that drift causally improves future frames

Correlation isn't enough.

Maybe spectral drift is just a symptom.

Your intervention must cause:

longer coherence horizon.

This is your real scientific claim.

H3. The intervention doesn't suppress desired motion

Critical.

You need:

coherence ↑

without:

motion ↓

Given your existing freeze problem, this must be a first-class metric.

H4. State transfer works across generators

Test at least:

one autoregressive video model
one diffusion/rolling model
one image-to-video chaining pipeline

If Crumb only works on your own architecture, it isn't yet the plugin product.

H5. Intervention after decoding isn't too late

This is the one I am most skeptical about.

If generator segment n+1 conditions on its own latent/history, but Crumb only repairs the exported RGB video afterward, you've corrected the recording but not the generator's internal trajectory.

The next segment still inherits corrupted state.

Therefore:

Crumb must ideally sit inside the recurrent generation loop, before the corrected context is fed into the next chunk.

That's the right interface.

Correct interface

Not:

Plain text
Generate segment
↓
save mp4
↓
Crumb postprocess

but:

Plain text
Generate segment n
↓
Crumb.observe()
↓
Crumb.correct_context()
↓
corrected context fed back into generator
↓
Generate segment n+1

That makes it a true coherence engine.

MVP

Don't start with 5 minutes.

Make the progression brutally falsifiable.

Milestone 0: offline corruption test

Take real videos.

Artificially inject:

color drift
DC drift
low-frequency spatial drift
blur accumulation
gradual contrast shift

Can Crumb restore them without damaging motion?

Target:

80% reduction in injected low-band drift

<5% loss in motion energy

If not, stop.

Milestone 1: synthetic AR test

Use your moving geometry dataset.

Generate 256-1,024 frames.

Compare:

vanilla
RGB spectral state
amplitude-only state
amplitude + phase
motion-compensated state

Metrics:

copy_ratio
trajectory error
spectral divergence
long-horizon MSE
motion energy

Kill if:

Crumb improves MSE/coherence only by reducing motion.

That's fake success.

Milestone 2: one real generator, 60 seconds

Pick one pipeline with accessible recurrent context.

Not five.

A/B:

same prompt
same seed
same base generator
Crumb off/on

10-20 prompts.

Measure:

CLIP/DINO identity stability
optical-flow persistence
LPIPS temporal smoothness
low-band spectral drift
scene composition drift
human blind preference

Target:

≥2× coherence horizon with <15% degradation in motion metrics.

Milestone 3: two minutes

Only proceed if 60 s works.

Requirement:

70% prompt runs reach 120 s

no catastrophic freeze
no monotonic desaturation / blur
identity/scene stability meaningfully above control
memory approximately constant

Your recurrence already demonstrated 1,024-frame streaming with flat ~0.207 GB memory, so the state-storage proposition is credible. 
CONSULT_BRIEF_3

Milestone 4: generator transfer

Run exactly the same controller with only adapters changed.

Two unrelated pipelines.

If you need model-specific retraining for each one, stop calling it universal.

Milestone 5: 5 minutes

At this point run:

5 min
10 seeds/prompts
automated metrics every 5 sec
blind clips at 0s / 30s / 60s / 120s / 300s

Your headline metric should become:

Coherence half-life

Define a quality threshold Q_min.

Measure time until:

identity
motion
composition
spectral stability

cross threshold.

Then report:

baseline coherence half-life = 38s
Crumb = 171s

That is far more compelling than saying:

"generates 5-minute videos."

Hard kill criteria for the plugin

I would kill or radically reshape the RGB spectral version if any of these occur:

<1.5× coherence-horizon gain
Improvement disappears when motion preservation is controlled
λ needs hand-tuning per prompt
Scene changes constantly fool the anchor
Pixel correction helps appearance but doesn't improve subsequent generation
Latent/internal intervention dramatically outperforms RGB intervention
Different generators require fundamentally different controller logic
2-minute performance requires growing state proportional to history
The product claim I would pursue

Not:

"Crumb makes any video model generate 5-minute videos."

Too broad and easily falsified.

Instead:

Crumb is a stateful coherence layer for streaming video generators. It carries a tiny persistent visual state across generation windows and corrects long-horizon drift before it compounds.

Then the differentiator:

Generator context can stay short. Crumb memory stays long.

That is genuinely interesting.

The deeper opportunity

I think your most valuable idea may be less about Fourier filtering and more about externalizing long-term video memory from the generator.

The generator handles:

next few seconds.

Crumb handles:

what must remain true across minutes.

That state might eventually contain:

Plain text
spectral state
+ appearance state
+ geometry state
+ motion state
+ semantic/object state

The wave recurrence is then not "the video generator."

It becomes the persistent dynamics memory.

That is a much stronger architectural and product position.

What I would run next, in exact order
g32 8k, 5 seeds, LR schedule stretched 4×.
Plot copy_ratio + spectral energy + motion magnitude against tokens processed, not steps.
Audit frequency/kernel normalization between N=16 and N=32.
Run upsampled-g16 → g32 control.
If g32 recovers: add self-conditioned 2→4→8-step rollout training.
If it only partially recovers: add conditional λ_t.
Build Crumb MVP as a closed-loop context corrector, not MP4 post-processing.
Start amplitude-dominant low-frequency state, very weak phase anchoring.
Add motion-compensated anchor warping before increasing λ.
Prove 60 s → 120 s → cross-generator transfer → 300 s in that order.

The biggest thing I would not do right now is add hybrid attention/gated fusion before resolving the 4× token-budget confound. If 8k g32 suddenly moves, that result tells you your wave architecture was substantially healthier than the 2k experiments made it look.