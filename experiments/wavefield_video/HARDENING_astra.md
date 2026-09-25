Share
You said:
CURRENT_STATE.md
Document
Deep research request: harden and better our wave-field video model system at its current point (you'll recognize your own dilution diagnosis in the brief). Read the attached state brief. Produce: ranked concrete moves for (a) objective/architecture, (b) engineering rigor, (c) evaluation integrity, (d) productization toward brainsnn.com; what we're missing or mis-investing in; next 5-10 moves with expected payoff + kill-criteria. Be unsparing. Plain markdown.
Worked for 12s
ChatGPT said:
Bottom line

The project is now at a useful inflection point.

You have real evidence that the wave operator can beat attention in a controlled regime, but you do not yet have evidence that the current wave-field architecture scales into a better video model.

The most important finding in the state brief is not the 16×16 win. It is the combination of:

wave wins 5/5 seeds at 16×16,
32×32 collapses into under-motion,
latent-space models found a darkness shortcut,
your metrics were initially willing to reward that shortcut,
the streaming implementation still does not match the full intended architecture.

That tells me your biggest risk is prematurely optimizing the mechanism before establishing what role the mechanism should actually play. 
CURRENT_STATE

My strongest recommendation is:

Stop treating the wave field as “the attention replacement.” Start treating it as a long-term dynamical memory operator inside a hybrid video architecture.

That distinction matters enormously.

Current long-horizon research is converging on essentially the same principle: cheap persistent state handles long-range memory, while higher-capacity/local mechanisms handle immediate spatial and motion detail. Adobe's long-context SSM work combines a causal SSM with dense local attention rather than asking the SSM to do everything. VideoSSM similarly uses global state-space memory plus a local context window. 
CVPR
+1

Your wave recurrence could be a more structured, interpretable, ultra-cheap version of that global state.

That is the thesis worth trying to kill or prove.

What I think you are currently mis-investing in
1. Too much attention on FFT/kernel engineering

The fused Triton streaming kernel is not remotely a priority yet.

Your brief identifies the missing fused single-step kernel as a gap. Technically it is. Scientifically, it is not. 
CURRENT_STATE

You could make the current model 3× faster and still have a model that freezes.

Do not optimize a mechanism whose correct architectural role hasn't survived testing.

Defer Triton.

Optimization becomes important after:

wave hybrid > matched attention/SSM baseline

on a serious long-horizon task.

2. The three-parameters-per-head constraint is probably becoming ideology

This is elegant:

𝜆
(
𝑘
)
=
𝑒
−
𝛼
(
𝑘
)
𝑒
𝑖
𝜔
(
𝑘
)
λ(k)=e
−α(k)
e
iω(k)

or its damped-cosine equivalent.

But three parameters/head may simply not provide enough conditional dynamics.

Real video does not have one stationary dispersion law.

A ball:

collides,
occludes,
changes direction,
interacts with another object,
enters different spatial regions.

A fixed spectral law cannot know:

"Object A just collided with B, therefore change the propagating dynamics."

Your wave layer is currently content-independent or nearly content-independent.

That is probably the biggest architectural weakness.

Do not throw away the structured kernel.

Make its application conditional.

For example:

𝑧
𝑡
=
𝜆
𝑡
⊙
𝑧
𝑡
−
1
+
𝐵
𝑡
𝑥
𝑡
z
t
	​

=λ
t
	​

⊙z
t−1
	​

+B
t
	​

x
t
	​


where a tiny learned controller generates:

𝜆
𝑡
=
𝑓
𝜃
(
𝑥
𝑡
,
𝑧
𝑡
−
1
)
λ
t
	​

=f
θ
	​

(x
t
	​

,z
t−1
	​

)

but constrained around a stable wave prior.

Think:

physics-shaped state, neural control.

Not:

three sacred numbers forever.

A. Objective / architecture priorities
Rank 1 — Fix train/inference mismatch

This is more important than another loss-weight sweep.

Your long-video model ultimately consumes its own imperfect history.

If training overwhelmingly conditions on correct history, then long-horizon rollout is an entirely different distribution.

This is the classic exposure-bias problem.

Self-Forcing attacks exactly this by performing autoregressive model-generated rollouts during training and supervising the resulting trajectory rather than only ground-truth-conditioned next steps. 
arXiv
+1

You need a cheap version immediately.

Add scheduled self-conditioning

Training batches:

Plain text
0-30% training:
mostly GT history

30-60%:
25% generated history

60-100%:
50-75% generated history

evaluation:
100% generated history

Even better:

roll out 8–32 steps, detach periodically, then calculate trajectory loss.

Expected payoff

Very high.

It attacks the actual deployment failure mode:

compounding recurrent error.

Kill criterion

If wave retains no advantage over the attention/SSM baseline once both train on their own rollout distribution, the current apparent advantage is probably an artifact of teacher-forced dynamics.

Rank 2 — Turn WaveField into hybrid memory

I would change:

Plain text
spatial encoder
      ↓
wave temporal mixer
      ↓
decoder

toward:

Plain text
local spatial/motion block
        ↓
┌─────────────────────┐
│ short-term pathway  │
│ local attention /   │
│ conv / gated mixer  │
└─────────────────────┘
          +
┌─────────────────────┐
│ WaveMemory          │
│ recurrent spectral  │
│ long-term state     │
└─────────────────────┘
          ↓
gated fusion
          ↓
decoder

Something as simple as:

ℎ
𝑡
=
𝑔
𝑡
⊙
ℎ
𝑡
𝑙
𝑜
𝑐
𝑎
𝑙
+
(
1
−
𝑔
𝑡
)
⊙
ℎ
𝑡
𝑤
𝑎
𝑣
𝑒
h
t
	​

=g
t
	​

⊙h
t
local
	​

+(1−g
t
	​

)⊙h
t
wave
	​


could be enough.

This is aligned with recent SSM video work: cheap global state plus dense/local mechanisms appears considerably more promising than forcing one operator to capture both long-term memory and fine local dynamics. 
CVPR
+1

Expected payoff

Extremely high.

It allows wave recurrence to specialize in the thing where its inductive bias actually makes sense:

persistent dynamical memory.

Kill criterion

Matched parameter/FLOP comparison:

Plain text
local-only
wave-only
SSM-only
local + wave
local + generic SSM

If local + wave cannot outperform local + generic SSM on long-horizon degradation, your unique wave formulation is probably not pulling its weight.

That is an important result, even if painful.

Rank 3 — Make WaveMemory content-adaptive

Do not jump to unrestricted Mamba.

Preserve your structure first.

Try:

Plain text
base decay α
base frequency ω
base phase φ

plus tiny context-dependent residuals:

𝛼
𝑡
=
𝛼
0
+
Δ
𝛼
𝑡
α
t
	​

=α
0
	​

+Δα
t
	​

𝜔
𝑡
=
𝜔
0
+
Δ
𝜔
𝑡
ω
t
	​

=ω
0
	​

+Δω
t
	​

𝐵
𝑡
=
𝐵
0
+
Δ
𝐵
𝑡
B
t
	​

=B
0
	​

+ΔB
t
	​


with bounded deltas.

A very small gate network could generate them.

That gives you:

structured wave dynamics with adaptive control.

This is much more defensible scientifically than just growing the layer.

Expected payoff

High.

Particularly for:

collision changes,
speed changes,
occlusion,
appearance changes,
multimodal motion.
Kill criterion

If adaptive parameters improve short-horizon reconstruction but do not increase divergence horizon, kill the complexity.

Rank 4 — Stop solving freezing with scalar loss weights

Your 32×32 dilution diagnosis is credible.

The state brief says the residual term diluted as canvas sparsity increased while the K-ramp encouraged a low-motion solution. 
CURRENT_STATE

But there is a deeper lesson:

pixel-average losses are structurally wrong for sparse moving systems.

At 64×64 the problem becomes worse.

Instead of:

𝐿
=
𝐿
𝑝
𝑖
𝑥
𝑒
𝑙
+
𝜆
𝐿
𝑟
𝑒
𝑠
𝑖
𝑑
𝑢
𝑎
𝑙
L=L
pixel
	​

+λL
residual
	​


use object/motion-normalized objectives.

Example:

𝐿
=
𝐿
𝑟
𝑒
𝑐
𝑜
𝑛
+
𝜆
1
𝐿
𝑚
𝑜
𝑡
𝑖
𝑜
𝑛
+
𝜆
2
𝐿
𝑡
𝑟
𝑎
𝑗
𝑒
𝑐
𝑡
𝑜
𝑟
𝑦
+
𝜆
3
𝐿
𝑜
𝑐
𝑐
𝑢
𝑝
𝑎
𝑛
𝑐
𝑦
+
𝜆
4
𝐿
𝑣
𝑒
𝑙
𝑜
𝑐
𝑖
𝑡
𝑦
L=L
recon
	​

+λ
1
	​

L
motion
	​

+λ
2
	​

L
trajectory
	​

+λ
3
	​

L
occupancy
	​

+λ
4
	​

L
velocity
	​


where moving pixels or tracked objects contribute approximately equally regardless of canvas size.

For the synthetic domain, you know the ground truth.

Use it.

Track:

Plain text
position
velocity
identity
collision event
radius
ball count
trajectory

You already built blob count/position matching. That is the correct direction. 
CURRENT_STATE

Kill criterion

If the wave architecture only wins when heavily sculpted losses are tuned specifically for it, but the baseline works under several reasonable objectives, that is evidence against the architecture.

Rank 5 — Fix the latent representation before training anything serious in latent space

The latent result is currently a red flag:

both models can collapse toward decoded darkness while appearing good under pixel MSE.

That means you cannot draw meaningful conclusions about wave-vs-attention in latent space yet.

Your AE is now part of the experimental system.

Treat it as a model needing qualification.

Recent video-VAE research reinforces that compression design materially determines downstream generation quality; temporal/spatial compression isn't just an implementation detail. 
CVPR
+1

Before another latent model run, establish:

Plain text
AE reconstruction
AE motion preservation
AE object count preservation
AE velocity preservation
AE rollout sensitivity
AE decode brightness distribution

at:

Plain text
1x
4x
8x
16x compression

I suspect 16× compression is too aggressive for the current balls benchmark, because the important information occupies very few pixels.

Kill criterion

If latent encoding cannot reconstruct trajectory/object state with >~99% correctness on your deterministic synthetic dataset, it is not qualified to evaluate temporal architectures.

B. Engineering rigor

This is where I would become much stricter.

Every result needs an immutable run manifest

Every checkpoint:

YAML
git_sha:
model:
parameter_count:
seed:
dataset_sha:
resolution:
num_objects:
objective:
loss_weights:
optimizer:
lr:
batch:
sequence_length:
train_frames:
eval_frames:
teacher_forcing_ratio:
streaming_mode:
precision:
gpu:
runtime:

No exceptions.

One command must reproduce every figure

You want:

Bash
python reproduce.py runs/wf_g32_seed3

and it produces:

Plain text
checkpoint
metrics.json
rollout.mp4
trajectory.png
memory.csv
environment.txt

If a result cannot be reproduced with one command, don't treat it as evidence.

Reference implementation before optimized implementation

Maintain:

Plain text
wave_reference.py
wave_fft.py
wave_stream.py

All three must produce numerically equivalent output within defined tolerance.

Test:

𝐹
𝐹
𝑇
(
𝑠
𝑒
𝑞
𝑢
𝑒
𝑛
𝑐
𝑒
)
≈
𝑟
𝑒
𝑐
𝑢
𝑟
𝑟
𝑒
𝑛
𝑡
(
𝑠
𝑒
𝑞
𝑢
𝑒
𝑛
𝑐
𝑒
)
FFT(sequence)≈recurrent(sequence)

for random inputs and trained weights.

This is critical.

Your central economic claim eventually depends on the recurrence actually being equivalent to the trained operator.

Required CI test

For 100 random cases:

Plain text
max_abs_error
relative_error
gradient_error

Failure means merge blocked.

Determinism audit

Five seeds is good.

Now make the framework enforce:

Plain text
same initialization policy
same examples
same train updates
same parameter budget
same optimizer
same wall-clock or FLOP budget

Parameter matching alone isn't enough.

You need at least two comparisons:

Plain text
equal parameter count
equal compute budget

because the commercial thesis is computational efficiency.

C. Evaluation integrity

This needs a much bigger upgrade.

You already discovered metric gaming once.

Assume it will happen again.

Build a metric hierarchy
Level 1 — reconstruction
Plain text
MSE
PSNR
SSIM

Useful but low-value.

Level 2 — motion
Plain text
mean velocity error
acceleration error
optical-flow error
copy ratio
dynamic degree
Level 3 — object state
Plain text
blob count
position error
identity survival
radius preservation
collision correctness
Level 4 — rollout

This should become your main metric.

Define:

Divergence horizon

First frame 
𝑡
t where state error exceeds threshold.

For example:

𝐻
𝜖
=
min
⁡
{
𝑡
:
𝐸
𝑡
>
𝜖
}
H
ϵ
	​

=min{t:E
t
	​

>ϵ}

Report:

Plain text
H50
H90
median divergence frame

This is vastly more meaningful than final MSE.

Build adversarial anti-cheat metrics

Your evaluator should explicitly flag:

Plain text
FREEZE
BLACK
COPY
BLUR
EXPLODE
DISAPPEAR
DUPLICATE

A run that triggers one should not be eligible to "win."

For example:

Plain text
if motion < threshold:
    FAIL_FREEZE

if brightness collapse:
    FAIL_BLACK

if object count != target:
    FAIL_OBJECT

if frame_similarity ~1:
    FAIL_COPY

Do not average catastrophic failure into a nice score.

Your difficulty ladder is good, but change how you use it

You have:

D1 speed ×2
D2 8 balls
D3 12 balls + speed
D4 smaller radius
D5 64 grid + 8 balls
D6 1024-frame horizon. 
CURRENT_STATE

Do not run all six whenever you make a change.

Use gated progression.

Plain text
D0 sanity
↓ pass
D1
↓ pass
D2
↓ pass
D3
...

Stop immediately when it fails.

This will massively reduce 4090 waste.

Add two tests missing from the ladder
D7 — distribution shift

Train:

Plain text
2-8 balls
speed 1-3
radius 1

Evaluate:

Plain text
10 balls
speed 4
different radii

This tells you whether WaveField learned physics-like dynamics or merely fitted the training distribution.

D8 — memory probe

Create a scene where information disappears and matters later.

Example:

Plain text
red ball enters tunnel
invisible for 128 frames
emerges from one of two exits
exit determined by pre-tunnel velocity

Local attention should struggle once context exceeds its window.

Persistent wave state should not.

This is a much sharper test of your thesis than bouncing balls.

Adobe's SSM world-model work similarly uses long-range memory/retrieval-style tasks rather than only visual similarity because those tests isolate whether the recurrent state actually remembers old information. 
CVPR

This experiment could become one of your strongest demonstrations.

You need a standard external evaluation track

Eventually integrate:

VBench / VBench-Long.

Its dimensions include things like subject consistency, background consistency, temporal flicker, motion smoothness, dynamic degree, and overall consistency. 
GitHub
+1

But don't use VBench yet as your primary R&D loop.

Your synthetic state metrics are more diagnostic right now.

Later:

Plain text
internal deterministic benchmark
+
VBench-Long
+
human blinded comparison

That is much harder to fool.

D. Productization toward BrainSNN

This is where I would alter your framing considerably.

Do not put:

"A new video generator"

at the center of brainsnn.com yet.

You don't have that.

You may have something potentially more interesting:

a persistent neural dynamics engine for long-horizon world simulation.

That actually fits BrainSNN much better.

BrainSNN thesis

I would position the research stack as:

Plain text
BrainSNN
│
├── Persistent Neural Memory
│      └── WaveMemory
│
├── World Models
│      └── long-horizon visual prediction
│
├── Physical Agents
│      └── robots / drones
│
└── Simulation
       └── learned environment dynamics

Then WaveField becomes a BrainSNN research primitive, not an unrelated video side project.

That gives the work somewhere to go even if it never competes with Veo/Kling on cinematography.

Your highest-value product target is probably not text-to-video

At least not first.

There is an easier, stronger wedge:

Persistent visual world model

Input:

Plain text
camera frames
+
actions
+
optional state

Output:

Plain text
future latent state
+
future frames

for:

robotics simulation,
autonomous agents,
drone navigation,
synthetic environments,
interactive worlds.

The reason is straightforward:

Your architecture's theoretical advantage is not:

prettier 5-second clips.

It is:

cheap, persistent recurrent state across very long sequences.

That maps almost perfectly onto world models.

Recent long-context SSM work is explicitly targeting interactive world models for precisely this reason. 
CVPR

What you're missing
1. A proper hypothesis

Right now the implicit hypothesis is:

Wave kernels are better than attention.

Too broad.

Use:

A structured recurrent spectral state can retain useful visual dynamics over substantially longer horizons per byte of persistent state than finite-context attention or generic SSMs.

That is falsifiable.

And valuable.

2. A generic SSM baseline

This is mandatory.

Your serious competitors are no longer just:

Plain text
Wave vs Attention

They are:

Plain text
Wave
Attention
Mamba/SSM
Hybrid local+Wave
Hybrid local+SSM

SSMs already demonstrate strong long-video efficiency, including 256-frame generation and causal long-context modeling. 
Springer
+1

If you don't benchmark against one, the experiment is scientifically incomplete.

3. Memory efficiency as a first-class metric

Start reporting:

Plain text
VRAM at frame 64
VRAM at frame 256
VRAM at frame 1024
VRAM at frame 4096

ms/frame

persistent-state MB

FLOPs/frame

Your strongest graph may eventually be:

Plain text
quality
  ^
  |
  |         Wave
  |        /
  |       /
  | SSM /
  |    /
  | Attn
  +-------------------->
        context length

or:

Plain text
divergence horizon / MB of memory

That would actually be interesting.

Ranked next 9 moves
1. Complete the current g32 objective matrix

Do not interrupt it.

Purpose:

Determine whether under-motion collapse is objective-induced or architecture-induced.

Expected payoff: very high information value

Success:

Plain text
copy_ratio restored
+
trajectory improves
+
wave advantage survives

Kill:

If reasonable objective variants cannot recover motion without catastrophic trajectory error, stop tuning losses and move architecture.

2. Add generic SSM baseline

Implement the smallest credible Mamba/S4-style temporal comparator.

Same:

Plain text
encoder
decoder
params
training budget
dataset

Expected payoff: critical

Kill WaveField thesis if:

generic SSM matches or beats wave across:

Plain text
D1-D6
compute
VRAM
divergence horizon

with no meaningful downside.

3. Build Wave + local mixer hybrid

Do this before richer wave kernels.

Expected payoff: very high

Measure:

Plain text
wave-only
local-only
local+wave
local+SSM

Kill:

If hybrid Wave adds <5-10% divergence horizon over local-only while materially increasing complexity.

4. Add self-generated rollout training

Start with 8–16 autoregressive steps.

Don't build full diffusion forcing yet.

Expected payoff: very high

Recent Self-Forcing work strongly suggests that training on model-generated histories is essential for controlling long-horizon autoregressive degradation. 
arXiv

Kill:

If longer self-conditioning produces no meaningful improvement at 256/1024 frames.

5. Qualify or replace the AE

Run the AE independently.

No model training.

Create:

Plain text
compression | count acc | pos error | velocity error | PSNR
4x
8x
16x

Expected payoff: huge because it removes a confound

Kill 16×:

if semantic state reconstruction materially degrades.

I would rather run an 8× latent that works than a 16× latent that deletes motion.

6. Build divergence-horizon benchmark

Make this the primary headline metric.

For every run:

Plain text
H_position
H_velocity
H_identity
H_motion
H_semantic

Expected payoff: extremely high

This changes the project from:

"we got lower MSE"

to:

"the recurrent state remains correct 6.4× longer."

Much stronger.

7. Create memory-probe world

Tunnel / occlusion / delayed-reappearance test.

Expected payoff: very high scientific clarity

Success:

Wave remains correct after its competitor's local context has expired.

Kill:

If Wave has no long-memory advantage when directly tested for memory.

8. Add bounded content-adaptive wave parameters

Only after #2–#7.

Tiny controller.

Do not explode model size.

Expected payoff: medium-high

Kill:

<10% improvement on divergence horizon across ≥3 hard tasks.

9. Only then optimize recurrent kernel

Then do:

Plain text
fused CUDA/Triton update
persistent state buffers
CUDA graphs
torch.compile
FP16/BF16
batch-1 latency

Expected payoff:

high commercialization payoff, low scientific payoff.

Target:

Plain text
O(1) temporal state
<24 GB
4090
stream indefinitely

Your existing recurrence,

𝑧
𝑡
(
𝑘
)
=
𝜆
(
𝑘
)
𝑧
𝑡
−
1
(
𝑘
)
+
𝐵
𝑥
𝑡
(
𝑘
)
,
z
t
	​

(k)=λ(k)z
t−1
	​

(k)+Bx
t
	​

(k),

is exactly the part that potentially gives you a major systems advantage. 
CURRENT_STATE

But earn the right to optimize it first.

Priority table
Rank	Move	Expected payoff	Cost	Kill signal
1	Finish g32 objective matrix	Very high	Low	motion still collapses
2	Generic SSM baseline	Critical	Medium	SSM dominates Wave
3	Local + Wave hybrid	Very high	Medium	Wave adds little
4	Self-rollout training	Very high	Medium	no horizon gain
5	AE qualification	Very high	Low	latent destroys state
6	Divergence-horizon eval	Critical	Low	n/a
7	Memory probe	Very high	Low	no long-memory advantage
8	Adaptive WaveField	High	Medium	complexity without horizon gain
9	Triton recurrence	Product payoff	Medium	only after model survives
Things I would explicitly stop

For now:

Stop

chasing pixel MSE,
adding more loss coefficients without mechanistic hypotheses,
scaling resolution merely because the next grid is bigger,
optimizing Triton,
treating attention as the only serious baseline,
using 16× latent compression by assumption,
declaring long-video victory from balls,
adding parameters indiscriminately.

Continue

multi-seed runs,
semantic/object-state evaluation,
anti-cheat metrics,
difficulty ladder,
4090 constraint,
causal recurrence,
fair baseline protocol.
The experiment that matters most

If I had your 4090 and could choose one experiment after the current matrix finishes, I would run this:

Long-memory occlusion benchmark
Plain text
64×64
8 moving objects
one target object

frames 0-64:
target visible

frames 65-320:
target hidden behind occluder

frame ~320:
target emerges

correct exit/direction determined by its
pre-occlusion trajectory

rollout:
1024 frames

Compare:

Plain text
A — local attention
B — generic SSM
C — Wave
D — local + SSM
E — local + Wave

Equal compute.

Five seeds.

Metrics:

Plain text
target identity
exit prediction
position error
velocity error
motion preservation
divergence horizon
VRAM
ms/frame
state bytes

That single experiment tells us far more about the actual WaveField thesis than another fifty bouncing-ball MSE runs.

If local + Wave remembers the hidden object's dynamical state substantially longer with materially smaller persistent state, you have something.

If not, we know where to stop spending time.

Where this can become genuinely differentiated

I would no longer aim at:

"replace attention using FFTs."

That story is becoming crowded. SSM-based video architectures already establish the broad efficiency argument. 
Springer
+1

Aim at:

WaveMemory: an ultra-compact recurrent dynamical memory for persistent world models.

And the technical claim becomes:

Plain text
local neural perception
+
structured recurrent dynamics
+
bounded adaptive control
=
long-horizon world state

That is much more interesting.

It also connects cleanly to BrainSNN:

perception → persistent dynamical state → prediction → action.

The current research is therefore not failing. But the original framing is too narrow. Your 16×16 result earned further investigation; the 32×32 and latent failures are telling you how to reformulate the system, not merely which hyperparameter to tweak.

The next milestone I would demand before allowing this project to grow is very simple:

WaveMemory must beat a generic SSM on divergence-horizon-per-byte in a task deliberately requiring memory beyond the local context.

If it does, double down.

If it doesn't, don't spend months polishing the wave idea because it is elegant.


