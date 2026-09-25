Bottom line

I agree with roughly 70% of Claude’s direction, but I would change the ordering.

The biggest thing the review missed is that your current experiment is not yet measuring whether wave-fields are competitive. The smoke models have not even cleared a trivial persistence baseline, the attention arm is several times larger, the wave kernel has the wrong geometry for moving objects, and the dataset boundary condition conflicts with circular FFT convolution.

So I would not jump straight to latent video or spend tonight optimizing recurrence.

My order would be:

Fix the experimental harness and residual prediction.

Upgrade the kernel geometry, especially space-time coupling and cheap content gating.

See whether wave can match a fair efficient baseline on actual motion prediction.

Then convert the successful kernel to its recurrent SSM form.

Then move into latent space for real video scale.

1. Do I agree with the review?
Strongly agree

The recurrence insight is real.

For a causal damped cosine,

h
n
	​

=e
−αn
cos(ωn+ϕ)

you can represent the temporal dynamics with a complex recurrent state:

z
t
	​

=λz
t−1
	​

+Bx
t
	​

,λ=e
−α+iω

and recover the real output through Re(C z_t).

So the same family can have:

parallel convolution/FFT training

recurrent streaming inference

memory independent of sequence length

That is exactly the convolution/recurrent duality exploited by structured SSMs such as S4. 
arXiv

But “O(1) state” needs a qualifier: it is O(1) in temporal sequence length, not literally one tiny state. You still maintain state across spatial locations/frequencies, heads, channels and modes.

Latent space is eventually mandatory.

Your current predictor embeds every pixel into a token and keeps all T × H × W tokens. 

wfvideo

 That is fine for studying the mixer but not for generating 720p video for minutes.

Current video systems commonly compress video before expensive generative modeling; for example CogVideoX uses a 3D VAE, while GPDiT operates autoregressively in continuous latent space. 
arXiv
+1

Content independence is a real weakness.

Mamba specifically attributes part of the weakness of earlier efficient sequence models to lack of content-dependent behavior, then makes SSM parameters functions of the input. 
arXiv
 Hyena similarly combines long convolutions with data-controlled gating. 
arXiv

So Claude is right that a completely fixed wave filter will eventually hit a quality ceiling.

Where I disagree
Causality is not currently a correctness blocker

This is overstated in the review.

Your model receives:

frames 0 ... T-1

and predicts:

frame T

There is no frame T inside the input. 

train_compare

Therefore, bidirectional mixing among the observed context frames does not leak future target information. Even at autoregressive evaluation, each new forward pass only receives past/generated frames. 

train_compare

What is genuinely wrong is the circular boundary condition.

Your FFT is same-size:

Python
Run
X = rfftn(h)
y = irfftn(X * K)

so frame 0 and frame T-1 become neighbors modulo T. 

wfvideo

That's a bad temporal geometry for streaming video, but it is not target leakage.

Causality becomes mandatory when you want the model itself to maintain a persistent recurrent state rather than repeatedly reprocessing a finite context window.

Worse: the circular-boundary bug also exists in space

Claude mostly talks about time.

But the 3D FFT is circular over T, H and W. 

wfvideo

Your synthetic world uses reflecting boundaries:

Python
Run
pos = torch.where(low, -pos, pos)
pos = torch.where(high, 2 * maxs - pos, pos)

data

So the model assumes:

left edge ↔ right edge
top edge ↔ bottom edge

while the simulated physics assumes walls.

That mismatch could materially hurt the wave model, particularly at 8×8 where boundaries occupy a large fraction of the image.

The biggest architectural weakness isn't “six parameters”

It's separability.

Right now:

K(t,y,x)=K
t
	​

(t)K
y
	​

(y)K
x
	​

(x)

and, worse, the same spatial kernel is used for x and y. 

wfvideo

But moving video content looks more like:

I(x,y,t)=I
0
	​

(x−v
x
	​

t, y−v
y
	​

t)

The required correlations couple space and time.

A translating object produces the spectral relationship roughly:

ω
t
	​

=v
x
	​

k
x
	​

+v
y
	​

k
y
	​


Your present filter cannot directly express that because its temporal frequency is independent of spatial frequency.

That's a much more fundamental limitation than having only six scalars/head.

What you currently have resembles a standing separable field.

What you want is a propagating field.

S4ND is relevant here because it explicitly treats images/video as multidimensional signals rather than flattening everything into an ordinary temporal sequence. 
arXiv

The experimental flaw Claude completely missed

Your smoke result is:

wave: ~0.153

attention: ~0.092

as Claude notes. 

claude_opus_review

I executed your uploaded generator with 512 seeded T=17 sequences.

At 8×8:

Copying the last frame gives ~0.0155 MSE.

So:

copy last       ≈ 0.0155
attention       ≈ 0.092
wave            ≈ 0.153

Attention isn't beating wave meaningfully yet.

Both learned models are failing the trivial baseline.

That completely changes how I interpret the 0.15 vs 0.09 gap.

At 32×32 I get an even lower copy-last MSE of about 0.00154, because your Gaussian balls occupy a smaller fraction of the total pixels. That also means raw image MSE is a poor metric for comparing different resolutions.

Before doing architecture science, add:

zero predictor
copy-last predictor
constant-velocity predictor

to every result table.

Another major confound: attention isn't the same budget

Every attention block has:

Python
Run
self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))

wfvideo

and you instantiate a separate mixer for every layer. 

wfvideo

Using your code directly:

8×8-ish configuration
wave       ~85k params
attention ~241k params
             2.8× larger

default 32×32, D=384, L=8
wave        ~11.8M params
attention   ~67.7M params
              5.7× larger

So "same scaffold + budget" in train_compare.py isn't true.

I would not add another giant absolute positional table to wave as Claude suggests.

Instead give both models the same compact positional representation, ideally one shared factorized t + y + x encoding or relative/RoPE-style encoding.

2. Which single architecture upgrade first?
Richer kernels.

But not just “K=8 damped cosines instead of K=1.”

I would make the temporal pole depend on spatial frequency.

Something like:

λ
h,m
	​

(k
x
	​

,k
y
	​

)=e
−α
h,m
	​

(k
x
	​

,k
y
	​

)
e
iΩ
h,m
	​

(k
x
	​

,k
y
	​

)

with perhaps 2–4 modes/head.

A very simple initial dispersion function could be:

Ω(k
x
	​

,k
y
	​

)=v
x
	​

k
x
	​

+v
y
	​

k
y
	​

+β
k
x
2
	​

+k
y
2
	​

	​


Then each head can learn propagation direction/speed instead of applying the same temporal oscillator everywhere.

Conceptually:

frame
  ↓
2D spatial FFT
  ↓
each spatial frequency gets its own damped oscillator
  ↓
inverse FFT
  ↓
local/gated nonlinear block

And crucially, this architecture is already recurrence-compatible:

z_t(k) = λ(k) z_{t-1}(k) + B x_t(k)

So you aren't throwing away the SSM insight. You're designing the richer wave operator in a form that can become an SSM immediately afterward.

Why not the other three first?

Causal + linear FFT: needed eventually, but it probably won't turn 0.15 into 0.09. It mostly fixes semantics/boundaries.

SSM recurrence: hugely important for minutes, but mathematically equivalent recurrence does not increase representational capacity. Don't optimize streaming for a mixer that has not demonstrated useful prediction yet.

Latent tokens: unquestionably needed for serious video, but introducing a VAE now adds compression artifacts and another model to debug. First determine whether the mixer itself works.

So my sequence is:

richer propagating kernel → recurrent form → latent space.

3. Sharpest falsifiable one-4090 experiment tonight
Hypothesis

A propagation-aware wave mixer can learn long-horizon physical motion at comparable short-horizon quality to efficient sequence models, while degrading more gracefully under autoregressive rollout.

Do not test “minutes-long video generation” yet.

Test the temporal/spatial mixer.

Dataset

Keep it synthetic and cheap, but modify your ball generator.

Use:

16×16
T context = 16
3 colored balls
random independent velocities
elastic wall collisions
ball-ball collisions/occlusions
deterministic dynamics only

I would not use Claude's random future velocity kicks.

If a velocity kick is genuinely stochastic and hasn't been observed yet, no deterministic predictor can know it. You would be testing irreducible uncertainty rather than memory.

Later, if you want stochastic futures, switch to a probabilistic generative objective.

For tonight, make every future deterministic given the observed state.

Four arms
P0  copy-last baseline

W0  current separable wave

W1  propagation-aware wave
    λ(kx,ky), 2–4 modes/head
    + cheap data-dependent gate

S1  generic recurrent SSM/Mamba-like baseline

A1  attention baseline
    compact shared positional encoding
    parameter matched

You don't need an enormous Mamba implementation. A clean diagonal/complex SSM baseline is already much more informative than dense attention.

Long-video diffusion work has already shown SSM temporal modeling can handle sequences up to 256 frames with favorable memory/quality tradeoffs, so this is the serious efficient baseline your wave formulation has to justify itself against. 
arXiv

Architecture budget

Use approximately:

D = 128
layers = 4
heads = 8
grid = 16
T = 16

Match learned parameters to within roughly 10%.

No giant [N,D] absolute position matrices.

Training change that I consider mandatory

Predict a residual from the final frame:

Python
Run
delta = head(last)
pred = frames[:, -1] + delta

Zero-initialize head.

That means step zero begins at the strong persistence baseline rather than learning RGB reconstruction from scratch.

Then train with:

1-step loss
+
4-step autoregressive rollout loss

For example:

L=L
1
	​

+0.25L
2
	​

+0.25L
3
	​

+0.25L
4
	​


Your current training is entirely one-step teacher forcing, while the thing you care about is autoregressive stability. 

train_compare

That mismatch will otherwise dominate your divergence results.

Evaluation

Roll out 256 frames from 32 fixed unseen seeds.

Measure four things.

1. Persistence-normalized MSE

Report:

r=
MSE(copy−last)
MSE(model)
	​


rather than raw MSE.

You want:

r < 1

first.

If a model can't beat persistence, stop analyzing its architecture.

2. Object trajectory error

Because you know the ball colors, recover the centroids.

Measure:

mean centroid error
final centroid error
identity survival

This removes the huge-background problem inherent to pixel MSE.

3. Divergence horizon

Define it before running the experiment:

First frame where median ball-centroid error exceeds one ball radius for 8 consecutive frames.

Your radius is currently 1.6 pixels. 

data

No moving the threshold afterward.

4. Efficiency

Record:

peak VRAM
training steps/sec
rollout frames/sec
memory at rollout length 32
memory at rollout length 256

The existing smoke benchmark already demonstrates that the FFT operator scales dramatically better than dense attention spatially. 

RESULTS

The unanswered question is whether useful predictive quality survives.

Kill criteria

This is the important part.

Experiment invalid

If W1, SSM and attention cannot significantly beat copy-last one-step performance.

That means you still have a training/harness problem.

Kill the current separable wave

If W0 loses badly to W1.

That's actually a useful result: separability, not the general field concept, was the problem.

Kill the special “wave-field” hypothesis

If, after parameter matching and proper training:

W1 ≈ or worse than generic SSM

on centroid/divergence metrics and gives no efficiency advantage.

Then use an SSM and stop forcing a physics interpretation onto it.

Continue aggressively

If:

W1 ≈ attention on short-horizon quality
W1 > generic SSM on rollout horizon
W1 << attention in memory/compute

Then immediately reformulate W1 as the recurrent spectral SSM and test 512/1024-frame streaming.

That is a genuinely interesting result.

4. Quick wins for the current 0.15 vs 0.09 gap

In order:

1. Residual prediction

Biggest easy win.

Python
Run
pred = last_frame + delta

Zero-init delta head.

Your model instantly starts from ~0.0155 instead of ~0.15 on the 8×8 generator.

2. Stop drawing conclusions from 40 training steps

The smoke is useful for catching explosions and shape errors.

It is not an optimization comparison.

Your normal script defaults to 2,000 steps. 

train_compare

Use convergence curves, not one checkpoint.

3. Add a short rollout loss

Even 4 steps should expose instability that one-step RGB MSE hides.

4. Fix the parameter/position mismatch

One compact positional representation at the input for both architectures.

Delete the gigantic per-layer attention positional tables.

5. Fix spatial FFT boundaries

Either:

zero-pad → FFT → crop

or use reflection-aware padding before spatial filtering.

Right now the FFT thinks the image is a torus while your balls bounce off walls.

6. Add cheap content gating

Something Hyena-like:

Python
Run
v = value_proj(x)
g = sigmoid(gate_proj(x))

y = output_proj(g * wave(v))

Very cheap, retains FFT parallelism, and directly attacks the fixed-filter weakness. Hyena's combination of long convolutions and data-controlled gating is precisely relevant here. 
arXiv

7. Add a local path

Parallel:

3×3 depthwise conv
+
global wave field

then fuse them.

The global field should model transport/coherence; don't force it to also reconstruct every local edge.

The architecture I would pursue

Not:

3D separable FFT

and not simply:

generic Mamba

I would pursue:

latent frame
    ↓
local spatial features
    ↓
2D FFT
    ↓
frequency-conditioned complex oscillator bank
λ(kx,ky)
    ↓
causal recurrent temporal state
    ↓
content gate
    ↓
inverse FFT / local refinement

That is much closer to a genuine learned wave-field video SSM.

It gives you the useful parts of all three ideas:

wave physics      → structured propagation bias
SSM recurrence    → constant memory in time
latent video      → tractable spatial scale

And it fixes the most important conceptual weakness in the current implementation: real motion couples space and time; your current kernel factorizes them.

That is the direction I would test before spending serious effort on a full latent flow-matching generator.