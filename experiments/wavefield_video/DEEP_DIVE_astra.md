Bottom line

The next 48 hours should not be spent finishing the current g32 matrix unchanged**.** The 32² SSM collapsing to copy-last means the present loss/evaluation allows a degenerate solution, so a larger sweep would mostly measure which model exploits that shortcut best.

The strongest path is:

causal dispersion kernel → exact recurrent form → prove 1024-frame constant-memory equivalence → fix the persistence/copy-last confound → add signed spatial-frequency dispersion → only then test richer modes/content adaptivity.

The recurrence is especially important because this is not an approximation. A damped oscillatory kernel is naturally a state-space recurrence, which is exactly the convolution/recurrent duality exploited by S4/S5-style models. 
arXiv
+1

One caveat: the separate new RESULTS BRIEF itself did not surface as a readable attachment on my side, so I am using the exact facts you gave here, plus the existing wave-model context, and I will not invent numerical g16/g32 deltas.

1. Ranked upgrades
Rank	Upgrade	Hardening payoff	Quality payoff	Do now?
1	Exact causal streaming recurrence for the dispersion kernel	Critical. Converts temporal memory from O(T) to O(1) state and gives a clean 512/1024-frame claim	Neutral if implemented exactly	Immediately
2	Remove copy-last as a valid optimum with motion-balanced loss + persistence metrics	Makes g32 comparison trustworthy	High, especially for SSM	Immediately
3	Signed k-dependent dispersion rather than cosine-only propagation	Stable, interpretable dynamics	Very high for motion	Immediately after recurrence works
4	K=2–4 dispersion modes/head	Still constant-memory	High for multi-speed/multi-timescale motion	Yes, K=2 then K=4
5	Input-dependent injection/readout gating	Preserves stable recurrent transition	Medium-high for collisions/occlusion/nonstationary motion	After K modes
6	Latent-space representation	Massive spatial scaling	Necessary eventually for real video	Not during first proof
7	Full content-dependent transition / hypernetwork	More expressive	Potentially high	Defer
8	Bigger dense attention runs	Almost no useful hardening value	Benchmark only	Cut
Why #2 matters so much

The g32 SSM result is probably telling you at least as much about the objective as the SSM.

For mostly-static frames,

predict next frame ≈ copy current frame

can get deceptively good pixel MSE because most pixels did not change.

So every result should include the explicit persistence baseline:

y_copy[t+1] = x[t]

and training should stop rewarding it.

For the synthetic experiments I would use:

E = (prediction - target)^2

moving = abs(target - last_frame) > threshold
static = ~moving

L =
    0.50 * mean(E[moving])
  + 0.50 * mean(E[static])
  + 0.25 * MSE(
        prediction - last_frame,
        target - last_frame
    )

That makes moving pixels roughly as important as the entire static background.

Also report:

copy_ratio =
    ||prediction - last_frame|| /
    (||target - last_frame|| + eps)

Interpretation:

copy_ratio ≈ 0      collapsed to persistence
copy_ratio ≈ 1      roughly correct motion magnitude
copy_ratio >> 1     unstable / excessive motion

Do not interpret the SSM result until it has been rerun under this objective.

2. Constant-memory streaming experiment
The kernel

Suppose one dispersion mode has causal temporal impulse response

h[τ,k] = g(k) ρ(k)^τ cos(τ θ(k) + φ(k)),    τ >= 0

where:

ρ(k) = exp(-α(k))
0 < ρ(k) < 1

θ(k) = temporal phase increment at spatial frequency k

For the original non-dispersive version, ρ and θ can simply be per-head scalars.

For a true dispersion model they become functions of spatial frequency k=(kx,ky).

Exact two-state recurrence

Take each incoming frame feature map after the input projection:

x_t : [B, H, W, D]

Spatial FFT:

u_t(k) = FFT2(x_t)

For every head/mode/frequency/channel maintain two states:

c_t(k)
s_t(k)

Update:

c_t =
    ρ(k) [
        cos θ(k) * c_{t-1}
      - sin θ(k) * s_{t-1}
    ]
    + b(k) * u_t

s_t =
    ρ(k) [
        sin θ(k) * c_{t-1}
      + cos θ(k) * s_{t-1}
    ]

Readout:

v_t(k) =
    g(k) [
        cos φ(k) * c_t
      - sin φ(k) * s_t
    ]

For K modes:

v_t(k) = Σ_m v_t,m(k)

Then:

y_t = IFFT2(v_t)

and continue through the output projection/residual block.

This gives exactly:

v_t =
Σ_{τ=0...t}
g b ρ^τ cos(τθ + φ) u_{t-τ}

So the recurrent implementation and the causal convolution implementation are mathematically the same operator.

That convolution/recurrent duality is the core reason structured state-space models can train using parallel convolution-like forms and run sequentially using fixed state. 
arXiv
+1

Important distinction

This only exactly replaces a causal, one-sided linear convolution.

If any checkpoint was trained using:

symmetric temporal kernel
circular FFT wraparound
future-frame mixing

you cannot simply export that checkpoint into this recurrence and call it equivalent.

The training operator needs to be causal first.

Stable parameterization

Do not learn unrestricted ρ.

Parameterize the memory in terms of half-life:

half_life = h_min + (h_max - h_min) * sigmoid(raw_h)

ρ = exp(-ln(2) / half_life)

For the 1024-frame test I would use roughly:

h_min = 2 frames
h_max = 4096 frames

This guarantees:

0 < ρ < 1

while still permitting ~1000-frame modes.

Keep recurrence state in FP32, even if projections/MLPs run BF16.

Near-unit-circle states are exactly where I would not save a few megabytes by using BF16.

The quality upgrade: signed dispersion

The cosine kernel has a deeper limitation.

A cosine contains both:

+ω
-ω

components.

So it naturally resembles two counter-propagating waves. That is not ideal for directional translation.

For an object moving at velocity:

v = (vx, vy)

Fourier-domain translation obeys approximately:

X_{t+1}(k)
=
exp(-i k·v) X_t(k)

Therefore the more interesting recurrence is:

z_t(k) =
ρ(k) exp(-i θ(k)) z_{t-1}(k)
+ B(k) u_t(k)

with

θ(k) = kx vx + ky vy

or a learnable generalization:

θ_m(k) =
kx vx,m
+ ky vy,m
+ δθ_m(k)

That directly represents motion as phase propagation.

To preserve a real spatial output enforce:

ρ(-k) = ρ(k)
θ(-k) = -θ(k)

and the corresponding conjugate symmetry of B/C.

I would expect this change to matter more for quality than simply increasing channel count. The current wave hypothesis should live or die on whether its spectral dynamics actually represent translation well.

Frequency-dependent damping

Use an even function such as:

α(k) =
softplus(
    a0
  + ax kx²
  + ay ky²
  + axy kx ky
)

then:

ρ(k) = exp(-α(k))

This lets coarse structures have long memory while high-frequency detail decays more quickly.

Do not start with a large MLP over frequencies. Low-order parameterization first.

State size

With:

grid = 32²
dim = 384
layers = 8
spatial rFFT = 32 × 17 frequencies
FP32 complex state

the approximate recurrent-state footprint for the two-quadrature form is:

K=1   ~25.5 MiB total across 8 layers
K=2   ~51 MiB
K=4   ~102 MiB

per generated sample, excluding weights and ordinary one-frame working buffers.

Most importantly:

512 frames  → same recurrent state
1024 frames → same recurrent state
4096 frames → same recurrent state

Only generation time grows.

Exact verification battery

Run five equivalence tests before training anything new.

A. Impulse response

Input:

u[0] = 1
u[t>0] = 0

Compare recurrence against explicitly constructed:

h[t] = ρ^t cos(tθ + φ)

for at least 4096 steps.

Targets:

FP64 CPU:
max_abs_error < 1e-9

FP32 CUDA:
relative_L2 < 1e-5
max_abs_error preferably < 1e-4

Do not use BF16 for this test.

B. Random sequence equivalence

Generate random u[0:1024].

Compare:

streaming recurrence

against:

explicit causal convolution

and separately:

zero-padded temporal FFT convolution

All three should agree.

C. Chunk invariance

Process the identical 1024-frame sequence as:

1024 × 1-frame calls
8 × 128-frame chunks
64 × 16-frame chunks
1 × 1024 reference convolution

carrying state between chunks.

Outputs must agree.

This catches a huge class of bad state-reset and indexing bugs.

D. Gradient equivalence

At short T=17 only:

FFT causal implementation
vs
unrolled recurrent implementation

compare gradients for:

ρ
θ
φ
pi
po

Use FP32.

If relative gradient error is meaningfully larger than ~1e-4, investigate before using recurrence for training.

E. Constant-memory proof

Measure:

torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()

for:

T = 128
256
512
1024

Important: do not store generated GPU frames in a Python list, or your supposedly constant-memory benchmark will itself become O(T). Compute metrics online and move optional samples to CPU.

Pass condition:

streaming peak VRAM:
T=1024 within ~5% of T=128

after warm-up.

Also benchmark:

median ms/frame
p95 ms/frame
frames/sec

The S4 family explicitly targets fixed-size recurrent state for long sequence inference, so this is the claim worth demonstrating rather than merely another short-sequence FLOP comparison. 
arXiv

Rollout benchmark

Do two synthetic suites, not one.

Transport suite

Designed to favor the core hypothesis fairly:

1–4 objects
random signed velocities
multiple speeds
wall reflection
acceleration
different sizes/textures
crossing trajectories
occlusion without unpredictable randomness

This answers:

Can dispersion actually preserve motion better than a generic state model?

Interaction suite

Harder:

object-object collisions
temporary occlusion
velocity changes triggered by visible events
depth/order changes
boundary changes

Do not use unseen random velocity kicks.

If the future contains randomness that is not observable from the inputs, a deterministic model cannot predict the ground truth by construction. That muddies the architecture comparison.

If you want kicks, expose a control/event telling the model the kick occurred.

Metrics

The primary scoreboard should be:

Metric	What it catches
One-step MSE	Basic fitting
Motion-balanced MSE	Defeats static-background cheating
Copy-last MSE	Persistence baseline
Copy ratio	Explicit collapse detector
Centroid trajectory RMSE	Actual dynamics
Velocity error	Phase/motion correctness
Divergence horizon	Long-rollout stability
State norm vs time	Recurrent instability
Peak VRAM	Constant-memory claim
ms/frame	Streaming practicality

Define divergence horizon using a task quantity rather than arbitrary raw MSE.

For example:

first frame where centroid error > object radius
for 8 consecutive frames

Report median horizon across sequences.

Then separately report:

H_50
H_90

where H_50 is median divergence frame and H_90 is the 90th-percentile failure frame.

3. Concrete 48-hour plan on one RTX 4090
Hours 0–6: stop the experiment from lying

Do not continue the current g32 jobs yet.

Implement:

copy-last baseline
motion-balanced loss
copy_ratio metric
trajectory/velocity metrics
deterministic transport suite

Then run existi