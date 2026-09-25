Share
You said:
RESULTS_BRIEF_2(1).md
Document
Deep dive request (retry). Attached: BRIEF with fresh results from our wave-field video model research. A motion-balanced objective lifted coherence at 16x16 (copy_ratio 0.72) but at 32x32 the SAME hyperparams collapsed toward freezing (copy_ratio 0.115). Fair-protocol renders show wave beating attention 5/5 seeds; pixel-MSE loses to freezing at 256 frames. Questions: (1) diagnose the scale collapse; (2) rank 2-3 next experiments; (3) sanity-check the protocol reconciliation. Plain markdown.
Worked for 22s
ChatGPT said:
Diagnosis

The strongest explanation is resolution-induced loss dilution, not yet a wave-kernel capacity failure.

Your g16 result is healthy:

1-step/copy = 0.434
copy_ratio = 0.719

At g32, with the same objective:

1-step/copy = 1.012
copy_ratio = 0.115
div_h = 0

So the model has crossed from learning useful motion into a regime where doing almost nothing is favored. 
RESULTS_BRIEF_2

The critical detail is that RADIUS=1.6 px and SPEED=1.15 px/frame remain fixed while the grid doubles from 16×16 to 32×32. 
RESULTS_BRIEF_2

1. Why g32 collapses
A. The moving signal becomes ~4× less important to a pixel-averaged loss

If the object's physical size stays fixed in pixels:

g16 total pixels = 256
g32 total pixels = 1024
moving object footprint stays roughly constant

Therefore the moving region occupies about one quarter the fraction of the image at g32.

Suppose the object affects ~8 pixels:

Plain text
g16: 8 / 256  = 3.1%
g32: 8 / 1024 = 0.78%

A loss averaged over the whole image therefore gets roughly 4× less gradient from the thing you actually care about.

That alone can change the optimum from:

predict motion accurately

to:

keep the 99% static background perfect and tolerate errors on the tiny object.

That is almost exactly the failure mode your copy_ratio=0.115 is describing.

B. MOVE_THRESH=0.05 is probably not resolution invariant

You report:

MOVE_THRESH = 0.05; g16 moving-frac 0.746

but there is no corresponding g32 moving fraction in the brief. 
RESULTS_BRIEF_2

This is the first diagnostic I would inspect.

If motion detection is based on something like:

Python
mean(abs(frame_t - frame_t1)) > MOVE_THRESH

then doubling resolution while leaving the moving object unchanged will sharply reduce that mean.

Even though the object moves by the same number of pixels, its contribution is divided by 4× as many pixels.

So a sequence that is "moving" at g16 may suddenly look "nearly static" to the loss at g32.

This creates a nasty feedback loop:

Plain text
larger grid
→ smaller global motion statistic
→ fewer pixels/frames qualify for motion emphasis
→ weaker motion gradient
→ copy/static solution becomes attractive
→ model predicts less motion
→ motion-related objective becomes even less useful

This is my highest-probability diagnosis.

C. SPEED is also effectively halved in normalized coordinates

The absolute displacement remains:

Plain text
1.15 px/frame

But relative to frame width:

Plain text
g16: 1.15 / 16 = 7.2% width/frame
g32: 1.15 / 32 = 3.6% width/frame

So the g32 task is not simply "the same experiment at higher resolution."

It is a different motion regime.

The object is:

relatively smaller
relatively slower
responsible for a smaller fraction of total MSE

All three changes bias pixel MSE toward persistence.

D. K-ramp can amplify the problem, but I doubt it initiated it

Your working g16 setup uses:

Plain text
motion loss + K ramp 3 → 8

and the same schedule is used at g32. 
RESULTS_BRIEF_2

If K increases rollout horizon / temporal training difficulty, then once the g32 model begins under-predicting motion, longer-horizon supervision can reinforce the static shortcut.

For example:

Plain text
weak 1-step motion learning
      ↓
K increases
      ↓
incorrect predicted motion compounds over K steps
      ↓
static prediction has lower average pixel error
      ↓
optimizer suppresses motion further

So I would regard K-ramp as a likely amplifier, not the root cause.

Don't tune K extensively before fixing the scale invariance of the objective.

E. Kernel capacity is possible, but your current experiment does not establish it

There is a conceivable architectural problem at g32.

A fixed family of wave kernels may have:

insufficient spatial bandwidth
inappropriate decay lengths
frequency parameters whose meaning changes with resolution
too few modes/heads for the larger spatial spectrum

But right now objective scaling is a much simpler explanation.

The collapse is too aligned with the arithmetic of sparse-motion pixel MSE to conclude:

wave fields don't scale to 32×32.

You haven't isolated that yet.

Ranked next experiments
1. Make the objective resolution invariant

Most informative experiment by far.

Keep the g32 data exactly as it is.

Do not scale radius or speed yet.

Instead replace globally averaged motion supervision with a motion-normalized loss.

Conceptually:

Python
delta = abs(target - previous_frame)

motion_mask = delta > pixel_threshold

L_static = mse(pred[~motion_mask], target[~motion_mask])
L_motion = mse(pred[motion_mask], target[motion_mask])

loss = L_static + lambda_motion * L_motion

The important part is:

Plain text
mean over moving pixels

rather than:

Plain text
sum motion error / all 1024 pixels

A softer version is preferable if the mask is unstable:

Python
w = delta / (delta.mean() + eps)
w = clamp(w, 1, w_max)

L_motion = mean(w * (pred-target)**2)

Or explicitly correct for occupancy:

Python
motion_frac = motion_mask.float().mean()

lambda_eff =
    lambda_base * reference_motion_frac /
    clamp(motion_frac, min_motion_frac)
Also make MOVE_THRESH relative

Don't use one fixed global scalar whose meaning changes with resolution.

Better possibilities:

Plain text
threshold = q-th percentile of frame-difference magnitude

or normalize the motion statistic:

Python
motion_score =
    delta.sum() / (moving_area_estimate + eps)

or derive the mask locally per pixel rather than from global mean movement.

Training schedule

For this run I would also freeze K initially:

Plain text
K = 3

until the g32 one-step motion metric clearly learns.

Only then:

Plain text
3 → 5 → 8

Make the ramp conditional on learning rather than epoch number.

For example:

Plain text
advance K only when
copy_ratio > ~0.5
and validation motion error is still improving

I would not obsess over the exact 0.5 threshold. The principle matters.

What would falsify this diagnosis?

If g32 still freezes despite:

strong motion-mask activation
comparable motion-gradient magnitude to g16
K fixed low

then architecture/capacity becomes much more plausible.

2. Run a resolution-equivalent data control

This one is extremely cheap and scientifically clean.

At g32 use:

Plain text
RADIUS = 3.2 px
SPEED  = 2.30 px/frame

instead of:

Plain text
RADIUS = 1.6
SPEED  = 1.15

Now the normalized geometry approximately matches g16:

Plain text
radius / width
speed / width

Keep everything else identical.

You now have an excellent A/B:

Plain text
g32-A
radius 1.6
speed 1.15

g32-B
radius 3.2
speed 2.30

If B suddenly recovers toward the g16 behavior, you've learned something major:

the collapse was predominantly task/loss scaling, not inability of the wave model to operate at 32×32.

I'd actually expect a substantial recovery.

This experiment is diagnostically useful even if this isn't the dataset you ultimately want.

3. Only then test wave capacity / spatial scaling

If experiment 1 fixes the loss but g32 remains materially worse than g16, test architecture.

Don't just make the model indiscriminately bigger.

Test whether the kernel parameterization scales correctly with resolution.

The sharp experiment is something like:

Plain text
A: current wave kernel
B: same parameter count, but wavelengths/decays parameterized
   in normalized spatial coordinates
C: multi-scale wave bank

For normalized coordinates, make the learned length scale refer to fraction-of-image rather than raw grid cells.

Conceptually:

Python
x_norm = x / W
y_norm = y / H

and define wavelength/decay in that space.

A multi-scale bank could explicitly have:

Plain text
short-range field
medium-range field
global field

rather than asking one family of damped cosines to span everything.

But I would spend zero extra architecture complexity until experiments 1 and 2 are done.

The experiment matrix I'd actually run

You can extract a lot from four short g32 runs:

Run	Motion loss	Radius/speed	K
A	current	1.6 / 1.15	3→8
B	normalized	1.6 / 1.15	fixed 3
C	current	3.2 / 2.30	fixed 3
D	normalized	1.6 / 1.15	3→8 after learning

Interpretation:

Plain text
B works:
objective dilution / threshold was dominant.

C works:
relative scene scale was dominant.

B + C both fail:
start suspecting wave capacity / optimization.

B works but D fails:
K curriculum is the problem.

B and D work:
you basically have the diagnosis nailed.

That is much stronger than randomly adjusting λ, heads, or kernel count.

Protocol reconciliation

I think your reported results are internally compatible, with one important qualification.

You have:

Plain text
wave:      0.074–0.082
attention: 0.146–0.183
frozen:    0.043

over 256-frame model MSE, with wave beating attention on all five seeds. 
RESULTS_BRIEF_2

And:

Plain text
curve mean:  0.0777
render mean: 0.0819

for the same seed family. 
RESULTS_BRIEF_2

Those statements do not conflict.

They mean:

Plain text
wave >> attention

for the learned-model comparison,

while simultaneously:

Plain text
freeze > wave > attention

under long-horizon pixel MSE.

Both can be true.

The 0.0777 vs 0.0819 difference looks benign

That's roughly:

Plain text
0.0042 absolute
~5% relative

Given that they are "same seed family" rather than necessarily the identical exact aggregation procedure, I would not regard that as evidence of protocol contamination.

It actually gives reasonable evidence that the render and evaluation pipelines are measuring the same behavior.

I'd verify only:

Plain text
same normalization
same target frames
same autoregressive rollout path
same starting context
same frame count
same loss reduction
same channels
same seeds or explicitly documented seed mismatch

If those agree, reconciliation looks sound.

But the frozen baseline changes what you can claim

This is the important scientific distinction.

Your result supports:

Under the matched learned-model protocol, wave outperforms attention consistently across 5/5 tested seeds.

It does not yet support:

The wave model is a better long-horizon video predictor.

Because frozen-frame MSE is still:

Plain text
0.043

versus wave around:

Plain text
0.08

on 256-frame rollout. 
RESULTS_BRIEF_2

That is not a protocol contradiction.

It exposes a metric/task pathology.

Pixel MSE rewards persistence heavily when most pixels do not change.

The model can generate something that is more dynamically correct while receiving a worse MSE than:

Python
pred[t+1] = pred[t]
You need two classes of evaluation now

Don't abandon MSE. Keep it because it's useful.

But separate:

Fidelity
Plain text
full-frame MSE
PSNR

from:

Motion correctness

At minimum add:

Plain text
moving-pixel MSE
centroid trajectory error
velocity error
direction/angular error
temporal-difference error

For this synthetic dataset, centroid trajectory error is particularly valuable because you know exactly what is supposed to move.

Example:

Plain text
position error at t
velocity error at t
trajectory ADE
trajectory FDE

Then you may discover:

Plain text
Frozen:
excellent pixel MSE
terrible trajectory error

Wave:
worse global MSE
much better trajectory

Attention:
worse on both

That would reconcile the apparent paradox immediately.

One metric I'd add immediately

Report everything relative to the copy baseline.

For horizon t:

Plain text
relative_skill(t) =
1 - MSE_model(t) / MSE_copy(t)

Interpretation:

Plain text
> 0     model adds predictive value
= 0     equivalent to freezing
< 0     worse than freezing

You can calculate the same thing on the moving mask.

Then your plots become much harder to misinterpret.

One concern with copy_ratio

Be very explicit about what it means.

Right now:

Plain text
g16 copy_ratio = 0.719
g32 copy_ratio = 0.115

while the separate line says:

Plain text
1-step/copy:
g16 = 0.434
g32 = 1.012

Those values clearly are not simply the same ratio expressed twice. 
RESULTS_BRIEF_2

That's okay if they measure different things, but name them very carefully in the eventual paper/report.

For example:

Plain text
one_step_MSE / copy_MSE

versus

Plain text
fraction of predicted motion relative to GT motion

If that is indeed what they mean.

Otherwise readers are going to assume the two "copy" quantities should agree.

My current diagnosis

Approximately:

Plain text
Primary:
resolution-dependent motion-loss dilution / thresholding

Secondary:
K-ramp locking in the low-motion shortcut

Tertiary:
wave-kernel capacity / scale parameterization

The strongest clue is not merely that g32 got worse. It is that radius and speed stayed fixed in pixel units while image area increased 4×. 
RESULTS_BRIEF_2

That changes the optimization problem dramatically.

So I would not touch the architecture next.

Run the resolution-invariant motion objective at g32, hold K=3, and instrument three quantities during training:

Plain text
motion-mask fraction
gradient norm from motion loss vs reconstruction loss
predicted motion magnitude / GT motion magnitude

If motion gradient at g32 is currently ~¼ of the effective g16 signal, you've probably found the collapse mechanism before the next full training run even finishes.
