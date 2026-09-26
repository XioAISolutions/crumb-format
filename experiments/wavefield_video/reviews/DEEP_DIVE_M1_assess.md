Yes, there is real evidence here, but I would call M1 a successful mechanism transfer, not yet proof that the plugin solves general long-video coherence.

I inspected the 256-frame before/after. The effect is real but visually subtle at 16×16. More importantly, the metrics confirm that the plugin is changing the low-frequency behavior while leaving most higher-frequency structure alone: luminance-band variance drops 36.85%, centroid variance 25.58%, high-band flicker stays at 0.973×, and HF-SSIM is 0.9902. 
m1_metrics
 The file itself also explicitly confirms the biggest caveat: at 16×16, r < 0.03 contains only DC, so complex_mc == complex here. 
m1_metrics

1. Is the evidence convincing?
For the narrow claim: yes

The evidence now supports:

A tiny persistent spectral state can suppress a specific long-horizon low-frequency instability in an unseen generator rollout without behaving like generic temporal smoothing.

The EMA/median controls are useful because they attack the obvious criticism: “you just made the video steadier by damping everything.”

Your plugin instead gets meaningful low-band stabilization while retaining approximately 99% HF structural similarity. The literature also supports the general mechanism class: temporal phase variation in complex frequency representations really does encode motion and can be selectively attenuated without optical flow. 
Google Research
+1

But the strongest claim is still unsupported

You cannot yet say:

“We demonstrated k-space phase stabilization of spatial drift in a real generated video.”

Because M1 did not exercise that mechanism.

At 16×16:

1
/
16
=
0.0625
 cycles/pixel
1/16=0.0625 cycles/pixel

while your band is <0.03.

So there are no non-DC spatial coefficients available.

Your own metrics file says exactly that. 
m1_metrics

What M1 demonstrated is closer to:

persistent spectral-state stabilization transfers from synthetic tests to a generator rollout, on the DC/luminance axis.

That's still worthwhile. Just don't overstate it.

One thing I would revise from your oracle interpretation

I would not conclude yet that:

“the correlation estimator is better than ground-truth displacement because pixel-centre truth is noisy.”

That's possible, but the experiment doesn't uniquely establish it.

A more conservative interpretation is:

The band-limited estimator is better aligned with your DSM objective than geometric pixel-centre displacement is.

Those are different statements.

If your nuisance exists as a sampled/band-limited signal, then the displacement that minimizes spectral-state inconsistency does not necessarily equal the literal continuous-space center displacement.

So your “oracle” was an oracle for geometric translation, not necessarily an oracle for the plugin objective.

The fact that oracle component actively makes DSM worse strongly suggests exactly this objective/model mismatch.

I'd actually rename these internally:

Plain text
geometric-displacement control
known-component replacement

rather than “oracle.”

That removes a confusing conceptual contradiction where the algorithm supposedly beats perfect information.

2. Strongest single falsification test you haven't run

This is the big one:

Legitimate low-frequency motion trap

Give the plugin a video with:

NO nuisance drift at all, but with intentional content occupying exactly the frequencies the plugin targets.

For example:

one large soft object slowly translating across the frame
global camera pan
gradual illumination ramp
slow zoom
ideally small independently moving objects simultaneously

Then freeze the shipping configuration.

No tuning.

The plugin should do essentially nothing.

That is the test.

Right now all successful tests contain something you want reduced.

Therefore a skeptical reviewer can still say:

“This is a clever low-frequency variance suppressor.”

Your temporal EMA and median controls do not eliminate that explanation because they're broadband temporal suppressors.

The dangerous alternative hypothesis is much narrower:

Crumb suppresses low-frequency temporal change whether that change is pathological or semantically correct.

That's exactly what you need to kill.

Suggested falsification sequence

At 32×32 or preferably 64×64:

Plain text
large Gaussian object:
x(t) = x0 + 0.15 px/frame

small balls:
normal independent trajectories

brightness:
slow intentional sinusoidal modulation

NO injected wander

Run the frozen:

Plain text
complex_mc
band = 0.03
strength = 0.5
corr
hard edge

Measure:

𝐸
traj
=
∥
𝑥
^
plugin
(
𝑡
)
−
𝑥
clean
(
𝑡
)
∥
trajectory range
E
traj
	​

=
trajectory range
∥
x
^
plugin
	​

(t)−x
clean
	​

(t)∥
	​


plus motion amplitude ratio and HF fidelity.

My kill criterion would be roughly:

Plain text
intentional centroid-motion retention > 95%
low-frequency trajectory distortion < 5%
HF SSIM > 0.98

And crucially:

Plain text
plugin should NOT significantly reduce clean low-band variance

unless that variance is actually inconsistent with the state model.

Why I rank this above another dirty-video test

Because if it fails, the product claim gets much weaker.

Then you have built:

“a frequency-selective stabilizer”

rather than:

“a coherence engine that distinguishes drift from intended evolution.”

That distinction matters enormously.

Then run the real spatial M1

Immediately after that, I would repeat the generator test at 32×32 or 64×64.

Right now M1 does not test the phase/translation portion of complex_mc.

The first non-DC radial frequency must actually fall inside the active band.

Otherwise the signature part of the architecture remains validated only synthetically.

3. Current video as product-page material

I would not use the existing clip as the hero demonstration.

I've looked across the clip. At 16×16, the raw and corrected sides mostly look extremely similar frame-to-frame, which is scientifically encouraging but commercially weak.

A layperson won't know what to look for.

The result risks producing:

“Which one is supposed to be better?”

That's fatal for a product demo.

Make the failure visible, not the algorithm

Create one simple demo with:

LEFT — Normal generation
RIGHT — + Crumb

and choose footage where raw behavior develops an unmistakable long-horizon defect.

For example:

Plain text
0–10 sec:
both look identical

10–30 sec:
raw background / lighting / large structure begins wandering

30–60 sec:
raw accumulated drift becomes obvious

Crumb:
subject motion continues normally
environment remains anchored

The viewer should understand it with the sound off and without reading a metric.

Add one visual aid

Below both videos, display a tiny live trace:

Plain text
Scene drift

RAW     ~~~~~~/\/\/~~~~~~
CRUMB   ────────────────

Or, even stronger, place a faint fixed crosshair/reference outline over an object that should stay stationary.

Then the user literally sees:

Plain text
RAW    object wanders away from reference
CRUMB  object stays centered

Do not start the product page with FFTs, phase plots, rho, or spectral language.

Lead with:

Long videos drift. Crumb remembers.

Then:

A tiny state that runs beside your generator and suppresses accumulated scene drift without freezing motion.

Then show:

36.9% less low-frequency instability
99.0% detail fidelity
kilobytes of persistent state

But only use the numbers once the benchmark supports the exact marketing interpretation.

4. Red flags

There are five I would keep front-and-centre.

1. M1 is DC-degenerate

This is currently the biggest technical limitation.

Your luminance results are strong:

low-band magnitude variance −36.85%
centroid variance −25.58%
HF-SSIM .9902

m1_metrics

But your hotspot path is much weaker on centroid variance, only 6.86%, because the intended spatial band does not exist at this resolution. 
m1_metrics

Don't mix those two findings together.

2. Your success metrics are partly self-referential

Reducing:

Plain text
lowband variance
centroid variance

is exactly what the plugin tries to do.

So those establish effect, not necessarily correctness.

You need a metric representing:

Did the video retain what it was supposed to do?

Your HF metric helps, but HF says little about legitimate slowly moving large structures.

This is why the clean low-frequency-motion falsification test matters.

3. High-frequency preservation is necessary but insufficient

You can have:

Plain text
HF-SSIM = .99

while destroying semantically important low-frequency motion.

A person's entire torso moving slowly, a camera pan, a face changing position, or illumination changes may primarily inhabit low/mid bands.

So eventually you need a motion-compensated temporal metric or tracked semantic trajectory metric alongside HF-SSIM.

Optical-flow warping error with non-occlusion masks is already a standard type of temporal-consistency evaluation in video work. 
Wiley Online Library

4. Hard-edge success still makes me cautious

Your winning M0 setting being:

Plain text
hard edge

is interesting because ordinarily sharp spectral boundaries can create spatial ringing or leakage.

It may be completely fine because:

correction amplitude is small,
the band is tiny,
and your sampled synthetic distribution happens to tolerate it.

But I would explicitly test:

Plain text
impulse
hard edge
thin line
high-contrast moving boundary

and measure ringing around those structures.

Not because I think it kills the design, but because it's exactly the sort of thing that appears later at higher resolution.

5. Don't let the synthetic benchmark become the specification

The most interesting thing in your new results is actually that the geometric “oracle” loses.

That means DSM is measuring something slightly different from the intuitive physical notion of drift.

That can be good.

But it also means you need to be careful that you're not optimizing:

metric coherence
metric coherence

instead of:

perceptual / semantic coherence
.
perceptual / semantic coherence.

The clean intentional-motion trap is the fastest way to find out.

Where I think you are

I'd mark the program:

Plain text
M0 — mechanism:
PASS

M1a — generator transfer / DC:
PASS

M1b — real rollout spatial phase correction:
NOT TESTED

M1c — legitimate-motion non-interference:
NOT TESTED

The next test should not be “can we get an even bigger drift reduction?”

It should be:

Can frozen Crumb see strong legitimate low-frequency evolution and leave it alone?

If that passes at 32/64² and then complex_mc also reduces genuine spatial drift on real rollouts while preserving independent motion, you've crossed a much more meaningful threshold. At that point the claim starts changing from “interesting spectral filter” to credible generator-agnostic temporal-coherence sidecar.