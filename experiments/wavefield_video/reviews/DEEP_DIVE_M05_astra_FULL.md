Share
You said:
RESEARCH BRIEF M0.5 (for expert consult)

SYSTEM: an ultra-compact spectral coherence plugin (k-space state, kilobytes)
that runs alongside any frame generator. State = damped per-cell complex
amplitude+phase. Engines so far: (i) dc_only, (ii) magnitude (amplitude-only
anchoring), (iii) complex (slow phase EMA anchoring, rho=0.995), (iv) complex_mc
(motion-compensated re-centering of low-band phase, new).

SCENARIO (synthetic, g32/141 frames): a smooth low-band bump is injected and
its position WANDERS (drift, lowband_pos_var ~= 44 px^2). Legitimate content:
small balls with normal motion. Metrics: drift%pos = variance drop of the
low-band energy position; hf_ref = high-pass SSIM of corrected vs CLEAN control
(detail fidelity vs truth); hf_chg = same vs input (diagnostic); motion% =
tracker motion preservation.

MEASURED FRONTIER (all owner-run):
  complex rho=0.995        50.2%  hf_ref 0.9946  motion 99.7
  complex rho=0.997        48.2%  hf_ref 0.9945
  complex rho=0.999        45.9%  hf_ref 0.9943
  mc 0.5 lowband-only      70.2%  hf_ref 0.9346  (hf_chg 0.9341)
  mc 0.5 taper=True        41.6%  hf_ref 0.9929
  mc 0.85 lowband-only     59.4%  hf_ref 0.9297
GATES: drift%pos >= 60 AND hf_ref >= 0.98 AND |motion-100| <= 5. No setting
clears both primary gates.

HYPOTHESES:
H1 (band inconsistency): mc shifts low-band only -> ghosts on every edge ->
    hf_ref damage. Fix: broadband-consistent transform.
H2 (scale separation): the bump lives at the LOWEST band; balls at mid band.
    A per-radial-band phase correction selective to the bump's band should
    remove wander without touching ball detail (lineage: phase-based motion
    processing, Wadhwa et al.; per-scale phasor analysis).
H3 (metric semantics): under subpixel translation, pixel-level HF SSIM
    decorrelates even for perfect consistency (floor ~0.9x) - so the right
    "S" metric for translation-like corrections may need band-limited or
    flow-aware comparison.

QUESTIONS:
1. Which mechanism (math sketch welcome) can reach drift>=60 with hf_ref>=0.98
   and motion preserved, given these measurements? Consider per-band phase
   correction, phase correlation + trajectory filtering (video stabilization
   lineage: L1-optimal paths, Bundled Camera Paths/content-preserving warps),
   or other. What is the minimal-change variant of our engine that should work?
2. Is H3 correct? If the metric floor is real, what is the principled fidelity
   metric for judging a translation-type correction (so we do not falsely fail
   a perfect correction)?
3. What single decisive experiment would falsify the winning mechanism fastest?
…
Show more
Worked for 1m 8s
ChatGPT said:

Your data points to a fairly specific answer:

The next mechanism I would bet on is not stronger complex_mc. It is a two-stage “estimate narrowly, correct coherently” engine: estimate the nuisance displacement from the very-low-frequency phase, but apply the correction through a smooth, reconstruction-consistent spectral band that covers the bump’s actual support.

That is the smallest architectural change that plausibly gets you across both gates.

1. Mechanism most likely to clear 60% drift + 0.98 HF

Your frontier tells us something important:

Engine	Drift reduction	HF fidelity
complex .995	50.2%	0.9946
MC .5 lowband	70.2%	0.9346
MC .5 taper	41.6%	0.9929

That is almost a textbook bias between correction strength and spectral consistency.

The taper result is particularly diagnostic. It largely fixes fidelity, from 0.9346 → 0.9929, so H1 is very likely substantially correct. But it attenuates the very coefficients needed to move the nuisance, so drift control collapses.

I would build this next

Call it something like:

phase_path_band

Pipeline:

Plain text
frame
  ↓
very-low-k phase measurement
  ↓
robust displacement estimate Δ_t
  ↓
causal trajectory filter
  ↓
desired correction δ_t
  ↓
flat-top spectral correction mask
  ↓
smooth overlap / partition-of-unity reconstruction
  ↓
corrected frame

Critically:

The measurement band and correction band should NOT be the same mask.

I suspect that is the conceptual mistake in the current MC implementation.

A. Estimate translation from phase, not from shifted pixels

For a translated component

𝑓
(
𝑥
−
Δ
)
f(x−Δ)

the Fourier transform becomes

𝐹
𝑡
(
𝑘
)
=
𝐹
0
(
𝑘
)
𝑒
−
𝑖
𝑘
⊤
Δ
𝑡
.
F
t
	​

(k)=F
0
	​

(k)e
−ik
⊤
Δ
t
	​

.

Therefore relative phase against your state is approximately

𝜙
𝑡
(
𝑘
)
=
∠
(
𝐹
𝑡
(
𝑘
)
𝑍
𝑡
(
𝑘
)
‾
)
≈
−
𝑘
𝑥
Δ
𝑥
𝑡
−
𝑘
𝑦
Δ
𝑦
𝑡
.
ϕ
t
	​

(k)=∠(F
t
	​

(k)
Z
t
	​

(k)
	​

)≈−k
x
	​

Δx
t
	​

−k
y
	​

Δy
t
	​

.

Instead of physically re-centering the low-band and then measuring it, solve:

Δ
^
𝑡
=
arg
⁡
min
⁡
Δ
∑
𝑘
∈
𝐵
measure
𝑤
𝑘
 
𝜌
(
wrap
⁡
[
𝜙
𝑡
(
𝑘
)
+
𝑘
⊤
Δ
]
)
Δ
^
t
	​

=arg
Δ
min
	​

k∈B
measure
	​

∑
	​

w
k
	​

ρ(wrap[ϕ
t
	​

(k)+k
⊤
Δ])

with:

amplitude/coherence weights w_k
Huber loss or trimmed residuals
lowest useful non-DC frequencies
reject cells with insufficient magnitude

At 32×32, I actually prefer this weighted phase-plane fit over conventional phase-correlation peak finding. You have very few meaningful low-frequency bins, so extracting a subpixel shift from the phase slope uses them efficiently.

Phase variations in complex multiscale coefficients are directly related to motion, which is the foundation of the Wadhwa phase-processing work. 
MIT CSAIL
+1

B. Filter the trajectory, not the pixels

You don't really want

Δ
𝑡
→
0.
Δ
t
	​

→0.

You want to separate undesirable wandering from permissible trajectory.

Maintain only a few scalars:

Plain text
position: x, y
velocity: vx, vy
confidence

For M0.5 I would use a robust causal α-β filter rather than L1 stabilization.

Something approximately:

𝑝
𝑡
−
=
𝑝
𝑡
−
1
+
𝑣
𝑡
−
1
p
t
−
	​

=p
t−1
	​

+v
t−1
	​

𝑟
𝑡
=
𝑝
^
𝑡
−
𝑝
𝑡
−
r
t
	​

=
p
^
	​

t
	​

−p
t
−
	​

𝑝
𝑡
=
𝑝
𝑡
−
+
𝛼
𝑟
𝑡
p
t
	​

=p
t
−
	​

+αr
t
	​

𝑣
𝑡
=
𝑣
𝑡
−
1
+
𝛽
𝑟
𝑡
v
t
	​

=v
t−1
	​

+βr
t
	​


and define your stabilized nuisance path

𝑝
𝑡
∗
=
𝐿
𝑃
(
𝑝
𝑡
)
p
t
∗
	​

=LP(p
t
	​

)

with correction

𝛿
𝑡
=
𝑝
𝑡
∗
−
𝑝
𝑡
.
δ
t
	​

=p
t
∗
	​

−p
t
	​

.

Why not jump to L1-optimal paths?

Because Grundmann's approach is excellent when you are stabilizing a camera trajectory, where the path itself is the object being optimized. 
Google Research

Here you have a tiny online sidecar plugin, plus legitimate independently moving objects.

Likewise Bundled Camera Paths is powerful specifically because it models spatially varying camera motion using multiple trajectories/mesh warping. 
Microsoft
+1

Both are heavier than your problem demands.

Borrow their philosophy, not their machinery:

estimate motion → regularize trajectory → synthesize a consistent correction.

C. Use a flat-top correction band, not the current taper

This is probably the most important implementation detail.

You currently appear to have two bad choices:

Plain text
hard low-band:
████████|_____
strong correction
bad seam

versus

Plain text
taper:
████▓▒░_______
good seam
insufficient correction

Instead:

Plain text
        bump support
      <------------>
____░▒████████▒░____
    smooth   smooth
    edge     edge

Use

𝑊
(
𝑘
)
=
{
1
	
𝑟
<
𝑟
1


1
2
[
1
+
cos
⁡
(
𝜋
𝑟
−
𝑟
1
𝑟
2
−
𝑟
1
)
]
	
𝑟
1
≤
𝑟
<
𝑟
2


0
	
𝑟
≥
𝑟
2
.
W(k)=
⎩
⎨
⎧
	​

1
2
1
	​

[1+cos(π
r
2
	​

−r
1
	​

r−r
1
	​

	​

)]
0
	​

r<r
1
	​

r
1
	​

≤r<r
2
	​

r≥r
2
	​

.
	​


Then apply

𝐹
𝑡
′
(
𝑘
)
=
[
(
1
−
𝑊
(
𝑘
)
)
+
𝑊
(
𝑘
)
𝑒
−
𝑖
𝑘
⊤
𝛿
𝑡
]
𝐹
𝑡
(
𝑘
)
.
F
t
′
	​

(k)=[(1−W(k))+W(k)e
−ik
⊤
δ
t
	​

]F
t
	​

(k).

But ideally do it with overlapping analysis/synthesis bands rather than multiplication against a single arbitrary FFT cutoff.

Why I think this beats the current taper

The taper experiment says:

smooth boundaries are sufficient to recover hf_ref ≈ .993.

So keep the smooth boundary.

But don't taper inside the nuisance's main spectral support.

You want:

unity gain over the bump frequencies + narrow cosine transition outside them.

That could plausibly retain most of the 70.2% suppression while recovering something close to the 0.99 fidelity regime.

There is an even better variant

If you're willing to make one slightly bigger change, I would prefer:

Estimate the nuisance component, then move only that component

Your synthetic corruption is additive:

𝑋
𝑡
=
𝑆
𝑡
+
𝐵
𝑡
X
t
	​

=S
t
	​

+B
t
	​


with approximately

𝐵
𝑡
(
𝑘
)
=
𝐵
0
(
𝑘
)
𝑒
−
𝑖
𝑘
⊤
𝑝
𝑡
.
B
t
	​

(k)=B
0
	​

(k)e
−ik
⊤
p
t
	​

.

So the mathematically clean solution isn't

“warp part of the scene.”

It's:

estimate 
𝐵
𝑡
B
t
	​

, replace it with its stabilized counterpart, leave 
𝑆
𝑡
S
t
	​

 alone.

Thus:

𝑋
𝑡
′
=
𝑋
𝑡
−
𝐵
^
𝑡
+
𝐵
^
𝑡
𝑠
𝑡
𝑎
𝑏
𝑙
𝑒
.
X
t
′
	​

=X
t
	​

−
B
^
t
	​

+
B
^
t
stable
	​

.

In Fourier space:

𝐵
𝑡
𝑠
𝑡
𝑎
𝑏
𝑙
𝑒
(
𝑘
)
=
𝐵
^
0
(
𝑘
)
𝑒
−
𝑖
𝑘
⊤
𝑝
𝑡
∗
.
B
t
stable
	​

(k)=
B
^
0
	​

(k)e
−ik
⊤
p
t
∗
	​

.

Therefore:

𝑋
𝑡
′
(
𝑘
)
=
𝑋
𝑡
(
𝑘
)
+
𝑊
(
𝑘
)
𝐵
^
0
(
𝑘
)
(
𝑒
−
𝑖
𝑘
⊤
𝑝
𝑡
∗
−
𝑒
−
𝑖
𝑘
⊤
𝑝
𝑡
)
.
X
t
′
	​

(k)=X
t
	​

(k)+W(k)
B
^
0
	​

(k)(e
−ik
⊤
p
t
∗
	​

−e
−ik
⊤
p
t
	​

).

This is a better conceptual match for the product.

You aren't building a video stabilizer.

You're building a:

persistent spectral-state corrector that removes generator drift without disturbing intentional scene motion.

That's an important distinction.

And your existing EMA state may already be surprisingly close to giving you 
𝐵
^
0
B
^
0
	​

.

2. H3 is correct, with an important qualification

Yes. Pixel-aligned HF SSIM is a bad sole fidelity gate for a system that intentionally introduces subpixel translations.

SSIM assumes corresponding structures occupy corresponding pixel locations.

Even a visually perfect translation can hammer it.

This isn't speculative. Translation sensitivity of conventional similarity measures was precisely one reason Wang & Simoncelli proposed complex-wavelet similarity; CW-SSIM treats consistent local phase changes as structurally equivalent and was designed to tolerate small translations. 
CNS NYU
+1

Efficient subpixel registration methods have also explicitly been developed for translation-invariant comparison. 
PubMed

So I would not discard hf_ref, but change the evaluation into two metrics.

Keep metric A: literal fidelity

Your existing:

Plain text
HF-SSIM(corrected, clean)

This remains valuable because it catches:

ringing
ghosts
sharpened boundaries
ball deformation
invented textures

But don't make it the only structural fidelity measure.

Add metric B: motion-compensated HF fidelity

Estimate only the known nuisance/global correction 
𝛿
𝑡
δ
t
	​

, then compare after registration:

𝑆
aligned
=
𝑆
𝑆
𝐼
𝑀
(
𝐻
𝑃
(
𝑇
−
𝛿
𝑡
(
𝑌
𝑡
)
)
,
𝐻
𝑃
(
𝐶
𝑡
)
)
.
S
aligned
	​

=SSIM(HP(T
−δ
t
	​

	​

(Y
t
	​

)),HP(C
t
	​

)).

For the synthetic benchmark you have something even better:

use the oracle known injected displacement rather than estimated optical flow.

That avoids letting the metric “cheat.”

And I would add CW-SSIM

For this exact architecture, it's unusually well matched.

Compute:

Plain text
CW-SSIM(corrected, clean)

on the mid/high frequency bands only.

CW-SSIM explicitly regards consistent wavelet-phase changes from small translations as much less destructive than ordinary SSIM does. 
CNS NYU
+1

Don't replace everything with CW-SSIM though.

It can become overly tolerant of geometric discrepancies, which is why recent metric reviews also caution against treating its invariance as universally desirable. 
PubMed Central (PMC)
+1

So your new fidelity panel should be
Plain text
hf_ref_raw
hf_ref_registered
cwssim_mid_high
motion_preservation
drift_pos_reduction

I would eventually add:

Plain text
edge ghost energy

specifically around moving balls.

That will tell you much more than one SSIM number.

3. The single decisive experiment

Don't tune rho again.

Don't sweep MC strength again.

Run an ORACLE TRANSLATION TEST.

It can simultaneously falsify H1, H2 and H3.

You know the true injected bump and true displacement in this synthetic environment. Exploit that.

Four arms, same 141 frames
A. CLEAN

Ground truth.

B. CORRUPTED

Current wandering bump.

C. ORACLE BAND PHASE

Feed the engine the true bump displacement 
Δ
𝑡
𝐺
𝑇
Δ
t
GT
	​

.

No estimation.

Use:

flat-top radial band containing essentially all bump energy
cosine transition outside it
exact Fourier phase ramp
𝑒
−
𝑖
𝑘
⊤
Δ
𝑡
𝐺
𝑇
e
−ik
⊤
Δ
t
GT
	​

D. ORACLE COMPONENT

Because this is synthetic, use the actual injected bump tensor:

𝑌
𝑡
=
𝑋
𝑡
−
𝐵
𝑡
+
𝐵
𝑠
𝑡
𝑎
𝑏
𝑙
𝑒
.
Y
t
	​

=X
t
	​

−B
t
	​

+B
stable
	​

.

This is your theoretical ceiling.

Measure both raw and aligned fidelity

For C/D report:

Plain text
drift%pos
hf_ref_raw
hf_ref_oracle_registered
CW-SSIM
ball motion

Then the interpretation becomes extremely clean.

Outcome 1
Plain text
Oracle component:
drift >> 60
HF raw > .98

Great.

Problem is estimation / decomposition.

Your target is physically attainable.

Outcome 2
Plain text
Oracle component:
drift >> 60
HF raw < .98
HF aligned ≈ 1.0
CW-SSIM ≈ 1.0

Then H3 wins.

Your existing fidelity gate is rejecting valid geometric correction.

Change the metric.

Outcome 3
Plain text
Oracle component passes,
oracle band-phase fails

Then H2 in its simple band-selective form is falsified.

There is too much spectral overlap between nuisance and real scene.

Move to:

𝑋
−
𝐵
^
+
𝐵
^
stable
X−
B
^
+
B
^
stable
	​


rather than band warping.

Outcome 4
Plain text
Oracle band-phase passes
but estimated band-phase fails

Excellent result.

Architecture is fine.

Your phase estimator / trajectory filter is the bottleneck.

Then implement weighted phase-plane fitting.

My prediction

Given your current numbers, my expectation is:

Plain text
oracle component       >> passes
flat-top phase band    likely passes or gets very close
current hard lowband   fails HF
current taper          fails drift

I would not interpret mc 0.5 lowband → hf_ref .9346 as evidence that phase correction is inherently destructive.

The fact that:

Plain text
same basic idea + taper
0.9346 → 0.9929

is unusually strong evidence that the dominant failure is the way frequencies are stitched back together, not the translation concept itself.

And I would stop sweeping rho. Your .995 → .997 → .999 results are already telling you that you're on the wrong axis: stronger persistence is buying neither the extra 10 points of drift suppression nor a fidelity improvement.

M0.5 decision

If I were choosing exactly one implementation tonight:

Robust low-k phase-plane displacement estimator → causal trajectory smoother → flat-top radial correction mask with cosine overlap.

Not full-frame stabilization.

Not Bundled Paths.

Not another EMA sweep.

And before even training/tuning that, run the oracle four-arm test. It should tell you in one run whether the product mechanism is sound, whether your spectral decomposition is the failure, or whether the hf_ref ≥ .98 gate itself is incorrectly specified.

ChatGPT can make mistakes. Check important info.
Latest response




Medium