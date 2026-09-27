For your exact setup, I would make B the weekend build, keep C as the research direction, use A only as a control/baseline, and avoid D until you have a customer reason to reconstruct arbitrary footage into 3D.

1. Rank A–D
Path	4090/24 GB	Demo this week	Long-term defensibility
B. Blender → control video → diffusion → crumb	1	1	2
A. Pure Blender/TRELLIS.2 render	2	2	4
D. Video → 3DGS → rerender	3	3	3
C. GEN3C-style 3D-cache diffusion	4	4	1

B wins now. Wan2.2-Fun Control is already proven in ComfyUI on an RTX 4090-class 24 GB card: 640×640, 81 frames used ~83% VRAM in FP8, around 520 s/run; the 4-step Lightning path cut repeat generation to around 79 s at some dynamics cost. It natively supports depth, canny, pose and trajectory conditioning. 
GitHub
+1

C is the architecture worth stealing from long term. GEN3C's important idea is not “another generator”; it is that a persistent 3D cache explicitly carries scene geometry forward, so diffusion doesn't have to remember the world from tokens alone. That maps almost perfectly onto your thesis: persistent geometric state upstream + tiny spectral temporal state downstream. 
GitHub
+1

A is useful because TRELLIS.2 targets ≥24 GB GPUs, so your box is at the minimum viable hardware boundary, but a Blender render alone doesn't prove your coherence layer. 
GitHub

D is interesting, especially because 2026 work such as GaussFusion now uses depth, normals, opacity and covariance buffers to drive temporally coherent video refinement, but it adds reconstruction, pose and splat-quality failure modes you do not need for the first demo. 
arXiv

2. Pipeline B: the recipe I would run
Blender outputs

Render these from one deterministic camera path:

RGB clay/reference render
metric depth
Canny/edge pass derived from RGB or normals
object-ID masks
alpha/opacity
optionally normals for diagnostics/future models

For Wan Fun Control itself, depth and canny are the valuable direct control inputs; pose is useful for articulated humans. Opacity and masks are primarily for compositing, selective control and debugging rather than the core Fun-Control conditioning modes. Fun Control officially exposes canny, depth, pose and trajectory-style controls. 
GitHub
+1

Start with depth only

Don't stack every condition immediately.

Use:

Plain text
Blender
 ├─ beauty/reference frame
 └─ temporally exact depth sequence
          ↓
Wan2.2 Fun Control
          ↓
crumb_coherence

Then compare against depth+canny.

Depth supplies 3D layout while letting the generator invent texture. Canny tends to overconstrain geometry and can cause generated textures to cling to meaningless render edges.

Depth normalization

This is a major trap.

Do not independently min/max-normalize every frame:

Python
depth_t = (depth_t - min_t) / (max_t - min_t)

That makes a stationary object change apparent depth whenever some other object enters the frame.

Instead establish fixed clip-level near/far bounds:

𝑑
′
=
clip
⁡
𝑑
−
𝑑
𝑛
𝑒
𝑎
𝑟
𝑑
𝑓
𝑎
𝑟
−
𝑑
𝑛
𝑒
𝑎
𝑟
d
′
=clip
d
far
	​

−d
near
	​

d−d
near
	​

	​


and keep them identical for every frame in the sequence.

Better still, use inverse depth if the model's depth preprocessor behaves more like monocular depth:

𝑧
′
=
1
/
(
𝑧
+
𝜖
)
z
′
=1/(z+ϵ)

but again normalize across the entire shot, not framewise.

Resolution

For iteration:

512–640 square-ish, 49–81 frames.

Your verified Wan2.2 Fun benchmark is 640×640/81 frames on 24 GB. 
GitHub

Do not start at 1080p.

Generate structurally correct controlled footage first, then upscale. Your experiment is asking whether 3D conditioning defeats long-horizon geometry drift, not whether the 4090 can survive a giant latent.

Biggest B failure modes
per-frame depth normalization → breathing geometry
depth discontinuities at thin objects
hard Canny edges → texture locking/ringing
inconsistent lighting between Blender reference and prompt
prompt contradicts geometry
overly strong control → “CG render painted photorealistically”
overly weak control → generator ignores the camera path
independent segments changing grade/style
occluded surfaces appearing differently after re-entry

The last one is the important long-horizon test.

3. Would I prefer something newer than Wan2.2?

I would benchmark LTX-2.3 next, but I would not replace Wan for this weekend.

LTX-2 now has native ComfyUI workflows for depth, Canny and pose control using IC-LoRAs, keyframe generation, and spatial/temporal upscaling. LTX-2.3 is the current released branch as of September 2026. 
GitHub
+1

The attractive part is speed and long-sequence experimentation. Even the older LTX line had community 4090 workflows generating ~1 minute / 1,800 frames in one pass at roughly 22 GB VRAM, and current tooling supports control-video injection. 
GitHub

But there is a caveat: official LTX desktop tooling historically gated local inference below 32 GB until CPU-offload work was added, so 4090 operation depends heavily on the implementation/quantization path. 
GitHub

So:

Weekend: Wan2.2 Fun
Monday benchmark: LTX-2.3 depth IC-LoRA
Research: GEN3C-like cache

4. The 60-second+ strategy

Do not make first/last-frame autoregression your primary architecture.

That creates:

𝑒
𝑡
+
1
=
𝑓
(
𝑒
𝑡
)
+
𝜖
𝑡
e
t+1
	​

=f(e
t
	​

)+ϵ
t
	​


so identity/color/geometry errors become the next segment's conditioning.

That is exactly the compounding loop you're trying to eliminate.

Instead:

One persistent 3D scene, many independently generated chunks
Plain text
Blender world + fixed materials/camera path
        ↓
control frames 0…N
        ↓
segments:
0–120
96–216
192–312
...
        ↓
overlap selection/blending
        ↓
crumb

Use 10–20% temporal overlap.

Every segment gets:

same scene geometry
same seed/reference image where supported
same prompt/style spec
its own absolute camera slice
overlapping control frames

That means segment 8 depends on the world, not on errors in segment 7.

This is much stronger than frame-chaining.

What breaks first?

Probably not gross geometry.

I expect:

color/lighting drift
texture/material identity
newly exposed surfaces
dynamic object identity
seam discontinuities
only then camera/path geometry

Measure:

depth reprojection error
optical-flow discontinuity across joins
LPIPS/DINO identity distance on tracked objects
average RGB/chroma drift
your low-band crumb metrics
seam-region temporal error
reappearance consistency after object occlusion

A nasty but useful benchmark is:

orbit behind object → object disappears → orbit back → does the same face/material return?

5. Blender MCP

The ecosystem has matured considerably.

The current 218-tool mcp-blender exposes modeling, materials, animation, render, geometry nodes, baking, rigging and physics through an addon + local MCP process. 
GitHub
+1

For your use case, however, I would favor headless/saved-file execution over a GUI-only agent.

There are newer implementations built specifically around blender -b, and enhanced Blender-Lab-derived MCP variants support saved-file/headless commands and subprocess isolation. 
GitHub
+1

Important gotcha: long bakes/render jobs can be awkward through live-addon request/response loops; one Blender-Lab MCP client explicitly recommends headless transport for genuinely long-running jobs. 
GitHub

My architecture:

Plain text
agent
 ↓
deterministic scene JSON/spec
 ↓
Python/bpy
 ↓
blender -b scene.blend --python render_controls.py
 ↓
rgb/
depth/
mask/
edges/
camera.json

Use MCP for scene construction/debugging.

Use plain scripted headless Blender for production renders.

Also note the 218-tool fork changed from MIT to AGPL-3.0-or-later in v0.4.0, relevant if you're embedding/distributing it commercially. 
GitHub

6. My weekend
Hour 0–2

Get B boringly deterministic.

Create one Blender scene:

room
textured hero object
foreground occluder
camera orbit + dolly
animated secondary object

Render RGB + fixed-normalization depth + masks.

Hour 2–5

Wan Fun Control depth-only.

Run 3 control strengths at 640×640 / 81 frames.

No crumb yet.

Pick the setting with best geometry preservation.

Hour 5–7

Run:

depth-only vs depth+canny vs uncontrolled Wan.

This gives you the actual scientific comparison.

Hour 7–10

Make four consecutive camera-path chunks using identical scene/style/reference conditioning.

Add 12–16 frame overlap.

Do not last-frame chain.

Hour 10–13

Build stitching:

overlap perceptual score
choose/blend boundary
histogram/chroma matching only if necessary

Then run crumb once on the complete sequence.

Hour 13–16

Create one adversarial shot:

360° partial orbit with disappearance/reappearance behind occluder.

If B works there, you have something meaningful.

Hour 16–20

Measure:

Plain text
geometry/depth consistency
boundary seam score
object identity
low-band drift
HF preservation
motion preservation
Hour 20–24

Produce a 45–60 sec demo:

Uncontrolled Wan | 3D-Control + Crumb

Same prompt. Same planned motion. Same duration.

The technical thesis I would pursue is not “Blender makes AI video coherent.”

It's:

Separate long-term state from generation. Geometry remembers where the world is; Crumb remembers how the video behaves. Diffusion only has to render the next view.

That is substantially more defensible than merely chaining increasingly long diffusion clips, and GEN3C/GaussFusion provide strong external evidence that explicit persistent 3D state is the right direction. 
GitHub
+1