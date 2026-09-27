# 3D-FIRST VIDEO: THE BLENDER METHOD + THE CUTTING EDGE
Filed 2026-09-26 · Zeph deep-dive for Slava · status: research complete, pilots specified

## The question
"Why aren't we using the Blender method — develop 3D objects, render them, adjust placement — or any other cutting edge methods?"

## TL;DR
The frontier answer to long-form video coherence is **control-first generation**: build the scene in 3D, render control passes (depth/edges), and let video diffusion finish it photorealistically. The output follows the 3D structure exactly — drift is impossible *by construction*, not corrected after the fact. We own most of the pieces already (Blender installed, TRELLIS + Wan 2.2 14B on the box, Comfy3D). The missing pieces are small. And our crumb engine stays valuable as the final coherence pass for any pipeline.

## The four pipelines (ranked for us)

**A. Pure Blender render (deterministic).**
TRELLIS.2/Hunyuan3D assets → Blender scene (driven agentically) → Cycles render.
100% coherent, placement infinitely adjustable, no AI artifacts. Cost: pure-CG look unless assets/lighting are excellent.
Pieces: Blender (HAVE) + blender-mcp (MISSING) + asset gen (UPGRADE: TRELLIS.2, 24GB VRAM = our 4090 exactly, MIT).

**B. Blender + generative finish (THE PILOT).**
Blender scene → render passes (depth/edges) → **Wan 2.2 Fun Control** (pose/depth/edge-conditioned video, ComfyUI-native).
Photoreal AND drift-free: the model cannot wander because every frame is conditioned on the exact rendered geometry. Adjust placement = edit the scene, re-render controls, regenerate (or restyle without re-moving anything).
Pieces: Wan 2.2 14B (HAVE) + Fun-Control model (small download) + a rendered control pass (Blender, HAVE).

**C. 3D-informed generation (no Blender).**
Depth → point-cloud "3D cache" → video diffusion on camera rails (NVIDIA GEN3C, CVPR'25). Strong research, heavier stack, less placement control than B. Watch, don't build yet.

**D. 3D reconstruction loop.**
Video → 3DGS → re-render from stabilized camera. Kills flicker by re-rendering; needs reconstruction quality (fragile on artifact-heavy input). GaussVideoDreamer runs single-image→scene on a 4090 in ~25 min. Later.

## Landscape notes (2026-09)
- **blender-mcp** — dominant MCP server for Blender (socket port 9876, addon in Blender, headless-capable variants exist for full 218-tool control). This is how *I* drive Blender directly.
- **TRELLIS.2** (Microsoft, Dec 2025) — 4B params, O-Voxel representation, single image → textured asset, **24GB VRAM**, MIT. Successor to the TRELLIS we already smoke-tested. Hunyuan3D 3.0 = stronger textures (4K PBR) but heavier pipeline.
- **Wan 2.2 Fun Control** — official ComfyUI workflow exists (pose/depth/edge guides, 81 frames @ 16fps). This is the bridge product.
- **Stable Virtual Camera (SEVA)** — single image → orbit/spiral/dolly 3D video. **Non-commercial license** = internal R&D only, cannot ship.
- **ReCamMaster** (Kling AI, ICCV'25 Best Paper Finalist) — camera re-trajectory for existing video; interesting for the rescue upsell (re-angles!).
- **WonderWorld** (Google) — single image → explorable 3D scene; scene-level play.
- **GaussFusion (CVPR'26)** — fixes 3DGS flicker/floaters; refine step node.
- Our crumb_coherence engine = orthogonal and complementary: the **final temporal-coherence pass** any of A-D can feed into (and the $99 rescue service for footage that already exists).

## Why not sooner
The generative path was the fast path to *any* footage (Wan I2V = hours to first clips). The 3D path needed the pipeline wired end-to-end (asset gen → scene → control passes → conditioned gen) — researched but never assembled. Cost is now mostly assembly, not research.

## Test plan (bind to 4090, zero spend)
- P0 — blender-mcp wired to Hermes; headless house scene loads; camera path renders a control pass. Success: depth+edge PNG sequences for a 5s path.
- P1 — Wan Fun-Control on the box: control pass → 5s conditioned render of the same house shot. Success: photoreal, camera honors the pass, zero drift across 5s.
- P2 — adjust placement (move a piece of furniture) → re-render controls → regenerate. Success: output reflects the change; old control footage rejected.
- P3 — 60s+ chain: three 20s paths through one scene, one style pass. Success: coherent beyond the current 2-5s wall — the long-form demo.
- Always — final crumb_coherence pass over outputs; before/after for the rescue funnel.

## Sources (accessed 2026-09-26)
GEN3C (nv-tlabs, CVPR'25) · stable-virtual-camera (Stability) · ReCamMaster (KlingAIResearch) · blender-mcp (ahujasid + headless forks) · ComfyUI Wan 2.2 Fun Control workflow · WonderWorld (KovenYu) · TRELLIS.2 (microsoft) · GaussFusion (CVPR'26) · GaussVideoDreamer
