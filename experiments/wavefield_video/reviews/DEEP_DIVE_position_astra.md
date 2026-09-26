Recommendation: C, but sell A first

Brainsnn.com should have a two-layer story:

Today: Fix drift and flicker in AI-generated video.
Tomorrow: Make long-form AI video stay coherent from the start.

Do not launch as “the long-horizon video engine.” The streaming model is valuable IP and a partner magnet, but synthetic-grade output makes it a weak thing to ask customers to pay for today.

The coherence pass is different. It attacks a pain customers already recognize. Kling itself published a July 2026 guide specifically about fixing AI-video drift, which is unusually good market validation for the problem. 
Kling
 Google Research also just framed long-form video around visual drift, consistency, persistent memory and cascading failures, showing that the problem is strategic rather than niche. 
Google Research
+1

One important market correction: I would stop framing the problem as merely “models only make 8–25 second clips.” Veo's published benchmark outputs are still around eight seconds, but Kling now advertises workflows up to 30 seconds, and research systems already demonstrate minute-scale output. 
Google DeepMind
+2

Your stronger framing is:

Generation is getting longer. Coherence isn't keeping up.

That survives model improvements.

1. What to sell first
Sell this this month

“Send us your drifting AI video. We stabilize it without regenerating it.”

Not:

spectral coherence engine
k-space state
long-horizon architecture
BrainSNN research
streaming world model

Those belong further down the page.

The first commercial product should initially be a concierge service, not SaaS.

Why? Because you haven't validated 1080p yet.

Manual service lets you:

inspect incoming clips,
reject unsuitable footage,
learn actual failure distributions,
tune without pretending the product is fully generalized,
charge immediately,
collect before/after footage.

Once you've fixed 20–50 real customer clips, automate the workflow into an API/upload product.

Product hierarchy

BrainSNN Coherence

Fix accumulated drift, flicker and scene instability in generated video.

Then underneath:

Coherence API

Add persistent visual stabilization to an existing video workflow.

Then much further down:

Long-Horizon Engine — Research Preview

Experimental constant-state generation technology for multi-minute streaming video.

That is C structurally, but A commercially.

2. Your actual buyer

Do not start with casual people making six-second memes.

Their pain is low and willingness to pay is terrible.

I would target three segments.

Buyer #1: AI-first production studios and ad agencies

Best initial buyer.

Specifically teams already producing:

branded social ads
product launches
fashion/beauty campaigns
pitch films
previsualization
AI commercials
music-video work

They care because a failed generation isn't just ugly. It creates:

re-generation cost
editor time
client revision rounds
inconsistent shots
missed delivery deadlines

Someone already spending hundreds or thousands on a campaign will pay $100–$500 to rescue an otherwise usable asset.

Buyer #2: serious independent AI filmmakers

Not hobbyists.

Look for people producing:

1–20 minute AI shorts
trailers
music videos
narrative series
cinematic YouTube projects

The current r/aifilmmaking community is already centered around shorts, series, trailers and even 19–22 minute AI projects, and has grown quickly to roughly 8,000 members. 
GummySearch
 Recent discussion among practitioners also identifies consistency, control and longer-form storytelling as continuing problems. 
Reddit

Buyer #3: tooling/workflow companies

Later, but potentially your largest customer.

Think:

AI editing tools
generation front ends
model aggregators
automated ad-generation platforms
story/video agents
virtual production systems

They don't want to rebuild coherence research.

They want:

Plain text
POST /coherence
video → corrected video

That's where your API ultimately belongs.

Where they hang out

For the first month:

r/aifilmmaking
AI Filmmakers Discord
Banodoco Discord, especially technical/open-source creators
model-specific Runway/Kling communities
X/Twitter around AI filmmaking
LinkedIn for AI creative-agency founders and creative directors
YouTube channels publishing AI filmmaking workflows

Creators themselves report using AI-filmmaking Discord communities and Banodoco for workflow experimentation and model stress-testing. 
Reddit

Don't spend your first month buying Meta ads.

Go directly into communities where people are already complaining about the problem.

3. Pricing

Use USD, even from Canada. This is a global digital product.

Current creator tooling gives useful anchors. Runway ranges roughly from $12/year-billed creator entry tiers to $76/month heavy-user plans, while Topaz Video ranges around $39–$59/month for individual video enhancement. 
Runway
+1
 Runway's API generation itself can cost roughly $0.05–$0.40 per output second depending on model/configuration. 
Runway Dev

You shouldn't compete directly against those subscriptions yet.

A. Done-for-you coherence fix

Beta pricing:

Job	Price
≤15 sec	$49–$79
15–60 sec	$99–$199
1–3 min	$249–$499
Agency / difficult project	$500–$1,500+

But I'd launch with one simple offer:

$99 AI Video Rescue
Send one broken clip. We return the improved version + comparison.
If we can't materially improve it, you don't pay.

That is extremely easy to understand.

After validation, raise it.

B. API

Eventually:

$0.02–$0.08/video-second, depending on resolution and compute.

With minimum plans:

Creator: $29–$49/mo
Pro: $99–$199/mo
Studio/API: $399–$999/mo
Enterprise: custom

Your post-processing should ultimately be significantly cheaper than regeneration. That's part of the economic story.

Example:

Regenerate a 30-second sequence five times or run a $1–$3 correction pass.

That becomes powerful if the technology truly supports it.

C. Early engine access

Do not sell this for $29/month.

The engine isn't a consumer product yet.

Sell access as a design-partner program:

$500–$2,000/month
for researchers/small creative-tech companies.

Or:

$2,500–$10,000 project engagement
for custom integration / evaluation / joint experimentation.

You're selling:

technical access
founder/research support
influence over roadmap
evaluation builds
custom experiments

Not photorealistic output.

Label it clearly:

Research Preview — not a production generator.

4. Homepage journey

You have about ten seconds before the visitor decides whether this is real.

Screen 1: show the problem and result

No research introduction.

Headline

AI videos drift. We fix them.

Subhead:

Upload AI-generated video and remove accumulated scene drift and flicker without regenerating the clip.

Buttons:

Fix My Video

Watch Before / After

And immediately show:

Plain text
ORIGINAL             COHERENCE ON
[video]              [video]

Same clip. Synced playback.

This is the most important object on the website.

Screen 2: quantify what they're seeing

Something like:

Less drift. Same motion.

−86%
low-frequency drift on tested rollouts

0.99
high-frequency structural similarity

0.97
motion preservation

Then small text:

Results vary by video. Current validation includes synthetic benchmarks and experimental low-resolution model rollouts. HD validation underway.

Transparency will actually increase credibility here.

Screen 3: works after generation

Logos/text:

Kling • Runway • Veo • open-source models • your own pipeline

But phrase it carefully until tested broadly:

Designed to operate on generated video independently of the source model.

Do not claim compatibility you've never tested.

Diagram:

Plain text
Your generator
      ↓
Generated video
      ↓
BrainSNN Coherence
      ↓
More stable video

That's all.

Screen 4: show the failure magnified

Give them a slider or zoom:

Background drift

Raw:
← wandering →

Fixed:
│ anchored │

A skeptical creator needs visual proof more than another percentage.

Strongest proof possible

Take one difficult clip from Kling or Runway, not your own wave model.

Process it blind with frozen settings.

Publish:

original file
processed file
method/version
no prompt change
no regeneration
no cherry-picked frame comparison

That destroys far more doubt than ten synthetic metrics.

Three independent generators would be even stronger.

Screen 5: purchase

Don't make them understand your architecture.

Three steps:

1. Upload
Send us your generated clip.

2. We stabilize it
No regeneration required.

3. Download
Review the before/after.

Then:

$99 Beta Video Rescue

CTA:

Fix My Clip

Screen 6: only then show the bigger vision

We're building beyond post-processing.

Explain that BrainSNN is researching constant-state, long-horizon video generation.

Show your streaming engine.

Invite:

Apply for Engine Preview

Now the research strengthens the company instead of confusing the transaction.

5. What I would cut

Aggressively remove anything that makes the visitor ask:

“What exactly are these guys selling?”

Cut from the homepage:

neuromorphic/SNN history unless directly relevant
giant research-paper sections
formulas
spectral/k-space explanations above the fold
dozens of metrics
internal project names
claims about “revolutionizing video”
long-term photorealistic promises
multiple unrelated AI projects
vague “world model” language
8 different calls to action
giant team/about section
“AI research lab building the future” copy

Also avoid trying to be:

Runway + Topaz + video foundation model + research institute

on one page.

Topaz succeeds because the promise is extremely concrete: improve the video you already have. Its current product line directly sells enhancement, stabilization, interpolation and upscaling. 
Topaz Labs
+1

Borrow that clarity.

The technology can be sophisticated.

The pitch shouldn't be.

6. Fastest revenue experiment: next 72 hours

This is what I would actually do.

The $99 Drift Rescue Challenge

Create one landing page.

Headline:

Got an AI video that almost works? Send us the broken clip.

Copy:

We built a new coherence pass for generated video.
Send us a clip with drift, flicker or long-horizon instability.
We'll try to repair it without regenerating it.
$99 if the improvement is worth keeping. $0 if it isn't.

Then manually contact 50 people, not 5, across:

15 AI filmmakers
15 creative agencies
10 AI-video YouTubers/tutorial creators
10 technical creators from Reddit/Discord/X

Don't pitch BrainSNN.

Send:

Saw your AI video work. We're testing a post-process that reduces accumulated drift/flicker without regenerating the clip. I'm looking for difficult real footage. If you have one clip that almost works but falls apart over time, send it over. We'll process it and show you the before/after. $99 only if the result is useful.

Your goal for 72 hours is not $10,000.

It's this:

Can 3 strangers give you real broken footage, and will at least 1 pay to keep the fixed version?

If yes, you've discovered a genuine commercial wedge.

If nobody cares enough to send a clip, changing pricing or rebuilding brainsnn.com won't solve the underlying problem.

And if people send clips but your engine can't improve real 720p/1080p footage, you've learned the exact engineering milestone that comes before selling SaaS.

My order from here

1. Finish the non-interference test.
2. Run one 720p/1080p third-party clip.
3. Launch the $99 rescue offer.
4. Collect 10 ugly customer clips.
5. Turn successful ones into public before/after proof.
6. Only then build self-serve/API.

Google's new research makes the timing unusually good: it explicitly validates persistent memory, long-horizon consistency and test-time correction as important directions, while presenting its work as research rather than a generally available creator product. 
Google Research
+1

That leaves you a commercially interesting wedge: don't try to out-generate Google. Make everyone else's generated video more stable.