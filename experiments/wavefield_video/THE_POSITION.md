# Video rescue pilot and model research

Updated October 13, 2026. The September position was a commercial hypothesis,
not a validated rescue service. The fixed post-process and learned video-memory
engine have separate gates. CRUMB remains the context handoff product.

## The pilot offer

Evaluate one authorized clip, up to 15 seconds, under frozen settings.
Proposed price: **US$99 only for a useful result the creator accepts**.
Eligibility, comparison and acceptance precede payment. A failed detail or
motion gate makes that output ineligible for paid fulfillment.

The current engine targets limited brightness/color changes. It has not
established identity repair, geometric continuity, detail restoration or
general HD rescue. [Both native 1080p tests failed](reports/ENG1_1080.md);
their sources were rendered math reels, not natural or client footage.
The later [owned native 720p house gate](../../docs/evidence/native-720-house-20261013.json)
fails the pre-encoding motion measure (0.516330 versus the 0.90 floor), despite passing
the pre-encoding detail measure. Independent complete native RGB decoding of the
identically encoded A/B also [fails the motion-measure gate](../../docs/evidence/native-720-house-delivery-verification-20261013.json):
0.488231 or 2.048211 depending on the unrevealed role order, both outside [0.90,1.10].
Encoded detail was not remeasured. These measures do not identify intended motion.
The easy synthetic gain scenario is matched closely by a 24-byte EMA, and
[the harder phase-drift scenario fails](M0_RESULTS.md). Do not turn either
scenario's percentages into customer-facing promises.

## The 72 hour demand test

Start the clock with the first owner-approved manual invitation. Hold the
scope, US$99 price, processing settings and acceptance gates fixed. Record
each invitation, independent clip source, consent, refusal, native resolution,
blind comparison, creator decision, accepted payment and refund separately.

Success requires **three independent strangers to submit usable broken
footage and at least one to accept and pay for a useful result**. Owned clips,
draft invitations, UI analytics and AI reviews are not customer demand.
If no one pays, revise or stop the offer before further engine tuning.
If native-resolution clips fail, that failure is the engineering milestone.

Preparation does not authorize outreach, publication or billing. A page that
only prepares an email draft must say so and must not display “Received” or
count the draft as a submitted lead.

## Product and research scope

BrainSNN's current product remains its bounded content detector/comparison
engine and simulation interface. The clip rescue is a service evaluation.
The learned video engine is research, not an established multi-minute
coherence product or a deployed wave-field stage in the text analyzer.

Use a dedicated video repository and evidence index as the source of truth.
Keep old experiment paths for reproducibility; move no checkpoints or private
customer material merely to simplify the portfolio story.

## Public claims

The page should lead with the evaluation offer and a real, correctly labeled
comparison. Put numerical methods and failure gates on an evidence page.
Every result names its baseline, hardware, initialization seeds and trained
status. Missing metadata stays unknown.

Retire “417× fewer tokens,” untrained-stream coherence claims and global
drift/detail/motion percentages. Preserve scoped positive and negative results
in the [claims ledger](../../docs/VIDEO_CLAIMS.md).

## Research gate

Train and evaluate the same 256-frame dependency on at least three
initialization seeds. Compare explicit reference/latch memory, frozen frames,
bounded-window attention, generic SSM and factorized attention; publish the
complete result either way after provenance and leakage checks.

Use a frozen pretrained VAE only after encoder causality is verified. Codec
replacement does not supply a generative prior or cure the failed rollout
objective. Decode held-out predictions and report first collapse and blind
visual review before claiming better video. Bring the protocol and raw
receipts to one independent human reviewer in efficient sequence modeling.
