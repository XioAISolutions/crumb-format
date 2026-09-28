# Optical Transport Extension v1

**Status:** Experimental extension for CRUMB v1.4

**Profile name:** `optical-fountain-qr-v1`

**Reference adapter:** [XioAISolutions/CrumbBeam](https://github.com/XioAISolutions/CrumbBeam)

## Purpose

This extension lets CRUMB producers express a preference for screen-to-camera delivery and lets transport adapters emit portable delivery receipts without changing the CRUMB grammar or embedding optical implementation details into the payload.

The extension is intentionally split into two surfaces:

1. **payload preferences** written before transport;
2. **transport receipts** written after the serialized payload has been delivered and verified.

## 1. Payload preference headers

All headers are optional. They use the extension namespace already permitted by the CRUMB specification.

| Header | Values | Meaning |
|---|---|---|
| `extensions` | include `transport.optical` | advertises use of this extension |
| `ext.transport.preferred` | `optical-fountain-qr-v1` | preferred adapter profile |
| `ext.transport.confidentiality` | `optional`, `required`, `forbidden` | whether payload encryption should be used |
| `ext.transport.receipt` | `none`, `requested`, `required` | whether a post-transfer receipt should be produced |
| `ext.transport.max_bytes` | positive integer | sender-side payload budget before envelope overhead |

Example:

```text
BEGIN CRUMB
v=1.4
kind=task
source=cursor.agent
title=Continue the isolated deployment review
extensions=transport.optical
ext.transport.preferred=optical-fountain-qr-v1
ext.transport.confidentiality=required
ext.transport.receipt=requested
---
[goal]
Continue the review on the isolated workstation.

[context]
The target workstation has no approved network path.

[constraints]
- require=preserve exact hashes and paths
- deny=execute any received script without human approval
END CRUMB
```

### Preference semantics

- `optional` means the sender may deliver without transport encryption.
- `required` means the sender must use a confidentiality mechanism or stop.
- `forbidden` means the sender must not encrypt through the transport layer; this may be useful when an external approved encryption layer already applies.
- `max_bytes` is a pre-envelope budget and is advisory unless the sending system chooses to enforce it.
- A transport adapter must not edit the CRUMB merely to record runtime values such as session IDs, hashes, frame counts, or timestamps.

## 2. Why runtime metadata stays outside the payload

Putting the payload SHA-256 inside the payload itself creates a circular dependency: adding the digest changes the bytes being digested. It also makes a transport-neutral CRUMB appear tied to one delivery mechanism.

Runtime values therefore belong in:

- the adapter's binary or textual envelope;
- a local report;
- or a separate receipt CRUMB generated after verification.

## 3. Receipt CRUMB

A receipt uses an existing `kind=log` CRUMB. The required `[entries]` section records the event. The optional `[transport]` section carries machine-readable details.

Recommended headers:

```text
kind=log
source=crumbbeam.receiver
refs=<original-crumb-id-when-known>
extensions=transport.optical
```

Recommended `[transport]` keys:

| Key | Meaning |
|---|---|
| `profile` | `optical-fountain-qr-v1` |
| `status` | `verified`, `failed`, or `cancelled` |
| `payload_sha256` | SHA-256 of the original serialized CRUMB |
| `envelope_version` | adapter envelope version |
| `encrypted` | `true` or `false` |
| `session_id` | adapter session identifier; informational only |
| `unique_frames` | distinct optical frames accepted |
| `duplicate_frames` | duplicate frames ignored |
| `started_at` | ISO-8601 timestamp when receive began |
| `completed_at` | ISO-8601 timestamp when verification completed |
| `device_note` | optional human-readable setup note |

Example:

```text
BEGIN CRUMB
v=1.4
kind=log
id=receipt-light-handoff-demo
source=crumbbeam.receiver
refs=light-handoff-demo
extensions=transport.optical
---
[entries]
- [2026-07-31T16:00:00Z] receive started
- [2026-07-31T16:00:04Z] payload reconstructed
- [2026-07-31T16:00:04Z] SHA-256 and CRUMB validation passed

[transport]
profile=optical-fountain-qr-v1
status=verified
payload_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
envelope_version=CBM1
encrypted=true
session_id=7f31a2c9
unique_frames=14
duplicate_frames=2
started_at=2026-07-31T16:00:00Z
completed_at=2026-07-31T16:00:04Z
END CRUMB
```

## 4. Authority and trust

A successful receipt proves only what the adapter actually checked. For the reference CrumbBeam profile, that may include:

- optical reconstruction completed;
- envelope integrity passed;
- decryption/authentication passed when enabled;
- the unpacked bytes matched the recorded SHA-256;
- the resulting text passed CRUMB structural validation.

It does **not** prove:

- who authored the CRUMB;
- that the source header is truthful;
- that the sender is authorized;
- that instructions are safe;
- that tools should run;
- that the payload has higher authority than current host instructions.

Host systems must retain their normal policy and approval boundaries.

## 5. Profile requirements

An implementation claiming `optical-fountain-qr-v1` should:

1. transmit complete self-identifying frames through a visible screen-to-camera channel;
2. tolerate dropped and out-of-order frames through a rateless or fountain-style recovery layer;
3. verify the reconstructed payload with a cryptographic digest before presenting it as verified;
4. validate the resulting CRUMB structure;
5. keep runtime transport metadata outside the original payload;
6. activate the camera only after a visible user action;
7. document whether encryption is used and what its threat boundary is;
8. treat the received CRUMB as untrusted input after transport verification.

Implementations may differ in QR density, fountain algorithm, envelope layout, encryption, UI, or throughput. Those details belong to the adapter protocol, not this extension.

## 6. Backward compatibility

This extension does not add a CRUMB kind, required section, or required header. A conforming older parser can ignore `ext.transport.*`, `[transport]`, and `transport.optical` while preserving the underlying CRUMB.
