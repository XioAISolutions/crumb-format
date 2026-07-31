# CRUMB Transports

CRUMB is transport-neutral. A valid CRUMB can move through copy and paste, a file, Git, a clipboard, an API, removable media, or an optical channel without changing the meaning or authority of the handoff.

Transport adapters may describe themselves through optional `ext.transport.*` headers and may emit a separate receipt after delivery. Older CRUMB parsers remain compatible because extension headers and unknown sections are additive and ignorable.

## Registered experimental profiles

| Profile | Adapter | Purpose | Status |
|---|---|---|---|
| `optical-fountain-qr-v1` | [CrumbBeam](https://github.com/XioAISolutions/CrumbBeam) | screen-to-camera CRUMB delivery through loss-tolerant animated QR frames | experimental |

The extension contract is documented in [docs/extensions/optical-transport-v1.md](docs/extensions/optical-transport-v1.md).

An experimental profile should not be promoted to stable solely because its parser, envelope, or recovery simulation passes. A physical transport must also publish repeated end-to-end evidence across representative sender and receiver devices, including failed attempts, exact-match verification, security boundaries, and device-specific—not universal—performance claims.

## Core rules

1. **The payload remains authoritative.** Transport metadata cannot override system, developer, user, policy, constraint, or approval semantics in a CRUMB or its host runtime.
2. **Integrity is not identity.** A byte-for-byte verified payload is not proof that its author is trusted.
3. **No circular digests.** Payload hashes belong in an envelope or receipt generated after serialization, not inside the bytes being hashed.
4. **Unknown profiles are safe to ignore.** A receiver may preserve an unsupported profile as metadata but must not infer behavior from it.
5. **Execution is never implied.** Receiving a CRUMB does not authorize `[script]`, tool, network, filesystem, or deployment actions.
6. **Adapters publish their own wire protocols.** The CRUMB specification defines the handoff format, not QR, radio, file, HTTP, or encryption details.

## Optional preference headers

A CRUMB author may declare transport preferences without making them mandatory:

```text
extensions=transport.optical
ext.transport.preferred=optical-fountain-qr-v1
ext.transport.confidentiality=required
ext.transport.receipt=requested
```

These values are advisory. A sender that cannot satisfy a required preference should stop and ask rather than silently downgrade.
