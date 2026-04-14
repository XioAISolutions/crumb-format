# CRUMB Threat Model

This document describes the threats CRUMB is designed to handle, the threats it is **not** designed to handle, and the security boundaries an implementer or operator should respect. It complements `docs/SECURITY.md`, which covers the operational `crumb lint` workflow.

---

## 1. Trust boundaries

A `.crumb` file crosses three boundaries during its life cycle:

1. **Producer → File.** An AI tool, CLI, or human writes a `.crumb` to disk, clipboard, or a transport.
2. **File → Storage / Transport.** The file may be committed to git, pasted into a chat, attached to an issue, or sent through an MCP/REST/A2A bridge.
3. **Storage → Consumer.** Another AI tool, CLI, or human ingests the `.crumb` and acts on its contents.

Every threat below is described relative to one of these boundaries.

---

## 2. Assets

What CRUMB protects:

- **Plaintext context** — the goals, constraints, observations, and preferences captured inside a crumb.
- **Identity & policy** (AgentAuth) — agent passports, allowed/denied tool lists, audit records.
- **Operator integrity** — the assurance that running `crumb` on an untrusted file does not compromise the host.

What CRUMB does **not** protect:

- The confidentiality of a `.crumb` once it leaves the operator's control. CRUMB has no built-in encryption.
- The truthfulness of facts inside a crumb. CRUMB is a transport format; semantic correctness is the AI's job.
- Network-level secrecy. Use TLS, signed git tags, and platform secrets management.

---

## 3. Threats and mitigations

### T1. Secret leakage through pasted context

**Scenario:** A producer copies a crumb containing API keys, JWTs, or AWS credentials into a hosted AI or public issue.

**Mitigations in CRUMB:**

- `crumb lint --secrets` heuristically detects common credential patterns (see `docs/SECURITY.md`).
- `crumb lint --redact` rewrites obvious credentials in place or to `--output`.
- The spec recommends avoiding `[raw_sessions]`, `[logs]`, and `[raw]` for shared crumbs; pack/dream-passed crumbs are preferred.

**Residual risk:** Heuristics miss novel secret formats. Human review remains the last line of defense.

### T2. Malicious crumb crashing or hanging the parser

**Scenario:** A consumer ingests a crafted crumb that is megabytes long, deeply nested, or uses adversarial line endings.

**Mitigations in CRUMB:**

- The grammar (SPEC §2.1) is strictly line-oriented and bounded — there is no nesting, no escape sequences, no quoting layer to confuse the parser.
- Parsers MUST reject files lacking the `BEGIN CRUMB` / `END CRUMB` markers and the `---` separator (SPEC §2.1 conformance notes).
- Reference implementations enforce reasonable line and total-size caps and surface them as validation errors, not crashes.

**Residual risk:** Implementations must apply their own size cap appropriate to their environment (e.g., 1 MB per file by default).

### T3. Prompt injection through crumb contents

**Scenario:** A producer writes a crumb whose `[context]` or `[notes]` contains adversarial instructions intended to override the consuming AI's system prompt.

**Mitigations in CRUMB:**

- The spec defines `.crumb` as **data**, not as a prompt. Consumers are expected to embed crumb contents as quoted context, not as instruction text.
- The structured section model (`[goal]`, `[context]`, `[constraints]`) gives consumers a clear seam to apply their own trust policy per section.
- `kind=` distinguishes crumbs the consumer should treat as authoritative (e.g. `mem` from a trusted source) from crumbs that are merely advisory (e.g. `task` from an external agent).

**Residual risk:** This is fundamentally an AI consumer problem. CRUMB makes the data structure clear; it cannot prevent a credulous consumer from following injected instructions.

### T4. Tampering in transit

**Scenario:** A crumb is modified between producer and consumer (e.g., MITM on a webhook, malicious commit).

**Mitigations in CRUMB:**

- Crumbs are plain text and diff cleanly in git, so tampering is visible in code review.
- AgentAuth audit records (when used) provide a tamper-evident log of which agent emitted which crumb.

**Out of scope:** End-to-end signing of crumbs. A future minor version may define an optional `signature=` namespaced extension; until then, rely on transport-level integrity (signed commits, TLS, signed webhooks).

### T5. Unauthorized agent action via a stolen passport

**Scenario:** An attacker exfiltrates an AgentAuth passport file and uses it to invoke high-risk tools.

**Mitigations in AgentAuth:**

- Passports carry a TTL (`--ttl-days`) and a deny list.
- `crumb passport revoke` provides an instant kill switch.
- Per-tool policy (`crumb policy set`) restricts the blast radius even before revocation.
- Every action is logged with risk scoring; `crumb audit feed` allows live monitoring.

**Residual risk:** A stolen passport is valid until revoked. Operators should rotate passports on schedule and treat passport files as secrets.

### T6. Extension namespace squatting

**Scenario:** A third party ships a tool that emits `extensions=crumb.pack.v1` while implementing different semantics.

**Mitigations in CRUMB:**

- The spec reserves the unprefixed namespace (`crumb.*`) for the official project (SPEC §3.3 / §8.1).
- All third-party extensions MUST use a vendor prefix (e.g. `ext.acme.priority`, `x-acme-pack.strategy`).
- Conformance fixtures in `fixtures/extensions/` pin the canonical shape of `crumb.*` extensions.

### T7. Supply-chain compromise of the `crumb-format` package

**Scenario:** An attacker publishes a malicious release of `crumb-format` to PyPI.

**Mitigations:**

- Releases are tagged in git, with the tag SHA matching the published wheel.
- The CI pipeline that builds and uploads is reproducible from the public repo.
- A constraints file (planned for v0.5) pins exact transitive dependency versions.
- Operators MAY pin to a specific version and verify against the git tag.

**Residual risk:** Standard PyPI supply-chain risk applies. Use `--require-hashes` in security-sensitive deployments.

---

## 4. Out-of-scope threats

The following are **explicitly not** mitigated by CRUMB:

- Confidential storage of crumbs on disk. Use OS-level encryption.
- Multi-tenant isolation between agents on the same host. Run separate processes / containers.
- Rate-limiting or DoS against the REST/A2A bridges. Front them with a real gateway.
- Adversarial AI behavior (the consuming AI deciding to misuse provided context). This is the consumer's policy problem.
- Side-channel timing attacks on AgentAuth credential validation. Use HSMs or constant-time libraries if this matters to you.

---

## 5. Reporting

If you discover a security issue in CRUMB, please follow the disclosure process in `SECURITY.md` at the repo root (or open a private security advisory on the GitHub repository). Do not file public issues for vulnerabilities.
