#!/usr/bin/env python3
"""Generate the benchmark corpus deterministically.

The corpus is committed, not generated at test time, so the numbers in
``baseline.json`` describe fixed inputs. This script exists so the corpus can
be regenerated and reviewed as a diff rather than being opaque blobs.

Two things it deliberately provides:

1. **The same conversation in every supported format.** Compression and
   retention must not depend on which provider exported the transcript. Any
   divergence between these five is a reader bug, and only a shared corpus can
   show it.
2. **A long session.** Every extraction fault found so far appeared only at
   length: fixed item caps discarded most of a long transcript, head-truncation
   described stale state, and markdown headings and narration were mistaken for
   content. A 40-turn fixture cannot exercise any of that.

Run: ``python benchmarks/build_corpus.py``
"""

from __future__ import annotations

import json
from pathlib import Path

CORPUS = Path(__file__).resolve().parent / "corpus"

# ── the canonical short conversation ────────────────────────────────
# Mirrors examples/transcript-checkout-openai.json so the corpus numbers stay
# comparable with the example quoted in the README.

TURNS: list[tuple[str, str]] = [
    ("user",
     "Users lose their cart when they refresh on /checkout/payment. It only happens after "
     "returning from the Stripe redirect. You must not change the public CartService API, and "
     "don't touch the webhook handler in src/api/stripe-webhook.ts — that's owned by another team."),
]
_HEDGE = (
    "Let me think about this carefully. There are a few possible explanations and I want to "
    "rule them out one at a time before committing to a diagnosis. It could be a hydration "
    "mismatch, a cookie scope issue, or storage being cleared by the navigation itself. "
)
_FILES = ["src/checkout/CartService.ts", "src/middleware.ts", "tests/checkout_refresh.spec.ts"]
for _i in range(18):
    _f = _FILES[_i % 3]
    TURNS.append(("assistant",
                  _HEDGE + f"I read through {_f} again on this pass. The hydrate path looks "
                  f"reasonable in isolation, though it assumes storage survives the navigation."))
    TURNS.append(("user",
                  f"Okay, that makes sense. Can you keep digging? I'm worried about the "
                  f"interaction between {_f} and the redirect."))
TURNS.append(("assistant",
              "Found it. sessionStorage is cleared by the cross-origin redirect back from Stripe, "
              "so CartService.hydrate() reads null on the first paint. We're going with a signed "
              "cookie instead: move hydration into src/middleware.ts so it runs before React "
              "mounts. The hydrate call currently costs 450ms and the edge returns a 503 when it "
              "misses. Decision: read the cart from a signed cookie in middleware, keeping "
              "CartService.hydrate() as the public entry point so the API is unchanged. Spec for "
              "the cookie format is at https://example.com/docs/signed-cart. The next step is to "
              "add a regression test in tests/checkout_refresh.spec.ts covering the post-redirect "
              "refresh path."))
TURNS.append(("user",
              "Good. Make sure we keep the existing cart_v2 cookie name and never store PII in "
              "the payload. Write the fix."))


def openai_messages() -> dict:
    return {"model": "gpt-4o",
            "messages": [{"role": r, "content": c} for r, c in TURNS]}


def anthropic_messages() -> dict:
    messages = []
    for role, content in TURNS:
        blocks: list[dict] = [{"type": "text", "text": content}]
        if role == "assistant":
            blocks.insert(0, {"type": "thinking", "thinking": "private reasoning, must be dropped"})
        messages.append({"role": role, "content": blocks})
    return {"model": "claude-opus-4", "messages": messages}


def chatgpt_export() -> list:
    mapping = {
        f"node-{i}": {"message": {"create_time": 1700000000 + i,
                                  "author": {"role": role},
                                  "content": {"parts": [content]}}}
        for i, (role, content) in enumerate(TURNS)
    }
    return [{"title": "Checkout cart drop", "mapping": mapping}]


def claude_export() -> list:
    return [{"name": "Checkout cart drop",
             "chat_messages": [{"sender": "human" if r == "user" else "assistant", "text": c}
                               for r, c in TURNS]}]


def claude_code_transcript() -> str:
    lines = []
    for i, (role, content) in enumerate(TURNS):
        blocks: list[dict] = [{"type": "text", "text": content}]
        if role == "assistant":
            blocks.insert(0, {"type": "thinking", "thinking": "private reasoning, must be dropped"})
        lines.append(json.dumps({
            "type": role, "sessionId": "bench-short", "uuid": f"u{i}",
            "timestamp": f"2026-01-01T00:{i:02d}:00Z", "cwd": "/work", "gitBranch": "main",
            "message": {"role": role, "content": blocks},
        }))
        if i % 7 == 0:  # interleave the bookkeeping records a real file carries
            lines.append(json.dumps({"type": "queue-operation", "sessionId": "bench-short"}))
    return "\n".join(lines) + "\n"


# ── the long session ────────────────────────────────────────────────

def long_session() -> str:
    """~160 turns touching many files, so budgets and recency actually engage."""
    lines = []
    idx = 0

    def emit(role: str, content: str) -> None:
        nonlocal idx
        lines.append(json.dumps({
            "type": role, "sessionId": "bench-long", "uuid": f"u{idx}",
            "timestamp": f"2026-01-02T00:00:{idx % 60:02d}Z", "cwd": "/work",
            "message": {"role": role, "content": [{"type": "text", "text": content}]},
        }))
        idx += 1

    emit("user", "Migrate the billing service off the legacy ledger. You must not change the "
                 "public /v1/invoices contract, and never run a backfill on a Friday.")
    for i in range(78):
        emit("assistant",
             f"Working through module {i}. I inspected services/billing/module_{i:02d}.py and "
             f"traced its call path. Nothing conclusive on this pass.")
        emit("user", f"Understood, keep going on module {i}.")
    # The decisions that actually matter land at the end, which is what makes
    # head-truncation the wrong strategy for a handoff.
    emit("assistant",
         "Found the root cause. The ledger writer double-counts refunds because "
         "services/billing/ledger.py applies the delta twice on retry. Decision: make the writer "
         "idempotent keyed on transfer_id. Latency on the reconcile path is 1200ms and the "
         "endpoint returns a 502 under load. Spec: https://example.com/docs/ledger-idempotency. "
         "The next step is to add a regression test in tests/billing/test_ledger_retry.py.")
    emit("user", "Good. Keep the existing ledger_v3 table name and never drop rows. Ship it.")
    return "\n".join(lines) + "\n"


def tool_heavy_session() -> str:
    """An agent session dominated by tool traffic, which is the real shape.

    A sample of a genuine Claude Code transcript ran 427 tool_use and 426
    tool_result blocks against 162 text blocks — tool traffic is the majority of
    a real session, and none of the other fixtures contain any. Without this,
    tool-result-clearing strategies measure nothing and look free.
    """
    lines = []
    idx = 0

    def emit(role: str, blocks: list) -> None:
        nonlocal idx
        lines.append(json.dumps({
            "type": role, "sessionId": "bench-tools", "uuid": f"u{idx}",
            "timestamp": f"2026-01-03T00:00:{idx % 60:02d}Z", "cwd": "/work",
            "message": {"role": role, "content": blocks},
        }))
        idx += 1

    emit("user", [{"type": "text", "text":
                   "The nightly export job silently drops rows. Find out why. Do not change the "
                   "public exporter interface, and never delete from warehouse.exports."}])
    for i in range(30):
        emit("assistant", [
            {"type": "text", "text": f"Checking partition {i}."},
            {"type": "tool_use", "name": "read_file",
             "input": {"path": f"jobs/export/partition_{i:02d}.py"}},
        ])
        emit("user", [{"type": "tool_result", "content":
                       f"def run(): return fetch(limit=1000, offset={i * 1000})  # rows: {950 + i}"}])
    emit("assistant", [{"type": "text", "text":
                        "Root cause: fetch() pages with a hard limit of 1000 but the offset stride "
                        "assumes 1000 rows returned, so any short page skips the remainder. The "
                        "export loses roughly 4200 rows per night and the job reports a 200. "
                        "Decision: page by cursor on export_id instead of offset. Spec: "
                        "https://example.com/docs/cursor-paging. Next step is a regression test in "
                        "tests/export/test_paging.py."}])
    emit("user", [{"type": "text", "text":
                   "Good. Keep the exports_v2 table name and never backfill without a dry run."}])
    return "\n".join(lines) + "\n"


def main() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in [
        ("short-openai.json", openai_messages()),
        ("short-anthropic.json", anthropic_messages()),
        ("short-chatgpt-export.json", chatgpt_export()),
        ("short-claude-export.json", claude_export()),
    ]:
        (CORPUS / name).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        written.append(name)
    for name, text in [
        ("short-claude-code.jsonl", claude_code_transcript()),
        ("long-claude-code.jsonl", long_session()),
        ("tools-claude-code.jsonl", tool_heavy_session()),
    ]:
        (CORPUS / name).write_text(text, encoding="utf-8")
        written.append(name)
    for name in written:
        print(f"  wrote corpus/{name}  ({(CORPUS / name).stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
