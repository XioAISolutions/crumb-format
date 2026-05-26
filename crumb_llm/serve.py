"""Production multi-tenant API server for Crumb LLM.

Built for the sovereign AI micro-cloud model: run private AI
infrastructure, sell API access with tiered subscriptions,
oversubscribe because usage is bursty.

Economics (from real deployment data):
    2× H100 SXM @ $2.69/hr = ~$3,900/mo GPU cost
    Mixed subs ($49/$99/$199) × 80-180 customers = $14K+/mo
    Gross margin: ~$10K/mo (before bandwidth, support, tax)

The key trick is oversubscription + rate limiting. Most customers
don't use the API 24/7. Rate limits per tier ensure no single
customer can saturate the GPUs.

Features:
    - API key authentication
    - Per-key rate limiting (tokens/min, requests/min)
    - Tiered access levels (starter, pro, enterprise)
    - Usage tracking per key (for billing)
    - Request queuing under load
    - Health monitoring
    - Context pulling built in
    - OpenAI-compatible /v1/completions endpoint

Deploy:
    python -m crumb_llm.production_serve --ckpt /model --index /crumbs
    # → ready for Stripe webhook integration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

import torch

from .sample import load_checkpoint
from .cache import generate_cached
from .context_pull import CrumbIndex, build_index, pull_context


# ── Tier definitions ─────────────────────────────────────────────────


@dataclass
class Tier:
    name: str
    price_monthly: int          # USD
    tokens_per_minute: int      # rate limit
    requests_per_minute: int    # rate limit
    max_context_tokens: int     # max input + pulled context
    max_output_tokens: int      # max generation length
    context_pulling: bool       # access to /v1/pull-complete


TIERS = {
    "starter": Tier("starter", 49, tokens_per_minute=10_000, requests_per_minute=20,
                     max_context_tokens=512, max_output_tokens=128, context_pulling=False),
    "pro": Tier("pro", 99, tokens_per_minute=50_000, requests_per_minute=60,
                max_context_tokens=2048, max_output_tokens=512, context_pulling=True),
    "enterprise": Tier("enterprise", 199, tokens_per_minute=200_000, requests_per_minute=200,
                       max_context_tokens=8192, max_output_tokens=2048, context_pulling=True),
}


# ── API key + usage tracking ─────────────────────────────────────────


@dataclass
class APIKey:
    key: str
    tier: str
    owner: str
    created: float = field(default_factory=time.time)
    active: bool = True


@dataclass
class UsageRecord:
    tokens_in: int = 0
    tokens_out: int = 0
    requests: int = 0
    last_reset: float = field(default_factory=time.time)


class UsageTracker:
    """Per-key rate limiting and usage tracking."""

    def __init__(self):
        self._lock = threading.Lock()
        self._minute_usage: dict[str, UsageRecord] = defaultdict(UsageRecord)
        self._total_usage: dict[str, UsageRecord] = defaultdict(UsageRecord)

    def check_rate_limit(self, key: str, tier: Tier) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        with self._lock:
            now = time.time()
            rec = self._minute_usage[key]
            if now - rec.last_reset > 60:
                rec.tokens_in = 0
                rec.tokens_out = 0
                rec.requests = 0
                rec.last_reset = now
            if rec.requests >= tier.requests_per_minute:
                return False, f"rate limit: {tier.requests_per_minute} req/min"
            if rec.tokens_in + rec.tokens_out >= tier.tokens_per_minute:
                return False, f"rate limit: {tier.tokens_per_minute} tok/min"
            return True, ""

    def record(self, key: str, tokens_in: int, tokens_out: int) -> None:
        with self._lock:
            for store in (self._minute_usage, self._total_usage):
                rec = store[key]
                rec.tokens_in += tokens_in
                rec.tokens_out += tokens_out
                rec.requests += 1

    def get_usage(self, key: str) -> dict:
        with self._lock:
            total = self._total_usage.get(key, UsageRecord())
            return {
                "total_tokens_in": total.tokens_in,
                "total_tokens_out": total.tokens_out,
                "total_requests": total.requests,
            }


class KeyStore:
    """In-memory API key store. Swap for Redis/DB in production."""

    def __init__(self):
        self._keys: dict[str, APIKey] = {}
        demo = os.environ.get("CRUMB_LLM_DEMO_KEY", "crumb-demo-key")
        self._keys[demo] = APIKey(key=demo, tier="pro", owner="demo")

    def validate(self, key: str) -> Optional[APIKey]:
        ak = self._keys.get(key)
        if ak and ak.active:
            return ak
        return None

    def add(self, key: str, tier: str, owner: str) -> APIKey:
        ak = APIKey(key=key, tier=tier, owner=owner)
        self._keys[key] = ak
        return ak

    def list_keys(self) -> list[dict]:
        return [{"key": k.key[:8] + "...", "tier": k.tier, "owner": k.owner, "active": k.active}
                for k in self._keys.values()]


# ── Request queue ────────────────────────────────────────────────────


class RequestQueue:
    """Simple bounded queue for load management."""

    def __init__(self, max_concurrent: int = 4):
        self._sem = threading.Semaphore(max_concurrent)
        self._pending = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        acquired = self._sem.acquire(blocking=False)
        if acquired:
            with self._lock:
                self._pending += 1
        return acquired

    def release(self) -> None:
        self._sem.release()
        with self._lock:
            self._pending -= 1

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending


# ── HTTP handler ─────────────────────────────────────────────────────


class CrumbLLMHandler(BaseHTTPRequestHandler):
    model = None
    tokenizer = None
    index: Optional[CrumbIndex] = None
    key_store: KeyStore = KeyStore()
    usage: UsageTracker = UsageTracker()
    queue: RequestQueue = RequestQueue(max_concurrent=4)
    model_info: dict = {}
    start_time: float = time.time()
    production_mode: bool = False  # False = dev (no auth), True = full auth+tiers

    def _json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _auth(self) -> Optional[tuple[APIKey, Tier]]:
        if not self.production_mode:
            # Dev mode: no auth, enterprise tier for all requests.
            return APIKey(key="dev", tier="enterprise", owner="dev"), TIERS["enterprise"]
        auth = self.headers.get("Authorization", "")
        key = auth.replace("Bearer ", "").strip()
        if not key:
            key = self.headers.get("X-API-Key", "")
        ak = self.key_store.validate(key)
        if not ak:
            self._json(401, {"error": "invalid API key"})
            return None
        tier = TIERS.get(ak.tier)
        if not tier:
            self._json(403, {"error": f"unknown tier: {ak.tier}"})
            return None
        ok, reason = self.usage.check_rate_limit(ak.key, tier)
        if not ok:
            self._json(429, {"error": reason, "retry_after_seconds": 60})
            return None
        return ak, tier

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/v1/health":
            uptime = time.time() - self.start_time
            self._json(200, {
                "status": "ok",
                "uptime_seconds": round(uptime, 1),
                "model": self.model_info,
                "queue_pending": self.queue.pending,
                "index_sections": len(self.index.sections) if self.index else 0,
                "tiers": {name: {"price": t.price_monthly, "rpm": t.requests_per_minute,
                                 "tpm": t.tokens_per_minute}
                          for name, t in TIERS.items()},
            })
            return
        if self.path == "/v1/usage":
            auth = self._auth()
            if not auth:
                return
            ak, _ = auth
            self._json(200, self.usage.get_usage(ak.key))
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/v1/completions":
            self._handle_completion()
        elif self.path == "/v1/pull-complete":
            self._handle_pull_complete()
        elif self.path == "/v1/score":
            self._handle_score()
        else:
            self._json(404, {"error": "not found"})

    def _handle_completion(self) -> None:
        auth = self._auth()
        if not auth:
            return
        ak, tier = auth

        if not self.queue.try_acquire():
            self._json(503, {"error": "server busy, retry in a moment"})
            return

        try:
            data = self._read_json()
            prompt = data.get("prompt", "")
            max_tokens = min(data.get("max_tokens", 64), tier.max_output_tokens)
            temperature = data.get("temperature", 1.0)
            top_k = data.get("top_k", 40)

            t0 = time.time()
            ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
            ids = ids[:, :tier.max_context_tokens]
            out_ids = generate_cached(
                self.model, ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            text = self.tokenizer.decode(out_ids[0].tolist())
            dt = time.time() - t0
            n_in = ids.shape[1]
            n_out = out_ids.shape[1] - n_in

            self.usage.record(ak.key, n_in, n_out)
            self._json(200, {
                "text": text,
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
                "latency_ms": round(dt * 1000, 1),
                "tier": tier.name,
            })
        finally:
            self.queue.release()

    def _handle_pull_complete(self) -> None:
        auth = self._auth()
        if not auth:
            return
        ak, tier = auth

        if not tier.context_pulling:
            self._json(403, {"error": "context pulling requires pro or enterprise tier"})
            return

        if not self.queue.try_acquire():
            self._json(503, {"error": "server busy"})
            return

        try:
            data = self._read_json()
            query = data.get("query", "")
            max_tokens = min(data.get("max_tokens", 64), tier.max_output_tokens)

            pulled = ""
            if self.index:
                pulled = pull_context(query, self.index,
                                      max_tokens=min(256, tier.max_context_tokens // 2),
                                      top_k=data.get("top_k_sections", 5))
            augmented = (pulled + "\n" + query) if pulled else query

            t0 = time.time()
            ids = torch.tensor(self.tokenizer.encode(augmented), dtype=torch.long).unsqueeze(0)
            ids = ids[:, :tier.max_context_tokens]
            out_ids = generate_cached(self.model, ids, max_new_tokens=max_tokens,
                                      temperature=data.get("temperature", 1.0))
            text = self.tokenizer.decode(out_ids[0].tolist())
            dt = time.time() - t0
            n_in = ids.shape[1]
            n_out = out_ids.shape[1] - n_in

            self.usage.record(ak.key, n_in, n_out)
            self._json(200, {
                "text": text,
                "pulled_context_tokens": len(self.tokenizer.encode(pulled)),
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
                "latency_ms": round(dt * 1000, 1),
                "tier": tier.name,
            })
        finally:
            self.queue.release()

    def _handle_score(self) -> None:
        auth = self._auth()
        if not auth:
            return
        ak, tier = auth
        data = self._read_json()
        text = data.get("text", "")
        if not text:
            self._json(400, {"error": "missing 'text'"})
            return
        ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long).unsqueeze(0)
        ids = ids[:, :tier.max_context_tokens]
        with torch.no_grad():
            x, y = ids[:, :-1], ids[:, 1:]
            out = self.model(x, targets=y)
        loss = out["loss"].item()
        self.usage.record(ak.key, ids.size(1), 0)
        self._json(200, {
            "tokens": ids.size(1),
            "loss": round(loss, 4),
            "perplexity": round(math.exp(loss), 2),
        })

    def log_message(self, fmt, *args):
        pass


def serve(
    ckpt_dir: str | Path,
    port: int = 8090,
    host: str = "0.0.0.0",
    index_dir: str | Path | None = None,
    max_concurrent: int = 4,
    production: bool = False,
) -> None:
    model, tok = load_checkpoint(ckpt_dir)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    CrumbLLMHandler.model = model
    CrumbLLMHandler.tokenizer = tok
    CrumbLLMHandler.queue = RequestQueue(max_concurrent)
    CrumbLLMHandler.model_info = {
        "arch": "crumb_llm",
        "version": "0.2.0",
        "params": n_params,
        "field_size": getattr(model.cfg, "field_size", None),
    }

    CrumbLLMHandler.production_mode = production

    if index_dir:
        field_size = getattr(model.cfg, "field_size", 256)
        idx = build_index(index_dir, field_size=field_size)
        CrumbLLMHandler.index = idx
        print(f"[serve] indexed {len(idx.sections)} sections from {index_dir}")

    server = HTTPServer((host, port), CrumbLLMHandler)
    print(f"""
╔══════════════════════════════════════════════════════╗
║           Crumb LLM Production Server                ║
╠══════════════════════════════════════════════════════╣
║  http://{host}:{port}                               ║
║  params: {n_params:,}                               ║
║  concurrent slots: {max_concurrent}                  ║
║                                                      ║
║  POST /v1/completions    — generate text             ║
║  POST /v1/pull-complete  — context-pull + generate   ║
║  POST /v1/score          — perplexity scoring        ║
║  GET  /v1/health         — status + tier info        ║
║  GET  /v1/usage          — per-key usage stats       ║
║                                                      ║
║  Auth: Authorization: Bearer <key>                   ║
║  Demo key: crumb-demo-key (or CRUMB_LLM_DEMO_KEY)   ║
║                                                      ║
║  Tiers:                                              ║
║    starter  $49/mo   20 rpm   10K tpm                ║
║    pro      $99/mo   60 rpm   50K tpm  + pulling     ║
║    enterprise $199/mo 200 rpm 200K tpm + pulling     ║
╚══════════════════════════════════════════════════════╝
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[prod] shutting down")
        server.shutdown()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Crumb LLM Server")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--index", default=None)
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--production", action="store_true",
                    help="Enable API key auth, rate limiting, and tiered access.")
    args = ap.parse_args(argv)
    serve(args.ckpt, args.port, args.host, args.index, args.max_concurrent, args.production)


if __name__ == "__main__":
    main()
