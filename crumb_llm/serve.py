"""Lightweight HTTP server for Crumb LLM inference.

The cheapest deployment path: a single-file server that loads a
checkpoint and serves completions over HTTP. No CUDA required (runs
on CPU), no heavy framework dependencies (just stdlib + torch).

Endpoints:
    POST /v1/completions     — generate text from a prompt
    POST /v1/pull-complete   — context-pull from index + generate
    POST /v1/score           — score a crumb's perplexity
    GET  /v1/health          — health check + model info
    GET  /v1/index/stats     — context index statistics

Deploy options (cheapest first):
    1. Local:     python -m crumb_llm.serve --ckpt /path/to/run
    2. Docker:    docker run -v /ckpt:/ckpt crumb-llm --ckpt /ckpt
    3. Fly.io:    fly launch (see Dockerfile in repo)
    4. Lambda:    package as a zip, mount EFS for the checkpoint
    5. HF Spaces: push hub export, use Gradio/FastAPI frontend

The server is stateless per request but can maintain a CrumbIndex
for context-pulling across requests.

Usage:
    python -m crumb_llm.serve --ckpt /tmp/crumb_run --port 8090

    curl -X POST http://localhost:8090/v1/completions \\
        -H 'Content-Type: application/json' \\
        -d '{"prompt": "BEGIN CRUMB", "max_tokens": 100}'
"""

from __future__ import annotations

import argparse
import json
import math
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

import torch

from .sample import load_checkpoint
from .cache import generate_cached
from .context_pull import CrumbIndex, pull_context, build_index


class CrumbLLMHandler(BaseHTTPRequestHandler):
    """HTTP handler for Crumb LLM inference."""

    model = None
    tokenizer = None
    index: Optional[CrumbIndex] = None
    model_info: dict = {}

    def _json_response(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body) if body else {}

    def do_GET(self) -> None:
        if self.path == "/v1/health":
            self._json_response(200, {
                "status": "ok",
                "model": self.model_info,
                "index_sections": len(self.index.sections) if self.index else 0,
            })
            return
        if self.path == "/v1/index/stats":
            if self.index is None:
                self._json_response(200, {"indexed": False})
                return
            by_file: dict[str, int] = {}
            for s in self.index.sections:
                by_file[s.source_file] = by_file.get(s.source_file, 0) + 1
            self._json_response(200, {
                "indexed": True,
                "total_sections": len(self.index.sections),
                "files": len(by_file),
                "field_size": self.index.field_size,
                "sections_per_file": by_file,
            })
            return
        self._json_response(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            data = self._read_json()
        except (json.JSONDecodeError, ValueError) as e:
            self._json_response(400, {"error": f"invalid JSON: {e}"})
            return

        if self.path == "/v1/completions":
            self._handle_completions(data)
        elif self.path == "/v1/pull-complete":
            self._handle_pull_complete(data)
        elif self.path == "/v1/score":
            self._handle_score(data)
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_completions(self, data: dict) -> None:
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 64)
        temperature = data.get("temperature", 1.0)
        top_k = data.get("top_k", 40)

        t0 = time.time()
        ids = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.long
        ).unsqueeze(0)
        out_ids = generate_cached(
            self.model, ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        text = self.tokenizer.decode(out_ids[0].tolist())
        dt = time.time() - t0
        n_generated = out_ids.shape[1] - ids.shape[1]

        self._json_response(200, {
            "text": text,
            "prompt_tokens": ids.shape[1],
            "generated_tokens": n_generated,
            "total_tokens": out_ids.shape[1],
            "latency_ms": round(dt * 1000, 1),
            "tokens_per_second": round(n_generated / max(dt, 0.001), 1),
        })

    def _handle_pull_complete(self, data: dict) -> None:
        """Context-pulling completion: retrieve relevant crumb sections,
        prepend them to the prompt, then generate."""
        query = data.get("query", "")
        max_context = data.get("max_context_tokens", 256)
        max_tokens = data.get("max_tokens", 64)
        temperature = data.get("temperature", 1.0)
        top_k_sections = data.get("top_k_sections", 5)

        if self.index is None:
            self._json_response(400, {"error": "no crumb index loaded"})
            return

        # Pull relevant context.
        pulled = pull_context(
            query, self.index,
            max_tokens=max_context,
            top_k=top_k_sections,
        )
        augmented_prompt = pulled + "\n" + query if pulled else query

        t0 = time.time()
        ids = torch.tensor(
            self.tokenizer.encode(augmented_prompt), dtype=torch.long
        ).unsqueeze(0)
        out_ids = generate_cached(
            self.model, ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        text = self.tokenizer.decode(out_ids[0].tolist())
        dt = time.time() - t0

        self._json_response(200, {
            "text": text,
            "pulled_context": pulled,
            "pulled_tokens": len(self.tokenizer.encode(pulled)),
            "total_tokens": out_ids.shape[1],
            "latency_ms": round(dt * 1000, 1),
        })

    def _handle_score(self, data: dict) -> None:
        text = data.get("text", "")
        if not text:
            self._json_response(400, {"error": "missing 'text' field"})
            return

        ids = torch.tensor(
            self.tokenizer.encode(text), dtype=torch.long
        ).unsqueeze(0)
        max_ctx = getattr(self.model.cfg, "field_size", ids.size(1))
        ids = ids[:, :max_ctx]

        with torch.no_grad():
            x, y = ids[:, :-1], ids[:, 1:]
            out = self.model(x, targets=y)

        loss = out["loss"].item()
        self._json_response(200, {
            "tokens": ids.size(1),
            "loss": round(loss, 4),
            "perplexity": round(math.exp(loss), 2),
            "bpc": round(loss / math.log(2), 3),
        })

    def log_message(self, fmt, *args):
        pass  # Suppress default logging.


def serve(
    ckpt_dir: str | Path,
    port: int = 8090,
    host: str = "0.0.0.0",
    index_dir: str | Path | None = None,
) -> None:
    """Start the Crumb LLM inference server."""
    print(f"[serve] loading checkpoint from {ckpt_dir}")
    model, tok = load_checkpoint(ckpt_dir)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    CrumbLLMHandler.model = model
    CrumbLLMHandler.tokenizer = tok
    CrumbLLMHandler.model_info = {
        "arch": "crumb_llm",
        "params": n_params,
        "field_size": getattr(model.cfg, "field_size", None),
        "dim": getattr(model.cfg, "dim", None),
        "n_layers": getattr(model.cfg, "n_layers", None),
    }

    if index_dir:
        print(f"[serve] building context index from {index_dir}")
        field_size = getattr(model.cfg, "field_size", 256)
        idx = build_index(index_dir, field_size=field_size)
        CrumbLLMHandler.index = idx
        print(f"[serve] indexed {len(idx.sections)} sections")

    server = HTTPServer((host, port), CrumbLLMHandler)
    print(f"[serve] Crumb LLM ready on http://{host}:{port}")
    print(f"[serve] params={n_params:,}  field_size={CrumbLLMHandler.model_info['field_size']}")
    print(f"[serve] endpoints:")
    print(f"  POST /v1/completions     — generate text")
    print(f"  POST /v1/pull-complete   — context-pull + generate")
    print(f"  POST /v1/score           — perplexity scoring")
    print(f"  GET  /v1/health          — health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] shutting down")
        server.shutdown()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Serve Crumb LLM over HTTP.")
    ap.add_argument("--ckpt", required=True, help="Checkpoint directory.")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--index", default=None,
                    help="Directory of .crumb files to index for context pulling.")
    args = ap.parse_args(argv)
    serve(ckpt_dir=args.ckpt, port=args.port, host=args.host, index_dir=args.index)


if __name__ == "__main__":
    main()
