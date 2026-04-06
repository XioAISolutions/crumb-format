#!/usr/bin/env python3
"""Minimal REST API for CRUMB operations."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cli.crumb import parse_crumb, render_crumb  # noqa: E402


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON body: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


class CrumbApiHandler(BaseHTTPRequestHandler):
    server_version = "crumb-api/0.2"

    def do_OPTIONS(self) -> None:
        json_response(self, 200, {"ok": True})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            json_response(
                self,
                200,
                {
                    "service": "crumb-api",
                    "endpoints": [
                        "GET /health",
                        "POST /crumb/validate",
                        "POST /crumb/parse",
                        "POST /crumb/render",
                    ],
                },
            )
            return

        if path == "/health":
            json_response(self, 200, {"status": "ok"})
            return

        json_response(self, 404, {"error": f"unknown route: {path}"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = read_json_body(self)
        except ValueError as exc:
            json_response(self, 400, {"error": str(exc)})
            return

        if path == "/crumb/validate":
            self.handle_validate(body)
            return

        if path == "/crumb/parse":
            self.handle_parse(body)
            return

        if path == "/crumb/render":
            self.handle_render(body)
            return

        json_response(self, 404, {"error": f"unknown route: {path}"})

    def handle_validate(self, body: dict) -> None:
        text = body.get("text", "")
        if not text:
            json_response(self, 400, {"error": "missing 'text' field"})
            return
        try:
            parse_crumb(text)
        except ValueError as exc:
            json_response(self, 200, {"valid": False, "error": str(exc)})
            return
        json_response(self, 200, {"valid": True, "error": None})

    def handle_parse(self, body: dict) -> None:
        text = body.get("text", "")
        if not text:
            json_response(self, 400, {"error": "missing 'text' field"})
            return
        try:
            parsed = parse_crumb(text)
        except ValueError as exc:
            json_response(self, 400, {"error": str(exc)})
            return
        json_response(self, 200, parsed)

    def handle_render(self, body: dict) -> None:
        headers = body.get("headers")
        sections = body.get("sections")
        if headers is None or sections is None:
            json_response(self, 400, {"error": "missing 'headers' and/or 'sections' fields"})
            return
        try:
            text = render_crumb(headers, sections)
        except Exception as exc:
            json_response(self, 400, {"error": str(exc)})
            return
        json_response(self, 200, {"text": text})

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the CRUMB REST API server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8420, help="Port to listen on (default: 8420).")
    args = parser.parse_args(argv)

    server = HTTPServer((args.host, args.port), CrumbApiHandler)
    print(f"crumb-api listening on http://{args.host}:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
