"""A2A-compatible HTTP server for CRUMB AgentAuth.

Uses only the Python standard library (http.server + json).  Run directly::

    python -m a2a.server            # default port 8421
    python -m a2a.server --port 9000

Endpoints
---------
GET  /.well-known/agent.json   Agent card (A2A discovery)
POST /tasks/send               Handle an A2A task request
GET  /health                   Health check
"""

import argparse
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Ensure project root is importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from a2a.agent_card import build_agent_card, DEFAULT_PORT  # noqa: E402
from a2a.task_handler import handle_task                    # noqa: E402


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class A2ARequestHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler implementing the A2A protocol surface."""

    server_version = "CRUMBAgentAuth/0.2.0"

    # ── helpers ───────────────────────────────────────────────

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    # ── GET ───────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/.well-known/agent.json":
            host = self.headers.get("Host", f"localhost:{self.server.server_port}")
            hostname = host.split(":")[0]
            card = build_agent_card(hostname, self.server.server_port)
            self._send_json(card)

        elif self.path == "/health":
            self._send_json({"status": "ok"})

        else:
            self._send_json({"error": "not found"}, status=404)

    # ── POST ──────────────────────────────────────────────────

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/tasks/send":
            raw = self._read_body()
            try:
                task = json.loads(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                self._send_json(
                    {"error": f"invalid JSON: {exc}"}, status=400
                )
                return
            response = handle_task(task)
            status_code = 200 if response.get("status") == "completed" else 400
            self._send_json(response, status=status_code)
        else:
            self._send_json({"error": "not found"}, status=404)

    # ── OPTIONS (CORS pre-flight) ─────────────────────────────

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # Suppress default stderr logging for cleaner output.
    def log_message(self, fmt, *args) -> None:  # noqa: N802
        sys.stderr.write(f"[a2a] {fmt % args}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(port: int = DEFAULT_PORT) -> None:
    server = HTTPServer(("0.0.0.0", port), A2ARequestHandler)
    print(f"CRUMB AgentAuth A2A server listening on http://localhost:{port}")
    print(f"  Agent card:  http://localhost:{port}/.well-known/agent.json")
    print(f"  Task submit: POST http://localhost:{port}/tasks/send")
    print(f"  Health:      http://localhost:{port}/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRUMB AgentAuth A2A server")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    args = parser.parse_args()
    main(port=args.port)
