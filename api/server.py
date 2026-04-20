#!/usr/bin/env python3
"""Lightweight REST API server for crumb-format and agentauth.

Uses ONLY the Python standard library (http.server, json).
Run with: python api/server.py [--port 8420]
"""

import argparse
import json
import re
import sys
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import cli.crumb and agentauth
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cli.crumb import parse_crumb, render_crumb  # noqa: E402
from cli.metalk import encode as metalk_encode, compression_stats as metalk_stats  # noqa: E402
from cli.vowelstrip import (  # noqa: E402
    drift_stats as vs_drift_stats,
    encode_crumb as vs_encode_crumb,
    strip_text as vs_strip_text,
)
from agentauth.store import PassportStore  # noqa: E402
from agentauth.passport import AgentPassport  # noqa: E402
from agentauth.policy import ToolPolicy  # noqa: E402
from agentauth.credentials import CredentialBroker  # noqa: E402
from agentauth.audit import AuditLogger  # noqa: E402

VERSION = "0.3.0"

# ---------------------------------------------------------------------------
# Shared instances (created once at startup)
# ---------------------------------------------------------------------------
store = PassportStore()
passport_mgr = AgentPassport(store)
policy_mgr = ToolPolicy(store)
cred_broker = CredentialBroker(store)
audit_logger = AuditLogger(store)

# ---------------------------------------------------------------------------
# Route table
# ---------------------------------------------------------------------------
# Each entry: (method, regex_pattern) -> handler_function
# Handler signature: handler(handler_self, match, query_params, body) -> (status, data)

ROUTES: list = []


def route(method: str, pattern: str):
    """Decorator to register a route."""
    compiled = re.compile(f"^{pattern}$")

    def decorator(func):
        ROUTES.append((method, compiled, func))
        return func

    return decorator


# ---------------------------------------------------------------------------
# CRUMB endpoints
# ---------------------------------------------------------------------------

@route("POST", "/crumb/validate")
def crumb_validate(_req, _match, _qs, body):
    text = body.get("text", "")
    if not text:
        return 400, {"error": "missing 'text' field"}
    try:
        parse_crumb(text)
        return 200, {"valid": True, "error": None}
    except ValueError as exc:
        return 200, {"valid": False, "error": str(exc)}


@route("POST", "/crumb/parse")
def crumb_parse(_req, _match, _qs, body):
    text = body.get("text", "")
    if not text:
        return 400, {"error": "missing 'text' field"}
    try:
        result = parse_crumb(text)
        return 200, result
    except ValueError as exc:
        return 400, {"error": str(exc)}


@route("POST", "/crumb/render")
def crumb_render(_req, _match, _qs, body):
    headers = body.get("headers")
    sections = body.get("sections")
    if headers is None or sections is None:
        return 400, {"error": "missing 'headers' and/or 'sections' fields"}
    # sections values may come as lists of strings already
    try:
        text = render_crumb(headers, sections)
        return 200, {"text": text}
    except Exception as exc:
        return 400, {"error": str(exc)}


# ---------------------------------------------------------------------------
# MeTalk / vowel-strip compression endpoint (powers the playground)
# ---------------------------------------------------------------------------

@route("POST", "/metalk/compress")
def metalk_compress(_req, _match, _qs, body):
    """Compress arbitrary text or a CRUMB through MeTalk levels 1-5.

    Body:
      text                 (str, required) — input text
      level                (int 1-5, default 2) — MeTalk level
      vowel_min_length     (int, default 4)   — L4-5 word length floor
      adaptive_threshold   (float, default 0.85) — L5 sim floor
      mode                 (str, default "auto") — "auto" | "crumb" | "plain"
                            auto = detect by BEGIN CRUMB sentinel
    """
    text = body.get("text", "")
    if not text:
        return 400, {"error": "missing 'text' field"}
    try:
        level = int(body.get("level", 2))
    except (TypeError, ValueError):
        return 400, {"error": "'level' must be an integer 1-5"}
    if level not in (1, 2, 3, 4, 5):
        return 400, {"error": "'level' must be 1, 2, 3, 4, or 5"}

    try:
        vml = int(body.get("vowel_min_length") or 4)
    except (TypeError, ValueError):
        return 400, {"error": "'vowel_min_length' must be a positive integer"}
    if vml < 1:
        return 400, {"error": "'vowel_min_length' must be a positive integer"}

    try:
        threshold = float(body.get("adaptive_threshold") or 0.85)
    except (TypeError, ValueError):
        return 400, {"error": "'adaptive_threshold' must be a number between 0 and 1"}
    if not (0.0 <= threshold <= 1.0):
        return 400, {"error": "'adaptive_threshold' must be between 0 and 1"}

    mode = body.get("mode", "auto")

    is_crumb = (
        mode == "crumb"
        or (mode == "auto" and text.lstrip().startswith(("BEGIN CRUMB", "BC")))
    )

    try:
        if is_crumb:
            encoded = metalk_encode(
                text, level=level,
                vowel_min_length=vml, adaptive_threshold=threshold,
            )
        else:
            # Plain prose at any level: wrap in a synthetic mem crumb so the
            # full MeTalk pipeline (dict + grammar + condense + vowel-strip)
            # runs, then strip the wrapper.
            wrapped = (
                "BEGIN CRUMB\nv=1.1\nkind=mem\ntitle=playground\n---\n"
                "[consolidated]\n" + text + "\nEND CRUMB\n"
            )
            mt = metalk_encode(
                wrapped, level=level,
                vowel_min_length=vml, adaptive_threshold=threshold,
            )
            try:
                _, after = mt.split("---\n", 1)
                body_lines = after.rstrip().split("\n")
                cleaned = [
                    ln for ln in body_lines
                    if ln.strip() not in {"EC", "END CRUMB"}
                    and not (ln.strip().startswith("[") and ln.strip().endswith("]"))
                ]
                encoded = "\n".join(cleaned).strip() + "\n"
            except ValueError:
                encoded = mt
    except Exception as exc:
        traceback.print_exc()
        return 500, {"error": f"compression failed: {exc}"}

    stats = metalk_stats(text, encoded)
    drift = vs_drift_stats(text, encoded)
    stats["vowels_removed"] = drift["vowels_removed"]
    stats["vowel_retention_pct"] = drift["vowel_retention_pct"]
    stats["level"] = level
    stats["mode"] = "crumb" if is_crumb else "plain"
    return 200, {"encoded": encoded, "stats": stats}


@route("POST", "/metalk/compare")
def metalk_compare(_req, _match, _qs, body):
    """Run MeTalk levels 1-5 over the same input in one request.

    Body: {text, vowel_min_length?, adaptive_threshold?, mode?}
    Returns: {levels: [{level, encoded, stats}, ...], original_tokens}
    """
    text = body.get("text", "")
    if not text:
        return 400, {"error": "missing 'text' field"}
    vml = int(body.get("vowel_min_length", 4) or 4)
    threshold = float(body.get("adaptive_threshold", 0.85) or 0.85)
    mode = body.get("mode", "auto")

    results = []
    for level in (1, 2, 3, 4, 5):
        try:
            status, data = metalk_compress(
                _req, _match, _qs,
                {"text": text, "level": level,
                 "vowel_min_length": vml, "adaptive_threshold": threshold,
                 "mode": mode},
            )
        except Exception:
            traceback.print_exc()
            results.append({"level": level, "error": "compression failed"})
            continue
        if status != 200 or "encoded" not in data:
            results.append({"level": level, "error": data.get("error", "unknown")})
            continue
        results.append({
            "level": level,
            "encoded": data["encoded"],
            "stats": data["stats"],
        })
    return 200, {
        "levels": results,
        "original_tokens": metalk_stats(text, text)["original_tokens"],
    }


# ---------------------------------------------------------------------------
# Passport endpoints
# ---------------------------------------------------------------------------

@route("POST", "/passport/register")
def passport_register(_req, _match, _qs, body):
    name = body.get("name")
    if not name:
        return 400, {"error": "missing 'name' field"}
    result = passport_mgr.register(
        name=name,
        framework=body.get("framework", "unknown"),
        owner=body.get("owner", ""),
        tools_allowed=body.get("tools_allowed"),
        tools_denied=body.get("tools_denied"),
        ttl_days=body.get("ttl_days", 90),
    )
    return 201, result


@route("GET", "/passport/(?P<id>[^/]+)/verify")
def passport_verify(_req, match, _qs, _body):
    agent_id = match.group("id")
    result = passport_mgr.verify(agent_id)
    status = 200 if result["valid"] else 200
    return status, result


@route("POST", "/passport/(?P<id>[^/]+)/revoke")
def passport_revoke(_req, match, _qs, _body):
    agent_id = match.group("id")
    success = passport_mgr.revoke(agent_id)
    if not success:
        return 404, {"error": f"passport '{agent_id}' not found or already revoked"}
    return 200, {"revoked": True, "agent_id": agent_id}


@route("GET", "/passport/(?P<id>[^/]+)")
def passport_inspect(_req, match, _qs, _body):
    agent_id = match.group("id")
    result = passport_mgr.inspect(agent_id)
    if result is None:
        return 404, {"error": f"passport '{agent_id}' not found"}
    return 200, result


@route("GET", "/passports")
def passports_list(_req, _match, qs, _body):
    status_filter = qs.get("status", ["all"])[0]
    results = passport_mgr.list_all(status_filter=status_filter)
    return 200, {"passports": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Policy endpoints
# ---------------------------------------------------------------------------

@route("POST", "/policy/set")
def policy_set(_req, _match, _qs, body):
    agent_name = body.get("agent_name")
    if not agent_name:
        return 400, {"error": "missing 'agent_name' field"}
    result = policy_mgr.set_policy(
        agent_name=agent_name,
        tools_allowed=body.get("tools_allowed"),
        tools_denied=body.get("tools_denied"),
        data_classes=body.get("data_classes"),
        max_actions_per_session=body.get("max_actions_per_session", 1000),
    )
    return 201, result


@route("POST", "/policy/check")
def policy_check(_req, _match, _qs, body):
    agent_id = body.get("agent_id")
    tool = body.get("tool")
    if not agent_id or not tool:
        return 400, {"error": "missing 'agent_id' and/or 'tool' fields"}
    result = policy_mgr.check(agent_id, tool)
    return 200, result


# ---------------------------------------------------------------------------
# Credential endpoints
# ---------------------------------------------------------------------------

@route("POST", "/credential/issue")
def credential_issue(_req, _match, _qs, body):
    agent_id = body.get("agent_id")
    tool = body.get("tool")
    if not agent_id or not tool:
        return 400, {"error": "missing 'agent_id' and/or 'tool' fields"}
    try:
        result = cred_broker.issue(
            agent_id=agent_id,
            tool=tool,
            ttl_seconds=body.get("ttl_seconds", 300),
        )
        return 201, result
    except PermissionError as exc:
        return 403, {"error": str(exc)}


@route("POST", "/credential/validate")
def credential_validate(_req, _match, _qs, body):
    token = body.get("token")
    agent_id = body.get("agent_id")
    tool = body.get("tool")
    if not token or not agent_id or not tool:
        return 400, {"error": "missing 'token', 'agent_id', and/or 'tool' fields"}
    result = cred_broker.validate(token=token, agent_id=agent_id, tool=tool)
    return 200, result


# ---------------------------------------------------------------------------
# Audit endpoints
# ---------------------------------------------------------------------------

@route("POST", "/audit/start")
def audit_start(_req, _match, _qs, body):
    agent_id = body.get("agent_id")
    goal = body.get("goal")
    if not agent_id or not goal:
        return 400, {"error": "missing 'agent_id' and/or 'goal' fields"}
    session_id = audit_logger.start_session(agent_id=agent_id, goal=goal)
    return 201, {"session_id": session_id, "agent_id": agent_id}


@route("POST", "/audit/log")
def audit_log(_req, _match, _qs, body):
    session_id = body.get("session_id")
    tool = body.get("tool")
    detail = body.get("detail", "")
    allowed = body.get("allowed")
    if not session_id or not tool or allowed is None:
        return 400, {"error": "missing 'session_id', 'tool', and/or 'allowed' fields"}
    try:
        audit_logger.log_action(
            session_id=session_id,
            tool=tool,
            detail=detail,
            allowed=allowed,
            reason=body.get("reason", ""),
        )
        return 200, {"logged": True, "session_id": session_id}
    except ValueError as exc:
        return 404, {"error": str(exc)}


@route("POST", "/audit/end")
def audit_end(_req, _match, _qs, body):
    session_id = body.get("session_id")
    if not session_id:
        return 400, {"error": "missing 'session_id' field"}
    status = body.get("status", "completed")
    try:
        content = audit_logger.end_session(session_id=session_id, status=status)
        return 200, {"session_id": session_id, "status": status, "crumb": content}
    except ValueError as exc:
        return 404, {"error": str(exc)}


@route("GET", "/audit/export")
def audit_export(_req, _match, qs, _body):
    agent_id = qs.get("agent_id", [None])[0]
    since = qs.get("since", [None])[0]
    fmt = qs.get("format", ["crumb"])[0]
    result = audit_logger.export_evidence(
        agent_id=agent_id, since=since, output_format=fmt
    )
    if fmt == "json":
        # Already JSON string, parse so it gets re-serialized properly
        try:
            parsed = json.loads(result)
            return 200, {"format": fmt, "data": parsed}
        except json.JSONDecodeError:
            return 200, {"format": fmt, "data": result}
    return 200, {"format": fmt, "data": result}


@route("GET", "/audit/feed")
def audit_feed(_req, _match, qs, _body):
    agent_id = qs.get("agent_id", [None])[0]
    lines = audit_logger.feed(agent_id=agent_id)
    return 200, {"entries": lines, "count": len(lines)}


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@route("GET", "/health")
def health(_req, _match, _qs, _body):
    return 200, {"status": "ok", "version": VERSION}


# ---------------------------------------------------------------------------
# Static file serving (web/ directory) — powers the playground UI
# ---------------------------------------------------------------------------

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json",
}


def _serve_static(handler, rel_path: str) -> bool:
    """Serve a file from web/ if it exists. Returns True if handled."""
    if not rel_path or rel_path == "/":
        rel_path = "playground.html"
    rel_path = rel_path.lstrip("/")
    # Prevent directory traversal
    safe = (WEB_DIR / rel_path).resolve()
    try:
        safe.relative_to(WEB_DIR.resolve())
    except ValueError:
        return False
    if not safe.is_file():
        return False
    data = safe.read_bytes()
    ctype = _STATIC_TYPES.get(safe.suffix.lower(), "application/octet-stream")
    handler.send_response(200)
    handler._add_cors_headers()
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
    return True


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class CrumbAPIHandler(BaseHTTPRequestHandler):
    """Routes requests through the ROUTES table."""

    server_version = f"CrumbAPI/{VERSION}"

    # Silence per-request log lines (override to re-enable)
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[crumb-api] {fmt % args}\n")

    # ── CORS preflight ────────────────────────────────────────
    def do_OPTIONS(self):
        self._send_cors_preflight()

    def _send_cors_preflight(self):
        self.send_response(204)
        self._add_cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _add_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")

    # ── Dispatchers ───────────────────────────────────────────
    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _dispatch(self, method: str):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)

        for route_method, pattern, handler in ROUTES:
            if route_method != method:
                continue
            m = pattern.match(path)
            if m:
                body = self._read_json_body() if method == "POST" else {}
                try:
                    status_code, data = handler(self, m, qs, body)
                except Exception:
                    traceback.print_exc()
                    status_code, data = 500, {"error": "internal server error"}
                self._send_json(status_code, data)
                return

        # Fall through to static file serving for GETs (powers /playground.html
        # and any sibling assets dropped under web/).
        if method == "GET":
            if _serve_static(self, path):
                return

        self._send_json(404, {"error": f"not found: {method} {path}"})

    def _send_json(self, status_code: int, data):
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status_code)
        self._add_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

ENDPOINT_LIST = [
    ("POST", "/crumb/validate", "Validate a crumb document"),
    ("POST", "/crumb/parse", "Parse crumb text to JSON"),
    ("POST", "/crumb/render", "Render JSON to crumb text"),
    ("POST", "/metalk/compress", "Compress text via MeTalk levels 1-5"),
    ("GET",  "/playground.html", "Browser-based prompt compression UI"),
    ("POST", "/passport/register", "Register a new agent passport"),
    ("GET",  "/passport/:id", "Inspect a passport"),
    ("GET",  "/passport/:id/verify", "Verify passport validity"),
    ("POST", "/passport/:id/revoke", "Revoke a passport"),
    ("GET",  "/passports", "List all passports"),
    ("POST", "/policy/set", "Set tool policy for an agent"),
    ("POST", "/policy/check", "Check if tool is allowed"),
    ("POST", "/credential/issue", "Issue a short-lived credential"),
    ("POST", "/credential/validate", "Validate a credential token"),
    ("POST", "/audit/start", "Start an audit session"),
    ("POST", "/audit/log", "Log an action in a session"),
    ("POST", "/audit/end", "End an audit session"),
    ("GET",  "/audit/export", "Export audit evidence"),
    ("GET",  "/audit/feed", "Live audit feed"),
    ("GET",  "/health", "Health check"),
]


def print_banner(port: int):
    print(f"""
 ____                  _        _    ____ ___
/ ___|_ __ _   _ _ __ | |__    / \\  |  _ \\_ _|
| |   | '__| | | | '_ \\| '_ \\  / _ \\ | |_) | |
| |___| |  | |_| | | | | |_) |/ ___ \\|  __/| |
 \\____|_|   \\__,_|_| |_|_.__//_/   \\_\\_|  |___|
                                         v{VERSION}
Listening on http://0.0.0.0:{port}
""")
    print("Available endpoints:")
    print("-" * 60)
    for method, path, desc in ENDPOINT_LIST:
        print(f"  {method:<5} {path:<30} {desc}")
    print("-" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Crumb Format REST API server")
    parser.add_argument("--port", type=int, default=8420, help="Port to listen on (default: 8420)")
    args = parser.parse_args()

    print_banner(args.port)

    server = HTTPServer(("0.0.0.0", args.port), CrumbAPIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
