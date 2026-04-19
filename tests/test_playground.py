"""Tests for the /metalk/compress endpoint and playground static serving."""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.server import CrumbAPIHandler


@pytest.fixture(scope="module")
def server():
    """Start the API server on an ephemeral port for the duration of the module."""
    httpd = HTTPServer(("127.0.0.1", 0), CrumbAPIHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    # Tiny pause so the socket is accepting before the first request.
    time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def _post_json(base, path, body):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def _get(base, path):
    req = urllib.request.Request(base + path, method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type", "")


class TestCompressEndpoint:
    def test_plain_text_level_2(self, server):
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": "Please fix the authentication.", "level": 2})
        assert status == 200
        assert "encoded" in data
        assert data["stats"]["level"] == 2
        assert data["stats"]["mode"] == "plain"

    def test_plain_text_level_4_strips_vowels(self, server):
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": "Authentication middleware configuration.", "level": 4})
        assert status == 200
        assert "athntctn" in data["encoded"].lower() or "Athntctn" in data["encoded"]
        assert data["stats"]["vowel_retention_pct"] < 100

    def test_crumb_input_auto_detected(self, server):
        crumb = ("BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n"
                 "[goal]\nFix authentication.\nEND CRUMB\n")
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": crumb, "level": 4})
        assert status == 200
        assert data["stats"]["mode"] == "crumb"
        assert "vs=" in data["encoded"]
        assert "mt=4" in data["encoded"]

    def test_missing_text_returns_400(self, server):
        try:
            _post_json(server, "/metalk/compress", {"level": 2})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            return
        pytest.fail("expected HTTP 400")

    def test_invalid_level_returns_400(self, server):
        try:
            _post_json(server, "/metalk/compress", {"text": "hi", "level": 99})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            return
        pytest.fail("expected HTTP 400")

    def test_stats_shape(self, server):
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": "Authentication middleware test text.", "level": 3})
        assert status == 200
        for key in ("original_tokens", "encoded_tokens", "saved_tokens",
                    "pct_saved", "ratio", "vowels_removed",
                    "vowel_retention_pct", "level", "mode"):
            assert key in data["stats"], f"missing {key}"


class TestStaticServing:
    def test_playground_html_served(self, server):
        status, body, ctype = _get(server, "/playground.html")
        assert status == 200
        assert "text/html" in ctype
        assert b"crumb playground" in body.lower() or b"crumb" in body.lower()

    def test_root_serves_playground(self, server):
        status, body, ctype = _get(server, "/")
        assert status == 200
        assert "text/html" in ctype

    def test_directory_traversal_blocked(self, server):
        try:
            _get(server, "/../api/server.py")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            return
        pytest.fail("expected HTTP 404 for traversal attempt")
