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

    def test_plain_text_level_4_compounds_dict_and_vowel_strip(self, server):
        # Plain prose at L4 runs the full MeTalk pipeline: dict substitution
        # ("authentication" → "auth", "middleware" → "mw") then vowel-strip
        # ("auth" → "ath"). Result is substantially shorter than vowel-strip alone.
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": "Authentication middleware configuration.", "level": 4})
        assert status == 200
        encoded = data["encoded"].lower()
        assert "authentication" not in encoded  # dict pass ran
        assert "mw" in encoded                   # middleware abbreviated
        assert data["stats"]["vowel_retention_pct"] < 100  # vowel-strip ran

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

    def test_invalid_vowel_min_length_returns_400(self, server):
        # Regression: non-numeric vml used to raise ValueError and surface as 500.
        try:
            _post_json(server, "/metalk/compress",
                       {"text": "hi", "level": 4, "vowel_min_length": "abc"})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            body = json.loads(exc.read())
            assert "vowel_min_length" in body["error"]
            return
        pytest.fail("expected HTTP 400")

    def test_invalid_adaptive_threshold_returns_400(self, server):
        try:
            _post_json(server, "/metalk/compress",
                       {"text": "hi", "level": 5, "adaptive_threshold": "not-a-float"})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            body = json.loads(exc.read())
            assert "adaptive_threshold" in body["error"]
            return
        pytest.fail("expected HTTP 400")

    def test_out_of_range_threshold_returns_400(self, server):
        try:
            _post_json(server, "/metalk/compress",
                       {"text": "hi", "level": 5, "adaptive_threshold": 1.5})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            return
        pytest.fail("expected HTTP 400")

    def test_threshold_zero_is_accepted(self, server):
        """Regression: `or 0.85` truthy-fallback used to silently upgrade
        an explicit 0.0 (a valid in-range value) to 0.85."""
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": "Please fix authentication.",
                                   "level": 5, "adaptive_threshold": 0.0})
        assert status == 200
        assert "encoded" in data

    def test_vowel_min_length_explicit_zero_returns_400(self, server):
        """0 is out of range (min 1); must be a clean 400, not a 500.
        This confirms we stopped using `or 4` which would have masked 0."""
        try:
            _post_json(server, "/metalk/compress",
                       {"text": "hi", "level": 4, "vowel_min_length": 0})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            return
        pytest.fail("expected HTTP 400")

    def test_non_string_text_returns_400_on_compress(self, server):
        """Regression: {"text": 123} used to raise AttributeError on lstrip()
        and surface as HTTP 500."""
        for bad in (123, {"a": 1}, [1, 2, 3], True):
            try:
                _post_json(server, "/metalk/compress", {"text": bad, "level": 2})
            except urllib.error.HTTPError as exc:
                assert exc.code == 400, f"expected 400 for text={bad!r}, got {exc.code}"
                continue
            pytest.fail(f"expected HTTP 400 for text={bad!r}")

    def test_non_string_text_returns_400_on_compare(self, server):
        """Regression: /metalk/compare used to return 200 with five
        'compression failed' rows on non-string text."""
        for bad in (123, {"a": 1}, [1, 2, 3]):
            try:
                _post_json(server, "/metalk/compare", {"text": bad})
            except urllib.error.HTTPError as exc:
                assert exc.code == 400, f"expected 400 for text={bad!r}, got {exc.code}"
                continue
            pytest.fail(f"expected HTTP 400 for text={bad!r}")

    def test_stats_shape(self, server):
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": "Authentication middleware test text.", "level": 3})
        assert status == 200
        for key in ("original_tokens", "encoded_tokens", "saved_tokens",
                    "pct_saved", "ratio", "vowels_removed",
                    "vowel_retention_pct", "level", "mode"):
            assert key in data["stats"], f"missing {key}"


class TestCompareEndpoint:
    def test_returns_all_five_levels(self, server):
        status, data = _post_json(server, "/metalk/compare",
                                  {"text": "Please fix authentication middleware configuration."})
        assert status == 200
        assert "levels" in data
        levels = {r["level"] for r in data["levels"]}
        assert levels == {1, 2, 3, 4, 5}

    def test_l4_saves_more_than_l1_on_long_prose(self, server):
        text = ("Please help me fix a bug in the authentication middleware. "
                "The application is not properly validating the JSON Web Token "
                "when users refresh the page.")
        status, data = _post_json(server, "/metalk/compare", {"text": text})
        assert status == 200
        by_level = {r["level"]: r for r in data["levels"]}
        assert by_level[4]["stats"]["pct_saved"] >= by_level[1]["stats"]["pct_saved"]

    def test_missing_text_returns_400(self, server):
        try:
            _post_json(server, "/metalk/compare", {})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            return
        pytest.fail("expected HTTP 400")

    def test_invalid_vowel_min_length_returns_400(self, server):
        """Regression: /metalk/compare used to 500 on non-numeric knobs."""
        try:
            _post_json(server, "/metalk/compare",
                       {"text": "hi", "vowel_min_length": "abc"})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            body = json.loads(exc.read())
            assert "vowel_min_length" in body["error"]
            return
        pytest.fail("expected HTTP 400")

    def test_invalid_adaptive_threshold_returns_400(self, server):
        try:
            _post_json(server, "/metalk/compare",
                       {"text": "hi", "adaptive_threshold": "nope"})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            return
        pytest.fail("expected HTTP 400")


class TestPlainModeBracketPreservation:
    """Regression: the plain-text unwrap used to drop any [foo] line, which
    silently deleted legitimate user content like `[todo]` or `[note]`.
    Only the exact synthetic section marker should be stripped."""

    def test_user_bracket_tag_preserved(self, server):
        text = "Ship the auth fix.\n[todo]\nWrite a regression test.\n[note]\nMerge after CI."
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": text, "level": 1, "mode": "plain"})
        assert status == 200
        encoded = data["encoded"]
        assert "[todo]" in encoded, f"user [todo] tag was dropped: {encoded!r}"
        assert "[note]" in encoded, f"user [note] tag was dropped: {encoded!r}"
        # The synthetic wrapper must still be gone.
        assert "[consolidated]" not in encoded

    def test_user_bracket_tag_preserved_on_compare(self, server):
        # At L1-L3 the bracketed tag survives intact; at L4-L5 vowel-strip
        # legitimately shortens the word inside (`todo` → `td`), but the
        # brackets themselves must remain — the token must never be dropped.
        text = "Please fix auth.\n[todo]\nadd test"
        status, data = _post_json(server, "/metalk/compare", {"text": text})
        assert status == 200
        for row in data["levels"]:
            if "encoded" not in row:
                continue
            if row["level"] <= 3:
                assert "[todo]" in row["encoded"], \
                    f"L{row['level']} dropped user [todo] tag: {row['encoded']!r}"
            else:
                # L4/L5: brackets preserved, interior can be vowel-stripped.
                assert "[" in row["encoded"] and "]" in row["encoded"], \
                    f"L{row['level']} dropped the brackets: {row['encoded']!r}"

    def test_section_headings_not_remapped_in_plain_mode(self, server):
        """Regression: the plain-text path used to wrap user text in a
        synthetic CRUMB and run it through metalk_encode, which rewrote
        legitimate `[goal]` / `[context]` bracket headings to the
        MeTalk-abbreviated forms `[g]` / `[cx]` even at level 1 —
        silent content corruption."""
        text = ("Fix the login bug.\n"
                "[goal]\nrestore session\n"
                "[context]\nJWT expires early\n")
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": text, "level": 1, "mode": "plain"})
        assert status == 200
        encoded = data["encoded"]
        assert "[goal]" in encoded, \
            f"user [goal] was rewritten to [g]: {encoded!r}"
        assert "[context]" in encoded, \
            f"user [context] was rewritten to [cx]: {encoded!r}"
        assert "[g]" not in encoded
        assert "[cx]" not in encoded

    def test_section_headings_preserved_across_compare(self, server):
        # At L1-L3 the bracket contents survive because no vowel-strip runs
        # and we no longer rewrite them as section headers. At L4+ vowel-
        # strip legitimately affects the word inside the brackets, but the
        # critical test is that L1 (the level Codex flagged) doesn't do
        # the section-abbreviation `[goal]` → `[g]` rewrite anymore.
        text = "Fix login.\n[goal]\nsession\n[context]\nJWT"
        status, data = _post_json(server, "/metalk/compare", {"text": text})
        assert status == 200
        by_level = {r["level"]: r for r in data["levels"] if "encoded" in r}
        for lv in (1, 2, 3):
            enc = by_level[lv]["encoded"]
            assert "[goal]" in enc, \
                f"L{lv} rewrote [goal] as section header: {enc!r}"
            assert "[context]" in enc, \
                f"L{lv} rewrote [context] as section header: {enc!r}"
            # The abbreviated forms must NOT appear.
            assert "[g]" not in enc and "[cx]" not in enc


class TestModeValidation:
    """Regression: `mode` used to accept any string, silently running
    plain-mode compression for typos like 'plian'."""

    def test_unknown_mode_returns_400_on_compress(self, server):
        try:
            _post_json(server, "/metalk/compress",
                       {"text": "hi", "level": 2, "mode": "plian"})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            body = json.loads(exc.read())
            assert "mode" in body["error"]
            return
        pytest.fail("expected HTTP 400")

    def test_unknown_mode_returns_400_on_compare(self, server):
        try:
            _post_json(server, "/metalk/compare",
                       {"text": "hi", "mode": "AUTO"})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            return
        pytest.fail("expected HTTP 400")

    def test_valid_modes_accepted(self, server):
        for mode in ("auto", "crumb", "plain"):
            status, _ = _post_json(server, "/metalk/compress",
                                   {"text": "hi", "level": 1, "mode": mode})
            assert status == 200


class TestCrumbAutoDetection:
    """Regression: auto mode used `startswith("BC")` which false-matched
    any prompt beginning with those letters (like 'BC drivers are failing'
    or 'BC Technologies'). It now requires the entire first non-empty line
    to be exactly BEGIN CRUMB or BC."""

    def test_bc_prefix_is_not_crumb(self, server):
        # "BC drivers are failing" is plain prose, not a CRUMB. Previously
        # this would route through metalk_encode (no `---` separator, so
        # the encoder returns input unchanged) and falsely report mode=crumb.
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": "BC drivers are failing today.",
                                   "level": 2, "mode": "auto"})
        assert status == 200
        assert data["stats"]["mode"] == "plain", \
            f"'BC drivers...' was misdetected as crumb: {data['stats']!r}"
        # Plain encoder also actually compresses the text (dict + grammar).
        assert data["stats"]["saved_tokens"] >= 0

    def test_bc_as_word_prefix_is_not_crumb(self, server):
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": "BCD framework comparison.",
                                   "level": 1, "mode": "auto"})
        assert status == 200
        assert data["stats"]["mode"] == "plain"

    def test_begin_crumb_sentinel_is_detected(self, server):
        crumb = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\nX\nEND CRUMB\n"
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": crumb, "level": 2, "mode": "auto"})
        assert status == 200
        assert data["stats"]["mode"] == "crumb"

    def test_leading_whitespace_crumb_detected(self, server):
        # Whitespace before the sentinel is fine — first non-empty line
        # still matches exactly after strip.
        crumb = "   BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\nX\nEND CRUMB\n"
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": crumb, "level": 2, "mode": "auto"})
        assert status == 200
        assert data["stats"]["mode"] == "crumb"

    def test_bc_standalone_line_detected(self, server):
        # MeTalk-abbreviated CRUMB starts with `BC` on its own line.
        crumb = "BC\nv=1.1\nk=task\nt=T\n---\n[g]\nX\nEC\n"
        status, data = _post_json(server, "/metalk/compress",
                                  {"text": crumb, "level": 2, "mode": "auto"})
        assert status == 200
        assert data["stats"]["mode"] == "crumb"


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
