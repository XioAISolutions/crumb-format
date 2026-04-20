"""Tests for vowel-strip Layer 4 of MeTalk."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.vowelstrip import (
    DEFAULT_MIN_LENGTH,
    PROTECTED_WORDS,
    drift_stats,
    encode_crumb,
    strip_line,
    strip_text,
    strip_word,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


class TestStripWord:
    def test_first_letter_preserved(self):
        assert strip_word("authentication").startswith("a")
        assert strip_word("Middleware").startswith("M")

    def test_short_words_unchanged(self):
        for w in ["a", "an", "the", "fix", "bug"]:
            assert strip_word(w) == w

    def test_acronyms_unchanged(self):
        for w in ["HTTP", "API", "JWT", "SQL"]:
            assert strip_word(w) == w

    def test_protected_words_unchanged(self):
        for w in PROTECTED_WORDS:
            if len(w) >= DEFAULT_MIN_LENGTH:
                assert strip_word(w) == w

    def test_non_alpha_unchanged(self):
        assert strip_word("v1.2") == "v1.2"
        assert strip_word("snake_case") == "snake_case"

    def test_plural_s_preserved(self):
        # Distinguishes "user" (kept whole, in PROTECTED) from "users" (also protected).
        # But for non-protected plurals: "function" → "fnctn", "functions" → "fnctns".
        assert strip_word("functions").endswith("s")
        assert strip_word("modules").endswith("s")

    def test_known_examples(self):
        assert strip_word("authentication") == "athntctn"
        assert strip_word("middleware") == "mddlwr"
        assert strip_word("function") == "fnctn"
        assert strip_word("beautiful") == "btfl"

    def test_case_preserved_on_first(self):
        assert strip_word("Beautiful")[0] == "B"
        assert strip_word("beautiful")[0] == "b"

    def test_idempotent(self):
        for w in ["authentication", "configuration", "performance"]:
            once = strip_word(w)
            twice = strip_word(once)
            assert once == twice


class TestStripLine:
    def test_punctuation_preserved(self):
        out = strip_line("Authentication, please fix the middleware.")
        assert "," in out
        assert "." in out

    def test_urls_unchanged(self):
        out = strip_line("Visit https://example.com/docs for info.")
        assert "https://example.com/docs" in out

    def test_identifiers_unchanged(self):
        out = strip_line("The function snake_case_name returns json.")
        assert "snake_case_name" in out

    def test_url_paths_unchanged(self):
        # /login is a path fragment — leading slash must not be peeled away.
        out = strip_line("Redirect to /login after refresh.")
        assert "/login" in out

    def test_handles_unchanged(self):
        out = strip_line("Ping @alice about authentication.")
        assert "@alice" in out

    def test_trailing_sentence_punct_peeled(self):
        # "authentication." → "athntctn." — period preserved at end.
        out = strip_line("Fix authentication.")
        assert out.endswith(".")
        assert "athntctn" in out

    def test_mixed_short_and_long(self):
        out = strip_line("A simple authentication test.")
        assert "A" in out
        assert "athntctn" in out


class TestEncodeCrumb:
    def test_section_headers_preserved(self):
        text = (
            "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=Test\n---\n"
            "[goal]\nFix authentication middleware.\n[context]\nApp uses JWT.\nEND CRUMB\n"
        )
        out = encode_crumb(text)
        assert "[goal]" in out
        assert "[context]" in out

    def test_header_block_untouched(self):
        text = (
            "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=Authentication fix\n---\n"
            "[goal]\nFix authentication.\nEND CRUMB\n"
        )
        out = encode_crumb(text)
        assert "kind=task" in out
        assert "title=Authentication fix" in out  # header value preserved

    def test_body_stripped(self):
        text = (
            "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n"
            "[goal]\nFix authentication middleware function.\nEND CRUMB\n"
        )
        out = encode_crumb(text)
        assert "athntctn" in out
        assert "mddlwr" in out

    def test_vs_header_injected(self):
        text = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\nAuthentication.\nEND CRUMB\n"
        out = encode_crumb(text, min_length=4)
        assert "vs=4" in out

    def test_fenced_code_untouched(self):
        text = (
            "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n"
            "[context]\n```\nfunction authenticate() { return true; }\n```\nEND CRUMB\n"
        )
        out = encode_crumb(text)
        assert "function authenticate()" in out

    def test_typed_code_block_untouched(self):
        text = (
            "BEGIN CRUMB\nv=1.2\nkind=task\ntitle=T\n---\n"
            "[context]\n@type: code/typescript\nexport async function requireAuth(req) { return true; }\n"
            "[constraints]\nMust authenticate properly.\nEND CRUMB\n"
        )
        out = encode_crumb(text)
        assert "export async function requireAuth(req)" in out
        # but the next section after [constraints] is stripped
        assert "thntct" in out or "athntct" in out

    def test_non_crumb_passthrough(self):
        out = encode_crumb("just some plain text without separator")
        assert out == "just some plain text without separator"


class TestStripText:
    def test_multiline(self):
        text = "Authentication middleware.\nConfiguration loaded."
        out = strip_text(text)
        assert "\n" in out
        assert "Athntctn" in out
        assert "Cnfgrtn" in out

    def test_empty_lines_preserved(self):
        text = "Line one.\n\nLine two."
        out = strip_text(text)
        assert out.count("\n") == 2


class TestDriftStats:
    def test_savings_positive_on_prose(self):
        original = "Authentication middleware configuration documentation."
        stripped = strip_text(original)
        stats = drift_stats(original, stripped)
        assert stats["encoded_chars"] < stats["original_chars"]
        assert stats["vowels_removed"] > 0
        assert stats["vowel_retention_pct"] < 100

    def test_zero_when_identical(self):
        text = "ok bye"
        stats = drift_stats(text, text)
        assert stats["vowels_removed"] == 0
        assert stats["vowel_retention_pct"] == 100.0


class TestMetalkLevel4Integration:
    @pytest.mark.parametrize("crumb_file", sorted(EXAMPLES_DIR.glob("*.crumb")))
    def test_level4_produces_smaller_or_equal(self, crumb_file):
        from cli.metalk import encode, compression_stats
        text = crumb_file.read_text(encoding="utf-8")
        l3 = encode(text, level=3)
        l4 = encode(text, level=4)
        s3 = compression_stats(text, l3)
        s4 = compression_stats(text, l4)
        assert s4["encoded_chars"] <= s3["encoded_chars"]

    def test_level4_injects_vs_header(self):
        from cli.metalk import encode
        text = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\nAuthentication.\nEND CRUMB\n"
        out = encode(text, level=4)
        assert "vs=" in out

    def test_decode_strips_vs_header(self):
        from cli.metalk import encode, decode
        text = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\nAuthentication.\nEND CRUMB\n"
        encoded = encode(text, level=4)
        decoded = decode(encoded)
        assert "vs=" not in decoded

    def test_level5_loads_embedder_once_per_encode(self, monkeypatch):
        """Regression: L5 must load the SentenceTransformer model ONCE per
        encode() call, not once per line. Previously adaptive_strip_text
        was called with embedder=None every time, so a 20-line crumb
        would load the model 20 times (seconds + ~80MB each)."""
        from cli import metalk as _metalk
        from cli import vowelstrip as _vs

        call_count = {"n": 0}

        class FakeEmbedder:
            def encode(self, pairs):
                # Return two small vectors that are highly similar (cos ~ 1)
                # so every line keeps the vowel-stripped candidate.
                return [[1.0, 0.0, 0.0], [0.999, 0.001, 0.0]]

        def fake_load_embedder(*args, **kwargs):
            call_count["n"] += 1
            return FakeEmbedder()

        # Patch in both modules: encode() imports _load_embedder from vowelstrip,
        # and adaptive_strip_text falls back to it if embedder=None.
        monkeypatch.setattr(_vs, "_load_embedder", fake_load_embedder)

        crumb = (
            "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n"
            "[goal]\nFix the authentication middleware.\n"
            "[context]\n- Line one about authentication.\n"
            "- Line two about configuration.\n"
            "- Line three about deployment.\n"
            "- Line four about middleware.\n"
            "- Line five about implementation.\n"
            "END CRUMB\n"
        )
        _metalk.encode(crumb, level=5)
        # One load call for the whole encode, regardless of line count.
        assert call_count["n"] == 1, (
            f"expected _load_embedder called once per encode, got {call_count['n']}"
        )

    def test_level5_falls_back_when_no_st(self):
        # Without sentence-transformers installed (the test environment),
        # level 5 should produce the same body as level 4 — only the
        # mt= header differs.
        from cli.metalk import encode
        text = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\nFix authentication middleware.\nEND CRUMB\n"
        l4 = encode(text, level=4)
        l5 = encode(text, level=5)
        l4_body = l4.split('---', 1)[1]
        l5_body = l5.split('---', 1)[1]
        assert l4_body == l5_body
        assert "mt=4" in l4 and "mt=5" in l5
