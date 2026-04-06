"""Tests for MeTalk caveman compression."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.metalk import encode, decode, compression_stats, ABBREV, SECTION_MAP, HEADER_KEY_MAP

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


class TestDictSubstitution:
    def test_tech_terms_abbreviated(self):
        crumb = (
            "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=Test\n---\n"
            "[goal]\n  Fix the authentication middleware function.\nEND CRUMB\n"
        )
        result = encode(crumb, level=1)
        assert "auth" in result
        assert "mw" in result
        assert "fn" in result

    def test_no_substring_corruption(self):
        """'authentication' should become 'auth', not 'authcation'."""
        crumb = (
            "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=Test\n---\n"
            "[goal]\n  Fix authentication issue.\nEND CRUMB\n"
        )
        result = encode(crumb, level=1)
        assert "authcation" not in result
        assert "auth" in result

    def test_structural_markers(self):
        crumb = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\n  X\nEND CRUMB\n"
        result = encode(crumb, level=1)
        assert result.startswith("BC\n")
        assert result.strip().endswith("EC")

    def test_section_headers_abbreviated(self):
        crumb = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\n  X\n[context]\n  Y\n[constraints]\n  Z\nEND CRUMB\n"
        result = encode(crumb, level=1)
        assert "[g]" in result
        assert "[cx]" in result
        assert "[ct]" in result

    def test_header_keys_abbreviated(self):
        crumb = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=Test\nsource=test\n---\n[goal]\n  X\nEND CRUMB\n"
        result = encode(crumb, level=1)
        assert "k=task" in result
        assert "t=Test" in result
        assert "src=test" in result


class TestGrammarStrip:
    def test_articles_removed(self):
        crumb = (
            "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n"
            "[goal]\n  Fix the bug in the application.\nEND CRUMB\n"
        )
        result = encode(crumb, level=2)
        # "the" should be stripped
        assert "Fix bug in app." in result or "Fix bug in app" in result

    def test_filler_words_removed(self):
        crumb = (
            "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n"
            "[goal]\n  Just basically fix the very simple issue.\nEND CRUMB\n"
        )
        result = encode(crumb, level=2)
        assert "just" not in result.lower() or "Just" not in result
        assert "basically" not in result.lower()
        assert "very" not in result.lower()

    def test_phrase_rewrites(self):
        crumb = (
            "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n"
            "[goal]\n  Do not change the login UI.\nEND CRUMB\n"
        )
        result = encode(crumb, level=2)
        assert "don't" in result.lower() or "Don't" in result


class TestCompressionLevels:
    def test_level1_saves_tokens(self):
        crumb = open(EXAMPLES_DIR / "task-bug-fix.crumb").read()
        result = encode(crumb, level=1)
        stats = compression_stats(crumb, result)
        assert stats["saved_tokens"] > 0
        assert stats["pct_saved"] > 0

    def test_level2_saves_more_than_level1(self):
        crumb = open(EXAMPLES_DIR / "task-content-repurpose-handoff.crumb").read()
        r1 = encode(crumb, level=1)
        r2 = encode(crumb, level=2)
        s1 = compression_stats(crumb, r1)
        s2 = compression_stats(crumb, r2)
        assert s2["pct_saved"] >= s1["pct_saved"]

    def test_level3_saves_most(self):
        crumb = open(EXAMPLES_DIR / "task-content-repurpose-handoff.crumb").read()
        r2 = encode(crumb, level=2)
        r3 = encode(crumb, level=3)
        s2 = compression_stats(crumb, r2)
        s3 = compression_stats(crumb, r3)
        assert s3["pct_saved"] >= s2["pct_saved"]


class TestMetalkHeader:
    def test_mt_header_injected(self):
        crumb = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\n  X\nEND CRUMB\n"
        for level in [1, 2, 3]:
            result = encode(crumb, level=level)
            assert f"mt={level}" in result

    def test_decode_detects_mt_header(self):
        crumb = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\n  X\nEND CRUMB\n"
        encoded = encode(crumb, level=1)
        decoded = decode(encoded)
        assert "mt=" not in decoded
        assert "BEGIN CRUMB" in decoded


class TestDecode:
    def test_decode_expands_structural(self):
        crumb = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\n  X\nEND CRUMB\n"
        encoded = encode(crumb, level=1)
        decoded = decode(encoded)
        assert decoded.startswith("BEGIN CRUMB")
        assert "END CRUMB" in decoded

    def test_decode_expands_sections(self):
        crumb = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\n  X\n[context]\n  Y\nEND CRUMB\n"
        encoded = encode(crumb, level=1)
        decoded = decode(encoded)
        assert "[goal]" in decoded
        assert "[context]" in decoded

    def test_decode_expands_header_keys(self):
        crumb = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\nsource=test\n---\n[goal]\n  X\nEND CRUMB\n"
        encoded = encode(crumb, level=1)
        decoded = decode(encoded)
        assert "kind=task" in decoded
        assert "title=T" in decoded
        assert "source=test" in decoded

    def test_non_metalk_passthrough(self):
        crumb = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\n  X\nEND CRUMB\n"
        decoded = decode(crumb)
        assert decoded == crumb


class TestAllExamples:
    @pytest.mark.parametrize("crumb_file", sorted(EXAMPLES_DIR.glob("*.crumb")))
    def test_encode_produces_output(self, crumb_file):
        text = crumb_file.read_text(encoding="utf-8")
        for level in [1, 2, 3]:
            result = encode(text, level=level)
            assert "BC" in result or "BEGIN CRUMB" in result
            stats = compression_stats(text, result)
            assert stats["saved_tokens"] >= 0

    @pytest.mark.parametrize("crumb_file", sorted(EXAMPLES_DIR.glob("*.crumb")))
    def test_decode_after_encode(self, crumb_file):
        text = crumb_file.read_text(encoding="utf-8")
        encoded = encode(text, level=1)
        decoded = decode(encoded)
        # Structural elements should be restored
        assert "BEGIN CRUMB" in decoded
        assert "END CRUMB" in decoded


class TestCompressionStats:
    def test_stats_structure(self):
        crumb = "BEGIN CRUMB\nv=1.1\nkind=task\ntitle=T\n---\n[goal]\n  X\nEND CRUMB\n"
        encoded = encode(crumb, level=2)
        stats = compression_stats(crumb, encoded)
        assert "original_tokens" in stats
        assert "encoded_tokens" in stats
        assert "saved_tokens" in stats
        assert "pct_saved" in stats
        assert "ratio" in stats

    def test_positive_savings(self):
        crumb = open(EXAMPLES_DIR / "task-bug-fix.crumb").read()
        encoded = encode(crumb, level=2)
        stats = compression_stats(crumb, encoded)
        assert stats["saved_tokens"] > 0
        assert stats["ratio"] > 1.0


class TestDictionaryIntegrity:
    def test_no_duplicate_short_forms(self):
        """Every abbreviation maps to a unique short form."""
        short_forms = list(ABBREV.values())
        assert len(short_forms) == len(set(short_forms)), \
            f"Duplicate short forms: {[s for s in short_forms if short_forms.count(s) > 1]}"

    def test_no_section_collisions(self):
        short_forms = list(SECTION_MAP.values())
        assert len(short_forms) == len(set(short_forms))

    def test_no_header_key_collisions(self):
        short_forms = list(HEADER_KEY_MAP.values())
        assert len(short_forms) == len(set(short_forms))
