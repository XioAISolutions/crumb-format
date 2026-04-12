"""Tests for cli.classify — rule-based hall classifier."""

import pytest
from cli.classify import classify, classify_batch, score, explain, HALLS


# ── classify: known-good examples per hall ──────────────────────────────

@pytest.mark.parametrize("text, expected", [
    # facts — decisions and locked-in choices
    ("decided to use Postgres over SQLite", "facts"),
    ("picked React for the frontend", "facts"),
    ("selected AWS over GCP", "facts"),
    ("went with JWT for auth", "facts"),
    ("stack is Next.js + Prisma", "facts"),
    # events — things that happened with temporal markers
    ("shipped v0.2.0 yesterday", "events"),
    ("fixed the login redirect bug 2026-04-12", "events"),
    ("deployed the hotfix last week", "events"),
    ("service crashed at 3am", "events"),
    ("merged the PR this morning", "events"),
    # discoveries — realizations and learnings
    ("realized the cookie middleware runs before auth parsing", "discoveries"),
    ("learned that PG has row-level locking", "discoveries"),
    ("turns out the CDN was caching 404s", "discoveries"),
    ("found out the SDK supports streaming", "discoveries"),
    # preferences — working style and likes
    ("prefers direct technical answers", "preferences"),
    ("likes short commit messages", "preferences"),
    ("hates verbose logging", "preferences"),
    ("always uses dark mode", "preferences"),
    # advice — recommendations and warnings
    ("never commit to main without running tests", "advice"),
    ("should always use parameterized queries", "advice"),
    ("recommend adding retry logic for network calls", "advice"),
    ("make sure to update the changelog", "advice"),
])
def test_classify_known_examples(text, expected):
    assert classify(text) == expected


# ── default for ambiguous text ──────────────────────────────────────────

def test_classify_default():
    assert classify("the team is 5 people") == "facts"
    assert classify("a random thought") == "facts"


# ── score returns all five halls ────────────────────────────────────────

def test_score_structure():
    s = score("decided to use Postgres")
    assert set(s.keys()) == set(HALLS)
    assert all(isinstance(v, int) for v in s.values())
    assert s["facts"] > 0


# ── explain returns diagnostics ─────────────────────────────────────────

def test_explain_structure():
    info = explain("decided to use Postgres")
    assert info["hall"] == "facts"
    assert "scores" in info
    assert "matched_patterns" in info
    assert info["text"] == "decided to use Postgres"
    assert len(info["matched_patterns"]["facts"]) > 0


# ── batch classification ───────────────────────────────────────────────

def test_classify_batch():
    lines = [
        "decided to use Postgres",
        "shipped v0.2.0",
        "   ",
        "prefers dark mode",
        "",
    ]
    result = classify_batch(lines)
    assert len(result) == 3
    assert result[0] == ("facts", "decided to use Postgres")
    assert result[1] == ("events", "shipped v0.2.0")
    assert result[2] == ("preferences", "prefers dark mode")


def test_classify_batch_empty():
    assert classify_batch([]) == []
    assert classify_batch(["", "  ", "\t"]) == []


# ── case insensitivity ──────────────────────────────────────────────────

def test_classify_case_insensitive():
    assert classify("DECIDED to use Postgres") == "facts"
    assert classify("SHIPPED v1.0") == "events"
    assert classify("Realized the API changed") == "discoveries"
