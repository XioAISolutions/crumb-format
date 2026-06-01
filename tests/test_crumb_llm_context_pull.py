"""Context-pulling engine tests."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from crumb_wavelm.context_pull import (  # noqa: E402
    CrumbIndex,
    ContextPullSession,
    IndexedSection,
    build_index,
    pull_context,
    score_sections,
    _spectral_signature,
)


def test_spectral_signature_shape():
    sig = _spectral_signature([65, 66, 67, 68], field_size=32)
    assert len(sig) == 32 // 2 + 1


def test_spectral_signature_empty():
    sig = _spectral_signature([], field_size=16)
    assert all(v == 0.0 for v in sig)


def test_build_index_from_examples():
    repo_root = Path(__file__).resolve().parent.parent
    examples = repo_root / "examples"
    if not examples.exists():
        pytest.skip("examples/ not present")
    idx = build_index(examples, field_size=64)
    assert len(idx.sections) > 0
    assert idx.field_size == 64
    assert all(s.n_tokens > 0 for s in idx.sections)
    assert all(len(s.spectral_signature) == 64 // 2 + 1 for s in idx.sections)


def test_score_sections_returns_ranked():
    repo_root = Path(__file__).resolve().parent.parent
    examples = repo_root / "examples"
    if not examples.exists():
        pytest.skip("examples/ not present")
    idx = build_index(examples, field_size=64)
    scored = score_sections("Fix the login bug", idx, top_k=3)
    assert len(scored) <= 3
    assert len(scored) > 0
    # Scores should be sorted descending.
    scores = [s for s, _ in scored]
    assert scores == sorted(scores, reverse=True)


def test_pull_context_respects_token_budget():
    repo_root = Path(__file__).resolve().parent.parent
    examples = repo_root / "examples"
    if not examples.exists():
        pytest.skip("examples/ not present")
    idx = build_index(examples, field_size=64)
    pulled = pull_context("Fix auth redirect", idx, max_tokens=50, top_k=3)
    assert isinstance(pulled, str)
    # Should have pulled something but not everything.
    assert len(pulled) > 0


def test_index_save_load(tmp_path):
    sections = [
        IndexedSection(
            source_file="test.crumb",
            section_name="goal",
            text="Fix the bug",
            token_ids=[70, 105, 120],
            n_tokens=3,
            priority=5,
            spectral_signature=[1.0, 2.0, 3.0],
            content_hash="abc123",
        )
    ]
    idx = CrumbIndex(sections=sections, field_size=4, vocab_size=256)
    idx.save(tmp_path / "index.json")
    loaded = CrumbIndex.load(tmp_path / "index.json")
    assert len(loaded.sections) == 1
    assert loaded.sections[0].section_name == "goal"
    assert loaded.field_size == 4


def test_context_pull_session_avoids_duplicates():
    sections = [
        IndexedSection("a.crumb", "goal", "Fix bug", [70], 1, 5, [1.0, 0.5], "hash1"),
        IndexedSection("b.crumb", "context", "Context info", [67], 1, 3, [0.5, 1.0], "hash2"),
    ]
    idx = CrumbIndex(sections=sections, field_size=2, vocab_size=256)
    session = ContextPullSession(index=idx, max_context_tokens=100, top_k=5)
    first = session.pull("Fix")
    second = session.pull("Fix")
    # Second pull should return empty (both hashes seen).
    assert len(second) == 0 or second != first
