"""Tests for v1.2 efficiency layers: squeeze, hashing, delta, priority folds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli import crumb, delta, hashing, squeeze


TASK_WITH_FOLDS = """\
BEGIN CRUMB
v=1.2
kind=task
title=Budget test
source=test
refs=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, mem-keep-me
---
[goal]
Fix the login redirect loop that bounces authenticated users back to /login.

[fold:context/summary]
JWT middleware races the cookie parser on refresh.

[fold:context/full]
Full repro, 30 lines of investigation, stack trace, three branches worth of
experimentation. Long prose here just to take up token budget so the packer
has a reason to drop this fold variant first before touching anything else
the author cared about.

[constraints]
- Do not change the login UI.
- Preserve existing cookie names.

[notes]
@priority: 2
- Low-priority scratch pad. Safe to drop under token pressure.

[rationale]
@priority: 8
- High-priority explanation that should outlast [notes] under pressure.
END CRUMB
"""


TASK_PLAIN = """\
BEGIN CRUMB
v=1.1
kind=task
title=Plain task
source=test
---
[goal]
Plain goal.

[context]
- Thing one.
- Thing two.

[constraints]
- Keep it simple.
END CRUMB
"""


# ── Layer 4: @priority annotation parsing ──────────────────────────────

class TestPriorityAnnotation:
    def test_valid_priority_parses(self):
        parsed = crumb.parse_crumb(TASK_WITH_FOLDS)
        assert "notes" in parsed["sections"]
        assert "rationale" in parsed["sections"]

    def test_priority_must_be_integer(self):
        text = TASK_PLAIN.replace(
            "[context]",
            "[context]\n@priority: high",
        )
        with pytest.raises(ValueError, match="must be an integer"):
            crumb.parse_crumb(text)

    def test_priority_out_of_range(self):
        text = TASK_PLAIN.replace(
            "[context]",
            "[context]\n@priority: 42",
        )
        with pytest.raises(ValueError, match="between 1 and 10"):
            crumb.parse_crumb(text)

    def test_priority_empty_value(self):
        text = TASK_PLAIN.replace(
            "[context]",
            "[context]\n@priority:",
        )
        with pytest.raises(ValueError, match="empty value"):
            crumb.parse_crumb(text)

    def test_priority_after_type_is_allowed(self):
        text = TASK_PLAIN.replace(
            "[context]",
            "[context]\n@type: text/plain\n@priority: 3",
        )
        parsed = crumb.parse_crumb(text)
        assert "context" in parsed["sections"]


# ── Layer 1: crumb squeeze (budget-aware packer) ───────────────────────

class TestSqueeze:
    def test_squeeze_within_budget_is_noop(self):
        rendered, report = squeeze.squeeze_crumb(TASK_PLAIN, budget=10_000)
        assert report.final_tokens <= 10_000
        assert report.dropped_sections == []
        assert report.dropped_full_folds == []
        assert report.metalk_level == 0
        crumb.parse_crumb(rendered)

    def test_squeeze_drops_full_fold_first(self):
        rendered, report = squeeze.squeeze_crumb(TASK_WITH_FOLDS, budget=160)
        assert "context" in report.dropped_full_folds
        assert "fold:context/summary" in rendered
        assert "fold:context/full" not in rendered
        crumb.parse_crumb(rendered)

    def test_squeeze_drops_lowest_priority_optional(self):
        rendered, report = squeeze.squeeze_crumb(TASK_WITH_FOLDS, budget=130)
        assert "notes" in report.dropped_sections
        parsed = crumb.parse_crumb(rendered)
        assert "notes" not in parsed["sections"]
        assert "rationale" in parsed["sections"]

    def test_squeeze_preserves_required_sections(self):
        rendered, report = squeeze.squeeze_crumb(TASK_WITH_FOLDS, budget=140)
        parsed = crumb.parse_crumb(rendered)
        for required in ("goal", "constraints"):
            assert required in parsed["sections"] or f"fold:{required}/summary" in parsed["sections"]

    def test_squeeze_fails_if_budget_below_required_floor(self):
        # v0.7: error wording changed to include actionable recovery hints.
        with pytest.raises(ValueError, match="cannot fit"):
            squeeze.squeeze_crumb(TASK_WITH_FOLDS, budget=5)

    def test_squeeze_escalates_metalk_when_needed(self):
        # A budget that drops folds + notes + rationale still leaves too much
        # for the required sections — MeTalk has to fire.
        rendered, report = squeeze.squeeze_crumb(TASK_WITH_FOLDS, budget=95)
        assert report.metalk_level >= 1 or report.final_tokens <= 95

    def test_squeeze_elides_seen_refs(self):
        seen = {"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
        rendered, report = squeeze.squeeze_crumb(TASK_WITH_FOLDS, budget=10_000, seen=seen)
        assert "sha256:aaaa" not in rendered
        assert "mem-keep-me" in rendered
        assert len(report.elided_refs) == 1

    def test_squeeze_short_digest_elides_long_ref(self):
        rendered, report = squeeze.squeeze_crumb(
            TASK_WITH_FOLDS, budget=10_000, seen={"sha256:aaaaaaaaaaaaaaaa"}
        )
        assert "sha256:aaaa" not in rendered
        assert len(report.elided_refs) == 1


# ── Layer 2: content-addressed hashing and seen set ───────────────────

class TestContentHashing:
    def test_hash_is_stable_across_volatile_headers(self):
        text_a = TASK_PLAIN
        text_b = TASK_PLAIN.replace(
            "source=test",
            "source=test\nid=some-id\ndream_pass=1",
        )
        assert hashing.content_hash(text_a) == hashing.content_hash(text_b)

    def test_hash_changes_when_content_changes(self):
        text_a = TASK_PLAIN
        text_b = TASK_PLAIN.replace("Plain goal.", "A different goal.")
        assert hashing.content_hash(text_a) != hashing.content_hash(text_b)

    def test_hash_format(self):
        digest = hashing.content_hash(TASK_PLAIN)
        assert digest.startswith("sha256:")
        assert len(digest.split(":", 1)[1]) == 64

    def test_short_hash_truncates(self):
        digest = hashing.content_hash(TASK_PLAIN)
        short = hashing.short_hash(digest, length=12)
        assert short.startswith("sha256:")
        assert len(short.split(":", 1)[1]) == 12

    def test_crumb_accepts_sha256_refs(self):
        text = TASK_PLAIN.replace(
            "source=test",
            "source=test\nrefs=sha256:" + "a" * 64,
        )
        parsed = crumb.parse_crumb(text)
        assert "sha256:" in parsed["headers"]["refs"]

    def test_crumb_rejects_malformed_sha256_refs(self):
        text = TASK_PLAIN.replace(
            "source=test",
            "source=test\nrefs=sha256:not-hex!!",
        )
        with pytest.raises(ValueError, match="malformed sha256"):
            crumb.parse_crumb(text)


class TestSeenSet:
    def test_add_list_remove(self, tmp_path):
        store = tmp_path / "seen"
        hashing.add_seen(["sha256:" + "a" * 64], path=store)
        hashing.add_seen(["sha256:" + "b" * 64], path=store)
        assert len(hashing.load_seen(store)) == 2
        hashing.remove_seen(["sha256:" + "a" * 64], path=store)
        assert len(hashing.load_seen(store)) == 1

    def test_clear(self, tmp_path):
        store = tmp_path / "seen"
        hashing.add_seen(["sha256:" + "a" * 64], path=store)
        hashing.clear_seen(store)
        assert hashing.load_seen(store) == set()

    def test_is_seen_prefix_match(self, tmp_path):
        store = tmp_path / "seen"
        long_digest = "sha256:" + "a" * 64
        hashing.add_seen(["sha256:" + "a" * 16], path=store)
        assert hashing.is_seen(long_digest, path=store)

    def test_rejects_non_sha256_entries(self, tmp_path):
        store = tmp_path / "seen"
        with pytest.raises(ValueError):
            hashing.add_seen(["not-a-digest"], path=store)


# ── Layer 3: delta crumbs ──────────────────────────────────────────────

class TestDelta:
    def test_delta_kind_requires_base_header(self):
        text = """BEGIN CRUMB
v=1.2
kind=delta
source=test
---
[changes]
- +[context] added line
END CRUMB
"""
        with pytest.raises(ValueError, match="base"):
            crumb.parse_crumb(text)

    def test_delta_kind_requires_changes_section(self):
        text = """BEGIN CRUMB
v=1.2
kind=delta
source=test
base=sha256:abc123
---
[changes]
END CRUMB
"""
        with pytest.raises(ValueError):
            crumb.parse_crumb(text)

    def test_malformed_change_line_rejected(self):
        text = """BEGIN CRUMB
v=1.2
kind=delta
source=test
base=sha256:abc123
---
[changes]
- bogus entry with no marker
END CRUMB
"""
        with pytest.raises(ValueError, match="malformed"):
            crumb.parse_crumb(text)

    def test_roundtrip_section_changes(self):
        base = TASK_PLAIN
        target = TASK_PLAIN.replace(
            "- Thing two.",
            "- Thing two updated.",
        ).replace(
            "- Keep it simple.",
            "- Keep it simple.\n- Add a regression test.",
        )
        delta_text = delta.build_delta_crumb(base, target)
        rebuilt = delta.apply_delta(base, delta_text)
        assert hashing.content_hash(rebuilt) == hashing.content_hash(target)

    def test_roundtrip_header_changes(self):
        base = TASK_PLAIN
        target = TASK_PLAIN.replace("title=Plain task", "title=Renamed task")
        delta_text = delta.build_delta_crumb(base, target)
        assert "@headers" in delta_text
        rebuilt = delta.apply_delta(base, delta_text)
        assert "title=Renamed task" in rebuilt

    def test_apply_verify_rejects_wrong_base(self):
        base = TASK_PLAIN
        target = TASK_PLAIN.replace("Plain goal.", "New goal.")
        delta_text = delta.build_delta_crumb(base, target)
        other_base = TASK_PLAIN.replace("Plain goal.", "Yet another goal.")
        with pytest.raises(ValueError, match="mismatch"):
            delta.apply_delta(other_base, delta_text, verify=True)

    def test_identical_crumbs_raise(self):
        with pytest.raises(ValueError, match="no content"):
            delta.build_delta_crumb(TASK_PLAIN, TASK_PLAIN)
