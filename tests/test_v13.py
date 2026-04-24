"""Tests for CRUMB v1.3 additions (SPEC v1.3 §§1-11)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cli"))
import crumb
import ref_resolver
from squeeze import select_folds_size_greedy


AGENT_CRUMB = """\
BEGIN CRUMB
v=1.3
kind=agent
id=code-reviewer-v2
source=human.notes
---
[identity]
role=senior_reviewer

[rules]
- never approve without tests

[knowledge]
- expert=python, typescript
END CRUMB
"""


def _task_with_handoff(handoff_body: str) -> str:
    return f"""\
BEGIN CRUMB
v=1.3
kind=task
title=t
source=test
---
[goal]
g
[context]
c
[constraints]
x
[handoff]
{handoff_body}
END CRUMB
"""


def _task_with_workflow(workflow_body: str) -> str:
    return f"""\
BEGIN CRUMB
v=1.3
kind=task
title=t
source=test
---
[goal]
g
[context]
c
[constraints]
x
[workflow]
{workflow_body}
END CRUMB
"""


class TestKindAgent:
    def test_agent_with_identity_parses(self):
        parsed = crumb.parse_crumb(AGENT_CRUMB)
        assert parsed["headers"]["kind"] == "agent"
        assert parsed["headers"]["v"] == "1.3"
        assert "identity" in parsed["sections"]

    def test_agent_missing_identity_fails(self):
        text = AGENT_CRUMB.replace("[identity]\nrole=senior_reviewer\n\n", "")
        with pytest.raises(ValueError, match="missing required section"):
            crumb.parse_crumb(text)

    def test_agent_kind_without_v13_still_parses_if_marked_v12(self):
        """A v1.2 parser wouldn't accept kind=agent, but our writer could emit
        v=1.3 in the header. Verify v=1.3 is required for kind=agent when
        consumed by a v1.2-only parser would fail — which is the whole point
        of additive bumps."""
        text = AGENT_CRUMB.replace("v=1.3", "v=1.2")
        parsed = crumb.parse_crumb(text)
        assert parsed["headers"]["kind"] == "agent"


class TestSupportedVersions:
    def test_v13_parses(self):
        parsed = crumb.parse_crumb(AGENT_CRUMB)
        assert parsed["headers"]["v"] == "1.3"

    def test_v20_rejected(self):
        text = AGENT_CRUMB.replace("v=1.3", "v=2.0")
        with pytest.raises(ValueError, match="unsupported version"):
            crumb.parse_crumb(text)


class TestHandoffDependencies:
    def test_linear_dependency_parses(self):
        text = _task_with_handoff(
            "- id=a  to=any  do=step a\n"
            "- id=b  to=any  do=step b  after=a\n"
            "- id=c  to=any  do=step c  after=b\n"
        )
        crumb.parse_crumb(text)

    def test_multi_dependency_parses(self):
        text = _task_with_handoff(
            "- id=a  to=any  do=step a\n"
            "- id=b  to=any  do=step b\n"
            "- id=c  to=any  do=step c  after=a,b\n"
        )
        crumb.parse_crumb(text)

    def test_unknown_dependency_rejected(self):
        text = _task_with_handoff(
            "- id=a  to=any  do=step a  after=nonexistent\n"
        )
        with pytest.raises(ValueError, match="unknown after="):
            crumb.parse_crumb(text)

    def test_cycle_rejected(self):
        text = _task_with_handoff(
            "- id=a  to=any  do=step a  after=b\n"
            "- id=b  to=any  do=step b  after=a\n"
        )
        with pytest.raises(ValueError, match="dependency cycle"):
            crumb.parse_crumb(text)

    def test_duplicate_id_rejected(self):
        text = _task_with_handoff(
            "- id=a  to=any  do=step a\n"
            "- id=a  to=any  do=step a again\n"
        )
        with pytest.raises(ValueError, match="duplicate id"):
            crumb.parse_crumb(text)

    def test_id_with_invalid_chars_rejected(self):
        text = _task_with_handoff(
            "- id=a!b  to=any  do=step with invalid id\n"
        )
        with pytest.raises(ValueError, match="must match"):
            crumb.parse_crumb(text)

    def test_completed_line_ignored_for_deps(self):
        text = _task_with_handoff(
            "- id=a  to=any  do=step a\n"
            "- [x] something already done\n"
            "- id=b  to=any  do=step b  after=a\n"
        )
        crumb.parse_crumb(text)

    def test_no_id_implicit_position(self):
        """Lines without id= get implicit numeric position — no collision."""
        text = _task_with_handoff(
            "- to=any  do=step a\n"
            "- to=any  do=step b\n"
        )
        crumb.parse_crumb(text)


class TestWorkflowSection:
    def test_numbered_steps_parse(self):
        text = _task_with_workflow(
            "1. reproduce_bug  status=pending  owner=any\n"
            "2. write_test     status=blocked  owner=any  depends_on=1\n"
            "3. implement_fix  status=blocked  owner=any  depends_on=2\n"
        )
        crumb.parse_crumb(text)

    def test_unknown_depends_on_rejected(self):
        text = _task_with_workflow(
            "1. step_a  depends_on=nope\n"
        )
        with pytest.raises(ValueError, match="unknown depends_on"):
            crumb.parse_crumb(text)

    def test_cycle_rejected(self):
        text = _task_with_workflow(
            "1. a  id=a  depends_on=b\n"
            "2. b  id=b  depends_on=a\n"
        )
        with pytest.raises(ValueError, match="dependency cycle"):
            crumb.parse_crumb(text)

    def test_unnumbered_line_rejected(self):
        text = _task_with_workflow("not numbered\n")
        with pytest.raises(ValueError, match="must be numbered"):
            crumb.parse_crumb(text)


class TestFoldPriorityHeader:
    def test_valid_fold_priority(self):
        text = f"""\
BEGIN CRUMB
v=1.3
kind=task
title=t
source=test
fold_priority=context, constraints
---
[goal]
g
[fold:context/summary]
short
[fold:context/full]
long
[constraints]
x
END CRUMB
"""
        parsed = crumb.parse_crumb(text)
        assert parsed["headers"]["fold_priority"] == "context, constraints"

    def test_empty_fold_priority_rejected(self):
        text = f"""\
BEGIN CRUMB
v=1.3
kind=task
title=t
source=test
fold_priority=
---
[goal]
g
[context]
c
[constraints]
x
END CRUMB
"""
        with pytest.raises(ValueError, match="fold_priority"):
            crumb.parse_crumb(text)


class TestChecksSection:
    def test_valid_checks_parse(self):
        text = """\
BEGIN CRUMB
v=1.3
kind=task
title=t
source=test
---
[goal]
g
[context]
c
[constraints]
x
[checks]
- tests :: pass
- coverage :: 87% threshold=85
- lint :: fail note=unused
END CRUMB
"""
        crumb.parse_crumb(text)

    def test_missing_separator_rejected(self):
        text = """\
BEGIN CRUMB
v=1.3
kind=task
title=t
source=test
---
[goal]
g
[context]
c
[constraints]
x
[checks]
- tests pass
END CRUMB
"""
        with pytest.raises(ValueError, match="name :: status"):
            crumb.parse_crumb(text)


class TestScriptSection:
    def test_valid_script_parse(self):
        text = """\
BEGIN CRUMB
v=1.3
kind=task
title=t
source=test
---
[goal]
g
[context]
c
[constraints]
x
[script]
@type: weave
agent.can("shell-exec") -> false
END CRUMB
"""
        crumb.parse_crumb(text)

    def test_missing_type_rejected(self):
        text = """\
BEGIN CRUMB
v=1.3
kind=task
title=t
source=test
---
[goal]
g
[context]
c
[constraints]
x
[script]
agent.do_thing()
END CRUMB
"""
        with pytest.raises(ValueError, match="@type:"):
            crumb.parse_crumb(text)


class TestOptionalAdditiveSections:
    @pytest.mark.parametrize("section", ["guardrails", "capabilities", "invariants"])
    def test_section_does_not_break_parse(self, section):
        text = f"""\
BEGIN CRUMB
v=1.3
kind=task
title=t
source=test
---
[goal]
g
[context]
c
[constraints]
x
[{section}]
- anything goes here
END CRUMB
"""
        parsed = crumb.parse_crumb(text)
        assert section in parsed["sections"]


class TestRefResolver:
    def test_bare_id_resolves_to_local_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRUMB_HOME", str(tmp_path))
        ref_file = tmp_path / "my-ref.crumb"
        ref_file.write_text("BEGIN CRUMB\nEND CRUMB\n", encoding="utf-8")
        resolved = ref_resolver.resolve_ref("my-ref", search_paths=[tmp_path])
        assert resolved == ref_file

    def test_missing_ref_returns_none(self, tmp_path):
        assert ref_resolver.resolve_ref("nonexistent", search_paths=[tmp_path]) is None

    def test_url_not_fetched_by_default(self):
        assert (
            ref_resolver.resolve_ref("https://example.com/x.crumb") is None
        )

    def test_digest_without_store_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRUMB_STORE", str(tmp_path / "no-such"))
        assert ref_resolver.resolve_ref("sha256:" + "a" * 64) is None

    def test_digest_with_store_resolves(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRUMB_STORE", str(tmp_path))
        digest_file = tmp_path / ("a" * 64 + ".crumb")
        digest_file.write_text("BEGIN CRUMB\nEND CRUMB\n", encoding="utf-8")
        resolved = ref_resolver.resolve_ref("sha256:" + "a" * 64)
        assert resolved == digest_file

    def test_walk_respects_depth_limit(self, tmp_path, monkeypatch):
        # a -> b -> c -> d
        for name, refs in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "")]:
            content = (
                "BEGIN CRUMB\nv=1.3\nkind=mem\nsource=t\n"
                + (f"refs={refs}\n" if refs else "")
                + "---\n[consolidated]\n- x\nEND CRUMB\n"
            )
            (tmp_path / f"{name}.crumb").write_text(content, encoding="utf-8")
        walked = ref_resolver.walk_refs(
            "a", search_paths=[tmp_path], depth_limit=2
        )
        names = [ref for ref, _ in walked]
        assert "a" in names
        assert "b" in names
        assert "c" in names
        assert "d" not in names


class TestSizeGreedyFold:
    def test_selects_summary_when_tight(self):
        sections = {
            "fold:context/summary": ["short"],
            "fold:context/full": ["much longer body " * 100],
        }
        selection = select_folds_size_greedy(sections, budget=10)
        assert selection["context"] == "summary"

    def test_upgrades_to_full_when_budget_allows(self):
        sections = {
            "fold:context/summary": ["short"],
            "fold:context/full": ["a bit longer"],
        }
        selection = select_folds_size_greedy(sections, budget=10_000)
        assert selection["context"] == "full"

    def test_fold_priority_order_honored(self):
        sections = {
            "fold:a/summary": ["s"],
            "fold:a/full": ["long " * 10],
            "fold:b/summary": ["s"],
            "fold:b/full": ["long " * 10],
        }
        # budget can fit one /full upgrade, not two. b should win because it's
        # first in the declared fold_priority.
        selection = select_folds_size_greedy(sections, budget=15, fold_priority=["b", "a"])
        assert selection["b"] == "full"
        assert selection["a"] == "summary"
