"""Tests for the cli.handoff multi-agent helpers."""

from __future__ import annotations

import pytest

from cli import crumb as crumb_mod
from cli.handoff import (
    ChainError,
    emit_mem,
    emit_task,
    new_id,
    validate_chain,
    walk_chain,
)


class TestEmitTask:
    def test_minimal_task_parses_as_valid_crumb(self):
        result = emit_task(
            title="Do the thing",
            goal="Accomplish X.",
            context=["- context line 1", "- context line 2"],
            constraints=["- don't break Y"],
            source="unit.test",
        )
        reparsed = crumb_mod.parse_crumb(result["text"])
        assert reparsed["headers"]["kind"] == "task"
        assert reparsed["headers"]["v"] == "1.1"
        assert reparsed["headers"]["source"] == "unit.test"
        assert reparsed["headers"]["title"] == "Do the thing"
        assert "id" in reparsed["headers"]
        # required sections populated
        assert reparsed["sections"]["goal"]
        assert reparsed["sections"]["context"]
        assert reparsed["sections"]["constraints"]

    def test_upstream_pointer_is_namespaced(self):
        parent = emit_task(
            title="Parent",
            goal="root",
            context="-",
            constraints="-",
            source="a",
            namespace="rag",
        )
        child = emit_task(
            title="Child",
            goal="do next step",
            context="-",
            constraints="-",
            source="b",
            upstream=parent["id"],
            namespace="rag",
        )
        assert child["headers"]["ext.rag.upstream"] == parent["id"]
        extensions = child["headers"]["extensions"]
        assert "ext.rag.upstream.v1" in extensions

    def test_custom_id_is_preserved(self):
        result = emit_task(
            title="T",
            goal="g",
            context="c",
            constraints="k",
            source="s",
            crumb_id="manual-id-42",
        )
        assert result["id"] == "manual-id-42"
        assert result["headers"]["id"] == "manual-id-42"

    def test_extra_sections_and_headers(self):
        result = emit_task(
            title="T",
            goal="g",
            context="c",
            constraints="k",
            source="s",
            extra_sections={"notes": ["- extra"]},
            extra_headers={"project": "alpha", "env": "staging"},
        )
        reparsed = crumb_mod.parse_crumb(result["text"])
        assert "notes" in reparsed["sections"]
        assert reparsed["headers"]["project"] == "alpha"
        assert reparsed["headers"]["env"] == "staging"


class TestEmitMem:
    def test_minimal_mem_parses_as_valid_crumb(self):
        result = emit_mem(
            title="Preferences",
            consolidated=["- prefer concise answers"],
            source="human.notes",
        )
        reparsed = crumb_mod.parse_crumb(result["text"])
        assert reparsed["headers"]["kind"] == "mem"
        assert reparsed["sections"]["consolidated"] == ["- prefer concise answers", ""]

    def test_mem_with_chain_header(self):
        chain_ids = ["root-1", "mid-2"]
        result = emit_mem(
            title="Final",
            consolidated="the answer",
            source="reducer",
            upstream="mid-2",
            chain=chain_ids,
            namespace="debate",
        )
        assert result["headers"]["ext.debate.chain"] == "root-1,mid-2"
        assert result["headers"]["ext.debate.upstream"] == "mid-2"
        ext_line = result["headers"]["extensions"]
        assert "ext.debate.upstream.v1" in ext_line
        assert "ext.debate.chain.v1" in ext_line


class TestNewId:
    def test_ids_are_unique(self):
        ids = {new_id() for _ in range(100)}
        assert len(ids) == 100

    def test_prefix_is_respected(self):
        assert new_id("task").startswith("task-")


class TestWalkChain:
    def _make_chain(self, namespace="crumb"):
        a = emit_task(
            title="A", goal="g", context="c", constraints="k",
            source="s", namespace=namespace,
        )
        b = emit_mem(
            title="B", consolidated="out",
            source="s", upstream=a["id"], namespace=namespace,
        )
        c = emit_task(
            title="C", goal="g", context="c", constraints="k",
            source="s", upstream=b["id"], namespace=namespace,
        )
        by_id = {node["id"]: node for node in (a, b, c)}
        return by_id, a, b, c

    def test_walk_returns_root_first(self):
        by_id, a, b, c = self._make_chain()
        chain = walk_chain(by_id, c["id"])
        assert [n["id"] for n in chain] == [a["id"], b["id"], c["id"]]

    def test_walk_single_root(self):
        by_id, a, _, _ = self._make_chain()
        chain = walk_chain(by_id, a["id"])
        assert chain == [a]

    def test_walk_raises_on_missing_node(self):
        by_id, a, b, _ = self._make_chain()
        del by_id[a["id"]]
        with pytest.raises(ChainError, match="missing crumb"):
            walk_chain(by_id, b["id"])

    def test_walk_raises_on_cycle(self):
        # Manufacture a cycle by rewriting the upstream pointer of the root.
        by_id, a, b, c = self._make_chain()
        a["headers"]["ext.crumb.upstream"] = c["id"]
        with pytest.raises(ChainError, match="cycle"):
            walk_chain(by_id, c["id"])

    def test_namespace_isolation(self):
        by_id, a, b, c = self._make_chain(namespace="rag")
        # Walking with the wrong namespace treats every node as a root.
        chain = walk_chain(by_id, c["id"], namespace="debate")
        assert chain == [c]


class TestValidateChain:
    def test_valid_chain_returns_root_first(self):
        a = emit_task(title="A", goal="g", context="c", constraints="k",
                      source="s", namespace="rag")
        b = emit_mem(title="B", consolidated="r",
                     source="s", upstream=a["id"], namespace="rag")
        by_id = {n["id"]: n for n in (a, b)}
        chain = validate_chain(by_id, b["id"], namespace="rag")
        assert [n["id"] for n in chain] == [a["id"], b["id"]]

    def test_leaf_kind_enforced(self):
        a = emit_task(title="A", goal="g", context="c", constraints="k", source="s")
        b = emit_task(title="B", goal="g", context="c", constraints="k",
                      source="s", upstream=a["id"])
        by_id = {n["id"]: n for n in (a, b)}
        with pytest.raises(ChainError, match="expected 'mem'"):
            validate_chain(by_id, b["id"], expected_final_kind="mem")

    def test_upstream_declared_in_extensions(self):
        """A non-root crumb missing the ext declaration is rejected."""
        a = emit_task(title="A", goal="g", context="c", constraints="k", source="s")
        b = emit_mem(title="B", consolidated="r",
                     source="s", upstream=a["id"])
        # Strip the extension declaration from B, keeping the pointer.
        b["headers"]["extensions"] = ""
        by_id = {n["id"]: n for n in (a, b)}
        with pytest.raises(ChainError, match="does not declare"):
            validate_chain(by_id, b["id"])
