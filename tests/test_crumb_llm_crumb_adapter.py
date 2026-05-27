"""Crumb adapter: parse a real crumb, build priors, run through the model."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from crumb_llm.crumb_adapter import CrumbPriorBuilder, iter_crumb_files  # noqa: E402
from crumb_llm.model import WaveFieldLM, WaveFieldConfig  # noqa: E402


SAMPLE_CRUMB = """\
BEGIN CRUMB
v=1.2
kind=task
title=Fix login bug
source=cursor.agent
---
[goal]
@priority: 9
Fix the JWT redirect loop.

[context]
@priority: 5
- Affects 5% of users.
- Started after deploy on 2026-01-15.

[constraints]
- Must keep backwards compat.
END CRUMB
"""


def test_builder_emits_expected_tensor_shapes():
    builder = CrumbPriorBuilder()
    priors = builder.build(SAMPLE_CRUMB)
    assert priors.input_ids.dim() == 2
    assert priors.input_ids.shape == priors.scatter_weights.shape
    assert priors.input_ids.shape == priors.section_ids.shape
    assert priors.input_ids.size(0) == 1


def test_builder_priority_scales_weight():
    """Section with @priority: 9 should produce larger amplitudes than priority: 5."""
    builder = CrumbPriorBuilder()
    priors = builder.build(SAMPLE_CRUMB)
    # Distinct positive amplitudes appear; the largest one corresponds
    # to the highest-priority section.
    uniq = priors.scatter_weights.unique().tolist()
    assert max(uniq) > 1.0   # priority 9 → 9/5 = 1.8 (no fold boost on [goal])
    assert min(uniq) == 0.0  # section separators get weight 0


def test_builder_extracts_refs():
    crumb_with_refs = SAMPLE_CRUMB.replace(
        "source=cursor.agent",
        "source=cursor.agent\nrefs=sha256:" + "a" * 64,
    )
    builder = CrumbPriorBuilder()
    priors = builder.build(crumb_with_refs)
    assert len(priors.ref_digests) == 1
    assert priors.ref_digests[0].startswith("sha256:")


def test_priors_run_through_model():
    builder = CrumbPriorBuilder()
    priors = builder.build(SAMPLE_CRUMB)
    cfg = WaveFieldConfig(
        vocab_size=256, dim=16, n_layers=2, n_heads=2,
        field_size=max(64, priors.input_ids.size(1)),
    )
    model = WaveFieldLM(cfg)
    out = model(
        priors.input_ids[:, : cfg.field_size],
        scatter_weights=priors.scatter_weights[:, : cfg.field_size],
    )
    assert out["logits"].shape[-1] == cfg.vocab_size


def test_iter_crumb_files_finds_examples():
    repo_root = Path(__file__).resolve().parent.parent
    examples = repo_root / "examples"
    if not examples.exists():
        pytest.skip("examples/ not present")
    files = list(iter_crumb_files(examples))
    assert len(files) > 0
    assert all(f.suffix == ".crumb" for f in files)
