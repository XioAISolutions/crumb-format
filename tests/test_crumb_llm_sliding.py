"""Sliding-window field tests."""

import pytest

torch = pytest.importorskip("torch")

from crumb_llm.sliding import (  # noqa: E402
    shift_field,
    SlidingFieldManager,
    sliding_scatter,
)


def test_shift_field_zeros_right_edge():
    field = torch.ones(1, 1, 16, 1)
    shifted = shift_field(field, shift_cells=4)
    assert shifted.shape == field.shape
    # Left 12 cells should be 1, right 4 should be 0.
    assert shifted[0, 0, :12, 0].sum() == 12
    assert shifted[0, 0, 12:, 0].sum() == 0


def test_shift_field_zero_shift_is_identity():
    field = torch.randn(2, 3, 8, 4)
    torch.testing.assert_close(shift_field(field, 0), field)


def test_shift_field_full_shift_zeros_all():
    field = torch.ones(1, 1, 8, 1)
    assert shift_field(field, 8).sum() == 0
    assert shift_field(field, 100).sum() == 0


def test_sliding_manager_triggers_shift():
    mgr = SlidingFieldManager(field_size=64, stride=16)
    assert not mgr.should_shift(63)
    assert mgr.should_shift(64)
    field = torch.ones(1, 1, 64, 1)
    field = mgr.shift(field)
    assert mgr.window_start == 16
    assert not mgr.should_shift(64)  # now 64 - 16 = 48 < 64


def test_sliding_scatter_shape():
    v = torch.randn(1, 2, 8, 4)
    out = sliding_scatter(v, field_size=16, window_start=0)
    assert out.shape == (1, 2, 16, 4)
