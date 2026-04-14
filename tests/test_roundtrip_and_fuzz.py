"""Round-trip and Hypothesis fuzz tests for the CRUMB parser/renderer.

These tests pin two stability properties of the v=1.1 reference parser:

    1. ``render`` is a fixed point after one pass:
           render(parse(render(parse(text)))) == render(parse(text))

    2. ``parse`` is total: it either returns a well-shaped dict or raises
       ``ValueError`` — never a different exception, never a crash.

Property 1 is what conformance fixtures rely on. Property 2 is the parser-DoS
mitigation called out in ``docs/THREAT_MODEL.md`` (T2).
"""

from __future__ import annotations

import glob
import string
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from cli import crumb

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_FILES = sorted(
    glob.glob(str(REPO_ROOT / "fixtures" / "valid" / "*.crumb"))
    + glob.glob(str(REPO_ROOT / "fixtures" / "extensions" / "*.crumb"))
)


# ---------------------------------------------------------------------------
# Round-trip tests over the conformance fixture suite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_path", FIXTURE_FILES)
def test_render_is_fixed_point_after_one_pass(fixture_path):
    """render is idempotent: rendering twice yields the same string."""
    text = Path(fixture_path).read_text(encoding="utf-8")
    parsed = crumb.parse_crumb(text)
    once = crumb.render_crumb(parsed["headers"], parsed["sections"])
    twice = crumb.render_crumb(
        crumb.parse_crumb(once)["headers"],
        crumb.parse_crumb(once)["sections"],
    )
    assert once == twice, f"render not idempotent for {fixture_path}"


@pytest.mark.parametrize("fixture_path", FIXTURE_FILES)
def test_parse_is_stable_on_rendered_output(fixture_path):
    """parse(render(parsed)) is itself a fixed point under further round-trips."""
    text = Path(fixture_path).read_text(encoding="utf-8")
    parsed = crumb.parse_crumb(text)
    rendered = crumb.render_crumb(parsed["headers"], parsed["sections"])
    p2 = crumb.parse_crumb(rendered)
    p3 = crumb.parse_crumb(
        crumb.render_crumb(p2["headers"], p2["sections"])
    )
    assert p2 == p3, f"parse not stable for {fixture_path}"


@pytest.mark.parametrize("fixture_path", FIXTURE_FILES)
def test_render_preserves_required_headers(fixture_path):
    """Every rendered crumb still validates: required headers and sections survive."""
    text = Path(fixture_path).read_text(encoding="utf-8")
    parsed = crumb.parse_crumb(text)
    rendered = crumb.render_crumb(parsed["headers"], parsed["sections"])
    reparsed = crumb.parse_crumb(rendered)
    for required in ("v", "kind", "source"):
        assert required in reparsed["headers"]


# ---------------------------------------------------------------------------
# Hypothesis fuzz tests — parser must never crash with anything other than
# ValueError, no matter what bytes it sees.
# ---------------------------------------------------------------------------


# Strategy: arbitrary printable text with newlines, capped at a few KB.
_arbitrary_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Zs"),
        whitelist_characters="\n\r\t -=[]_.",
    ),
    max_size=4096,
)


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(_arbitrary_text)
def test_parser_never_crashes_on_arbitrary_input(text):
    """Property: parse_crumb on any string returns a dict or raises ValueError."""
    try:
        result = crumb.parse_crumb(text)
    except ValueError:
        return  # expected failure mode
    # If it parsed, the shape must be sane.
    assert isinstance(result, dict)
    assert "headers" in result and isinstance(result["headers"], dict)
    assert "sections" in result and isinstance(result["sections"], dict)


# Strategy: well-formed envelopes with random body bytes between the markers.
@st.composite
def _wrapped_envelope(draw):
    body = draw(st.text(max_size=2048))
    return f"BEGIN CRUMB\n{body}\nEND CRUMB\n"


@settings(max_examples=200, deadline=None)
@given(_wrapped_envelope())
def test_envelope_with_random_body_does_not_crash(text):
    """Even with markers in place, the parser must reject cleanly."""
    try:
        crumb.parse_crumb(text)
    except ValueError:
        pass  # expected for almost all random bodies


# Strategy: a structurally valid crumb with hypothesis-generated header values
# and section bodies. The parser MUST accept it, and round-trip through render.
_safe_value_chars = string.ascii_letters + string.digits + " _-./:"


# Post-filter strings so that ``.strip()`` yields a non-empty string —
# purely-whitespace values would produce an empty section body, which the
# parser rightly rejects but the rest of this strategy is trying to avoid.
_nonblank = st.text(alphabet=_safe_value_chars, min_size=1, max_size=80).filter(
    lambda s: bool(s.strip())
)
_nonblank_short = st.text(alphabet=_safe_value_chars, min_size=1, max_size=40).filter(
    lambda s: bool(s.strip())
)


@st.composite
def _valid_task_crumb(draw):
    title = draw(_nonblank_short)
    source = draw(
        st.text(alphabet=string.ascii_lowercase + ".", min_size=1, max_size=20).filter(
            lambda s: bool(s.strip())
        )
    )
    goal = draw(_nonblank)
    ctx_lines = draw(st.lists(_nonblank_short, min_size=1, max_size=5))
    constr_lines = draw(st.lists(_nonblank_short, min_size=1, max_size=5))
    body = (
        f"BEGIN CRUMB\n"
        f"v=1.1\n"
        f"kind=task\n"
        f"title={title.strip()}\n"
        f"source={source.strip()}\n"
        f"---\n"
        f"[goal]\n{goal.strip()}\n\n"
        f"[context]\n" + "\n".join(f"- {l.strip()}" for l in ctx_lines) + "\n\n"
        f"[constraints]\n" + "\n".join(f"- {l.strip()}" for l in constr_lines) + "\n"
        f"END CRUMB\n"
    )
    return body


@settings(max_examples=150, deadline=None)
@given(_valid_task_crumb())
def test_well_formed_task_crumbs_round_trip(text):
    """Hypothesis-generated valid task crumbs must parse and survive a render pass."""
    parsed = crumb.parse_crumb(text)
    assert parsed["headers"]["kind"] == "task"
    rendered = crumb.render_crumb(parsed["headers"], parsed["sections"])
    re_parsed = crumb.parse_crumb(rendered)
    # Headers preserved exactly.
    assert re_parsed["headers"] == parsed["headers"]
    # Required sections preserved (their *content* is preserved modulo a possible
    # trailing blank line that ``render`` adds — we check non-empty equivalence).
    for section in ("goal", "context", "constraints"):
        original = [l for l in parsed["sections"][section] if l.strip()]
        replayed = [l for l in re_parsed["sections"][section] if l.strip()]
        assert original == replayed
