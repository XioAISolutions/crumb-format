"""Tests for cli.reflect — self-learning gap detection for palace."""

import time
import pytest
from pathlib import Path
from cli.palace import (
    init_palace, add_observation, rebuild_index,
)
from cli.reflect import (
    reflect, render_report, render_report_crumb, Gap, ReflectReport,
)
from cli.crumb import parse_crumb


# ── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def palace(tmp_path):
    return init_palace(tmp_path)


@pytest.fixture
def populated_palace(palace):
    add_observation(palace, "orion", "facts", "db-choice", "decided to use Postgres")
    add_observation(palace, "orion", "events", "launch", "shipped v0.1 on 2026-04-01")
    add_observation(palace, "orion", "discoveries", "caching", "turns out CDN was caching 404s")
    add_observation(palace, "orion", "preferences", "style", "prefers concise commits")
    add_observation(palace, "orion", "advice", "ci-tips", "never skip integration tests")
    add_observation(palace, "nova", "facts", "db-choice", "went with SQLite for nova")
    rebuild_index(palace)
    return palace


# ── empty palace ───────────────────────────────────────────────────────

def test_empty_palace(palace):
    report = reflect(palace)
    assert report.health_score < 60
    assert report.wing_count == 0
    assert report.room_count == 0
    assert any(g.kind == "empty_palace" for g in report.gaps)


def test_empty_palace_grade(palace):
    report = reflect(palace)
    assert report.grade in ("D", "F")


# ── populated palace ───────────────────────────────────────────────────

def test_populated_has_score(populated_palace):
    report = reflect(populated_palace)
    assert 0 <= report.health_score <= 100
    assert report.wing_count == 2
    assert report.room_count == 6


def test_populated_detects_thin_wing(populated_palace):
    report = reflect(populated_palace)
    thin = [g for g in report.gaps if g.kind == "thin_wing"]
    assert len(thin) >= 1
    assert thin[0].wing == "nova"


def test_populated_detects_missing_halls(populated_palace):
    report = reflect(populated_palace)
    missing = [g for g in report.gaps if g.kind == "missing_hall"]
    # nova only has facts, so it's missing events/discoveries/preferences/advice
    nova_missing = [g for g in missing if g.wing == "nova"]
    assert len(nova_missing) >= 3


def test_no_preferences_gap(populated_palace):
    """nova has 1 room in facts, no preferences. But it's only 1 room,
    so no_preferences won't trigger (requires >= 2 rooms)."""
    report = reflect(populated_palace)
    no_prefs = [g for g in report.gaps if g.kind == "no_preferences" and g.wing == "nova"]
    # nova has only 1 room, threshold is 2, so no_preferences shouldn't fire
    assert len(no_prefs) == 0


# ── single wing ────────────────────────────────────────────────────────

def test_single_wing_gap(palace):
    add_observation(palace, "solo", "facts", "db", "uses Postgres")
    add_observation(palace, "solo", "events", "launch", "shipped v1")
    add_observation(palace, "solo", "preferences", "style", "prefers short code")
    report = reflect(palace)
    singles = [g for g in report.gaps if g.kind == "single_wing"]
    assert len(singles) == 1


# ── stale detection ────────────────────────────────────────────────────

def test_stale_room_detection(palace):
    add_observation(palace, "old", "facts", "ancient", "some old fact")
    # Manually backdate the file's mtime
    from cli.palace import _room_path
    room_file = _room_path(palace, "old", "facts", "ancient")
    old_time = time.time() - (60 * 86400)  # 60 days ago
    import os
    os.utime(room_file, (old_time, old_time))

    report = reflect(palace, stale_days=30)
    stale = [g for g in report.gaps if g.kind == "stale_room"]
    assert len(stale) >= 1
    assert "ancient" in stale[0].detail


def test_not_stale_when_recent(populated_palace):
    report = reflect(populated_palace, stale_days=30)
    stale = [g for g in report.gaps if g.kind == "stale_room"]
    assert len(stale) == 0


# ── health score ───────────────────────────────────────────────────────

def test_full_palace_has_higher_score(palace):
    for h in ["facts", "events", "discoveries", "preferences", "advice"]:
        add_observation(palace, "alpha", h, f"room-{h}", f"content for {h}")
        add_observation(palace, "beta", h, f"room-{h}", f"content for {h}")
    report = reflect(palace)
    assert report.health_score >= 80
    assert report.grade in ("A", "B")


def test_score_never_negative(palace):
    report = reflect(palace)
    assert report.health_score >= 0


# ── grade boundaries ───────────────────────────────────────────────────

def test_grade_a():
    r = ReflectReport(health_score=95)
    assert r.grade == "A"

def test_grade_b():
    r = ReflectReport(health_score=85)
    assert r.grade == "B"

def test_grade_c():
    r = ReflectReport(health_score=72)
    assert r.grade == "C"

def test_grade_d():
    r = ReflectReport(health_score=65)
    assert r.grade == "D"

def test_grade_f():
    r = ReflectReport(health_score=55)
    assert r.grade == "F"


# ── rendering ──────────────────────────────────────────────────────────

def test_render_text(populated_palace):
    report = reflect(populated_palace)
    text = render_report(report)
    assert "Palace Health:" in text
    assert "/100" in text
    assert "gap(s)" in text or "No gaps" in text


def test_render_crumb_is_valid(populated_palace):
    report = reflect(populated_palace)
    crumb_text = render_report_crumb(report)
    parsed = parse_crumb(crumb_text)
    assert parsed["headers"]["kind"] == "map"
    assert "reflection" in parsed["headers"]["title"].lower()


def test_render_empty_palace(palace):
    report = reflect(palace)
    text = render_report(report)
    assert "Palace Health:" in text


# ── gap sorting ────────────────────────────────────────────────────────

def test_gaps_sorted_by_priority(populated_palace):
    report = reflect(populated_palace)
    if len(report.gaps) >= 2:
        priorities = [g.priority for g in report.gaps]
        order = {"high": 0, "medium": 1, "low": 2}
        assert priorities == sorted(priorities, key=lambda p: order[p])


# ── global empty hall detection ────────────────────────────────────────

def test_global_empty_hall(palace):
    add_observation(palace, "proj", "facts", "db", "uses Postgres")
    report = reflect(palace)
    empty_halls = [g for g in report.gaps if g.kind == "empty_hall"]
    empty_hall_names = [g.detail for g in empty_halls]
    # events, discoveries, preferences, advice should all be flagged
    assert len(empty_halls) == 4
