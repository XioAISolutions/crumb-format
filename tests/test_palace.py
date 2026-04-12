"""Tests for cli.palace — hierarchical spatial memory and wake-up crumbs."""

import pytest
from pathlib import Path
from cli.palace import (
    slugify, find_palace, init_palace, add_observation,
    list_wings, list_halls, list_rooms, palace_stats,
    palace_search, rebuild_tunnels, rebuild_index, build_wake_crumb,
    PALACE_DIRNAME, WINGS_DIR, TUNNELS_FILE, INDEX_FILE,
)
from cli.crumb import parse_crumb


# ── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def palace(tmp_path):
    """Create a fresh palace and return its root."""
    return init_palace(tmp_path)


@pytest.fixture
def populated_palace(palace):
    """Palace with several wings/halls/rooms for search and tunnel tests."""
    add_observation(palace, "orion", "facts", "db-choice", "decided to use Postgres")
    add_observation(palace, "orion", "facts", "db-choice", "migration deadline 2026-05-01")
    add_observation(palace, "orion", "events", "launch", "shipped v0.1 on 2026-04-01")
    add_observation(palace, "orion", "discoveries", "caching", "turns out the CDN was caching 404s")
    add_observation(palace, "nova", "facts", "db-choice", "went with SQLite for nova")
    add_observation(palace, "nova", "preferences", "style", "prefers concise commits")
    add_observation(palace, "nova", "advice", "testing", "never skip integration tests")
    rebuild_index(palace)
    return palace


# ── slugify ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("inp, expected", [
    ("orion-project", "orion-project"),
    ("My Cool Project", "my-cool-project"),
    ("auth/login bug", "auth-login-bug"),
    ("  spaces  ", "spaces"),
    ("UPPERCASE", "uppercase"),
    ("", "untitled"),
    ("---", "untitled"),
])
def test_slugify(inp, expected):
    assert slugify(inp) == expected


# ── init ────────────────────────────────────────────────────────────────

def test_init_palace(tmp_path):
    root = init_palace(tmp_path)
    assert root == tmp_path / PALACE_DIRNAME
    assert (root / WINGS_DIR).is_dir()
    assert (root / INDEX_FILE).exists()
    # Index is valid crumb
    parse_crumb((root / INDEX_FILE).read_text())


def test_init_palace_idempotent(tmp_path):
    r1 = init_palace(tmp_path)
    r2 = init_palace(tmp_path)
    assert r1 == r2


# ── find_palace ─────────────────────────────────────────────────────────

def test_find_palace(palace, tmp_path):
    found = find_palace(tmp_path)
    assert found == palace


def test_find_palace_from_subdirectory(palace, tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    found = find_palace(nested)
    assert found == palace


def test_find_palace_missing(tmp_path):
    empty = tmp_path / "nope"
    empty.mkdir()
    assert find_palace(empty) is None


# ── add_observation ─────────────────────────────────────────────────────

def test_add_creates_room(palace):
    path = add_observation(palace, "orion", "facts", "db-choice", "using Postgres")
    assert path.exists()
    content = path.read_text()
    parsed = parse_crumb(content)
    assert parsed["headers"]["kind"] == "mem"
    assert parsed["headers"]["wing"] == "orion"
    assert parsed["headers"]["hall"] == "facts"
    assert parsed["headers"]["room"] == "db-choice"
    assert any("Postgres" in line for line in parsed["sections"]["consolidated"])


def test_add_appends_to_existing_room(palace):
    add_observation(palace, "orion", "facts", "db-choice", "using Postgres")
    path = add_observation(palace, "orion", "facts", "db-choice", "migration planned for May")
    content = path.read_text()
    parsed = parse_crumb(content)
    bullets = [l.strip() for l in parsed["sections"]["consolidated"] if l.strip().startswith("-")]
    assert len(bullets) == 2
    assert any("Postgres" in b for b in bullets)
    assert any("migration" in b for b in bullets)


def test_add_bad_hall(palace):
    with pytest.raises(ValueError, match="unknown hall"):
        add_observation(palace, "orion", "bogus", "room", "text")


# ── list functions ──────────────────────────────────────────────────────

def test_list_wings(populated_palace):
    assert list_wings(populated_palace) == ["nova", "orion"]


def test_list_halls(populated_palace):
    halls = list_halls(populated_palace, "orion")
    assert "facts" in halls
    assert "events" in halls
    assert "discoveries" in halls


def test_list_rooms_all(populated_palace):
    rooms = list_rooms(populated_palace)
    assert len(rooms) == 6


def test_list_rooms_filtered_by_wing(populated_palace):
    rooms = list_rooms(populated_palace, wing="nova")
    assert all(w == "nova" for w, _, _, _ in rooms)
    # nova has: facts/db-choice, preferences/style, advice/testing
    assert len(rooms) == 3


def test_list_rooms_filtered_by_hall(populated_palace):
    rooms = list_rooms(populated_palace, hall="facts")
    assert all(h == "facts" for _, h, _, _ in rooms)
    assert len(rooms) == 2


def test_list_rooms_filtered_both(populated_palace):
    rooms = list_rooms(populated_palace, wing="orion", hall="facts")
    assert len(rooms) == 1
    assert rooms[0][:3] == ("orion", "facts", "db-choice")


# ── stats ───────────────────────────────────────────────────────────────

def test_palace_stats(populated_palace):
    stats = palace_stats(populated_palace)
    assert stats["wings"] == 2
    assert stats["rooms"] == 6
    assert stats["by_hall"]["facts"] == 2
    assert stats["by_hall"]["events"] == 1
    assert stats["by_hall"]["discoveries"] == 1
    assert stats["by_hall"]["preferences"] == 1
    assert stats["by_hall"]["advice"] == 1


# ── search ──────────────────────────────────────────────────────────────

def test_search_basic(populated_palace):
    results = palace_search(populated_palace, "postgres")
    assert len(results) >= 1
    assert results[0][2] == "db-choice"


def test_search_no_match(populated_palace):
    assert palace_search(populated_palace, "nonexistent42") == []


def test_search_filtered(populated_palace):
    results = palace_search(populated_palace, "db-choice", wing="nova")
    # Should only find nova's db-choice (file path contains the slug)
    assert all(r[0] == "nova" for r in results)


# ── tunnels ─────────────────────────────────────────────────────────────

def test_rebuild_tunnels(populated_palace):
    path = rebuild_tunnels(populated_palace)
    assert path.exists()
    content = path.read_text()
    parsed = parse_crumb(content)
    assert parsed["headers"]["kind"] == "map"
    # db-choice exists in both orion and nova → should be a tunnel
    modules_text = "\n".join(parsed["sections"]["modules"])
    assert "db-choice" in modules_text


def test_tunnels_empty_palace(palace):
    path = rebuild_tunnels(palace)
    content = path.read_text()
    parsed = parse_crumb(content)
    assert "no cross-wing tunnels yet" in "\n".join(parsed["sections"]["modules"])


# ── index ───────────────────────────────────────────────────────────────

def test_rebuild_index(populated_palace):
    path = rebuild_index(populated_palace)
    assert path.exists()
    parsed = parse_crumb(path.read_text())
    assert parsed["headers"]["kind"] == "map"
    modules_text = "\n".join(parsed["sections"]["modules"])
    assert "orion" in modules_text
    assert "nova" in modules_text


# ── wake-up crumb ───────────────────────────────────────────────────────

def test_build_wake_crumb(populated_palace):
    wake = build_wake_crumb(populated_palace)
    parsed = parse_crumb(wake)
    assert parsed["headers"]["kind"] == "wake"
    assert "identity" in parsed["sections"]
    assert "facts" in parsed["sections"]
    assert "rooms" in parsed["sections"]
    # Should have facts from the facts hall
    facts_text = "\n".join(parsed["sections"]["facts"])
    assert "Postgres" in facts_text or "SQLite" in facts_text


def test_wake_crumb_empty_palace(palace):
    wake = build_wake_crumb(palace)
    parsed = parse_crumb(wake)
    assert parsed["headers"]["kind"] == "wake"
    assert "no facts recorded yet" in "\n".join(parsed["sections"]["facts"])
    assert "palace is empty" in "\n".join(parsed["sections"]["rooms"])


def test_wake_max_facts(populated_palace):
    wake = build_wake_crumb(populated_palace, max_facts=1)
    parsed = parse_crumb(wake)
    fact_bullets = [l for l in parsed["sections"]["facts"] if l.strip().startswith("-")]
    assert len(fact_bullets) == 1


# ── room crumbs validate ───────────────────────────────────────────────

def test_all_rooms_validate(populated_palace):
    """Every room crumb in the palace must pass parse_crumb without error."""
    for wing, hall, room, path in list_rooms(populated_palace):
        content = path.read_text()
        parsed = parse_crumb(content)
        assert parsed["headers"]["kind"] == "mem"
        assert parsed["headers"]["wing"] == wing.replace("-", "-")
        assert parsed["headers"]["hall"] == hall
