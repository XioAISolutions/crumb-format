"""Palace — hierarchical spatial memory for CRUMB.

A palace organizes `kind=mem` crumbs into a four-level layout:

    wings   — a person, project, or topic (top-level container)
    halls   — memory types within a wing (facts/events/discoveries/
              preferences/advice)
    rooms   — specific named topics inside a hall (one .crumb per room)
    tunnels — cross-wing links between rooms that share a name

Unlike database-backed memory systems, a palace is **pure filesystem** —
every room is a standalone .crumb file, so everything is grep-able, git-able,
diff-able, validatable with `crumb validate`, searchable with `crumb search`,
and compressible with `crumb metalk`. No external deps, no servers.

Layout on disk (rooted at `.crumb-palace/`):

    .crumb-palace/
        wings/
            <wing-slug>/
                <hall>/
                    <room-slug>.crumb     # kind=mem
        tunnels.crumb                     # kind=map, cross-wing links
        index.crumb                       # kind=map, full room index

Wake-up:
    `build_wake_crumb()` emits a compact `kind=wake` crumb containing
    identity, top facts, and a room index — designed to be pasted at the
    start of a new AI session so the next tool knows your world before
    you type a single message.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cli.classify import HALLS

PALACE_DIRNAME = ".crumb-palace"
WINGS_DIR = "wings"
TUNNELS_FILE = "tunnels.crumb"
INDEX_FILE = "index.crumb"


# ── path helpers ────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    """Convert a wing/room name into a filesystem-safe slug."""
    s = (name or "").lower().strip()
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-") or "untitled"


def find_palace(start: Optional[Path] = None) -> Optional[Path]:
    """Walk upward from `start` looking for a `.crumb-palace/` directory."""
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        candidate = parent / PALACE_DIRNAME
        if candidate.is_dir():
            return candidate
    return None


def init_palace(path: Path) -> Path:
    """Create a fresh palace rooted at `path / .crumb-palace`."""
    root = Path(path) / PALACE_DIRNAME
    (root / WINGS_DIR).mkdir(parents=True, exist_ok=True)
    index_path = root / INDEX_FILE
    if not index_path.exists():
        index_path.write_text(_render_index([]))
    return root


def _room_path(root: Path, wing: str, hall: str, room: str) -> Path:
    return root / WINGS_DIR / slugify(wing) / hall / f"{slugify(room)}.crumb"


# ── crumb builders ──────────────────────────────────────────────────────

def _new_room_crumb(wing: str, hall: str, room: str, first_entry: str) -> str:
    """Build an initial mem crumb for a brand-new room."""
    lines = [
        "BEGIN CRUMB",
        "v=1.1",
        "kind=mem",
        f"title={room}",
        "source=palace",
        f"wing={wing}",
        f"hall={hall}",
        f"room={room}",
        "---",
        "[consolidated]",
        _bullet(first_entry),
        "",
        "END CRUMB",
    ]
    return "\n".join(lines) + "\n"


def _bullet(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return "-"
    if stripped.startswith("-"):
        return stripped
    return f"- {stripped}"


def _append_to_consolidated(content: str, text: str) -> str:
    """Append a bullet to the [consolidated] section of a room crumb."""
    lines = content.rstrip().splitlines()
    while lines and lines[-1].strip() in ("END CRUMB", ""):
        lines.pop()
    lines.append(_bullet(text))
    lines.append("")
    lines.append("END CRUMB")
    return "\n".join(lines) + "\n"


# ── core ops ────────────────────────────────────────────────────────────

def add_observation(root: Path, wing: str, hall: str, room: str, text: str) -> Path:
    """File an observation into `wing/hall/room`. Creates the room if missing.

    Raises ValueError for unknown halls.
    """
    if hall not in HALLS:
        raise ValueError(f"unknown hall: {hall!r}. valid: {HALLS}")
    path = _room_path(root, wing, hall, room)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        content = path.read_text()
        if "[consolidated]" not in content:
            raise ValueError(f"room crumb missing [consolidated]: {path}")
        path.write_text(_append_to_consolidated(content, text))
    else:
        path.write_text(_new_room_crumb(wing, hall, room, text))
    return path


def list_wings(root: Path) -> List[str]:
    wings_dir = root / WINGS_DIR
    if not wings_dir.is_dir():
        return []
    return sorted(d.name for d in wings_dir.iterdir() if d.is_dir())


def list_halls(root: Path, wing: str) -> List[str]:
    wing_dir = root / WINGS_DIR / slugify(wing)
    if not wing_dir.is_dir():
        return []
    return sorted(d.name for d in wing_dir.iterdir() if d.is_dir() and d.name in HALLS)


def list_rooms(
    root: Path,
    wing: Optional[str] = None,
    hall: Optional[str] = None,
) -> List[Tuple[str, str, str, Path]]:
    """Return `(wing, hall, room, path)` tuples, optionally filtered."""
    results: List[Tuple[str, str, str, Path]] = []
    wings_filter = slugify(wing) if wing else None
    for w in list_wings(root):
        if wings_filter and w != wings_filter:
            continue
        for h in list_halls(root, w):
            if hall and h != hall:
                continue
            hall_dir = root / WINGS_DIR / w / h
            for f in sorted(hall_dir.glob("*.crumb")):
                results.append((w, h, f.stem, f))
    return results


def palace_stats(root: Path) -> Dict[str, object]:
    rooms = list_rooms(root)
    wings = list_wings(root)
    by_hall: Dict[str, int] = {h: 0 for h in HALLS}
    for _, h, _, _ in rooms:
        by_hall[h] = by_hall.get(h, 0) + 1
    return {
        "wings": len(wings),
        "rooms": len(rooms),
        "by_hall": by_hall,
        "wing_names": wings,
    }


def palace_search(
    root: Path,
    query: str,
    wing: Optional[str] = None,
    hall: Optional[str] = None,
) -> List[Tuple[str, str, str, Path, int]]:
    """Substring search across room bodies. Returns hits sorted by count."""
    needle = query.lower()
    matches: List[Tuple[str, str, str, Path, int]] = []
    for w, h, r, p in list_rooms(root, wing=wing, hall=hall):
        try:
            body = p.read_text().lower()
        except OSError:
            continue
        hits = body.count(needle)
        if hits:
            matches.append((w, h, r, p, hits))
    matches.sort(key=lambda m: -m[4])
    return matches


# ── tunnels + index ─────────────────────────────────────────────────────

def rebuild_tunnels(root: Path) -> Path:
    """Scan rooms and record cross-wing name matches in `tunnels.crumb`."""
    by_name: Dict[str, List[Tuple[str, str]]] = {}
    for w, h, r, _ in list_rooms(root):
        by_name.setdefault(r, []).append((w, h))
    tunnels: Dict[str, List[Tuple[str, str]]] = {}
    for name, locs in by_name.items():
        if len({loc[0] for loc in locs}) >= 2:
            tunnels[name] = locs
    path = root / TUNNELS_FILE
    path.write_text(_render_tunnels(tunnels))
    return path


def _render_tunnels(tunnels: Dict[str, List[Tuple[str, str]]]) -> str:
    lines = [
        "BEGIN CRUMB",
        "v=1.1",
        "kind=map",
        "title=Palace tunnels",
        "source=palace.tunnel",
        "project=palace",
        "---",
        "[project]",
        "Cross-wing room name index. Rooms with the same slug in multiple",
        "wings are linked here so related context can be followed across",
        "wings without full-palace search.",
        "",
        "[modules]",
    ]
    if not tunnels:
        lines.append("- (no cross-wing tunnels yet)")
    else:
        for room in sorted(tunnels.keys()):
            locs = tunnels[room]
            refs = ", ".join(f"{w}/{h}" for w, h in sorted(set(locs)))
            lines.append(f"- {room}: {refs}")
    lines += ["", "END CRUMB"]
    return "\n".join(lines) + "\n"


def rebuild_index(root: Path) -> Path:
    path = root / INDEX_FILE
    path.write_text(_render_index(list_rooms(root)))
    return path


def _render_index(rooms: List[Tuple[str, str, str, Path]]) -> str:
    lines = [
        "BEGIN CRUMB",
        "v=1.1",
        "kind=map",
        "title=Palace index",
        "source=palace.index",
        "project=palace",
        "---",
        "[project]",
        "Full room index grouped by wing and hall.",
        "",
        "[modules]",
    ]
    if not rooms:
        lines.append("- (palace is empty)")
    else:
        by_wing: Dict[str, List[Tuple[str, str]]] = {}
        for w, h, r, _ in rooms:
            by_wing.setdefault(w, []).append((h, r))
        for w in sorted(by_wing.keys()):
            lines.append(f"- {w}:")
            for h, r in sorted(by_wing[w]):
                lines.append(f"  - {h}/{r}")
    lines += ["", "END CRUMB"]
    return "\n".join(lines) + "\n"


# ── wake-up ─────────────────────────────────────────────────────────────

def _harvest_facts(root: Path, limit: int) -> List[str]:
    """Pull top bullets from hall=facts rooms across all wings."""
    harvested: List[str] = []
    for w, h, r, p in list_rooms(root, hall="facts"):
        try:
            body = p.read_text()
        except OSError:
            continue
        in_consolidated = False
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_consolidated = stripped == "[consolidated]"
                continue
            if not in_consolidated:
                continue
            if not stripped.startswith("-"):
                continue
            bullet = stripped.lstrip("- ").strip()
            if bullet:
                harvested.append(f"{w}/{r}: {bullet}")
                if len(harvested) >= limit:
                    return harvested
    return harvested


def build_wake_crumb(root: Path, max_facts: int = 8) -> str:
    """Build a `kind=wake` crumb summarizing the palace for session bootstrap.

    The crumb has three sections — [identity], [facts], [rooms] — designed
    to be pasted into an AI's system prompt or opening message so the next
    tool has a compact map of your world before the first real turn.
    """
    stats = palace_stats(root)
    rooms = list_rooms(root)
    facts = _harvest_facts(root, max_facts)

    lines = [
        "BEGIN CRUMB",
        "v=1.1",
        "kind=wake",
        "title=Session wake-up",
        "source=palace.wake",
        "---",
        "[identity]",
        f"- palace: {root}",
        f"- wings: {stats['wings']}, rooms: {stats['rooms']}",
        "",
        "[facts]",
    ]
    if not facts:
        lines.append("- (no facts recorded yet)")
    else:
        for f in facts:
            lines.append(f"- {f}")

    lines += ["", "[rooms]"]
    if not rooms:
        lines.append("- (palace is empty)")
    else:
        by_wing: Dict[str, Dict[str, int]] = {}
        for w, h, _, _ in rooms:
            by_wing.setdefault(w, {})
            by_wing[w][h] = by_wing[w].get(h, 0) + 1
        for w in sorted(by_wing.keys()):
            summary = ", ".join(f"{h}({c})" for h, c in sorted(by_wing[w].items()))
            lines.append(f"- {w}: {summary}")

    lines += ["", "END CRUMB"]
    return "\n".join(lines) + "\n"
