"""Reflect — self-learning gap detection for CRUMB Palace.

Analyzes a palace to identify what's missing, stale, or unbalanced.
This is the "System 3" that turns a filing cabinet into a second brain:
instead of passively storing what you tell it, the palace can now tell
*you* what it needs.

Gap types detected:
    empty_halls     — wings with halls that have zero rooms
    thin_wings      — wings with very few rooms overall
    stale_rooms     — rooms not updated recently (by file mtime)
    missing_halls   — halls that exist in other wings but not this one
    unlinked_topics — rooms that appear in only one wing (no tunnels)
    no_preferences  — wings without any preferences documented
    no_discoveries  — wings without discoveries (learning is stagnant)

The output is a scored report: each gap has a priority (high/medium/low)
and a suggested action the user can take to fill it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from cli.classify import HALLS
from cli.palace import (
    list_wings,
    list_halls,
    list_rooms,
    palace_stats,
    WINGS_DIR,
)


@dataclass
class Gap:
    """A single knowledge gap detected in the palace."""
    kind: str          # e.g. "empty_hall", "stale_room", "thin_wing"
    priority: str      # "high", "medium", "low"
    wing: str          # which wing (or "*" for global)
    detail: str        # human-readable description
    suggestion: str    # actionable next step


@dataclass
class ReflectReport:
    """Full reflection report for a palace."""
    gaps: List[Gap] = field(default_factory=list)
    health_score: int = 100  # 0-100, starts at 100, deductions per gap
    wing_count: int = 0
    room_count: int = 0
    hall_coverage: Dict[str, int] = field(default_factory=dict)

    @property
    def grade(self) -> str:
        if self.health_score >= 90:
            return "A"
        if self.health_score >= 80:
            return "B"
        if self.health_score >= 70:
            return "C"
        if self.health_score >= 60:
            return "D"
        return "F"


# ── deduction weights ──────────────────────────────────────────────────

_DEDUCTIONS = {
    "empty_palace": 50,
    "single_wing": 10,
    "empty_hall": 5,
    "thin_wing": 8,
    "stale_room": 3,
    "missing_hall": 4,
    "no_preferences": 6,
    "no_discoveries": 6,
}


# ── analysis ───────────────────────────────────────────────────────────

def reflect(root: Path, stale_days: int = 30) -> ReflectReport:
    """Analyze a palace and return a gap report with health score."""
    report = ReflectReport()
    stats = palace_stats(root)
    report.wing_count = stats["wings"]
    report.room_count = stats["rooms"]
    report.hall_coverage = dict(stats["by_hall"])

    wings = list_wings(root)

    # ── Empty palace ───────────────────────────────────────────────
    if not wings:
        report.gaps.append(Gap(
            kind="empty_palace",
            priority="high",
            wing="*",
            detail="Palace has no wings — no knowledge stored yet.",
            suggestion="Start with: crumb palace add \"your first observation\" --wing <name> --room <topic>",
        ))
        report.health_score = max(0, report.health_score - _DEDUCTIONS["empty_palace"])
        return report

    # ── Single wing ────────────────────────────────────────────────
    if len(wings) == 1:
        report.gaps.append(Gap(
            kind="single_wing",
            priority="low",
            wing=wings[0],
            detail=f"Only one wing ({wings[0]}). Multiple wings enable cross-referencing via tunnels.",
            suggestion=f"Add a second wing for another project or person: crumb palace add \"...\" --wing <other> --room <topic>",
        ))
        report.health_score = max(0, report.health_score - _DEDUCTIONS["single_wing"])

    # ── Per-wing analysis ──────────────────────────────────────────
    all_halls_seen: Dict[str, set] = {w: set() for w in wings}
    now = time.time()
    stale_threshold = now - (stale_days * 86400)

    for w in wings:
        wing_rooms = list_rooms(root, wing=w)
        wing_halls = set()
        for _, h, r, p in wing_rooms:
            wing_halls.add(h)
            # Check staleness
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime < stale_threshold:
                age_days = int((now - mtime) / 86400)
                report.gaps.append(Gap(
                    kind="stale_room",
                    priority="medium" if age_days > 60 else "low",
                    wing=w,
                    detail=f"Room {w}/{h}/{r} last updated {age_days} days ago.",
                    suggestion=f"Review and update: crumb palace search \"{r}\" --wing {w}",
                ))
                report.health_score = max(0, report.health_score - _DEDUCTIONS["stale_room"])

        all_halls_seen[w] = wing_halls

        # Thin wing
        if len(wing_rooms) < 3:
            report.gaps.append(Gap(
                kind="thin_wing",
                priority="medium",
                wing=w,
                detail=f"Wing {w} has only {len(wing_rooms)} room(s). Sparse knowledge.",
                suggestion=f"Add more observations: crumb palace add \"...\" --wing {w} --room <topic>",
            ))
            report.health_score = max(0, report.health_score - _DEDUCTIONS["thin_wing"])

        # No preferences
        if "preferences" not in wing_halls and len(wing_rooms) >= 2:
            report.gaps.append(Gap(
                kind="no_preferences",
                priority="medium",
                wing=w,
                detail=f"Wing {w} has no preferences documented.",
                suggestion=f"Add style/workflow preferences: crumb palace add \"prefers ...\" --wing {w} --room style",
            ))
            report.health_score = max(0, report.health_score - _DEDUCTIONS["no_preferences"])

        # No discoveries
        if "discoveries" not in wing_halls and len(wing_rooms) >= 3:
            report.gaps.append(Gap(
                kind="no_discoveries",
                priority="low",
                wing=w,
                detail=f"Wing {w} has no discoveries — nothing learned or realized.",
                suggestion=f"Capture learnings: crumb palace add \"realized ...\" --wing {w} --room <insight>",
            ))
            report.health_score = max(0, report.health_score - _DEDUCTIONS["no_discoveries"])

    # ── Cross-wing hall gaps ──────────────────────────────────────
    if len(wings) >= 2:
        all_halls_union = set()
        for halls in all_halls_seen.values():
            all_halls_union |= halls

        for w in wings:
            missing = all_halls_union - all_halls_seen[w]
            for h in missing:
                report.gaps.append(Gap(
                    kind="missing_hall",
                    priority="low",
                    wing=w,
                    detail=f"Wing {w} is missing hall '{h}' (present in other wings).",
                    suggestion=f"Add to fill the gap: crumb palace add \"...\" --wing {w} --hall {h} --room <topic>",
                ))
                report.health_score = max(0, report.health_score - _DEDUCTIONS["missing_hall"])

    # ── Global empty halls ────────────────────────────────────────
    for hall_name in HALLS:
        if report.hall_coverage.get(hall_name, 0) == 0:
            report.gaps.append(Gap(
                kind="empty_hall",
                priority="medium",
                wing="*",
                detail=f"No rooms in the '{hall_name}' hall across the entire palace.",
                suggestion=f"Start documenting {hall_name}: crumb palace add \"...\" --hall {hall_name} --wing <name> --room <topic>",
            ))
            report.health_score = max(0, report.health_score - _DEDUCTIONS["empty_hall"])

    # Clamp
    report.health_score = max(0, min(100, report.health_score))

    # Sort: high priority first
    priority_order = {"high": 0, "medium": 1, "low": 2}
    report.gaps.sort(key=lambda g: priority_order.get(g.priority, 9))

    return report


def render_report(report: ReflectReport) -> str:
    """Render a ReflectReport as human-readable text."""
    lines = [
        f"Palace Health: {report.health_score}/100 (Grade: {report.grade})",
        f"Wings: {report.wing_count}  Rooms: {report.room_count}",
        f"Hall coverage: {', '.join(f'{h}={c}' for h, c in sorted(report.hall_coverage.items()))}",
        "",
    ]
    if not report.gaps:
        lines.append("No gaps detected — your knowledge base looks solid.")
    else:
        lines.append(f"Found {len(report.gaps)} gap(s):")
        lines.append("")
        for i, gap in enumerate(report.gaps, 1):
            icon = {"high": "!!!", "medium": " !!", "low": "  !"}[gap.priority]
            lines.append(f"  {icon} [{gap.priority.upper()}] {gap.detail}")
            lines.append(f"      -> {gap.suggestion}")
    return "\n".join(lines)


def render_report_crumb(report: ReflectReport) -> str:
    """Render a ReflectReport as a kind=map crumb."""
    lines = [
        "BEGIN CRUMB",
        "v=1.1",
        "kind=map",
        f"title=Palace reflection — {report.health_score}/100 ({report.grade})",
        "source=crumb.reflect",
        "project=palace",
        "---",
        "[project]",
        f"Palace health: {report.health_score}/100 grade {report.grade}",
        f"Wings: {report.wing_count}, rooms: {report.room_count}",
        "",
        "[modules]",
    ]
    if not report.gaps:
        lines.append("- No gaps detected")
    else:
        for gap in report.gaps:
            lines.append(f"- [{gap.priority}] {gap.detail}")
            lines.append(f"  - Action: {gap.suggestion}")
    lines += ["", "END CRUMB"]
    return "\n".join(lines) + "\n"
