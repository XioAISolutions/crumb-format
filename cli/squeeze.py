"""Budget-aware CRUMB packer — the ``crumb squeeze`` command.

Takes a CRUMB and a token budget and produces a rendered CRUMB that fits,
by composing the pieces CRUMB already has:

  1. Elide refs whose content the receiver has seen (Layer 2).
  2. Drop ``[fold:X/full]`` variants, keeping ``/summary``.
  3. Drop optional sections by ``@priority`` (Layer 4), lowest first.
  4. Escalate MeTalk from level 1 → 2 → 3.
  5. If still over budget, raise.

Required sections are never dropped. ``@priority`` overrides the default
priority ordering so the caller can say "never drop [notes] before
[context]", which is closer to the PolarQuant trick of keeping the
components that carry the most signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from cli import crumb, hashing, metalk


DEFAULT_PRIORITY = 5
REQUIRED_PRIORITY = 10
SUMMARY_FOLD_PRIORITY = 9
FULL_FOLD_PRIORITY = 4


@dataclass
class SqueezeReport:
    original_tokens: int = 0
    final_tokens: int = 0
    metalk_level: int = 0
    dropped_sections: List[str] = field(default_factory=list)
    dropped_full_folds: List[str] = field(default_factory=list)
    elided_refs: List[str] = field(default_factory=list)
    budget: int = 0


def _required_section_names(kind: str) -> set[str]:
    names = set(crumb.REQUIRED_SECTIONS.get(kind, []))
    # A fold pair substituting a required section is also required — mark both sides.
    result: set[str] = set()
    for name in names:
        result.add(name)
        result.add(f"fold:{name}/summary")
        result.add(f"fold:{name}/full")
    return result


def _strip_annotations(body: List[str]) -> Tuple[List[str], int | None, str | None]:
    """Strip leading ``@type:`` / ``@priority:`` annotations and return remainder.

    Returns (cleaned_body, priority_or_None, type_or_None).
    """
    priority: int | None = None
    content_type: str | None = None
    cleaned: List[str] = list(body)
    for _ in range(2):
        idx = next((i for i, line in enumerate(cleaned) if line.strip()), None)
        if idx is None:
            break
        stripped = cleaned[idx].strip()
        if stripped.startswith("@priority:"):
            try:
                priority = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                priority = None
            cleaned.pop(idx)
            continue
        if stripped.startswith("@type:"):
            content_type = stripped.split(":", 1)[1].strip()
            cleaned.pop(idx)
            continue
        break
    return cleaned, priority, content_type


def _section_priority(name: str, body: List[str], required: set[str]) -> int:
    if name in required:
        return REQUIRED_PRIORITY
    _, explicit, _ = _strip_annotations(body)
    if explicit is not None:
        return explicit
    if name.startswith("fold:") and name.endswith("/summary"):
        return SUMMARY_FOLD_PRIORITY
    if name.startswith("fold:") and name.endswith("/full"):
        return FULL_FOLD_PRIORITY
    return DEFAULT_PRIORITY


def _elide_refs(headers: Dict[str, str], seen: set[str], report: SqueezeReport) -> None:
    value = headers.get("refs")
    if not value:
        return
    kept: List[str] = []
    for raw in value.split(","):
        ref = raw.strip()
        if not ref:
            continue
        if ref.startswith("sha256:") and hashing.digest_matches_set(ref, seen):
            report.elided_refs.append(ref)
            continue
        kept.append(ref)
    if kept:
        headers["refs"] = ", ".join(kept)
    else:
        headers.pop("refs", None)


def _drop_fold_full_variants(
    sections: Dict[str, List[str]],
    report: SqueezeReport,
    fold_priority: List[str] | None = None,
) -> bool:
    """Drop a single [fold:X/full] variant; return True if something was dropped.

    If ``fold_priority`` is supplied (v1.3 §2.2), folds NOT in the list are dropped
    first, and folds at the end of the list are dropped before folds at the front.
    """
    fold_pairs: Dict[str, Dict[str, str]] = {}
    for name in sections:
        match = crumb.FOLD_SECTION_RE.match(name)
        if not match:
            continue
        fold_name, variant = match.group(1), match.group(2)
        fold_pairs.setdefault(fold_name, {})[variant] = name

    candidates: List[Tuple[int, int, str, str]] = []
    priority_map = {name: idx for idx, name in enumerate(fold_priority or [])}
    unlisted_rank = len(priority_map)
    for fold_name, variants in fold_pairs.items():
        if "full" in variants and "summary" in variants:
            full_name = variants["full"]
            body = sections.get(full_name, [])
            priority = _section_priority(full_name, body, set())
            listed_rank = priority_map.get(fold_name, unlisted_rank)
            drop_rank = -listed_rank if listed_rank < unlisted_rank else 1
            candidates.append((drop_rank, priority, fold_name, full_name))

    if not candidates:
        return False

    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    _, _, fold_name, full_name = candidates[0]
    sections.pop(full_name, None)
    report.dropped_full_folds.append(fold_name)
    return True


def select_folds_size_greedy(
    sections: Dict[str, List[str]],
    budget: int,
    fold_priority: List[str] | None = None,
) -> Dict[str, str]:
    """v1.3 §2.1 — size-greedy fold selection.

    Returns a map {fold_name: "summary" | "full"} describing which variant of
    each fold pair fits within ``budget``. `/summary` is always loaded first;
    upgrades happen in declaration order (or ``fold_priority`` order if given).

    The returned map is advisory — callers decide what to do with dropped /full
    variants.
    """
    fold_pairs: Dict[str, Dict[str, str]] = {}
    declaration_order: List[str] = []
    for name in sections:
        match = crumb.FOLD_SECTION_RE.match(name)
        if not match:
            continue
        fold_name, variant = match.group(1), match.group(2)
        if fold_name not in fold_pairs:
            declaration_order.append(fold_name)
        fold_pairs.setdefault(fold_name, {})[variant] = name

    selection: Dict[str, str] = {}
    total = 0
    for fold_name, variants in fold_pairs.items():
        if "summary" in variants:
            selection[fold_name] = "summary"
            body = sections.get(variants["summary"], [])
            total += crumb.estimate_tokens("\n".join(body))

    order = fold_priority or declaration_order
    for fold_name in order:
        variants = fold_pairs.get(fold_name, {})
        if "full" not in variants or "summary" not in variants:
            continue
        summary_body = sections.get(variants["summary"], [])
        full_body = sections.get(variants["full"], [])
        delta = crumb.estimate_tokens("\n".join(full_body)) - crumb.estimate_tokens(
            "\n".join(summary_body)
        )
        if total + delta <= budget:
            selection[fold_name] = "full"
            total += delta
    return selection


def _drop_lowest_priority_optional(
    sections: Dict[str, List[str]], required: set[str], report: SqueezeReport
) -> bool:
    optional: List[Tuple[int, str]] = []
    for name, body in sections.items():
        if name in required:
            continue
        if crumb.FOLD_SECTION_RE.match(name) and name.endswith("/summary"):
            # never drop a /summary while its /full is still absent-or-present; that's
            # covered by the fold-pair rule above.
            continue
        priority = _section_priority(name, body, required)
        if priority >= REQUIRED_PRIORITY:
            continue
        optional.append((priority, name))

    if not optional:
        return False

    optional.sort(key=lambda row: (row[0], row[1]))
    _, name = optional[0]
    sections.pop(name, None)
    report.dropped_sections.append(name)
    return True


def squeeze_crumb(
    text: str,
    budget: int,
    seen: set[str] | None = None,
    metalk_max_level: int = 3,
) -> Tuple[str, SqueezeReport]:
    """Squeeze ``text`` down so its estimated tokens fit ``budget``.

    Raises ValueError if the budget cannot be met without dropping required
    sections.
    """
    if budget <= 0:
        raise ValueError("budget must be a positive integer")

    seen = seen or set()
    original_parsed = crumb.parse_crumb(text)
    headers = dict(original_parsed["headers"])
    sections = {name: list(body) for name, body in original_parsed["sections"].items()}
    kind = headers.get("kind", "")
    required = _required_section_names(kind)

    report = SqueezeReport(budget=budget, original_tokens=crumb.estimate_tokens(text))

    _elide_refs(headers, seen, report)

    def _render() -> str:
        return crumb.render_crumb(headers, sections)

    fold_priority_header = headers.get("fold_priority", "").strip()
    fold_priority = (
        [p.strip() for p in fold_priority_header.split(",") if p.strip()]
        if fold_priority_header
        else None
    )

    current = _render()
    while crumb.estimate_tokens(current) > budget:
        if _drop_fold_full_variants(sections, report, fold_priority):
            current = _render()
            continue
        if _drop_lowest_priority_optional(sections, required, report):
            current = _render()
            continue
        break

    if crumb.estimate_tokens(current) > budget:
        for level in range(1, max(1, metalk_max_level) + 1):
            candidate = metalk.encode(current, level=level)
            report.metalk_level = level
            current = candidate
            if crumb.estimate_tokens(current) <= budget:
                break

    report.final_tokens = crumb.estimate_tokens(current)
    if report.final_tokens > budget:
        raise ValueError(
            f"cannot fit {report.final_tokens} tokens in budget {budget} without dropping required sections. "
            f"try: --metalk-max-level 3, --seen <digests> the receiver already has, "
            f"or raise the budget."
        )
    return current, report


def format_report(report: SqueezeReport) -> str:
    saved = report.original_tokens - report.final_tokens
    pct = (saved / report.original_tokens * 100) if report.original_tokens else 0.0
    lines = [
        f"squeeze: {report.original_tokens} → {report.final_tokens} tokens "
        f"(budget {report.budget}, saved {saved} / {pct:.0f}%)",
    ]
    if report.elided_refs:
        lines.append(f"  elided refs: {', '.join(report.elided_refs)}")
    if report.dropped_full_folds:
        lines.append(f"  dropped /full folds: {', '.join(report.dropped_full_folds)}")
    if report.dropped_sections:
        lines.append(f"  dropped sections: {', '.join(report.dropped_sections)}")
    if report.metalk_level:
        lines.append(f"  MeTalk level applied: {report.metalk_level}")
    return "\n".join(lines)
