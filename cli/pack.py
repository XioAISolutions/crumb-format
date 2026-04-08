"""Context pack builder for CRUMB."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import re
from dataclasses import dataclass
from pathlib import Path

from extensions import SPEC_URL, append_extension
from local_ai import DEFAULT_OLLAMA_MODEL, LocalAIError, ensure_ollama_available, extract_crumb_block, generate_text


@dataclass(frozen=True)
class ScoredLine:
    section: str
    text: str
    score: float
    origin: str
    category: str = "general"


def _crumb():
    return importlib.import_module("crumb")


def _query_terms(query: str) -> list[str]:
    crumb = _crumb()
    extracted = sorted(crumb.extract_keywords(query))
    return extracted or sorted(part for part in query.lower().split() if part.strip())


def _path_sort_key(path: Path) -> str:
    return path.as_posix()


def _matches_project(parsed: dict[str, object], path: Path, project: str | None) -> bool:
    if not project:
        return True
    project_lower = project.lower()
    headers = parsed["headers"]
    if headers.get("project", "").lower() == project_lower:
        return True
    if project_lower in path.as_posix().lower():
        return True
    title = headers.get("title", "").lower()
    return project_lower in title


def _query_hits_text(text: str, query_terms: list[str]) -> int:
    crumb = _crumb()
    text_keywords = crumb.extract_keywords(text)
    return sum(1 for term in query_terms if term in text_keywords)


def _file_relevance(path: Path, parsed: dict[str, object], query_terms: list[str]) -> int:
    headers = parsed["headers"]
    sections = parsed["sections"]
    title = headers.get("title", "")
    project = headers.get("project", "")
    body = " ".join(" ".join(lines) for lines in sections.values())
    path_text = path.as_posix()
    return max(
        _query_hits_text(title, query_terms),
        _query_hits_text(project, query_terms),
        _query_hits_text(path_text, query_terms),
        _query_hits_text(body, query_terms),
    )


def _pack_dedupe_key(text: str) -> str:
    crumb = _crumb()
    normalized = crumb.normalize_entry(text)
    normalized = re.sub(r"^[a-z ]+:\s*", "", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _keyword_signature(text: str) -> tuple[str, ...]:
    crumb = _crumb()
    keywords = sorted(crumb.extract_keywords(text))
    return tuple(keywords[:8])


def _constraint_signature(text: str) -> tuple[str, ...]:
    crumb = _crumb()
    keywords = crumb.extract_keywords(text)
    noise = {
        "keep",
        "preserve",
        "change",
        "add",
        "reuse",
        "maintain",
        "required",
        "packed",
        "handoff",
        "declared",
        "budget",
        "existing",
        "current",
    }
    core = sorted(keyword for keyword in keywords if keyword not in noise)
    return tuple(core[:6])


def _kind_weight(kind: str, mode: str) -> float:
    if mode == "debug":
        return {
            "task": 1.15,
            "mem": 0.95,
            "map": 1.0,
            "todo": 1.0,
            "log": 1.6,
        }.get(kind, 1.0)
    if mode == "review":
        return {
            "task": 1.2,
            "mem": 1.0,
            "map": 1.15,
            "todo": 0.95,
            "log": 0.85,
        }.get(kind, 1.0)
    return {
        "task": 1.35,
        "mem": 1.2,
        "map": 1.1,
        "todo": 1.0,
        "log": 0.65,
    }.get(kind, 1.0)


def _source_weight(headers: dict[str, str], mode: str) -> float:
    source = headers.get("source", "").strip().lower()
    extensions = {
        item.strip().lower()
        for item in headers.get("extensions", "").split(",")
        if item.strip()
    }
    if source == "crumb.pack" or "crumb.pack.v1" in extensions:
        return {
            "implement": 0.72,
            "review": 0.8,
            "debug": 0.88,
        }.get(mode, 0.8)
    return 1.0


def _rank_files(search_dir: Path, query_terms: list[str], strategy: str, project: str | None, mode: str) -> list[tuple[Path, str, dict[str, object], float]]:
    crumb = _crumb()
    files = [
        (path, text, parsed)
        for path, text, parsed in crumb._load_crumb_files(search_dir)
        if _matches_project(parsed, path, project)
    ]
    if not files:
        return []

    keyword_scores = {item["path"]: float(item["score"]) for item in crumb._search_keyword(query_terms, files)}
    ranked_scores = {item["path"]: float(item["score"]) for item in crumb._search_ranked(query_terms, files)}

    recency_order = sorted(files, key=lambda item: (item[0].stat().st_mtime, _path_sort_key(item[0])), reverse=True)
    recency_scores: dict[Path, float] = {}
    total = max(len(recency_order), 1)
    for index, (path, _, _) in enumerate(recency_order):
        recency_scores[path] = float(total - index)

    results: list[tuple[Path, str, dict[str, object], float]] = []
    for path, text, parsed in files:
        headers = parsed["headers"]
        kind = headers["kind"]
        keyword = keyword_scores.get(path, 0.0)
        ranked = ranked_scores.get(path, 0.0)
        recent = recency_scores.get(path, 0.0)
        relevance = _file_relevance(path, parsed, query_terms)
        kind_weight = _kind_weight(kind, mode)
        source_weight = _source_weight(headers, mode)

        if strategy == "keyword":
            score = keyword + (relevance * 4.0)
        elif strategy == "ranked":
            score = ranked + (relevance * 3.0)
        elif strategy == "recent":
            score = recent + (keyword * 0.5) + (relevance * 3.0)
        else:
            score = (keyword * 3.0) + (ranked * 2.0) + (relevance * 6.0) + (recent * 0.2)

        score *= kind_weight * source_weight

        has_relevance = relevance > 0 or ranked > 0
        if strategy != "recent" and not has_relevance:
            continue
        if score <= 0:
            continue
        results.append((path, text, parsed, score))

    results.sort(
        key=lambda item: (
            -item[3],
            -item[0].stat().st_mtime,
            item[2]["headers"].get("title", ""),
            _path_sort_key(item[0]),
        )
    )
    return results


def _bullet(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("- "):
        return stripped
    return f"- {stripped}"


def _clip_line(text: str, limit: int = 220) -> str:
    stripped = re.sub(r"\s+", " ", text.strip())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1].rstrip() + "…"


def _line_overlap(line: str, query_terms: list[str]) -> int:
    return _query_hits_text(line, query_terms)


def _line_signal(line: str) -> float:
    crumb = _crumb()
    norm = crumb.normalize_entry(line)
    keywords = crumb.extract_keywords(line)
    if not norm:
        return 0.0
    return float(len(keywords)) + (1.0 if re.search(r"[/_.:-]", line) else 0.0) + (1.0 if re.search(r"\d", line) else 0.0)


def _candidate_score(line: str, file_score: float, query_terms: list[str], section_bonus: float) -> float:
    overlap = _line_overlap(line, query_terms)
    signal = _line_signal(line)
    relevance_floor = 2.0 if overlap > 0 else 0.0
    return (file_score * 4.0) + (overlap * 8.0) + signal + section_bonus + relevance_floor


def _looks_like_constraint(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "prefer",
            "avoid",
            "must",
            "never",
            "always",
            "don't",
            "do not",
            "keep ",
            "preserve",
            "reuse",
            "maintain",
            "required",
        )
    )


def _is_relevant_line(text: str, query_terms: list[str], min_overlap: int = 1) -> bool:
    return _line_overlap(text, query_terms) >= min_overlap


def _task_candidates(
    ranked_files: list[tuple[Path, str, dict[str, object], float]],
    query_terms: list[str],
    mode: str,
) -> tuple[list[str], list[ScoredLine], list[ScoredLine], list[str]]:
    goal_lines: list[str] = []
    context_lines: list[ScoredLine] = []
    constraint_lines: list[ScoredLine] = []
    source_lines: list[str] = []

    seen_goal: set[str] = set()
    for path, _, parsed, file_score in ranked_files:
        headers = parsed["headers"]
        sections = parsed["sections"]
        crumb_kind = headers["kind"]
        title = headers.get("title", path.stem)
        source_lines.append(f"- {path.name} — {title}")

        if crumb_kind == "task":
            for raw in sections.get("goal", []):
                if not raw.strip():
                    continue
                goal = _clip_line(raw, 180)
                dedupe_key = _pack_dedupe_key(goal)
                if dedupe_key and dedupe_key not in seen_goal:
                    seen_goal.add(dedupe_key)
                    goal_lines.append(goal)

        for section_name, lines in sections.items():
            for raw in lines:
                if not raw.strip():
                    continue
                text = _clip_line(raw)
                origin = f"{path.name}[{section_name}]"

                if crumb_kind == "task" and section_name == "constraints":
                    constraint_lines.append(
                        ScoredLine("constraints", _bullet(text), _candidate_score(text, file_score, query_terms, 4.0), origin, "constraint")
                    )
                    continue

                if crumb_kind == "mem":
                    if _looks_like_constraint(text):
                        constraint_lines.append(
                            ScoredLine("constraints", _bullet(text), _candidate_score(text, file_score, query_terms, 3.0), origin, "constraint")
                        )
                    elif _is_relevant_line(text, query_terms):
                        context_lines.append(
                            ScoredLine("context", _bullet(text), _candidate_score(text, file_score, query_terms, 2.5), origin, "memory")
                        )
                    continue

                if crumb_kind == "todo":
                    if "[x]" in text.lower():
                        continue
                    if _is_relevant_line(text, query_terms):
                        context_lines.append(
                            ScoredLine("context", _bullet(f"Open task: {text}"), _candidate_score(text, file_score, query_terms, 2.0), origin, "todo")
                        )
                    continue

                if crumb_kind == "map" and section_name == "modules":
                    module_name = text.lstrip("- ").strip()
                    if _is_relevant_line(module_name, query_terms):
                        context_lines.append(
                            ScoredLine(
                                "context",
                                _bullet(f"Relevant module: {module_name}"),
                                _candidate_score(module_name, file_score, query_terms, 2.0),
                                origin,
                                "module",
                            )
                        )
                    continue

                if crumb_kind == "map" and section_name == "project":
                    if _is_relevant_line(text, query_terms):
                        context_lines.append(
                            ScoredLine("context", _bullet(text), _candidate_score(text, file_score, query_terms, 2.5), origin, "project")
                        )
                    continue

                if crumb_kind == "log":
                    if _is_relevant_line(text, query_terms) and mode in {"debug", "review"}:
                        context_lines.append(
                            ScoredLine("context", _bullet(text), _candidate_score(text, file_score, query_terms, 1.5), origin, "log")
                        )
                    continue

                if crumb_kind == "task" and section_name == "goal":
                    continue

                if crumb_kind == "task" and section_name == "context":
                    if _is_relevant_line(text, query_terms):
                        context_lines.append(
                            ScoredLine("context", _bullet(text), _candidate_score(text, file_score, query_terms, 3.0), origin, "evidence")
                        )
                    continue

                if crumb_kind == "task" and section_name == "notes" and _is_relevant_line(text, query_terms):
                    context_lines.append(
                        ScoredLine("context", _bullet(text), _candidate_score(text, file_score, query_terms, 1.5), origin, "note")
                    )

    return goal_lines, context_lines, constraint_lines, source_lines


def _mem_candidates(
    ranked_files: list[tuple[Path, str, dict[str, object], float]],
    query_terms: list[str],
) -> list[ScoredLine]:
    lines: list[ScoredLine] = []
    for path, _, parsed, file_score in ranked_files:
        sections = parsed["sections"]
        for section_name, section_lines in sections.items():
            for raw in section_lines:
                if not raw.strip():
                    continue
                text = _clip_line(raw)
                if not (_is_relevant_line(text, query_terms) or section_name in {"consolidated", "invariants"}):
                    continue
                bonus = 3.0 if section_name in {"consolidated", "invariants", "goal"} else 1.0
                lines.append(
                    ScoredLine(
                        "consolidated",
                        _bullet(text),
                        _candidate_score(text, file_score, query_terms, bonus),
                        f"{path.name}[{section_name}]",
                        "memory",
                    )
                )
    return lines


def _map_candidates(
    ranked_files: list[tuple[Path, str, dict[str, object], float]],
    query_terms: list[str],
) -> tuple[list[ScoredLine], list[ScoredLine]]:
    project_lines: list[ScoredLine] = []
    module_lines: list[ScoredLine] = []
    for path, _, parsed, file_score in ranked_files:
        sections = parsed["sections"]
        headers = parsed["headers"]

        for raw in sections.get("project", []):
            if raw.strip():
                text = _clip_line(raw, 180)
                if _is_relevant_line(text, query_terms):
                    project_lines.append(
                        ScoredLine("project", text, _candidate_score(text, file_score, query_terms, 4.0), f"{path.name}[project]", "project")
                    )

        for raw in sections.get("modules", []):
            if raw.strip():
                text = _bullet(raw.strip())
                if _is_relevant_line(text, query_terms):
                    module_lines.append(
                        ScoredLine("modules", text, _candidate_score(text, file_score, query_terms, 4.0), f"{path.name}[modules]", "module")
                    )

        title = headers.get("title")
        if title and _is_relevant_line(title, query_terms):
            project_lines.append(
                ScoredLine("project", _clip_line(title, 180), _candidate_score(title, file_score, query_terms, 2.0), f"{path.name}[title]", "project")
            )
    return project_lines, module_lines


def _dedupe_scored_lines(lines: list[ScoredLine]) -> list[ScoredLine]:
    chosen: dict[str, ScoredLine] = {}
    chosen_keys: list[str] = []
    constraint_signatures: list[tuple[str, ...]] = []
    keyword_signatures: list[tuple[str, ...]] = []
    for item in sorted(lines, key=lambda row: (-row.score, row.origin, row.text)):
        key = _pack_dedupe_key(item.text)
        if not key:
            continue
        bucket = _semantic_context_category(item)
        if bucket in {"scope", "module", "todo"}:
            key = f"{bucket}:{key}"
        if item.category == "constraint":
            signature = _constraint_signature(item.text)
            if signature and any(
                signature == existing
                or (
                    len(set(signature) & set(existing)) >= 2
                    and (
                        set(signature).issubset(set(existing))
                        or set(existing).issubset(set(signature))
                    )
                )
                for existing in constraint_signatures
            ):
                continue
        if key in chosen:
            continue
        if any(_similarity(key, existing) >= 0.9 for existing in chosen_keys):
            continue
        if item.category in {"evidence", "memory", "log", "note"}:
            signature = _keyword_signature(item.text)
            if signature and any(
                len(set(signature) & set(existing)) >= 3
                and (
                    set(signature).issubset(set(existing))
                    or set(existing).issubset(set(signature))
                )
                for existing in keyword_signatures
            ):
                continue
        chosen[key] = item
        chosen_keys.append(key)
        if item.category == "constraint":
            signature = _constraint_signature(item.text)
            if signature:
                constraint_signatures.append(signature)
        if item.category in {"evidence", "memory", "log", "note"}:
            signature = _keyword_signature(item.text)
            if signature:
                keyword_signatures.append(signature)
    return sorted(chosen.values(), key=lambda row: (-row.score, row.origin, row.text))


def _dedupe_plain_lines(lines: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        key = _pack_dedupe_key(line)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(line)
        if limit is not None and len(output) >= limit:
            break
    return output


def _rewrite_goal_for_mode(primary: str, query: str, mode: str, needs_regression: bool) -> str:
    stripped = primary.strip().rstrip(".")
    if not stripped:
        stripped = f"Resolve the work described by this pack for: {query}"

    if mode == "debug":
        focus = stripped
        lowered = focus.lower()
        if lowered.startswith("fix "):
            focus = "Diagnose " + focus[4:]
        elif lowered.startswith("implement "):
            focus = "Diagnose and verify " + focus[10:]
        elif not lowered.startswith(("diagnose ", "debug ", "investigate ")):
            focus = f"Diagnose the issue behind: {focus}"
        if needs_regression and "regression" not in focus.lower() and "test" not in focus.lower():
            return f"{focus} and confirm the fix with a regression check."
        return f"{focus} and confirm the root cause before changing code."

    if mode == "review":
        focus = stripped
        lowered = focus.lower()
        if lowered.startswith("fix "):
            focus = focus[4:]
        elif lowered.startswith("implement "):
            focus = focus[10:]
        if focus:
            review_goal = f"Review the work scoped in this pack for {focus.lower()}"
        else:
            review_goal = f"Review the work scoped in this pack for {query}"
        if needs_regression:
            return f"{review_goal}, correctness, edge cases, and regression risk before merging."
        return f"{review_goal}, correctness, and merge risk before proceeding."

    if needs_regression and "test" not in stripped.lower() and "regression" not in stripped.lower():
        return f"{stripped} and add a regression check."
    return stripped + "."


def _synthesize_task_goal(query: str, goal_lines: list[str], constraints: list[ScoredLine], mode: str) -> list[str]:
    best_goals = _dedupe_plain_lines(goal_lines, limit=3)
    needs_regression = any("regression" in item.text.lower() or "test" in item.text.lower() for item in constraints)
    if not best_goals:
        fallback = _rewrite_goal_for_mode("", query, mode, needs_regression)
        return [fallback]

    primary = best_goals[0]
    return [_rewrite_goal_for_mode(primary, query, mode, needs_regression)]


def _strip_known_context_label(body: str) -> str:
    known_prefixes = (
        "review scope:",
        "working tree changes:",
        "changed files:",
        "relevant module:",
        "affected module:",
        "start in:",
        "open task:",
        "next step:",
        "next check:",
        "missing check:",
        "current baseline:",
        "known baseline:",
        "current invariant:",
        "likely cause:",
        "implementation detail:",
        "observed symptom:",
        "observed issue:",
        "regression evidence:",
        "evidence:",
        "supporting context:",
        "review risk:",
        "known risk:",
        "guardrail:",
    )
    lowered = body.lower()
    for prefix in known_prefixes:
        if lowered.startswith(prefix):
            return body[len(prefix) :].strip()
    return body


def _context_label(category: str, mode: str) -> str | None:
    labels = {
        "implement": {
            "module": "Start in",
            "decision": "Current baseline",
            "todo": "Next step",
            "cause": "Implementation detail",
            "symptom": "Observed issue",
            "scope": "Changed files",
            "risk": "Guardrail",
            "evidence": "Supporting context",
        },
        "debug": {
            "module": "Relevant module",
            "decision": "Known baseline",
            "todo": "Next check",
            "cause": "Likely cause",
            "symptom": "Observed symptom",
            "scope": "Changed files",
            "risk": "Known risk",
            "evidence": "Evidence",
        },
        "review": {
            "module": "Affected module",
            "decision": "Current invariant",
            "todo": "Missing check",
            "cause": "Implementation detail",
            "symptom": "Regression evidence",
            "scope": "Review scope",
            "risk": "Review risk",
            "evidence": "Evidence",
        },
    }
    return labels.get(mode, {}).get(category)


def _edit_scored_line(item: ScoredLine, mode: str, bucket: str | None = None) -> ScoredLine:
    text = re.sub(r"\s+", " ", item.text.strip())
    if text.startswith("- "):
        body = text[2:].strip()
    else:
        body = text

    bucket = bucket or _semantic_context_category(item)
    label = _context_label(bucket, mode) if item.section == "context" else None
    if label:
        body = _strip_known_context_label(body)

    if bucket == "todo":
        body = body[0].upper() + body[1:]
    elif bucket in {"evidence", "decision", "project", "risk", "cause", "symptom"} and body and body[0].islower():
        body = body[0].upper() + body[1:]

    if bucket in {"evidence", "decision", "project", "todo", "risk", "cause", "symptom", "scope", "module"} and not body.endswith((".", "!", "?")):
        body += "."

    if label:
        body = f"{label}: {body}"

    return ScoredLine(item.section, f"- {body}", item.score, item.origin, item.category)


def _semantic_context_category(item: ScoredLine) -> str:
    lower = item.text.lower().lstrip("- ").strip()

    if lower.startswith(("review scope:", "working tree changes:", "changed files:")):
        return "scope"
    if lower.startswith(("relevant module:", "affected module:", "start in:")):
        return "module"
    if lower.startswith(("open task:", "next step:", "next check:", "missing check:")):
        return "todo"
    if lower.startswith(("review risk:", "known risk:", "guardrail:")):
        return "risk"
    if lower.startswith(("likely cause:", "implementation detail:")):
        return "cause"
    if lower.startswith(("observed symptom:", "observed issue:", "regression evidence:")):
        return "symptom"
    if lower.startswith(("evidence:", "supporting context:")):
        return "evidence"
    if lower.startswith(("current baseline:", "known baseline:", "current invariant:")):
        return "decision"

    if item.category == "git":
        return "scope"
    if item.category == "module":
        return "module"
    if item.category == "todo":
        return "todo"
    if item.category == "log":
        return "evidence"

    if any(token in lower for token in ("still need", "rollout", "compatib", "backwards", "validate during", "risk", "edge case")):
        return "risk"
    if any(token in lower for token in ("reproduces", "repro", "happens", "only on", "redirect loop", "error", "failure", "bug", "crash", "broken")):
        return "symptom"
    if any(token in lower for token in ("because", "before", "after", "due", "reads auth state", "parsing", "cookie parsing", "root cause", "timing", "race", "settles", "moved refresh")):
        return "cause"
    if any(token in lower for token in ("already", "uses ", "is in place", "moved", "migration", "decision", "prefers")):
        return "decision"
    if item.category in {"memory", "project"}:
        return "decision"
    if item.category == "note":
        return "evidence"
    return item.category


def _select_task_context(context_lines: list[ScoredLine], mode: str, limit: int = 14) -> list[ScoredLine]:
    deduped = _dedupe_scored_lines(context_lines)
    categorized: list[tuple[str, ScoredLine]] = [
        (_semantic_context_category(item), item)
        for item in deduped
    ]
    by_category: dict[str, list[ScoredLine]] = {}
    for bucket, item in categorized:
        by_category.setdefault(bucket, []).append(item)

    if mode == "debug":
        order = ["symptom", "cause", "evidence", "module", "scope", "decision", "risk", "todo", "general"]
        caps = {"symptom": 4, "cause": 3, "evidence": 3, "module": 2, "scope": 1, "decision": 2, "risk": 2, "todo": 1, "general": 1}
    elif mode == "review":
        order = ["scope", "module", "risk", "decision", "cause", "symptom", "evidence", "todo", "general"]
        caps = {"scope": 2, "module": 3, "risk": 2, "decision": 2, "cause": 2, "symptom": 2, "evidence": 2, "todo": 1, "general": 1}
    else:
        order = ["module", "decision", "todo", "cause", "symptom", "scope", "risk", "evidence", "general"]
        caps = {"module": 3, "decision": 3, "todo": 2, "cause": 2, "symptom": 2, "scope": 1, "risk": 2, "evidence": 2, "general": 1}

    selected: list[ScoredLine] = []
    selected_keys: set[str] = set()
    for category in order:
        pool = by_category.get(category, [])
        for item in pool[: caps.get(category, 2)]:
            key = _pack_dedupe_key(item.text)
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)

    if len(selected) < limit:
        for category in order:
            for item in by_category.get(category, []):
                key = _pack_dedupe_key(item.text)
                if key in selected_keys:
                    continue
                selected.append(item)
                selected_keys.add(key)
                if len(selected) >= limit:
                    break
            if len(selected) >= limit:
                break

    edited: list[ScoredLine] = []
    for item in selected[:limit]:
        bucket = _semantic_context_category(item)
        edited.append(_edit_scored_line(item, mode, bucket=bucket))
    return edited


def _select_constraints(constraint_lines: list[ScoredLine], mode: str, limit: int = 6) -> list[ScoredLine]:
    chosen = _dedupe_scored_lines(constraint_lines)
    if mode == "debug":
        chosen = sorted(
            chosen,
            key=lambda item: (
                -("test" in item.text.lower() or "regression" in item.text.lower()),
                -item.score,
                item.origin,
            ),
        )
    elif mode == "review":
        chosen = sorted(
            chosen,
            key=lambda item: (
                -("preserve" in item.text.lower() or "keep" in item.text.lower()),
                -item.score,
                item.origin,
            ),
        )
    return [_edit_scored_line(item, mode) for item in chosen[:limit]]


def _fit_task_sections(
    headers: dict[str, str],
    query: str,
    mode: str,
    goal_lines: list[str],
    context_lines: list[ScoredLine],
    constraint_lines: list[ScoredLine],
    source_lines: list[str],
    max_total_tokens: int,
) -> str:
    crumb = _crumb()
    goal = _synthesize_task_goal(query, goal_lines, constraint_lines, mode)
    context = _select_task_context(context_lines, mode=mode, limit=14)
    constraints = _select_constraints(constraint_lines, mode=mode, limit=6)
    sources = _dedupe_plain_lines(source_lines, limit=5)

    if not context:
        context = [ScoredLine("context", "- No directly matching CRUMBs were found; use the query, repo map, and local diff as the primary context.", 0.0, "fallback", "general")]
    if not constraints:
        constraints = [ScoredLine("constraints", "- Preserve backwards compatibility and prefer minimal edits that respect the packed context.", 0.0, "fallback", "constraint")]

    while True:
        sections = {
            "goal": goal,
            "context": [item.text for item in context[:14]],
            "constraints": [item.text for item in constraints[:6]],
        }
        if sources:
            sections["sources"] = sources[:5]
        rendered = crumb.render_crumb(headers, sections)
        if crumb.estimate_tokens(rendered) <= max_total_tokens:
            return rendered

        removable: list[tuple[float, str, int]] = []
        if sources:
            sources.pop()
            continue
        if len(context) > 1:
            removable.append((context[-1].score, "context", len(context) - 1))
        if len(constraints) > 1:
            removable.append((constraints[-1].score, "constraints", len(constraints) - 1))
        if not removable:
            break
        _, target, index = sorted(removable, key=lambda row: (row[0], row[1]))[0]
        if target == "context":
            context.pop(index)
        else:
            constraints.pop(index)

    raise ValueError(f"Could not fit task pack within {max_total_tokens} estimated tokens without dropping required sections.")


def _fit_mem_sections(headers: dict[str, str], lines: list[ScoredLine], max_total_tokens: int) -> str:
    crumb = _crumb()
    chosen = _dedupe_scored_lines(lines)
    if not chosen:
        chosen = [ScoredLine("consolidated", "- No durable facts matched the requested query.", 0.0, "fallback")]

    while True:
        sections = {"consolidated": [item.text for item in chosen[:40]]}
        rendered = crumb.render_crumb(headers, sections)
        if crumb.estimate_tokens(rendered) <= max_total_tokens:
            return rendered
        if len(chosen) <= 1:
            break
        chosen.pop()
    raise ValueError(f"Could not fit mem pack within {max_total_tokens} estimated tokens.")


def _fit_map_sections(
    headers: dict[str, str],
    project_lines: list[ScoredLine],
    module_lines: list[ScoredLine],
    max_total_tokens: int,
) -> str:
    crumb = _crumb()
    project = _dedupe_scored_lines(project_lines)
    modules = _dedupe_scored_lines(module_lines)

    if not project:
        project = [ScoredLine("project", "Project context assembled from the local CRUMB corpus.", 0.0, "fallback")]
    if not modules:
        modules = [ScoredLine("modules", "- No specific modules matched; inspect the repository root directly.", 0.0, "fallback")]

    while True:
        sections = {
            "project": [item.text for item in project[:4]],
            "modules": [item.text for item in modules[:30]],
        }
        rendered = crumb.render_crumb(headers, sections)
        if crumb.estimate_tokens(rendered) <= max_total_tokens:
            return rendered

        removable: list[tuple[float, str, int]] = []
        if len(modules) > 1:
            removable.append((modules[-1].score, "modules", len(modules) - 1))
        if len(project) > 1:
            removable.append((project[-1].score, "project", len(project) - 1))
        if not removable:
            break
        _, target, index = sorted(removable, key=lambda row: (row[0], row[1]))[0]
        if target == "modules":
            modules.pop(index)
        else:
            project.pop(index)

    raise ValueError(f"Could not fit map pack within {max_total_tokens} estimated tokens.")


def _maybe_git_context(search_dir: Path, query_terms: list[str], output_kind: str) -> list[ScoredLine]:
    crumb = _crumb()
    repo_root = crumb._git_repo_root(search_dir)
    if repo_root is None:
        return []

    lines: list[ScoredLine] = []
    diff_names = crumb._git_completed(["diff", "--name-only", "HEAD"], cwd=search_dir)
    if diff_names.returncode == 0 and diff_names.stdout.strip():
        changed_files = [item.strip() for item in diff_names.stdout.splitlines() if item.strip()]
        relevant_files = [
            item
            for item in changed_files
            if _is_relevant_line(item, query_terms)
        ]
        changed_files = relevant_files or []
    else:
        changed_files = []
    if changed_files:
        summary = ", ".join(changed_files[:8])
        if len(changed_files) > 8:
            summary += ", …"
        target_section = "modules" if output_kind == "map" else ("consolidated" if output_kind == "mem" else "context")
        category = "module" if output_kind == "map" else ("memory" if output_kind == "mem" else "git")
        lines.append(ScoredLine(target_section, _bullet(f"Working tree changes: {summary}"), 80.0, "git diff --name-only", category))

    return lines


def _maybe_repo_tree_context(search_dir: Path, query_terms: list[str], output_kind: str) -> list[ScoredLine]:
    crumb = _crumb()
    root_lines = crumb._build_repo_tree(search_dir)
    if not root_lines:
        return []
    matched: list[str] = []
    for line in root_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if output_kind != "map" and stripped.endswith(".crumb"):
            continue
        if any(term in stripped.lower() for term in query_terms):
            matched.append(stripped)
    if not matched and output_kind == "map":
        matched = [line.strip() for line in root_lines[:10] if line.strip()]
    if not matched:
        return []

    target = "modules" if output_kind == "map" else "context"
    category = "module" if output_kind == "map" else "module"
    return [ScoredLine(target, _bullet(item), 45.0, "repo tree", category) for item in matched[:12]]


def _apply_optional_local_compression(rendered: str, args: argparse.Namespace) -> str:
    if not getattr(args, "ollama", False):
        return rendered

    crumb = _crumb()
    ensure_ollama_available(args.ollama_model)
    prompt = (
        "You are compressing a CRUMB context pack.\n"
        f"Return one valid CRUMB v1.1 document only.\n"
        f"Keep kind={args.kind} and preserve the required sections.\n"
        f"Target budget: <= {args.max_total_tokens} estimated tokens.\n"
        "Prefer deleting low-signal repetition over paraphrasing away concrete facts.\n"
        "Keep it human-readable.\n\n"
        "CRUMB to compress:\n"
        f"{rendered}"
    )
    response = generate_text(prompt, model=args.ollama_model)
    candidate = extract_crumb_block(response)
    crumb.parse_crumb(candidate)
    if crumb.estimate_tokens(candidate) > args.max_total_tokens:
        raise LocalAIError(
            f"Local model returned a pack above budget ({crumb.estimate_tokens(candidate)} > {args.max_total_tokens} tokens)."
        )
    return candidate if candidate.endswith("\n") else candidate + "\n"


def _pack_id(search_dir: Path, query: str, kind: str) -> str:
    digest = hashlib.sha1(f"{search_dir.resolve()}::{kind}::{query}".encode("utf-8")).hexdigest()
    return f"crumb-pack-{digest[:12]}"


def _headers_for_pack(args: argparse.Namespace) -> dict[str, str]:
    headers = {
        "v": "1.1",
        "kind": args.kind,
        "title": args.title or f"Context pack: {args.query}",
        "source": "crumb.pack",
        "id": _pack_id(Path(args.dir), args.query, args.kind),
        "url": SPEC_URL,
        "max_total_tokens": str(args.max_total_tokens),
        "tags": f"pack, {args.strategy}, {args.kind}, {args.mode}",
    }
    if args.project:
        headers["project"] = args.project
    headers["x-crumb-pack.mode"] = args.mode
    append_extension(headers, "crumb.pack.v1")
    return headers


def build_pack(args: argparse.Namespace) -> str:
    search_dir = Path(args.dir).resolve()
    if not search_dir.is_dir():
        raise ValueError(f"{search_dir} is not a directory")

    query_terms = _query_terms(args.query)
    ranked_files = _rank_files(search_dir, query_terms, args.strategy, args.project, args.mode)
    if not ranked_files:
        raise ValueError(f"No matching CRUMBs found in {search_dir} for query '{args.query}'.")

    headers = _headers_for_pack(args)

    if args.kind == "task":
        goal_lines, context_lines, constraint_lines, source_lines = _task_candidates(ranked_files, query_terms, args.mode)
        context_lines.extend(_maybe_git_context(search_dir, query_terms, args.kind))
        context_lines.extend(_maybe_repo_tree_context(search_dir, query_terms, args.kind))
        rendered = _fit_task_sections(headers, args.query, args.mode, goal_lines, context_lines, constraint_lines, source_lines, args.max_total_tokens)
    elif args.kind == "mem":
        lines = _mem_candidates(ranked_files, query_terms)
        lines.extend(_maybe_git_context(search_dir, query_terms, args.kind))
        rendered = _fit_mem_sections(headers, lines, args.max_total_tokens)
    else:
        project_lines, module_lines = _map_candidates(ranked_files, query_terms)
        module_lines.extend(_maybe_git_context(search_dir, query_terms, args.kind))
        module_lines.extend(_maybe_repo_tree_context(search_dir, query_terms, args.kind))
        rendered = _fit_map_sections(headers, project_lines, module_lines, args.max_total_tokens)

    return _apply_optional_local_compression(rendered, args)


def run_pack(args: argparse.Namespace) -> None:
    crumb = _crumb()
    rendered = build_pack(args)
    crumb.parse_crumb(rendered)
    output = args.output
    path = Path(output)
    if output != "-":
        path.parent.mkdir(parents=True, exist_ok=True)
    crumb.write_text(output, rendered)
    if output != "-":
        print(f"Packed {args.kind} context → {output} (~{crumb.estimate_tokens(rendered)} tokens)")
