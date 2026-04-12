"""Rule-based hall classifier for memory observations.

Sorts each observation into one of five halls — facts, events, discoveries,
preferences, advice — based on linguistic cues. Pure regex/keyword heuristics,
no LLM calls, deterministic, fast.

Halls:
    facts        — decisions, locked-in choices, versions, stack selections
    events       — things that happened, timestamps, debugging, deploys
    discoveries  — realizations, findings, learnings, breakthroughs
    preferences  — likes, dislikes, habits, working style
    advice       — recommendations, warnings, tips, best practices

Used internally by `crumb palace add` when no explicit `--hall` is given,
and exposed directly via `crumb classify`.
"""

import re
from typing import Dict, List, Tuple

HALLS = ["facts", "events", "discoveries", "preferences", "advice"]

# (pattern, weight) tuples per hall. Higher weight = stronger signal.
_FACT_PATTERNS = [
    (r"\b(decided|chose|picked|selected)\b", 3),
    (r"\b(locked in|went with|settled on)\b", 3),
    (r"\bwill use\b", 2),
    (r"\bdefault (is|to)\b", 2),
    (r"\bstack is\b", 2),
    (r"\b(using|use) the\b", 1),
    (r"\bv\d+\.\d+\b", 1),
]

_EVENT_PATTERNS = [
    (r"\b\d{4}-\d{2}-\d{2}\b", 3),
    (r"\b(yesterday|today|tomorrow)\b", 2),
    (r"\b(last|this|next) (week|month|year|sprint)\b", 2),
    (r"\b(shipped|deployed|launched|released|merged)\b", 3),
    (r"\b(fixed|debugged|patched|hotfixed)\b", 3),
    (r"\b(broke|crashed|failed|went down|regressed)\b", 3),
    (r"\bsession\b", 1),
    (r"\bstand[- ]?up\b", 1),
]

_DISCOVERY_PATTERNS = [
    (r"\b(realized|discovered|found out)\b", 3),
    (r"\b(learned|figured out|noticed)\b", 3),
    (r"\bturns out\b", 3),
    (r"\bdidn[^a-z]t know\b", 2),
    (r"\b(breakthrough|aha|insight)\b", 3),
    (r"\bit works because\b", 2),
]

_PREFERENCE_PATTERNS = [
    (r"\bprefers?\b", 3),
    (r"\b(likes?|loves?|enjoys?)\b", 2),
    (r"\b(hates?|dislikes?|avoids?)\b", 3),
    (r"\bfavou?rite\b", 3),
    (r"\bwants? to\b", 2),
    (r"\bdon[^a-z]t want\b", 3),
    (r"\balways\b", 1),
    (r"\bnever\b", 1),
    (r"\bstyle is\b", 2),
]

_ADVICE_PATTERNS = [
    (r"\b(should|shouldn[^a-z]t)\b", 2),
    (r"\b(recommend|suggest|propose)\b", 3),
    (r"\b(try|avoid|beware|watch out)\b", 2),
    (r"\bbest to\b", 3),
    (r"\btip:\b", 3),
    (r"\bwarning:\b", 3),
    (r"\bmake sure\b", 2),
    (r"\bdo not\b", 2),
    (r"\bnever (commit|push|force|skip|disable)\b", 3),
]

_PATTERN_TABLE: Dict[str, List[Tuple[str, int]]] = {
    "facts": _FACT_PATTERNS,
    "events": _EVENT_PATTERNS,
    "discoveries": _DISCOVERY_PATTERNS,
    "preferences": _PREFERENCE_PATTERNS,
    "advice": _ADVICE_PATTERNS,
}

# Tie-break priority: more specific halls win over more general ones.
_TIE_PRIORITY = ["discoveries", "advice", "events", "preferences", "facts"]


def score(text: str) -> Dict[str, int]:
    """Score an observation against every hall. Returns a dict of hall→score."""
    lower = text.lower()
    out: Dict[str, int] = {hall: 0 for hall in HALLS}
    for hall, patterns in _PATTERN_TABLE.items():
        total = 0
        for pattern, weight in patterns:
            matches = len(re.findall(pattern, lower))
            total += matches * weight
        out[hall] = total
    return out


def classify(text: str, default: str = "facts") -> str:
    """Return the best-matching hall for an observation.

    If every hall scores zero, returns `default` (facts — the catch-all for
    statements of record with no stronger signal).
    """
    scores = score(text)
    best = max(scores.values())
    if best == 0:
        return default
    for hall in _TIE_PRIORITY:
        if scores[hall] == best:
            return hall
    return default


def classify_batch(lines: List[str]) -> List[Tuple[str, str]]:
    """Classify a batch of non-empty lines. Returns [(hall, line), ...]."""
    return [(classify(line), line.strip()) for line in lines if line.strip()]


def explain(text: str) -> Dict[str, object]:
    """Return a dict with the classification, full scores, and matched patterns.

    Useful for debugging / showing why a line landed in a particular hall.
    """
    lower = text.lower()
    hits: Dict[str, List[str]] = {hall: [] for hall in HALLS}
    for hall, patterns in _PATTERN_TABLE.items():
        for pattern, _ in patterns:
            if re.search(pattern, lower):
                hits[hall].append(pattern)
    return {
        "text": text,
        "hall": classify(text),
        "scores": score(text),
        "matched_patterns": hits,
    }
