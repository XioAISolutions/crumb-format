"""Agent integrity checks for CRUMB-powered agents.

This module cannibalizes the useful engineering pattern from experimental
LLM/signal-feedback repos without adopting their speculative claims:
small deterministic canaries, confidence scoring, action-risk checks, and
rolling drift detection.

It is intentionally dependency-free so it can run inside local desktop
agent loops before a model writes memory, edits files, calls tools, or
hands work to another AI.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Deque, Dict, Iterable, List, Literal, Optional

Recommendation = Literal["proceed", "retry", "ask_user", "block"]
Severity = Literal["info", "warning", "error", "critical"]


@dataclass
class CanaryResult:
    """Result from one deterministic integrity check."""

    name: str
    passed: bool
    score: float
    severity: Severity = "warning"
    message: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrityResult:
    """Combined result from all checks."""

    passed: bool
    score: float
    failures: List[str]
    recommendation: Recommendation
    checks: List[CanaryResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "failures": self.failures,
            "recommendation": self.recommendation,
            "checks": [asdict(check) for check in self.checks],
            "metadata": self.metadata,
        }


_SECRET_PATTERNS = [
    ("openai_api_key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{20,}", re.I)),
    ("private_key", re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----")),
]

_DESTRUCTIVE_PATTERNS = [
    re.compile(r"\brm\s+-rf\b"),
    re.compile(r"\bdelete\b.*\b(production|prod|database|db|all|everything)\b", re.I),
    re.compile(r"\bdrop\s+table\b", re.I),
    re.compile(r"\btruncate\s+table\b", re.I),
    re.compile(r"\bchmod\s+777\b", re.I),
    re.compile(r"\bforce\s+push\b|\bgit\s+push\s+--force\b", re.I),
]

_HIGH_RISK_FILE_PATTERNS = [
    re.compile(r"(^|/|\\)\.env(\.|$|/|\\)?", re.I),
    re.compile(r"(^|/|\\)(package-lock|pnpm-lock|yarn.lock|poetry.lock)$", re.I),
    re.compile(r"(^|/|\\)(settings|config|secrets|credentials)\.(json|yml|yaml|toml|env)$", re.I),
    re.compile(r"(^|/|\\)(migrations?|database|db)(/|\\)", re.I),
    re.compile(r"(^|/|\\)\.github(/|\\)workflows(/|\\)", re.I),
]

_NEGATING_WORDS = ("not", "don't", "do not", "never", "avoid", "without")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _contains_any(text: str, words: Iterable[str]) -> bool:
    haystack = text.lower()
    return any(word in haystack for word in words)


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def check_output_presence(input_text: str, proposed_action: str, model_output: str) -> CanaryResult:
    """Basic presence and usefulness check."""
    output = model_output.strip()
    if not output:
        return CanaryResult(
            name="output_presence",
            passed=False,
            score=0.0,
            severity="error",
            message="Model output is empty.",
        )
    if len(output) < 12:
        return CanaryResult(
            name="output_presence",
            passed=False,
            score=0.3,
            severity="warning",
            message="Model output is too short to trust for an agent action.",
            evidence={"length": len(output)},
        )
    return CanaryResult(
        name="output_presence",
        passed=True,
        score=1.0,
        severity="info",
        message="Model output has enough surface signal to inspect.",
        evidence={"length": len(output)},
    )


def check_physical_logic_canary(input_text: str, proposed_action: str, model_output: str) -> CanaryResult:
    """Catch the car-wash style quantization failure.

    This does not pretend to prove reasoning. It catches a cheap and useful
    class of physical-location mistakes before an agent proceeds.
    """
    combined = f"{input_text}\n{proposed_action}\n{model_output}".lower()

    car_wash_context = "car wash" in combined or "wash the car" in combined
    if not car_wash_context:
        return CanaryResult(
            name="physical_logic_car_wash",
            passed=True,
            score=1.0,
            severity="info",
            message="No car-wash physical-location canary triggered.",
        )

    says_walk = re.search(r"\bwalk(?:ing)?\b", model_output, re.I) is not None
    says_drive = re.search(r"\bdrive|bring|take\s+the\s+car|take\s+it\b", model_output, re.I) is not None

    passed = says_drive and not says_walk
    return CanaryResult(
        name="physical_logic_car_wash",
        passed=passed,
        score=1.0 if passed else 0.0,
        severity="error" if not passed else "info",
        message=(
            "Car-wash canary passed: output keeps the car physically involved."
            if passed
            else "Car-wash canary failed: output may ignore that the car must be at the car wash."
        ),
        evidence={"says_drive": says_drive, "says_walk": says_walk},
    )


def check_instruction_following(input_text: str, proposed_action: str, model_output: str) -> CanaryResult:
    """Look for obvious violation of user constraints."""
    source = input_text.lower()
    output = model_output.lower()
    action = proposed_action.lower()

    # Simple but valuable: if user says not to do a named thing, don't do it.
    violations: List[str] = []
    for pattern in (
        r"do not ([^.\n]+)",
        r"don't ([^.\n]+)",
        r"never ([^.\n]+)",
        r"avoid ([^.\n]+)",
    ):
        for match in re.finditer(pattern, source):
            forbidden = match.group(1).strip().strip(" .")
            if not forbidden or len(forbidden) < 4:
                continue
            # Use the first few meaningful words to avoid overfitting full clauses.
            key_terms = [w for w in re.findall(r"[a-z0-9_/-]+", forbidden) if len(w) > 3][:4]
            if key_terms and all(term in output or term in action for term in key_terms[:2]):
                violations.append(forbidden[:120])

    if violations:
        return CanaryResult(
            name="instruction_following",
            passed=False,
            score=0.25,
            severity="error",
            message="Output/action appears to violate an explicit user constraint.",
            evidence={"violations": violations},
        )

    return CanaryResult(
        name="instruction_following",
        passed=True,
        score=1.0,
        severity="info",
        message="No obvious explicit-constraint violation found.",
    )


def check_contradiction_detection(input_text: str, proposed_action: str, model_output: str) -> CanaryResult:
    """Detect simple self-contradictions that should trigger retry."""
    text = f"{proposed_action}\n{model_output}".lower()
    contradictions = []

    pairs = [
        ("safe", "unsafe"),
        ("pass", "fail"),
        ("passed", "failed"),
        ("proceed", "block"),
        ("allowed", "denied"),
        ("yes", "no"),
    ]
    for left, right in pairs:
        if re.search(rf"\b{left}\b", text) and re.search(rf"\b{right}\b", text):
            contradictions.append(f"{left}/{right}")

    # Stronger signal: recommends proceed while also describing a critical issue.
    if re.search(r"\b(proceed|continue|ship|merge)\b", text) and re.search(
        r"\b(critical|dangerous|unsafe|data loss|secret leaked|credential)\b", text
    ):
        contradictions.append("proceed_with_critical_risk")

    passed = not contradictions
    return CanaryResult(
        name="contradiction_detection",
        passed=passed,
        score=1.0 if passed else 0.4,
        severity="warning" if not passed else "info",
        message=(
            "No obvious self-contradiction found."
            if passed
            else "Potential self-contradiction found; retry or ask the user."
        ),
        evidence={"contradictions": contradictions},
    )


def check_tool_use_safety(input_text: str, proposed_action: str, model_output: str) -> CanaryResult:
    """Block or escalate obviously destructive actions."""
    text = f"{proposed_action}\n{model_output}"
    hits = [pattern.pattern for pattern in _DESTRUCTIVE_PATTERNS if pattern.search(text)]

    if hits:
        return CanaryResult(
            name="tool_use_safety",
            passed=False,
            score=0.0,
            severity="critical",
            message="Proposed action contains destructive or high-risk tool use.",
            evidence={"patterns": hits},
        )

    if _contains_any(text, ("delete", "overwrite", "deploy", "migration", "production", "secret", "token")):
        return CanaryResult(
            name="tool_use_safety",
            passed=True,
            score=0.7,
            severity="warning",
            message="Action includes elevated-risk terms; review before execution.",
        )

    return CanaryResult(
        name="tool_use_safety",
        passed=True,
        score=1.0,
        severity="info",
        message="No destructive tool-use pattern found.",
    )


def check_memory_write_validation(input_text: str, proposed_action: str, model_output: str) -> CanaryResult:
    """Prevent writing secrets or low-signal garbage into CRUMB memory."""
    text = f"{proposed_action}\n{model_output}"
    secret_hits = [label for label, pattern in _SECRET_PATTERNS if pattern.search(text)]

    is_memory_action = _contains_any(proposed_action, ("memory", "remember", "crumb", "palace", "consolidated"))
    too_large = len(text) > 20_000

    if secret_hits:
        return CanaryResult(
            name="memory_write_validation",
            passed=False,
            score=0.0,
            severity="critical",
            message="Potential secret detected; do not write this to memory.",
            evidence={"secret_types": secret_hits},
        )

    if is_memory_action and too_large:
        return CanaryResult(
            name="memory_write_validation",
            passed=False,
            score=0.35,
            severity="warning",
            message="Memory write is too large; summarize before storing.",
            evidence={"chars": len(text)},
        )

    return CanaryResult(
        name="memory_write_validation",
        passed=True,
        score=1.0,
        severity="info",
        message="No obvious memory-write issue found.",
    )


def check_file_edit_risk(input_text: str, proposed_action: str, model_output: str) -> CanaryResult:
    """Score file-edit risk without blocking normal implementation work."""
    text = f"{proposed_action}\n{model_output}"
    paths = re.findall(r"(?:[\w.-]+/)+[\w.-]+|\.env(?:\.[\w.-]+)?|[\w.-]+\.(?:py|ts|tsx|js|json|yml|yaml|toml|env)", text)
    risky_paths = []
    for path in paths:
        if any(pattern.search(path) for pattern in _HIGH_RISK_FILE_PATTERNS):
            risky_paths.append(path)

    if risky_paths:
        return CanaryResult(
            name="file_edit_risk",
            passed=True,
            score=0.65,
            severity="warning",
            message="Proposed action touches high-risk files; review diff before applying.",
            evidence={"paths": sorted(set(risky_paths))},
        )

    return CanaryResult(
        name="file_edit_risk",
        passed=True,
        score=1.0,
        severity="info",
        message="No high-risk file edits detected.",
        evidence={"path_count": len(paths)},
    )


DEFAULT_CHECKS = [
    check_output_presence,
    check_physical_logic_canary,
    check_instruction_following,
    check_contradiction_detection,
    check_tool_use_safety,
    check_memory_write_validation,
    check_file_edit_risk,
]


def _recommendation(checks: List[CanaryResult], score: float) -> Recommendation:
    if any(c.severity == "critical" and not c.passed for c in checks):
        return "block"
    if any(c.severity == "error" and not c.passed for c in checks):
        return "ask_user"
    if score < 0.65 or any(c.severity == "warning" and not c.passed for c in checks):
        return "retry"
    if any(c.severity == "warning" and c.score < 0.75 for c in checks):
        return "ask_user"
    return "proceed"


def run_agent_integrity_check(
    input_text: Any,
    proposed_action: Any = "",
    model_output: Any = "",
    *,
    checks: Optional[Iterable[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> IntegrityResult:
    """Run deterministic integrity checks around an agent decision.

    Args:
        input_text: User request, CRUMB goal/context, or system task.
        proposed_action: Tool call, file edit summary, memory write, or plan.
        model_output: The model response/output being evaluated.
        checks: Optional custom check callables. Defaults to built-in checks.
        metadata: Optional caller metadata, such as repo, branch, model, run id.

    Returns:
        IntegrityResult with the requested stable shape:
        {passed, score, failures, recommendation}
    """
    normalized_input = _normalize_text(input_text)
    normalized_action = _normalize_text(proposed_action)
    normalized_output = _normalize_text(model_output)

    active_checks = list(checks) if checks is not None else DEFAULT_CHECKS
    results: List[CanaryResult] = []
    for check in active_checks:
        try:
            results.append(check(normalized_input, normalized_action, normalized_output))
        except Exception as exc:  # defensive: checks should not crash the agent loop
            results.append(
                CanaryResult(
                    name=getattr(check, "__name__", "unknown_check"),
                    passed=False,
                    score=0.0,
                    severity="error",
                    message=f"Integrity check crashed: {exc}",
                )
            )

    score = _clip(mean([r.score for r in results]) if results else 0.0)
    failures = [r.message for r in results if not r.passed]
    recommendation = _recommendation(results, score)
    passed = recommendation == "proceed"

    return IntegrityResult(
        passed=passed,
        score=score,
        failures=failures,
        recommendation=recommendation,
        checks=results,
        metadata=metadata or {},
    )


class DriftMonitor:
    """Rolling score monitor for long-running agent sessions."""

    def __init__(self, window: int = 20, warn_threshold: float = 0.72, block_threshold: float = 0.45):
        self.window = max(3, int(window))
        self.warn_threshold = warn_threshold
        self.block_threshold = block_threshold
        self._scores: Deque[float] = deque(maxlen=self.window)
        self._recommendations: Deque[str] = deque(maxlen=self.window)

    def observe(self, result: IntegrityResult) -> Dict[str, Any]:
        self._scores.append(result.score)
        self._recommendations.append(result.recommendation)
        rolling = mean(self._scores) if self._scores else 0.0
        block_count = sum(1 for r in self._recommendations if r == "block")
        retry_count = sum(1 for r in self._recommendations if r == "retry")

        if rolling <= self.block_threshold or block_count >= 2:
            status = "block"
        elif rolling <= self.warn_threshold or retry_count >= max(3, self.window // 3):
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "rolling_score": round(rolling, 4),
            "observations": len(self._scores),
            "window": self.window,
            "retry_count": retry_count,
            "block_count": block_count,
        }


def _read_arg_or_stdin(value: Optional[str]) -> str:
    if value == "-":
        return sys.stdin.read()
    return value or ""


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run CRUMB agent integrity checks.")
    parser.add_argument("--input", default="", help="User input/task text, or '-' for stdin.")
    parser.add_argument("--action", default="", help="Proposed action/tool/file edit summary.")
    parser.add_argument("--output", default="", help="Model output to evaluate.")
    parser.add_argument("--metadata", default="{}", help="Optional JSON metadata.")
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args(argv)

    try:
        metadata = json.loads(args.metadata) if args.metadata else {}
    except json.JSONDecodeError:
        metadata = {"metadata_parse_error": args.metadata}

    result = run_agent_integrity_check(
        _read_arg_or_stdin(args.input),
        proposed_action=args.action,
        model_output=args.output,
        metadata=metadata,
    )

    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Agent integrity: {result.recommendation.upper()} score={result.score:.2f}")
        for failure in result.failures:
            print(f"- {failure}")
        if not result.failures:
            print("- All checks passed.")

    sys.exit(0 if result.recommendation in ("proceed", "ask_user") else 1)


if __name__ == "__main__":
    main()
