"""Translate v1.3 [guardrails] lines into AgentAuth ToolPolicy calls.

SPEC v1.3 §21.2 says AgentAuth-aware runtimes SHOULD translate applicable
[guardrails] entries into their policy engine. This module is that bridge.

Example:

    parsed = crumb.parse_crumb(text)
    guardrails = parsed["sections"].get("guardrails", [])
    result = apply_guardrails_to_policy(guardrails, agent_name="my-agent")
    # result["denied"] lists the tools added to the deny list
    # result["required"] lists the tools added to the allow list (require=)
    # result["skipped"] lists lines the translator didn't understand

The translator is conservative: unknown `type=` values or malformed lines
are skipped with a reason, never raised. Same contract as the parser —
[guardrails] is advisory at the CRUMB layer; enforcement is opt-in.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

KV_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^\s]+)")


def parse_guardrail_line(line: str) -> Optional[Dict[str, str]]:
    """Parse one [guardrails] bullet into a key=value dict.

    Returns None for blank lines or non-bullet lines. Prose bullets without
    any key=value pairs return an empty dict.
    """
    stripped = line.strip()
    if not stripped or not stripped.startswith("-"):
        return None
    body = stripped[1:].strip()
    return {m.group(1): m.group(2) for m in KV_RE.finditer(body)}


def translate_guardrails(
    lines: Iterable[str],
) -> Dict[str, List[Dict[str, str]]]:
    """Group guardrail lines by semantic action.

    Returns a dict with keys ``deny``, ``require``, ``approval``, ``scope``,
    ``skipped``. Each value is a list of the parsed key=value dicts.
    A line may appear under multiple keys if it expresses more than one
    constraint. Unparseable lines go to ``skipped`` with a ``_raw`` key so
    callers can surface them.
    """
    buckets: Dict[str, List[Dict[str, str]]] = {
        "deny": [],
        "require": [],
        "approval": [],
        "scope": [],
        "skipped": [],
    }
    for line in lines:
        parsed = parse_guardrail_line(line)
        if parsed is None:
            continue
        if not parsed:
            buckets["skipped"].append({"_raw": line.strip(), "_reason": "no key=value pairs"})
            continue
        gtype = parsed.get("type", "").lower()
        if "deny" in parsed:
            buckets["deny"].append(parsed)
        if "require" in parsed:
            buckets["require"].append(parsed)
        if gtype == "approval":
            buckets["approval"].append(parsed)
        if gtype == "scope":
            buckets["scope"].append(parsed)
        if not (
            "deny" in parsed
            or "require" in parsed
            or gtype in {"approval", "scope"}
        ):
            buckets["skipped"].append(
                {"_raw": line.strip(), "_reason": f"no actionable key (type={gtype!r})"}
            )
    return buckets


def apply_guardrails_to_policy(
    lines: Iterable[str],
    agent_name: str,
    policy=None,
) -> Dict[str, object]:
    """Translate [guardrails] lines into AgentAuth ToolPolicy entries.

    ``policy`` is an optional ``agentauth.ToolPolicy`` instance. If omitted,
    the function computes what would be set without touching any store —
    useful for `crumb resolve --guardrails` dry-runs or tests.

    Returns a summary dict:

    .. code-block:: python

        {
            "agent_name": "my-agent",
            "tools_denied": ["shell-exec", "filesystem.write(/etc/**)"],
            "tools_required": ["tests"],
            "approvals": [{"action": "merge", "who": "human", ...}],
            "scope": [{"max": "files=5", ...}],
            "skipped": [{"_raw": "...", "_reason": "..."}],
            "applied": bool,
        }
    """
    buckets = translate_guardrails(lines)

    tools_denied: List[str] = []
    for entry in buckets["deny"]:
        value = entry.get("deny")
        if value:
            tools_denied.append(value)

    tools_required: List[str] = []
    for entry in buckets["require"]:
        value = entry.get("require")
        if value:
            tools_required.append(value)

    summary: Dict[str, object] = {
        "agent_name": agent_name,
        "tools_denied": tools_denied,
        "tools_required": tools_required,
        "approvals": buckets["approval"],
        "scope": buckets["scope"],
        "skipped": buckets["skipped"],
        "applied": False,
    }

    if policy is not None and (tools_denied or tools_required):
        existing = getattr(policy, "store", None)
        tools_allowed = tools_required if tools_required else None
        policy.set_policy(
            agent_name=agent_name,
            tools_allowed=tools_allowed,
            tools_denied=tools_denied or None,
        )
        summary["applied"] = True

    return summary
