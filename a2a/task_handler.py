"""A2A Task Handler — routes incoming A2A task requests to AgentAuth functions.

Each task arrives as a JSON object:

    {"id": "task-123", "skill_id": "passport.register", "input": { ... }}

The handler validates the envelope, dispatches to the correct AgentAuth
subsystem, and returns an A2A-compliant task response.
"""

import sys
import traceback
from pathlib import Path

# Ensure the project root is on sys.path so imports work when running the
# server directly (e.g.  ``python -m a2a.server``).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cli.crumb import parse_crumb, render_crumb  # noqa: E402
from agentauth.passport import AgentPassport      # noqa: E402
from agentauth.policy import ToolPolicy            # noqa: E402
from agentauth.audit import AuditLogger            # noqa: E402


# ---------------------------------------------------------------------------
# Shared singletons (created once per process)
# ---------------------------------------------------------------------------

_passport = AgentPassport()
_policy = ToolPolicy()
_audit = AuditLogger()


# ---------------------------------------------------------------------------
# Individual skill handlers
# ---------------------------------------------------------------------------

def _handle_crumb_parse(inp: dict) -> dict:
    text = inp.get("text", "")
    if not text:
        return {"error": "missing required field 'text'"}
    try:
        return parse_crumb(text)
    except ValueError as exc:
        return {"error": f"parse error: {exc}"}


def _handle_crumb_render(inp: dict) -> dict:
    headers = inp.get("headers")
    sections = inp.get("sections")
    if headers is None or sections is None:
        return {"error": "missing required fields 'headers' and/or 'sections'"}
    try:
        rendered = render_crumb(headers, sections)
        return {"text": rendered}
    except Exception as exc:
        return {"error": f"render error: {exc}"}


def _handle_crumb_validate(inp: dict) -> dict:
    text = inp.get("text", "")
    if not text:
        return {"error": "missing required field 'text'"}
    try:
        parse_crumb(text)
        return {"valid": True, "errors": []}
    except ValueError as exc:
        return {"valid": False, "errors": [str(exc)]}


def _handle_passport_register(inp: dict) -> dict:
    name = inp.get("name")
    if not name:
        return {"error": "missing required field 'name'"}
    return _passport.register(
        name=name,
        framework=inp.get("framework", "unknown"),
        owner=inp.get("owner", ""),
        tools_allowed=inp.get("tools_allowed"),
        tools_denied=inp.get("tools_denied"),
        data_classes=inp.get("data_classes"),
        ttl_days=inp.get("ttl_days", 90),
    )


def _handle_passport_verify(inp: dict) -> dict:
    agent_id = inp.get("agent_id")
    if not agent_id:
        return {"error": "missing required field 'agent_id'"}
    return _passport.verify(agent_id)


def _handle_passport_revoke(inp: dict) -> dict:
    agent_id = inp.get("agent_id")
    if not agent_id:
        return {"error": "missing required field 'agent_id'"}
    success = _passport.revoke(agent_id)
    return {"revoked": success, "agent_id": agent_id}


def _handle_policy_check(inp: dict) -> dict:
    agent_id = inp.get("agent_id")
    tool = inp.get("tool")
    if not agent_id or not tool:
        return {"error": "missing required fields 'agent_id' and/or 'tool'"}
    return _policy.check(
        agent_id_or_name=agent_id,
        tool=tool,
        action=inp.get("action"),
        data_class=inp.get("data_class"),
    )


def _handle_audit_log(inp: dict) -> dict:
    agent_id = inp.get("agent_id")
    tool = inp.get("tool")
    detail = inp.get("detail", "")
    if not agent_id or not tool:
        return {"error": "missing required fields 'agent_id' and/or 'tool'"}

    goal = inp.get("goal", "a2a-task")
    allowed = inp.get("allowed", True)
    reason = inp.get("reason", "")

    session_id = _audit.start_session(agent_id, goal)
    _audit.log_action(session_id, tool, detail, allowed, reason)
    crumb_text = _audit.end_session(session_id)
    return {"session_id": session_id, "audit_crumb": crumb_text}


def _handle_scan_shadow(inp: dict) -> dict:
    scan_path = inp.get("path", ".")
    target = Path(scan_path).resolve()
    if not target.is_dir():
        return {"error": f"path is not a directory: {scan_path}"}

    # Import the internal scan helpers from the CLI module.
    from cli.crumb import (
        _scan_config_files,
        _scan_env_files,
        _scan_dependencies,
        _scan_mcp_configs,
        _scan_code_imports,
        RISK_ORDER,
    )

    findings = []
    findings.extend(_scan_config_files(target))
    findings.extend(_scan_env_files(target))
    findings.extend(_scan_dependencies(target))
    findings.extend(_scan_mcp_configs(target))
    findings.extend(_scan_code_imports(target))

    min_risk = RISK_ORDER.get(inp.get("min_risk", "low"), 0)
    findings = [f for f in findings if RISK_ORDER.get(f["risk_level"], 0) >= min_risk]
    findings.sort(key=lambda f: (-RISK_ORDER.get(f["risk_level"], 0), f["path"]))

    # Convert Path objects to strings for JSON serialisation.
    for f in findings:
        if isinstance(f.get("path"), Path):
            f["path"] = str(f["path"])

    return {"scan_root": str(target), "findings": findings}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_SKILL_DISPATCH = {
    "crumb.parse": _handle_crumb_parse,
    "crumb.render": _handle_crumb_render,
    "crumb.validate": _handle_crumb_validate,
    "passport.register": _handle_passport_register,
    "passport.verify": _handle_passport_verify,
    "passport.revoke": _handle_passport_revoke,
    "policy.check": _handle_policy_check,
    "audit.log": _handle_audit_log,
    "scan.shadow": _handle_scan_shadow,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def handle_task(task: dict) -> dict:
    """Process an A2A task request and return a task response.

    Parameters
    ----------
    task : dict
        A2A task envelope with at least ``id``, ``skill_id``, and ``input``.

    Returns
    -------
    dict
        A2A task response with ``id``, ``status``, and ``result`` or ``error``.
    """
    task_id = task.get("id", "unknown")
    skill_id = task.get("skill_id")
    inp = task.get("input", {})

    if not skill_id:
        return {
            "id": task_id,
            "status": "failed",
            "error": {"message": "missing required field 'skill_id'"},
        }

    handler = _SKILL_DISPATCH.get(skill_id)
    if handler is None:
        return {
            "id": task_id,
            "status": "failed",
            "error": {"message": f"unknown skill: {skill_id}"},
        }

    try:
        result = handler(inp)
        # If the handler itself returned an error dict, surface it.
        if isinstance(result, dict) and "error" in result:
            return {
                "id": task_id,
                "status": "failed",
                "error": {"message": result["error"]},
            }
        return {
            "id": task_id,
            "status": "completed",
            "result": result,
        }
    except Exception:
        return {
            "id": task_id,
            "status": "failed",
            "error": {"message": traceback.format_exc()},
        }
