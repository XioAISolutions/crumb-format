"""protect — decorator that enforces AgentAuth policy before tool execution."""

import functools

from .credentials import CredentialBroker
from .store import PassportStore


def protect(agent_id: str, tool: str = None, store: PassportStore = None):
    """Decorator that checks policy + issues a credential before running the function.

    Usage:
        @protect(agent_id="ap_abc12345", tool="database.query")
        def query_database(sql):
            ...
    """

    def decorator(fn):
        resolved_tool = tool or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            broker = CredentialBroker(store=store)
            # issue() internally verifies passport + checks policy
            cred = broker.issue(agent_id, resolved_tool)
            # Attach credential to kwargs so the function can inspect it
            kwargs["_agentauth_credential"] = cred
            return fn(*args, **kwargs)

        return wrapper

    return decorator
