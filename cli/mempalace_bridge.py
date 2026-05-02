"""Compatibility shim — `cli.mempalace_bridge` was renamed to
`cli.memory_bridge` in v0.11.0.

The module is the same. The new name reflects that the bridge
framework is generic (it has an adapter registry) — MemPalace is
just the only adapter currently implemented. Code importing
``cli.mempalace_bridge`` continues to work for one release; remove
this shim in v0.12.
"""

# Re-export every public name. We use a wildcard import so the shim
# doesn't go stale when memory_bridge gains or loses helpers; explicit
# import lists in a previous draft of this shim referenced names that
# don't exist on the new module (cmd_mempalace, adapters) and broke
# `import cli.mempalace_bridge` outright.
from cli.memory_bridge import *  # noqa: F401,F403
