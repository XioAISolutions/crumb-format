"""Compatibility shim — `cli.mempalace_bridge` was renamed to
`cli.memory_bridge` in v0.11.0.

The module is the same. The new name reflects that the bridge
framework is generic (it has an adapter registry) — MemPalace is
just the only adapter currently implemented. Code importing
``cli.mempalace_bridge`` continues to work for one release; remove
this shim in v0.12.
"""

from cli.memory_bridge import *  # noqa: F401,F403
from cli.memory_bridge import (  # noqa: F401
    MempalaceAdapter,
    cmd_mempalace,
    adapters,
)
