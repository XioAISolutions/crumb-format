"""CrumbLLM — AI analysis over CRUMB files and CRUMB packs.

CrumbLLM reads CRUMB files (the project-memory handoff format), understands
project memory, and produces summaries, risks, next actions, PR notes, and
improved handoffs.

It is *not* the CRUMB spec (that lives in ``crumb-format``) and it is *not*
IBM Bob session capture (that lives in ``Crumb-Bob``). CrumbLLM only reads
CRUMB artifacts and reasons over them.
"""

from crumb_llm.models import (
    AnalysisResult,
    CrumbDoc,
    CrumbPack,
    ProviderConfig,
)

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
    "CrumbDoc",
    "CrumbPack",
    "ProviderConfig",
    "__version__",
]
