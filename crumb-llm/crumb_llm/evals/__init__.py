"""Quality gates for CrumbLLM output.

The guiding principle: **never silently trust model output.** Every gate
returns a list of human-readable warnings that are attached to the
:class:`AnalysisResult`.
"""

from crumb_llm.evals.hallucination_check import check_hallucinated_paths
from crumb_llm.evals.json_check import check_json
from crumb_llm.evals.quality_gate import run_quality_gates

__all__ = [
    "check_hallucinated_paths",
    "check_json",
    "run_quality_gates",
]
