"""CrumbEngine — the analysis core.

Responsibilities:
  - turn a CrumbDoc / CrumbPack into a prompt (using the prompt templates),
  - call the configured provider,
  - run quality gates over the result,
  - return an :class:`AnalysisResult` whose warnings are always populated.

The engine knows nothing about argparse or the filesystem layout of packs;
that is the CLI's and the crumb/ layer's job.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from crumb_llm.evals.quality_gate import run_quality_gates
from crumb_llm.models import AnalysisResult, CrumbDoc, CrumbPack, ProviderConfig
from crumb_llm.providers import BaseProvider, ProviderError, get_provider

# Map task name -> prompt template file.
_PROMPTS = {
    "analyze": "analyze_session.md",
    "summarize": "summarize_memory.md",
    "risks": "classify_risks.md",
    "next": "next_actions.md",
    "improve": "improve_handoff.md",
}

# Tasks that may legitimately ask for JSON output.
_JSON_CAPABLE = {"risks"}


@lru_cache(maxsize=None)
def _load_prompt(name: str) -> str:
    return (
        resources.files("crumb_llm.prompts")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


class CrumbEngine:
    def __init__(
        self,
        provider: BaseProvider | None = None,
        config: ProviderConfig | None = None,
    ):
        self.provider = provider or get_provider(config)

    # -- prompt assembly ---------------------------------------------------

    def _render_prompt(self, task: str, context: str, as_json: bool) -> str:
        template = _load_prompt(_PROMPTS[task])
        prompt = template.replace("{context}", context)
        if as_json and task in _JSON_CAPABLE:
            prompt += "\n\nRespond with ONLY valid JSON, no prose, no code fences."
        return prompt

    # -- the single run path -----------------------------------------------

    def _run(
        self,
        task: str,
        context: str,
        source_paths: set[str],
        as_json: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> AnalysisResult:
        prompt = self._render_prompt(task, context, as_json)
        warnings: list[str] = []
        text = ""
        try:
            text = self.provider.generate(
                prompt, max_tokens=max_tokens, temperature=temperature
            )
        except ProviderError as exc:
            warnings.append(f"provider error: {exc}")

        expect_json = as_json and task in _JSON_CAPABLE
        parsed, gate_warnings = run_quality_gates(
            text, source_paths=source_paths, expect_json=expect_json
        )
        warnings.extend(gate_warnings)

        return AnalysisResult(
            task=task,
            text=text,
            data=parsed if isinstance(parsed, dict) else None,
            warnings=warnings,
            provider=getattr(self.provider, "name", "unknown"),
            model=getattr(self.provider, "model", ""),
            source_paths=sorted(source_paths),
        )

    # -- public tasks ------------------------------------------------------

    def analyze(self, doc: CrumbDoc, **kw) -> AnalysisResult:
        return self._run("analyze", doc.as_text(), doc.file_paths(), **kw)

    def analyze_pack(self, pack: CrumbPack, **kw) -> AnalysisResult:
        return self._run("analyze", pack.as_text(), pack.file_paths(), **kw)

    def summarize(self, source: CrumbDoc | CrumbPack, **kw) -> AnalysisResult:
        return self._run("summarize", source.as_text(), source.file_paths(), **kw)

    def risks(
        self, source: CrumbDoc | CrumbPack, as_json: bool = False, **kw
    ) -> AnalysisResult:
        return self._run(
            "risks", source.as_text(), source.file_paths(), as_json=as_json, **kw
        )

    def next_actions(self, source: CrumbDoc | CrumbPack, **kw) -> AnalysisResult:
        return self._run("next", source.as_text(), source.file_paths(), **kw)

    def improve_handoff(self, doc: CrumbDoc, **kw) -> AnalysisResult:
        return self._run("improve", doc.as_text(), doc.file_paths(), **kw)
