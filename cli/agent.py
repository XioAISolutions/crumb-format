from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from crumb import estimate_tokens, render_crumb
except ImportError:  # pragma: no cover
    from .crumb import estimate_tokens, render_crumb


DEFAULT_TEMPLATE: dict[str, Any] = {
    "title": "Implement auth refresh fix",
    "goal": "Fix the auth redirect loop that happens after a hard refresh.",
    "context": [
        "App uses JWT cookie auth.",
        "Redirect loop only happens on a full page refresh.",
        "Middleware reads auth state before cookie parsing stabilizes.",
    ],
    "constraints": [
        "Do not change the login UI.",
        "Preserve existing cookie names.",
        "Add a regression check before merging.",
    ],
    "agent": {
        "name": "crumb-agent",
        "model": "local",
        "mode": "implement",
        "max_total_tokens": 1800,
        "telemetry_opt_in": False,
    },
    "backend": {
        "kind": "echo",
        "command": [],
        "cwd": ".",
        "env": {},
    },
    "evaluation": {
        "required_terms": ["auth", "cookie", "test"],
        "banned_terms": ["ignore safety", "disable guardrails"],
        "min_chars": 80,
        "max_chars": 5000,
    },
    "metadata": {
        "project": "web-app",
        "source": "crumb.agent",
        "tags": ["agent", "experiment"],
    },
    "seed": 7,
}


class AgentError(RuntimeError):
    pass


@dataclass(slots=True)
class BackendConfig:
    kind: str = "echo"
    command: List[str] = field(default_factory=list)
    cwd: str = "."
    env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackendConfig":
        return cls(
            kind=str(data.get("kind", "echo")),
            command=[str(item) for item in data.get("command", [])],
            cwd=str(data.get("cwd", ".")),
            env={str(k): str(v) for k, v in data.get("env", {}).items()},
        )


@dataclass(slots=True)
class AgentConfig:
    name: str = "crumb-agent"
    model: str = "local"
    mode: str = "implement"
    max_total_tokens: int = 1800
    telemetry_opt_in: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        return cls(
            name=str(data.get("name", "crumb-agent")),
            model=str(data.get("model", "local")),
            mode=str(data.get("mode", "implement")),
            max_total_tokens=int(data.get("max_total_tokens", 1800)),
            telemetry_opt_in=bool(data.get("telemetry_opt_in", False)),
        )


@dataclass(slots=True)
class EvaluationConfig:
    required_terms: List[str] = field(default_factory=list)
    banned_terms: List[str] = field(default_factory=list)
    min_chars: int = 0
    max_chars: int = 100_000

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        return cls(
            required_terms=[str(item) for item in data.get("required_terms", [])],
            banned_terms=[str(item) for item in data.get("banned_terms", [])],
            min_chars=int(data.get("min_chars", 0)),
            max_chars=int(data.get("max_chars", 100_000)),
        )


@dataclass(slots=True)
class ExperimentConfig:
    title: str
    goal: str
    context: List[str]
    constraints: List[str]
    agent: AgentConfig
    backend: BackendConfig
    evaluation: EvaluationConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            title=str(data.get("title", "Untitled agent run")),
            goal=str(data.get("goal", "")),
            context=[str(item) for item in data.get("context", [])],
            constraints=[str(item) for item in data.get("constraints", [])],
            agent=AgentConfig.from_dict(data.get("agent", {})),
            backend=BackendConfig.from_dict(data.get("backend", {})),
            evaluation=EvaluationConfig.from_dict(data.get("evaluation", {})),
            metadata=dict(data.get("metadata", {})),
            seed=int(data.get("seed", 0)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


class EchoBackend:
    def generate(
        self,
        prompt: str,
        *,
        model: str,
        mode: str,
        max_total_tokens: int,
        run_dir: Path,
    ) -> str:
        summary = [
            f"Agent mode: {mode}",
            f"Model: {model}",
            f"Budget: {max_total_tokens}",
            "",
            "Suggested implementation plan:",
            "1. Reproduce the issue from the task crumb.",
            "2. Isolate the auth and cookie handling path.",
            "3. Add a targeted regression test before changing the behavior.",
            "4. Ship the smallest safe fix and re-run the check.",
            "",
            f"Prompt size estimate: ~{estimate_tokens(prompt)} tokens.",
        ]
        return "\n".join(summary).strip() + "\n"


class CommandBackend:
    def __init__(self, config: BackendConfig) -> None:
        if not config.command:
            raise AgentError("command backend requires backend.command in the config")
        self.config = config

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        mode: str,
        max_total_tokens: int,
        run_dir: Path,
    ) -> str:
        prompt_file = run_dir / "prompt.txt"
        output_file = run_dir / "response.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        replacements = {
            "prompt_file": str(prompt_file),
            "output_file": str(output_file),
            "run_dir": str(run_dir),
            "model": model,
            "mode": mode,
        }
        command = [replace_tokens(part, replacements) for part in self.config.command]
        env = os.environ.copy()
        env.update(self.config.env)
        env.setdefault("CRUMB_AGENT_MODEL", model)
        env.setdefault("CRUMB_AGENT_MODE", mode)
        env.setdefault("CRUMB_AGENT_MAX_TOTAL_TOKENS", str(max_total_tokens))

        completed = subprocess.run(
            command,
            cwd=self.config.cwd,
            env=env,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AgentError(
                "command backend failed"
                f"\ncommand: {shlex.join(command)}"
                f"\nstdout:\n{completed.stdout.strip()}"
                f"\nstderr:\n{completed.stderr.strip()}"
            )

        if output_file.exists():
            return output_file.read_text(encoding="utf-8")
        if completed.stdout.strip():
            return completed.stdout
        raise AgentError("command backend produced no output file and no stdout")


class TelemetryStore:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.path = Path.home() / ".crumb" / "agent_telemetry.jsonl"

    def append(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def replace_tokens(text: str, replacements: Dict[str, str]) -> str:
    result = text
    for key, value in replacements.items():
        result = result.replace("{" + key + "}", value)
    return result


def slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "run"


def stable_run_id(experiment: ExperimentConfig) -> str:
    payload = json.dumps(
        {
            "title": experiment.title,
            "goal": experiment.goal,
            "context": experiment.context,
            "constraints": experiment.constraints,
            "agent": asdict(experiment.agent),
            "backend": asdict(experiment.backend),
            "metadata": experiment.metadata,
            "seed": experiment.seed,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:12]


def build_task_crumb(experiment: ExperimentConfig) -> str:
    headers = {
        "v": "1.1",
        "kind": "task",
        "title": experiment.title,
        "source": str(experiment.metadata.get("source", "crumb.agent")),
    }
    sections = {
        "goal": [experiment.goal],
        "context": [f"- {item}" for item in experiment.context] or ["- No context supplied."],
        "constraints": [f"- {item}" for item in experiment.constraints]
        or ["- No additional constraints supplied."],
    }
    return render_crumb(headers, sections)


def build_log_crumb(
    experiment: ExperimentConfig,
    metrics: Dict[str, Any],
    runtime_seconds: float,
    output_path: Path,
) -> str:
    headers = {
        "v": "1.1",
        "kind": "log",
        "title": f"{experiment.title} run log",
        "source": "crumb.agent.runner",
    }
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entries = [
        f"- [{now}] backend={experiment.backend.kind}",
        f"- [{now}] output={output_path}",
        f"- [{now}] runtime_seconds={runtime_seconds:.3f}",
        f"- [{now}] metrics={json.dumps(metrics, sort_keys=True)}",
    ]
    return render_crumb(headers, {"entries": entries})


def build_mem_crumb(experiment: ExperimentConfig, metrics: Dict[str, Any]) -> str:
    headers = {
        "v": "1.1",
        "kind": "mem",
        "title": f"{experiment.title} findings",
        "source": "crumb.agent.eval",
    }
    consolidated = [
        f"- Agent: {experiment.agent.name}",
        f"- Model: {experiment.agent.model}",
        f"- Mode: {experiment.agent.mode}",
        f"- Passed: {metrics['passed']}",
        f"- Required terms present: {', '.join(metrics['required_terms_present']) or 'none'}",
        f"- Missing required terms: {', '.join(metrics['missing_required_terms']) or 'none'}",
        f"- Banned terms found: {', '.join(metrics['banned_terms_found']) or 'none'}",
        f"- Output chars: {metrics['chars']}",
        f"- Output tokens_estimate: {metrics['tokens_estimate']}",
    ]
    return render_crumb(headers, {"consolidated": consolidated})


def evaluate_output(text: str, config: EvaluationConfig) -> Dict[str, Any]:
    lowered = text.lower()
    required_present = [term for term in config.required_terms if term.lower() in lowered]
    missing_required = [term for term in config.required_terms if term.lower() not in lowered]
    banned_found = [term for term in config.banned_terms if term.lower() in lowered]
    chars = len(text)
    passed = not missing_required and not banned_found and config.min_chars <= chars <= config.max_chars
    return {
        "passed": passed,
        "required_terms_present": required_present,
        "missing_required_terms": missing_required,
        "banned_terms_found": banned_found,
        "chars": chars,
        "tokens_estimate": estimate_tokens(text),
        "min_chars": config.min_chars,
        "max_chars": config.max_chars,
    }


def select_backend(config: BackendConfig):
    if config.kind == "echo":
        return EchoBackend()
    if config.kind == "command":
        return CommandBackend(config)
    raise AgentError(f"unsupported backend kind: {config.kind}")


def run_experiment(config_path: str | Path, output_dir: str | Path | None = None, backend_override: str | None = None) -> Dict[str, Any]:
    experiment = ExperimentConfig.from_file(config_path)
    if backend_override:
        experiment.backend.kind = backend_override

    run_slug = slugify(experiment.title)
    run_id = stable_run_id(experiment)
    base_output = Path(output_dir or ".runs") / f"{run_slug}-{run_id}"
    base_output.mkdir(parents=True, exist_ok=True)

    task_crumb = build_task_crumb(experiment)
    task_path = base_output / "task.crumb"
    task_path.write_text(task_crumb, encoding="utf-8")

    telemetry = TelemetryStore(enabled=experiment.agent.telemetry_opt_in)
    backend = select_backend(experiment.backend)

    start = time.perf_counter()
    response = backend.generate(
        task_crumb,
        model=experiment.agent.model,
        mode=experiment.agent.mode,
        max_total_tokens=experiment.agent.max_total_tokens,
        run_dir=base_output,
    )
    runtime_seconds = time.perf_counter() - start

    response_path = base_output / "response.txt"
    response_path.write_text(response, encoding="utf-8")

    metrics = evaluate_output(response, experiment.evaluation)
    metrics_path = base_output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "run_dir": str(base_output),
        "title": experiment.title,
        "seed": experiment.seed,
        "backend": experiment.backend.kind,
        "model": experiment.agent.model,
        "mode": experiment.agent.mode,
        "runtime_seconds": runtime_seconds,
        "files": {
            "task": str(task_path),
            "response": str(response_path),
            "metrics": str(metrics_path),
            "log": str(base_output / "run.log.crumb"),
            "mem": str(base_output / "run.mem.crumb"),
        },
    }
    manifest_path = base_output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    log_path = base_output / "run.log.crumb"
    log_path.write_text(build_log_crumb(experiment, metrics, runtime_seconds, response_path), encoding="utf-8")

    mem_path = base_output / "run.mem.crumb"
    mem_path.write_text(build_mem_crumb(experiment, metrics), encoding="utf-8")

    telemetry.append(
        {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": run_id,
            "backend": experiment.backend.kind,
            "passed": metrics["passed"],
            "tokens_estimate": metrics["tokens_estimate"],
            "runtime_seconds": round(runtime_seconds, 3),
        }
    )

    return {
        "run_id": run_id,
        "run_dir": str(base_output),
        "passed": metrics["passed"],
        "metrics": metrics,
        "manifest": manifest,
    }


def run_eval(
    file_path: str | Path,
    *,
    config_path: str | Path | None = None,
    required_terms: Iterable[str] = (),
    banned_terms: Iterable[str] = (),
    min_chars: int = 0,
    max_chars: int = 100_000,
) -> Dict[str, Any]:
    text = Path(file_path).read_text(encoding="utf-8")
    if config_path:
        evaluation = ExperimentConfig.from_file(config_path).evaluation
    else:
        evaluation = EvaluationConfig(
            required_terms=list(required_terms),
            banned_terms=list(banned_terms),
            min_chars=min_chars,
            max_chars=max_chars,
        )
    return evaluate_output(text, evaluation)


def cmd_agent(args: argparse.Namespace) -> None:
    action = args.agent_action

    if action == "template":
        rendered = json.dumps(DEFAULT_TEMPLATE, indent=2) + "\n"
        output = getattr(args, "output", "-")
        if output in (None, "-"):
            sys.stdout.write(rendered)
        else:
            Path(output).write_text(rendered, encoding="utf-8")
            print(f"Wrote template to {output}")
        return

    if action == "run":
        result = run_experiment(
            config_path=args.config,
            output_dir=args.output_dir,
            backend_override=args.backend,
        )
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return

    if action == "eval":
        result = run_eval(
            args.file,
            config_path=args.config,
            required_terms=args.required_terms,
            banned_terms=args.banned_terms,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return

    raise AgentError(f"unsupported agent action: {action}")
