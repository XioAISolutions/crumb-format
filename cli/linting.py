"""Safety and protocol linting for CRUMBs."""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from cli.extensions import is_known_header, is_namespaced_extension_name, is_namespaced_header, is_valid_header_key, parse_extensions


def _crumb():
    return importlib.import_module("cli.crumb")


SECRET_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}"), "possible OpenAI API key"),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "possible GitHub token"),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}"), "possible AWS access key"),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "possible Slack token"),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}", re.IGNORECASE), "possible bearer token"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9._-]{10,}\.[A-Za-z0-9._-]{10,}\b"), "possible JWT"),
    (
        "generic_secret",
        re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*[\"']?[A-Za-z0-9/_+.-]{12,}[\"']?"),
        "possible credential assignment",
    ),
]

RAW_SECTIONS = {"raw", "logs", "raw_sessions", "entries"}


@dataclass(frozen=True)
class LintFinding:
    level: str
    code: str
    message: str
    path: str


def _redact_secret(pattern: re.Pattern[str], text: str, label: str) -> str:
    return pattern.sub(f"[REDACTED:{label}]", text)


def _read_headers_from_text(text: str) -> dict[str, str]:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    try:
        sep_index = lines.index("---")
    except ValueError:
        return {}
    headers: dict[str, str] = {}
    for line in lines[1:sep_index]:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers


def _index_tokens(parsed: dict[str, object]) -> int:
    crumb = _crumb()
    sections = parsed["sections"]
    index_sections = {"goal", "constraints", "consolidated", "project", "modules", "tasks"}
    text = "\n".join(
        line
        for name, lines in sections.items()
        if name in index_sections
        for line in lines
        if line.strip()
    )
    return crumb.estimate_tokens(text)


def lint_text(path: str, text: str, args: argparse.Namespace) -> tuple[list[LintFinding], str]:
    crumb = _crumb()
    findings: list[LintFinding] = []
    redacted = text

    try:
        parsed = crumb.parse_crumb(text)
    except Exception as exc:
        return [LintFinding("ERROR", "parse_error", str(exc), path)], redacted

    headers = parsed["headers"]
    raw_headers = _read_headers_from_text(text)

    for key in raw_headers:
        if not is_valid_header_key(key):
            findings.append(LintFinding("WARN", "header_key_format", f"suspicious header key '{key}'", path))
        elif not is_known_header(key) and not is_namespaced_header(key):
            findings.append(
                LintFinding(
                    "WARN",
                    "header_extension_namespace",
                    f"unknown header '{key}' is not namespaced; prefer x-... or ext.namespace.key headers",
                    path,
                )
            )

    for name in parse_extensions(headers.get("extensions")):
        if not is_namespaced_extension_name(name):
            findings.append(
                LintFinding(
                    "WARN",
                    "extension_name",
                    f"extension '{name}' is not namespaced; prefer names like crumb.pack.v1",
                    path,
                )
            )

    if args.secrets or args.redact:
        for label, pattern, description in SECRET_PATTERNS:
            if pattern.search(redacted):
                findings.append(LintFinding("ERROR", label, description, path))
                if args.redact:
                    redacted = _redact_secret(pattern, redacted, label)

    max_size = args.max_size
    if max_size is not None:
        total_tokens = crumb.estimate_tokens(text)
        if total_tokens > max_size:
            findings.append(
                LintFinding("WARN", "max_size_total", f"estimated size {total_tokens} exceeds --max-size {max_size}", path)
            )

    for section_name, lines in parsed["sections"].items():
        if section_name in RAW_SECTIONS:
            section_text = "\n".join(line for line in lines if line.strip())
            if section_text:
                section_tokens = crumb.estimate_tokens(section_text)
                if max_size is not None and section_tokens > max_size:
                    findings.append(
                        LintFinding(
                            "WARN",
                            "raw_section_size",
                            f"section [{section_name}] is very large ({section_tokens} tokens > {max_size})",
                            path,
                        )
                    )
                if section_name in {"entries", "raw_sessions"} and section_tokens > 1000:
                    findings.append(
                        LintFinding(
                            "WARN",
                            "raw_log_volume",
                            f"section [{section_name}] looks like a large raw log ({section_tokens} tokens)",
                            path,
                        )
                    )

    if "max_total_tokens" in headers:
        try:
            budget = int(headers["max_total_tokens"])
            actual = crumb.estimate_tokens(text)
            if actual > budget:
                findings.append(
                    LintFinding("WARN", "budget_total", f"estimated size {actual} exceeds max_total_tokens={budget}", path)
                )
        except ValueError:
            findings.append(LintFinding("WARN", "budget_total_format", "max_total_tokens is not an integer", path))

    if "max_index_tokens" in headers:
        try:
            budget = int(headers["max_index_tokens"])
            actual = _index_tokens(parsed)
            if actual > budget:
                findings.append(
                    LintFinding("WARN", "budget_index", f"estimated index size {actual} exceeds max_index_tokens={budget}", path)
                )
        except ValueError:
            findings.append(LintFinding("WARN", "budget_index_format", "max_index_tokens is not an integer", path))

    if getattr(args, "check_refs", False):
        try:
            from cli import ref_resolver
        except ImportError:
            import ref_resolver  # type: ignore[no-redef]
        refs_value = headers.get("refs", "").strip()
        if refs_value:
            base_dir = Path(path).resolve().parent
            search_paths = [base_dir]
            for ref in (r.strip() for r in refs_value.split(",")):
                if not ref:
                    continue
                resolved = ref_resolver.resolve_ref(ref, search_paths=search_paths)
                if resolved is None:
                    findings.append(
                        LintFinding(
                            "WARN",
                            "unresolved_ref",
                            f"ref {ref!r} did not resolve (SPEC v1.3 §17.3)",
                            path,
                        )
                    )

    return findings, redacted


def _write_redacted(path: str, content: str, output: str | None) -> None:
    crumb = _crumb()
    destination = output or path
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    crumb.write_text(destination, content)


def run_lint(args: argparse.Namespace) -> None:
    all_findings: list[LintFinding] = []
    parse_failures = 0
    security_failures = 0

    for filepath in args.files:
        text = Path(filepath).read_text(encoding="utf-8")
        findings, redacted = lint_text(filepath, text, args)
        all_findings.extend(findings)

        if args.redact and redacted != text:
            if len(args.files) > 1 and args.output:
                output_path = str(Path(args.output) / Path(filepath).name)
            else:
                output_path = args.output
            _write_redacted(filepath, redacted, output_path)

        for finding in findings:
            stream = sys.stderr if finding.level == "ERROR" else sys.stdout
            print(f"{finding.level} {finding.path} {finding.code} {finding.message}", file=stream)

        parse_failures += sum(1 for item in findings if item.code == "parse_error")
        security_failures += sum(1 for item in findings if item.level == "ERROR" and item.code != "parse_error")

    warning_count = sum(1 for item in all_findings if item.level == "WARN")
    if parse_failures:
        raise SystemExit(2)
    if security_failures:
        raise SystemExit(1)
    if args.strict and warning_count:
        raise SystemExit(1)
    raise SystemExit(0)
