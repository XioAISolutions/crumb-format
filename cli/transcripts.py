"""Real-conversation ingest — the ``crumb from-messages`` command.

``from-chat`` parses ad-hoc chat text with prefix heuristics ("User:",
"Claude:"). Almost no conversation is actually stored that way. This module
reads the shapes conversations really live in:

  - **OpenAI Chat Completions** — ``{"messages": [{"role", "content"}]}``,
    a bare ``[...]`` array, or a full response object with ``choices``.
    ``tool_calls`` / ``role=tool`` turns are understood.
  - **Anthropic Messages API** — the same envelope with content-block
    arrays: ``text``, ``tool_use``, ``tool_result``, ``thinking``.
  - **ChatGPT data export** — ``conversations.json``, the mapping-tree form
    where each node holds ``message.author.role`` and ``message.content.parts``.
  - **Claude.ai data export** — conversation objects with ``chat_messages``.
  - **JSONL** — one message or one conversation per line, any of the above.

and normalises them into a v1.4 ``kind=task`` handoff: the goal, the
decisions that were reached, the artifacts touched, the constraints stated
along the way, and what is still open — without the turn-by-turn noise that
makes pasting a raw transcript so expensive.

The extractors are deliberately lexical and deterministic. No model is
called, so the same transcript always yields the same crumb and the output
can be diffed in CI.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple


# ── normalised message ───────────────────────────────────────────────

@dataclass
class Message:
    """One turn, flattened out of whichever provider shape it arrived in."""

    role: str = "user"
    text: str = ""
    tool_calls: List[str] = field(default_factory=list)
    tool_results: List[str] = field(default_factory=list)
    code_blocks: List[Dict[str, str]] = field(default_factory=list)

    def is_substantive(self) -> bool:
        return bool(self.text.strip()) or bool(self.tool_calls)


class TranscriptError(ValueError):
    """Raised when the input is not a transcript we can read."""


# ── format detection ─────────────────────────────────────────────────

def _looks_like_chatgpt_export(obj: Any) -> bool:
    candidates = obj if isinstance(obj, list) else [obj]
    for item in candidates[:4]:
        if isinstance(item, dict) and isinstance(item.get("mapping"), dict):
            return True
    return False


def _looks_like_claude_export(obj: Any) -> bool:
    candidates = obj if isinstance(obj, list) else [obj]
    for item in candidates[:4]:
        if isinstance(item, dict) and isinstance(item.get("chat_messages"), list):
            return True
    return False


def _messages_of(obj: Any) -> List[Any] | None:
    """Pull a messages array out of the common API envelopes."""
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, dict):
        return None
    for key in ("messages", "input", "conversation"):
        value = obj.get(key)
        if isinstance(value, list):
            return value
    choices = obj.get("choices")
    if isinstance(choices, list):
        collected = [c["message"] for c in choices
                     if isinstance(c, dict) and isinstance(c.get("message"), dict)]
        if collected:
            return collected
    return None


def _has_content_blocks(messages: Sequence[Any]) -> bool:
    for msg in messages:
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            return True
    return False


def detect_format(obj: Any) -> str:
    """Name the transcript shape: one of the ``FORMATS`` keys."""
    if _looks_like_chatgpt_export(obj):
        return "chatgpt-export"
    if _looks_like_claude_export(obj):
        return "claude-export"
    messages = _messages_of(obj)
    if messages is None:
        raise TranscriptError(
            "no messages found — expected an OpenAI/Anthropic messages array, "
            "a ChatGPT conversations.json, or a Claude.ai export"
        )
    if _has_content_blocks(messages):
        return "anthropic-messages"
    return "openai-messages"


FORMATS = (
    "openai-messages",
    "anthropic-messages",
    "chatgpt-export",
    "claude-export",
)


# ── per-format readers ───────────────────────────────────────────────

_ROLE_ALIASES = {
    "human": "user",
    "model": "assistant",
    "ai": "assistant",
    "bot": "assistant",
    "chatgpt": "assistant",
    "claude": "assistant",
}


def _normalise_role(role: Any) -> str:
    role = str(role or "user").strip().lower()
    return _ROLE_ALIASES.get(role, role)


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _read_content(content: Any) -> Tuple[str, List[str], List[str]]:
    """Flatten a content field into (text, tool_calls, tool_results)."""
    if content is None:
        return "", [], []
    if isinstance(content, str):
        return content, [], []
    if not isinstance(content, list):
        return str(content), [], []

    parts: List[str] = []
    calls: List[str] = []
    results: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in ("text", "input_text", "output_text"):
            parts.append(str(block.get("text", "")))
        elif btype == "thinking":
            # Reasoning traces are not handoff material — they are the most
            # expensive and least reusable part of a transcript. Dropped.
            continue
        elif btype == "tool_use":
            name = block.get("name", "tool")
            args = block.get("input")
            calls.append(_describe_tool_call(name, args))
        elif btype == "tool_result":
            results.append(_truncate(_stringify(block.get("content")), 200))
        elif btype in ("image", "input_image"):
            parts.append("[image]")
    return "\n".join(p for p in parts if p), calls, results


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out = []
        for block in value:
            if isinstance(block, dict) and "text" in block:
                out.append(str(block["text"]))
            else:
                out.append(_stringify(block))
        return " ".join(out)
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _describe_tool_call(name: Any, args: Any) -> str:
    """Render a tool call as ``name(key=value, …)`` with values clipped."""
    label = str(name or "tool")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (TypeError, ValueError):
            return f"{label}({_truncate(args, 60)})"
    if isinstance(args, dict) and args:
        rendered = ", ".join(
            f"{k}={_truncate(_stringify(v), 40)}" for k, v in list(args.items())[:3]
        )
        return f"{label}({rendered})"
    return f"{label}()"


def _read_api_message(raw: Dict[str, Any]) -> Message:
    text, calls, results = _read_content(raw.get("content"))
    # OpenAI keeps tool calls beside the content rather than inside it.
    for call in raw.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        calls.append(_describe_tool_call(fn.get("name", call.get("name")), fn.get("arguments")))
    role = _normalise_role(raw.get("role"))
    if role == "tool":
        results.append(_truncate(text, 200))
        text = ""
    return Message(role=role, text=text, tool_calls=calls, tool_results=results)


def _read_api_messages(obj: Any) -> List[Message]:
    messages = _messages_of(obj) or []
    out = []
    for raw in messages:
        if isinstance(raw, dict):
            out.append(_read_api_message(raw))
    return out


def _read_chatgpt_export(obj: Any) -> List[Message]:
    """Walk a ChatGPT export mapping tree in creation order."""
    conversations = obj if isinstance(obj, list) else [obj]
    out: List[Message] = []
    for convo in conversations:
        if not isinstance(convo, dict):
            continue
        mapping = convo.get("mapping")
        if not isinstance(mapping, dict):
            continue
        nodes = []
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            msg = node.get("message")
            if not isinstance(msg, dict):
                continue
            nodes.append((msg.get("create_time") or 0, msg))
        nodes.sort(key=lambda pair: pair[0])
        for _, msg in nodes:
            author = msg.get("author") or {}
            role = _normalise_role(author.get("role"))
            if role == "system":
                continue
            content = msg.get("content") or {}
            parts = content.get("parts")
            if isinstance(parts, list):
                text = "\n".join(_stringify(p) for p in parts if p)
            else:
                text = _stringify(content.get("text", ""))
            if text.strip():
                out.append(Message(role=role, text=text))
    return out


def _read_claude_export(obj: Any) -> List[Message]:
    conversations = obj if isinstance(obj, list) else [obj]
    out: List[Message] = []
    for convo in conversations:
        if not isinstance(convo, dict):
            continue
        for msg in convo.get("chat_messages") or []:
            if not isinstance(msg, dict):
                continue
            role = _normalise_role(msg.get("sender") or msg.get("role"))
            text = msg.get("text")
            if not text:
                text, _, _ = _read_content(msg.get("content"))
            if str(text).strip():
                out.append(Message(role=role, text=str(text)))
    return out


_READERS = {
    "openai-messages": _read_api_messages,
    "anthropic-messages": _read_api_messages,
    "chatgpt-export": _read_chatgpt_export,
    "claude-export": _read_claude_export,
}


def load_messages(raw: str, fmt: str | None = None) -> Tuple[str, List[Message]]:
    """Parse transcript text into ``(format_name, messages)``.

    Accepts JSON or JSONL. ``fmt`` forces a reader; otherwise it is detected.
    """
    obj = _parse_json_or_jsonl(raw)
    resolved = fmt or detect_format(obj)
    if resolved not in _READERS:
        raise TranscriptError(f"unknown format {resolved!r}; expected one of {', '.join(FORMATS)}")
    messages = _READERS[resolved](obj)
    for msg in messages:
        msg.code_blocks = _extract_code_blocks(msg.text)
    messages = [m for m in messages if m.is_substantive() or m.tool_results]
    if not messages:
        raise TranscriptError("transcript parsed but contained no usable messages")
    return resolved, messages


def _parse_json_or_jsonl(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        raise TranscriptError("empty input")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    records = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise TranscriptError(
                f"input is neither JSON nor JSONL (line {lineno}: {exc.msg})"
            ) from exc
    if not records:
        raise TranscriptError("empty input")
    # A JSONL file of bare messages is itself the messages array.
    if all(isinstance(r, dict) and "role" in r for r in records):
        return records
    return records if len(records) > 1 else records[0]


# ── extraction ───────────────────────────────────────────────────────

CODE_FENCE_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
PATH_RE = re.compile(r"\b[\w.-]+/[\w./-]+\.[A-Za-z]{1,6}\b|\b[\w-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|sql|yaml|yml|json|toml|md|sh|css|html)\b")
URL_RE = re.compile(r"https?://[^\s<>\"')]+")

# Sentence boundaries require whitespace *and* a capitalised opener after the
# stop, so `tests/checkout_refresh.spec.ts` and `v1.4` stay in one piece. A
# naive split on [.!?] truncates exactly the filenames a handoff needs most.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\-(])|\n+")

CONSTRAINT_WORDS = re.compile(
    r"\b(?:must not|must|never|do not|don't|cannot|can't|avoid|should not|"
    r"shouldn't|only use|only if|keep|preserve|without changing|make sure|"
    r"ensure|required to|has to)\b",
    re.IGNORECASE,
)
OPEN_THREAD_WORDS = re.compile(
    r"\b(?:next step|next,|still need|still needs|todo|to do|remaining|"
    r"not yet|left to do|follow[- ]up|after that|then we)\b",
    re.IGNORECASE,
)

# The diagnosis is separate from the decision: "sessionStorage is cleared by
# the redirect" is why the fix works, and a handoff that carries only the fix
# forces the next model to re-derive it. Measured retention on real
# transcripts showed error codes and latencies dropping out without this.
FINDING_WORDS = re.compile(
    r"\b(?:root cause|the bug is|found it|because|caused by|turns out|"
    r"the problem is|fails? when|breaks? when|reproduces?)\b",
    re.IGNORECASE,
)
# Diagnostic values worth carrying even when stated in passing prose.
MEASUREMENT_RE = re.compile(
    r"\b(?:[45]\d{2}\b|\d+(?:\.\d+)?\s?(?:ms|kb|mb|gb|%)\b|[A-Z][a-zA-Z]*(?:Error|Exception)\b)"
)

# Verbs that mark a line as an action already taken rather than a plan.
_NOISE_PREFIXES = (
    "sure", "certainly", "of course", "great", "thanks", "thank you",
    "let me know", "happy to", "no problem", "you're welcome", "i'd be happy",
    "here's", "here is", "absolutely",
)


def _extract_code_blocks(text: str) -> List[Dict[str, str]]:
    return [
        {"lang": (m.group(1) or "").strip(), "code": m.group(2)}
        for m in CODE_FENCE_RE.finditer(text or "")
    ]


def _strip_code(text: str) -> str:
    return CODE_FENCE_RE.sub(" ", text or "")


def _is_noise(sentence: str) -> bool:
    low = sentence.strip().lower()
    return (not low) or low.startswith(_NOISE_PREFIXES) or len(low) < 12


def _dedupe(items: Iterable[str], limit: int) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        item = " ".join(str(item).split()).strip(" -•\t")
        key = re.sub(r"[^a-z0-9 ]", "", item.lower())
        if not item or key in seen or _is_noise(item):
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def sentences(text: str) -> List[str]:
    """Split prose into sentences without breaking dotted identifiers."""
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(_strip_code(text or "")) if s.strip()]


def _sentences_matching(
    pattern: re.Pattern, messages: Sequence[Message], roles: Sequence[str]
) -> List[str]:
    """Whole sentences (never mid-token slices) that mention the pattern."""
    hits: List[str] = []
    for msg in messages:
        if msg.role not in roles:
            continue
        for sentence in sentences(msg.text):
            if pattern.search(sentence):
                hits.append(sentence)
    return hits


def _drop_contained(items: Sequence[str]) -> List[str]:
    """Drop entries wholly contained in a longer sibling.

    Two decision patterns firing on the same sentence otherwise yields one
    line and a near-duplicate prefix of it.
    """
    keys = [re.sub(r"[^a-z0-9 ]", "", i.lower()) for i in items]
    out = []
    for i, item in enumerate(items):
        if any(j != i and keys[i] and keys[i] in keys[j] and len(keys[j]) > len(keys[i])
               for j in range(len(items))):
            continue
        out.append(item)
    return out


def derive_goal(messages: Sequence[Message]) -> str:
    """The most recent substantive user instruction — what to continue doing."""
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        body = _strip_code(msg.text).strip()
        if len(body) < 12:
            continue
        # Lead with the first substantive sentence; acknowledgements like
        # "Good." are not the goal, so skip past them.
        for sentence in sentences(body):
            if len(sentence) > 24:
                return _truncate(sentence, 300)
        return _truncate(body, 300)
    return "Continue this work from where the previous assistant left off."


def derive_title(messages: Sequence[Message]) -> str:
    for msg in messages:
        if msg.role == "user" and _strip_code(msg.text).strip():
            return _truncate(_strip_code(msg.text).strip().splitlines()[0], 70)
    return "Continued session"


def collect(messages: Sequence[Message]) -> Dict[str, List[str]]:
    """Pull the reusable signal out of a transcript."""
    from cli.crumb import DECISION_PATTERNS

    # A decision is reported as the whole sentence that carries it — the
    # match itself is a fragment starting mid-clause ("going with X").
    decisions: List[str] = []
    for msg in messages:
        for sentence in sentences(msg.text):
            if any(p.search(sentence) for p in DECISION_PATTERNS):
                decisions.append(_truncate(sentence, 200))

    # Paths and links appear in tool arguments and results as often as in
    # prose. A spec URL is load-bearing: dropping it makes the next model
    # guess at a contract it could have read.
    paths: List[str] = []
    links: List[str] = []
    for msg in messages:
        haystack = " ".join([msg.text or "", *msg.tool_calls, *msg.tool_results])
        paths.extend(PATH_RE.findall(haystack))
        links.extend(u.rstrip(".,;:!?") for u in URL_RE.findall(haystack))

    tools: List[str] = []
    for msg in messages:
        tools.extend(msg.tool_calls)

    languages: List[str] = []
    for msg in messages:
        for block in msg.code_blocks:
            if block["lang"]:
                languages.append(block["lang"])

    # A question is a request, never a constraint — "Can you keep digging?"
    # otherwise lands in [constraints] on the strength of the word "keep".
    constraints = [s for s in _sentences_matching(CONSTRAINT_WORDS, messages, ("user", "system"))
                   if not s.rstrip().endswith("?")]
    open_threads = _sentences_matching(OPEN_THREAD_WORDS, messages, ("assistant", "user"))
    questions = [s for s in _sentences_matching(re.compile(r"\?"), messages[-3:], ("assistant",))
                 if s.endswith("?")]

    findings = [
        s for s in _sentences_matching(FINDING_WORDS, messages, ("assistant",))
        if not s.rstrip().endswith("?")
    ]
    findings += [s for s in _sentences_matching(MEASUREMENT_RE, messages, ("assistant", "user"))
                 if not s.rstrip().endswith("?")]

    return {
        "decisions": _dedupe(_drop_contained(decisions), 8),
        "findings": _dedupe((_truncate(f, 200) for f in _drop_contained(findings)), 5),
        "paths": _dedupe(paths, 12),
        "links": _dedupe(links, 4),
        "tools": _dedupe(tools, 8),
        "languages": _dedupe(languages, 6),
        "constraints": _dedupe((_truncate(c, 200) for c in _drop_contained(constraints)), 6),
        "open": _dedupe((_truncate(o, 200) for o in _drop_contained(open_threads)), 5),
        "questions": _dedupe((_truncate(q, 160) for q in questions), 3),
    }


# ── rendering ────────────────────────────────────────────────────────

def to_crumb(
    messages: Sequence[Message],
    *,
    fmt: str,
    title: str | None = None,
    source: str | None = None,
    goal: str | None = None,
    project: str | None = None,
    extra_constraints: Sequence[str] = (),
    version: str = "1.4",
) -> str:
    """Render normalised messages as a v1.4 ``kind=task`` handoff crumb."""
    from cli.crumb import render_crumb

    signal = collect(messages)
    headers: Dict[str, str] = {
        "v": version,
        "kind": "task",
        "title": title or derive_title(messages),
        "source": source or f"transcript.{fmt}",
    }
    if project:
        headers["project"] = project

    context: List[str] = []
    if signal["findings"]:
        context.append("- What was diagnosed:")
        context.extend(f"  - {f}" for f in signal["findings"])
    if signal["decisions"]:
        context.append("- Decisions reached:")
        context.extend(f"  - {d}" for d in signal["decisions"])
    if signal["paths"]:
        context.append(f"- Files in play: {', '.join(signal['paths'])}")
    if signal["links"]:
        context.append("- References:")
        context.extend(f"  - {u}" for u in signal["links"])
    if signal["tools"]:
        context.append("- Tools already called:")
        context.extend(f"  - {t}" for t in signal["tools"])
    if signal["languages"]:
        context.append(f"- Code discussed: {', '.join(signal['languages'])}")
    if not context:
        # Fall back to the last exchange rather than emitting an empty section.
        tail = [m for m in messages if m.role in ("user", "assistant")][-2:]
        context.extend(f"- {m.role}: {_truncate(_strip_code(m.text), 200)}" for m in tail if m.text.strip())
    if not context:
        context.append("- No reusable context extracted from this transcript.")

    constraints = [f"- {c}" for c in signal["constraints"]]
    constraints.extend(f"- {c.strip()}" for c in extra_constraints if str(c).strip())
    if not constraints:
        constraints.append("- No constraints were stated in this conversation.")

    sections: Dict[str, List[str]] = {
        "goal": [goal or derive_goal(messages), ""],
        "context": context + [""],
        "constraints": constraints + [""],
    }

    handoff = [f"- to=any    do={o}" for o in signal["open"]]
    handoff.extend(f"- to=human  do=answer: {q}" for q in signal["questions"])
    if handoff:
        sections["handoff"] = handoff + [""]

    turns = len([m for m in messages if m.role in ("user", "assistant")])
    sections["notes"] = [
        f"- Derived from {fmt} transcript: {turns} turns, "
        f"{len(messages)} messages, {sum(len(m.code_blocks) for m in messages)} code blocks.",
        "- Reasoning traces and turn-by-turn phrasing were dropped; decisions, "
        "artifacts, constraints, and open threads were kept.",
        "",
    ]

    return render_crumb(headers, sections)


def transcript_text(messages: Sequence[Message]) -> str:
    """Re-render the transcript as plain text — the baseline to measure against."""
    lines = []
    for msg in messages:
        lines.append(f"{msg.role}: {msg.text}".rstrip())
        for call in msg.tool_calls:
            lines.append(f"{msg.role} tool_call: {call}")
        for result in msg.tool_results:
            lines.append(f"tool_result: {result}")
    return "\n".join(lines) + "\n"
