"""Chat templates for Crumb LLM.

Two conventions are shipped:

* ``chatml``  — OpenAI-style ``<|im_start|>role\\ncontent<|im_end|>``.
                Widest interop with local LLM tooling. Default.
* ``crumb``   — CRUMB-shaped: ``BEGIN CRUMB\\n[goal] ...\\n[context] ...\\n
                [response] ...``. Plays to the wave-field's
                section-boundary strength.

Both templates accept the same OpenAI-compatible message dicts:

    messages = [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
    ]

Call :func:`apply_chat_template` to render messages to a single string
that can be passed straight into ``tokenizer.encode(...)``. By default
the renderer appends the assistant-turn prefix so the model continues
with the assistant's reply.
"""

from __future__ import annotations

from typing import Iterable


Message = dict  # {"role": str, "content": str}


# ── ChatML ───────────────────────────────────────────────────────────


CHATML_STOP_TOKENS = ("<|im_end|>",)


def render_chatml(messages: Iterable[Message], add_generation_prompt: bool = True) -> str:
    parts: list[str] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# ── CRUMB-shaped ─────────────────────────────────────────────────────


CRUMB_STOP_TOKENS = ("<|crumb_end|>", "END CRUMB")


def render_crumb(messages: Iterable[Message], add_generation_prompt: bool = True) -> str:
    """Render messages as a CRUMB document.

    Maps OpenAI roles onto CRUMB sections:
        system    → [context] (always first)
        user      → [goal]
        assistant → [response]

    Multi-turn conversations concatenate turns inside a single CRUMB.
    """
    msgs = list(messages)
    parts: list[str] = ["<|crumb_begin|>BEGIN CRUMB\n"]
    sys_msgs = [m for m in msgs if m["role"] == "system"]
    convo_msgs = [m for m in msgs if m["role"] != "system"]
    if sys_msgs:
        parts.append(f"<|context|>[context]\n{sys_msgs[-1]['content']}\n")
    for m in convo_msgs:
        if m["role"] == "user":
            parts.append(f"<|goal|>[goal]\n{m['content']}\n")
        elif m["role"] == "assistant":
            parts.append(f"<|response|>[response]\n{m['content']}\n")
    if add_generation_prompt:
        parts.append("<|response|>[response]\n")
    return "".join(parts)


# ── Dispatcher ───────────────────────────────────────────────────────


_TEMPLATES = {
    "chatml": (render_chatml, CHATML_STOP_TOKENS),
    "crumb": (render_crumb, CRUMB_STOP_TOKENS),
}


def apply_chat_template(
    messages: Iterable[Message],
    template: str = "chatml",
    add_generation_prompt: bool = True,
) -> str:
    """Render messages into a single prompt string.

    Args:
        messages: OpenAI-style list of ``{"role", "content"}`` dicts.
        template: One of ``"chatml"`` or ``"crumb"``.
        add_generation_prompt: If True, append the assistant-turn prefix
            so the model continues with the reply. Set to False when you
            just want a transcript representation of completed turns.
    """
    if template not in _TEMPLATES:
        raise ValueError(f"unknown template: {template!r}; pick one of {sorted(_TEMPLATES)}")
    render, _ = _TEMPLATES[template]
    return render(messages, add_generation_prompt=add_generation_prompt)


def stop_tokens_for(template: str) -> tuple[str, ...]:
    """Stop strings appropriate for sampling under ``template``."""
    if template not in _TEMPLATES:
        raise ValueError(f"unknown template: {template!r}")
    return _TEMPLATES[template][1]
