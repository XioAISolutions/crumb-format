"""Local Ollama helpers for CRUMB generation and compression."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3"


class LocalAIError(RuntimeError):
    """Raised when the local Ollama workflow cannot complete."""


def _post_json(payload: dict, timeout: float) -> dict:
    request_model = str(payload.get("model") or DEFAULT_OLLAMA_MODEL)
    request = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            raw = response.read().decode(charset)
        return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        message = ""
        try:
            error_payload = json.loads(body)
            message = str(error_payload.get("error") or "").strip()
        except Exception:
            message = body.strip()
        if message and "not found" in message.lower() and "model" in message.lower():
            raise LocalAIError(
                f"Ollama is running, but model '{request_model}' is not available. "
                f"Run `ollama pull {request_model}` first."
            ) from exc
        detail = message or str(exc.reason)
        raise LocalAIError(f"Ollama request failed: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LocalAIError(
            "Could not reach Ollama at http://localhost:11434. Start Ollama with `ollama serve` and try again."
        ) from exc
    except json.JSONDecodeError as exc:
        raise LocalAIError("Ollama returned invalid JSON.") from exc


def ensure_ollama_available(model: str = DEFAULT_OLLAMA_MODEL, timeout: float = 2.0) -> None:
    payload = {
        "model": model,
        "prompt": "Reply with OK and nothing else.",
        "stream": False,
        "options": {"num_predict": 8},
    }
    response = _post_json(payload, timeout=timeout)
    text = str(response.get("response") or "").strip()
    if not text:
        raise LocalAIError("Ollama responded, but no text was returned from /api/generate.")


def generate_text(prompt: str, model: str = DEFAULT_OLLAMA_MODEL, timeout: float = 90.0) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = _post_json(payload, timeout=timeout)
    text = str(response.get("response") or "").strip()
    if not text:
        raise LocalAIError("Ollama returned an empty response.")
    return text


def extract_crumb_block(text: str) -> str:
    match = re.search(r"BEGIN CRUMB[\s\S]*?END CRUMB", text)
    if match:
        return match.group(0).strip()
    raise LocalAIError("Ollama did not return a valid CRUMB block bounded by BEGIN CRUMB and END CRUMB.")
