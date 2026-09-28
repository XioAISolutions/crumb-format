"""MCP tool definitions for Crumb LLM — context-pulling inference.

Adds four tools to the existing CRUMB MCP server, enabling any MCP
client (Claude Desktop, Cursor, etc.) to:

  1. crumb_llm_complete  — generate text from a prompt
  2. crumb_llm_pull      — retrieve relevant crumb sections for a query
  3. crumb_llm_score     — score text perplexity under the model
  4. crumb_llm_index     — build/rebuild the context index

These tools implement the "context pulling" pattern: instead of
stuffing the entire crumb library into the context window, the LLM
client calls crumb_llm_pull to dynamically retrieve only the sections
spectrally relevant to its current task, then optionally feeds them
to crumb_llm_complete for wave-field generation.
"""

from __future__ import annotations

TOOLS = [
    {
        "name": "crumb_llm_complete",
        "description": (
            "Generate text using the Crumb LLM wave-field model. "
            "Accepts a prompt and returns generated text. Uses the field-state "
            "cache for efficient O(F log F) generation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Input text to continue."},
                "max_tokens": {"type": "integer", "default": 64, "description": "Max tokens to generate."},
                "temperature": {"type": "number", "default": 1.0},
                "top_k": {"type": "integer", "default": 40},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "crumb_llm_pull",
        "description": (
            "Context pulling: retrieve the most relevant crumb sections for a "
            "query using spectral matching. Returns section text ranked by "
            "wave-field resonance score. Use this to dynamically load only the "
            "context you need instead of stuffing the entire crumb library."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query to match against indexed crumb sections."},
                "max_tokens": {"type": "integer", "default": 256, "description": "Token budget for pulled context."},
                "top_k": {"type": "integer", "default": 5, "description": "Max sections to consider."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "crumb_llm_score",
        "description": "Score a text's perplexity under the Crumb LLM model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to score."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "crumb_llm_index",
        "description": (
            "Build or rebuild the context-pulling index from a directory of "
            ".crumb files. Each section is spectrally fingerprinted for fast "
            "retrieval via crumb_llm_pull."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Path to directory containing .crumb files."},
            },
            "required": ["directory"],
        },
    },
]


def handle_tool(name: str, args: dict, model=None, tokenizer=None, index=None) -> str:
    """Handle an MCP tool call. Returns result text."""
    import math
    import torch
    from .cache import generate_cached
    from .context_pull import build_index, pull_context, score_sections

    if name == "crumb_llm_complete":
        if model is None:
            return "Error: no model loaded. Set CRUMB_LLM_CKPT environment variable."
        prompt = args["prompt"]
        max_tokens = args.get("max_tokens", 64)
        temperature = args.get("temperature", 1.0)
        top_k = args.get("top_k", 40)
        ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
        out = generate_cached(model, ids, max_new_tokens=max_tokens,
                              temperature=temperature, top_k=top_k)
        return tokenizer.decode(out[0].tolist())

    if name == "crumb_llm_pull":
        if index is None:
            return "Error: no index built. Call crumb_llm_index first."
        query = args["query"]
        max_tokens = args.get("max_tokens", 256)
        top_k = args.get("top_k", 5)
        pulled = pull_context(query, index, max_tokens=max_tokens, top_k=top_k)
        if not pulled:
            return "No relevant sections found."
        return pulled

    if name == "crumb_llm_score":
        if model is None:
            return "Error: no model loaded."
        text = args["text"]
        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0)
        max_ctx = getattr(model.cfg, "field_size", ids.size(1))
        ids = ids[:, :max_ctx]
        with torch.no_grad():
            x, y = ids[:, :-1], ids[:, 1:]
            out = model(x, targets=y)
        loss = out["loss"].item()
        return f"tokens={ids.size(1)} loss={loss:.4f} ppl={math.exp(loss):.2f} bpc={loss / math.log(2):.3f}"

    if name == "crumb_llm_index":
        directory = args["directory"]
        field_size = getattr(model.cfg, "field_size", 256) if model else 256
        new_index = build_index(directory, field_size=field_size)
        return f"Indexed {len(new_index.sections)} sections from {directory}"

    return f"Unknown tool: {name}"
