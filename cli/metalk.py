"""MeTalk — caveman-style token compression for CRUMB.

Layers:
  Layer 1 (lossless):    dictionary word→abbreviation swaps
  Layer 2 (lossy):       strip articles/filler, rewrite verbose phrases
  Layer 3 (aggressive):  + sentence condensing
  Layer 4 (skeleton):    + interior vowel stripping (cli/vowelstrip)
  Layer 5 (adaptive):    + embedding-aware vowel stripping (requires
                         the [embeddings] extra; falls back to L4)

Usage:
    from cli.metalk import encode, decode, compression_stats
    slim = encode(crumb_text, level=2)
    full = decode(slim)  # reverses Layer 1 only
"""

import re

# ── Layer 1: Dictionary substitutions (lossless, reversible) ─────────

STRUCTURAL = {
    "BEGIN CRUMB": "BC",
    "END CRUMB": "EC",
}

SECTION_MAP = {
    "goal": "g",
    "context": "cx",
    "constraints": "ct",
    "consolidated": "cs",
    "project": "pj",
    "modules": "md",
    "entries": "en",
    "tasks": "tk",
    "identity": "id",
    "permissions": "pm",
    "actions": "ac",
    "verdict": "vd",
    "notes": "nt",
    "dream": "dm",
    "raw": "rw",
    "archived": "ar",
    "summary": "sm",
    "findings": "fd",
    "recommendations": "rc",
}

HEADER_KEY_MAP = {
    "kind": "k",
    "source": "src",
    "title": "t",
    "project": "pj",
    "max_index_tokens": "mit",
    "max_total_tokens": "mtt",
    "dream_pass": "dp",
    "dream_sessions": "ds",
    "agent_name": "an",
    "agent_framework": "af",
    "status": "st",
    "issued": "iss",
    "expires": "exp",
    "metalk": "mt",
}

ABBREV = {
    # Long tech terms → short forms (sorted longest-first at runtime)
    "authentication": "auth",
    "authorization": "authz",
    "configuration": "cfg",
    "implementation": "impl",
    "infrastructure": "infra",
    "documentation": "docs",
    "specification": "spec",
    "asynchronous": "async",
    "synchronous": "sync",
    "architecture": "arch",
    "notification": "notif",
    "application": "app",
    "environment": "env",
    "development": "dev",
    "performance": "perf",
    "distributed": "distrib",
    "dependencies": "deps",
    "requirements": "reqs",
    "deployment": "deploy",
    "integration": "integ",
    "certificate": "cert",
    "credential": "cred",
    "continuous": "cont",
    "expression": "expr",
    "connection": "conn",
    "transaction": "txn",
    "dependency": "dep",
    "repository": "repo",
    "kubernetes": "k8s",
    "middleware": "mw",
    "production": "prod",
    "components": "cmps",
    "properties": "props",
    "parameters": "params",
    "attributes": "attrs",
    "permission": "perm",
    "validation": "val",
    "definition": "defn",
    "execution": "exec",
    "component": "cmp",
    "container": "ctnr",
    "publisher": "pub",
    "subscriber": "sub",
    "variables": "vars",
    "operation": "op",
    "reference": "ref",
    "parameter": "param",
    "attribute": "attr",
    "directory": "dir",
    "interface": "ifc",
    "exception": "exc",
    "postgresql": "pg",
    "typescript": "ts",
    "javascript": "js",
    "template": "tmpl",
    "property": "prop",
    "function": "fn",
    "database": "db",
    "response": "resp",
    "variable": "var",
    "callback": "cb",
    "message": "msg",
    "request": "rq",
    "information": "info",
}

# Build reverse maps
_REV_STRUCTURAL = {v: k for k, v in STRUCTURAL.items()}
_REV_SECTION = {v: k for k, v in SECTION_MAP.items()}
_REV_HEADER_KEY = {v: k for k, v in HEADER_KEY_MAP.items()}
_REV_ABBREV = {v: k for k, v in ABBREV.items()}

# Pre-sort by length descending for longest-match-first
_ABBREV_SORTED = sorted(ABBREV.items(), key=lambda x: len(x[0]), reverse=True)
_REV_ABBREV_SORTED = sorted(_REV_ABBREV.items(), key=lambda x: len(x[0]), reverse=True)


# ── Layer 2: Grammar stripping (lossy) ──────────────────────────────

STRIP_WORDS = frozenset({
    "the", "a", "an", "that", "which", "just", "very", "really",
    "basically", "actually", "currently", "essentially", "simply",
    "furthermore", "moreover", "however", "therefore", "consequently",
    "additionally", "specifically", "particularly", "approximately",
})

PHRASE_REWRITES = {
    "in order to": "to",
    "make sure to": "ensure",
    "as well as": "and",
    "due to the fact that": "because",
    "at this point in time": "now",
    "in the event that": "if",
    "it is necessary to": "must",
    "there is a need to": "must",
    "is able to": "can",
    "be able to": "can",
    "needs to be": "must be",
    "do not": "don't",
    "does not": "doesn't",
    "should not": "shouldn't",
    "is not": "isn't",
    "can not": "can't",
    "cannot": "can't",
    "has been": "was",
    "have been": "were",
    "will need to": "must",
    "you should": "",
    "please note that": "",
    "it is important to note that": "",
    "it should be noted that": "",
}

# Sort phrase rewrites longest-first
_PHRASE_SORTED = sorted(PHRASE_REWRITES.items(), key=lambda x: len(x[0]), reverse=True)


# ── Core encode/decode ──────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _apply_dict_sub(text: str, mapping: list, body_only_start: int = 0) -> str:
    """Apply whole-word substitutions from a sorted mapping list."""
    for long_form, short_form in mapping:
        pattern = re.compile(r'\b' + re.escape(long_form) + r'\b', re.IGNORECASE)
        def _replace(m):
            orig = m.group(0)
            # Preserve capitalization
            if orig[0].isupper() and short_form[0].islower():
                return short_form[0].upper() + short_form[1:]
            return short_form
        text = pattern.sub(_replace, text)
    return text


def _reverse_dict_sub(text: str, mapping: list) -> str:
    """Reverse whole-word substitutions."""
    for short_form, long_form in mapping:
        pattern = re.compile(r'\b' + re.escape(short_form) + r'\b')
        def _replace(m, lf=long_form):
            orig = m.group(0)
            if orig[0].isupper() and lf[0].islower():
                return lf[0].upper() + lf[1:]
            return lf
        text = pattern.sub(_replace, text)
    return text


def _strip_grammar(text: str) -> str:
    """Remove articles, filler words, and rewrite verbose phrases."""
    # Apply phrase rewrites first (longest-first)
    for phrase, replacement in _PHRASE_SORTED:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub(replacement, text)

    # Strip filler words (whole-word, preserve line structure)
    lines = text.split('\n')
    result = []
    for line in lines:
        words = line.split()
        filtered = []
        for w in words:
            # Check stripped version (without punctuation) against strip list
            clean = w.strip('.,;:!?()[]"\'').lower()
            if clean in STRIP_WORDS:
                continue
            filtered.append(w)
        new_line = ' '.join(filtered)
        # Clean up double spaces
        new_line = re.sub(r'  +', ' ', new_line).strip()
        # Fix orphaned leading punctuation
        new_line = re.sub(r'^- +', '- ', new_line)
        result.append(new_line)
    return '\n'.join(result)


def _condense_aggressive(text: str) -> str:
    """Level 3: aggressive sentence condensing."""
    lines = text.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        # Remove trailing periods on bullet points
        if stripped.startswith('-') and stripped.endswith('.'):
            line = line.rstrip()
            if line.endswith('.'):
                line = line[:-1]
        # Remove empty bullet points
        if stripped == '-' or stripped == '- ':
            continue
        result.append(line)

    # Collapse multiple blank lines
    text = '\n'.join(result)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def encode(text: str, level: int = 2, *,
           vowel_min_length: int = 4,
           adaptive_threshold: float = 0.85) -> str:
    """Compress crumb text using MeTalk.

    Args:
        text: Full crumb text (BEGIN CRUMB ... END CRUMB)
        level: 1=dict only (lossless), 2=dict+grammar (lossy),
               3=aggressive, 4=skeleton (vowel-strip),
               5=adaptive (embedding-aware vowel-strip; needs the
               [embeddings] extra, falls back to 4 if unavailable).
        vowel_min_length: Min word length eligible for vowel stripping
                          at levels 4-5. Default 4.
        adaptive_threshold: Cosine similarity floor for level 5 — a
                            line keeps its stripped form only if drift
                            stays under this. Default 0.85.

    Returns:
        MeTalk-encoded crumb text with mt=N header injected.
    """
    lines = text.strip().split('\n')

    # Find the separator
    sep_idx = None
    for i, line in enumerate(lines):
        if line.strip() == '---':
            sep_idx = i
            break

    if sep_idx is None:
        # Not a valid crumb structure, just apply body transforms
        return text

    # Phase A: Structural markers
    if lines[0].strip() == 'BEGIN CRUMB':
        lines[0] = 'BC'
    if lines[-1].strip() == 'END CRUMB':
        lines[-1] = 'EC'

    # Phase B: Header keys (before ---)
    header_lines = []
    mt_injected = False
    for i in range(1, sep_idx):
        line = lines[i]
        if '=' in line:
            key, _, val = line.partition('=')
            key_stripped = key.strip()
            if key_stripped in HEADER_KEY_MAP:
                line = f"{HEADER_KEY_MAP[key_stripped]}={val}"
        header_lines.append(line)

    # Inject mt=N header
    header_lines.append(f"mt={level}")

    # Phase C: Section headers and body (after ---)
    body_lines = []
    for i in range(sep_idx + 1, len(lines) - 1):  # exclude EC
        line = lines[i]
        stripped = line.strip()

        # Section headers
        if stripped.startswith('[') and stripped.endswith(']'):
            section_name = stripped[1:-1].strip().lower()
            if section_name in SECTION_MAP:
                indent = line[:len(line) - len(line.lstrip())]
                line = f"{indent}[{SECTION_MAP[section_name]}]"
            body_lines.append(line)
            continue

        # Body text: apply dictionary substitutions
        line = _apply_dict_sub(line, _ABBREV_SORTED)

        # Layer 2: grammar stripping
        if level >= 2:
            line = _strip_grammar(line)

        body_lines.append(line)

    # Layer 3: aggressive condensing
    if level >= 3:
        body_text = '\n'.join(body_lines)
        body_text = _condense_aggressive(body_text)
        body_lines = body_text.split('\n')

    # Reassemble
    result_lines = [lines[0]]  # BC
    result_lines.extend(header_lines)
    result_lines.append('---')
    result_lines.extend(body_lines)
    result_lines.append(lines[-1])  # EC

    assembled = '\n'.join(result_lines) + '\n'

    # Layers 4 and 5: vowel-strip pass over the body. Done last so that
    # dictionary substitutions (Layer 1) have already canonicalized the
    # most ambiguous tech terms before vowels are removed.
    if level >= 4:
        from cli.vowelstrip import encode_crumb as _vs_encode, adaptive_strip_text, strip_line
        if level >= 5:
            transform = lambda line: adaptive_strip_text(
                line, threshold=adaptive_threshold, min_length=vowel_min_length
            )
        else:
            transform = lambda line: strip_line(line, vowel_min_length)
        assembled = _vs_encode(assembled, min_length=vowel_min_length, transform=transform)

    return assembled


def decode(text: str) -> str:
    """Decode MeTalk-encoded crumb. Reverses Layer 1 (dictionary) only.

    Layers 2-5 are lossy and cannot be fully reversed. Vowel-stripped
    words from L4/L5 pass through unchanged (their consonant skeletons
    don't match the dictionary). The `vs=` header is stripped on decode.
    """
    lines = text.strip().split('\n')

    # Check for mt= header
    mt_level = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('mt='):
            try:
                mt_level = int(stripped.split('=', 1)[1])
            except ValueError:
                pass
            break

    if mt_level is None:
        return text  # Not MeTalk-encoded, passthrough

    # Phase A: Structural markers
    if lines[0].strip() == 'BC':
        lines[0] = 'BEGIN CRUMB'
    if lines[-1].strip() == 'EC':
        lines[-1] = 'END CRUMB'

    # Find separator
    sep_idx = None
    for i, line in enumerate(lines):
        if line.strip() == '---':
            sep_idx = i
            break

    if sep_idx is None:
        return '\n'.join(lines) + '\n'

    # Phase B: Header keys
    header_lines = []
    for i in range(1, sep_idx):
        line = lines[i]
        stripped = line.strip()
        # Skip mt= and vs= MeTalk-internal headers (remove them)
        if stripped.startswith('mt=') or stripped.startswith('vs='):
            continue
        if '=' in line:
            key, _, val = line.partition('=')
            key_stripped = key.strip()
            if key_stripped in _REV_HEADER_KEY:
                line = f"{_REV_HEADER_KEY[key_stripped]}={val}"
        header_lines.append(line)

    # Phase C: Section headers and body
    body_lines = []
    for i in range(sep_idx + 1, len(lines) - 1):
        line = lines[i]
        stripped = line.strip()

        # Section headers
        if stripped.startswith('[') and stripped.endswith(']'):
            section_name = stripped[1:-1].strip().lower()
            if section_name in _REV_SECTION:
                indent = line[:len(line) - len(line.lstrip())]
                line = f"{indent}[{_REV_SECTION[section_name]}]"
            body_lines.append(line)
            continue

        # Reverse dictionary substitutions
        line = _reverse_dict_sub(line, _REV_ABBREV_SORTED)
        body_lines.append(line)

    # Reassemble
    result_lines = [lines[0]]  # BEGIN CRUMB
    result_lines.extend(header_lines)
    result_lines.append('---')
    result_lines.extend(body_lines)
    result_lines.append(lines[-1])  # END CRUMB

    return '\n'.join(result_lines) + '\n'


def compression_stats(original: str, encoded: str) -> dict:
    """Return compression statistics."""
    orig_tokens = _estimate_tokens(original)
    enc_tokens = _estimate_tokens(encoded)
    saved = orig_tokens - enc_tokens
    pct = (saved / orig_tokens * 100) if orig_tokens > 0 else 0
    ratio = orig_tokens / enc_tokens if enc_tokens > 0 else 1
    return {
        "original_tokens": orig_tokens,
        "encoded_tokens": enc_tokens,
        "saved_tokens": saved,
        "pct_saved": round(pct, 1),
        "ratio": round(ratio, 2),
        "original_chars": len(original),
        "encoded_chars": len(encoded),
    }
