#!/usr/bin/env bash
# Install the CRUMB Claude Code integration.
#
# Idempotent. Safe to run more than once.
# Removes nothing the user added.
set -euo pipefail

CLAUDE_DIR="${HOME}/.claude"
COMMANDS_DIR="${CLAUDE_DIR}/commands"
MCP_FILE="${CLAUDE_DIR}/.mcp.json"

# The asset root contains commands/, mcp.json.template, CLAUDE.md.template.
# It's normally <repo>/integrations/claude-code (the dir containing this
# script). But the README advertises `bash <(curl ...)` direct install,
# which runs from /dev/fd/* — there are no sibling files there. In that
# case (or when BASH_SOURCE points at a tempfile / process-substitution),
# fetch the assets from GitHub raw to a tmp dir and use that as ASSETS.
# (Codex P1 caught this: the previous version assumed SCRIPT_DIR worked
# regardless of invocation mode and aborted on cp.)
SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" 2>/dev/null && pwd || echo "")"

CRUMB_INSTALL_BRANCH="${CRUMB_INSTALL_BRANCH:-main}"
ASSETS_BASE_URL="${CRUMB_ASSETS_BASE_URL:-https://raw.githubusercontent.com/XioAISolutions/crumb-format/${CRUMB_INSTALL_BRANCH}/integrations/claude-code}"

if [[ -n "$SCRIPT_DIR" && -f "${SCRIPT_DIR}/commands/crumb-export.md" ]]; then
    # Local checkout — use the script's own directory.
    ASSETS="$SCRIPT_DIR"
else
    # Process-substitution / curl-pipe / oddly-relocated install. Fetch
    # the assets to a tmp dir.
    if ! command -v curl >/dev/null 2>&1; then
        echo "ERROR: cannot locate integration assets locally and curl isn't available to fetch them." >&2
        echo "       Either clone the repo and run integrations/claude-code/install.sh from there," >&2
        echo "       or install curl and re-run." >&2
        exit 1
    fi
    ASSETS="$(mktemp -d)"
    trap 'rm -rf "$ASSETS"' EXIT
    echo "==> Fetching install assets from ${ASSETS_BASE_URL}"
    mkdir -p "${ASSETS}/commands"
    for path in commands/crumb-export.md commands/crumb-import.md mcp.json.template CLAUDE.md.template; do
        if ! curl -fsSL "${ASSETS_BASE_URL}/${path}" -o "${ASSETS}/${path}"; then
            echo "ERROR: failed to fetch ${path} from ${ASSETS_BASE_URL}" >&2
            exit 1
        fi
    done
fi

# ── Prereq: crumb-format must be on PATH ──────────────────────────
if ! command -v crumb >/dev/null 2>&1; then
    echo "ERROR: crumb-format is not installed (or 'crumb' is not on PATH)."  >&2
    echo "       Install with: pip install crumb-format"                       >&2
    exit 1
fi

# ── Prereq: ~/.claude/ exists (Claude Code is set up) ─────────────
if [[ ! -d "$CLAUDE_DIR" ]]; then
    echo "ERROR: ${CLAUDE_DIR} does not exist."                                >&2
    echo "       Open Claude Code at least once to create it, then re-run."    >&2
    exit 1
fi

mkdir -p "$COMMANDS_DIR"

# ── 1. Slash commands ──────────────────────────────────────────────
echo "==> Installing slash commands to ${COMMANDS_DIR}/"
CRUMB_MANAGED_MARKER="<!-- managed by crumb-format/integrations/claude-code/install.sh - safe to overwrite -->"

for cmd in crumb-export crumb-import; do
    src="${ASSETS}/commands/${cmd}.md"
    dst="${COMMANDS_DIR}/${cmd}.md"
    # Compare against the EXACT bytes we'd write (marker + src body).
    # Comparing against $src alone would always trigger "differs"
    # because dst has the marker prepended.
    if [[ -f "$dst" ]] && cmp -s <(printf '%s\n' "$CRUMB_MANAGED_MARKER"; cat "$src") "$dst"; then
        echo "  =  ${cmd} already up to date"
        continue
    fi
    if [[ -f "$dst" ]] && ! grep -qF "$CRUMB_MANAGED_MARKER" "$dst" 2>/dev/null; then
        # File exists, differs from ours, and doesn't carry the
        # "managed by us" marker — assume the user customized it.
        # Back it up before overwriting. (Codex P2: previous version
        # silently clobbered user edits.)
        backup="${dst}.bak.$(date +%Y%m%d-%H%M%S)"
        cp "$dst" "$backup"
        echo "  !  ${cmd} appears user-modified; backed up to $(basename "$backup")"
    fi
    # Prepend the marker comment to the installed file so subsequent
    # runs recognize it as ours and overwrite cleanly.
    {
        echo "$CRUMB_MANAGED_MARKER"
        cat "$src"
    } > "$dst"
    echo "  +  ${cmd}"
done

# ── 2. MCP server registration ────────────────────────────────────
# We merge into existing .mcp.json instead of overwriting. Merge logic
# is naive (jq required) — if jq isn't present, print the snippet and
# let the user paste it.
echo "==> Registering MCP server in ${MCP_FILE}"
MCP_TEMPLATE="${ASSETS}/mcp.json.template"

# Resolution strategy for the install path of mcp/server.py, in order:
#
#   1. Ask `crumb` itself which interpreter it uses, then import via
#      that interpreter. Handles pipx and venv installs where system
#      python3 doesn't have crumb_cli on its sys.path. (Codex P1.)
#   2. Fall back to `pip show crumb-format` parsing. Works whenever pip
#      can see the package, even without crumb_cli being importable
#      from system python3.
#   3. Fall back to system `python3 -c 'import crumb_cli'`. Works in
#      typical pip-installed environments.
#   4. Fall back to walking up from $SCRIPT_DIR. Only valid when
#      running from a real local checkout (not curl-pipe).
#   5. Existence-check on mcp/server.py catches every miss; loud error
#      if all four strategies produced a wrong path.

CRUMB_INSTALL_PATH=""
CRUMB_PYTHON=""  # path to the interpreter the MCP server should use

# Strategy 1: invoke crumb's own interpreter.
if command -v crumb >/dev/null 2>&1; then
    CRUMB_BIN="$(command -v crumb)"
    # Read the shebang from the entry-point script. Shebang is line 1.
    CRUMB_PY="$(head -1 "$CRUMB_BIN" 2>/dev/null | sed -n 's|^#!\(.*\)|\1|p')"
    if [[ -n "$CRUMB_PY" && -x "$CRUMB_PY" ]]; then
        CRUMB_INSTALL_PATH="$("$CRUMB_PY" -c 'import crumb_cli, os; print(os.path.dirname(crumb_cli.__file__))' 2>/dev/null || echo "")"
        if [[ -n "$CRUMB_INSTALL_PATH" ]]; then
            CRUMB_PYTHON="$CRUMB_PY"
        fi
    fi
fi

# Strategy 2: pip show crumb-format.
if [[ -z "$CRUMB_INSTALL_PATH" ]] && command -v pip >/dev/null 2>&1; then
    PIP_LOC="$(pip show crumb-format 2>/dev/null | sed -n 's|^Location: ||p' | head -1)"
    if [[ -n "$PIP_LOC" && -d "${PIP_LOC}" ]]; then
        CRUMB_INSTALL_PATH="$PIP_LOC"
        # If pip itself is on the same interpreter, use its python.
        # Resolve via `pip --version` which prints the python it's bound to.
        PIP_PY="$(python3 -c 'import sys; print(sys.executable)' 2>/dev/null || echo "")"
        if [[ -n "$PIP_PY" && -x "$PIP_PY" ]]; then
            CRUMB_PYTHON="$PIP_PY"
        fi
    fi
fi

# Strategy 3: system python3.
if [[ -z "$CRUMB_INSTALL_PATH" ]]; then
    CRUMB_INSTALL_PATH="$(python3 -c 'import crumb_cli, os; print(os.path.dirname(crumb_cli.__file__))' 2>/dev/null || echo "")"
    if [[ -n "$CRUMB_INSTALL_PATH" ]]; then
        CRUMB_PYTHON="$(command -v python3)"
    fi
fi

# Strategy 4: walk up from a real script dir (only valid when not curl-piped).
if [[ -z "$CRUMB_INSTALL_PATH" && -n "$SCRIPT_DIR" && -d "$SCRIPT_DIR" ]]; then
    candidate="$(dirname "$(dirname "$SCRIPT_DIR")")"
    if [[ -f "${candidate}/mcp/server.py" ]]; then
        CRUMB_INSTALL_PATH="$candidate"
        # No interpreter signal here; fall through to the default below.
    fi
fi

# Sanity check: the path must actually contain mcp/server.py. Without
# this, a wrong CRUMB_INSTALL_PATH silently writes a broken .mcp.json
# and the user only finds out when Claude Code can't start the server.
if [[ ! -f "${CRUMB_INSTALL_PATH}/mcp/server.py" ]]; then
    echo "ERROR: could not locate mcp/server.py under ${CRUMB_INSTALL_PATH}." >&2
    echo "       Confirm crumb-format is installed and reachable, then re-run." >&2
    exit 1
fi

# If no interpreter was identified (Strategy 4 only), fall back to
# system python3. We'd rather record a path that's close-enough and
# fail visibly than refuse to install. (Codex P1: hardcoded `python3`
# in the template broke pipx/venv users; now it's resolved.)
if [[ -z "$CRUMB_PYTHON" ]]; then
    CRUMB_PYTHON="$(command -v python3 || echo python3)"
fi

if command -v jq >/dev/null 2>&1; then
    if [[ ! -f "$MCP_FILE" ]]; then
        echo '{"mcpServers": {}}' > "$MCP_FILE"
    fi
    SERVER_JSON=$(sed -e "s|__CRUMB_INSTALL_PATH__|${CRUMB_INSTALL_PATH}|g" \
                      -e "s|__CRUMB_PYTHON__|${CRUMB_PYTHON}|g" \
                      "$MCP_TEMPLATE")
    # Merge .mcpServers.crumb from the template into the existing file.
    tmp=$(mktemp)
    jq --argjson new "$(echo "$SERVER_JSON" | jq '.mcpServers')" \
        '.mcpServers = (.mcpServers // {}) * $new' \
        "$MCP_FILE" > "$tmp"
    mv "$tmp" "$MCP_FILE"
    echo "  +  crumb MCP server registered (interpreter: ${CRUMB_PYTHON})"
else
    echo "  !  jq not found; skipping automatic MCP merge."
    echo "     Manually merge this into ${MCP_FILE}:"
    echo "     ----"
    sed -e "s|__CRUMB_INSTALL_PATH__|${CRUMB_INSTALL_PATH}|g" \
        -e "s|__CRUMB_PYTHON__|${CRUMB_PYTHON}|g" \
        "$MCP_TEMPLATE" | sed 's/^/     /'
    echo "     ----"
fi

# ── 3. CLAUDE.md verbal-trigger block (optional, prompted) ─────────
if [[ -f "./CLAUDE.md" ]]; then
    # Use the HTML-comment marker the install appends, not the prose,
    # as the duplicate-detection signal. The prose can drift across
    # template revisions (and previously DID — the v1.0.0 template
    # uses `**"crumb it"**` with bold asterisks while the prior check
    # looked for `"crumb it"` without). The marker is stable.
    if grep -qF "<!-- Added by crumb-format/integrations/claude-code/install.sh -->" ./CLAUDE.md 2>/dev/null; then
        echo "==> ./CLAUDE.md already has the 'crumb it' block — skipping"
    elif [[ ! -t 0 ]]; then
        # Non-interactive stdin (CI, redirected, curl-piped through sh).
        # Don't prompt — silently skip and tell the user how to do it
        # manually. Otherwise `read` would either block indefinitely
        # or hit EOF and (with set -e) abort the install AFTER the
        # commands+MCP have already been written, leaving a partial
        # install with exit 1. (Codex P1.)
        echo "==> ./CLAUDE.md exists; not prompting (non-interactive stdin)."
        echo "    To append the 'crumb it' verbal-trigger block manually:"
        echo "    cat ${ASSETS}/CLAUDE.md.template >> ./CLAUDE.md"
    else
        echo "==> ./CLAUDE.md exists. Append the 'crumb it' verbal-trigger block? [y/N]"
        # `|| yn=""` defends against EOF on stdin if it slipped past
        # the -t 0 check (e.g. tty closed mid-prompt).
        read -r yn || yn=""
        if [[ "$yn" =~ ^[Yy] ]]; then
            {
                echo
                echo "<!-- Added by crumb-format/integrations/claude-code/install.sh -->"
                cat "${ASSETS}/CLAUDE.md.template"
            } >> ./CLAUDE.md
            echo "  +  appended"
        else
            echo "  =  skipped (you can append manually from ${ASSETS}/CLAUDE.md.template)"
        fi
    fi
fi

# ── Done ───────────────────────────────────────────────────────────
echo
echo "==> Installed."
echo "    Try /crumb-export in your next Claude Code session."
echo "    Or say 'crumb it' if you appended the verbal trigger block to CLAUDE.md."
