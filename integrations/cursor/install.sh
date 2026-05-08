#!/usr/bin/env bash
# Install the CRUMB Cursor integration.
#
# Idempotent. Safe to run more than once.
# Removes nothing the user added.
set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=1 ;;
        --help|-h)
            cat <<EOF
Usage: install.sh [--dry-run]

Installs the CRUMB Cursor integration:
  - Global MCP server entry at ~/.cursor/mcp.json
  - Optional rule files copied into ./.cursor/rules/ (project-level)

--dry-run    Show what would change without writing anything.
EOF
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

CURSOR_DIR="${HOME}/.cursor"
MCP_FILE="${CURSOR_DIR}/mcp.json"
PROJECT_RULES_DIR="./.cursor/rules"

# The asset root contains rules/, mcp.json.template.
SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" 2>/dev/null && pwd || echo "")"

CRUMB_INSTALL_BRANCH="${CRUMB_INSTALL_BRANCH:-main}"
ASSETS_BASE_URL="${CRUMB_ASSETS_BASE_URL:-https://raw.githubusercontent.com/XioAISolutions/crumb-format/${CRUMB_INSTALL_BRANCH}/integrations/cursor}"

if [[ -n "$SCRIPT_DIR" && -f "${SCRIPT_DIR}/rules/crumb-export.mdc" ]]; then
    ASSETS="$SCRIPT_DIR"
else
    if ! command -v curl >/dev/null 2>&1; then
        echo "ERROR: cannot locate integration assets locally and curl isn't available to fetch them." >&2
        echo "       Either clone the repo and run integrations/cursor/install.sh from there," >&2
        echo "       or install curl and re-run." >&2
        exit 1
    fi
    ASSETS="$(mktemp -d)"
    trap 'rm -rf "$ASSETS"' EXIT
    echo "==> Fetching install assets from ${ASSETS_BASE_URL}"
    mkdir -p "${ASSETS}/rules"
    for path in rules/crumb-export.mdc rules/crumb-import.mdc rules/crumb-it.mdc mcp.json.template; do
        if ! curl -fsSL "${ASSETS_BASE_URL}/${path}" -o "${ASSETS}/${path}"; then
            echo "ERROR: failed to fetch ${path} from ${ASSETS_BASE_URL}" >&2
            exit 1
        fi
    done
fi

# ── Helpers ───────────────────────────────────────────────────────
# Run a shell command unless --dry-run is set, in which case just print it.
do_or_show() {
    if (( DRY_RUN )); then
        printf '   [dry-run] %s\n' "$*"
    else
        eval "$@"
    fi
}

if (( DRY_RUN )); then
    echo "==> DRY RUN — no files will be written."
fi

# ── Prereq: crumb-format must be on PATH ──────────────────────────
if ! command -v crumb >/dev/null 2>&1; then
    echo "ERROR: crumb-format is not installed (or 'crumb' is not on PATH)."  >&2
    echo "       Install with: pip install crumb-format"                       >&2
    exit 1
fi

# ── Resolve the install path of mcp/server.py and the interpreter ──
# Same 4-tier strategy as the Claude Code installer. See
# integrations/claude-code/install.sh for the rationale.
CRUMB_INSTALL_PATH=""
CRUMB_PYTHON=""

if command -v crumb >/dev/null 2>&1; then
    CRUMB_BIN="$(command -v crumb)"
    CRUMB_PY="$(head -1 "$CRUMB_BIN" 2>/dev/null | sed -n 's|^#!\(.*\)|\1|p')"
    if [[ -n "$CRUMB_PY" && -x "$CRUMB_PY" ]]; then
        CRUMB_INSTALL_PATH="$("$CRUMB_PY" -c 'import crumb_cli, os; print(os.path.dirname(crumb_cli.__file__))' 2>/dev/null || echo "")"
        if [[ -n "$CRUMB_INSTALL_PATH" ]]; then
            CRUMB_PYTHON="$CRUMB_PY"
        fi
    fi
fi

if [[ -z "$CRUMB_INSTALL_PATH" ]] && command -v pip >/dev/null 2>&1; then
    PIP_LOC="$(pip show crumb-format 2>/dev/null | sed -n 's|^Location: ||p' | head -1)"
    if [[ -n "$PIP_LOC" && -d "${PIP_LOC}" ]]; then
        CRUMB_INSTALL_PATH="$PIP_LOC"
        PIP_PY="$(python3 -c 'import sys; print(sys.executable)' 2>/dev/null || echo "")"
        if [[ -n "$PIP_PY" && -x "$PIP_PY" ]]; then
            CRUMB_PYTHON="$PIP_PY"
        fi
    fi
fi

if [[ -z "$CRUMB_INSTALL_PATH" ]]; then
    CRUMB_INSTALL_PATH="$(python3 -c 'import crumb_cli, os; print(os.path.dirname(crumb_cli.__file__))' 2>/dev/null || echo "")"
    if [[ -n "$CRUMB_INSTALL_PATH" ]]; then
        CRUMB_PYTHON="$(command -v python3)"
    fi
fi

if [[ -z "$CRUMB_INSTALL_PATH" && -n "$SCRIPT_DIR" && -d "$SCRIPT_DIR" ]]; then
    candidate="$(dirname "$(dirname "$SCRIPT_DIR")")"
    if [[ -f "${candidate}/mcp/server.py" ]]; then
        CRUMB_INSTALL_PATH="$candidate"
    fi
fi

if [[ ! -f "${CRUMB_INSTALL_PATH}/mcp/server.py" ]]; then
    echo "ERROR: could not locate mcp/server.py under ${CRUMB_INSTALL_PATH}." >&2
    echo "       Confirm crumb-format is installed and reachable, then re-run." >&2
    exit 1
fi

if [[ -z "$CRUMB_PYTHON" ]]; then
    CRUMB_PYTHON="$(command -v python3 || echo python3)"
fi

# ── 1. Global MCP server registration ─────────────────────────────
echo "==> Registering MCP server in ${MCP_FILE}"
if (( DRY_RUN )); then
    echo "   [dry-run] mkdir -p ${CURSOR_DIR}"
else
    mkdir -p "$CURSOR_DIR"
fi
MCP_TEMPLATE="${ASSETS}/mcp.json.template"

if command -v jq >/dev/null 2>&1; then
    if [[ ! -f "$MCP_FILE" ]]; then
        if (( DRY_RUN )); then
            echo "   [dry-run] would create ${MCP_FILE} with {}"
        else
            echo '{"mcpServers": {}}' > "$MCP_FILE"
        fi
    fi
    SERVER_JSON=$(sed -e "s|__CRUMB_INSTALL_PATH__|${CRUMB_INSTALL_PATH}|g" \
                      -e "s|__CRUMB_PYTHON__|${CRUMB_PYTHON}|g" \
                      "$MCP_TEMPLATE")
    if (( DRY_RUN )); then
        echo "   [dry-run] would merge into ${MCP_FILE}:"
        echo "$SERVER_JSON" | sed 's/^/   [dry-run]   /'
    else
        # Merge .mcpServers.crumb from the template into the existing file.
        # If the file existed but lacks a top-level mcpServers, jq's
        # `(.mcpServers // {}) * $new` handles the null-init.
        tmp=$(mktemp)
        jq --argjson new "$(echo "$SERVER_JSON" | jq '.mcpServers')" \
            '.mcpServers = (.mcpServers // {}) * $new' \
            "$MCP_FILE" > "$tmp"
        mv "$tmp" "$MCP_FILE"
    fi
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

# ── 2. Project rules (optional, prompted) ─────────────────────────
# Cursor's rule files live per-project at ./.cursor/rules/*.mdc. We
# only install them if the user is currently in a directory where they
# want them (CWD has either a .cursor/ or the user explicitly opts in).
echo "==> Project rules"
CRUMB_MANAGED_MARKER="<!-- managed by crumb-format/integrations/cursor/install.sh - safe to overwrite -->"

install_rules() {
    if (( DRY_RUN )); then
        echo "   [dry-run] mkdir -p ${PROJECT_RULES_DIR}"
    else
        mkdir -p "$PROJECT_RULES_DIR"
    fi
    for rule in crumb-export crumb-import crumb-it; do
        src="${ASSETS}/rules/${rule}.mdc"
        dst="${PROJECT_RULES_DIR}/${rule}.mdc"
        if [[ -f "$dst" ]] && cmp -s <(printf '%s\n' "$CRUMB_MANAGED_MARKER"; cat "$src") "$dst"; then
            echo "  =  ${rule} already up to date"
            continue
        fi
        if [[ -f "$dst" ]] && ! grep -qF "$CRUMB_MANAGED_MARKER" "$dst" 2>/dev/null; then
            backup="${dst}.bak.$(date +%Y%m%d-%H%M%S)"
            if (( DRY_RUN )); then
                echo "   [dry-run] would back up ${dst} → $(basename "$backup")"
            else
                cp "$dst" "$backup"
            fi
            echo "  !  ${rule} appears user-modified; would back up to $(basename "$backup")"
        fi
        if (( DRY_RUN )); then
            echo "   [dry-run] would write ${dst} (with marker)"
        else
            {
                echo "$CRUMB_MANAGED_MARKER"
                cat "$src"
            } > "$dst"
        fi
        echo "  +  ${rule}"
    done
}

if [[ -d "./.cursor" ]]; then
    install_rules
elif [[ ! -t 0 ]]; then
    echo "   ./.cursor not found; not prompting (non-interactive stdin)."
    echo "   To install rules into the current project later, run:"
    echo "     mkdir -p ./.cursor/rules && cp ${ASSETS}/rules/*.mdc ./.cursor/rules/"
else
    echo "   ./.cursor not found in this directory."
    echo "   Install rule templates into ./.cursor/rules/? [y/N]"
    read -r yn || yn=""
    if [[ "$yn" =~ ^[Yy] ]]; then
        install_rules
    else
        echo "  =  skipped (you can copy them later from ${ASSETS}/rules/)"
    fi
fi

# ── Done ───────────────────────────────────────────────────────────
echo
if (( DRY_RUN )); then
    echo "==> Dry run complete. Re-run without --dry-run to apply."
else
    echo "==> Installed."
    echo "    Global MCP server is registered. Open Cursor; the 'crumb' tools will be available."
    echo "    If you installed project rules, mention 'crumb it' in chat to trigger an export."
fi
