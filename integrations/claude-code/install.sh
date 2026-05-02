#!/usr/bin/env bash
# Install the CRUMB Claude Code integration.
#
# Idempotent. Safe to run more than once.
# Removes nothing the user added.
set -euo pipefail

CLAUDE_DIR="${HOME}/.claude"
COMMANDS_DIR="${CLAUDE_DIR}/commands"
MCP_FILE="${CLAUDE_DIR}/.mcp.json"

# Resolve this script's directory so the source paths work regardless
# of where the user invokes it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
for cmd in crumb-export crumb-import; do
    src="${SCRIPT_DIR}/commands/${cmd}.md"
    dst="${COMMANDS_DIR}/${cmd}.md"
    if [[ -f "$dst" ]] && cmp -s "$src" "$dst"; then
        echo "  =  ${cmd} already up to date"
    else
        cp "$src" "$dst"
        echo "  +  ${cmd}"
    fi
done

# ── 2. MCP server registration ────────────────────────────────────
# We merge into existing .mcp.json instead of overwriting. Merge logic
# is naive (jq required) — if jq isn't present, print the snippet and
# let the user paste it.
echo "==> Registering MCP server in ${MCP_FILE}"
MCP_TEMPLATE="${SCRIPT_DIR}/mcp.json.template"
CRUMB_INSTALL_PATH="$(python3 -c 'import crumb_cli, os; print(os.path.dirname(os.path.dirname(crumb_cli.__file__)))' 2>/dev/null || echo "")"

if [[ -z "$CRUMB_INSTALL_PATH" ]]; then
    # Fallback: use the source path of this script's parent.
    CRUMB_INSTALL_PATH="$(dirname "$SCRIPT_DIR")"
fi

if command -v jq >/dev/null 2>&1; then
    if [[ ! -f "$MCP_FILE" ]]; then
        echo '{"mcpServers": {}}' > "$MCP_FILE"
    fi
    SERVER_JSON=$(sed "s|__CRUMB_INSTALL_PATH__|${CRUMB_INSTALL_PATH}|g" "$MCP_TEMPLATE")
    # Merge .mcpServers.crumb from the template into the existing file.
    tmp=$(mktemp)
    jq --argjson new "$(echo "$SERVER_JSON" | jq '.mcpServers')" \
        '.mcpServers = (.mcpServers // {}) * $new' \
        "$MCP_FILE" > "$tmp"
    mv "$tmp" "$MCP_FILE"
    echo "  +  crumb MCP server registered"
else
    echo "  !  jq not found; skipping automatic MCP merge."
    echo "     Manually merge this into ${MCP_FILE}:"
    echo "     ----"
    sed "s|__CRUMB_INSTALL_PATH__|${CRUMB_INSTALL_PATH}|g" "$MCP_TEMPLATE" | sed 's/^/     /'
    echo "     ----"
fi

# ── 3. CLAUDE.md verbal-trigger block (optional, prompted) ─────────
if [[ -f "./CLAUDE.md" ]]; then
    if grep -q "When I say \"crumb it\"" ./CLAUDE.md 2>/dev/null; then
        echo "==> ./CLAUDE.md already has the 'crumb it' block — skipping"
    else
        echo "==> ./CLAUDE.md exists. Append the 'crumb it' verbal-trigger block? [y/N]"
        read -r yn
        if [[ "$yn" =~ ^[Yy] ]]; then
            {
                echo
                echo "<!-- Added by crumb-format/integrations/claude-code/install.sh -->"
                cat "${SCRIPT_DIR}/CLAUDE.md.template"
            } >> ./CLAUDE.md
            echo "  +  appended"
        else
            echo "  =  skipped (you can append manually from ${SCRIPT_DIR}/CLAUDE.md.template)"
        fi
    fi
fi

# ── Done ───────────────────────────────────────────────────────────
echo
echo "==> Installed."
echo "    Try /crumb-export in your next Claude Code session."
echo "    Or say 'crumb it' if you appended the verbal trigger block to CLAUDE.md."
