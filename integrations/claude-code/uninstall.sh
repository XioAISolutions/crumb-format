#!/usr/bin/env bash
# Uninstall the CRUMB Claude Code integration.
# Removes only what install.sh added. User-saved handoffs are preserved.
set -euo pipefail

CLAUDE_DIR="${HOME}/.claude"
COMMANDS_DIR="${CLAUDE_DIR}/commands"
MCP_FILE="${CLAUDE_DIR}/.mcp.json"

removed=0

for cmd in crumb-export crumb-import; do
    dst="${COMMANDS_DIR}/${cmd}.md"
    if [[ -f "$dst" ]]; then
        rm "$dst"
        echo "  -  ${cmd} slash command"
        removed=$((removed+1))
    fi
done

if [[ -f "$MCP_FILE" ]] && command -v jq >/dev/null 2>&1; then
    if jq -e '.mcpServers.crumb' "$MCP_FILE" >/dev/null 2>&1; then
        tmp=$(mktemp)
        jq 'del(.mcpServers.crumb)' "$MCP_FILE" > "$tmp"
        mv "$tmp" "$MCP_FILE"
        echo "  -  crumb MCP server entry"
        removed=$((removed+1))
    fi
fi

# We do NOT touch ./CLAUDE.md — the user owns that file. The
# install.sh appended block is recognizable but removing it would
# also catch hand-edited copies. Leave it.

if [[ $removed -eq 0 ]]; then
    echo "Nothing to remove."
else
    echo "==> Uninstalled $removed item(s)."
    echo "    Note: any 'crumb it' block you may have added to ./CLAUDE.md"
    echo "    was left in place — remove it manually if desired."
fi
