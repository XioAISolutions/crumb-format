#!/usr/bin/env bash
# Uninstall the CRUMB Cursor integration.
# Removes only what install.sh added. User-saved rules are preserved.
set -euo pipefail

CURSOR_DIR="${HOME}/.cursor"
MCP_FILE="${CURSOR_DIR}/mcp.json"
PROJECT_RULES_DIR="./.cursor/rules"

removed=0
preserved=0

CRUMB_MANAGED_MARKER="<!-- managed by crumb-format/integrations/cursor/install.sh - safe to overwrite -->"

# Project-level rules.
for rule in crumb-export crumb-import crumb-it; do
    dst="${PROJECT_RULES_DIR}/${rule}.mdc"
    if [[ ! -f "$dst" ]]; then
        continue
    fi
    if grep -qF "$CRUMB_MANAGED_MARKER" "$dst" 2>/dev/null; then
        rm "$dst"
        echo "  -  ${rule}.mdc rule"
        removed=$((removed+1))
    else
        echo "  !  ${rule}.mdc appears user-managed (no install.sh marker); preserved"
        preserved=$((preserved+1))
    fi
done

# Global MCP entry.
if [[ -f "$MCP_FILE" ]] && command -v jq >/dev/null 2>&1; then
    if jq -e '.mcpServers.crumb' "$MCP_FILE" >/dev/null 2>&1; then
        if jq -e '.mcpServers.crumb._managed_by == "crumb-format/integrations/cursor/install.sh"' "$MCP_FILE" >/dev/null 2>&1; then
            tmp=$(mktemp)
            jq 'del(.mcpServers.crumb)' "$MCP_FILE" > "$tmp"
            mv "$tmp" "$MCP_FILE"
            echo "  -  crumb MCP server entry"
            removed=$((removed+1))
        else
            echo "  !  crumb MCP server entry has no install.sh marker; preserved"
            preserved=$((preserved+1))
        fi
    fi
fi

if [[ $removed -eq 0 && $preserved -eq 0 ]]; then
    echo "Nothing to remove."
else
    if [[ $removed -gt 0 ]]; then
        echo "==> Uninstalled $removed item(s)."
    fi
    if [[ $preserved -gt 0 ]]; then
        echo "==> Preserved $preserved file(s) without the install.sh marker."
        echo "    Delete them yourself if you want them gone."
    fi
fi
