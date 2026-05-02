#!/usr/bin/env bash
# Uninstall the CRUMB Claude Code integration.
# Removes only what install.sh added. User-saved handoffs are preserved.
set -euo pipefail

CLAUDE_DIR="${HOME}/.claude"
COMMANDS_DIR="${CLAUDE_DIR}/commands"
MCP_FILE="${CLAUDE_DIR}/.mcp.json"

removed=0
preserved=0

# Marker that install.sh writes at the top of every file it manages.
# Must match the literal in install.sh exactly.
CRUMB_MANAGED_MARKER="<!-- managed by crumb-format/integrations/claude-code/install.sh - safe to overwrite -->"

for cmd in crumb-export crumb-import; do
    dst="${COMMANDS_DIR}/${cmd}.md"
    if [[ ! -f "$dst" ]]; then
        continue
    fi
    # Only delete files that carry our marker. install.sh writes the
    # marker at the top of every file it manages and backs up any
    # pre-existing user-modified file before overwriting. So a file
    # without the marker either pre-dates install.sh or is a hand-
    # written custom command that happens to share the name —
    # either way, deleting it would violate the contract that
    # uninstall removes "only what install.sh added". (Codex P2.)
    if grep -qF "$CRUMB_MANAGED_MARKER" "$dst" 2>/dev/null; then
        rm "$dst"
        echo "  -  ${cmd} slash command"
        removed=$((removed+1))
    else
        echo "  !  ${cmd} appears user-managed (no install.sh marker); preserved"
        preserved=$((preserved+1))
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
    echo "    Note: any 'crumb it' block you may have added to ./CLAUDE.md"
    echo "    was left in place — remove it manually if desired."
fi
