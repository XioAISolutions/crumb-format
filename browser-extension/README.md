# CRUMB Browser Extension

A Chrome extension that lets you right-click any AI chat conversation and generate a structured CRUMB handoff document, copied straight to your clipboard.

## Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Select the `browser-extension/` directory from this repository
5. The CRUMB extension icon will appear in your toolbar

## Usage

1. Select text in any AI chat (ChatGPT, Claude, etc.)
2. Right-click the selection
3. Click **Crumb it**
4. The structured CRUMB handoff is copied to your clipboard
5. Paste it into your next AI session, a document, or a ticket

## What it extracts

- **Goal** — inferred from the first meaningful sentence
- **Context** — conversation summary with language and snippet counts
- **Constraints** — "must not", "cannot", "avoid", etc.
- **Decisions** — "decided to", "going with", "let's use", etc.
- **Action items** — TODO, FIXME, "next step", "need to", etc.
- **Code snippets** — all fenced code blocks with language tags

## Permissions

- `contextMenus` — adds the "Crumb it" right-click menu item
- `clipboardWrite` — copies the generated crumb to your clipboard
- `activeTab` — reads the selected text on the current page

## Icon

Place an `icon.png` file (128x128 recommended) in this directory to use a custom icon. A placeholder reference is included in the manifest.
