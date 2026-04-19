# CRUMB Browser Extension

A Chrome extension that adds a one-click **Copy as CRUMB** handoff flow to supported AI chat interfaces and can also convert selected text into a portable CRUMB block.

## Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Select the `browser-extension/` directory from this repository
5. The CRUMB extension icon will appear in your toolbar

## Usage

The extension now supports two capture paths.

### One-click chat capture

1. Open a supported chat in ChatGPT, Claude, or Gemini.
2. Click the floating **Copy as CRUMB** button.
3. The extension scrapes the most recent visible user and assistant exchanges.
4. A `kind=log` CRUMB v1.2 block is generated locally and copied to your clipboard.
5. Paste it into your next AI session, a document, or a ticket.

### Selection-based capture

1. Select text in a supported AI chat page.
2. Right-click the selection.
3. Click **Copy as CRUMB**.
4. The selected text is converted into a `kind=task` CRUMB and copied to your clipboard.

## Supported sites

- `https://chatgpt.com/*`
- `https://chat.openai.com/*`
- `https://claude.ai/*`
- `https://gemini.google.com/*`

## What it extracts

For one-click chat capture, the extension builds a compact `kind=log` handoff from the latest visible exchanges on the page.

For selection-based capture, it extracts:

- **Goal** — inferred from the first meaningful sentence
- **Context** — the most relevant lines from the selected excerpt
- **Constraints** — phrases such as "must not", "cannot", and "avoid"
- **Handoff** — `[handoff]` bullets (v1.2) inferred from TODO / should / need-to / implement / add phrases, so the next AI has an explicit action list

## Permissions

- `contextMenus` — adds the **Copy as CRUMB** right-click menu item
- `clipboardWrite` — copies the generated crumb to your clipboard
- `activeTab` — allows page-level capture from the current supported chat tab

## Icon

Place an `icon.png` file (128x128 recommended) in this directory to use a custom icon. A placeholder reference is included in the manifest.
