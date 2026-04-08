# CRUMB Studio

CRUMB Studio is the desktop UI for the existing CRUMB Python engine.

It turns the CLI workflow into a visual before/after product:

- left pane: raw source context
- right pane: structured CRUMB output
- top bar: mode selector and run action
- footer bar: copy, save, export, clear, and history

## Architecture

CRUMB Studio keeps the current Python engine as the source of truth.

```text
raw text input
  -> Studio UI (HTML/CSS/JS in pywebview)
  -> Studio bridge (studio/app.py)
  -> existing CRUMB engine (cli/crumb.py)
  -> rendered .crumb / markdown / json output
```

Key points:

- The desktop shell is built with `pywebview` for a lightweight cross-platform MVP.
- The UI is static web-style assets inside `studio/static/`.
- The bridge layer lives in `studio/app.py`.
- Generation logic lives in `studio/engine.py`.
- Parsing, rendering, markdown export, JSON export, token estimation, and clipboard behavior still come from the existing Python engine in [`cli/crumb.py`](../cli/crumb.py).
- The CLI remains intact; Studio is exposed as an additional command: `crumb studio`.

## Folder Layout

```text
studio/
  app.py                # desktop bridge + pywebview launcher
  engine.py             # Studio-specific transformation layer
  history.py            # recent output persistence
  static/
    index.html          # split-pane interface
    app.css             # UI styling
    app.js              # frontend bridge and interaction logic
  packaging/
    build_studio.py     # PyInstaller helper
```

## Run Locally

Install the package with the optional Studio dependencies:

```bash
python -m pip install -e ".[studio]"
```

Launch the desktop app:

```bash
crumb studio
```

Useful local checks:

```bash
# verify the Studio pipeline without opening a window
crumb studio --smoke-test

# launch with webview debug tools enabled
crumb studio --debug
```

## Build a Desktop Binary

Install the build dependency:

```bash
python -m pip install pyinstaller
```

Build the desktop app:

```bash
python studio/packaging/build_studio.py
```

This produces a packaged desktop build under `dist/CRUMB-Studio/` on macOS/Linux or the platform equivalent on Windows.

## MVP Scope

The current desktop MVP supports:

- paste raw text into the left pane
- choose `task`, `mem`, `map`, `log`, or `todo`
- generate structured output with one action
- copy the CRUMB output
- save a `.crumb` file
- export markdown, json, or plain text
- reload recent generations from local history
- show clear errors instead of crashing on invalid input

## Roadmap

### 1. Desktop MVP

- stabilize the current two-pane app
- improve heuristics for messy chat logs and issue threads
- add more example presets for demos

### 2. Packaged release

- sign and notarize the macOS build
- produce a Windows installer
- add CI packaging and release artifacts

### 3. Possible web version

- reuse the same transformation layer concepts behind an HTTP bridge
- keep `studio/engine.py` as the portable domain layer
- replace the pywebview bridge with an API boundary

### 4. Future share/export/viral features

- one-click “Copy for Claude / ChatGPT / Cursor / Codex”
- screenshot-ready demo presets
- shareable before/after links
- drag-and-drop file import
- prompt packs and saved workflows
