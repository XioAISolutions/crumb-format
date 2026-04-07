# CRUMB — AI Handoff Format (VS Code Extension)

Create, validate, and manage CRUMB handoff files for switching between AI tools.

## Features

- Syntax highlighting for `.crumb` files (markers, headers, sections, list items)
- Snippets for all CRUMB kinds: task, mem, map, log, todo
- Commands to create, validate, compress, and bench CRUMB files
- "Crumb It" command to generate a CRUMB from selected text

## Commands

Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and type "CRUMB":

| Command | Description |
|---------|-------------|
| **CRUMB: New Task Handoff** | Create a new task crumb in a fresh editor tab |
| **CRUMB: New Memory Crumb** | Create a new memory crumb template |
| **CRUMB: Validate Current File** | Run `crumb validate` on the active file |
| **CRUMB: Compress Current File** | Run `crumb compress` on the active file |
| **CRUMB: Bench Current File** | Run `crumb bench` and show results in the output panel |
| **CRUMB: Crumb It** | Generate a CRUMB from selected text using `crumb from-chat` |

## Snippets

In any `.crumb` file, type one of these prefixes and press `Tab`:

- `!crumb-task` — `kind=task` handoff with `[goal]`, `[context]`, and `[constraints]`
- `!crumb-mem` — `kind=mem` block with `[consolidated]`
- `!crumb-map` — `kind=map` block with `[project]` and `[modules]`
- `!crumb-log` — `kind=log` block with timestamped `[entries]`
- `!crumb-todo` — `kind=todo` block with `[items]`

## Installation

### From VSIX

1. Build the extension:
   ```bash
   npm install
   npm run compile
   npx vsce package
   ```
2. Install the `.vsix` file:
   - Open VS Code
   - `Ctrl+Shift+P` > "Extensions: Install from VSIX..."
   - Select the generated `.vsix` file

### From Marketplace

Search for "CRUMB" in the VS Code Extensions marketplace and click Install.

### Prerequisites

The `crumb` CLI must be installed and available on your PATH for the validate, compress, bench, and "Crumb It" commands to work. Install it from the [crumb-format repository](https://github.com/xioaisolutions/crumb-format).

## File Association

The extension automatically associates `.crumb` files with the CRUMB language mode, enabling syntax highlighting and snippets.

## Screenshots

- **Syntax Highlighting**: CRUMB markers (`BEGIN CRUMB` / `END CRUMB`) appear as keywords, section headers (`[goal]`, `[context]`) as headings, and key=value pairs are color-coded.
- **Snippets in Action**: Type `!crumb-task` and press Tab to expand a CRUMB v1.1 task handoff template with tab stops for each field.
- **Command Palette**: All CRUMB commands grouped under the "CRUMB:" prefix for easy discovery.
