# CRUMB ClawHub Skill

A ClawHub skill that gives OpenClaw agents the ability to create, manage, and search `.crumb` handoff files.

## Install

### From ClawHub (when published)

```bash
openclaw skills install crumb
```

### Manual install

Copy the `clawhub-skill/` directory into your OpenClaw skills folder:

```bash
cp -r clawhub-skill/ ~/.openclaw/skills/crumb/
```

Or symlink it:

```bash
ln -s $(pwd)/clawhub-skill ~/.openclaw/skills/crumb
```

### Dependencies

The skill requires the `crumb` CLI. Install it with pip:

```bash
pip install crumb-format
```

This creates the `crumb` command globally.

## What it does

Once installed, your OpenClaw agent can:

- **"crumb it"** — generate a structured handoff crumb from the current conversation
- **Parse incoming crumbs** — act on `BEGIN CRUMB / END CRUMB` blocks automatically
- **Manage memory** — append observations, run dream passes, search across crumbs
- **Track todos** — add tasks, mark done, archive completed
- **Export/import** — convert crumbs to JSON, markdown, or clipboard format
- **Use templates** — scaffold from 6 built-in templates (bug-fix, feature, code-review, etc.)

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `crumbDir` | `./crumbs` | Where to store .crumb files |
| `autoSource` | `openclaw` | Source label for generated crumbs |
| `autoDreamThreshold` | `5` | Auto-dream when raw entries exceed this |

## Example usage

```
User: crumb it
Agent: [generates a task crumb summarizing current work]

User: [pastes a BEGIN CRUMB block]
Agent: [parses and acts on the crumb directly]

User: save that I prefer TypeScript and terse responses
Agent: crumb append prefs.crumb "Prefers TypeScript" "Terse responses"
```
