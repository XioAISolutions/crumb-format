# CRUMB CLI reference

_This document is auto-generated from the argparse tree by `tools/generate_cli_reference.py`. Do not edit by hand — rerun the generator after changing the CLI surface._

See [`docs/STABILITY.md`](STABILITY.md) for which pieces of this surface are frozen for the 0.x line.

## Top-level usage

```
usage: crumb [-h] [--version]
             {new,from-chat,from-git,validate,inspect,append,dream,search,merge,compact,diff,init,log,todo-add,todo-done,todo-list,todo-dream,watch,export,import,hooks,template,compress,bench,share,handoff,receive,context,passport,policy,audit,scan,comply,dashboard,bridge,pack,lint,webhook,metalk,palace,classify,wake,reflect}
             ...
```

## Subcommands (43)

- [`crumb append`](#crumb-append)
- [`crumb audit`](#crumb-audit)
- [`crumb bench`](#crumb-bench)
- [`crumb bridge`](#crumb-bridge)
- [`crumb classify`](#crumb-classify)
- [`crumb compact`](#crumb-compact)
- [`crumb comply`](#crumb-comply)
- [`crumb compress`](#crumb-compress)
- [`crumb context`](#crumb-context)
- [`crumb dashboard`](#crumb-dashboard)
- [`crumb diff`](#crumb-diff)
- [`crumb dream`](#crumb-dream)
- [`crumb export`](#crumb-export)
- [`crumb from-chat`](#crumb-from-chat)
- [`crumb from-git`](#crumb-from-git)
- [`crumb handoff`](#crumb-handoff)
- [`crumb hooks`](#crumb-hooks)
- [`crumb import`](#crumb-import)
- [`crumb init`](#crumb-init)
- [`crumb inspect`](#crumb-inspect)
- [`crumb lint`](#crumb-lint)
- [`crumb log`](#crumb-log)
- [`crumb merge`](#crumb-merge)
- [`crumb metalk`](#crumb-metalk)
- [`crumb new`](#crumb-new)
- [`crumb pack`](#crumb-pack)
- [`crumb palace`](#crumb-palace)
- [`crumb passport`](#crumb-passport)
- [`crumb policy`](#crumb-policy)
- [`crumb receive`](#crumb-receive)
- [`crumb reflect`](#crumb-reflect)
- [`crumb scan`](#crumb-scan)
- [`crumb search`](#crumb-search)
- [`crumb share`](#crumb-share)
- [`crumb template`](#crumb-template)
- [`crumb todo-add`](#crumb-todo-add)
- [`crumb todo-done`](#crumb-todo-done)
- [`crumb todo-dream`](#crumb-todo-dream)
- [`crumb todo-list`](#crumb-todo-list)
- [`crumb validate`](#crumb-validate)
- [`crumb wake`](#crumb-wake)
- [`crumb watch`](#crumb-watch)
- [`crumb webhook`](#crumb-webhook)

---

### `crumb append`

**Arguments**

- `file` — Path to an existing kind=mem .crumb file.
- `entries ENTRIES...` — Observations to append to [raw].


### `crumb audit`

#### `crumb audit export`

**Options**

- `--agent AGENT`
- `--since SINCE`
- `-f, --format FORMAT` (default: `crumb`)
- `-o, --output OUTPUT` (default: `-`)

#### `crumb audit feed`

**Options**

- `--agent AGENT`


### `crumb bench`

**Arguments**

- `file` — .crumb file to benchmark.


### `crumb bridge`

#### `crumb bridge export`

**Arguments**

- `input` — Input .crumb file.

**Options**

- `--to TO` — Target format.
- `-o, --output OUTPUT` — Output file. (default: `-`)

#### `crumb bridge import`

**Arguments**

- `input` — Input file.

**Options**

- `--from SOURCE_FORMAT` — Source format.
- `-o, --output OUTPUT` — Output .crumb file. (default: `-`)

#### `crumb bridge list`

#### `crumb bridge mempalace`


### `crumb classify`

**Options**

- `--text TEXT` — Text to classify.
- `--file FILE` — File of lines to classify (one per line).
- `--explain` — Show matched patterns and scores.


### `crumb compact`

**Arguments**

- `file [FILE]` — .crumb file to compact (default: stdin).

**Options**

- `--output, -o OUTPUT` — Output file or - for stdout. (default: `-`)


### `crumb comply`

**Options**

- `-f, --format FORMAT` (default: `text`)
- `-o, --output OUTPUT` (default: `-`)
- `--framework FRAMEWORK` (default: `general`)


### `crumb compress`

**Arguments**

- `file` — .crumb file to compress.

**Options**

- `-o, --output OUTPUT` — Output path (default: stdout). (default: `-`)
- `--target TARGET` — Target retention ratio 0.0-1.0 (default: 0.5 = keep top 50%%). (default: `0.5`)
- `--metalk` — Apply MeTalk caveman compression as Stage 3.
- `--metalk-level METALK_LEVEL` — MeTalk level (default: 2). (default: `2`)


### `crumb context`

**Options**

- `--commits COMMITS` — Number of recent commits to include (default: 5). (default: `5`)
- `--goal GOAL` — Override the auto-detected goal.
- `--title TITLE` — Override the auto-generated title.
- `--source SOURCE` — Override source label (default: crumb.context).
- `-o, --output OUTPUT` — Output file or - for stdout. (default: `-`)
- `--metalk` — Apply MeTalk compression.
- `--metalk-level METALK_LEVEL` — MeTalk level (default: 2). (default: `2`)
- `--clipboard` — Copy result to clipboard instead of printing.
- `--max-facts MAX_FACTS` — Max palace facts to include (default: 8). (default: `8`)


### `crumb dashboard`

**Options**

- `-o, --output OUTPUT` (default: `agentauth-dashboard.html`)


### `crumb diff`

**Arguments**

- `file_a` — First .crumb file.
- `file_b` — Second .crumb file.


### `crumb dream`

**Arguments**

- `file` — Path to a kind=mem .crumb file.

**Options**

- `--dry-run` — Print result to stdout instead of writing.


### `crumb export`

**Arguments**

- `file [FILE]` — .crumb file to export (default: stdin).

**Options**

- `--format, -f FORMAT` — Output format.
- `--output, -o OUTPUT` — Output file or - for stdout. (default: `-`)


### `crumb from-chat`

**Options**

- `--input, -i INPUT` — Input file or - for stdin. (default: `-`)
- `--output, -o OUTPUT` — Output file or - for stdout. (default: `-`)
- `--title TITLE` — Title for the crumb.
- `--source SOURCE` — Source label (e.g. chatgpt.chat, claude.chat).
- `--goal GOAL` — Override the default goal text.
- `--kind, -k KIND` — Output kind: task (default) or mem (extracts decisions). (default: `task`)
- `--constraints, -c [CONSTRAINTS...]` — Constraints as separate arguments.


### `crumb from-git`

**Options**

- `--commits COMMITS` — Number of recent commits to include (default: 5). (default: `5`)
- `--branch BRANCH` — Base branch to compare against (default: auto-detect main/master).
- `--title TITLE` — Override the auto-generated title.
- `--source SOURCE` — Override source label (default: git).
- `--output, -o OUTPUT` — Output file or - for stdout. (default: `-`)


### `crumb handoff`

**Arguments**

- `file` — .crumb file to hand off.

**Options**

- `--target TARGET` — Target AI tool (optional).


### `crumb hooks`

**Options**

- `--dir DIR` — Project directory (default: current). (default: `.`)


### `crumb import`

**Options**

- `--from FROM` — Source format.
- `--input, -i INPUT` — Input file or - for stdin. (default: `-`)
- `--output, -o OUTPUT` — Output file or - for stdout. (default: `-`)


### `crumb init`

**Options**

- `--dir DIR` — Project directory (default: current). (default: `.`)
- `--project, -p PROJECT` — Project name (default: directory name).
- `--description, -d DESCRIPTION` — One-line project description.
- `--claude-md` — Create/update CLAUDE.md with CRUMB instructions.
- `--cursor-rules` — Create .cursor/rules with CRUMB instructions.
- `--windsurf-rules` — Create .windsurfrules with CRUMB instructions.
- `--chatgpt-rules` — Print ChatGPT custom instructions.
- `--gemini` — Create .gemini/settings.json with CRUMB instructions.
- `--copilot` — Create .github/copilot-instructions.md with CRUMB instructions.
- `--cody` — Create .sourcegraph/cody.json with CRUMB instructions.
- `--continue-dev` — Create .continue/config.json with CRUMB system message.
- `--aider` — Create .aider.conf.yml with CRUMB conventions.
- `--replit` — Create .replit with CRUMB instructions.
- `--devin` — Create devin.md with CRUMB instructions.
- `--bolt` — Create .bolt/config.json with CRUMB instructions.
- `--lovable` — Create .lovable/config.json with CRUMB instructions.
- `--all` — Seed all AI tools at once.


### `crumb inspect`

**Arguments**

- `file [FILE]` — .crumb file to inspect (default: stdin).

**Options**

- `--headers-only, -H` — Show only headers and section names.


### `crumb lint`

**Arguments**

- `files FILES...` — One or more .crumb files to lint.

**Options**

- `--secrets` — Enable secret detection checks.
- `--redact` — Redact obvious credentials in-place unless --output is set.
- `--max-size MAX_SIZE` — Warn when estimated total or raw section tokens exceed this value.
- `--strict` — Return a non-zero exit code for warnings.
- `--output OUTPUT` — Optional output file or directory for redacted content.


### `crumb log`

**Arguments**

- `file` — Path to a log crumb (created if missing).
- `entries ENTRIES...` — Entries to log.

**Options**

- `--title, -t TITLE` — Title (for new log crumbs).
- `--source, -s SOURCE` — Source label.


### `crumb merge`

**Arguments**

- `files FILES...` — Mem .crumb files to merge.

**Options**

- `--output, -o OUTPUT` — Output file or - for stdout. (default: `-`)
- `--title, -t TITLE` — Title for the merged crumb.


### `crumb metalk`

**Arguments**

- `file` — .crumb file to encode/decode.

**Options**

- `--level LEVEL` — 1=dict only (lossless), 2=dict+grammar strip, 3=aggressive (default: 2). (default: `2`)
- `--decode` — Decode MeTalk back to full form.
- `-o, --output OUTPUT` — Output path. (default: `-`)


### `crumb new`

**Arguments**

- `kind` — Kind of crumb to create.

**Options**

- `--title, -t TITLE` — Title for the crumb.
- `--source, -s SOURCE` — Source label (e.g. claude.chat, cursor.agent).
- `--output, -o OUTPUT` — Output file or - for stdout. (default: `-`)
- `--goal GOAL` — Goal text (task only).
- `--context [CONTEXT...]` — Context items (task only).
- `--constraints, -c [CONSTRAINTS...]` — Constraint items (task only).
- `--entries, -e [ENTRIES...]` — Consolidated entries (mem only).
- `--project, -p PROJECT` — Project name (map only).
- `--description, -d DESCRIPTION` — Project description (map only).
- `--modules, -m [MODULES...]` — Module entries (map only).


### `crumb pack`

**Options**

- `--dir DIR` — Directory containing source .crumb files.
- `--query QUERY` — Query describing the handoff you want to build.
- `--project PROJECT` — Optional project filter/header to apply while selecting crumbs.
- `--kind KIND` — Output kind for the packed CRUMB.
- `--mode MODE` — Pack shaping mode: implement (default), debug, or review. (default: `implement`)
- `--max-total-tokens MAX_TOTAL_TOKENS` — Estimated token budget for the final packed CRUMB.
- `--strategy STRATEGY` — Ranking strategy for selecting and merging context (default: hybrid). (default: `hybrid`)
- `--title TITLE` — Optional title override for the packed CRUMB.
- `--ollama, --use-local` — Optionally run a final local-model compression pass with Ollama.
- `--ollama-model OLLAMA_MODEL` — Local Ollama model to use when --ollama is set (default: llama3.2:3b). (default: `llama3.2:3b`)
- `--output, -o OUTPUT` — Output .crumb file path.


### `crumb palace`

#### `crumb palace init`

**Options**

- `--path PATH` — Parent directory (default: cwd).

#### `crumb palace add`

**Arguments**

- `text` — Observation text.

**Options**

- `--wing WING` — Wing name (person/project/topic).
- `--room ROOM` — Room name (specific topic).
- `--hall HALL` — Hall (auto-classified via `crumb classify` if omitted).
- `--path PATH` — Start directory for palace lookup.

#### `crumb palace list`

**Options**

- `--wing WING` — Filter by wing.
- `--hall HALL`
- `--path PATH` — Start directory.

#### `crumb palace search`

**Arguments**

- `query` — Search query.

**Options**

- `--wing WING` — Restrict to one wing.
- `--hall HALL`
- `--path PATH` — Start directory.

#### `crumb palace tunnel`

**Options**

- `--path PATH` — Start directory.

#### `crumb palace stats`

**Options**

- `--path PATH` — Start directory.

#### `crumb palace wiki`

**Options**

- `--path PATH` — Start directory.
- `-o, --output OUTPUT` — Output file or - for stdout. (default: `-`)


### `crumb passport`

#### `crumb passport register`

**Arguments**

- `name` — Agent name.

**Options**

- `--framework FRAMEWORK` (default: `unknown`)
- `--owner OWNER` (default: ``)
- `--tools-allowed [TOOLS_ALLOWED...]` (default: `[]`)
- `--tools-denied [TOOLS_DENIED...]` (default: `[]`)
- `--ttl-days TTL_DAYS` (default: `90`)
- `-o, --output OUTPUT`

#### `crumb passport inspect`

**Arguments**

- `agent_id` — Agent ID or name.

#### `crumb passport revoke`

**Arguments**

- `agent_id` — Agent ID to revoke.

#### `crumb passport list`

**Options**

- `--status STATUS` (default: `all`)


### `crumb policy`

#### `crumb policy set`

**Arguments**

- `agent_name`

**Options**

- `--allow [ALLOW...]` (default: `[]`)
- `--deny [DENY...]` (default: `[]`)
- `--max-actions MAX_ACTIONS` (default: `1000`)

#### `crumb policy test`

**Arguments**

- `agent_name`
- `tool`


### `crumb receive`

**Options**

- `--file FILE` — Read crumb from this file instead of clipboard.
- `-o, --output OUTPUT` — Save received crumb to file.
- `--palace` — Auto-file observations into the palace.
- `--wing WING` — Palace wing (default: derived from source header).
- `--hall HALL` — Palace hall (default: auto-classified).


### `crumb reflect`

**Options**

- `--path PATH` — Start directory for palace lookup.
- `-o, --output OUTPUT` — Output file or - for stdout. (default: `-`)
- `--format, -f FORMAT` — Output format (default: text). (default: `text`)
- `--stale-days STALE_DAYS` — Days before a room is considered stale (default: 30). (default: `30`)


### `crumb scan`

**Options**

- `--path PATH` — Directory to scan (default: current directory). (default: `.`)
- `--format, -f FORMAT` — Output format (default: text). (default: `text`)
- `--min-risk MIN_RISK` — Minimum risk level to report (default: low). (default: `low`)


### `crumb search`

**Arguments**

- `query` — Search query (space-separated terms).

**Options**

- `--dir DIR` — Directory to search (default: current). (default: `.`)
- `--method, -m METHOD` — Search method: keyword (exact), fuzzy (approximate), ranked (TF-IDF). (default: `keyword`)
- `--limit, -n LIMIT` — Max results to show.


### `crumb share`

**Arguments**

- `file` — .crumb file to share.


### `crumb template`

**Arguments**

- `action` — Template action.
- `name [NAME]` — Template name (for use/add).
- `source_file [SOURCE_FILE]` — Source .crumb file (for add).

**Options**

- `--output, -o OUTPUT` — Output file (for use). (default: `-`)


### `crumb todo-add`

**Arguments**

- `file` — Path to a todo crumb (created if missing).
- `tasks TASKS...` — Tasks to add.

**Options**

- `--title, -t TITLE` — Title (for new todo crumbs).
- `--source, -s SOURCE` — Source label.


### `crumb todo-done`

**Arguments**

- `file` — Path to a todo crumb.
- `query` — Substring to match against open tasks.


### `crumb todo-dream`

**Arguments**

- `file` — Path to a todo crumb.


### `crumb todo-list`

**Arguments**

- `file [FILE]` — Path to a todo crumb.

**Options**

- `--all, -a` — Show completed tasks too.


### `crumb validate`

**Arguments**

- `files FILES...` — .crumb files to validate.


### `crumb wake`

**Options**

- `--path PATH` — Start directory for palace lookup.
- `-o, --output OUTPUT` — Output file or - for stdout. (default: `-`)
- `--max-facts MAX_FACTS` — Max facts to include (default: 8). (default: `8`)
- `--metalk` — Pipe output through MeTalk compression.
- `--metalk-level METALK_LEVEL` (default: `2`)
- `--reflect` — Include top knowledge gaps in the wake crumb.


### `crumb watch`

**Arguments**

- `target` — File or directory to watch.

**Options**

- `--threshold THRESHOLD` — Raw entries before auto-dream (default: 5). (default: `5`)
- `--interval INTERVAL` — Poll interval in seconds (default: 3). (default: `3`)


### `crumb webhook`

#### `crumb webhook add`

**Arguments**

- `url` — Webhook URL.

**Options**

- `--events EVENTS...` — Events to subscribe to (e.g. passport.revoked policy.denied).

#### `crumb webhook list`

#### `crumb webhook remove`

**Arguments**

- `webhook_id` — Webhook ID to remove.

#### `crumb webhook test`

**Arguments**

- `webhook_id` — Webhook ID to test.


