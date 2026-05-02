---
description: Export the current Claude Code session as a CRUMB handoff for pasting into another AI tool.
allowed-tools: ["Bash", "Read", "Write"]
---

Export the current session as a CRUMB. Use the `crumb_new` MCP tool (or shell out to `crumb new` if MCP isn't available).

Process:

1. Determine the output path. If the user gave one in the slash-command argument, use it. Otherwise default to `~/.claude/handoffs/<short-session-name>.crumb`.

2. Pick the right `kind` based on what's been happening in this session:
   - **Most active goal** → `kind=task`. Set `[goal]` from the most recent user-issued objective; `[context]` from the surrounding conversation; `[constraints]` from anything the user said NOT to change.
   - **Stable preferences accumulated over multiple sessions** → `kind=mem`. Pull `[consolidated]` from the durable observations (style preferences, decisions, key facts).
   - **Repo overview discussed** → `kind=map`. Set `[project]` and `[modules]`.
   - When in doubt, default to `kind=task`.

3. Use `v=1.4`. Set `source=claude-code`. Set `title` from the first user message or the user's stated goal.

4. After writing, run `crumb validate <path>` to confirm well-formedness.

5. Print:
   - The output path
   - A 3-line summary of the crumb (kind, title, sections)
   - The exact text of `[goal]` (or `[consolidated]` for mem) so the user can copy-paste it without opening the file

Constraints:
- Do not include secrets, credentials, or unredacted error stacks. Match the same posture as a code-review hand-off.
- Keep the crumb under ~600 tokens unless the user asks for `--full`. Use `[fold:context/summary]`+`[fold:context/full]` if the context section would otherwise blow the budget.
- Validate before printing the path. Do not emit a crumb that fails validation.
