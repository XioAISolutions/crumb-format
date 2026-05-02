---
description: Import a CRUMB into the current Claude Code session and continue the work.
allowed-tools: ["Bash", "Read"]
---

Import a CRUMB and pick up where the previous session/AI left off.

Process:

1. The argument is the path to the crumb file. If no argument given, ask the user for the path.

2. Run `crumb validate <path>` first. If validation fails, surface the error and stop — don't try to consume malformed input.

3. Read the file and parse it. The format is:
   ```
   BEGIN CRUMB
   v=<1.1|1.2|1.3|1.4>
   kind=<task|mem|map|log|todo|agent>
   ...
   ---
   [section_name]
   body
   ...
   END CRUMB
   ```

4. Summarize what was imported:
   - The `kind=` and `title=` headers
   - For `kind=task`: surface `[goal]`, `[constraints]`, and the most recent `[handoff]` line if present
   - For `kind=mem`: surface the top 5 lines of `[consolidated]`
   - For `kind=map`: surface `[project]` and the count of `[modules]` entries
   - For `kind=agent`: surface `[identity]` and the count of `[rules]`

5. Tell the user: "Loaded the crumb. Ready to continue. The goal/context above is now part of our shared context. What would you like to do first?"

Special handling:

- If the crumb has a `[handoff]` section with `after=` dependencies, surface the dependency graph and pick the first unblocked step.
- If the crumb has `[checks]`, surface any check that's `:: fail` or has a canonical failure-mode name (`hallucinated_tool_call`, `refusal_loop`, etc. per SPEC §21.1.2).
- If the crumb has a `deadline=` annotation in `[handoff]`, run `crumb lint <path> --check-deadlines` and surface any overdue or malformed deadlines.

Do NOT execute `[script]` bodies. Per SPEC §21.4 they are advisory carriers, not executable.
