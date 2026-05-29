# Cleaning up the XIO CRUMB ecosystem

This document is the reference for how the CRUMB projects are separated and how
to finish the cleanup. It exists because responsibilities between
`crumb-format`, `Crumb-Bob`, and the LLM code had drifted into one another.

## Target architecture: three repos, three jobs

| Repo                    | Owns                                                                 | Does NOT own                          |
|-------------------------|---------------------------------------------------------------------|---------------------------------------|
| **crumb-format**        | The CRUMB spec, parser, validator, linter, schemas, base `crumb` CLI | AI analysis; session capture          |
| **Crumb-Bob**           | IBM Bob session capture; CRUMB **pack** generation                   | The spec; AI reasoning                |
| **CrumbLLM** (`crumb-llm`) | AI analysis over CRUMB files and packs (summaries, risks, next, handoffs) | The spec; Bob capture; training by default |

Dependency direction is strictly one-way:

```
Crumb-Bob ──▶ crumb-format ◀── CrumbLLM
   (generates packs)            (reads + reasons)
```

Both Crumb-Bob and CrumbLLM depend on `crumb-format`. Neither re-implements the
spec. CrumbLLM has **no** dependency on Crumb-Bob — it reads packs as plain
directories of `.crumb` files.

## What was messy, and the fix

1. **The `crumb_llm` import name was occupied by a research artifact.**
   crumb-format shipped a physics / wave-field language model as a top-level
   `crumb_llm` package (requiring `torch` on import). That squatted on the name
   the actual product needed, and it would have collided with `pip install
   crumb-llm` (two distributions exporting the same top-level module).

   **Fix (done in this branch):** the wave-field LM was renamed
   `crumb_llm → crumb_wavelm` inside crumb-format. The `crumb llm ...`
   sub-CLI, configs, and tests were repointed. The `crumb_llm` name is now free
   for the analysis product.

2. **AI logic lived inside the format repo.** Analysis prompts, providers, and
   quality gates did not belong in the spec repo.

   **Fix:** the new standalone `crumb-llm` package (this PR's `crumb-llm/`
   directory) holds all analysis logic. It depends on crumb-format and contains
   zero spec code.

3. **No clean import surface for the parser.** Consumers had to reach into
   `cli.crumb.parse_crumb`.

   **Recommended next step (crumb-format):** expose a stable public API:

   ```python
   # crumb_format/__init__.py  (recommended)
   from .parser import parse_crumb, validate_crumb
   ```

   CrumbLLM's `parser_adapter.py` already prefers a `crumb_format` import and
   only falls back to `cli.crumb`, so adding this API is non-breaking and lets
   CrumbLLM drop the fallback later.

## Boundaries to keep enforced

- CrumbLLM must **not** copy the CRUMB spec — always parse via crumb-format.
- CrumbLLM must **not** contain Bob-specific logic — packs are just directories.
- CrumbLLM must **not** train by default — scratch training is opt-in.
- **No secrets** in config files or any database — keys come from env vars only.
- **No models, checkpoints, or datasets** committed to any repo.

## Remaining follow-ups

- [ ] Split `crumb-llm/` out of crumb-format into `XioAISolutions/CrumbLLM`
      (see the PR description for the `git subtree split` command).
- [ ] Publish `crumb-format` ≥ 1.1.0 to PyPI so `crumb-llm`'s dependency
      resolves for end users.
- [ ] Add the recommended `crumb_format` public parser API.
- [ ] Reconcile the crumb-format version string (`pyproject` says `1.1.0`,
      some tests/release notes say `0.4.0`).
