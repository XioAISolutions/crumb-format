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

CrumbLLM now lives in its own repository: **XioAISolutions/CrumbLLM**.

```
Crumb-Bob ──▶ crumb-format        CrumbLLM
   (generates packs)              (reads + reasons)
```

Crumb-Bob depends on `crumb-format`. **CrumbLLM is standalone**: it bundles its
own CRUMB reader, so it has *no required runtime dependency* on crumb-format
(crumb-format remains an optional extra it prefers when installed). CrumbLLM has
no dependency on Crumb-Bob either — it reads packs as plain directories of
`.crumb` files. Neither project re-implements the canonical spec.

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

   **Fix:** the standalone `crumb-llm` package — now in its own repository,
   `XioAISolutions/CrumbLLM` — holds all analysis logic and contains zero spec
   code. It bundles its own CRUMB reader, so it is fully standalone (crumb-format
   is an optional extra it prefers when present).

3. **No clean import surface for the parser.** Consumers had to reach into
   `cli.crumb.parse_crumb`.

   **Recommended next step (crumb-format):** expose a stable public API:

   ```python
   # crumb_format/__init__.py  (recommended)
   from .parser import parse_crumb, validate_crumb
   ```

   CrumbLLM's `parser_adapter.py` already prefers a `crumb_format` import,
   then `cli.crumb`, then its own bundled reader, so adding this API is
   non-breaking — it would simply become CrumbLLM's preferred path when a user
   installs the optional `crumb-format` extra.

## Boundaries to keep enforced

- CrumbLLM must **not** copy the CRUMB spec — always parse via crumb-format.
- CrumbLLM must **not** contain Bob-specific logic — packs are just directories.
- CrumbLLM must **not** train by default — scratch training is opt-in.
- **No secrets** in config files or any database — keys come from env vars only.
- **No models, checkpoints, or datasets** committed to any repo.

## Remaining follow-ups

- [x] Split the standalone package out of crumb-format into
      `XioAISolutions/CrumbLLM`. Done — CrumbLLM is its own repo, and the
      duplicated `crumb-llm/` directory has been removed from crumb-format.
- [x] Make CrumbLLM installable without crumb-format on PyPI. Done a different
      way: CrumbLLM bundles its own CRUMB reader, so `pip install crumb-llm` has
      zero required dependencies (crumb-format is an optional extra).
- [ ] Add the recommended `crumb_format` public parser API (`parse_crumb`,
      `validate_crumb`) so the optional `crumb-llm[crumb-format]` path uses a
      stable import instead of `cli.crumb`.
- [ ] Publish `crumb-format` to PyPI (optional now for CrumbLLM, still wanted for
      Crumb-Bob and for the `crumb-llm[crumb-format]` extra).
- [ ] Reconcile the crumb-format version string across release notes (resolved
      in code: `pyproject` and the CLI now both report `1.1.0`).
