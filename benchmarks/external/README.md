# Measuring another tool on this corpus

Every row in `benchmarks/compare.py` is a strategy *this repository implements*.
That is a real limit: replicating a competitor's documented mechanism and then
scoring it is not the same as measuring the competitor, and the `*-style`
naming in the main table exists to keep that distinction visible.

This directory is how a real tool gets measured instead of approximated.

## Why the corpus makes this possible

`benchmarks/corpus/` holds **committed transcripts** in five ingest formats.
Because the input is fixed and public, any tool that turns a conversation into
a handoff can be run over the same bytes, and the results are comparable
without anyone trusting anyone else's harness. That is the whole point of
publishing the corpus rather than a score.

## Protocol

1. Pick a corpus transcript, for example `benchmarks/corpus/long-claude-code.jsonl`.
2. Run your tool over it and capture **what it would actually hand the next
   agent** — the text that gets injected or pasted into the fresh session. Not
   its internal database, not a summary of its summary.
3. Save that text as `benchmarks/external/<tool>/<corpus-stem>.txt`:

   ```
   benchmarks/external/
     ai-memory/
       long-claude-code.txt
       tools-claude-code.txt
     your-tool/
       long-claude-code.txt
   ```

4. Add a `SOURCE.md` in your tool's directory recording the version, the exact
   command, and the date. Provenance is the difference between a measurement
   and an assertion.
5. Re-run `python benchmarks/compare.py`. Your artifacts appear as their own
   rows, measured by the same `cli/measure.py` as everything else.

## What gets reported

External rows are named for the tool with no `-style` suffix, because they are
real output rather than a replicated mechanism. They are measured identically:
token count, savings against the full transcript, lexical fact retention, and —
the column that matters — **which facts were dropped**.

Cases you have not supplied are reported as `missing`, never skipped and never
interpolated. An empty slot is a fact about what has been measured so far, and
hiding it would make the table dishonest in exactly the way this harness is
supposed to prevent.

## Notes for specific tools

**ai-memory** (`akitaonrails/ai-memory`) stores handoff records in SQLite,
reachable only through its MCP tools or server API — but its session summary
pages are written to the wiki directory as `sessions/<id>.md` and committed to
git, so those are plain files you can copy here. Use the page content that the
next agent receives, and record in `SOURCE.md` which of the two you captured;
they are not the same artifact and should not be silently mixed.

**Hosted compaction** (OpenAI, Anthropic) cannot be captured this way, because
the provider compacts server-side and does not return the compacted context.
That is why the main table replicates the documented mechanisms instead, and
labels those rows `*-style`. If a provider ever exposes the compacted context,
it belongs here as a real row.

## A standing offer

If you maintain a handoff or memory tool and think this harness measures you
unfairly, the corpus and the measurement code are both in this repository.
Open a PR with your artifacts and a `SOURCE.md`. Results that show CRUMB losing
get merged — `tests/test_compare.py` already fails if no strategy beats CRUMB
on any axis anywhere, because a comparison its author always wins is marketing.
