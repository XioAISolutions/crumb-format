# Vowel-Strip Compression — MeTalk Layer 4 / 5

CRUMB's MeTalk pipeline ships two new tiers:

- **Layer 4 — skeleton**: rule-based interior-vowel removal. Deterministic,
  no model, no network. Runs after MeTalk's dictionary and grammar passes
  so the most ambiguous tech terms have already been canonicalized.
- **Layer 5 — adaptive**: same strip, but per-line drift is measured against
  a sentence-transformers embedding and the strip is *kept* only if cosine
  similarity stays above a threshold (default 0.85). Requires the optional
  `[embeddings]` extra; without it L5 falls back to L4 with a warning.

## Why bother

Vowels carry few bits in English consonant-skeleton text (`btfl dy` ≈
"beautiful day"), but token-budget-constrained AI handoffs care about
every character. The interesting question is not *can we compress* but
*at what compression level does embedding similarity break down*. Layers
4 and 5 give you a knob and a measurement so you can answer it for your
own corpus.

## What gets stripped

`cli/vowelstrip.py:strip_word` removes interior `a/e/i/o/u` (case-insensitive)
when:

- the word is at least `--min-length` chars (default 4),
- it is not all-uppercase (acronyms like `HTTP`, `JWT`, `API` survive),
- it is not in `PROTECTED_WORDS` — a tiny allowlist of confusable
  consonant skeletons (`read`/`lead`/`feed`, `node`/`mode`/`code`,
  `user`/`users`, programming literals).

The first letter (and case) is preserved. Trailing `s` on plurals is
preserved so singular/plural pairs don't collapse onto the same skeleton
(`fnctn` vs `fnctns`).

## What is left alone

`encode_crumb` skips the strip entirely for:

- header lines (everything before `---`),
- section markers (`[goal]`, `[g]`, `[fold:context/summary]`, …),
- fenced code blocks (` ``` `… ` ``` `),
- v1.2 typed-content sections (`@type: code/*`, `diff`, `json`, `yaml`),
- the `BEGIN/END CRUMB` and `BC/EC` sentinels.

`strip_line` additionally treats any whitespace-delimited token containing
`/`, `:`, `_`, `.`, `@`, `\`, `=`, or `#` as **opaque** and passes it
through whole. That covers URLs, file paths, snake_case identifiers,
version strings, emails, contractions like `don't`, and most code-like
fragments embedded in prose.

## Round-trip

Vowel removal is lossy. `decode()` strips the `vs=` header and runs the
Layer 1 dictionary reverse, but vowel-stripped words pass through
unchanged because their consonant skeletons don't match the dictionary.
If you need a round-trippable encoding, stay at level 1.

## Measuring drift on your corpus

```bash
# Lexical (char 4-gram) — fast, no deps
python scripts/measure_drift.py --dir examples

# Semantic (sentence-transformers, all-MiniLM-L6-v2)
pip install crumb-format[embeddings]
python scripts/measure_drift.py --backend st --dir examples

# Regenerate the bundled benchmark
python scripts/measure_drift.py --md > docs/vowel-drift-benchmark.md
```

The script prints, for every `.crumb` in the directory, the cosine
similarity of each MeTalk level (1-5) against the original plus the
percentage of tokens saved.

## Reading the bundled benchmark

[`docs/vowel-drift-benchmark.md`](vowel-drift-benchmark.md) is the
ngram-backend report against `examples/`. Across the 17 fixtures:

| Level | Avg lexical sim | Avg tokens saved |
|-------|-----------------|------------------|
| L1    | 0.92            | 8.9%             |
| L2    | 0.88            | 11.5%            |
| L3    | 0.87            | 11.8%            |
| L4    | 0.51            | 25.2%            |
| L5    | 0.51            | 25.2%            |

The L4 lexical-sim cliff (0.87 → 0.51) is expected — char-4-gram cosine
counts vowel n-grams, which all disappear at L4. **A semantic-embedding
backend will hold sim much higher** because the consonant skeleton
preserves enough information for a Transformer to reconstruct meaning.
Re-run with `--backend st` to see the semantic curve for your model.

## Tuning the knob

- Stay at L3 for human-facing crumbs you might re-read.
- Use L4 for AI-to-AI handoffs where token budget dominates and you
  control both ends (the receiving model has been instructed to expect
  vowel-stripped consonant skeletons).
- Use L5 when you want the safety rail of an embedding-based drift
  guarantee per line. Tune `--adaptive-threshold` (default 0.85) up for
  fidelity or down for compression.
- Compose with v1.2 foldable sections: ship a `[fold:context/summary]`
  at L3 and a `[fold:context/vowless]` at L4 in the same crumb so the
  consumer picks the tier their token budget allows.

## Caveats

- Lexical similarity is not semantic similarity. The bundled benchmark
  uses ngram by default for portability — install `[embeddings]` for the
  number that matches what models actually "see".
- Layer 4 is irreversible. `decode()` is best-effort.
- The PROTECTED_WORDS list is intentionally small. If your domain has
  high-frequency confusable pairs (`hand`/`hund`, `pad`/`pod`/`pud`,
  etc.) extend it before relying on L4 in production.
