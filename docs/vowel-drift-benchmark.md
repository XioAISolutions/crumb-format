# Vowel-Strip Drift Benchmark (ngram)

Cosine similarity vs original, per MeTalk level.
`ngram` backend: char 4-gram lexical similarity (no model)

| File | Tokens | L1 sim/save | L2 sim/save | L3 sim/save | L4 sim/save | L5 sim/save |
|------|--------|-------------|-------------|-------------|-------------|-------------|
| log-deployment.crumb | 147 | 0.971 / 5.4% | 0.971 / 5.4% | 0.971 / 5.4% | 0.817 / 15.0% | 0.817 / 15.0% |
| map-client-takeover.crumb | 290 | 0.960 / 5.2% | 0.956 / 5.9% | 0.933 / 6.6% | 0.417 / 26.6% | 0.417 / 26.6% |
| map-repo-onboarding.crumb | 166 | 0.886 / 12.0% | 0.869 / 13.3% | 0.868 / 13.3% | 0.472 / 25.9% | 0.472 / 25.9% |
| mem-mempalace-auth-migration.crumb | 120 | 0.919 / 9.2% | 0.902 / 10.8% | 0.887 / 11.7% | 0.715 / 20.0% | 0.715 / 20.0% |
| mem-terse-output.crumb | 336 | 0.965 / 4.2% | 0.949 / 5.7% | 0.913 / 6.2% | 0.419 / 23.8% | 0.419 / 23.8% |
| mem-user-preferences.crumb | 135 | 0.841 / 14.8% | 0.828 / 16.3% | 0.828 / 16.3% | 0.425 / 30.4% | 0.425 / 30.4% |
| task-bug-fix.crumb | 127 | 0.842 / 15.0% | 0.813 / 17.3% | 0.813 / 17.3% | 0.367 / 31.5% | 0.367 / 31.5% |
| task-content-repurpose-handoff.crumb | 250 | 0.930 / 9.2% | 0.867 / 12.8% | 0.820 / 14.0% | 0.432 / 29.6% | 0.432 / 29.6% |
| task-cross-tool-feature-handoff.crumb | 249 | 0.924 / 9.2% | 0.884 / 12.0% | 0.820 / 13.7% | 0.373 / 30.9% | 0.373 / 30.9% |
| task-feature-continuation.crumb | 134 | 0.874 / 14.9% | 0.784 / 19.4% | 0.784 / 19.4% | 0.441 / 28.4% | 0.441 / 28.4% |
| task-packed-auth-context.crumb | 233 | 0.929 / 9.4% | 0.906 / 11.6% | 0.892 / 12.4% | 0.699 / 21.5% | 0.699 / 21.5% |
| todo-sprint.crumb | 90 | 0.887 / 8.9% | 0.887 / 8.9% | 0.887 / 8.9% | 0.631 / 20.0% | 0.631 / 20.0% |
| v12-fold.crumb | 317 | 0.961 / 4.7% | 0.895 / 8.8% | 0.895 / 8.8% | 0.344 / 27.8% | 0.344 / 27.8% |
| v12-handoff.crumb | 204 | 0.958 / 4.9% | 0.783 / 14.2% | 0.783 / 14.2% | 0.423 / 29.4% | 0.423 / 29.4% |
| v12-refs.crumb | 182 | 0.945 / 6.0% | 0.913 / 8.8% | 0.913 / 8.8% | 0.499 / 22.5% | 0.499 / 22.5% |
| v12-typed-content.crumb | 218 | 0.930 / 8.7% | 0.858 / 13.8% | 0.846 / 14.2% | 0.717 / 21.1% | 0.717 / 21.1% |
| wake-session.crumb | 195 | 0.912 / 9.7% | 0.912 / 9.7% | 0.912 / 9.7% | 0.595 / 22.6% | 0.595 / 22.6% |
| **average** |  | **0.920 / 8.9%** | **0.881 / 11.5%** | **0.869 / 11.8%** | **0.517 / 25.1%** | **0.517 / 25.1%** |

**Reading the table:** `sim` is cosine similarity vs the original (1.0 = identical). `save` is the % token reduction vs the original. Higher sim with higher save is better.

