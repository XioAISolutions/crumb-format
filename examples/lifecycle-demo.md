# Memory lifecycle demo

This walkthrough shows the full append → dream → diff workflow.

## 1. Create a mem crumb

```bash
crumb new mem \
  --title "Project prefs" \
  --source human.notes \
  --entries "Use TypeScript" "No ORMs" "Prefer Postgres" \
  -o prefs.crumb
```

## 2. Append raw observations over time

As you work, append new observations:

```bash
crumb append prefs.crumb "Switched from npm to pnpm"
crumb append prefs.crumb "Team agreed on Tailwind v4" "Dropped Styled Components"
```

Your crumb now has a `[raw]` section with timestamped entries alongside the original `[consolidated]` section.

## 3. Run a dream pass

Consolidate everything — deduplicates, merges `[raw]` into `[consolidated]`, and prunes to budget:

```bash
# preview first
crumb dream prefs.crumb --dry-run

# then write
crumb dream prefs.crumb
```

Output:
```
Dream pass complete on prefs.crumb
  3 existing + 3 raw → 6 consolidated
```

## 4. Track changes with diff

Save versions and compare:

```bash
cp prefs.crumb prefs-v1.crumb

# ... more appends and another dream pass ...
crumb append prefs.crumb "Moved CI to GitHub Actions"
crumb dream prefs.crumb

crumb diff prefs-v1.crumb prefs.crumb
```

Output:
```
Headers:
  - dream_sessions=1
  + dream_sessions=2

[consolidated]:
  + - Moved CI to GitHub Actions
```

## 5. Merge across team members

Combine multiple peoples' preference crumbs:

```bash
crumb merge alice-prefs.crumb bob-prefs.crumb \
  --title "Team preferences" \
  -o team.crumb
```

Duplicates are automatically removed.

## 6. Search across crumbs

Find relevant crumbs by keyword:

```bash
crumb search "TypeScript" --dir ./crumbs/
```
