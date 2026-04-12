# OpenCrumb terminal banner patch

This patch package adds an OpenClaw-style splash for `crumb`, but branded with bread:

- keeps the real command as `crumb`
- shows a bread emoji header
- shows a large ASCII `CRUMB` wordmark
- prints on empty launch and `--help`

## Why this is packaged as a patch artifact

The GitHub connector available in this session can create new files, but it cannot update existing files in-place through the exposed file-edit method for this repository. So this branch includes the ready replacement file you can drop in directly.

## File to replace

Replace the repo root `crumb_cli.py` with:

- `patches/opencrumb-banner/crumb_cli.py`

## Resulting terminal vibe

```text
🍞  CRUMB 0.2.0 — AI handoff with bread crumbs

 ██████╗██████╗ ██╗   ██╗███╗   ███╗██████╗
██╔════╝██╔══██╗██║   ██║████╗ ████║██╔══██╗
██║     ██████╔╝██║   ██║██╔████╔██║██████╔╝
██║     ██╔══██╗██║   ██║██║╚██╔╝██║██╔══██╗
╚██████╗██║  ██║╚██████╔╝██║ ╚═╝ ██║██████╔╝
 ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═════╝

              🍞  CRUMB  🍞
```

## Optional next step

If you want the help text itself to feel more branded too, a second follow-up change can be made in `cli/crumb.py` to tweak the parser description.
