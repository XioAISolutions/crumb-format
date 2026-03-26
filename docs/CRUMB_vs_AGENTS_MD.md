# CRUMB vs `AGENTS.md`

Both help agents work better. They operate at different layers.

## `AGENTS.md`

Best for:
- repository-level instructions
- conventions, commands, paths, and workflow hints
- local guidance for coding agents inside a repo

Weak points:
- mostly repo-scoped
- not designed as a small cross-tool handoff block
- often too broad for a single next-step task transfer

## CRUMB

Best for:
- moving work between tools or sessions
- task continuation without a giant transcript
- portable memory and repo-map summaries
- copy-paste handoff in ordinary chats and notes

Weak points:
- intentionally smaller than a full repo instruction system
- does not replace richer local agent files

## Simple rule

Use `AGENTS.md` for **repo-native guidance**.

Use CRUMB for the **portable handoff**.

A good pattern is:
1. keep repo conventions in `AGENTS.md`
2. compile the current task or repo slice into a `.crumb`
3. move that crumb across tools

That gives you local depth plus portable continuity.
