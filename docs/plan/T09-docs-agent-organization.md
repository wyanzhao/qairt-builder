# T09 — Docs and agent organization

Status: done (2026-08-29)
Depends on: —
Effort: S

## Goal

Organize the program as agent-executable long-horizon work: a plan board with
self-contained task files, a rewritten root `CLAUDE.md` reflecting the settled
scope, `AGENTS.md` as a symlink to `CLAUDE.md`, and `QWEN.md` removed.

## Delivered (2026-08-29, review session)

- `docs/plan/README.md` — decisions record, status board, execution protocol.
- `docs/plan/review-findings-2026-08-29.md` — condensed evidence base with
  file:line references for the tasks.
- Task files T01–T09.
- Root `CLAUDE.md` rewritten: adds the program scope/plan section
  (primary Qwen3.5, secondary Qwen3, decided measurement scope, out-of-scope
  list, plan pointer) while preserving the operating contract; existing
  behavior statements stay truthful — planned changes live in `docs/plan/`
  until they land.
- `AGENTS.md` converted from a pointer file to a symlink -> `CLAUDE.md`.
- `QWEN.md` deleted (decision: only `AGENTS.md` and `CLAUDE.md` remain).

## Ongoing duties (every future task)

- Keep the status board current; a task is `done` only with its acceptance
  criteria met and gates run.
- Keep root `CLAUDE.md` describing **current** behavior only; aspirational
  statements belong in `docs/plan/`. When a task lands behavior, move the
  description into `CLAUDE.md`/`docs/` in the same change.
- English-only for all landed docs and code.
