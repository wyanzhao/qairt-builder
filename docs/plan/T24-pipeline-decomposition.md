# T24 — Pipeline decomposition and runner dedup

Status: done (2026-08-30)
Depends on: T23 (typed seams first); T22 recommended
Effort: L

## Goal

Break the 8,285-line `pipeline.py` god class into stage modules with the
lane and family logic in their own homes, and collapse the duplicated
docker/apple-container runners onto one base. Pure refactor: behavior,
reports, and the CLI surface must not change.

## Context

Review findings 27 and 31 in
[review-findings-2026-08-30.md](review-findings-2026-08-30.md). `QairtAgent`
spans `pipeline.py:650-8285` with ~110 methods; `validate` ~780 lines,
`benchmark` ~1,220 with a giant nested `run_one` closure; 55 inline "qwen"
mentions carry family logic that belongs beside `families/`. The mitigating
assets — DI seams, a fast fake-driven suite, the T23 Protocol — are exactly
what makes an incremental split safe. Separately, `docker/runner.py` and
`apple_container/runner.py` are ~300 near-duplicate lines (identical
build-arg assembly, env blocks, parallel run functions) so every container
contract change is made twice.

## Design

1. **Stage extraction, one stage per landing.** Move build, validate,
   benchmark, and diagnose bodies into `pipeline_stages/` modules (or
   per-stage classes) that take the typed adapter Protocol and the shared
   run/manifest context. `QairtAgent` remains the facade with unchanged
   public methods — the CLI, jobs, and MCP layers must not notice. Extract
   in dependency order: diagnose (most self-contained) → benchmark →
   validate → build.
2. **Family logic placement.** Inline `qwen`-conditional blocks move behind
   the family profile/preset surface (extended in T22) or into clearly named
   per-lane helpers — no behavioral branching keyed on family strings inside
   stage bodies.
3. **Runner base.** One shared container-runner core (build-arg assembly,
   env block, mount translation, smoke argv) with backend-specific
   subclasses holding only genuine differences (file-bind staging for Apple
   `container`, `--network none` vs `--no-dns`, command word). The existing
   backend-specific tests keep passing against the subclasses.
4. **Refactor discipline.** Every landing keeps the suite green and
   compileall clean; report/JSON output is asserted byte-stable by the T23
   golden tests; no landing mixes a refactor with a behavior change. If a
   real bug is found mid-refactor, it gets its own finding/task first.

## Files

`src/qairt_agent/pipeline.py` → `src/qairt_agent/pipeline_stages/*`,
`src/qairt_agent/families/*`, `src/qairt_agent/docker/runner.py`,
`src/qairt_agent/apple_container/runner.py`, a new shared runner module,
tests updated mechanically (imports/patch targets), `docs/architecture.md`
module map, `CLAUDE.md` if any documented module path changes.

## Acceptance criteria

- `pipeline.py` (facade) is under ~1,500 lines; no extracted stage module
  exceeds ~1,200; the nested `run_one` closure is a named, testable unit.
- Zero report/JSON output change (golden tests) and zero CLI surface change.
- Family-string conditionals no longer appear inside stage bodies (grep
  gate).
- Container build-arg/env/smoke logic exists once; both backends' existing
  tests pass against the shared core.
- Suite, compileall, and the T23 type gate clean after every landing.

## Result

Landed 2026-08-30. Pure refactor: no report, JSON, or CLI change.

**Decomposition.** `pipeline.py` went from 8,285 lines to **943**. Shared
helpers moved to `pipeline_support.py` (re-exported, so existing imports still
work); the stage bodies moved to twelve modules under `pipeline_stages/`, the
largest 1,197 lines: planning, build, qwen35_derivation, validate,
vectors_for_quality, float_reference, benchmark, benchmark_one, optrace,
execution, diagnose, stage_tools.

The extraction is a **verbatim move into mixins** composed onto `QairtAgent`, not
a signature rewrite. Nothing was re-indented into free functions, no `self` was
rebound, every decorator survived, and helpers one stage calls on another
resolve across the MRO exactly as they did in one class body. That is why a
refactor of this size could be verified: the suite is the proof, and it stayed
green through every landing. Deviation from the design, which suggested
functions taking a typed context — mixins were chosen because signature surgery
across ~7,000 moved lines could not have been verified to the same standard.

Three defects the move surfaced and fixed: a hard `QairtAgent.` reference inside
a moved staticmethod (twice), the facade's trailing `__all__` following the
first extraction, and `run_one` capturing `manifest_sha256` from its enclosing
scope.

**`run_one`.** The 870-line closure inside `benchmark` is now
`_benchmark_one`, a named method in its own module with every input as an
argument — the free variable it captured is an explicit parameter.

**Family logic.** The nine family-string conditionals in stage bodies are gone.
`family_registry` declares `has_decoder_lane`, `has_vision_component`,
`has_audio_component`, and `requires_derivation_evidence` on the canonical
records, and the stages ask the capability. Each predicate was checked to give
the identical answer for every family before the branches were switched.

**Runner dedup.** `container_runtime.py` holds what must be identical between
the two backends: harness-path resolution, build-arg assembly, the smoke
environment, and the env flattening. What genuinely differs — Docker's
`--network none` against Apple `container`'s weaker `--no-dns`, the file-bind
staging, the Rosetta flag, the availability probe, the exception type — stays in
each backend, and a test asserts both that the shared parts render identically
and that the isolation difference is still visible. A shared module rather than a
base class: the two classes have different constructors and probes, and
flattening them would have added risk without removing duplication.

**Guarded.** `tests/test_pipeline_decomposition.py` pins the shape: facade and
stage line budgets, the family-conditional grep gate, no stage importing the
facade, the public facade surface, and `run_one` staying a named unit.

Suite 698 passed / 2 skipped (34 new), compileall clean, mypy clean, and
`qairt-agent plan` renders the same keys against the smoke fixture.
