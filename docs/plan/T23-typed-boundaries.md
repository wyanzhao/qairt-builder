# T23 — Typed boundaries: adapter Protocol and report contracts

Status: done (2026-08-30)
Depends on: —
Effort: M-L

## Goal

Give the two largest untyped surfaces real types: a `Protocol` for the
pipeline↔adapter boundary (with a type checker configured to enforce it) and
pydantic models for the report payloads that today are schema-string-tagged
dicts. This is the enabling step for T24's decomposition — a monolith can be
split safely only along typed seams.

## Context

Review findings 28–29 in
[review-findings-2026-08-30.md](review-findings-2026-08-30.md). The pipeline
consumes ~30 adapter methods through `adapter_factory: Callable[[], Any]`
(`pipeline.py:657`); the only Protocol in the codebase is
`runtime.chain.SliceExecutor`; `tests/test_pipeline.py` hand-maintains a
500+ line `FakeAdapter`, so signature drift between adapter, fake, and
call sites is caught only by eye. Report payloads
(`multi-ar-sqnr-report.v1`, `optrace-evidence.v1`,
`float-reference-report/1`, ...) are dicts validated by scattered
isinstance guards (`pipeline.py:7369`), unlike the typed
spec/manifest contracts.

## Design

1. **`QairtAdapterProtocol`.** Define the consumed surface as a
   `typing.Protocol` next to the adapter types; annotate
   `adapter_factory` and the fakes with it. The real adapter and the test
   fake are both checked against the same Protocol.
2. **Type checker.** Configure mypy or pyright (pick one, pin it in dev
   extras) scoped initially to the Protocol boundary, `contracts.py`,
   `families/`, and `diagnostics/` — not a big-bang whole-repo strictness
   jump. Gate in CI/dev checks alongside pytest/compileall.
3. **Typed reports.** Model each published report schema as a pydantic model
   in `contracts.py` (or a `contracts_reports.py` sibling): construction
   sites in `pipeline.py` build the model then dump; consumption sites
   (`_quality_divergence_attributions`, compare in T17) parse the model
   instead of isinstance-walking. Schema-version strings stay — they become
   a field with a literal type per version. On-disk JSON stays byte-stable
   (assert with golden serialization tests) so existing manifests remain
   readable.
4. Migrate incrementally by report family (sqnr → latency → footprint →
   float-reference → optrace), each landing green; do not hold the task open
   for a full sweep if a later family is better done inside T24.

## Files

`src/qairt_agent/qairt_adapter/types.py` (Protocol),
`src/qairt_agent/pipeline.py`, `src/qairt_agent/contracts.py` (+ possible
`contracts_reports.py`), `pyproject.toml` (type-checker dev dependency +
config), tests (`test_pipeline.py` FakeAdapter annotation, golden
serialization tests), `CLAUDE.md` development-checks section (the new gate).

## Acceptance criteria

- A deliberate signature change in the real adapter fails the type check
  against the Protocol (and the fake fails the same way) before any test
  runs.
- At least the SQNR and latency report families are constructed and consumed
  through typed models, with golden tests proving on-disk JSON is unchanged.
- The type-check gate is documented in `CLAUDE.md` and runs with the standard
  development checks.
- No behavior change; suite clean.

## Result

Landed 2026-08-30. No behavior change; the on-disk JSON is asserted byte-stable.

**`QairtAdapterProtocol`.** Declared in `qairt_adapter/types.py` for the surface
the pipeline always calls, with `QairtAdapterOptionalProtocol` for the six it
probes with `hasattr`/`getattr` and degrades without (a missing
`capture_device_execution` publishes `available=false` with a reason). Signatures
are permissive where the real method takes many keyword-only options: what this
catches is a method that vanished or was renamed, which is the drift that used
to go unnoticed. `QairtAgent.__init__` and `_new_adapter` are annotated with it.

The Protocol immediately found two real gaps: the pipeline calls
`build_qwen35_omni_components`, which no boundary declared, and `FakeAdapter`
never implemented `ar_convert`, `transform`, `convert`, or `quantize` — the
low-level stage tools it claims to stand in for. Both are fixed, and a test
greps `pipeline.py` for adapter calls to keep the declaration honest.

**Type gate.** mypy 2.3 pinned in the `dev` extra and configured in
`pyproject.toml`, scoped to `contracts.py`, `contracts_reports.py`,
`family_registry.py`, `families/`, `diagnostics/`, and the Protocol module. It
runs clean (15 files) and is documented in `CLAUDE.md`'s development checks. Six
genuine type defects were fixed to get there, including a `**kwargs` splat in
the T13 cross-check that mypy could not verify and an `Any | None` fed to
`int()` in target-tuple resolution.

**Typed reports.** `contracts_reports.py` models `multi-ar-sqnr-report.v1`,
`multi-ar-latency-report.v1`, and the `device-execution/2` block. The aggregates
are now **constructed** through their model in `pipeline.py` (validate, then
dump) and `compare.py` **consumes** the device block through its model instead
of isinstance-walking. Models allow and preserve extra keys deliberately: a
published report is content-addressed evidence, so a lossy round trip would make
its recorded hash unreproducible — proven by a test that adds an unmodelled key
and by golden tests that build real reports through the pipeline and assert
canonical byte equality.

Per the task's incremental rule, the per-AR reports the aggregates wrap, plus
the optrace and float-reference families, stay plain payloads until their own
landing.

Suite 664 passed / 2 skipped (25 new), mypy clean, compileall clean.
