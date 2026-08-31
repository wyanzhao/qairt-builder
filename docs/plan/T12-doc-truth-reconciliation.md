# T12 — Documentation truth reconciliation

Status: done (2026-08-30)
Depends on: —
Effort: S

## Goal

Make every landed document tell the truth about landed code again, in one
reviewable patch. This repository is operated by agents that treat `CLAUDE.md`
as authority; a contract that *understates* capabilities is as damaging as one
that overstates them.

## Context

Review findings 1–4 in
[review-findings-2026-08-30.md](review-findings-2026-08-30.md). The code moved
ahead of the contract: layer-level float-reference drilldown and per-slice
chain device capture both landed, but three documents still deny them, one
plan-output key is misnamed in two documents, `docs/mcp-tools.md` contradicts
the latency-is-device-time decision, and two shipped example specs are outside
the examples-resolve test.

## Design

1. `CLAUDE.md`: rewrite the two float-reference passages (program-scope bullet
   and the validation section) to state that `granularity` accepts
   `slice_boundary` and `layer`, with layer granularity requiring executed,
   hash-verified diagnostic contexts and remaining debug-only. Rewrite the
   no-device-meter paragraph: chain scope now records each slice's exact
   inputs during the measured pass and publishes per-slice
   `device_execution`; the GenAI generation scope remains the declared gap.
   Keep the multi-slice-layer-drilldown caveat honest: the success path has
   not run on hardware (finding 18 context, T04 note).
2. `docs/native-workflow.md`: same two corrections (float-reference layer
   granularity, chain device meter).
3. `CLAUDE.md` and `docs/native-workflow.md`: replace
   `effective_config.benchmark` with the real key `effective_benchmark`, and
   add one clarifying sentence that `effective_config` in plan output is an
   output-layout role, not the benchmark block.
4. `docs/mcp-tools.md`: replace the "warmed production-wall latency" sentence
   with the device-time contract (production latency is
   `accelerator_compute_us`; wall time is `harness_diagnostics` only).
5. `tests/test_presets.py`: add
   `examples/qwen3_dense_float_reference_debug.json` and
   `examples/qwen3_dense_float_reference_layer_debug.json` to the
   examples-resolve parametrization (they must parse as `WorkflowSpec` and
   resolve).
6. Opportunistic, same patch: align `tools/make_smoke_fixture.py`'s default
   `--target` with the harness-active target (or make the flag required) so a
   first run cannot silently plan for the non-active chip.
7. Affected skills: check the five `.claude/skills/` files for the same stale
   claims; the two diagnosis skills reference the drilldown and chain meters.

## Files

`CLAUDE.md`, `docs/native-workflow.md`, `docs/mcp-tools.md`,
`tests/test_presets.py`, `tools/make_smoke_fixture.py`,
`.claude/skills/qairt-diagnose-quality/SKILL.md`,
`.claude/skills/qairt-diagnose-latency/SKILL.md`.

## Acceptance criteria

- `grep -rn "not available yet\|has not landed\|slice_boundary.*only" CLAUDE.md
  docs/` returns no claim contradicting `pipeline.py:2041` (layer granularity)
  or the chain device capture (`pipeline.py:5029`).
- `grep -rn "effective_config.benchmark" CLAUDE.md docs/` is empty; the plan
  section names `effective_benchmark`.
- `docs/mcp-tools.md` contains no wall-latency claim about benchmark reports.
- The two float-reference examples are parametrized in the examples-resolve
  test and the suite passes.
- No behavior change outside `tools/make_smoke_fixture.py`'s default-target
  handling; `.venv/bin/pytest -q` and `compileall` clean.

## Result

Landed 2026-08-30. `CLAUDE.md` and `docs/native-workflow.md` now state that
`granularity` accepts `slice_boundary` and `layer` (layer requiring an executed,
hash-verified diagnostic context per slice, with the unexercised multi-slice path
named), and `CLAUDE.md` no longer lists chain scope as meterless — it describes
the recorded-input per-slice capture that landed. Both documents name
`effective_benchmark` and warn that the plan key `effective_config` is an
output-layout role; `CLAUDE.md` also places `measurement_scope` inside
`harness_diagnostics` where the code publishes it. `docs/mcp-tools.md` states the
device-time contract instead of "warmed production-wall latency".
`tests/test_presets.py` now resolves the two float-reference debug examples, and
`tools/make_smoke_fixture.py` reads its default target from
`harness/constraints.json` (the hardcoded `sm8750` disagreed with the active
`sm8850`), covered by a new test. The five skills were checked: only the
multi-slice drilldown caveat was missing, added to `qairt-diagnose-quality`.
Suite 555 passed / 2 skipped, compileall clean.
