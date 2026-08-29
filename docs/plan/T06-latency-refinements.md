# T06 — Latency refinements (GenAI defaults, token accounting)

Status: done (2026-08-29) — GenAI lane resolves 3+10 through both spec entry
points, reports label the token source and wall scope.
Depends on: — (metrics enrichment consumes T01's probe answer when available)
Effort: S

## Goal

Keep the decided methodology (host wall-clock + warmed production contexts +
optrace attribution) and fix its cost/ergonomics for the GenAI lane.

## Context

- Defaults `warmup_runs=10, measured_runs=50` plus A/A calibration (on by
  default) mean one GenAI benchmark performs **180 full `generate()` calls**
  (`contracts.py:453-458`, `pipeline.py:5443-5456`) — far too expensive for
  LLM generation on-device.
- `p50_ms_per_token` exists only when the caller supplies `token_count`
  (`pipeline.py:5895-5903`). Prefill/decode split is not observable through
  the public executor (SDK limitation; reports already state it).

## Design

1. **GenAI-lane defaults**: effective `warmup_runs=3, measured_runs=10` when
   the resolved lane is `genai_generation` and the spec does not set explicit
   values. Resolution happens at plan time so `qairt-agent plan` shows the
   effective numbers; explicit spec values always win. A/A stays available;
   its doubled cost is documented next to the default.
2. **Token accounting**: keep `token_count` explicit (no tokenizer dependency
   in the agent). Additionally, when the SDK's `generation_metrics`
   passthrough contains a trustworthy generated-token count (T01 probe
   answer), surface `ms_per_token_source = "sdk_metrics" | "caller"` and
   compute from the SDK value only when the caller did not supply one. Never
   mix sources silently.
3. **Documentation**: state plainly in the report and docs that wall samples
   include host->SDK->device round trip (no device-side sync barrier exists;
   the `synchronize` hook stays unwired unless the SDK grows one), and that
   optrace cycles remain the on-device attribution evidence.

## Out of scope

- Prefill/decode split or TTFT via the public executor (SDK-limited; revisit
  only if T01's probe finds new metrics).
- Changing low-level lane defaults (10/50 stays).
- Power/thermal (program out of scope).

## Files

`contracts.py` (default resolution note), `pipeline.py` (lane-aware effective
defaults + token source), `diagnostics/latency.py` (untouched math), tests
(`test_latency.py`, `test_pipeline.py` effective-default assertions), docs
(`docs/native-workflow.md`, root `CLAUDE.md` one line), examples with explicit
prompts unchanged.

## Acceptance criteria

- GenAI lane with defaults plans/executes 3+10 (and 2x that with A/A);
  explicit spec values override; low-level lane unchanged — all asserted in
  tests.
- `plan` output shows effective benchmark numbers per lane.
- Token metric source is explicit in the report; no silent SDK/caller mixing.

## Result (2026-08-29)

1. **Lane-aware defaults, resolved once — in one shared place.**
   `families.presets.apply_lane_benchmark_defaults` materializes
   `warmup_runs=3, measured_runs=10` for a GenAI-lane family, reusing the
   pattern already established for `compile.enable_intermediate_outputs`.
   Resolution must happen where the caller's input is still distinguishable
   from the schema default: pydantic's `model_fields_set` separates "the caller
   chose 10" from "10 is the default" only before a manifest round-trip, after
   which every field reads as set and a later resolution point would silently
   stop overriding. It is per field, so a spec that sets only `warmup_runs`
   still gets the lane's `measured_runs`. The lane comes from the preset
   registry (`family -> preset_id_for_family -> get_preset().pipeline`) rather
   than a second copy of the routing table. `benchmark` is not in the worker's
   `_BUILD_SPEC_REUSE_FIELDS`, so this does not invalidate build reuse.

   The first implementation put this only in `QairtAgent._parse_spec`, which
   was wrong in a way the API-level tests could not see: the **CLI**
   `qairt-agent plan` builds its payload through
   `presets.to_build_spec(workflow_spec)`, a second entry point that never
   reaches `_parse_spec`. It would have printed 10/50 for a run that executes
   3/10 — precisely the plan-versus-execution drift this task's acceptance
   criterion exists to prevent. Both entry points now call the same helper, and
   `effective_benchmark_policy` renders the same block for both, so the two
   cannot diverge again.
2. **Token accounting — the SDK answer is already known.** T06 deferred this to
   T01's probe, but the 2.49 source settles it: `GenerationMetrics`
   (`qairt/gen_ai_api/executors/gen_ai_executable.py`) exposes
   `token_generation_rate` and `token_generation_time` and **no token count**;
   `_parse_execution_metrics` never reads the `num-generated-tokens` record the
   underlying Genie profile carries. So `sdk_metrics` is unreachable on this
   SDK, and `p50_ms_per_token` comes from the caller's `token_count` alone.
   `ms_per_token_source` is now always published next to it.
   `_sdk_generated_token_count` reads only an explicitly reported count and
   returns `None` for a rate/duration pair — deriving `rate x time` would
   manufacture a number the SDK never reported. If a later SDK surfaces a
   count, the mechanism activates without further change; the guard against
   fabricating one stays.
3. **Measurement scope stated in the report.** Every latency payload carries a
   `measurement_scope` block (`clock`, `includes`,
   `device_side_sync_barrier: false`, `sample_unit`, and the reason), so the
   host round-trip caveat travels with the numbers instead of living only in
   documentation.

Tests: `test_genai_lane_resolves_cheaper_benchmark_defaults` (plan output,
recorded BuildSpec, and an executed 3+10 GenAI benchmark),
`test_explicit_benchmark_values_beat_the_genai_lane_defaults`,
`test_low_level_lane_keeps_the_original_benchmark_defaults`,
`test_latency_report_labels_the_token_metric_source_and_wall_scope`,
`test_sdk_generated_token_count_never_derives_from_a_rate`, plus
`tests/test_cli.py` `test_plan_shows_the_genai_lane_benchmark_defaults` and an
`effective_benchmark` assertion in `test_plan_resolves_preset` covering the CLI
surface. Documentation: root `CLAUDE.md` and `docs/native-workflow.md` (new
"Benchmark sampling and token accounting" section).
