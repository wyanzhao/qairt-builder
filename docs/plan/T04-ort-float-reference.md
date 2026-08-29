# T04 — ORT float reference and layerwise debug comparison

Status: tier 1 done (2026-08-29); tier 2 still planned, and the real-device
acceptance run is still owed.
Depends on: T01 (tier 2 capability answer; device acceptance) — tier 1 code
can be developed against the fake-adapter seams earlier
Effort: L

## Goal

Add an **ONNX Runtime float-graph reference** as an explicit second reference
beside AIMET goldens, enabling on-target vs float comparison at slice
boundaries and — in a deeper debug tier — layer by layer.

**Policy (decided): this is a debug-only mode.** It is never enabled by
default, production validation output is unchanged unless the debug config is
explicitly present, and all its reports remain report-only (no verdicts).

## Context

- Today ORT is only a *fallback* used when a manifest has inputs but no
  goldens (`pipeline.py:3733-3771`); it cannot coexist with supplied AIMET
  goldens. The decided methodology wants both: device vs AIMET-quantsim golden
  (production) and device vs ORT float (debug, error attribution against the
  float graph).
- Diagnostic contexts are built and verified but never executed
  (`pipeline.py:3652-3730`, `6812-6820`) — tier 2 closes exactly that gap.
- Slice-boundary activations ("boundary tensors") are the activations at the
  split points (embedding → decoder_00 → ... → lm_head). The ORT float run
  captures them; no extra export is required from the AIMET side.

## Design

### Tier 1 — slice-boundary comparison (standard debug entry)

1. New explicit debug config, e.g.
   `stage_configs.validation.float_reference = {"granularity":
   "slice_boundary", "ar": <int>, ...}` (exact schema is the implementer's
   choice; it must be content-addressed like other stage configs, off unless
   present, and single-AR — a debug run pins one AR explicitly, consistent
   with the existing single-AR debug override philosophy).
2. Host side: run the **float source ONNX** (the AR-matching export; encodings
   are not applied) under ONNX Runtime, capturing boundary activations by
   promoting the mapped internal tensors to graph outputs in an in-memory
   model copy. Record model hash, ORT version, providers — same provenance
   discipline as the existing fallback capture.
3. Feed the captured boundaries as the per-slice reference set through the
   existing `teacher_forced`/`chain` machinery, labeled
   `reference_source = "onnxruntime_float"`. AIMET-golden `full_reference`
   results are reported alongside, never replaced. Where both references
   exist for the same tensor, the report may include the derived
   float-vs-golden metrics so quantization error and backend error separate.
4. Lanes: low-level (Qwen3) uses transformed-slice boundaries as today; GenAI
   (Qwen3.5) uses the container raw-slice routes from the runtime index. If
   the container exposes no auditable tensor route, the debug mode fails
   closed exactly like GenAI SQNR does.

### Tier 2 — layer-level drilldown (deep debug)

1. Device side needs intermediate tensors:
   - **Low-level lane:** execute the already-built diagnostic contexts
     (selective `set_output_tensors` taps preferred; full
     `enable_intermediate_outputs` dumps only with an explicit tensor list or
     AR1 — CL4096 full dumps are large). This adds the missing
     "run diagnostic context and collect tensors" stage.
   - **GenAI lane:** gated on the T01 capability probe ("does 2.49 expose a
     debug/intermediate-output option for built containers?"). If yes, use it;
     if no, the documented fallback is an explicit, fail-closed experimental
     low-level diagnostic build for Qwen3.5 — a separate opt-in, never
     automatic.
2. Name alignment: map source-graph tensor names to transformed/on-device
   names using the transform lineage/tracing info (the same evidence class the
   diagnosis path already requires; never name heuristics). MHA2SHA head
   splits map N:1 — aggregate per mapped group and say so in the report.
   Unmappable tensors are listed explicitly (`unmapped_tensors`), never
   guessed — same philosophy as `unresolved_external_inputs`.
3. Report: per-tensor SQNR/RMSE/cosine against the ORT float value, ordered by
   graph topology, `claim_scope: "first_observed_divergence_not_root_cause"`,
   `reference_source: "onnxruntime_float"`, granularity and tap list recorded.
   Diagnostic-context latency is never reported as production latency
   (existing rule).

## Out of scope

- Enabling any of this by default, or letting it touch production reports.
- Token-level metrics (program out-of-scope decision).
- Multi-AR fan-out of debug runs.

## Files

`contracts.py` (debug config schema + report fields), `pipeline.py`
(validate/debug paths, diagnostic-context execution), `vectors.py` (ORT
intermediate capture), `qairt_adapter/adapter.py` (diagnostic-context run
support if a new call shape is needed), lineage plumbing from transform
results, new example `examples/` debug spec (marked debug-only in
`examples/README.md`), tests (unit + fake-adapter pipeline tests), root
`CLAUDE.md`, `docs/architecture.md`.

## Acceptance criteria

- With no `float_reference` config, byte-identical validation behavior
  (regression tests prove it).
- Tier 1 on the fake-adapter seam: boundary capture, dual-reference report,
  fail-closed on missing routes — all tested.
- Tier 2 low-level: a diagnostic-context execution path with selective taps,
  lineage-mapped comparison, unmapped tensors listed; tested against fakes.
- GenAI tier 2 decision recorded here after T01's probe (implemented, or
  fallback documented + experimental path spec'd).
- One real-device debug run on SM8850 producing a slice-boundary float
  comparison report (after T01).
- Docs updated: debug-only status stated everywhere the mode is mentioned.

## Tier 1 result (2026-08-29)

Landed as `stage_configs.validation.float_reference`, handled by
`QairtAgent._float_reference_report` and
`VectorPreparer.capture_onnx_float_activations`.

- **Off by default, provably.** The whole path is behind one config key; with
  it absent the payload has no `float_reference`, no artifact is published, and
  the stage metric `float_reference_debug` is `false` — all asserted by
  `test_validation_without_the_debug_config_publishes_no_float_reference`, and
  the 496 pre-existing tests continued to pass unchanged when the feature
  landed.
- **Activation promotion without touching the model.** Requested tensors are
  appended as graph outputs on an in-memory proto copy. A self-contained model
  runs from serialized bytes; a model with **external data** cannot, because a
  byte-loaded model cannot resolve relative initializer locations. That case
  materializes the instrumented copy beside the original (the only place those
  relative paths resolve) and always removes it, failing closed with a clear
  message when the directory is not writable. Both branches are tested,
  including that the on-disk model's outputs are unchanged afterwards.
- **Names are bound, never guessed.** A device tensor binds to a float tensor
  by exact name match or by an explicit `tensor_map` entry; the map may be
  flat or nested per slice. Everything else lands in `unmapped_tensors` with
  the reason, and a run that binds nothing fails closed instead of publishing a
  partial comparison. This is the same discipline as
  `unresolved_external_inputs` and is what makes the mode safe to run against
  an export whose lineage the caller has not fully mapped.
- **Additive, not substitutive.** The AIMET golden comparison, its
  `reference_source`, and the mode reports are untouched; the float result is a
  separate `float_reference_report` artifact plus one payload key.
- **Single-AR by construction.** `float_reference.ar` is required and must
  agree with the bound runtime AR, matching the existing single-AR debug
  override philosophy.

Tests: `test_float_reference_compares_device_boundaries_with_the_float_graph`,
`test_float_reference_lists_unmapped_boundaries_instead_of_guessing`,
`test_float_reference_uses_an_explicit_tensor_map`,
`test_float_reference_fails_closed_on_debug_misuse`,
`test_validation_without_the_debug_config_publishes_no_float_reference`,
plus `tests/test_vectors.py`
`test_float_capture_resolves_external_data_and_restores_the_directory` and
`test_producible_tensor_names_include_internal_activations`. Example:
`examples/qwen3_dense_float_reference_debug.json`, labelled debug-only in
`examples/README.md`. Documentation: root `CLAUDE.md` and
`docs/native-workflow.md` ("Float-graph reference (debug only)").

## Still open

- **Tier 2 (layer-level drilldown).** Requesting any granularity other than
  `slice_boundary` fails closed today. The blocker is unchanged: the low-level
  lane must actually execute the diagnostic contexts it already builds, and the
  GenAI lane needs T01's capability probe.
- **GenAI lane tier 1.** The implementation is lane-neutral — it compares
  whatever per-slice device outputs the validation run produced — so a GenAI
  raw-tensor route works in principle, but no GenAI-lane test exercises it yet.
- **Real-device acceptance.** No SM8850 run has produced a slice-boundary float
  comparison report; the acceptance criterion above still stands.
