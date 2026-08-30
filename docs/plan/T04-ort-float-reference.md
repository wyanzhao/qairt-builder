# T04 — ORT float reference and layerwise debug comparison

Status: tier 1 done (2026-08-29); tier 1 and tier 2 device acceptance done
(2026-08-30 on SM8750). Low-level tier 2 landed; GenAI tier 2 stays on the
documented fallback.
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

## Tier 1 real-device acceptance (2026-08-30) — done

Run on the SM8750 handset (serial RFCY30B296K, `ro.soc.model SM8750`) from
`models/acceptance/spec-t04-float-reference.json`; job
`20260830T110455Z-b71f5a66`, run `0601fee2-a12d-4a99-bb67-2a5829cc1c79`,
workflow succeeded through build, validate and benchmark.

The published `float_reference_report` records
`comparison: device_chain_vs_onnxruntime_float_graph`, ONNX Runtime 1.17.1 with
`CPUExecutionProvider`, the reference model hash
`b4f52600b4a717ce4fbbd0cb35591029ab766d4750c8e8e7c9b5e5af296d54d7`, an explicit
`tensor_map` of `{"model": {"output": "output"}}` resolved by exact name match,
`unmapped_tensors: []`, and SQNR 41.634 dB / cosine 0.99997 / normalized RMSE
0.00829 at the slice boundary.

**Honest scope of what this proves.** The acceptance model is a single-slice
3-node graph, so it has exactly one boundary and its float reference coincides
with the supplied golden — the same 41.63 dB appears in both. The run therefore
proves the *mechanism* end to end on hardware (ORT capture, name binding,
report publication, artifact immutability), not that the two references diverge
under quantization. A multi-slice model is still needed to show separation
between quantization error and backend error, and that requires the T08 vectors.

The acceptance criterion named SM8850; it was run on SM8750 because that is the
handset attached, and SM8750 is equally verified in the target registry.

## Still open

- **Tier 2 (layer-level drilldown).** Requesting any granularity other than
  `slice_boundary` fails closed today. The blocker is unchanged: the low-level
  lane must actually execute the diagnostic contexts it already builds. T01's
  probe answered the GenAI half negative — 2.49 exposes no intermediate/debug
  output for built containers — so that lane takes the documented fail-closed
  experimental low-level diagnostic build.
- **GenAI lane tier 1.** The implementation is lane-neutral — it compares
  whatever per-slice device outputs the validation run produced — so a GenAI
  raw-tensor route works in principle, but no GenAI-lane test exercises it yet.
- **Multi-slice divergence.** See the scope note above; blocked on
  [T08](T08-aimet-vector-import.md).

## Tier 2 low-level half — landed (2026-08-30)

`granularity: "layer"` is implemented. `QairtAgent._diagnostic_device_outputs`
executes the diagnostic contexts the build has always compiled and hash-verified
but **never ran**, and feeds their tapped tensors into tier 1's existing binding
and comparison path, so the name binding, `unmapped_tensors` discipline and
report shape are shared rather than duplicated.

- **Multi-slice reuses the production chain wiring** — the diagnostic contexts
  are substituted into `SliceChainRunner` with the same routes, so each slice is
  fed exactly what it is fed in a real run instead of a guess at its inputs.
  Several diagnostic contexts with no routes fails closed rather than guessing.
- **No silent degradation.** `granularity: "layer"` without executed diagnostic
  contexts fails closed; a build that produced none says so and names the flag
  to set. Publishing a slice-boundary report under a layer-level label would
  have been the same overclaim in a new place.
- **`op_level_dump_available` now means something.** It was previously set from
  the mere *existence* of a hash-verified context; it is now `true` only when
  tapped tensors were actually collected.
- Observations are sorted by the float graph's topology, so the first
  divergence is the first row.

### Acceptance on SM8750, job `20260830T204543Z-39cd04c7`

One diagnostic context (`vit/tiny`) executed, 3 tensors collected, zero
unmapped:

| tensor | SQNR | cosine |
| --- | --- | --- |
| `h0` (MatMul output) | 41.46 dB | 0.999964 |
| `h1` (bias_add output) | **3.21 dB** | **0.722643** |
| `output` (Relu output) | 41.63 dB | 0.999966 |

**This is the capability working, and it is also a live demonstration of why
the report says `first_observed_divergence_not_root_cause`.** A naive reading
of `h1` at 3.21 dB is "the bias add is broken". It is not: `h1` feeds a Relu,
which clamps the negative half of the range, and that is exactly where the
quantization error lives relative to the signal. The error never reaches the
output, which matches the boundary result to two decimals. Tier 2 surfaces the
divergence; it deliberately does not adjudicate it.

### Coverage gap to be honest about

The multi-slice branch (several diagnostic contexts routed through
`SliceChainRunner`) is implemented and its no-routes guard fails closed, but
**no fixture test exercises the multi-slice success path** — the existing fake
adapters do not emit diagnostic contexts through the build path the pipeline
takes, and the acceptance model is single-slice. A multi-slice model, which
needs [T08](T08-aimet-vector-import.md), is what would prove it.
