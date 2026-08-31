# Code review findings — 2026-08-30

Condensed evidence base for plan tasks T12–T24. An eight-dimension full-repo
review (build pipeline, model routing, target registry, measurement
correctness, maintainability, docs/onboarding, regression diagnosis, API
boundary) produced 102 findings; the highest-impact gap/bug/risk claims were
independently re-verified against the code. File:line references are as of
commit `c9f551c`.

## Confirmed capabilities (verified against code, not docs)

These claims were checked in code and hold. They are recorded so later tasks
do not re-litigate them.

- Both lanes are complete, real code. The low-level lane chains inspection →
  per-AR/CL `GraphContext` conversion → `split_llm`+MHA2SHA transform →
  `qairt.convert` with AIMET encodings → weight-shared `qairt.compile` per
  semantic slice, with native-KV config sourced live from the SDK and separate
  diagnostic contexts. The GenAI lane drives
  `Qwen3_5BuilderHTP.from_pretrained` with fail-closed per-AR attached
  models+encodings through build/save and extracts public raw slices for
  tensor SQNR.
- SQNR/RMSE/cosine math is numerically correct with tested zero-signal,
  zero-noise, and NaN edges. Supplied goldens always win; the ORT fallback is
  fully audited (model hash, ORT version, providers); neither-goldens-nor-
  inputs fails closed.
- Latency is genuinely device time: `accelerator_compute_us` from ten
  `level="detailed"` profiled executes with spread and per-sample values kept,
  capture forced before `initialize_execution`, wall time quarantined under
  `harness_diagnostics` with `not_latency: true`.
- Target resolution is strict (exact name or full tuple, no default, spec-time
  failure); the SM8750 empty-`device_custom_configs` guard exists and is
  tested on the low-level compile path; no soc constant is hardcoded in `src/`
  outside registry data.
- The no-vendor-CLI boundary holds: every `subprocess` call site in `src/` is
  adb, docker/apple-container, a python worker, or the pickle-isolation child;
  all SDK work goes through importlib-loaded QAIRT Python modules.
- Context length is not pinned to 4096: `sequence.context_lengths` is a spec
  tuple and any CL divisible by 256 under native KV (e.g. 8192) flows through
  the build path (`src/qairt_agent/contracts.py:271-319`).
- `docs/first-run.md` is accurate to the code down to error strings and JSON
  shapes; the five skills describe only capabilities that exist; examples and
  configs are resolved by tests.

## Confirmed issues (drive tasks T12–T24)

Ordering is by theme, not severity. "(re-verified)" marks claims that an
independent adversarial pass confirmed with its own evidence.

### Documentation truth drift

1. **CLAUDE.md denies two shipped capabilities** (re-verified). Layer-level
   float-reference drilldown landed (commit `ffed2c9`): `pipeline.py:2041`
   accepts `granularity="layer"`, `pipeline.py:1928-2009` executes diagnostic
   contexts, `tests/test_pipeline.py:2490-2538` covers it, and
   `docs/first-run.md:163-192` plus
   `examples/qwen3_dense_float_reference_layer_debug.json` demonstrate it —
   but `CLAUDE.md:30-31` and `CLAUDE.md:276-277` still say layer-level
   drilldown "has not [landed]" / "is not available yet", and
   `docs/native-workflow.md:320-322` says "not wired up yet". Similarly the
   benchmark records per-slice chain device capture
   (`pipeline.py:5029-5107`) while `CLAUDE.md:~345` still lists chain scope as
   a no-device-meter scope. → T12.
2. **Plan-output key misnamed in two docs**. `CLAUDE.md:384` and
   `docs/native-workflow.md:330` tell the reader to check
   `effective_config.benchmark`; the CLI publishes `effective_benchmark`, and
   a key literally named `effective_config` *does* exist in plan output as an
   output-layout role, so the wrong name resolves to the wrong object. → T12.
3. **`docs/mcp-tools.md:155` still says benchmark reports "warmed
   production-wall latency"**, contradicting the T10/T11 latency-is-device-time
   contract every other document enforces. → T12.
4. **The two float-reference debug examples are outside the examples-resolve
   test.** `tests/test_presets.py:251` parametrizes six example files;
   `examples/qwen3_dense_float_reference_debug.json` and
   `examples/qwen3_dense_float_reference_layer_debug.json` are parsed by no
   test and can rot silently. → T12.

### Identity and trust guards

5. **The declared preset is never cross-checked against the supplied HF
   config.** `resolve_family_profile` short-circuits when a family is passed
   (`src/qairt_agent/families/profiles.py:257`), and the pipeline always
   passes `family=spec.family.value` (`pipeline.py:1346`), so the
   architecture-detection code that exists is never consulted on the build
   path. A mis-declared export (a Qwen3.5 hybrid export declared
   `qwen3_dense`) silently bypasses every family-specific gate. → T13.
6. **The GenAI lane has no analogue of the compiler-target guard**
   (re-verified). `_validate_compiler_target`
   (`src/qairt_agent/qairt_adapter/adapter.py:1114-1147`) is invoked only from
   the low-level `compile_context` (`adapter.py:1323`). GenAI paths call
   `builder.set_targets([target_spec])` (`adapter.py:2352`, vision `:2408`,
   Omni `:2869/:2871`) and record the *input* registry tuple
   (`adapter.py:2546-2550`), never anything the SDK resolved. → T13.
7. **GenAI raw-slice AR→graph binding is positional** (re-verified).
   `adapter.py:1960-1964` binds `zip(ar_values, graphs)` when counts match;
   the ABI check at `:1980-1996` compares tensor names only, so an order
   inversion is undetectable. The binding flows into
   `runtime/index.py:253-263` and `runtime/chain.py:118-125`. Unlike the
   MHA2SHA start points, this SDK ordering assumption has no reviewed
   fingerprint. → T13.
8. **Nothing verifies the ADB-attached handset is the registered SoC.** The
   registry records per-target Android `soc_id` lists for exactly this
   purpose (`harness/targets/*.json`), but no code reads
   `/sys/devices/soc0/soc_id` or `ro.soc.*` (grep across `device/`,
   `runtime/`, `diagnostics/`), and the device doctor's target check is a
   tautology (`src/qairt_agent/device/doctor.py:193`). With
   `QAIRT_AGENT_ADB_SERIAL` pointing at the wrong handset, reports publish
   under the wrong target identity. → T14.

### CLI, jobs, and stage-tool robustness

9. **Invalid specs escape as raw pydantic tracebacks.** `cli.main` catches
   only `QairtAgentError` (`src/qairt_agent/cli.py:1394`), but spec
   normalization calls `model_validate`/`json.loads` unwrapped
   (`src/qairt_agent/agent.py:221`), so a typo field or malformed JSON prints
   a traceback to stderr instead of the JSON error contract. → T15.
10. **`job watch --follow` hangs forever after an uncleanly dead worker.**
    `_follow` loops until `status.state.terminal` (`cli.py:684`), terminal
    excludes ORPHANED (`contracts.py:1355`), and the watch path never checks
    heartbeat staleness — `mark_orphaned_if_stale` runs only when a new worker
    attempts the job. → T15.
11. **MCP `submit_job` runs the job in-process**, in a daemon thread of the
    MCP server (`src/qairt_agent/mcp_server.py:79`,
    `src/qairt_agent/agent.py:414`), bypassing the detached pinned-container
    worker the contract mandates; on macOS that is native execution, and a
    server exit kills the job with no journal finalization. It also defaults
    to `stages=('build',)` while describing itself as submitting a workflow.
    → T15.
12. **Standalone `compile_context` stage tool cannot succeed with JSON
    native-KV expectations.** `pipeline.compile_context` reconstructs
    `NativeKvGraphExpectation` from JSON without `model_path`
    (`pipeline.py:4841`), but the adapter's audit path requires it
    (`qairt_adapter/types.py:189`) and raises `NativeKvConfigError`. → T15.

### Measurement report honesty

13. **The multi-AR aggregate latency report hardcodes
    `latency_metric="device_execution"`** (`pipeline.py:7321`) even when a
    per-AR capture degraded to `available=false` (`pipeline.py:940-946`,
    `:6570-6577`), so the aggregate label can overstate per-AR evidence. → T16.
14. **`chain_sequence` device evidence covers only the last step per slice.**
    `_recording_chain_executors` overwrites `recorded[slice]` per invocation
    (`pipeline.py:5052`), so after prefill+decode only the final step's inputs
    survive; `_chain_device_execution` (`pipeline.py:5107`) then publishes
    scope `"chain"` with no step/AR provenance. → T16.
15. **`aggregate_device_executions` can average partial samples without an
    explicit marker**, and non-finite device outputs abort validation with an
    exception instead of a localizing quality report. → T16.

### Regression detection

16. **`qairt-agent diagnose` never detects a regression.**
    `_automatic_diagnosis` (`pipeline.py:7838`) triggers the quality path on
    any observation with `noise_energy > 0` — i.e. on every healthy quantized
    run — so the latency delta path is unreachable after any validate stage,
    and `stage_configs.diagnose.kind="latency"` is silently ignored in
    automatic mode (`pipeline.py:8184`). → T17.
17. **No cross-run comparison exists for headline metrics.** Detecting a
    `production_latency_us` or SQNR regression between two runs is manual
    report-diffing; the only automated delta is the hard-to-reach op-cycle
    path. → T17.
18. **Layer/operator quality attribution needs a hand-built map.** Device-to-
    float binding is exact-name or explicit `float_reference.tensor_map`
    (correctly never guessed — `pipeline.py:2109`), and `layer_attributions`
    come only from caller-supplied lineage (`pipeline.py:5756`); no tool
    extracts the MHA2SHA/transform name mapping the caller needs. → T17
    (stretch) / recorded for a future task.

### Onboarding gaps

19. **No skill or landed runbook for the golden-pickle import** — the
    mandatory first step for the primary model's real vectors. The runbook
    text exists only inside blocked task T08
    (`docs/plan/T08-aimet-vector-import.md:28`); `.claude/skills/` has no
    entry for it. → T18.
20. **No consolidated spec-authoring reference** (re-verified). Field
    knowledge is scattered: `quality.sqnr_modes` only in
    `docs/first-run.md:212` / `examples/README.md:96`,
    `slice_vector_manifests` only in `examples/README.md:99`, the benchmark
    prompt requirement across three files, `float_reference` only in
    `docs/native-workflow.md:278-315`. There is no `docs/` spec reference and
    no `plan --schema`. → T18.
21. **AR/CL/native-KV decisions for non-linear-attention models have no
    decision support.** Policy is preset defaults + spec overrides + `plan`
    preview (by design, agent-native); nothing guides a human or agent through
    choosing the AR set, CL, or native-KV for a wide export. → T18.

### Packaging and registry seams

22. **`pyproject.toml` force-includes version- and target-pinned filenames**
    (`harness/targets/sm8850.json`, `sm8750.json`,
    `docker/requirements-qairt-2.49.0.260730.txt`) — renaming the lock or
    adding a target ships a wheel missing the file, and neither the upgrade
    procedure nor the two skills mention it. → T19.
23. **`tests/test_harness.py` hardcodes the registry population.**
    `test_harness.py:164` asserts `set(registry) == {"sm8750", "sm8850"}` and
    `:206` requires every entry verified with the pinned `sdk_build`, so the
    documented add-target intermediate state (entry committed, acceptance
    pending) fails the suite. → T19.

### Real-model scale

24. **No real model has ever been through either lane** (re-verified). Every
    hardware acceptance record traces to the generated smoke fixture: both
    `harness/targets/*.json` verified blocks record the fixture's 41.63 dB,
    which `docs/plan/T01-sdk-upgrade-2.49.md:317-334` identifies as the tiny
    64×32 MatMul graph; `docs/plan/T11-device-only-latency.md:145` says its
    chain fixture came from `tools/make_smoke_fixture.py --chain`; the GenAI
    half is blocked on T08. Real-model-scale behaviors are unexercised. → T20
    (preparation), T21 (the run itself).
25. **The build stage retains every live SDK object for the whole run**
    (re-verified). `adapter.py:248/:347/:406` attach live
    `graph_context`/`sdk_model` objects to artifacts; `adapter.build`
    accumulates them across the whole CL×AR×slice loop
    (`adapter.py:3290-3296`, appends at `:3349/:3374/:3465`) and returns them
    all (`:3817-3825`); the pipeline holds the full result through vector
    prep/route publishing/footprint (`pipeline.py:3679-3750`). No release
    anywhere. Crash recovery is workflow-stage granular; the monolithic build
    restarts from zero. → T20.
26. **Every continuation stage re-hashes all cumulative artifacts**, a real
    wall-time cost at model scale (tens of GB per stage boundary). → T20.

### Structure (long-horizon)

27. **`pipeline.py` is an 8,285-line single-class monolith**: `QairtAgent`
    spans `pipeline.py:650-8285` with ~110 methods; `validate` is ~780 lines
    (`:5352-6134`), `benchmark` ~1,220 (`:6134-7358`) with a giant nested
    `run_one` closure; 55 "qwen" mentions inline family logic that belongs in
    `families/`. Mitigations exist (DI seams, fast fake-driven tests), which
    is what makes a split feasible. → T24.
28. **The pipeline↔adapter boundary is `Any`-typed** with no `Protocol` and no
    type checker configured; `tests/test_pipeline.py` re-implements the ~30
    method surface by hand in a 500+ line `FakeAdapter`, so signature drift is
    caught only by eye. → T23.
29. **Report payloads are untyped dicts** with schema-string tags
    (`multi-ar-sqnr-report.v1`, ...) validated by scattered hand-rolled
    isinstance guards (`pipeline.py:7369`), while specs/manifests are typed
    pydantic models. → T23.
30. **Family identity lives in four hand-synced registries** with diverging
    alias sets: `contracts.ModelFamily` (`contracts.py:70`, alias `qwen3_4b`),
    `families/profiles.FamilyId` (`profiles.py:131`, `qwen_3`, no VIT),
    `vector_retarget._FAMILY_ALIASES` (`vector_retarget.py:139-194`,
    `qwen3-4b` only here), and `presets._PRESET_TO_FAMILY`. Adding a family
    touches 5–6 files with no documented procedure. → T22.
31. **`docker/` and `apple_container/` runners are ~300 near-duplicate
    lines** (`docker/runner.py:177`, `apple_container/runner.py:361`): same
    build-arg logic, same env block, parallel run functions; every mount/env
    contract change is made twice. → T24.

### Recorded, no task (accepted or low)

- `docker/smoke.py` executes the SDK's own `bin/check-python-dependency`
  script inside the worker — the single borderline vendor-file execution;
  accepted as a Python dependency check, not a build tool.
- No `logging` anywhere in `src/`; observability inside a long stage is
  journal events plus the final structured result. Revisit if long real-model
  builds prove hard to supervise.
- `cli.py` carries ~250 lines of torch-pickle container-dispatch logic,
  drifting from the thin-CLI layering. Fold into `project.py`/`agent.py`
  opportunistically.
- The smoke-fixture generator defaults to `--target sm8750` while the harness
  activates sm8850 — a first-run user on SM8850 silently builds for the other
  chip unless they pass `--target`. Cheap fix; bundle with T12 if touched.
- The CLI accepts specs only as file paths (no stdin/inline JSON) — a minor
  friction for spec-composing agents.
- No test exercises `EngineStageRunner` with `pipeline='genai_builder'` — the
  dispatch point keeping Qwen3.5 out of the low-level lane in a normal
  workflow is itself untested (unit gates exist at three other layers).
