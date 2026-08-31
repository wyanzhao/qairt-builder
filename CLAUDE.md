# QAIRT Agent Maintainer Guide

This is the shared operating contract for Claude Code, Codex, and human
maintainers. Prefer the `qairt-agent` CLI for normal work. Import
`QairtAgent` only for focused debugging or custom analysis; MCP is a
compatibility surface, not the primary automation interface.

## Program scope and plan

Long-horizon work is organized as agent-executable tasks under
[`docs/plan/`](docs/plan/README.md): pick one task, execute it against its
acceptance criteria, update its status. This file describes **current**
behavior only; planned changes live in the task files until they land, then
their behavior descriptions move here in the same change.

Settled program decisions (2026-08-29; details and rationale in
`docs/plan/README.md`):

- **Primary model: Qwen3.5** (hybrid/linear attention) through the GenAI
  Builder lane, built from two independent exports — AR1 and AR128 ONNX +
  AIMET encodings — at context length 4096, split policy per GenAI Builder
  defaults. **Secondary: Qwen3 dense/MoE** through the low-level lane from one
  wide export (for example AR2073/CL4096) with AR/CL conversion.
- Golden vectors come from the AIMET-quantized model as trusted local pickle
  and are imported into immutable per-AR manifests before use.
- Measurement scope: tensor-level SQNR/RMSE/cosine; device-side execute time
  and per-op cycles from QAIRT's profiling log, beside a host-orchestrated wall
  latency that is not device time; static artifact footprint as the only RAM
  metric. An ONNX Runtime float-graph second reference is a **debug-only**
  mode: `granularity` accepts `slice_boundary` (T04 tier 1) and `layer` (T04
  tier 2), the latter only over executed, hash-verified diagnostic contexts.
  Neither is ever a default, and neither alters production reports.
- Hardware: **SM8850 and SM8750, through the reviewed target registry** under
  `harness/targets/`. Both are accepted on real hardware. `soc_model` is the
  `Qnn_SocModel_t` value the compiler consumes — SM8850 is 87, SM8750 is 69 —
  and is a different scheme from the Android `soc_id` a device reports (660,
  and 618/639); conflating the two is what made an earlier pin wrong.
- **Out of scope by decision** — do not build or claim: direct Genie API
  integration, power/thermal measurement, token-level accuracy metrics,
  end-to-end Omni audio runtime, end-to-end Qwen3-VL multimodal execution.
- All landed documentation and code are English-only.

## Non-negotiable boundaries

- Use only QAIRT Python APIs: the GenAI Builder Python API and the low-level
  Python API.
- Do not construct or invoke QAIRT/QNN CLI commands, vendor executables, or
  the QAIRT C++ API.
- Keep build intent in a JSON `BuildSpec`/`WorkflowSpec`; do not hide model
  policy in shell scripts.
- Keep production contexts free of diagnostic outputs. Build a separate
  diagnostic context for intermediate tensors or selected output tensors.
- Treat ONNX, external ONNX data, AIMET encodings, vector manifests, raw
  tensors, context binaries, reports, and run manifests as content-addressed
  evidence. Never reuse an artifact after its hash fails verification.
- Do not claim device latency, SQNR, transform equivalence, or runtime support
  without reopenable reports from the corresponding run.

## Onboarding and skills

[`docs/spec-reference.md`](docs/spec-reference.md) is the single reference for
every `WorkflowSpec` field, every `stage_configs` key, and the AR/CL/native-KV
decisions a wide export needs; a test keeps it in step with the contracts in
both directions.

A newcomer starts at [`docs/first-run.md`](docs/first-run.md): clone to a real
report on a real device, using a fixture this repository generates. Because no
model payload is committed, every spec in `examples/` and `configs/` points at a
model the user must supply, so `tools/make_smoke_fixture.py` is the only
runnable entry point. It emits a tiny ONNX, AIMET-style encodings computed from
real tensor ranges, vectors whose golden is the float graph's own output, and
two specs (one plain, one wired for the layer-level float reference). It is
deterministic from a fixed seed, which is what lets an acceptance result
recorded under `docs/plan/` be reproduced on another machine. Artifacts must not
be written under the models directory: the worker mounts it read-only.
`tests/test_smoke_fixture.py` keeps that entry point from rotting.

Recurring procedures are packaged as skills under `.claude/skills/`:
`qairt-first-run`, `qairt-diagnose-quality`, `qairt-diagnose-latency`,
`qairt-add-target`, `qairt-sdk-upgrade`, `qairt-import-vectors`,
`qairt-author-spec`. They restate the operational sequence
and the traps; this file remains the authority on the contract itself, so when
behavior changes the affected skill moves with it.

## CLI-first workflow

```bash
qairt-agent init --root .
qairt-agent doctor --root .
qairt-agent plan --spec spec.json
qairt-agent workflow --spec spec.json
qairt-agent job watch JOB_ID --follow
qairt-agent diagnose --from-job JOB_ID [--baseline BASELINE_JOB] [--kind quality|latency]
qairt-agent compare --from-job A --to-job B
qairt-agent rerun --from-job JOB_ID --spec adjusted.json
```

Commands write JSON/JSONL to stdout. `workflow` runs build, validate, and
benchmark; diagnosis is conditional and starts from that job only after a
reported regression. Long work must use the detached worker and job journal;
reserve `--inline` for tests or a short run inside a compatible Ubuntu worker.
On macOS the Ubuntu 22.04 worker uses Apple `container`; on Linux it uses
Docker. Do not silently fall back to native macOS execution.

Every `rerun` that reuses a production manifest mints a new `run_id`. Before
the first changed stage, the worker publishes a content-verified revision-zero
snapshot of the last reused manifest; an all-reused rerun snapshots at the end.
The source chain is never branched or modified. The snapshot copies cumulative
stages/artifacts and records `forked_from_manifest`, run/revision/job
provenance in metadata. Its stable stage-key identity is the verified source
manifest SHA so crash recovery reuses the same fork and key.
`name` and `output_root` are build identity fields: changing either invalidates
build reuse. In particular, relocating the output root must rebuild there; it
must not copy a prior `BuildSpec` or prior artifact paths into the new root.
When only continuation policy changes, the fork snapshot carries the current
validated effective `BuildSpec` so benchmark and per-stage configuration do not
remain stale. Rebasing is allowed only when build-relevant identities match and
a verified ancestor build receipt has the current build key; family, source,
output-root, transform, quantization, and other build changes fail closed.
Copied stages/artifacts and source/effective BuildSpec hashes preserve the
provenance of reused build outputs.

Golden vectors supplied as a trusted local pickle must first be converted into
the manifest-plus-raw representation:

```bash
qairt-agent vectors import-pickle golden.pkl \
  --output-dir artifacts/imported-vectors \
  --trusted-local --format auto --section auto --isolate
```

Pickle import is explicit because pickle can execute code. Production
validation consumes the resulting immutable vector manifest, not the pickle.
`auto` recognizes modern `torch.save` zip archives; Torch archives are always
loaded in an rlimit subprocess with `torch.load(weights_only=True,
map_location="cpu")` and normalized to NumPy before the parent accepts them.
Direct `pickle.dump(torch.Tensor)` is not supported. For separate input-only or
golden-only files, use `--section inputs` or `--section goldens`; the whole
pickle is assigned to that manifest section.

With the normal `apple_container` or `docker` backend, the host CLI keeps NumPy
imports local but dispatches a Torch archive to the configured pinned Ubuntu
worker, where Torch is installed. Docker mounts the exact archive read-only.
Apple `container` cannot bind a regular file, so the CLI creates a private
temporary directory containing one content-verified `archive.pt`, mounts that
directory read-only, verifies the original, staged copy, and returned manifest
against the original path/SHA, then removes the staging directory. The exact
output directory is mounted read-write, and the container runs as the host
uid/gid.
Docker uses `--network none`; Apple `container` 1.0 only provides `--no-dns`,
which is not hard IP-egress isolation. This requires an initialized project and
a built worker image.

## Model routing

| Preset/family | Required lane | Source policy |
| --- | --- | --- |
| `qwen3_5` | GenAI Builder | Independent AR1 and AR128 ONNX + AIMET encodings |
| `qwen3_5_omni_thinker` | GenAI Builder text lane | Independent AR1 and AR128 Thinker ONNX + encodings; no audio source |
| `qwen3_5_omni` | GenAI Builder workflow packaging | Thinker AR1/AR128 plus audio ONNX + encodings; end-to-end audio runtime remains capability-gated |
| `qwen3_dense`, `qwen3_moe` | Low-level Python API | A source graph such as AR2073/CL4096 may be converted to AR1 and AR128 |
| `qwen3_vl` | Low-level Python API | Text ONNX plus vision ONNX with its projector already integrated |
| `vit` | Standalone low-level Python API | One ONNX, AR1 only, no MHA2SHA/native-KV/weight sharing |

Never route Qwen3.5 or Omni Thinker through the low-level production lane.
Never route Qwen3/Qwen3-VL/standalone ViT through GenAI Builder. The preset,
not filename heuristics, is the final routing authority — but it is
cross-checked, never merely trusted. When a model config is supplied, its
`architectures` are compared against the family table at spec/plan time: an
architecture the table maps to a *different* family fails closed naming the
preset, the config value, and the file, while an architecture the table does
not know is recorded under `effective_config.family_cross_check` as a warning
so an incomplete table cannot block a new family. A nested decoder `model_type`
is the weaker signal (Qwen3-VL legitimately nests `model_type: "qwen3"`) and is
never grounds for failure on its own.

Both lanes verify the target they actually resolved. The low-level lane refuses
a compile whose resolved device target is not the named one; the GenAI lane
reads the same `CompileConfig` back off the builder after `set_targets` and
applies the identical guard, including the empty-`device_custom_configs` case.
Its receipt carries `target.verification` with status `resolved_verified` or —
if a builder exposes no readable resolved value — `input_only`. An input echo
is never labelled as a resolved value.

GenAI raw slices are bound to ARs by proof, not by list order. The tensor names
of an AR1 and an AR128 graph are identical by construction, so a name-only ABI
check cannot see an inverted order; the shapes can, and a dimension that takes
each requested AR exactly once across the graphs is what binds them. No such
dimension means the binding is unprovable, and that fails closed naming the
graphs.

Qwen3-VL validation and benchmarking are component-scoped until an audited
vision-to-text bridge exists. Never describe the default as multimodal
end-to-end: an unscoped run must fail. Use an explicit `component = "text"` or
`component = "vision"` stage config, preserve the resulting `text_only` or
`vision_only` label, and require separate vision vectors for the vision graph.

Canonical examples live in `examples/`. Run `qairt-agent plan` after changing a
spec; the resolved JSON must show the expected `pipeline`, AR policy, native-KV
policy, target, and output layout before starting a build. `examples/README.md`
states which examples are normal CLI workflows and which capability-gated or
legacy files must not be treated as production templates.

Deployable cells live under `configs/{preset}/{target}.json` and are what a real
run is launched from; `configs/README.md` carries the conventions. A test
resolves every cell and requires its directory/file names to agree with the
preset and target inside, and requires that target to be verified.

## Input and transformation contract

The canonical inputs are:

- `sources.*.onnx_path`, including all referenced external-data files;
- AIMET `encodings_path` for `apply_encodings`;
- Hugging Face-style `config_path`;
- a validation vector manifest containing raw inputs and preferred golden
  outputs (Qwen3.5/Omni Thinker use one manifest per AR through
  `validation_manifests_by_ar`); and
- a calibration vector manifest when the standalone quantizer calibrates.

For a low-level LLM build the explicit stage order is model inspection,
test-vector preparation, AR/CL conversion, semantic model split, MHA2SHA,
converter, optional standalone quantizer, and context-binary generation.
The standalone quantizer is deprecated for this program — production input is
always AIMET `apply_encodings` — and survives only as a debugging comparison.
Weight sharing packages the exact AR set `{1, 128}` per semantic slice.
For a wider source such as AR2073/CL4096, the build exports independent
AR1/AR128 ONNX and AIMET-encoding artifacts and target-ABI vector manifests;
supplied compatible goldens win, otherwise ORT capture is recorded.
Embedding, decoder slices, and LM head retain explicit boundaries.

**Per-model knowledge comes from the SDK, not from a copy here.** Whatever the
GenAI Builder already knows about a family is read from it at build time, so an
upper-layer change cannot leave a stale duplicate in this repository:

- MHA2SHA start points are read from the SDK's own family builder
  (`Qwen3_5BuilderHTP._QWEN3_5_START_POINTS` for Qwen3.5) and passed through
  unchanged. The profile stores only where to find them plus a fingerprint of
  what was reviewed; a fingerprint mismatch fails closed naming the new values,
  because these decide where attention heads are cut.
- The native-KV/HMX selection is QAIRT's own `gen_kv_format_config`, including
  its rule that a graph's outputs are marked only when its AR is a positive
  multiple of 32. `GraphContext.export` leaves `ExportedFiles._info` unset and
  the SDK reads the AR from there, so the adapter stamps it exactly as the SDK's
  builder does; unstamped, every graph would silently demote to inputs-only.
  One documented subtraction is applied on top of QAIRT's answer — names whose
  role proves they are not caches (mask, padding, position, index, ...) are
  removed, and what was removed is reported.

The one thing still reproduced locally is the `split_llm` layer distribution
(with `split_lm_head`, `N-1` layers across the decoder slices, remainder
front-loaded, the final layer folded into the lm_head split). It cannot be
sourced: planning must work without the SDK, and the SDK's split graphs carry
no per-split layer range. Captured slice boundaries are therefore marked
`advisory`; what a build verifies is the split count.

Native KV must preserve exact tensor names, graph routing, shape/layout, and CL
alignment.

GenAI Builder owns its internal transform/convert/quantize/compile sequence;
do not invoke the low-level build behind it. Qwen3.5 production specs supply
`metadata.attached_models_by_ar` with `model_path` and `encodings_path` for
both AR1 and AR128.

## Output contract

`output_root` is the only artifact root. Every preset contains a serializable
relative `output_layout`, and `qairt-agent plan` renders it beneath that root.
The current layouts are:

- immutable manifests: `manifests/{run_id}`;
- run state: `runs/{run_id}`;
- effective config: `runs/{run_id}/config`;
- vectors: `runs/{run_id}/vectors`;
- diagnostic reports: `runs/{run_id}/diagnostics`;
- stage attempts: `runs/{run_id}/stages`;
- low-level variants/slices/DLCs/contexts:
  `runs/{run_id}/build/{variants,transformed,converted,contexts}`;
- low-level diagnostic contexts:
  `runs/{run_id}/build/diagnostic_contexts`; and
- GenAI output/cache:
  `runs/{run_id}/genai/{container,cache}`.

Source models are not copied. Their hashes and original paths are recorded by
the immutable manifest under the `source_records` layout role.

## Validation, benchmark, and diagnosis

Prefer supplied goldens for SQNR. If the selected manifest has executable raw
inputs but no golden outputs, validation automatically captures a reference
with ONNX Runtime and records that fallback, model hash, ORT version, and
providers in the immutable report. A manifest with neither usable goldens nor
inputs fails closed; ORT never replaces supplied goldens.

A second ONNX Runtime reference is available as an explicit **debug-only** mode:
`stage_configs.validation.float_reference` runs the float source graph and
compares it against the device slice boundaries. It is off unless that config
is present, requires an explicit single `ar`, requires a device chain run, and
publishes a separate `float_reference_report` artifact plus a `float_reference`
block — the supplied-golden comparison is untouched and remains the production
reference. Internal activations are promoted to outputs in an in-memory copy of
the graph; the model on disk is never rewritten. A device tensor binds to a
float tensor only by an exact name match or an explicit
`float_reference.tensor_map` entry; everything else is listed in
`unmapped_tensors` rather than guessed, and a run that can bind nothing fails
closed. `granularity` accepts `slice_boundary` and `layer`. Layer granularity
compares the tapped intermediates, so it requires an executed, hash-verified
diagnostic context for **every** slice in scope and fails closed naming the
slices that lack one rather than degrading silently. A chain assembled from two
independently built slices can never satisfy that — a diagnostic context belongs
to the build that produced it — so the multi-slice success path still has not
run on hardware; what has run is the single-slice path.

When no custom graph/routes/outputs or explicit `stage_configs.*.ar` override
is present, low-level validation and benchmarking execute every AR requested by
the spec. Each AR is bound to its exact graph and
`runtime_index.vectors.validation_manifests_by_ar` entry. The stage publishes
immutable `sqnr_report_arN`/`latency_report_arN` artifacts plus a canonical
aggregate with `coverage` and `results_by_ar`; optrace follows the same rule.
Missing per-AR evidence fails closed. An explicit `ar` is intentionally a
single-AR debug override, and custom graph/routes remain caller-scoped rather
than being fanned out implicitly.

- `quality.sqnr_modes` is executable workflow policy, not report metadata.
  Validation runs exactly the listed `full_reference`, `teacher_forced`, and
  `chain` modes and records the requested/executed modes independently.
- `full_reference` compares the final device output with the supplied full
  golden, or with an audited ONNX Runtime fallback when no full golden exists.
- `teacher_forced` feeds every slice inputs from its own golden boundary. Those
  per-slice inputs and outputs must come from
  `stage_configs.validation.slice_vector_manifests` or from an exact ONNX
  reference run over the transformed slice models. Device boundary outputs are
  never accepted as teacher inputs; missing slice models/tensors fail closed.
- `chain` feeds device output from one slice into the next and compares every
  slice boundary with the same per-slice reference set, so local and propagated
  errors remain distinguishable.

When `dump_intermediates_on_failure=true`, the effective build enables separate
diagnostic contexts before validation; `qairt-agent plan` exposes this under
`effective_compile`. A failing validation may claim operator-intermediate
evidence only when those contexts are present and hash-verified. Otherwise the
report explicitly degrades to verified slice/tensor evidence and sets
`op_level_dump_available=false`.

Every build publishes a report-only `static_footprint` block — per-artifact
bytes read from the published content-addressed references, per-role totals, and
a `total_bytes` that sums only the roles named in `total_includes` (context
binaries and the saved GenAI container). Converted DLCs are reported but never
summed into it, diagnostic contexts sit in a separate `diagnostic` section with
`counted_in_totals=false`, and a role with no outputs has no total field rather
than a zero. Benchmark reports embed the block copied verbatim from the verified
build receipt rather than re-measuring. This is the only RAM metric.

**Latency means device time, and production latency is
`accelerator_compute_us`** — QAIRT's "Accelerator (execute excluding wait)
time", the cost of the model on the hardware with host orchestration and device
queueing/memory wait both outside it. It is published as
`production_latency_us` with `production_latency_source` and
`production_latency_cv_percent`; its small absolute value makes it the most
dispersed metric in the block (8-17% CV against ~2% for accelerator execute), so
a change is read against that dispersion. Deployment latency for Qwen3.5 is a
different meter — Genie's device-measured token rate and time-to-first-token —
and the two are never combined. A latency report names its metric in `latency_metric`, and
that names the `device_execution` block or the string `unavailable` — never a
host number. `device_execution` comes from QAIRT's own
profiling log: accelerator compute and execute time, QNN execute time, per-op
cycles, and per-process overhead kept separate. It is read at `level="detailed"`
with **no** profiling option — `option="optrace"` additionally requires a
schematic binary that this program's compile does not emit, and fails with "No
op trace raw data found." without one; per-op cycles need no optrace. The
profiled execute is repeated **ten times and averaged**: `statistic` says
`mean`, and `spread` plus `samples` keep the per-sample values so an average is
never taken on faith. The block names its `meter` (`qnn_accelerator`) and
`lane`, because the GenAI lane's meter is a different one and the two must
never be conflated.

A scope with no device meter sets `latency_metric: "unavailable"` and publishes
`device_execution.available = false` **with a reason**. Today that is the GenAI
generation scope: `generate()` reaches Genie as `GenieDialog_query`, so
`qairt.Profiler` observes nothing. An absent device number is always a declared
gap, never a silent omission.

Chain scope **is** measured. The timed pass records the exact inputs each slice
was actually fed — in a chain they come from the previous slice at run time —
and every slice is then profiled with those recorded inputs, so `by_slice`
carries one `device_execution` block per slice. `totals` is a sum of per-slice
means, labelled as such because the slices run sequentially; it is not a
measured end-to-end number. A *sequence* runs each slice once per step, and
every recorded step is profiled: the block then reports `scope:
"chain_sequence"` with `by_step` (per-step `by_slice`, graph, and AR) and
`steps_covered`/`steps_total`, never last-step-only evidence under an
unqualified `chain` label.

An aggregate never claims more than its parts. The multi-AR latency report sets
`latency_metric` to `device_execution` only when **every** AR carries an
available block; otherwise it is `partial`, and `coverage` names
`metered_ars`, `unmetered_ars`, and the per-AR reason. A `device_execution`
block likewise states `samples_requested`/`samples_used` and sets `partial:
true` when fewer profiled executes were aggregated than the contract's ten, or
when a metric was absent from some of them.

The host wall-clock number is kept only under `harness_diagnostics`, marked
`not_latency: true`, because it still detects ADB, container and transport
degradation. It never grounds a regression verdict, and `aa_calibration` and
`p50_ms_per_token` live there too since both derive from it. Its
`measurement_scope` lists `excluded_from_timer` (context loading, ADB staging,
device construction, graph-runner setup — what *we* control) against
`included_in_sample` (qnn-net-run process launch, per-call context load,
HVX/HMX power-on, per-call deinit, the ADB round trip — what the SDK does
inside one call). Benchmark warmed production contexts only. The gap between
the two metrics is large and expected: on SM8750 the tiny acceptance graph
measured ~4900 ms of wall time around 79 µs of accelerator compute.

`initialize_execution` establishes QAIRT's persistent execution context before
the timer and `release_execution` frees it afterwards; without it the SDK
rebuilds the backend and inferencer on every call. Device capture must run
**before** initialization — an initialized model carries an execution context
created with profiling disabled, so profiling it silently yields nothing — and
the adapter fails closed rather than letting that happen.

Diagnosis selects its path from the request or from measurement, never from a
heuristic that is always true. `stage_configs.diagnose.kind` (or `diagnose
--kind`) runs exactly one path and fails closed when that path has no evidence;
without a kind, **both** paths run and the report's `considered` block says what
each found. The old selector keyed on "some SQNR observation has nonzero noise",
which is the steady state of every healthy quantized run, so the latency path
was unreachable after any validate stage. Nonzero noise is still what the
quality path *attributes*; it is no longer what selects it.

`qairt-agent compare --from-job A --to-job B` (or `--from-manifest` /
`--to-manifest`) is the cross-run delta. It loads both runs' hash-verified
reports and refuses a non-comparable pair fail-closed, naming the field that
differs — preset, family, target, AR set, context lengths, `sqnr_modes`, or the
latency meter/lane. It emits per-AR `production_latency_us` deltas in absolute
terms **and in units of the pooled CV**, because the program contract says a
latency change is read against `production_latency_cv_percent`; per-tap
SQNR/RMSE/cosine deltas, worst movers first; and provenance for both sides
(run ids, manifest SHAs, report SHAs, identity fields). It is report-only: no
threshold and no pass/fail verdict anywhere. `diagnose --baseline` runs the same
comparison first and publishes an `implicated` block saying which path the
measured change points at, with the rule stated inline — both paths are still
reported.

For a quality regression, generate a diagnostic context and bisect component,
slice, layer, tensor, then operator. For a latency regression, compare the
`device_execution` block; wall time moves with host and transport conditions
that have nothing to do with the model, so it diagnoses the harness, not the
model.
Diagnostic-context latency is not production latency.
For a multi-AR GenAI container, raw-tensor SQNR still covers each exact AR.
Production generation latency is one public-executor prefill/decode workload,
so its report states that internal graph-AR selection is executor-managed.
Multi-AR GenAI optrace fails closed; use an explicit `ar` for a raw
CompiledModel profiling run instead of presenting one AR as complete coverage.

Benchmark sampling is lane-aware. The low-level lane keeps 10 warmup and 50
measured graph invocations. A GenAI sample is a whole `generate()` call, so
that lane resolves 3 warmup and 10 measured at spec-parse time and records the
result in the `BuildSpec`; `qairt-agent plan` renders the effective numbers
under `effective_benchmark`, and any value the spec sets explicitly wins. The
plan key literally named `effective_config` is a different thing — an
output-layout role naming where the effective config is written — so reading
the benchmark policy out of it finds the wrong object. A/A calibration doubles
whichever numbers apply. The `measurement_scope` block describing that sampling
lives inside `harness_diagnostics`, with the wall-clock number it qualifies: it
states that samples are warmed host wall-clock around one call including the
host-to-SDK-to-device round trip — the QAIRT Python API exposes no device-side
synchronization barrier — and lists explicitly what is `excluded_from_timer`
and what is `included_in_sample`. `p50_ms_per_token` is published only
with an explicit `ms_per_token_source`: `caller` for a supplied `token_count`,
`sdk_metrics` only if the SDK reports a generated-token count. QAIRT 2.49 does
not (its `GenerationMetrics` carries a rate and a duration but no count), and a
count is never derived from their product.

Before any device stage, the handset behind `QAIRT_AGENT_ADB_SERIAL` is checked
against the resolved target's registered Android `soc_id` list — a plain adb
read of `ro.soc.id`, `ro.soc.model`, then `/sys/devices/soc0/soc_id`, with no
SDK involvement. A reported id outside the list fails closed before the lease
and before the device is constructed, naming serial, observed id, target, and
the registry list; an unreadable id is a recorded warning with the raw source
outputs kept, because absence is a gap in what we can see rather than evidence
of the wrong chip. `QAIRT_AGENT_TARGET_ACCEPTANCE=<name>` downgrades a
contradiction to a warning for the qualifying run, since confirming that
target's `soc_id` list is part of what the run exists to do. The observed value
is recorded in the stage's device section, and `device doctor` reports the same
comparison instead of the tautological check it used to publish.

Device work requires `QAIRT_AGENT_ADB_SERIAL` and
`QAIRT_AGENT_ADB_SERVER`. Remote artifacts live only under the exact leased
`/data/local/tmp/qairt-agent/<job>/<stage>/<attempt>/` path and must be cleaned
after collection. Never broaden cleanup to a parent directory.

## Version and dependency updates

`harness/constraints.json` is the reviewed source of truth for the QAIRT
version/build, Ubuntu image, Python ABI, worker image, runtime CLI versions,
dependency lock path, Torch version, and the *name* of the active target; the
target's own values live in `harness/targets/<name>.json`. A project may select a
different reviewed constraints file with
`QAIRT_AGENT_HARNESS_CONSTRAINTS`; do not override individual pins ad hoc.
`qairt-agent init` materializes the selected Dockerfile/lock and a deterministic
source archive from the exact running editable checkout or installed wheel.
Do not hand-edit `docker/.generated/`; rerun `init` or `image build`. Keep the
managed `.dockerignore` block last so `qnn/`, models, artifacts, and model
payloads cannot be re-included by earlier user rules.

The SDK installation root `qnn/qnn` may be a symlink to a real install
elsewhere; discovery and container mounts must resolve it (T01 verifies this).

For an upgrade:

1. Change `harness/constraints.json` in one reviewable patch.
2. Add/rename the pinned dependency file referenced by
   `worker.dependencies_file`; do not reuse an old release lock silently.
3. Update Docker/Apple-container image inputs only through the harness values.
4. Update SDK signature probes and family capability tests for the new build.
   `tools/sdk_signature_probe.py` is that probe; run it **inside the worker
   container**, where the SDK imports, and it fails naming any bound surface
   the new build dropped.
5. Run the complete test suite, compile check, project doctor, worker SDK
   import smoke, and at least one real-device golden/latency acceptance run.
6. Update examples and capability claims only after those gates pass.

Do not relax a version mismatch into a warning. Until a new SDK is proven,
preflight must fail closed. The pin is QAIRT 2.49.0.260730 (build
`260730134355`), landed by [T01](docs/plan/T01-sdk-upgrade-2.49.md).

Targets live in a reviewed registry, one `harness/targets/<name>.json` per
target, and `harness/constraints.json` only names which one is active. Each
entry carries `chipset`, `dsp_arch`, `soc_model` (the `Qnn_SocModel_t` value),
the Android `soc_id` list, and a `verified` block recording the real-device
acceptance run that qualified it.

A spec selects a target by `name`, or supplies the complete
`chipset`/`dsp_arch`/`soc_model` tuple, which is accepted only if it matches a
registered entry exactly. A partial tuple is never completed implicitly, an
unregistered name or tuple fails at spec time, and there is no built-in default
— the harness names one. `qairt-agent plan` renders the resolved target under
`effective_target`, including whether it is verified.

A target with no `verified` block still plans, but build and device stages
refuse it: it has never been proven on hardware. Because a target cannot become
verified without a run and a run is refused while it is unverified, the
qualifying run is the one explicit exception — set
`QAIRT_AGENT_TARGET_ACCEPTANCE=<name>` for that run only, and record its
outcome in the registry entry afterwards.

One guard needs care on SM8750: QAIRT's own compile default is `v79`/`soc_model
69`, which is exactly the SM8750 tuple, so a resolved-value check cannot
distinguish an intended target from a silent fallback. An empty
`device_custom_configs` list — the SDK's "skipping device config creation"
path — therefore fails closed in its own right, whichever target was named.

## Development checks

Use small targeted tests while editing, then run:

```bash
.venv/bin/pytest -q
.venv/bin/python -m compileall -q src tests
.venv/bin/python -m mypy
```

The type gate is scoped, not a whole-repo strictness jump: the pipeline↔adapter
`QairtAdapterProtocol` boundary, `contracts.py`, `contracts_reports.py`,
`family_registry.py`, `families/`, and `diagnostics/`. It is meant to fail on
real drift, which means it has to stay green — do not widen the scope and leave
findings behind. `mypy` is in the `dev` extra.

The pipeline consumes its adapter through `QairtAdapterProtocol`
(`qairt_adapter/types.py`): the methods it always calls, plus
`QairtAdapterOptionalProtocol` for the ones it probes for and degrades without.
The real adapter and the test fake are checked against the same declaration, so
a method renamed in one and not the other fails a test rather than a call site.
Published report payloads for the multi-AR aggregates and the
`device_execution` block are pydantic models in `contracts_reports.py`; they
allow and preserve extra keys, because a published report is content-addressed
evidence and a lossy round trip would make its recorded hash unreproducible.

Preserve unrelated workspace changes. Use `apply_patch` for hand edits. Avoid
committing generated contexts, SDK contents under `qnn/`, device dumps, caches,
or secrets. When behavior changes, update the typed contract, CLI plan output,
canonical example, tests, and documentation together — including the
`docs/plan/` status board when the change completes a task. `AGENTS.md` is a
symlink to this file; edit only `CLAUDE.md`.
