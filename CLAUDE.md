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
- Measurement scope: tensor-level SQNR/RMSE/cosine; warmed production wall
  latency plus optrace attribution; static artifact footprint as the only RAM
  metric. An ONNX Runtime float-graph second reference is a **debug-only**
  mode: slice-boundary comparison has landed (T04 tier 1), layer-level
  drilldown has not. Neither is ever a default, and neither alters production
  reports.
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

## CLI-first workflow

```bash
qairt-agent init --root .
qairt-agent doctor --root .
qairt-agent plan --spec spec.json
qairt-agent workflow --spec spec.json
qairt-agent job watch JOB_ID --follow
qairt-agent diagnose --from-job JOB_ID
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
not filename heuristics, is the final routing authority.

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
Embedding, decoder slices, and LM head retain explicit boundaries. The planned
boundaries reproduce `split_llm` exactly: with `split_lm_head` the final decoder
layer is folded into the lm_head split, so `N-1` layers are distributed across
the decoder slices with the remainder front-loaded, and captured slice
boundaries record the folded layer instead of losing it. Native KV must
preserve exact tensor names, graph routing, shape/layout, and CL alignment; it
marks key/value cache tensors only, never mask/position-style tensors that
merely share a substring, and marks output tensors only for a graph whose AR is
a positive multiple of 32.

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
closed. Only `granularity="slice_boundary"` is implemented; layer-level
drilldown needs executed diagnostic contexts and is not available yet.

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

Benchmark warmed production contexts only; context loading, ADB staging, and
setup are outside the latency sample. For a quality regression, generate a
diagnostic context and bisect component, slice, layer, tensor, then operator.
For a latency regression, use production wall time first and per-op profiling
for attribution. Diagnostic-context latency is not production latency.
For a multi-AR GenAI container, raw-tensor SQNR still covers each exact AR.
Production generation latency is one public-executor prefill/decode workload,
so its report states that internal graph-AR selection is executor-managed.
Multi-AR GenAI optrace fails closed; use an explicit `ar` for a raw
CompiledModel profiling run instead of presenting one AR as complete coverage.

Benchmark sampling is lane-aware. The low-level lane keeps 10 warmup and 50
measured graph invocations. A GenAI sample is a whole `generate()` call, so
that lane resolves 3 warmup and 10 measured at spec-parse time and records the
result in the `BuildSpec`; `qairt-agent plan` renders the effective numbers
under `effective_config.benchmark`, and any value the spec sets explicitly
wins. A/A calibration doubles whichever numbers apply. Every latency report
carries a `measurement_scope` block stating that samples are warmed host
wall-clock around one call including the host-to-SDK-to-device round trip —
the QAIRT Python API exposes no device-side synchronization barrier — and that
per-op attribution comes from optrace. `p50_ms_per_token` is published only
with an explicit `ms_per_token_source`: `caller` for a supplied `token_count`,
`sdk_metrics` only if the SDK reports a generated-token count. QAIRT 2.49 does
not (its `GenerationMetrics` carries a rate and a duration but no count), and a
count is never derived from their product.

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
```

Preserve unrelated workspace changes. Use `apply_patch` for hand edits. Avoid
committing generated contexts, SDK contents under `qnn/`, device dumps, caches,
or secrets. When behavior changes, update the typed contract, CLI plan output,
canonical example, tests, and documentation together — including the
`docs/plan/` status board when the change completes a task. `AGENTS.md` is a
symlink to this file; edit only `CLAUDE.md`.
