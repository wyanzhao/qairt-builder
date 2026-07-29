# QAIRT Agent Maintainer Guide

This is the shared operating contract for Claude Code, Codex, and human
maintainers. Prefer the `qairt-agent` CLI for normal work. Import
`QairtAgent` only for focused debugging or custom analysis; MCP is a
compatibility surface, not the primary automation interface.

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
policy, and output layout before starting a build. `examples/README.md` states
which examples are normal CLI workflows and which capability-gated or legacy
files must not be treated as production templates.

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
Weight sharing packages the exact AR set `{1, 128}` per semantic slice.
For a wider source such as AR2073/CL4096, the build exports independent
AR1/AR128 ONNX and AIMET-encoding artifacts and target-ABI vector manifests;
supplied compatible goldens win, otherwise ORT capture is recorded.
Embedding, decoder slices, and LM head retain explicit boundaries. Native KV
must preserve exact tensor names, graph routing, shape/layout, and CL
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

Device work requires `QAIRT_AGENT_ADB_SERIAL` and
`QAIRT_AGENT_ADB_SERVER`. Remote artifacts live only under the exact leased
`/data/local/tmp/qairt-agent/<job>/<stage>/<attempt>/` path and must be cleaned
after collection. Never broaden cleanup to a parent directory.

## Version and dependency updates

`harness/constraints.json` is the reviewed source of truth for the QAIRT
version/build, Ubuntu image, Python ABI, worker image, runtime CLI versions,
dependency lock path, Torch version, and target tuple. A project may select a
different reviewed constraints file with
`QAIRT_AGENT_HARNESS_CONSTRAINTS`; do not override individual pins ad hoc.
`qairt-agent init` materializes the selected Dockerfile/lock and a deterministic
source archive from the exact running editable checkout or installed wheel.
Do not hand-edit `docker/.generated/`; rerun `init` or `image build`. Keep the
managed `.dockerignore` block last so `qnn/`, models, artifacts, and model
payloads cannot be re-included by earlier user rules.

For an upgrade:

1. Change `harness/constraints.json` in one reviewable patch.
2. Add/rename the pinned dependency file referenced by
   `worker.dependencies_file`; do not reuse an old release lock silently.
3. Update Docker/Apple-container image inputs only through the harness values.
4. Update SDK signature probes and family capability tests for the new build.
5. Run the complete test suite, compile check, project doctor, worker SDK
   import smoke, and at least one real-device golden/latency acceptance run.
6. Update examples and capability claims only after those gates pass.

Do not relax a version mismatch into a warning. Until a new SDK is proven,
preflight must fail closed.

## Development checks

Use small targeted tests while editing, then run:

```bash
.venv/bin/pytest -q
.venv/bin/python -m compileall -q src tests
```

Preserve unrelated workspace changes. Use `apply_patch` for hand edits. Avoid
committing generated contexts, SDK contents under `qnn/`, device dumps, caches,
or secrets. When behavior changes, update the typed contract, CLI plan output,
canonical example, tests, and documentation together.
