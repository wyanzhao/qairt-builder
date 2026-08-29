# Native workflow

`qairt-agent` keeps the content-addressed artifact store, immutable manifest
revisions, ONNX inspection, and SQNR/latency diagnostics, and layers a
persistent job journal, family presets, a thin CLI, and four asynchronous MCP
tools on top of the synchronous stage engine.

The default workflow is:

```text
spec.json
  -> background build
  -> validate
  -> benchmark
       |
       +-- regression --> diagnose
                            -> Claude Code/Codex adjusts the spec
                            -> rerun, reusing unchanged stages
```

`diagnose` is never part of the default `workflow`; run it from the completed
job only when validation or benchmark evidence shows a regression.

## Public surface

### CLI

```text
qairt-agent init                         # write qairt-agent.toml + state dirs
qairt-agent image build --root .         # build pinned Ubuntu/Python worker
qairt-agent doctor                       # verify SDK metadata, ABI, target
qairt-agent plan --spec spec.json        # resolve preset/workflow without the SDK
qairt-agent build|validate|benchmark|diagnose --spec ... [--from-job ...]
qairt-agent workflow --spec spec.json    # build -> validate -> benchmark
qairt-agent diagnose --from-job JOB_ID   # conditional regression attribution
qairt-agent rerun --from-job JOB_ID [--spec adjusted.json]
qairt-agent job status|watch|cancel|resume|list JOB_ID
qairt-agent vectors import-pickle PATH --output-dir DIR --trusted-local \
  --format auto --section auto --isolate
qairt-agent device doctor|gc
qairt-agent artifact verify PATH --sha256 ...
```

Long-running commands submit a persistent job and, by default, spawn a
**detached worker process** (`start_new_session`) so the job survives the CLI
exiting or being killed. They print one JSON line `{job_id, state,
status_path}` and return. `--follow` streams JSONL events until the job is
terminal; `--inline` runs the worker in the current process. `job watch
--after-seq N` resumes a stream from event sequence `N` using only the journal.
Before returning, the launcher waits up to
`QAIRT_AGENT_WORKER_STARTUP_TIMEOUT` seconds (default 30) for the worker to
leave `QUEUED`. Immediate runtime exits and startup timeouts are recorded in
`logs/worker-launch.log` and transition the journal to structured `FAILED`.

A rerun never continues an already-published parent revision. After its final
reused stage, it snapshots that exact cumulative manifest into revision zero of
a new run and continues there; if every stage is reused, that snapshot is the
rerun's final manifest. `metadata.forked_from_manifest` preserves the verified
source reference without creating a parent edge or allowing one `run_id` to
branch. The `manifest_forked` journal event makes the snapshot recoverable
after a hard worker loss.

`name` and `output_root` participate in the build stage key. Relocating
`output_root` therefore starts a real build under the new root instead of
reusing or copying a manifest whose `BuildSpec` and artifacts still point at
the old root.

For a continuation-only adjustment, the new snapshot stores the current
validated effective `BuildSpec`, not the parent's stale workflow policy.
`forked_from_build_spec_sha256`, `effective_build_spec_sha256`, the copied
stage/artifact records, and `forked_from_manifest` keep both effective policy
and reused-output provenance explicit. The worker permits this rebase only
after matching the build-relevant identity and locating a verified ancestor
build receipt with the current build key. A family, source, output-root,
transform, quantization, build-stage-config, or other build-relevant change
must execute build again and cannot enter through the snapshot path.

### Python API

```python
from qairt_agent import QairtAgentClient

client = QairtAgentClient.from_project(".")
handle = client.submit(spec)                 # background; returns immediately
handle = client.workflow(spec)               # build + validate + benchmark
handle = client.rerun(from_job, adjusted)    # reuse unchanged stages
handle = client.resume(job_id)               # continue from last verified receipt

handle.status()        # JobStatus
handle.events(after_seq=0)
handle.cancel()
handle.wait()
```

The synchronous `qairt_agent.pipeline.QairtAgent` remains the stage engine the
worker delegates to.

### MCP

Default (four short asynchronous tools):

```text
submit_job(spec, stages?, from_job?)  -> {job_id, state, status_path}
get_job(job_id, after_seq?)           -> {status, events}
cancel_job(job_id)                    -> {ok, job_id}
resume_job(job_id)                    -> {job_id, state, status_path}
```

The original ~18 synchronous tools remain behind `qairt-agent-mcp --legacy`
(or `QAIRT_AGENT_MCP_LEGACY=1`) and are marked deprecated. The agent no longer
orchestrates a dozen fine-grained tools.

## Spec and presets

`WorkflowSpec` replaces the flat `family` enum with a `preset` reference
plus an optional `SkuOverlay`. `BuildSpec` is still readable and is
converted with `to_workflow_spec`; all new artifacts use these schemas.

| Preset | Pipeline | Default policy |
|---|---|---|
| `qwen3_dense` / `qwen3_moe` | low-level Python API | AR1+AR128, 4 decoder slices, independent embedding/lm_head, weight sharing, native KV |
| `qwen3_vl` | low-level Python API | reusable ViT/projector components + text decoder chain |
| `vit` | low-level Python API | single ONNX → DLC → context; no AR/split/MHA2SHA/KV/weight sharing |
| `qwen3_5` | GenAI Builder | independent ONNX+encodings per AR; must dispatch `Qwen3_5BuilderHTP` |
| `qwen3_5_omni_thinker` | GenAI Builder text lane | independent Thinker ONNX+encodings and validation manifest per AR; no audio source |
| `qwen3_5_omni` | GenAI Builder component packaging | `Qwen3OmniAudioEncoderBuilderHTP` + pinned `Qwen3_5BuilderHTP`, independent text ONNX+encodings per AR; packaging supported, `runtime_supported=false` |

Qwen3-VL has no automatic end-to-end runtime binding in this release.
Validation and benchmarking without an explicit component fail closed.
`stage_configs.validation.component` and
`stage_configs.benchmark.component` may select `text` or `vision`; reports are
labelled with that partial coverage, and a vision-only run supplies its own
vision vectors.

The Omni package is an SDK `WorkflowContainer` created with
`WorkflowBuilder.from_builders` and an
`AUDIO_ENCODER -> TEXT_GENERATOR` graph. Both component containers are saved,
but the agent does not call or claim end-to-end audio execution: QAIRT 2.48's
workflow executor does not orchestrate that audio node. Missing audio
ONNX/encodings, per-AR text ONNX/encodings, explicit builder classes, token IDs,
or factory methods fail closed before packaging.

`preset capture` binds a reference overlay to a model SHA, architecture, tensor
ABI, and exact slice boundaries into a reproducible `SkuOverlay`.

### Per-stage inputs

The continuation stages deliberately do not share one catch-all configuration.
`stage_configs` is part of `WorkflowSpec`/`BuildSpec`, is content-addressed per
stage, and has this shape:

```json
{
  "stage_configs": {
    "build": {},
    "validation": {
      "actual_manifest": "/models/qwen3/vectors/device-chain.json"
    },
    "benchmark": {
      "context_path": "/artifacts/qwen3/context.bin",
      "graph_name": "decoder_00_ar1",
      "vector_manifest": "/models/qwen3/vectors/validation.json",
      "optrace": true
    },
    "diagnose": {
      "kind": "quality",
      "config": {}
    }
  }
}
```

`validation` also accepts the input alias `validate`. A single-graph benchmark
requires `context_path`, `graph_name`, and `vector_manifest`; a chain benchmark
instead supplies `routes` plus contexts and steps/vectors. With an empty
diagnose config (the default for `qairt-agent diagnose --from-job JOB_ID`), the
engine selects evidence from the parent job's cumulative verified manifest:

- a validate-published SQNR report with positive noise selects quality
  diagnosis and localizes the first observable slice/tensor boundary;
  layer/operator attribution is emitted only when that report contains
  explicit lineage, never inferred from names;
- otherwise a benchmark-published `optrace_evidence` plus its bound latency
  report selects latency diagnosis. A compatible parent/fork profile with
  stable matching op IDs enables per-op cycle deltas. Without one, the result
  is explicitly `candidate_hotspot_only` and does not claim a regression;
- absence of both forms of provable evidence fails closed with a structured
  error.

Explicit diagnosis remains available for custom analysis: quality accepts
`reference_trace` plus `actual_trace`, while latency selects `"kind":
"latency"` and accepts `baseline_ops` plus `candidate_ops`.

These are explicit artifact bindings, not guessed links. In particular, the
orchestrator may select only the exact AR/CL route and vector manifest recorded
in the build's content-verified `runtime_index`; it never chooses an arbitrary
context, trace, or device output from a multi-slice manifest. An explicit stage
configuration can replace that binding. Missing or ambiguous bindings fail as
a structured stage error, and changing a bound file changes only that stage and
its downstream cache keys.

For an automatic low-level continuation with multiple requested ARs, the
orchestrator runs every AR independently. It retains content-addressed
`sqnr_report_arN`, `latency_report_arN`, and optional
`optrace_evidence_arN` artifacts, then publishes a canonical aggregate whose
`coverage` lists requested/executed/missing ARs and whose `results_by_ar`
contains the exact report references. An explicit `stage_configs.*.ar` remains
a single-AR debug override. Explicit graphs, routes, steps, or result manifests
are caller-owned scopes and are never automatically multiplied.

Qwen3.5 and Omni Thinker specs provide
`vectors.validation_manifests_by_ar` for AR1 and AR128 alongside matching
`metadata.attached_models_by_ar` ONNX+encoding pairs. Supplied goldens are
preferred. If an AR manifest has raw inputs but no goldens, validation captures
an ONNX Runtime reference and records its provenance; a manifest without either
usable goldens or executable inputs fails closed.

Their build records an auditable raw-tensor route from the saved container's
public compiled-model splits when the SDK exposes one. SQNR fails closed when
that route is unavailable. GenAI latency uses the saved container's public
executor and requires an explicit `stage_configs.benchmark.prompt` or
`prompt_path`. Multi-AR SQNR executes each exact raw-tensor route. Production
generation latency is one executor-managed prefill/decode workload; its
coverage does not claim that the internal graph AR is observable. A multi-AR
GenAI optrace request fails closed. To collect public raw `CompiledModel`
profiling evidence, set an explicit benchmark `ar`; that evidence is labelled
`raw_compiled_slices_not_generation_wall_latency`.

For low-level graph and chain runtimes, `benchmark.optrace` executes the exact
selected graph/slice sequence through QAIRT's public Python profiler after the
warmed wall-latency measurement. It publishes captured raw reports and one
normalized, content-addressed `optrace_evidence` artifact. Thread-cycle records
use the maximum overlapping thread value rather than a sum, and all per-op
claims remain reported work attribution rather than additive wall latency.

### Benchmark sampling and token accounting

Sampling policy is lane-aware. The low-level lane measures 10 warmup and 50
measured graph invocations. One GenAI sample is an entire `generate()` call, so
that lane resolves 3 warmup and 10 measured — materialized into the `BuildSpec`
when the spec is parsed, so `qairt-agent plan` shows the numbers that will run
under `effective_config.benchmark` (with `lane` and `sample_unit`), and every
later stage reads the same values. Explicit spec values always win, per field.
A/A calibration doubles whichever numbers apply.

Every latency report carries a `measurement_scope` block: samples are warmed
host `perf_counter_ns` wall time around one call and include the
host-to-SDK-to-device round trip, because the QAIRT Python API exposes no
device-side synchronization barrier. On-device per-op attribution comes from
optrace, never from arithmetic on these samples.

`p50_ms_per_token` appears only alongside an explicit `ms_per_token_source`.
`caller` means the benchmark config supplied `token_count`. `sdk_metrics` is
reserved for an SDK that reports a generated-token count; QAIRT 2.49 does not —
its public `GenerationMetrics` exposes `token_generation_rate` and
`token_generation_time` but no count, and multiplying them would manufacture a
number the SDK never reported. Sources are never mixed silently.

### Static footprint

Every build publishes a `static_footprint` block in its stage data and metrics:
one entry per published output with its exact byte size and SHA, per-role
totals (`contexts_total_bytes`, `converted_models_total_bytes`,
`genai_container_total_bytes`), and a headline `total_bytes` that sums only the
roles named in `total_includes` — the context binaries and the saved GenAI
container, that is, what the device has to hold. Converted DLCs are reported as
build intermediates but never summed into that total, and diagnostic contexts
live in their own `diagnostic` section with `counted_in_totals: false`, the same
separation the latency reports keep.

Sizes come from the published content-addressed references, so nothing is
estimated: a role with no outputs has no total field rather than a zero. The
policy is `report_only`; there are no thresholds. Benchmark reports embed the
block copied verbatim from the hash-verified build receipt (with `source:
"build_receipt"` and the measuring stage recorded), so a latency report answers
"how big is what I just measured" without re-measuring. This static footprint
is the program's only RAM metric; on-device RSS/PSS and VTCM/DDR accounting are
out of scope.

### Output layout

`output_root` is model-specific and is the only artifact root. `plan` renders
the selected preset's relative roles beneath it:

```text
manifests/{run_id}
runs/{run_id}/config
runs/{run_id}/vectors
runs/{run_id}/diagnostics
runs/{run_id}/stages
runs/{run_id}/build/{variants,transformed,converted,contexts}
runs/{run_id}/build/diagnostic_contexts
runs/{run_id}/genai/{container,cache}
```

Only the `build/...` roles exist for low-level presets; only `genai/...` exists
for GenAI presets. Source models remain at their original paths and are
content-addressed from the manifest rather than copied.

## File job journal

Jobs live under `.qairt-agent/jobs/<job-id>/` (no database):

```text
spec.original.json     immutable original spec
spec.resolved.json     immutable resolved workflow
launcher.json          launcher provenance + intended stages
state.json             atomic JobStatus snapshot
heartbeat.json         last heartbeat timestamp/pid
cancel                 presence requests cancellation
events/0000000001.json append-only, one event per sequence number
receipts/<...>.json    immutable verified StageReceipt
logs/<stage>.log       per-stage logs
logs/worker-launch.log detached-runtime startup stdout/stderr
```

- State machine: `QUEUED -> STAGING -> RUNNING -> COLLECTING -> COMMITTING ->
  SUCCEEDED`, plus `FAILED/CANCELLED/ORPHANED`. Terminal states cannot be
  left (resume continues an `ORPHANED` job; `rerun` retries after `FAILED`).
- `state.seq` always equals the last event sequence number, so a watch can
  resume from the journal alone.
- Each stage produces an immutable `StageReceipt`; the worker verifies every
  input/output SHA before recording a receipt, and a single `ManifestPublisher`
  is the only thing that advances `state.manifest` (parallel slices submit
  receipts; they never race a revision).
- `resume` replays the journal and continues from the last verified receipt;
  incomplete attempts are never reused. A `stage_started` event reserves its
  attempt number, so a hard-killed stage restarts in a new `attempt-NNN`
  directory rather than reading partial output from the interrupted attempt.
- The production heartbeat runs in a helper process, not a Python thread, so a
  long QAIRT pybind call that holds the GIL still updates
  `heartbeat.json`/`state.heartbeat_at` with the QAIRT worker pid. A worker
  holds an exclusive per-job lease; after
  `QAIRT_AGENT_HEARTBEAT_STALE_AFTER` seconds (default 30), a new worker marks
  a stale in-flight state `ORPHANED` before resuming. The heartbeat interval is
  controlled by `QAIRT_AGENT_HEARTBEAT_INTERVAL` (default 5 seconds).
- QAIRT stages remain atomic recovery units. A verified receipt and its
  manifest are reused even if the previous process died before final publish;
  a stage without such a receipt is rerun in full. There are no synthetic
  checkpoints inside AR conversion, quantization, conversion, or context
  generation yet.
- `rerun` creates a new job with `parent_job_id` and reuses any stage whose key
  is unchanged. Stage keys fold in the stage inputs, the resolved preset, the
  SDK build, the adapter capability, the resolved worker-image digest, and the platform
  ABI; device stages also fold in a device/runtime fingerprint. Keys are
  stage-aware: a benchmark-only spec change reuses build and validate.

## Project, worker runtimes, device

- `init` writes `qairt-agent.toml` with installation root `./qnn/qnn`. It
  accepts either a direct SDK or a versioned
  `./qnn/qnn/qairt/<release>/sdk.yaml`, preferring the pinned build without
  moving it. It also writes the editable `harness/constraints.json` version
  contract, copies the matching image inputs from the editable checkout or
  installed wheel, stages the exact agent sources, and applies the mandatory
  build-context exclusions. It never installs a host runtime.
- `worker.backend = "auto"` resolves to Apple `container` on macOS and Docker
  on Linux. Native execution is opt-in and must satisfy the pinned ABI.
  `qairt-agent image build --root .` refreshes and builds the project-local worker image and
  then accepts it only after an import smoke test against the read-only mounted SDK. Run
  `qairt-agent image smoke --root .` to repeat that gate without rebuilding.
- `doctor` verifies SDK metadata, QAIRT 2.48 capability, Ubuntu/x86_64 ABI, and
  the `SM8850/v81/soc_model 660` target. On a host without the SDK or selected
  runtime it reports those checks as failing rather than silently passing.
- Both runtimes use the same Ubuntu 22.04 / Python 3.10 / `linux/amd64`
  image. The SDK, models, and artifacts are mounted, never baked into the
  image. Docker smoke disables the network; Apple `container` 1.0 smoke uses
  `--no-dns` because that CLI has no `--network none` equivalent. Apple
  Silicon explicitly enables Rosetta and records emulation in provenance.
  The resolved image digest and selected ADB device identity participate in
  stage provenance, so rebuilding the same tag or changing devices invalidates
  unsafe reuse. If the selected runtime or configured image is unavailable the runner
  fails closed.
- ADB is only for transfer/lifecycle. `QAIRT_AGENT_ADB_SERIAL` and
  `QAIRT_AGENT_ADB_SERVER` are required (no auto-select; fail closed). Remote
  work lives at `/data/local/tmp/qairt-agent/<job-id>/<stage-key>/<attempt-id>/`,
  staged via `incoming -> verify -> ready`. Context binaries, vector manifests,
  and their raw tensors are copied there as a content-verified lifecycle/audit
  sandbox. QAIRT 2.48 does not expose a Python option that makes
  `CompiledModel` reuse this remote working directory: execution still receives
  an explicit `qairt.Device`, and QAIRT owns its internal runtime deployment.
  The exact audit-sandbox attempt directory is removed in a `finally` on
  success, failure, and cancel. A file lease per server+serial prevents
  concurrent use; its exact attempt path is recorded before the first push and
  forgotten only after cleanup succeeds. Lease identity canonicalizes
  `localhost`, IPv4/IPv6 loopback, Docker's `host.docker.internal`, and
  Apple's `host.container.internal`, while
  the ADB client continues to use the configured connection address. A
  process-sidecar heartbeat is independent of the QAIRT caller's GIL and makes
  container PIDs diagnostic-only. Owner records are complete before atomic
  no-clobber publication. `QAIRT_AGENT_LEASES_DIR` selects the shared lease
  store (`/state/leases` in a worker container). `device gc` requires the explicit ADB
  environment for a real cleanup, waits out a creation grace for legacy
  malformed records, and then rechecks heartbeat freshness plus owner-token
  and content CAS while holding the per-device acquisition lock. It processes
  only leases matching that server+serial and its shared parser accepts only
  `/data/local/tmp/qairt-agent/<job>/<stage>/<attempt>/`, never broad
  recursion.
- Prefer project-relative paths or the stable container roots `/models` and
  `/artifacts`. The journal preserves the supplied values; host-specific
  absolute paths are supported through compatibility mounts but make a job
  non-portable.

## Golden vectors and pickle

- `vectors import-pickle --trusted-local` is the only sanctioned pickle path;
  normal builds never call `pickle.load`. `--format numpy-pickle` uses the
  restricted NumPy-only global allowlist, rejects `persistent_load`, caps input
  size, and validates the tree (nested containers plus NumPy ndarray/scalar;
  no custom classes, callables, or object dtype). `--format torch` accepts a
  `torch.save` archive only through a mandatory rlimit subprocess and calls
  `torch.load(weights_only=True, map_location="cpu")`; tensors are immediately
  normalized to a NumPy-only tree before the parent revalidates it. Direct
  `pickle.dump(torch.Tensor)` is intentionally unsupported. `--format auto`
  recognizes modern torch zip archives and otherwise retains the restricted
  NumPy path. NumPy imports execute in the host CLI. For the normal
  `apple_container`/`docker` backend, a Torch archive is dispatched once to the
  pinned Ubuntu worker. Docker mounts the exact archive read-only. Apple
  `container` cannot bind a regular file, so the CLI mounts a private temporary
  directory containing only a content-verified `archive.pt`; the original,
  staged copy, and returned manifest are bound to the original path/SHA and
  the staging directory is removed afterward. The exact output directory is
  read-write, the process uses the host uid/gid, and the runner's
  build-isolation mode disables Docker networking (Apple `container` 1.0
  supplies `--no-dns`, not a full IP-egress namespace). An internal marker
  prevents recursive redispatch. The inner loader still uses its rlimit
  subprocess. Resource limits are not a complete security boundary and
  `RLIMIT_AS` is best-effort on macOS.
- The preferred payload is
  `{"inputs": {name: tensor}, "goldens": {name: tensor}}`. For a separate
  unwrapped file, `--section inputs` or `--section goldens` assigns the whole
  tree to that manifest section. Auto-sectioned payloads reject an explicitly
  present empty section, and all modes reject flattened tensor-name collisions
  instead of silently dropping a tensor. An unsectioned `auto` payload remains
  golden-only and therefore needs an explicit actual manifest for offline SQNR.
- `VectorBundle` records component/slice/layer/op, AR/CL/phase/step,
  dtype/shape/layout, valid region, source key/hash, graph binding, and
  representation. `logical_fp`, `graph_native`, and `hmx_native` are kept
  separate: SQNR compares decoded logical floats only; native-KV physical
  bytes/layout are integrity-checked separately.

## What is verified here vs gated

This workspace contains the pinned SDK under
`./qnn/qnn/qairt/2.48.0.260626`. Host-runtime and device acceptance remain
separate gates:

- **Verified by tests (no SDK/container runtime/device needed):** the contracts and
  spec conversion; preset registry, SKU merge/capture, and Qwen3.5-Omni
  component packaging contract/runtime boundary; the file journal (atomic
  state, events, verified receipts,
  resume, rerun reuse, stage keys); the async orchestration state machine with
  a fake engine (submit/workflow/rerun/resume/cancel/failure); the CLI command
  tree; the restricted pickle loader (legit nested NumPy accepted; malicious
  GLOBAL/custom/object-dtype/oversized rejected); Docker and Apple `container`
  argv/mounts/platform/provenance and fail-closed runners; ADB
  argv/lease/staging-cleanup/doctor/gc; low-level and GenAI raw-slice optrace
  evidence normalization; automatic SQNR attribution, compatible fork-profile
  deltas, and fail-closed hotspot-only latency diagnosis.
- **Gated pending environment prep:** live worker-image construction/execution
  and SM8850 on-device build/validate/benchmark/optrace. These run
  through the same code paths once the selected runtime is ready and the image is built;
  until then `doctor` reports them as not-ready and the engine fails closed at
  preflight.
