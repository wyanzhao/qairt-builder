# QAIRT Agent

`qairt-agent` is a Python-only, agent-native orchestration layer for Qualcomm
AI Runtime (QAIRT) 2.48 GenAI model preparation, compilation, execution, and
diagnostics. It is designed for Claude Code, Codex, and direct Python callers.
The thin JSON-producing CLI is the primary automation interface; direct Python
imports are for focused debugging and custom analysis, while MCP is retained as
a compatibility surface.
It never constructs QAIRT CLI commands, invokes QAIRT executables through
`subprocess`, or uses the QAIRT C++ API.

The framework is intentionally separate from the unpacked QAIRT SDK. It exposes
typed Python and MCP interfaces for:

- ONNX and AIMET encoding validation
- test-vector and golden preparation
- AR/CL conversion, model splitting, and MHA2SHA
- converter and standalone quantizer workflows
- per-slice AR1 + AR128 weight-sharing context binaries
- production `LLMContainer`/`WorkflowContainer` packaging through the QAIRT
  GenAI Builder Python API
- native-KV chain execution
- full-reference, teacher-forced, and chained SQNR analysis
- production latency and per-op profiling
- family-specific Qwen3 Dense, Qwen3 MoE, Qwen3-VL, and Qwen3.5 configuration

## Supported environment

The QAIRT worker is pinned by the editable
[`harness/constraints.json`](harness/constraints.json) contract:

- Ubuntu 22.04 x86_64
- Python 3.10
- QAIRT `2.48.0.260626` (build `260626120635`)
- HTP target `SM8850 / v81 / soc_model 660`

The Python package can be imported without QAIRT installed. Operations that
need the SDK fail with a structured preflight error.

## Install

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install -e '.[mcp,dev]'
```

Initialize the repository after unpacking QAIRT beneath `qnn/qnn`. The
installation root may itself be an SDK or contain a versioned SDK such as
`qnn/qnn/qairt/2.48.0.260626`; discovery does not move it.

```bash
qairt-agent init --root .
qairt-agent image build --root .
qairt-agent doctor --root .
```

`worker.backend = "auto"` selects Apple `container` on macOS and Docker on
Linux; native Ubuntu execution is opt-in. Both container backends run the same
Ubuntu 22.04, Python 3.10, `linux/amd64` worker. Apple Silicon uses Rosetta.
`init` works from either an editable checkout or an installed wheel: it copies
the harness-selected Dockerfile and dependency lock, stages the exact running
agent sources under `docker/.generated/`, and appends a final managed
`.dockerignore` block that excludes `qnn/`, models, artifacts, and tensor/model
payloads from the image build context. The worker build never treats an
unrelated project `pyproject.toml`, `README.md`, or `src/` tree as agent code.
`image build` accepts the image only after a mounted-SDK import smoke test;
`image smoke` repeats the gate without rebuilding. Docker smoke is
network-isolated. Apple `container` 1.0 has no equivalent to Docker's
`--network none`, so its smoke disables DNS but is explicitly recorded as
best-effort rather than hard network isolation.

The detached CLI worker receives the resolved SDK, image/device provenance,
harness path, and ADB settings. It must claim the journal within the bounded
startup timeout (default 30 seconds); otherwise the CLI writes
`logs/worker-launch.log`, marks the job `FAILED`, and returns a structured
error instead of leaving it queued. Direct Python use should bind the editable
project harness with `QairtAgentClient.from_project(".")`; `--inline` and
Python execution still require the QAIRT environment in that process. See
[Worker runtimes](docs/worker-runtimes.md) for initial Apple service/DNS setup
and the version-upgrade procedure.

Real-device stages fail closed unless both `QAIRT_AGENT_ADB_SERIAL` and
`QAIRT_AGENT_ADB_SERVER` are set. Before calling the QAIRT Python API they put
the context and raw vector artifacts in an integrity-checked, leased sandbox
under `/data/local/tmp/qairt-agent/`, then clean that exact attempt directory.
This sandbox is lifecycle/audit evidence; QAIRT 2.48's explicit `Device` object
still performs its own internal runtime deployment because the Python API does
not expose reuse of that remote sandbox as its execution working directory.
Native loopback, Docker's `host.docker.internal`, and Apple container's
`host.container.internal` share one canonical server+serial lease identity
without changing the actual ADB connection address. A separate process
heartbeat keeps long GIL-holding QAIRT calls live;
`device gc` rechecks its owner token under a per-device lock before removing
only an exact `<job>/<stage>/<attempt>/` sandbox.

## Canonical build specification

The native CLI consumes a complete `WorkflowSpec` with a `preset`. Legacy
`BuildSpec.family` input remains accepted and is normalized through
`to_workflow_spec`; it is not a second routing policy. The JSON files listed in
`examples/README.md` are the canonical references. Important fields are:

- `sources.text`: ONNX, AIMET encodings, and Hugging Face `config_path`
- `sources.vision`: required for Qwen3-VL
- `sequence`: ARs, context lengths, weight sharing, and native KV
- `split.embedding_mode`: `lut`, `compiled`, or `external`
- `vectors`: supplied validation/calibration manifests or `capture` mode;
  Qwen3.5/Omni Thinker use `validation_manifests_by_ar`
- `target`: `chipset`, `dsp_arch`, and `soc_model`
- `stage_configs`: distinct inputs for build, validation, benchmark, and
  quality/latency diagnosis
- `output_root`: the only root beneath which a run publishes artifacts

Qwen3-VL accepts only
`sources.vision_projector_location = "inside_vision_onnx"`; the vision ONNX
must already contain the projector. Normal Qwen3.5 and Omni Thinker workflows
always resolve to GenAI Builder and require independent AR1/AR128
ONNX+encoding pairs. The separately gated direct low-level experiment can
derive AR1 and AR128 from one source only with
`sequence.qwen35_experimental_auto_ar = true` plus
`metadata.qwen35_runtime_validation.cases`; it is not the CLI preset path.
Caller-supplied pass/fail booleans are not accepted as evidence. Weight sharing
is exactly AR1 + AR128, and native KV requires every context length to be
divisible by 256.

There are two intentionally separate build lanes:

- `qairt_build` uses the low-level QAIRT Python APIs for deterministic
  AR/CL conversion, semantic splitting, MHA2SHA, conversion, quantization, and
  per-slice context binaries. This is also the lane used for diagnostic taps,
  chain SQNR, and optrace.
- `qairt_build_genai_container` lets the selected public SDK family builder own
  production transformation/compilation and saves an `LLMContainer` or
  `WorkflowContainer`. Qwen3.5 and Omni Thinker use
  `Qwen3_5BuilderHTP.from_pretrained` directly; supported other families may
  use `GenAIBuilderFactory`. It does not call the low-level build in the same
  invocation, so the model is not compiled twice.

QAIRT 2.48's factory explicitly dispatches Qwen3 MoE. Qwen3.5 is instead
pinned to the public `Qwen3_5BuilderHTP.from_pretrained` constructor because
the factory does not recognize every Qwen3.5/Omni Thinker architecture name.
Qwen3 Dense uses the SDK's generic HTP builder and is marked as requiring
device golden validation. The normal `qwen3_vl` preset always uses the
low-level lane. Its build publishes the vision component and text slices, but
this version does not claim an automatic end-to-end image-to-text runtime route. Unscoped
validation/benchmarking fails closed. An intentionally partial run must set
`stage_configs.validation.component` and
`stage_configs.benchmark.component` to `text` or `vision`; the resulting report
is labelled `text_only` or `vision_only`, and vision-only execution requires a
separate vision vector manifest.

For Qwen3.5, the GenAI Builder lane does not use experimental single-source AR
rewriting. Its `metadata.attached_models_by_ar` must supply independent AR1 and
AR128 ONNX+encoding pairs.

For low-level Qwen3/Qwen3-VL inputs such as AR2073/CL4096, AR conversion exports
separate AR1 and AR128 ONNX plus AIMET-encoding artifacts under the resolved
variant/transform directories. It also publishes exact per-AR raw vector
manifests. A supplied golden is retained only when it proves the target ABI;
otherwise target-shape goldens are captured with ONNX Runtime and the
provenance is recorded.

With the default CLI stage configuration, validation and benchmark fan out
over every requested low-level AR and bind each run to the exact graph and
per-AR vector manifest in the content-verified runtime index. Per-AR
`sqnr_report_arN`, `latency_report_arN`, and optional
`optrace_evidence_arN` artifacts are retained; the canonical report exposes
`coverage` and `results_by_ar`. A missing AR binding fails closed. Setting
`stage_configs.validation.ar` or `stage_configs.benchmark.ar` is an intentional
single-AR debug override. Explicit custom graph/routes are executed exactly as
provided and are not implicitly fanned out.

Supplied golden tensors are the first-choice SQNR reference. When a selected
validation manifest contains raw inputs but no goldens, validation
automatically captures ONNX Runtime outputs and records the fallback
provenance; it never silently replaces supplied goldens. Trusted local pickle
vectors must first be converted to immutable raw tensors plus a vector
manifest with `qairt-agent vectors import-pickle`. The importer accepts nested
NumPy pickle trees and modern `torch.save` archives. Torch loading is
weights-only, CPU-mapped, and subprocess-isolated; direct
`pickle.dump(torch.Tensor)` is rejected. Use `--section inputs` or
`--section goldens` for a separate unwrapped file. NumPy imports stay local.
Torch archives are automatically sent to the configured Ubuntu worker
(Apple `container` on macOS, Docker on Linux), with a read-write output mount
and host uid/gid. Docker mounts the source file read-only. Because Apple
`container` cannot bind a regular file, the CLI mounts a private temporary
directory containing only a content-verified `archive.pt`, rechecks the
original/copy/result provenance, and deletes the directory afterward. Docker
uses `--network none`; Apple `container` 1.0 supplies `--no-dns`, not hard
IP-egress isolation.

For Qwen3.5 and Omni Thinker, build records the saved container's public
compiled-model split route when the SDK exposes it. SQNR runs against that raw
tensor route and fails closed if no auditable route is available. Latency
benchmarking loads the saved container through its public executor and requires
an explicit `stage_configs.benchmark.prompt` or `prompt_path`; generated text
is hashed rather than copied into the performance claim. Multi-AR raw-tensor
SQNR validates every exact AR route. The public generation benchmark is one
executor-managed prefill/decode workload and does not claim which internal AR
graph ran. Multi-AR GenAI optrace fails closed; set an explicit benchmark `ar`
when collecting raw CompiledModel profiling evidence.

All embedding modes retain an explicit embedding semantic split:

- `lut`: export the embedding table as a lookup artifact
- `compiled`: compile the embedding split into a context binary
- `external`: route embeddings supplied by an external producer; the split
  boundary remains in the manifest

Every model spec has its own `output_root`. The resolved preset renders all
artifact directories beneath it:

| Role | Relative directory |
|---|---|
| immutable manifests | `manifests/{run_id}` |
| run state/config/vectors/reports/stages | `runs/{run_id}/{config,vectors,diagnostics,stages}` |
| low-level variants/transformed/converted/contexts | `runs/{run_id}/build/{variants,transformed,converted,contexts}` |
| low-level diagnostic contexts | `runs/{run_id}/build/diagnostic_contexts` |
| GenAI container/cache | `runs/{run_id}/genai/{container,cache}` |

See [the example index](examples/README.md) for the model-specific roots and
input policy.

## Native workflow

The native workflow adds a persistent file job journal, family presets, a thin
CLI, and four asynchronous MCP tools on top of the synchronous stage engine.
The core workflow is `spec -> build -> validate -> benchmark`. Run `diagnose`
from that job only after a quality or latency regression, then adjust the spec
and rerun while reusing unchanged stages.

```bash
qairt-agent init                       # write qairt-agent.toml + state dirs
qairt-agent image build --root .       # build the pinned Ubuntu worker
qairt-agent doctor                     # verify SDK metadata, ABI, target
qairt-agent plan --spec spec.json
qairt-agent workflow --spec spec.json  # background job; prints {job_id,state,status_path}
qairt-agent job watch JOB_ID --follow --after-seq 0
qairt-agent rerun --from-job JOB_ID --spec adjusted.json
```

`WorkflowSpec` replaces the flat `family` enum with a `preset` reference plus
an optional SKU overlay; `BuildSpec` is still readable (`to_workflow_spec`). The
`vit` uses a standalone low-level `qairt.convert -> qairt.compile` lane. The
`qwen3_5_omni` preset packages a dedicated
`Qwen3OmniAudioEncoderBuilderHTP` plus a pinned `Qwen3_5BuilderHTP`; it remains
`runtime_supported=false` because QAIRT 2.48 does not provide validated
end-to-end audio workflow execution. It never aliases the text or audio model
family. See
[Native workflow](docs/native-workflow.md) for the journal, presets, worker
runtimes, ADB, per-stage configuration, pickle security, and what is verified here
versus gated.

## MCP

```bash
qairt-agent-mcp            # four asynchronous tools by default
qairt-agent-mcp --legacy   # deprecated synchronous tool set
```

Default tools (backed by the job journal):

```text
submit_job(spec, stages?, from_job?)  -> {job_id, state, status_path}
get_job(job_id, after_seq?)           -> {status, events}
cancel_job(job_id)                    -> {ok, job_id}
resume_job(job_id)                    -> {job_id, state, status_path}
```

The original ~18 synchronous tools (`qairt_build`, `qairt_validate`, the expert
stages, ...) remain behind `--legacy` / `QAIRT_AGENT_MCP_LEGACY=1` and are
marked deprecated. SQNR and latency stay report-only; structural failures
(hash mismatch, missing tensors, non-finite values, invalid routing, transform
inequivalence) remain errors.

See [MCP tools](docs/mcp-tools.md) for the call contract and
[Architecture](docs/architecture.md) for artifact and execution details.
