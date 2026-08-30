# MCP tools

## Default: asynchronous job tools

By default the server exposes four short tools backed by the file job journal.
The agent no longer orchestrates a dozen fine-grained tools; it submits a job
and polls/watches it by id.

```text
submit_job(spec, stages?, from_job?)  -> {job_id, state, status_path}
get_job(job_id, after_seq?)           -> {status, events}
cancel_job(job_id)                    -> {ok, job_id}
resume_job(job_id)                    -> {job_id, state, status_path}
```

- `submit_job` returns immediately; the job keeps running in a detached worker.
  `stages` defaults to `["build"]`; pass
  `["build","validate","benchmark"]` for the standard workflow. Submit
  `["diagnose"]` from that job only after a reported regression. `from_job`
  seeds the initial manifest for a continuation
  (`validate`/`benchmark`/`diagnose`).
- `get_job` returns the current `JobStatus` plus events after `after_seq`, so
  a watch can resume from the last sequence number it saw.
- `cancel_job` sets the journal cancel flag; the worker stops before the next
  stage. `resume_job` continues an interrupted (orphaned) job from its last
  verified receipt.

Failures return the same structured error envelope as below.

## Legacy synchronous tools (deprecated)

`qairt-agent-mcp --legacy` (or `QAIRT_AGENT_MCP_LEGACY=1`) exposes the original
~18 synchronous tools. They are deprecated; prefer the asynchronous API. Their
stateless continuation contract is documented below for reference.

Every response is a JSON-serializable `ToolResult`. QAIRT objects and device
handles never cross the MCP boundary. The server calls QAIRT Python APIs only;
none of these tools constructs or executes a QAIRT CLI command.

## Stateless continuation

Initial tools accept a complete canonical `BuildSpec`. A successful stage
returns an `ArtifactRef` for an immutable manifest:

```json
{
  "ok": true,
  "data": {},
  "manifest": {
    "path": "/artifacts/run-id/manifest-r000001-<sha256>.json",
    "sha256": "<64 lowercase hexadecimal characters>",
    "size_bytes": 1234,
    "kind": "manifest",
    "media_type": "application/json"
  }
}
```

Every continuation call must send both `manifest_uri` (the manifest filesystem
path) and `manifest_sha256`. The server hashes the file before parsing it. A
path without its SHA256, a stale hash, or a manifest revision conflict returns a
structured error; there is no hidden job ID or current-session state.

Native KV, recurrent tensors, and device handles cannot be resumed across MCP
calls. A complete prefill/decode sequence must remain inside one `run_chain` or
`benchmark` call.

`qairt_build` publishes one `slice_routes_cl<CL>.json` per context length.
Pass its `routes` and `contexts` fields to `qairt_run_chain` or to
`qairt_benchmark` for warmed chain/sequence timing. Any
`unresolved_external_inputs` must be supplied explicitly before execution.

## Pipeline tools

- `qairt_plan(spec, offline=False)`
- `qairt_generate_config(spec)`
- `qairt_build(spec)`
- `qairt_build_genai_container(spec, config={})`
- `qairt_validate(manifest_uri, manifest_sha256, vector_manifest=None, config={})`
- `qairt_benchmark(manifest_uri, manifest_sha256, config={})`
- `qairt_diagnose_quality(manifest_uri, manifest_sha256, config={})`
- `qairt_diagnose_latency(manifest_uri, manifest_sha256, config={})`

`spec` must include `sources.*.config_path`, `output_root`, `sequence`,
`split`, `vectors`, and an explicit target. The canonical target uses
a registered target name (or a tuple that matches one exactly); `htp_arch` is not a contract field.
See the canonical JSON files and routing notes under `examples/`.

`qairt_build` is the low-level Python API lane and publishes explicit
per-slice context binaries. `qairt_build_genai_container` is a separate
production packaging lane owned by the selected public SDK family builder.
Qwen3.5 and Omni Thinker use `Qwen3_5BuilderHTP.from_pretrained` directly;
supported other families may use `GenAIBuilderFactory`. The lane saves an
`LLMContainer` or `WorkflowContainer` and never calls the low-level build in
the same invocation.

For Qwen3.5, the GenAI tool requires independently exported AR1 and AR128
models and encodings:

```json
{
  "metadata": {
    "attached_models_by_ar": {
      "1": {
        "model_path": "/models/qwen3.5/decode_ar1.onnx",
        "encodings_path": "/models/qwen3.5/decode_ar1.encodings"
      },
      "128": {
        "model_path": "/models/qwen3.5/prefill_ar128.onnx",
        "encodings_path": "/models/qwen3.5/prefill_ar128.encodings"
      }
    }
  }
}
```

Qwen3 Dense requires device golden validation before release. The canonical
`qwen3_vl` preset uses `qairt_build` and the low-level Python API; it is not
automatically redirected to GenAI Builder.

Qwen3.5 and Omni Thinker workflow specs pair the attached model map above with
`vectors.validation_manifests_by_ar`, normally one manifest for AR1 and one for
AR128. Supplied goldens take priority. If a selected manifest contains raw
inputs but no goldens, validation captures an ONNX Runtime reference and
records that fallback in the report.

For experimental Qwen3.5 multi-AR builds,
`metadata.qwen35_runtime_validation.cases` must resolve every decoder
slice/AR (or exact graph name) to a vector manifest with inputs and goldens.
`qairt_build` performs standalone-vs-golden, joint-vs-golden, and
standalone-vs-joint execution before the adapter authorizes a production
weight-sharing context.

## Expert tools

- `qairt_prepare_vectors(manifest_uri, manifest_sha256, config={})`
- `qairt_ar_convert(manifest_uri, manifest_sha256, config={})`
- `qairt_split(manifest_uri, manifest_sha256, config={})`
- `qairt_mha2sha(manifest_uri, manifest_sha256, config={})`
- `qairt_convert(manifest_uri, manifest_sha256, config={})`
- `qairt_quantize(manifest_uri, manifest_sha256, config={})`
- `qairt_compile_context(manifest_uri, manifest_sha256, config={})`
- `qairt_run_graph(manifest_uri, manifest_sha256, config={})`
- `qairt_run_chain(manifest_uri, manifest_sha256, config={})`
- `qairt_profile(manifest_uri, manifest_sha256, config={})`

Expert tools consume explicit artifact references. They do not mutate a hidden
pipeline session. They are a deprecated compatibility/debugging surface; normal
long transform, compile, and profiling work should use the CLI's detached job
worker and resumable journal.

## Reports and errors

Quality operations report `full_reference`, `teacher_forced`, and `chain`
comparisons. Benchmark operations report warmed production-wall latency and
optional op work attribution. Neither operation applies an SQNR or latency
pass/fail threshold.

Tool failures use:

```json
{
  "ok": false,
  "error": {
    "code": "artifact_hash_mismatch",
    "message": "artifact integrity check failed",
    "stage": "validate",
    "retryable": false,
    "details": {}
  }
}
```

Hash mismatch, missing artifacts/tensors, invalid family configuration,
non-finite values, transform inequivalence, and graph-routing errors are hard
errors rather than low-quality measurements.
