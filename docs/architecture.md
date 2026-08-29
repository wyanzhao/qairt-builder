# Architecture

## Trust boundaries

`qairt-agent` is installed beside, not inside, the QAIRT SDK. The package never
constructs a QAIRT command line or calls a QNN/QAIRT executable through
`subprocess`. The vendor's Python modules may load their own native libraries;
that behavior remains inside the QAIRT trust boundary. The integration surface
is limited to the QAIRT GenAI Builder and low-level Python APIs; the QAIRT CLI
and C++ APIs are out of scope.

The QAIRT adapter is pinned by `harness/constraints.json` to QAIRT
`2.49.0.260730`, build `260730134355`. Any other version is rejected before a
transform or build. The same contract owns the worker image, Ubuntu/Python,
dependency file, and host-runtime CLI versions.
`SM8750 / HTP v79 / soc_model 69` must resolve explicitly through the SDK.
The canonical target fields are `chipset`, `dsp_arch`, and `soc_model`. There is
no V79 or default-SoC fallback.

## Canonical request contract

`WorkflowSpec` is the native CLI source of build intent and names a `preset`.
The compatibility `BuildSpec` names a `family` and is immediately normalized
with `to_workflow_spec`; model policy is never inferred from filenames. The
shared canonical top-level structure is:

```text
preset (or compatibility family)
sources
  text:   onnx, encodings, config_path
  vision: onnx, encodings, config_path       # Qwen3-VL only
  vision_projector_location                  # inside_vision_onnx
output_root
sequence
  ars, context_lengths, weight_sharing, native_kv
  qwen35_experimental_auto_ar
split
  decoder_slice_count, embedding_mode, split_lm_head
quantization
vectors
target
quality
benchmark
```

Paths are validated structurally by the contract and for existence/content by
preflight. Nested `onnx` and `encodings` are accepted input aliases; manifests
serialize their canonical `onnx_path` and `encodings_path` names. Weight sharing accepts only the AR set `{1, 128}`. Native KV defaults on and
requires every context length to be divisible by 256.

`vectors.mode` is either `provided` or `capture`. Provided vector sets name an
immutable shared validation manifest, an AR-keyed
`validation_manifests_by_ar` map, an optional calibration manifest, or a valid
combination. Qwen3.5 and Omni Thinker require the AR-keyed form. Vector
manifests content-address every raw tensor file and record name, dtype, shape,
role, and byte order.

## Stateless control plane

The source of truth is the typed contract in `qairt_agent.contracts` and a chain
of immutable run manifests:

```text
WorkflowSpec / compatibility BuildSpec
    |
    v
manifest-r000000-<sha256>.json --stage--> manifest-r000001-<sha256>.json
              |                                      |
              +-- path + SHA256                      +-- path + SHA256
```

The synchronous stage engine has no implicit "current model." A stage
invocation receives the manifest path and expected SHA256, verifies the content
before parsing, and then publishes a new revision atomically. A path without
its hash is rejected. Each revision points to the immediately preceding
manifest artifact; publishing different content for an existing run/revision
is a conflict. The native controller adds an explicit file job journal around
that engine; it does not weaken the manifest/hash boundary.

The stage key is derived from input hashes, normalized configuration, and the
QAIRT build. A retry may reuse a completed artifact only when that artifact is
already named by the explicit manifest.

Native KV, recurrent state, and device handles are invocation-local. Prefill and
all requested decode steps run inside one `run_chain` or `benchmark` call.
Cross-call native-HMX continuation is deliberately unsupported.

## Build data flow

The framework exposes two production entrypoints. They share `BuildSpec`,
preflight, content-addressed artifacts, and immutable manifests, but they do
not invoke one another:

```text
qairt_build_genai_container
  Qwen3.5 -> Qwen3_5BuilderHTP.from_pretrained --+
  other supported families -> GenAIBuilderFactory +-> build -> container.save
                                                      |
                                                      +--> LLMContainer / WorkflowContainer

qairt_build
  low-level Python APIs -> explicit per-stage artifacts and contexts
                                     |
                                     +--> diagnostics / SQNR / optrace
```

This separation is deliberate because `GenAIBuilderHTP.build()` already
performs transform, conversion, quantization, and compilation. Calling the
low-level build behind it would compile the model twice.

The low-level lane is:

```text
inspect source ONNX + config_path + family profile
          |
          v
provided/captured validation and calibration vectors
          |
          v
AR/CL conversion
          |
          v
balanced split -> MoE adapt -> MHA2SHA
          |
          v
convert -> quantize
          |
          v
one context per semantic slice and CL
  [AR1 graph + AR128 graph, shared weights]
          |
          +------> production execution / wall latency
          |
          +------> separate selective-output diagnostic context
```

When the source graph is exported at a wider AR/CL (for example
AR2073/CL4096), the variant stage exports independent target ONNX and AIMET
encodings for AR1 and AR128. Vector retargeting publishes target-ABI raw
manifests beside those artifacts. A supplied golden is used only when it proves
the exact target output ABI; otherwise ONNX Runtime captures a new per-AR
reference with explicit provenance.

Compilation never implies execution order. `qairt.compile(list[Model])` packages
graphs; `SliceChainRunner` owns embedding -> decoder slices -> LM-head routing
and selects a graph by its exact manifest name.

After a build, the orchestrator inspects each transformed slice and publishes a
`slice_routes_cl<CL>.json` artifact containing context paths, AR-to-graph-name
mappings, exact boundary routes, and invocation-local native-state slots.
Inputs that cannot be proven from exact tensor names remain explicitly listed
as `unresolved_external_inputs`; they are never guessed or silently wired.

Embedding is always a semantic split. Its packaging mode controls what happens
to that split:

- `lut`: publish a lookup table and route its output into the first decoder
- `compiled`: publish and execute an embedding context binary
- `external`: accept embeddings from an external producer while retaining the
  graph boundary and route in the manifest

## Family profiles

Profiles do more than choose a builder class. Each profile declares:

- architecture fingerprints and decoder-layer discovery
- the Hugging Face `config_path` used to resolve architecture metadata
- tensor roles for hidden state, logits, mask, position, KV and recurrent state
- embedding and LM-head boundaries
- MHA2SHA start points and head mapping
- native-KV eligible inputs and outputs
- Genie/Workflow configuration fields

Qwen3-VL uses two source models. The contract accepts only
`vision_projector_location = "inside_vision_onnx"`: the Vision ONNX includes the
projector and must emit `text_hidden_size` visual embeddings. The vision graph
is compiled as its own workflow node; the Text ONNX supplies the text-side
embedding/decoder/LM-head slices.

The canonical Qwen3-VL preset remains in the low-level lane. It publishes the
vision component and text decoder slices, but automatic unscoped
validation/benchmarking fails closed. Callers may explicitly select a
`text_only` or `vision_only` component run; the runtime index records excluded
components, and vision-only execution requires its own vectors. The framework
does not claim an unproven end-to-end ImageT2T execution path.

QAIRT 2.49's factory explicitly dispatches Qwen3 MoE. Qwen3.5 and Omni Thinker
bind the public `Qwen3_5BuilderHTP.from_pretrained` constructor directly
because the factory does not recognize every supported architecture name. The
canonical Qwen3 Dense preset remains low-level and requires device golden
validation before release.

Qwen3.5 single-source AR conversion requires the explicit
`sequence.qwen35_experimental_auto_ar = true` opt-in. The adapter checks
recurrent/conv/KV contracts, initializer identity, source-vs-derived ONNX
equivalence, standalone-vs-joint context equivalence, and the SDK's
`16 KV heads * 256 head_dim` start-point assumption. A failed check produces no
weight-sharing artifact.

That experimental single-source conversion belongs only to the low-level lane.
The GenAI Builder lane requires independent ONNX and AIMET encodings for every
requested Qwen3.5 AR through `attached_models_by_ar`, disables
`skip_ar_conversion`, attaches each model explicitly, and only then enables
weight sharing.

The legacy direct low-level Qwen3.5 experiment obtains device evidence from
`metadata.qwen35_runtime_validation.cases`. Cases may be keyed by exact graph
name or nested as `decoder_slice -> AR -> vector manifest`. Each vector
manifest contains that graph's inputs and goldens. During the same build call,
the adapter compiles standalone and joint diagnostic contexts, stages those
contexts and vectors in a leased, content-verified ADB audit sandbox, executes
every named graph through an explicit QAIRT Android `Device`, writes reopenable
comparison reports, hashes those reports, and then mints invocation-scoped
evidence. The sandbox does not replace QAIRT's own internal runtime deployment;
QAIRT 2.49 exposes no Python working-directory override for that reuse. The
request cannot authorize compilation by supplying boolean evidence.

The canonical Qwen3.5 and Omni Thinker GenAI build instead records the public
compiled-model splits exposed by the saved `LLMContainer`. The runtime index
binds their exact graph order, contexts, raw tensor ABI, and AR-specific vector
manifest. SQNR fails closed if that auditable tensor route is unavailable.
Latency loads the saved container through its public executor and requires an
explicit prompt or prompt file.

## Evidence separation

Production contexts contain no intermediate outputs. Wall latency is measured
on a warmed production context. Diagnostic contexts contain only requested
taps; their latency is never reported as production performance.

Quality reports distinguish:

- `full_reference`: end-to-end output against golden
- `teacher_forced`: each slice receives its own golden boundary input
- `chain`: device output feeds the next slice

SQNR uses reference energy and float64 accumulation. Low SQNR or high latency is
informational by product decision; quality and benchmark records never contain
a product threshold or pass/fail verdict. Latency reports warmed production-wall
samples, p50/p95, dispersion, and optional A/A noise calibration. Op trace
cycles are work attribution and are not added together as wall latency.

Supplied golden tensors have reference priority. If the exact selected vector
manifest has raw inputs but no goldens, validation automatically captures
goldens with ONNX Runtime and records the model hash, ORT version, providers,
and fallback reason. It never replaces supplied goldens; missing both usable
goldens and executable inputs is an error.

Missing tensors, invalid mappings, non-finite data, transform inequivalence,
manifest hash mismatch, and graph/context mismatches remain execution errors.
