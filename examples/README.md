# Canonical model examples

Use the CLI first:

```bash
qairt-agent plan --spec examples/qwen3_dense.json
qairt-agent workflow --spec examples/qwen3_dense.json
qairt-agent build --spec examples/qwen3_5_omni.json
```

`plan` must resolve the pipeline, AR policy, native-KV policy, and every output
role before a build starts. The paths are deployment-shaped placeholders; make
them point at mounted files in your project.

These files show shapes; [`docs/spec-reference.md`](../docs/spec-reference.md)
explains every field one can carry, including the `stage_configs` keys none of
these examples exercise.

| Example | Pipeline | Required model/vector inputs | Output root |
|---|---|---|---|
| `qwen3_dense.json` | low-level Python API | source ONNX (for example AR2073/CL4096), AIMET encodings, config, validation vectors | `/artifacts/qwen3` |
| `qwen3_vl.json` | low-level Python API | text ONNX+encodings, vision-with-projector ONNX+encodings, configs, explicitly scoped text-chain vectors | `/artifacts/qwen3-vl` |
| `vit.json` | low-level Python API | one ONNX+encodings pair, config, vectors; AR1 only | `/artifacts/vit` |
| `qwen3_5.json` | GenAI Builder | independent AR1/AR128 ONNX+encodings and validation manifest per AR | `/artifacts/qwen3.5` |
| `qwen3_5_omni_thinker.json` | GenAI Builder text lane | independent Thinker AR1/AR128 ONNX+encodings and validation manifest per AR | `/artifacts/qwen3.5-omni-thinker` |
| `qwen3_5_omni.json` | GenAI Builder packaging | Thinker AR1/AR128 plus audio ONNX+encodings; validation manifest per text AR | `/artifacts/qwen3.5-omni` |
| `qwen3_dense_float_reference_debug.json` | low-level Python API (**debug only**) | same as `qwen3_dense.json` plus the AR-matching float ONNX | `/artifacts/qwen3-float-reference-debug` |
| `qwen3_dense_float_reference_layer_debug.json` | low-level Python API (**debug only**) | same, plus a build that emitted diagnostic contexts | `/artifacts/qwen3-float-reference-layer-debug` |

`qwen3_5_omni.json` is build/package-only on the pinned SDK because its preset
sets `runtime_supported=false`; do not run the default validate/benchmark
workflow or claim end-to-end audio latency/SQNR.

The Qwen3-VL build includes both components, but unscoped automatic
validation/benchmarking fails closed because there is no audited vision-to-text
tensor bridge. This example explicitly sets
`stage_configs.validation.component = "text"` and
`stage_configs.benchmark.component = "text"`; its reports are therefore marked
`text_only` and exclude `vision_projector`. A vision-only run must select
`component = "vision"` and provide its own vision input vector manifest.
End-to-end image-to-text SQNR and latency are not claimed.

For a low-level AR2073/CL4096 source, the build exports separate AR1/AR128 ONNX
and AIMET-encoding artifacts, then writes target-ABI raw vector manifests under
the run's vector directory. Supplied target-compatible goldens win; when
retargeting makes them incompatible, ONNX Runtime captures new per-AR goldens
and records why.

`qwen3_dense_float_reference_debug.json` is a **debug-only** variant of
`qwen3_dense.json`, not a production template. It adds
`stage_configs.validation.float_reference`, which runs the float source ONNX
under ONNX Runtime and compares the device slice boundaries against it. The
mode is single-AR by construction, never enabled by default, and its report is
published beside — never instead of — the AIMET golden comparison. Copy the
block into your own spec only while investigating a divergence, and remove it
before a production run: `tensor_map` names must come from the transform
lineage of your exact export, because a boundary name is never guessed. Any
device tensor it cannot bind by an exact name match is listed under
`unmapped_tensors` rather than silently dropped.

`qwen3_dense_float_reference_layer_debug.json` is the deeper tier of the same
debug mode: `granularity: "layer"`. Instead of comparing only slice
boundaries, it **executes the diagnostic contexts** the build compiled and
compares every tapped tensor against the float graph, ordered by the float
graph's topology so the first divergence is the first row. It therefore needs a
build that actually produced those contexts, which is why the example also sets
`quality.dump_intermediates_on_failure` and `compile.enable_intermediate_outputs`
— run it against a build made with those, or the stage fails closed and names
the flag to set. It never degrades to slice boundaries: publishing a
boundary-only report under a layer-level label would be an overclaim.

Read its output with the same discipline as the boundary tier. The report is
labelled `first_observed_divergence_not_root_cause` for a reason — an
intermediate can read far worse than the tensors on either side of it when a
downstream operator discards the range the error lives in, so the first bad row
is where to start looking, not the answer.

`legacy/qwen3_5_low_level_experimental.json` preserves an old direct low-level
experiment for single-source Qwen3.5 AR rewriting. Files under `legacy/` are
not CLI workflow templates: the `qwen3_5` preset always resolves to GenAI
Builder and requires independent `attached_models_by_ar`. Prefer
`qwen3_5.json`.

## Vector reference policy

Raw inputs and supplied goldens live in content-addressed vector manifests.
Qwen3.5 and Omni Thinker use `validation_manifests_by_ar` so AR1 and AR128 are
never guessed from one shared file. If a selected manifest has inputs but no
goldens, validation captures ONNX Runtime goldens and records the fallback.
Their benchmark config must also provide an explicit `prompt` or `prompt_path`;
the two executable GenAI examples include a prompt.

For a default low-level workflow, validation and benchmark run every requested
AR against the exact runtime-index vector/graph pair and retain per-AR reports
plus a `results_by_ar` aggregate. An explicit stage `ar` is a single-AR debug
override. GenAI validation likewise covers both raw tensor routes, while its
generation benchmark is executor-managed; request raw GenAI optrace with one
explicit AR at a time.

For sliced low-level examples, `quality.sqnr_modes` drives validation directly.
`teacher_forced` and `chain` use a per-slice reference chain captured from the
exact transformed ONNX slices unless
`stage_configs.validation.slice_vector_manifests` supplies an explicit manifest
for every slice. Teacher forcing never reuses a device-produced boundary.
Because the canonical Qwen3 and Qwen3-VL examples request intermediate dumps on
failure, `qairt-agent plan` reports
`effective_compile.enable_intermediate_outputs=true`; the resulting diagnostic
contexts stay separate from production latency contexts.

Trusted local pickle data must first be converted:

```bash
qairt-agent vectors import-pickle golden.pkl \
  --output-dir artifacts/imported-vectors \
  --trusted-local --format auto --section auto --isolate
```

The preferred wrapped payload is
`{"inputs": {name: tensor}, "goldens": {name: tensor}}`. A separate
input-only or golden-only pickle uses `--section inputs` or
`--section goldens`. `auto` accepts restricted NumPy pickles and recognizes
modern `torch.save` zip archives; it does not accept a direct
`pickle.dump(torch.Tensor)`. NumPy imports execute locally. Torch archives use
the initialized project's Apple-container/Docker Ubuntu worker, so build and
smoke-test the pinned image before importing them.

## Output layout

Each example owns only its `output_root`; preset-relative directories are:

```text
manifests/{run_id}
runs/{run_id}/{config,vectors,diagnostics,stages}
runs/{run_id}/build/{variants,transformed,converted,contexts}
runs/{run_id}/build/diagnostic_contexts
runs/{run_id}/genai/{container,cache}
```

Low-level examples use the `build/...` roles. GenAI examples use the
`genai/...` roles. Source files are not copied into the output root.
