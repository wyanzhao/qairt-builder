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

| Example | Pipeline | Required model/vector inputs | Output root |
|---|---|---|---|
| `qwen3_dense.json` | low-level Python API | source ONNX (for example AR2073/CL4096), AIMET encodings, config, validation vectors | `/artifacts/qwen3` |
| `qwen3_vl.json` | low-level Python API | text ONNX+encodings, vision-with-projector ONNX+encodings, configs, explicitly scoped text-chain vectors | `/artifacts/qwen3-vl` |
| `vit.json` | low-level Python API | one ONNX+encodings pair, config, vectors; AR1 only | `/artifacts/vit` |
| `qwen3_5.json` | GenAI Builder | independent AR1/AR128 ONNX+encodings and validation manifest per AR | `/artifacts/qwen3.5` |
| `qwen3_5_omni_thinker.json` | GenAI Builder text lane | independent Thinker AR1/AR128 ONNX+encodings and validation manifest per AR | `/artifacts/qwen3.5-omni-thinker` |
| `qwen3_5_omni.json` | GenAI Builder packaging | Thinker AR1/AR128 plus audio ONNX+encodings; validation manifest per text AR | `/artifacts/qwen3.5-omni` |
| `qwen3_4b_2layer_genai.json` | GenAI Builder (debug) | 2-layer Qwen3-4B sliced ONNX+encodings per AR | `/artifacts/qwen3-4b-2layer-genai` |
| `qwen3_4b_2layer_lowlevel.json` | low-level Python API (debug) | 2-layer Qwen3-4B sliced ONNX+encodings, vectors | `/artifacts/qwen3-4b-2layer-lowlevel` |

⚠️ `qwen3_4b_2layer_*.json` are **debug/slicing-test templates**. The GenAI
example deliberately applies the `qwen3_5` preset to a Qwen3-4B model; the
low-level example uses the route-correct `qwen3_dense` preset. They exist for
validating the ONNX slicing pipeline and do not represent supported production
configurations.

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
