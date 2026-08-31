# `WorkflowSpec` reference

One place for every field a spec can carry. Spec knowledge used to be spread
across `README.md`, `docs/first-run.md`, `docs/native-workflow.md` and
`examples/README.md`, each describing a part; this file is the reference those
now point at.

`tests/test_spec_reference.py` asserts that every field on the contract models
appears here and that every field documented here exists on a model, so the two
cannot drift apart in either direction.

Run `qairt-agent plan --spec spec.json` after every edit. The resolved output is
the contract: it renders `pipeline`, the AR policy, the native-KV policy,
`effective_target` (including whether the target is verified),
`effective_compile`, and `effective_benchmark`.

## Top level

| Field | Default | What it is |
| --- | --- | --- |
| `name` | `"model"` | Build identity. Changing it invalidates build reuse. |
| `preset` | *required* | The routing authority: `qwen3_dense`, `qwen3_moe`, `qwen3_vl`, `qwen3_5`, `qwen3_5_omni_thinker`, `qwen3_5_omni`, `vit`. Never inferred from a filename. |
| `sku` | `null` | A captured `SkuOverlay` pinning a reference model's identity and slice boundaries. |
| `sources` | *required* | Where the exports live. See below. |
| `output_root` | *required* | The only artifact root. Build identity: relocating it must rebuild there, never copy a prior `BuildSpec` or prior artifact paths in. |
| `sequence` | see below | AR set, context lengths, weight sharing, native KV. |
| `split` | see below | Decoder slices, embedding mode, LM-head split. |
| `transforms` | see below | MHA2SHA and KV-cache IO permutation. |
| `quantization` | see below | `apply_encodings` (production) or `calibrate` (deprecated debug). |
| `vectors` | see below | Validation and calibration vector manifests. |
| `compile` | see below | Diagnostic outputs and compiler options. |
| `target` | active harness target | Named or full tuple; must match the reviewed registry exactly. |
| `quality` | see below | Which SQNR modes run; whether a failure dumps intermediates. |
| `benchmark` | see below | Warmup/measured sampling and optrace. |
| `stage_configs` | `{}` | Per-stage continuation inputs. |
| `metadata` | `{}` | Free-form, plus the keys named under **Metadata** below. |

## `sources` (`ModelSourcesSpec`)

| Field | Default | What it is |
| --- | --- | --- |
| `text` | *required* | The text/decoder export (`ModelSourceSpec`). |
| `vision` | `null` | Qwen3-VL only: the vision ONNX with its projector already integrated. |
| `audio` | `null` | Omni packaging only: the audio encoder ONNX. |
| `vision_projector_location` | `null` | Must be `inside_vision_onnx`; a separate projector is refused. |

Each `ModelSourceSpec`:

| Field | Default | What it is |
| --- | --- | --- |
| `onnx_path` | *required* | The export. Its external-data side-car files are canonical inputs too. |
| `encodings_path` | `null` | AIMET encodings. Required when `quantization.mode = "apply_encodings"`. |
| `tokenizer_path` | `null` | Hugging Face tokenizer directory, for the GenAI container. |
| `config_path` | `null` | Hugging Face-style `config.json`. When present its `architectures` are cross-checked against the declared preset; a contradiction fails at plan time. |
| `aimet_config_path` | `null` | AIMET quantsim config, when the export needs a non-default one. |

## `sequence` (`SequenceSpec`)

| Field | Default | What it is |
| --- | --- | --- |
| `ars` | `(1, 128)` | The AR set. Weight sharing packages exactly `{1, 128}` per semantic slice. |
| `context_lengths` | `(4096,)` | Not pinned to 4096: any CL divisible by 256 under native KV flows through. Use one CL per workflow so vectors bind automatically. |
| `weight_sharing` | `true` | One context per semantic slice with both ARs inside. |
| `native_kv` | `true` | HMX-native KV layout. Must preserve exact tensor names, graph routing, shape/layout and CL alignment. |
| `qwen35_experimental_auto_ar` | `false` | Derive AR variants from one Qwen3.5 export. Fail-closed and gated by adapter validation evidence; production supplies independent per-AR exports instead. |

## `split` (`SplitSpec`)

| Field | Default | What it is |
| --- | --- | --- |
| `decoder_slice_count` | `1` | Decoder slices. The layer distribution is reproduced locally and marked `advisory`; what a build verifies is the split count. |
| `embedding_mode` | `lut` | `lut` and `external` publish an embedding boundary instead of compiling it. |
| `split_lm_head` | `true` | Keep the LM head in its own split; the final layer folds into it. |

## `transforms` (`TransformSpec`)

| Field | Default | What it is |
| --- | --- | --- |
| `mha2sha` | `true` | Multi-head to single-head. Start points are read from the SDK's own family builder, never copied here. |
| `mha2sha_validate` | `false` | Run the SDK's own transform validation. Slow; a debugging aid. |
| `permute_kv_cache_io` | `false` | Permute KV-cache IO independently of `sequence.native_kv`. |
| `family_options` | `{}` | Family-specific passthrough (`m2s_head_split_map`, `adapt_moe`). |

## `quantization` (`QuantizationSpec`)

| Field | Default | What it is |
| --- | --- | --- |
| `mode` | `apply_encodings` | Production input is always AIMET `apply_encodings`. `calibrate` drives the standalone quantizer, which is **deprecated for this program** and survives only as a debugging comparison. |
| `act_precision` | `16` | Activation bitwidth. |
| `bias_precision` | `32` | Bias bitwidth. |
| `weights_precision` | `8` | Weight bitwidth. |
| `act_calibration_method` | `min-max` | Standalone quantizer only. |
| `param_calibration_method` | `min-max` | Standalone quantizer only. |

## `vectors` (`VectorSpec`)

| Field | Default | What it is |
| --- | --- | --- |
| `mode` | `capture` | `provided` consumes supplied manifests; `capture` records a reference during the run. |
| `validation_manifest` | `null` | One immutable manifest for a single-AR workflow. |
| `validation_manifests_by_ar` | `{}` | One manifest per AR. Qwen3.5 and Omni Thinker require this; low-level multi-AR runs bind each AR to its own entry and fail closed when one is missing. |
| `calibration_manifest` | `null` | Only when the standalone quantizer calibrates. |

Golden vectors arrive as a trusted local pickle and must be imported into an
immutable manifest first — see the `qairt-import-vectors` skill.

## `compile` (`CompileSpec`)

| Field | Default | What it is |
| --- | --- | --- |
| `enable_intermediate_outputs` | `false` | Build separate **diagnostic** contexts. Production contexts stay free of diagnostic outputs. |
| `output_tensors` | `()` | Select specific output tensors for a diagnostic context. |
| `compiler_options` | `{}` | Passed to QAIRT's compile config. |

`quality.dump_intermediates_on_failure` also turns diagnostic contexts on in the
effective build; `qairt-agent plan` shows the result under `effective_compile`.

## `target` (`TargetSpec`)

| Field | Default | What it is |
| --- | --- | --- |
| `backend` | `HTP` | The only backend. |
| `name` | active harness target | A registered entry name. |
| `chipset` / `dsp_arch` / `soc_model` | `""` / `""` / `0` | Supply all three or none: a partial tuple is never completed implicitly, and a complete one is accepted only on an exact registry match. `soc_model` is the `Qnn_SocModel_t` value, **not** the Android `soc_id`. |
| `platform` | `android` | |
| `device_id` | `null` | Recorded only; the device comes from `QAIRT_AGENT_ADB_SERIAL`. |
| `qairt_version` / `qairt_build_id` | pinned | Must match the harness; a mismatch fails closed. |

## `quality` (`QualitySpec`)

| Field | Default | What it is |
| --- | --- | --- |
| `sqnr_modes` | `()` | Executable policy, not report metadata: exactly `full_reference`, `teacher_forced`, `chain` in any subset. Empty means the stage's default. |
| `dump_intermediates_on_failure` | `true` | Build diagnostic contexts so a failing validation can claim operator-level evidence; without them the report degrades explicitly and sets `op_level_dump_available=false`. |

## `benchmark` (`BenchmarkSpec`)

| Field | Default | What it is |
| --- | --- | --- |
| `warmup_runs` | `10` | Low-level lane. The GenAI lane resolves `3` at spec-parse time; an explicit value always wins. |
| `measured_runs` | `50` | Low-level lane. The GenAI lane resolves `10`. |
| `optrace` | `false` | Publish per-op evidence. Multi-AR GenAI optrace fails closed. |

## `stage_configs` (`WorkflowStageConfigs`)

| Field | Default | What it is |
| --- | --- | --- |
| `build` | `{}` | Build continuation inputs. |
| `validation` | `{}` | Validation continuation inputs (accepts `validate` as an alias; supplying both is refused). |
| `benchmark` | `{}` | Benchmark continuation inputs. |
| `diagnose` | `{kind: "quality", config: {}}` | `DiagnoseStageConfig`. |

`DiagnoseStageConfig`:

| Field | Default | What it is |
| --- | --- | --- |
| `kind` | `quality` | `quality` or `latency`. The named path runs and fails closed when it has no evidence. |
| `config` | `{}` | `baseline_manifest` and `kind` keep the stage in automatic mode; anything else switches it to explicit-trace mode. |

Keys used inside `stage_configs.validation`:

- `ar` — a single-AR debug override. Without it (and without custom
  graph/routes/outputs) validation executes **every** AR the spec requests.
- `component` — Qwen3-VL only: `text` or `vision`. An unscoped VL run fails.
- `slice_vector_manifests` — per-slice boundary vectors for `teacher_forced`.
- `float_reference` — the debug-only ONNX Runtime float reference:
  `granularity` (`slice_boundary` or `layer`), an explicit single `ar`, and an
  optional `tensor_map`. Layer granularity needs an executed, hash-verified
  diagnostic context for every slice in scope.

Keys used inside `stage_configs.benchmark`:

- `ar`, `component` — as above.
- `prompt` / `prompt_path` — the GenAI generation workload.
- `token_count` — makes `p50_ms_per_token` publishable with
  `ms_per_token_source: "caller"`. It is never derived from a rate times a
  duration.
- `aa_calibration` — doubles the sample counts to measure the harness against
  itself.

## Metadata keys the pipeline reads

| Key | Used by | What it is |
| --- | --- | --- |
| `attached_models_by_ar` | Qwen3.5 / Omni Thinker | `{ar: {model_path, encodings_path}}` for AR1 and AR128. Required; fail-closed. |
| `model_config` | any | An inline Hugging Face config instead of `sources.text.config_path`. |
| `model_config_path` | any | A config file path, when `sources.text.config_path` is not used. |
| `qwen35_runtime_validation` | Qwen3.5 multi-AR | Per-graph/AR golden cases for the derivation check. |

## Choosing AR, CL, weight sharing and native KV

A wide export (for example AR2073 / CL4096) is converted by the low-level lane.
These are decisions, not defaults to inherit silently — the `qairt-author-spec`
skill walks them with the user before a spec is written. Each is enforced at
spec time, so a wrong answer fails in seconds rather than hours into a build.

**AR set — use `{1, 128}`.** Weight sharing packages exactly that set per
semantic slice, so a third AR does not get shared weights and a different pair
is not what the program's evidence covers. AR1 is the decode step, AR128 the
prefill chunk. `sequence.ars` is validated against `context_lengths` at
spec time: `max(ars) > context_length` fails
(`pipeline._generate_family_config`).

**Weight sharing — on by default, and it constrains the AR set.** With
`sequence.weight_sharing=true` the AR set must be **exactly** `{1, 128}`: not a
subset, not a superset (`contracts.SequenceSpec.validate_ar_context_product`).
One context is produced per semantic slice with both ARs inside, sharing one
weight copy; with it off, one context per (slice, AR). Qwen3.5 and Omni
**require** it for any multi-AR build, and standalone ViT **forbids** it along
with native KV. Adding a third AR therefore means turning weight sharing off —
there is no partial form. At compile time this becomes QAIRT's own
`CompileConfig.set_mode("weight_sharing", ...)` with every converted graph handed
to `qairt.compile` together.

It has unit-test coverage against the SDK fake but **no device acceptance run**:
the smoke fixture is single-AR with weight sharing off, and the config cell that
enables it needs a real export ([T21](plan/T21-real-model-acceptance.md)). Do
not describe it as hardware-proven.

**CL — one per workflow, divisible by 256 under native KV.** The 256 alignment
is enforced in `contracts.SequenceSpec`. One CL per workflow is not a hard
constraint but a practical one: validation vectors bind to a CL automatically
only when there is a single one, and a second CL doubles the build without
doubling what the reports can say. 4096 is the program's setting; 8192 flows
through the same path.

**Native KV — on, unless you can say why not.** It is the HMX-native layout the
target expects, and QAIRT's own `gen_kv_format_config` selects the tensors (with
one documented subtraction: names whose role proves they are not caches are
removed, and what was removed is reported). Turn it off only when the export's
KV tensor names, routing, shapes or CL alignment cannot be preserved — those are
exactly what the native-KV audit checks, and it fails closed rather than
compiling something that will not bind. A graph's outputs are marked only when
its AR is a positive multiple of 32, which is QAIRT's rule, not ours: AR1 graphs
are inputs-only by construction.

**Before building, read the plan.** `qairt-agent plan --spec spec.json` must
show the expected `pipeline`, AR policy, native-KV policy, `effective_target`
(and that it is verified), `effective_compile`, and `effective_benchmark`. A
surprise there is cheaper than a surprise after a multi-hour build.


## Adding a model family

Family identity is declared **once**, in `src/qairt_agent/family_registry.py`.
Everything else derives its own view from those records: `ModelFamily` alias
resolution and the family-to-preset map in `contracts`, the preset-to-family map
in `families/presets`, the profile alias sets in `families/profiles`, and the
vector-retarget policy table. `tests/test_family_registry.py` fails if an alias
spelling reappears in any of those derived views.

1. **Add one `FamilyRecord`.** Give it a `key`, the `model_family` build lane it
   uses, its `profile_id` (`None` when it has no decoder — standalone ViT), its
   `preset_ids`, a `canonical_name`, every `alias` spelling, and the
   retarget/KV/AR policy. List both hyphen and dot spellings: the consumers
   normalize differently, one folding `_` to `-` and the other stripping every
   separator. Two records may share a `model_family` — Omni Thinker builds
   through the Qwen3.5 lane but is its own family for routing.
2. **Add the `FamilyPreset`** in `families/presets.py` if the family is
   launchable: pipeline, output layout, and default policy.
3. **Add the `FamilyProfile`** in `families/profiles.py` if it has a decoder:
   architecture names, model types, AR policy, and — if the SDK knows its
   MHA2SHA start points — an `SdkStartPointSource` rather than a copy. Per-model
   knowledge is read from the SDK at build time; the only thing reproduced
   locally is the `split_llm` layer distribution, and it is marked `advisory`.
4. **What must come from the SDK, not from here:** MHA2SHA start points and the
   native-KV/HMX tensor selection. Both have a reviewed fingerprint or a live
   read; a hand-maintained copy goes stale silently and two of the three we once
   had produced real bugs.
5. **The tests that must exist:** routing enforcement (the family cannot be
   built through the wrong lane), preset resolution, retarget policy, and — if
   the family has a decoder profile — the preset↔config cross-check. Use
   `test_a_synthetic_family_becomes_visible_everywhere_from_one_record` in
   `tests/test_family_registry.py` as the template: it declares one record and
   asserts every view picks it up with no further edits.
