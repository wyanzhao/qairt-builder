---
name: qairt-author-spec
description: Author a QAIRT WorkflowSpec by asking the user the decisions that cannot be guessed — AR set, weight sharing, context length, native KV, decoder slices, vector source — then writing the spec and checking the resolved plan. Use when someone wants to build a new model, adapt a wide export, or asks "what should I choose for AR/CL/weight sharing?".
---

# Author a spec

The CLI is JSON in, JSON out — there is no wizard, by design. The asking happens
**here**, at the agent layer: you ask, the user decides, you write the spec.

Do not guess these. Each one changes what gets built, and several of them make a
build fail hours later rather than at spec time.

## Ask before writing

Work through these in order. Skip a question only when the preset already
settles it — and say so rather than staying silent.

### 1. Preset — the routing authority

`qwen3_dense`, `qwen3_moe`, `qwen3_vl`, `qwen3_5`, `qwen3_5_omni_thinker`,
`qwen3_5_omni`, `vit`. Never infer it from a filename. If a model config is
supplied its `architectures` are cross-checked against the preset at plan time
and a contradiction fails closed — so a wrong answer here is caught, but only
after the user has supplied the config.

### 2. AR set

**Ask:** "Which ARs do you need — the standard `{1, 128}`, or something else?"

- `1` is the decode step, `128` the prefill chunk. `{1, 128}` is the program's
  set and what all its evidence covers.
- **Standalone ViT is AR1 only** and refuses anything else.
- Every AR must fit every context length (`max(ars) > min(context_lengths)`
  fails at spec time).

**For a wide export (AR2073/CL4096 and the like), also ask:** "Should the
low-level lane convert this to AR1 and AR128, or do you have independent
per-AR exports?" Qwen3.5 and Omni Thinker take independent exports through
`metadata.attached_models_by_ar`; deriving several ARs from one Qwen3.5 export
needs `sequence.qwen35_experimental_auto_ar=true` and is **experimental and
fail-closed**, gated on runtime-validated derivation evidence. Say that when
you offer it.

### 3. Weight sharing — ask explicitly, every time

**Ask:** "Weight sharing packages AR1 and AR128 into one context per semantic
slice. Do you want it on?"

It defaults to **on**, so silence means yes — which is exactly why it is worth
asking rather than letting the default decide for the user.

What the answer commits them to:

| | On | Off |
| --- | --- | --- |
| AR set | must be **exactly** `{1, 128}` | any |
| Contexts | one per semantic slice, both ARs inside | one per (slice, AR) |
| Footprint | one shared weight copy | one per AR |

Hard rules, all enforced at spec time:

- **`weight_sharing=true` requires the AR set to be exactly `{1, 128}`.** Not a
  subset, not a superset. Adding AR64 means turning weight sharing off.
- **Qwen3.5 and Omni multi-AR *require* it.** `weight_sharing=false` with more
  than one AR fails.
- **Standalone ViT forbids it**, along with native KV.

At compile time this becomes QAIRT's own
`CompileConfig.set_mode("weight_sharing", graph_names=..., soc_model=...,
dsp_arch=...)` with both converted graphs handed to `qairt.compile` together.

**Say this when it comes up:** weight sharing has unit-test coverage against the
SDK fake, but **no device acceptance run has exercised it** — the smoke fixture
is single-AR with weight sharing off, and the config cell that turns it on needs
a real model export. Do not describe it as hardware-proven.

### 4. Context length

**Ask:** "What context length? One per workflow, or several?"

- Not pinned to 4096; 4096 is the program's setting.
- **Under native KV, every CL must be divisible by 256** — enforced at spec
  time.
- Prefer **one CL per workflow**: validation vectors bind to a CL automatically
  only when there is a single one, and a second CL doubles the build without
  doubling what the reports can say.

### 5. Native KV

**Ask:** "Native KV layout on? It is the HMX-native layout the target expects."

Default on. Turn it off only when the export's KV tensor names, routing, shapes
or CL alignment cannot be preserved — those are exactly what the native-KV audit
checks, and it fails closed rather than compiling something that will not bind.
The tensor selection is QAIRT's own `gen_kv_format_config`, with one documented
subtraction (names whose role proves they are not caches), and what was removed
is reported. A graph's outputs are marked only when its AR is a positive
multiple of 32 — QAIRT's rule, not ours, so AR1 graphs are inputs-only by
construction.

### 6. Decoder slices and LM head

**Ask:** "How many decoder slices?" Default 4 for the Qwen3 presets, 1 for ViT.
The layer distribution is reproduced locally and marked `advisory`; what a build
verifies is the split count.

### 7. Vectors

**Ask:** "Do you have golden vectors, or should the run capture a reference?"

- `mode: "provided"` with `validation_manifest`, or
  `validation_manifests_by_ar` when there is more than one AR — Qwen3.5 and Omni
  Thinker require the per-AR form, and a low-level multi-AR run fails closed
  when one AR's entry is missing.
- Goldens delivered as a pickle must be imported first — see
  `qairt-import-vectors`.
- Inputs without goldens trigger the audited ONNX Runtime fallback capture,
  which is recorded as a fallback. Supplied AIMET goldens are the decided
  production reference.

### 8. Target

**Ask** only if it is not the active harness target. Name it, or supply the
complete `chipset`/`dsp_arch`/`soc_model` tuple — a partial tuple is never
completed implicitly, and a complete one is accepted only on an exact registry
match. An unverified target plans but refuses device stages.

## Then write it, then check the plan

```bash
qairt-agent plan --spec spec.json
```

Read these back to the user before any build starts:

| Field | What to confirm |
| --- | --- |
| `resolved.pipeline` | `low_level` or `genai_builder` — the lane the preset chose |
| `resolved` AR / native-KV policy | matches what they asked for |
| `effective_target` | the right chip, and `verified` is truthy |
| `effective_compile` | whether diagnostic contexts are on |
| `effective_benchmark` | the sampling that will actually run |

A surprise here costs seconds. The same surprise after a multi-hour build costs
the build.

## Never

- Never let a default stand in for an answer on AR set or weight sharing without
  saying which default you applied.
- Never claim weight sharing, or any capability, is hardware-verified without a
  reopenable report from a run that exercised it.
- Never write `weight_sharing: true` beside an AR set that is not exactly
  `{1, 128}` — it fails at spec time, and the failure is easier to avoid than to
  read.

## Related

- [`docs/spec-reference.md`](../../../docs/spec-reference.md) — every field, and
  where each constraint is enforced.
- `qairt-import-vectors` — turning a delivered pickle into a manifest.
- `qairt-first-run` — the end-to-end path, with a fixture that needs no model.
