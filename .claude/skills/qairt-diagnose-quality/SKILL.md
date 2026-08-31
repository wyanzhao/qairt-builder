---
name: qairt-diagnose-quality
description: Investigate a SQNR/accuracy regression in a QAIRT build by bisecting component, slice, layer, tensor and operator, including the debug-only ONNX Runtime float reference and layer-level drilldown. Use when SQNR is lower than expected, when device output diverges from the golden, or when someone asks which layer broke.
---

# Diagnose a quality regression

## Order of evidence

Bisect component → slice → layer → tensor → operator. Do not skip to operator
level; each step narrows what the next one has to explain.

1. **Supplied goldens are the production reference.** ONNX Runtime is a
   *fallback* only when a manifest has executable inputs but no goldens, and
   that fallback is recorded in the report. ORT never replaces supplied
   goldens.
2. **`quality.sqnr_modes` is executable policy, not metadata.** `full_reference`
   compares the final output; `teacher_forced` feeds every slice its own golden
   boundary; `chain` feeds device output forward so local and propagated error
   stay distinguishable. Device boundary outputs are never accepted as teacher
   inputs.
3. **Slice boundaries vs the float graph** — `stage_configs.validation.float_reference`
   with `granularity: "slice_boundary"`. Debug-only, single-AR, published beside
   the golden comparison and never instead of it.
4. **Layer drilldown** — same config with `granularity: "layer"`. This executes
   the diagnostic contexts and compares every tapped tensor, ordered by the
   float graph's topology.

## Compare before you drill

Nonzero noise is the steady state of a quantized model, not a regression. To
know whether quality actually moved, compare against a baseline run:

```bash
qairt-agent compare --from-job BASELINE_JOB --to-job CANDIDATE_JOB
```

`quality.by_tap` lists per-tap SQNR/RMSE/cosine deltas with the worst movers
first, and `quality.worst_mover` is where to start bisecting. Non-comparable
pairs are refused by name rather than differenced. `qairt-agent diagnose
--from-job CANDIDATE --baseline BASELINE` runs the same comparison and reports
which path the measured change implicates before drilling down.

## Requirements that fail closed

- Layer granularity needs a build that emitted diagnostic contexts: set
  `quality.dump_intermediates_on_failure` or `compile.enable_intermediate_outputs`
  and rebuild. It will not degrade to slice boundaries under a layer label.
- Names are **bound, never guessed**: exact match or an explicit
  `float_reference.tensor_map` entry. Anything else lands in `unmapped_tensors`.
  Map names from the transform lineage of the exact export, not by eye.
- Several diagnostic contexts with no routes fails closed — each slice must be
  fed what the previous slice produced, not a guess.
- A chain stitched from two independent builds cannot be drilled down: a
  diagnostic context belongs to the build that produced it, so one slice has
  none. The stage names the slices instead of failing deep in the runner. Only
  the single-slice layer path has run on hardware so far; say so when reporting
  a multi-slice drilldown result.

## Reading the report without over-concluding

`claim_scope: first_observed_divergence_not_root_cause` is load-bearing. An
intermediate can read far worse than the tensors on either side of it when a
downstream operator discards the range the error lives in. On the smoke fixture
`h1` reads ~2.5 dB while `h0` and the output read ~38–40 dB, because `h1` feeds
a Relu that clamps the negative half where quantization error dominates.

**The first bad row is where to start looking, not the answer.** Say so when
reporting; do not tell the user a layer is broken because it has the worst SQNR.

`device_tensor_source` distinguishes a diagnostic-context tap from a production
context boundary when both produced the same tensor name.

## Never

- Claim operator-level evidence unless diagnostic contexts were hash-verified
  *and executed*; otherwise the report degrades to slice/tensor evidence and
  sets `op_level_dump_available=false`.
- Put diagnostic outputs in a production context.
