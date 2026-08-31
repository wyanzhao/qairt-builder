# T16 — Measurement report honesty fixes

Status: done (2026-08-30)
Depends on: —
Effort: S

## Goal

Close the three places where a report's label can claim more than its
evidence: the multi-AR aggregate latency metric, chain-sequence device
coverage, and partial-sample aggregation. The default single-graph and
single-AR paths are already honest; this task extends the same discipline to
the aggregate and multi-step paths.

## Context

Review findings 13–15 in
[review-findings-2026-08-30.md](review-findings-2026-08-30.md). The multi-AR
aggregate hardcodes `latency_metric="device_execution"` (`pipeline.py:7321`)
even when a per-AR capture degraded to `available=false`;
`_recording_chain_executors` keeps only the last step's inputs per slice
(`pipeline.py:5052`) so a prefill+decode sequence publishes scope `"chain"`
covering only decode; `aggregate_device_executions` can average partial
samples without a marker, and non-finite device outputs abort validation with
an exception instead of a localizing report.

## Design

1. **Aggregate metric.** Derive the aggregate `latency_metric` from the per-AR
   results: `device_execution` only when every AR carries an available block;
   otherwise `"partial"` with an explicit list of ARs whose meter was
   unavailable and the recorded per-AR reasons. `coverage` already exists —
   extend it rather than inventing a parallel field.
2. **Chain-step provenance.** Record per-step, per-slice inputs (keyed by
   step index and graph/AR) instead of overwriting. Either profile every
   recorded step per slice, or — if measurement cost rules that out — profile
   the recorded steps the caller names and label the block with exact
   `steps_covered` / `steps_total` and per-step AR provenance. A chain block
   must never present last-step-only evidence under an unqualified
   scope="chain" label.
3. **Partial samples.** When `aggregate_device_executions` receives fewer
   samples than requested, the block carries `samples_requested` /
   `samples_used` and a `partial: true` marker; the statistic never silently
   averages a different N than the contract's ten.
4. **Non-finite outputs.** Convert the abort into a failing validation report
   that names the slice/tensor with the non-finite values (report-only
   structural failure stays an error at the stage level — the point is that
   the error payload localizes, not that validation passes).

## Files

`src/qairt_agent/pipeline.py`, `src/qairt_agent/diagnostics/device_metrics.py`,
`src/qairt_agent/contracts.py` (if aggregate fields are typed), tests
(`test_pipeline.py`, `test_device_metrics.py`, `test_chain.py`),
`docs/native-workflow.md` report-field notes, `CLAUDE.md` one-line scope note
if the chain-coverage rule changes the contract text.

## Acceptance criteria

- A multi-AR benchmark where one AR's capture failed publishes an aggregate
  whose metric/coverage names the degraded AR; no unconditional
  `device_execution` label (test).
- A two-step chain sequence publishes device evidence with per-step
  provenance, or an explicit `steps_covered` subset label (test fails on
  pre-fix last-step-only behavior).
- Partial-sample aggregation is marked and carries both counts (test).
- A non-finite device output yields a structured failure naming slice and
  tensor (test).

## Result

Landed 2026-08-30, all four items.

**Aggregate metric.** The multi-AR latency aggregate derives `latency_metric`
from the per-AR reports: `device_execution` only when every AR published an
available `device_execution` block, `partial` otherwise. `coverage` was extended
rather than duplicated — `metered_ars`, `unmetered_ars`, `device_meter_complete`,
and `unmetered_ar_reasons` carrying each AR's recorded reason.

**Chain-step provenance.** `_recording_chain_executors` appends every invocation
(with `step_index`, graph name, and AR) instead of overwriting, and
`_chain_device_execution` profiles every recorded step. One pass keeps the
documented `scope: "chain"` / `by_slice` shape byte for byte; a sequence
publishes `scope: "chain_sequence"` with `by_step`, `steps_covered`, and
`steps_total`, and its total is an explicitly labelled sum over all steps.

**Partial samples.** `aggregate_device_executions` takes `requested` and
publishes `samples_requested`, `samples_used`, `samples_used_by_metric`, and
`partial` (with `partial_reason` / `partial_metrics`), so a mean over three
executes is never presented as the contract's ten.

**Non-finite outputs.** `QualityDiagnoser._compare_localized` wraps each tap
comparison: a NaN/inf tensor still fails the stage, but as an `InvalidSpecError`
naming slice, tensor, which side (`teacher_forced`/`device_chain`), the
non-finite element count, and the element count.

Nine new tests across `test_chain.py`, `test_device_metrics.py`, `test_sqnr.py`,
and `test_pipeline.py` (including a fake whose AR128 capture fails). Suite 597
passed / 2 skipped, compileall clean. `CLAUDE.md` and `docs/native-workflow.md`
updated.
