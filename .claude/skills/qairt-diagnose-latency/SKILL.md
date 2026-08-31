---
name: qairt-diagnose-latency
description: Investigate QAIRT latency, or explain why a measured number looks far too large for the NPU. Covers the device_execution metric, per-op cycles, and why the host wall-clock number is not latency. Use when someone asks how fast a model runs, reports a suspicious latency, or wants per-operator attribution.
---

# Diagnose latency

## The one thing to get right

**Latency means device time.** A latency report names its metric in
`latency_metric`, which points at `device_execution` or at the string
`unavailable`. Never quote the host wall-clock number as the model's speed.

QAIRT implements a remote call by relaunching `qnn-net-run` on the device, so a
wall sample contains process launch, per-call context load, HVX/HMX power-on,
deinit and the ADB round trip. Measured on SM8750, a 49 KB context took ~4900 ms
of wall time around **79 µs** of accelerator compute — a factor of ~60,000. If a
reported latency looks absurd for the hardware, this is almost always why, and
the answer is to read the right block rather than to re-measure.

The wall number is kept under `harness_diagnostics`, marked `not_latency: true`,
because it still detects ADB, container and transport degradation. `aa_calibration`
and `p50_ms_per_token` live there too, since both derive from it.

## What `device_execution` contains

Read from QAIRT's own profiling log at `level="detailed"`:

- `accelerator_compute_us` — accelerator time excluding wait. **This is what
  this program calls production latency**, republished as
  `production_latency_us`. Host orchestration and the device's queueing/memory
  wait are both outside it. Its small absolute value makes it the most dispersed
  metric here (8-17% CV against ~2% for accelerator execute), so read
  `production_latency_cv_percent` alongside it.
- `accelerator_execute_us`, `qnn_execute_us` — progressively wider scopes.
- `per_op_cycles` — per-operator cycle counts.
- `per_process_overhead_us` — load-binary, power-on, deinit, kept **separate**
  from execute time and never folded into it.
- Ten profiled executes averaged: `statistic: "mean"`, with `spread` and
  `samples`. Check the spread before trusting the mean — on real hardware the
  same fixed work has ranged 26083 to 46467 accelerator cycles.

For chain scope, `by_slice` carries one block per slice and `totals` is an
explicitly labelled sum of per-slice means, not a measured end-to-end number.

## Traps

- **`option="optrace"` is not how per-op cycles are obtained.** It needs a
  schematic binary this program's compile does not emit and fails with "No op
  trace raw data found." The detailed log already carries per-op cycles.
- **Profiling must happen before `initialize_execution`.** An initialized model
  carries an execution context created with profiling disabled, so profiling it
  yields nothing at all. The adapter fails closed on this.
- **QAIRT writes its profiling log to a relative `output/` directory.** The
  worker mounts the project root read-only, so the capture needs a writable
  `working_dir` or the execute fails outright with `ExecutionError`.

## Compare before you drill

A single run's numbers cannot tell you whether anything changed. Start from the
delta:

```bash
qairt-agent compare --from-job BASELINE_JOB --to-job CANDIDATE_JOB
```

It refuses a non-comparable pair by name (preset, family, target, AR set,
context lengths, `sqnr_modes`, latency meter/lane), then reports per-AR
`production_latency_us` deltas both in microseconds and in units of the **pooled
CV**. Read `delta_in_pooled_cv`, not `delta_us` alone: production latency is the
most dispersed metric in the block (8-17% CV against ~2% for accelerator
execute), so a change smaller than its own dispersion is not yet a finding. The
output is report-only and carries no pass/fail verdict; that judgement is yours.

`qairt-agent diagnose --from-job CANDIDATE --baseline BASELINE` runs the same
comparison first and publishes an `implicated` block naming which path the
measured change points at. Use `--kind latency` to run only the latency path;
it now fails closed when there is no optrace evidence instead of quietly
producing a quality report.

## Where a device number is unavailable

`latency_metric: "unavailable"` with a reason in
`device_execution.available/reason`. Today that is GenAI generation: `generate()`
reaches Genie as `GenieDialog_query` rather than `CompiledModel.__call__`, so
`qairt.Profiler` cannot observe it. Genie has its own on-device profiling with
dialog timings and token rates, but it is a **different meter** — never present
it as the same number as accelerator execute time. See
`docs/plan/T11-device-only-latency.md`.

## Never

- Report diagnostic-context latency as production latency.
- Add per-op cycles together and call the sum wall latency.
- Derive `p50_ms_per_token` from a token count the SDK did not report;
  `ms_per_token_source` must be `caller` unless the SDK actually supplies one.
