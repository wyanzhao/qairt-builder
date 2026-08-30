# T10 — Latency measurement is host-harness time, not device time

Status: done (2026-08-30)
Depends on: —
Effort: M

## The defect

Every latency number this program has published is dominated by per-call host
and process overhead, not by NPU execution. On SM8750, one benchmark sample of
the tiny acceptance graph (a 49,824-byte context binary) measures **4903 ms**
while QAIRT's own device-side profiler reports **77 µs** of accelerator compute
for the same graph — roughly a 60,000x gap.

Two published claims are therefore wrong as written:

- `latency_report.setup_excluded = true` is **false**. Per-call setup happens
  *inside* the timed call, so it cannot have been excluded by timing around it.
- `measurement_scope.includes = "host_to_sdk_to_device_round_trip"` is literally
  true but reads as a small IPC tax. It is the dominant term.

## Evidence

Measured on the SM8750 handset (serial RFCY30B296K, `ro.soc.model SM8750`),
context `ca6fa637…` — the same verified artifact as the sm8750 acceptance run.

**1. QAIRT executes a fresh `qnn-net-run` process per call.** While a benchmark
runs, a `tmpXXXXXXXX/` directory holding `input_list.txt` plus an
`input_NN.raw` appears and is deleted on the device once per invocation. The
persistent staging directory alongside it contains `qnn-net-run`,
`libQnnHtp.so`, `libQnnHtpV79Skel.so`, `libQnnHtpV79Stub.so` and the context
binary (97 MB). The profiling log names its producer: `qnn-net-run
v2.49.0.260730134355`.

**2. The device's own accounting** (`qairt.Profiler(context={"level":
"detailed"})`, one execute):

| QAIRT profiling event | value |
| --- | --- |
| Accelerator (execute excluding wait) time | **77 µs** |
| Accelerator (execute) time | 1 734 µs |
| QNN accelerator (execute) time | 2 381 µs |
| QNN (execute) time | 3 001 µs |
| RPC (execute) time | 2 886 µs |
| Time for HVX + HMX power on and acquire | 29 218 µs |
| Time for initial VTCM acquire | 804 µs |
| QNN (load binary) time | 9 009 µs |
| QNN (deinit) time | 19 359 µs |

Per-op cycles are also present: `Input OpId_2` 8405, `fc:OpId_17` 6137,
`bias_add:OpId_21` 0, `act:OpId_23` 8567, `Output OpId_3` 6130.

Even the most inclusive on-device figure — load binary + power-on + execute +
deinit — is about 60 ms. The remaining ~4.8 s is host-side: ADB push/pull, an
`adb shell` process launch per call, the container-to-host ADB hop through
`host.container.internal`, and Rosetta emulation of the linux/amd64 worker on
Apple silicon. None of it is a property of the model or the NPU.

**3. `CompiledModel.initialize()` is never called.** QAIRT's docstring: *"should
be used if you intend to call execute multiple times with the same model,
backend, and device."* Without it, `CompiledModel._execute` re-enters
`_create_execution_context` on every call and `NetRunnerModule.run` takes its
`InferenceIdentifier` branch, which creates a backend and an `mlapi.Inferencer`
per call and unloads at the end of the function. Measured on the same device,
same context, one process:

| path | p50 |
| --- | --- |
| no `initialize()` (today) | 3990 ms |
| `initialize()` once, then call (50 samples) | 2492 ms |

Outputs are bit-identical between the two paths (max abs diff 0.0). So
`initialize()` is a real 38% improvement and the API's documented usage — but it
is **not** the fix: 2492 ms is still ~32,000x the accelerator compute, because
`qnn-net-run` is still relaunched per call.

## What to change

1. **Stop claiming `setup_excluded: true`.** Rename the wall metric to what it
   is (host-orchestrated end-to-end call latency) and state in
   `measurement_scope` that per-call process launch, context load, HVX/HMX
   power-on and deinit are inside the sample.
2. **Publish the device-side numbers as the NPU metric.** `level="detailed"`
   parses straight from the profiling log with no schematic binary, and yields
   accelerator execute time, QNN execute time, and per-op cycles. This is the
   number that belongs in a report next to SQNR.
3. **Call `initialize()` before the timer** for the low-level lane, mirroring
   what the GenAI lane already does with `prepare_environment()` /
   `clean_environment()`, and pair it with `destroy()` in a `finally`.
4. **Decide what production latency means for this program.** A per-call
   `qnn-net-run` launch can never be a production number. Either the reported
   metric becomes device-side execute time, or a genuinely persistent execution
   path is required.

## Landmine found while probing

`initialize()` outside a `Profiler` scope **silently disables profiling**: the
cached execution context is built with profiling off, and `generate_reports()`
then dies with `Profiling data: None is not valid`. Any fix that adds
`initialize()` must initialize inside the profiler scope, or skip it for
profiling runs. A test must cover this, because the failure appears only on the
profiling path.

Separately, `option="optrace"` requires `backend_profiling_artifacts` (a
schematic binary) that our compile does not currently emit, so the optrace path
fails with `No op trace raw data found.` on these contexts. `level="detailed"`
without the optrace option is unaffected. CLAUDE.md's claim that "per-op
attribution comes from optrace" needs revisiting: per-op cycles are available
from the detailed log without optrace.

## Acceptance criteria

- No report claims setup is excluded when it is not.
- A device-side execute metric (accelerator/QNN execute time, per-op cycles) is
  published from the profiling log, with its own provenance.
- `initialize()`/`destroy()` bracket low-level benchmark execution, with the
  profiler interaction covered by a test.
- The optrace claim in root `CLAUDE.md` is corrected or the schematic-binary
  requirement is documented.
- One real-device run on a registered target publishes both metrics.

## Result (2026-08-30)

Landed across `diagnostics/device_metrics.py` (new), `qairt_adapter/adapter.py`
(`initialize_execution`, `release_execution`, `capture_device_execution`) and
the benchmark stage in `pipeline.py`.

**The report no longer lies about scope.** `setup_excluded` is gone. In its
place `harness_setup_excluded` states what we do control, and
`sdk_per_call_setup_included` states what we do not; `measurement_scope` lists
`excluded_from_timer` and `included_in_sample` by name, and the wall metric is
labelled `host_orchestrated_call_latency`.

**The device metric is published.** Every low-level benchmark now carries a
`device_execution` block read from QAIRT's detailed profiling log — accelerator
compute, accelerator execute, QNN execute, per-op cycles, and per-process
overhead kept separate from execute time. It is report-only enrichment: an
adapter that cannot profile still produces a valid benchmark, and the block
records `available: false` with a reason rather than inventing a number.

**Acceptance run** on SM8750 (serial RFCY30B296K), job
`20260830T113624Z-d356efb4`, output root
`artifacts/sm8750-t10-device-latency-v3`:

| metric | value |
| --- | --- |
| wall p50 (`host_orchestrated_call_latency`) | 2411 ms |
| accelerator compute | 79 us |
| accelerator execute | 1746 us |
| QNN execute | 2965 us |
| accelerator execute cycles | 30208 |

Per-op cycles: `Input OpId_2` 8262, `fc:OpId_17` 6227, `bias_add:OpId_21` 0,
`act:OpId_23` 9271, `Output OpId_3` 6448. Per-process overhead is reported
separately, including `QNN (load binary) time` 10341 us and `QNN (deinit) time`
19719 us. Producer: `qnn-net-run v2.49.0.260730134355`.

Wall time is 813x the QNN execute time and 30525x the accelerator compute. That
ratio is the point of the change: it is now visible in the report instead of
being hidden behind a false `setup_excluded`.

**`initialize()` landed too**, and the effect is reproducible on hardware: the
same spec measured 4974 ms p50 before this change and 2411 ms after, matching
the 3990 -> 2492 ms seen in the isolated probe.

**One implementation trap beyond the profiler one.** QAIRT writes the profiling
log to a *relative* `output/` directory, and the worker mounts the project root
read-only at the process cwd, so the profiled execute failed outright with
`ExecutionError: Failed to execute model` until the capture was given a
writable `working_dir` (the stage's own attempt directory). This is the same
read-only-cwd failure class already known from the standalone quantizer's
calibrate path. The cwd is restored in a `finally`, and a test covers the
raising case.

Tests: `tests/test_device_metrics.py` (8, against a report captured verbatim
from the device), plus `test_capture_device_execution_reads_qairt_own_device_numbers`,
`test_capture_device_execution_refuses_an_initialized_model`,
`test_capture_device_execution_restores_the_working_directory`,
`test_initialize_execution_establishes_and_releases_the_context`,
`test_initialize_execution_fails_closed_without_the_sdk_method`,
`test_latency_report_publishes_device_side_execution_beside_wall_time`,
`test_latency_report_no_longer_claims_per_call_setup_is_excluded`,
`test_benchmark_captures_device_metrics_before_initializing_and_releases_after`,
and `test_benchmark_survives_an_adapter_that_cannot_profile`.

## Still open

- **The GenAI lane is unmeasured.** `sdk_per_call_setup_included` is
  `"unverified"` there rather than a guess: the executor's per-call behaviour
  has not been probed, and `generate()` is a different path from `run_graph`.
  It needs the same treatment before a GenAI latency number is trusted.
- **Chain-scope runs** get the honest scope block but no `device_execution`;
  capture is wired for the single-graph path only.
- **What "production latency" should mean** for this program is still open. A
  per-call `qnn-net-run` launch is not a deployment path, so neither number
  describes a shipped application; the device block is the one that describes
  the model.
