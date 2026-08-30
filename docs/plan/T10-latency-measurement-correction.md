# T10 — Latency measurement is host-harness time, not device time

Status: planned (opened 2026-08-30)
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
