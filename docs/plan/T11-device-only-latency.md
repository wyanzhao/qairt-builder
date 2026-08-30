# T11 — Make device the only latency metric

Status: low-level half done (2026-08-30); GenAI half blocked on T08
Depends on: [T10](T10-latency-measurement-correction.md)
Effort: M

## Decision

Latency means device time. The host-orchestrated wall number that
[T10](T10-latency-measurement-correction.md) renamed and made honest is not a
latency metric for this program; it stays only as harness diagnostics, because
it still detects ADB/container/transport degradation.

## GenAI capability probe (2026-08-30)

Read from QAIRT 2.49.0.260730's own sources; no hardware run, because a GenAI
container needs the Qwen3.5 sources that [T08](T08-aimet-vector-import.md) is
still blocked on. Every claim below is a source-level fact, not a measurement.

**`qairt.Profiler` cannot see the GenAI lane.** `T2TExecutor.generate()` calls
`self._runner.query(GenieQueryConfig(...))`, which reaches Genie as
`GenieDialog_query` — not `CompiledModel.__call__`. The profiler hooks the
net-runner execute path, so it observes nothing. This is the same root cause as
the existing multi-AR GenAI optrace fail-closed, and it means
`capture_device_execution` cannot be pointed at `generate()`.

**Genie has its own on-device profiling.** It writes `profile.json` on the
device; the run module pulls it, deletes the remote copy, and parses it into a
`GenieProfileRecord`:

| event | fields |
| --- | --- |
| `GenieDialog_create` | `init-time` |
| `GenieDialog_query` | `num-prompt-tokens`, `prompt-processing-rate`, `time-to-first-token`, `num-generated-tokens`, `token-generation-rate`, `token-generation-time`, `lora-adapter-switching-time` |
| every event | `duration`, `start`, `stop` |

These are measured on the device. They are **a different meter** from the
low-level lane's `Accelerator (execute) time`: Genie reports dialog-level
timings and token rates, with no accelerator time and no per-operator cycles.
A report must never present the two as the same number.

**The raw record is discarded before any public API.**
`parse_genie_profile_record` maps it onto `GenerationMetrics`, which has fields
for `init_time`, `prompt_processing_time/rate`, `token_generation_time/rate`,
`time_to_first_token`, `token_acceptance_rate` and `adapter_switch_time` — and
**no token-count field**. `TextGenerationResult` exposes only `output`,
`error` and `metrics`, never `profile_record`. All three runner paths
(`genie_t2t_run_module`, `genie_app_run_module`, `native_t2t_module`) convert
and drop it identically.

This sharpens a claim in root `CLAUDE.md`. Saying "QAIRT 2.49 does not report a
generated-token count" is right about the public surface but wrong about where
the number goes: **the device measures `num-generated-tokens` and the SDK
throws it away in parsing.** The conclusion for us is unchanged — a count still
cannot be obtained through the public API, so `ms_per_token_source` stays
`caller` — but the reason should be stated correctly, because it is the kind of
thing a future SDK build may simply stop doing.

## Design

1. **Two named meters, never conflated.** Keep `device_execution` for the
   low-level lane (accelerator/QNN execute time, per-op cycles) and add
   `genie_execution` for the GenAI lane (on-device Genie dialog events). Each
   carries its own `source` and `sample_unit`; nothing computes a ratio across
   them or presents one as the other.
2. **Publish Genie's measured rate, do not derive it.** `token-generation-rate`
   and `prompt-processing-rate` are device-measured tokens/second. Publishing
   them directly avoids the count problem entirely — a derived `ms_per_token`
   is what needs a count, a reported rate does not. `time_to_first_token` is
   the other headline.
3. **Ten samples, report the mean** (maintainer decision, 2026-08-30).
   The profiled execute repeats 10 times and the block publishes the average.
   *Correction to the original plan:* this does **not** reuse
   `summarize_latency`. That summarizer names every field `*_ms`, and putting
   microseconds or cycle counts behind a millisecond label is precisely the
   class of mislabelling this metric exists to correct, so
   `aggregate_device_executions` does its own small aggregation. A/A
   calibration stays with the wall number, which is what it was calibrating.
4. **Cover the remaining scopes.** Chain-scope runs get no device capture
   today; the capture is wired only for the single-graph path.
5. **Demote the wall metric** to a `harness_diagnostics` block: still recorded,
   never called latency, never the basis of a regression verdict.
6. **Per-op attribution on the GenAI lane stays unavailable** through
   `generate()`. The documented route is the explicit single-`ar` raw
   `CompiledModel` profiling run, which `capture_device_execution` already
   supports — a separate, opt-in measurement that must never be presented as
   coverage of the whole container.

## Out of scope

- Reading Genie's `profile_record` through private attributes. It is not a
  public QAIRT API, and the program's boundary is public Python APIs only.
- Power/thermal measurement (program out-of-scope decision).

## Acceptance criteria

- No report calls a host wall number "latency".
- Low-level lane publishes `device_execution` averaged over 10 profiled
  executes.
- GenAI lane publishes `genie_execution` with the device-measured rates and
  time-to-first-token, labelled as its own meter.
- Chain scope is covered or explicitly fails closed.
- The `CLAUDE.md` token-count claim is restated precisely.
- One real-device run per lane. The GenAI half is blocked on
  [T08](T08-aimet-vector-import.md).

## Low-level half — landed (2026-08-30)

`aggregate_device_executions` averages N parsed executes; the benchmark stage
takes `DEVICE_EXECUTION_SAMPLES = 10`. The payload was restructured: the wall
number moved under `harness_diagnostics` with `not_latency: true`, carrying
`measurement`, `measurement_scope`, `aa_calibration` and `p50_ms_per_token`
(all wall-derived), and a top-level `latency_metric` names the block that is
the latency. A scope with no device meter publishes
`device_execution.available = false` **with a reason** — GenAI generation and
chain scope each state theirs — so an absent device number is a declared gap.

**Acceptance on SM8750**, job `20260830T203637Z-2fc25926`, 10 profiled
executes:

| metric | mean | p50 | min | max | stddev |
| --- | --- | --- | --- | --- | --- |
| accelerator compute (us) | 76.8 | 74.0 | 67 | 111 | 12.8 |
| accelerator execute (us) | 1730.8 | 1720.5 | 1711 | 1814 | 30.5 |
| QNN execute (us) | 3043.6 | 2987.5 | 2960 | 3374 | 132.1 |
| accelerator execute (cycles) | 30038 | 28446 | 26083 | 46467 | 5982 |

Per-op cycle means: `Input` 8828.7, `act` 9095.2, `Output` 6217.8, `fc` 5896.3,
`bias_add` 0.0. Wall p50 was 2552.9 ms, published under `harness_diagnostics`.

**The spread vindicates sampling.** Accelerator cycles ranged 26083 to 46467 —
a 78% spread with stddev ~6000 on a graph doing fixed work. A single profiled
execute, which is what T10 shipped, could have reported anything in that band
as *the* number. This is why `spread` and `samples` are published rather than
the mean alone.

### Still open in this task

- GenAI half (`genie_execution`): blocked on [T08](T08-aimet-vector-import.md)
  for a real container.
- Chain scope: declares its gap with a reason rather than measuring; wiring a
  per-slice profiled execute needs the runner's dynamically produced inputs.
