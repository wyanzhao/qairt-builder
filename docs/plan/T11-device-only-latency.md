# T11 — Make device the only latency metric

Status: planned (opened 2026-08-30)
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
   `device_execution` is currently a single profiled execute; as the primary
   metric it repeats the profiled execute 10 times and publishes the average.
   No new statistics code is needed — `diagnostics.latency.summarize_latency`
   already returns mean alongside p50/p90/stddev/MAD from a sample list, so the
   device samples go through the same summarizer and the mean is the headline
   with the rest available beside it. A/A calibration moves onto the device
   numbers or is dropped there; it was calibrating host noise, which is no
   longer the metric.
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
