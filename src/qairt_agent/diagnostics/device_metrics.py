"""Device-side execution metrics parsed from QAIRT's own profiling log.

A latency sample in this program is host wall time around one SDK call, and on
the low-level lane QAIRT implements that call by relaunching ``qnn-net-run`` on
the device.  Per-call context load, HVX/HMX power-on and deinit therefore sit
*inside* the sample, and the host-side ADB round trip dominates it.  On the
tiny acceptance graph a sample measures ~4.9 s while the accelerator reports
77 us of compute.

QAIRT answers the real question itself.  ``qairt.Profiler(context={"level":
"detailed"})`` writes a profiling log that the SDK parses into named events
with device-side values, including per-operator cycle counts.  This module
turns that report into the block a latency report publishes.

``option="optrace"`` is deliberately not used here: it additionally requires a
schematic binary in ``backend_profiling_artifacts`` which our compile does not
emit, and it fails with "No op trace raw data found." without one.  The
detailed level needs no such asset and already carries per-op cycles.
"""

from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence

DEVICE_EXECUTION_SCHEMA = "qairt-agent.device-execution/2"

# The meter this block reports. The GenAI lane cannot use it: generate() reaches
# Genie as GenieDialog_query rather than CompiledModel.__call__, so the profiler
# observes nothing there and that lane needs its own, differently named block.
DEVICE_EXECUTION_METER = "qnn_accelerator"

# The metric this program calls production latency (maintainer decision).
PRODUCTION_LATENCY_SOURCE = "accelerator_compute_us"

# Headline scalars averaged across samples.
_AGGREGATED_SCALARS = (
    "accelerator_compute_us",
    "accelerator_execute_us",
    "qnn_execute_us",
    "accelerator_execute_cycles",
)

_EXECUTE_METHOD = "BACKEND_EXECUTE"
_INIT_METHOD = "BACKEND_CREATE_FROM_BINARY"
_DEINIT_METHOD = "BACKEND_DEINIT"

# The accelerator's own compute time, with waiting excluded: the closest thing
# QAIRT reports to "what the NPU spent on this graph".
COMPUTE_IDENTIFIER = "Accelerator (execute excluding wait) time"
ACCELERATOR_IDENTIFIER = "Accelerator (execute) time"
QNN_EXECUTE_IDENTIFIER = "QNN (execute) time"
CYCLES_IDENTIFIER = "Accelerator (execute) time (cycles)"

_MICROSECOND_UNIT = "MICROSEC"
_CYCLE_UNIT = "CYCLES"
_NODE_TYPE = "NODE"


class DeviceMetricsError(RuntimeError):
    """Raised when a profiling report cannot yield device-side execute values."""


def _messages(report_data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = report_data.get("messages")
    if not isinstance(raw, Sequence):
        raise DeviceMetricsError(
            "profiling report has no 'messages' list; it is not a QAIRT "
            "detailed profiling report"
        )
    return [item for item in raw if isinstance(item, Mapping)]


def _events(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = message.get("profilingEvents")
    if not isinstance(raw, Sequence):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _microseconds(events: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for event in events:
        if str(event.get("unit")) != _MICROSECOND_UNIT:
            continue
        identifier = str(event.get("identifier", ""))
        value = event.get("value")
        if identifier and isinstance(value, (int, float)):
            values[identifier] = float(value)
    return values


def _per_op_cycles(events: Sequence[Mapping[str, Any]]) -> tuple[
    list[dict[str, Any]], float | None
]:
    for event in events:
        if str(event.get("identifier", "")) != CYCLES_IDENTIFIER:
            continue
        total = event.get("value")
        operators: list[dict[str, Any]] = []
        for sub_event in event.get("sub-events", ()) or ():
            if not isinstance(sub_event, Mapping):
                continue
            if str(sub_event.get("unit")) != _CYCLE_UNIT:
                continue
            if str(sub_event.get("type")) != _NODE_TYPE:
                continue
            value = sub_event.get("value")
            operators.append(
                {
                    "identifier": str(sub_event.get("identifier", "")),
                    "cycles": float(value) if isinstance(value, (int, float)) else None,
                }
            )
        return operators, (float(total) if isinstance(total, (int, float)) else None)
    return [], None


def parse_device_execution(report_data: Mapping[str, Any]) -> dict[str, Any]:
    """Turn one QAIRT detailed profiling report into a publishable block.

    Fails closed rather than publishing a partial block: a report without a
    ``BACKEND_EXECUTE`` message, or without the accelerator's execute time, is
    not evidence of device-side latency.
    """

    if not isinstance(report_data, Mapping):
        raise DeviceMetricsError("profiling report data must be a mapping")

    execute_message: Mapping[str, Any] | None = None
    init_events: list[Mapping[str, Any]] = []
    deinit_events: list[Mapping[str, Any]] = []
    for message in _messages(report_data):
        method = str(message.get("method", ""))
        if method == _EXECUTE_METHOD and execute_message is None:
            execute_message = message
        elif method == _INIT_METHOD:
            init_events.extend(_events(message))
        elif method == _DEINIT_METHOD:
            deinit_events.extend(_events(message))

    if execute_message is None:
        raise DeviceMetricsError(
            f"profiling report contains no {_EXECUTE_METHOD} message; "
            "device-side execute time cannot be claimed"
        )

    execute_events = _events(execute_message)
    execute_us = _microseconds(execute_events)
    if ACCELERATOR_IDENTIFIER not in execute_us:
        raise DeviceMetricsError(
            f"profiling report has no {ACCELERATOR_IDENTIFIER!r}; the backend "
            "reported no device-side execute time"
        )

    operators, total_cycles = _per_op_cycles(execute_events)
    metadata = report_data.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}

    block: dict[str, Any] = {
        "schema": DEVICE_EXECUTION_SCHEMA,
        "policy": "report_only",
        "source": "qairt_profiler_detailed_log",
        "profiler_level": "detailed",
        "profiler_option": None,
        "sample_unit": "one_profiled_graph_execute",
        "accelerator_execute_us": execute_us[ACCELERATOR_IDENTIFIER],
        "execute_events_us": execute_us,
        "per_op_cycles": operators,
        "accelerator_execute_cycles": total_cycles,
        "per_process_overhead_us": {
            **_microseconds(init_events),
            **_microseconds(deinit_events),
        },
        "producer": {
            "app_name": metadata.get("appName"),
            "app_version": metadata.get("appVersion"),
            "backend_version": metadata.get("backendVersion"),
        },
        "claim_scope": "one_profiled_execute_not_the_timed_samples",
        "note": (
            "device-side values reported by QAIRT for a single profiled "
            "execute; the wall-clock samples in 'measurement' are host time "
            "around one SDK call and are not comparable to these"
        ),
    }
    if COMPUTE_IDENTIFIER in execute_us:
        block["accelerator_compute_us"] = execute_us[COMPUTE_IDENTIFIER]
    if QNN_EXECUTE_IDENTIFIER in execute_us:
        block["qnn_execute_us"] = execute_us[QNN_EXECUTE_IDENTIFIER]
    return block


def _spread(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "stddev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def _mean_by_key(
    blocks: Sequence[Mapping[str, Any]], field: str
) -> dict[str, float]:
    collected: dict[str, list[float]] = {}
    for block in blocks:
        for name, value in (block.get(field) or {}).items():
            if isinstance(value, (int, float)):
                collected.setdefault(str(name), []).append(float(value))
    return {name: statistics.fmean(values) for name, values in collected.items()}


def aggregate_device_executions(
    blocks: Sequence[Mapping[str, Any]],
    *,
    requested: int | None = None,
) -> dict[str, Any]:
    """Average N parsed executes into one block, with the mean as the headline.

    Deliberately not routed through ``summarize_latency``: that summarizer
    names every field ``*_ms``, and putting microseconds or cycle counts behind
    a millisecond label is the exact class of mislabelling this metric exists
    to correct.

    Per-sample values are kept so a reader can see the spread rather than
    trusting an average of ten.

    ``requested`` is how many profiled executes the contract asked for. When
    fewer arrive -- or when a metric is missing from some of them -- the block
    says so with ``partial``, ``samples_requested`` and ``samples_used``, rather
    than presenting a mean over a different N under the same label.
    """

    if not blocks:
        raise DeviceMetricsError("no device execute samples to aggregate")

    samples: dict[str, list[float]] = {}
    for name in _AGGREGATED_SCALARS:
        values = [
            float(block[name])
            for block in blocks
            if isinstance(block.get(name), (int, float))
        ]
        if values:
            samples[name] = values

    if not samples:
        raise DeviceMetricsError(
            "device execute samples carried no numeric metrics to average"
        )

    per_op = _mean_by_key(
        [
            {
                "per_op": {
                    str(item.get("identifier", "")): item.get("cycles")
                    for item in block.get("per_op_cycles", ())
                    if isinstance(item, Mapping)
                }
            }
            for block in blocks
        ],
        "per_op",
    )

    first = blocks[0]
    # Production latency is the accelerator's own compute time, excluding wait
    # (maintainer decision 2026-08-31): the cost of the model on the hardware,
    # with this program's test harness and the device's queueing/memory waits
    # both outside it.
    #
    # Its absolute value is small -- on SM8750 roughly 4% of accelerator execute
    # time -- which makes it the most dispersed metric in this block, measured
    # at 8-17% CV against ~2% for accelerator execute. The dispersion is
    # published with it so a change can be read against it.
    aggregated: dict[str, Any] = {
        "schema": DEVICE_EXECUTION_SCHEMA,
        "meter": DEVICE_EXECUTION_METER,
        "lane": "low_level",
        "policy": "report_only",
        "source": first.get("source"),
        "profiler_level": first.get("profiler_level"),
        "profiler_option": first.get("profiler_option"),
        "statistic": "mean",
        "sample_count": len(blocks),
        "samples_requested": int(requested) if requested is not None else len(blocks),
        "samples_used": len(blocks),
        "samples_used_by_metric": {
            name: len(values) for name, values in sorted(samples.items())
        },
        "sample_unit": "one_profiled_graph_execute",
        "producer": first.get("producer"),
        "per_op_cycles": [
            {"identifier": name, "cycles": value}
            for name, value in sorted(per_op.items())
        ],
        "execute_events_us": _mean_by_key(blocks, "execute_events_us"),
        "per_process_overhead_us": _mean_by_key(blocks, "per_process_overhead_us"),
        "spread": {name: _spread(values) for name, values in samples.items()},
        "samples": samples,
        "claim_scope": "profiled_executes_not_the_host_wall_samples",
        "note": (
            "device-side values QAIRT reported for each profiled execute, "
            "averaged; the host wall samples under 'harness_diagnostics' "
            "measure a different thing and are not comparable"
        ),
    }
    for name, values in samples.items():
        aggregated[name] = statistics.fmean(values)

    expected = aggregated["samples_requested"]
    short_metrics = {
        name: len(values) for name, values in samples.items() if len(values) < expected
    }
    if len(blocks) < expected or short_metrics:
        aggregated["partial"] = True
        aggregated["partial_reason"] = (
            f"{len(blocks)} of {expected} profiled executes were aggregated"
            if len(blocks) < expected
            else "some metrics were absent from part of the profiled executes"
        )
        if short_metrics:
            aggregated["partial_metrics"] = dict(sorted(short_metrics.items()))
    else:
        aggregated["partial"] = False

    production = aggregated.get(PRODUCTION_LATENCY_SOURCE)
    if isinstance(production, (int, float)):
        spread = aggregated["spread"][PRODUCTION_LATENCY_SOURCE]
        aggregated["production_latency_us"] = float(production)
        aggregated["production_latency_source"] = PRODUCTION_LATENCY_SOURCE
        aggregated["production_latency_cv_percent"] = (
            100.0 * spread["stddev"] / production if production else None
        )
        aggregated["production_latency_note"] = (
            "accelerator compute excluding wait: the model-and-hardware cost "
            "this program reports as production latency, with host orchestration "
            "and device queueing/memory wait both outside it; its small absolute "
            "value makes it the most dispersed metric here, so read "
            "production_latency_cv_percent alongside it"
        )
    return aggregated


__all__ = [
    "ACCELERATOR_IDENTIFIER",
    "DEVICE_EXECUTION_METER",
    "aggregate_device_executions",
    "COMPUTE_IDENTIFIER",
    "CYCLES_IDENTIFIER",
    "DEVICE_EXECUTION_SCHEMA",
    "DeviceMetricsError",
    "PRODUCTION_LATENCY_SOURCE",
    "QNN_EXECUTE_IDENTIFIER",
    "parse_device_execution",
]
