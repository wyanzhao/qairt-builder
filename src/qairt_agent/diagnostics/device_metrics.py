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

from typing import Any, Mapping, Sequence

DEVICE_EXECUTION_SCHEMA = "qairt-agent.device-execution/1"

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


__all__ = [
    "ACCELERATOR_IDENTIFIER",
    "COMPUTE_IDENTIFIER",
    "CYCLES_IDENTIFIER",
    "DEVICE_EXECUTION_SCHEMA",
    "DeviceMetricsError",
    "QNN_EXECUTE_IDENTIFIER",
    "parse_device_execution",
]
