"""Wall-latency statistics, A/A calibration, and non-additive op attribution."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


def _finite_positive_samples(samples_ms: Iterable[float]) -> np.ndarray:
    samples = np.asarray(tuple(float(value) for value in samples_ms), dtype=np.float64)
    if samples.size == 0:
        raise ValueError("At least one latency sample is required")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Latency samples must be finite")
    if np.any(samples < 0.0):
        raise ValueError("Latency samples cannot be negative")
    return samples


@dataclass(frozen=True)
class LatencySummary:
    """Descriptive wall-time statistics in milliseconds."""

    samples_ms: tuple[float, ...]
    count: int
    minimum_ms: float
    maximum_ms: float
    mean_ms: float
    stddev_ms: float
    p10_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    mad_ms: float
    robust_cv_percent: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples_ms": list(self.samples_ms),
            "count": self.count,
            "minimum_ms": self.minimum_ms,
            "maximum_ms": self.maximum_ms,
            "mean_ms": self.mean_ms,
            "stddev_ms": self.stddev_ms,
            "p10_ms": self.p10_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "mad_ms": self.mad_ms,
            "robust_cv_percent": self.robust_cv_percent,
            "metric_scope": "wall_latency",
        }


def summarize_latency(samples_ms: Iterable[float]) -> LatencySummary:
    """Summarize already-measured wall latency without applying a policy gate."""

    samples = _finite_positive_samples(samples_ms)
    median = float(np.percentile(samples, 50))
    mad = float(np.median(np.abs(samples - median)))
    robust_cv = None if median == 0.0 else 100.0 * 1.4826 * mad / median
    return LatencySummary(
        samples_ms=tuple(float(value) for value in samples),
        count=int(samples.size),
        minimum_ms=float(np.min(samples)),
        maximum_ms=float(np.max(samples)),
        mean_ms=float(np.mean(samples)),
        stddev_ms=float(np.std(samples)),
        p10_ms=float(np.percentile(samples, 10)),
        p50_ms=median,
        p90_ms=float(np.percentile(samples, 90)),
        p95_ms=float(np.percentile(samples, 95)),
        mad_ms=mad,
        robust_cv_percent=robust_cv,
    )


@dataclass(frozen=True)
class LatencyMeasurement:
    summary: LatencySummary
    warmup_count: int
    repeat_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup_count": self.warmup_count,
            "repeat_count": self.repeat_count,
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True)
class AACalibration:
    """Two same-path latency runs; deliberately report-only."""

    first: LatencySummary
    second: LatencySummary
    median_delta_ms: float
    median_delta_percent: float | None
    observed_noise_percent: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "first": self.first.to_dict(),
            "second": self.second.to_dict(),
            "median_delta_ms": self.median_delta_ms,
            "median_delta_percent": self.median_delta_percent,
            "observed_noise_percent": self.observed_noise_percent,
            "policy": "report_only",
        }


def compare_aa(
    first_samples_ms: Iterable[float],
    second_samples_ms: Iterable[float],
) -> AACalibration:
    """Compare A/A samples and estimate a robust observed noise percentage."""

    first = summarize_latency(first_samples_ms)
    second = summarize_latency(second_samples_ms)
    delta_ms = second.p50_ms - first.p50_ms
    delta_percent = None if first.p50_ms == 0.0 else 100.0 * delta_ms / first.p50_ms
    noise_candidates = [
        abs(delta_percent) if delta_percent is not None else None,
        first.robust_cv_percent,
        second.robust_cv_percent,
    ]
    finite_candidates = [value for value in noise_candidates if value is not None and math.isfinite(value)]
    observed_noise = max(finite_candidates) if finite_candidates else None
    return AACalibration(
        first=first,
        second=second,
        median_delta_ms=delta_ms,
        median_delta_percent=delta_percent,
        observed_noise_percent=observed_noise,
    )


@dataclass(frozen=True)
class OpCycleRecord:
    op_id: str
    cycles: float
    cycle_basis: str = "reported"
    critical_path: bool = False
    lineage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpAttribution:
    op_id: str
    baseline_cycles: float | None
    candidate_cycles: float | None
    delta_cycles: float | None
    delta_percent: float | None
    cycle_basis: str
    critical_path: bool
    status: str
    lineage: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "baseline_cycles": self.baseline_cycles,
            "candidate_cycles": self.candidate_cycles,
            "delta_cycles": self.delta_cycles,
            "delta_percent": self.delta_percent,
            "cycle_basis": self.cycle_basis,
            "critical_path": self.critical_path,
            "status": self.status,
            "lineage": dict(self.lineage),
            "claim_scope": "work_attribution_not_additive_wall_latency",
        }


def _record_from_value(op_id: str, value: Any) -> OpCycleRecord:
    if isinstance(value, OpCycleRecord):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return OpCycleRecord(op_id=op_id, cycles=float(value))
    if not isinstance(value, Mapping):
        raise TypeError(f"Cannot interpret op record {op_id!r} from {type(value).__name__}")

    record_id = str(
        value.get("op_id")
        or value.get("source_id")
        or value.get("name")
        or value.get("op_name")
        or op_id
    )
    cycle_basis = str(value.get("cycle_basis", "reported"))
    if value.get("cycles") is not None:
        cycles = float(value["cycles"])
    elif value.get("thread_cycles") is not None:
        thread_cycles_value = value["thread_cycles"]
        if isinstance(thread_cycles_value, Mapping):
            thread_cycles = [float(item) for item in thread_cycles_value.values()]
        else:
            thread_cycles = [float(item) for item in thread_cycles_value]
        if not thread_cycles:
            raise ValueError(f"Op record {record_id!r} has no thread-cycle values")
        # Threads overlap; summing them would falsely imply wall latency.
        cycles = max(thread_cycles)
        cycle_basis = "max_thread"
    else:
        raise ValueError(f"Op record {record_id!r} has neither cycles nor thread_cycles")

    if not math.isfinite(cycles) or cycles < 0.0:
        raise ValueError(f"Op record {record_id!r} has invalid cycles: {cycles}")
    lineage = value.get("lineage", {})
    if not isinstance(lineage, Mapping):
        raise TypeError(f"Op record {record_id!r} lineage must be a mapping")
    return OpCycleRecord(
        op_id=record_id,
        cycles=cycles,
        cycle_basis=cycle_basis,
        critical_path=bool(value.get("critical_path", False)),
        lineage=dict(lineage),
    )


def _normalize_op_records(
    records: Mapping[str, Any] | Sequence[Mapping[str, Any] | OpCycleRecord],
) -> dict[str, OpCycleRecord]:
    normalized: dict[str, OpCycleRecord] = {}
    items: Iterable[tuple[str, Any]]
    if isinstance(records, Mapping):
        items = records.items()
    else:
        items = ((str(index), value) for index, value in enumerate(records))
    for fallback_id, value in items:
        record = _record_from_value(str(fallback_id), value)
        if record.op_id in normalized:
            raise ValueError(f"Duplicate op attribution key {record.op_id!r}")
        normalized[record.op_id] = record
    return normalized


def attribute_op_cycles(
    baseline: Mapping[str, Any] | Sequence[Mapping[str, Any] | OpCycleRecord],
    candidate: Mapping[str, Any] | Sequence[Mapping[str, Any] | OpCycleRecord],
) -> tuple[OpAttribution, ...]:
    """Match op-cycle records without claiming their sum is wall latency."""

    baseline_records = _normalize_op_records(baseline)
    candidate_records = _normalize_op_records(candidate)
    attributions: list[OpAttribution] = []

    for op_id in sorted(set(baseline_records) | set(candidate_records)):
        baseline_record = baseline_records.get(op_id)
        candidate_record = candidate_records.get(op_id)
        if baseline_record is None:
            status = "candidate_only"
            baseline_cycles = None
            candidate_cycles = candidate_record.cycles if candidate_record else None
            delta_cycles = None
            delta_percent = None
        elif candidate_record is None:
            status = "baseline_only"
            baseline_cycles = baseline_record.cycles
            candidate_cycles = None
            delta_cycles = None
            delta_percent = None
        else:
            status = "matched"
            baseline_cycles = baseline_record.cycles
            candidate_cycles = candidate_record.cycles
            delta_cycles = candidate_cycles - baseline_cycles
            delta_percent = (
                None if baseline_cycles == 0.0 else 100.0 * delta_cycles / baseline_cycles
            )

        chosen = candidate_record or baseline_record
        assert chosen is not None
        lineage = dict(baseline_record.lineage if baseline_record else {})
        if candidate_record:
            lineage.update(candidate_record.lineage)
        attributions.append(
            OpAttribution(
                op_id=op_id,
                baseline_cycles=baseline_cycles,
                candidate_cycles=candidate_cycles,
                delta_cycles=delta_cycles,
                delta_percent=delta_percent,
                cycle_basis=chosen.cycle_basis,
                critical_path=bool(
                    (baseline_record and baseline_record.critical_path)
                    or (candidate_record and candidate_record.critical_path)
                ),
                status=status,
                lineage=lineage,
            )
        )

    def sort_key(item: OpAttribution) -> tuple[int, float]:
        if item.delta_cycles is None:
            return (1, 0.0)
        if item.delta_cycles > 0.0:
            return (2, item.delta_cycles)
        return (0, item.delta_cycles)

    # Put matched positive regressions first, then unmatched records, then improvements.
    return tuple(
        sorted(
            attributions,
            key=sort_key,
            reverse=True,
        )
    )


class LatencyDiagnoser:
    """Measure a warm context, calibrate A/A noise, and attribute op work."""

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        self.clock_ns = clock_ns
        self.synchronize = synchronize

    def _synchronize(self) -> None:
        if self.synchronize is not None:
            self.synchronize()

    def measure(
        self,
        invocation: Callable[[], Any],
        *,
        warmup: int = 10,
        repeats: int = 50,
    ) -> LatencyMeasurement:
        """Measure only invocation wall time; setup should occur before this call."""

        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        if repeats <= 0:
            raise ValueError("repeats must be positive")

        for _ in range(warmup):
            invocation()
            self._synchronize()

        samples_ms: list[float] = []
        for _ in range(repeats):
            self._synchronize()
            start_ns = self.clock_ns()
            invocation()
            self._synchronize()
            end_ns = self.clock_ns()
            if end_ns < start_ns:
                raise RuntimeError("Latency clock moved backwards")
            samples_ms.append((end_ns - start_ns) / 1_000_000.0)

        return LatencyMeasurement(
            summary=summarize_latency(samples_ms),
            warmup_count=warmup,
            repeat_count=repeats,
        )

    def calibrate_aa(
        self,
        invocation: Callable[[], Any],
        *,
        warmup: int = 10,
        repeats: int = 50,
    ) -> AACalibration:
        first = self.measure(invocation, warmup=warmup, repeats=repeats)
        second = self.measure(invocation, warmup=warmup, repeats=repeats)
        return compare_aa(first.summary.samples_ms, second.summary.samples_ms)

    @staticmethod
    def compare_aa(
        first_samples_ms: Iterable[float],
        second_samples_ms: Iterable[float],
    ) -> AACalibration:
        return compare_aa(first_samples_ms, second_samples_ms)

    @staticmethod
    def attribute_ops(
        baseline: Mapping[str, Any] | Sequence[Mapping[str, Any] | OpCycleRecord],
        candidate: Mapping[str, Any] | Sequence[Mapping[str, Any] | OpCycleRecord],
    ) -> tuple[OpAttribution, ...]:
        return attribute_op_cycles(baseline, candidate)


__all__ = [
    "AACalibration",
    "LatencyDiagnoser",
    "LatencyMeasurement",
    "LatencySummary",
    "OpAttribution",
    "OpCycleRecord",
    "attribute_op_cycles",
    "compare_aa",
    "summarize_latency",
]
