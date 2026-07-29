"""Report-only quality and latency diagnostics."""

from qairt_agent.diagnostics.latency import (
    AACalibration,
    LatencyDiagnoser,
    LatencyMeasurement,
    LatencySummary,
    OpAttribution,
    OpCycleRecord,
    attribute_op_cycles,
    compare_aa,
    summarize_latency,
)
from qairt_agent.diagnostics.sqnr import (
    QualityDiagnoser,
    QualityObservation,
    QualityReport,
    TensorQuality,
    TraceObservation,
    TraceQualityReport,
    compute_sqnr,
    compute_tensor_quality,
)

__all__ = [
    "AACalibration",
    "LatencyDiagnoser",
    "LatencyMeasurement",
    "LatencySummary",
    "OpAttribution",
    "OpCycleRecord",
    "QualityDiagnoser",
    "QualityObservation",
    "QualityReport",
    "TensorQuality",
    "TraceObservation",
    "TraceQualityReport",
    "attribute_op_cycles",
    "compare_aa",
    "compute_sqnr",
    "compute_tensor_quality",
    "summarize_latency",
]
