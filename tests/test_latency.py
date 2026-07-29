from __future__ import annotations

import inspect

import pytest

from qairt_agent.diagnostics.latency import (
    LatencyDiagnoser,
    attribute_op_cycles,
    compare_aa,
    summarize_latency,
)


def test_summarize_latency_uses_wall_time_robust_statistics() -> None:
    summary = summarize_latency([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary.count == 5
    assert summary.p50_ms == 3.0
    assert summary.p90_ms == pytest.approx(4.6)
    assert summary.mad_ms == 1.0
    assert summary.robust_cv_percent == pytest.approx(49.42)
    assert summary.to_dict()["metric_scope"] == "wall_latency"


def test_aa_calibration_reports_noise_without_policy_gate() -> None:
    calibration = compare_aa([10.0, 10.0, 10.0], [10.5, 10.5, 10.5])

    assert calibration.median_delta_ms == 0.5
    assert calibration.median_delta_percent == pytest.approx(5.0)
    assert calibration.observed_noise_percent == pytest.approx(5.0)
    assert calibration.to_dict()["policy"] == "report_only"
    assert "passed" not in calibration.to_dict()


def test_measure_excludes_warmup_and_uses_supplied_clock() -> None:
    timestamps = iter([0, 1_000_000, 1_000_000, 3_000_000])
    calls: list[int] = []
    diagnoser = LatencyDiagnoser(clock_ns=lambda: next(timestamps))

    measurement = diagnoser.measure(lambda: calls.append(1), warmup=1, repeats=2)

    assert len(calls) == 3
    assert measurement.warmup_count == 1
    assert measurement.summary.samples_ms == (1.0, 2.0)
    assert measurement.summary.p50_ms == 1.5


def test_op_attribution_uses_max_thread_and_never_claims_wall_sum() -> None:
    baseline = [
        {
            "op_id": "decoder/layer0/matmul",
            "thread_cycles": [100, 80, 40],
            "lineage": {"layer": 0, "op": "MatMul"},
        },
        {"op_id": "decoder/layer0/add", "cycles": 50},
    ]
    candidate = [
        {
            "op_id": "decoder/layer0/matmul",
            "thread_cycles": [130, 70, 60],
            "critical_path": True,
        },
        {"op_id": "decoder/layer0/add", "cycles": 40},
        {"op_id": "decoder/layer0/reshape", "cycles": 5},
    ]

    attribution = attribute_op_cycles(baseline, candidate)
    matmul = next(item for item in attribution if item.op_id.endswith("matmul"))
    reshape = next(item for item in attribution if item.op_id.endswith("reshape"))

    assert attribution[0].op_id == "decoder/layer0/matmul"
    assert matmul.baseline_cycles == 100
    assert matmul.candidate_cycles == 130
    assert matmul.delta_percent == pytest.approx(30.0)
    assert matmul.cycle_basis == "max_thread"
    assert matmul.critical_path is True
    assert matmul.lineage == {"layer": 0, "op": "MatMul"}
    assert reshape.status == "candidate_only"
    assert (
        matmul.to_dict()["claim_scope"]
        == "work_attribution_not_additive_wall_latency"
    )


def test_latency_rejects_empty_negative_and_duplicate_ops() -> None:
    with pytest.raises(ValueError, match="At least one"):
        summarize_latency([])
    with pytest.raises(ValueError, match="cannot be negative"):
        summarize_latency([1.0, -1.0])
    with pytest.raises(ValueError, match="Duplicate op"):
        attribute_op_cycles(
            [{"op_id": "x", "cycles": 1}, {"op_id": "x", "cycles": 2}],
            {"x": 1},
        )


def test_measurement_defaults_match_benchmark_contract() -> None:
    parameters = inspect.signature(LatencyDiagnoser.measure).parameters
    assert parameters["warmup"].default == 10
    assert parameters["repeats"].default == 50
