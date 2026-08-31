"""Typed report models must not change a byte of what is published.

Published reports are content-addressed evidence: their hashes are recorded in
manifests and receipts, so a model that dropped, reordered or re-typed a key
would make an existing report unreproducible. These tests build real reports
through the pipeline and assert the round trip through the model is exact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qairt_agent.artifacts import canonical_json_bytes
from qairt_agent.contracts_reports import (
    DeviceExecutionBlock,
    MultiArLatencyReport,
    MultiArSqnrReport,
)

from test_pipeline import (  # noqa: E402 - shared fixtures live with the suite
    FakeAdapterFactory,
    MultiArFakeAdapterFactory,
    _fake_agent,
    _load_run,
    _make_spec,
    _vector_case,
)


def _multi_ar_run(tmp_path: Path) -> Any:
    ar1 = _vector_case(
        tmp_path / "ar1",
        inputs={"x": np.array([1.0], dtype=np.float32)},
        goldens={"y": np.array([1.0], dtype=np.float32)},
        case_id="report-ar1",
    )
    ar128 = _vector_case(
        tmp_path / "ar128",
        inputs={"x": np.array([128.0], dtype=np.float32)},
        goldens={"y": np.array([128.0], dtype=np.float32)},
        case_id="report-ar128",
    )
    spec = _make_spec(
        tmp_path,
        vectors={
            "mode": "provided",
            "validation_manifests_by_ar": {1: ar1, 128: ar128},
        },
        sequence={
            "ars": [1, 128],
            "context_lengths": [4096],
            "weight_sharing": True,
            "native_kv": False,
        },
    )
    agent = _fake_agent(MultiArFakeAdapterFactory())
    built = agent.build(spec)
    assert built.ok, built.error
    validated = agent.validate(built.manifest.path, built.manifest.sha256)
    assert validated.ok, validated.error
    benchmarked = agent.benchmark(
        validated.manifest.path,
        validated.manifest.sha256,
        config={"warmup_runs": 0, "measured_runs": 1, "aa_calibration": False},
    )
    assert benchmarked.ok, benchmarked.error
    return benchmarked.manifest


def _published(manifest: Any, logical_name: str) -> dict[str, Any]:
    ref = next(
        artifact
        for artifact in _load_run(manifest).artifacts
        if artifact.logical_name == logical_name
    )
    return json.loads(ref.path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "logical_name, model",
    [
        ("sqnr_report", MultiArSqnrReport),
        ("latency_report", MultiArLatencyReport),
    ],
)
def test_the_model_round_trip_is_byte_identical(
    tmp_path: Path, logical_name: str, model: Any
) -> None:
    manifest = _multi_ar_run(tmp_path)
    published = _published(manifest, logical_name)

    rendered = model.model_validate(published).to_payload()

    # Published reports are written with sorted keys, so byte stability is
    # exactly this: same keys, same values, canonically serialized.
    assert canonical_json_bytes(rendered) == canonical_json_bytes(published)
    assert set(rendered) == set(published)


def test_an_unmodelled_key_survives_the_round_trip() -> None:
    # Reports grow fields; a model that silently dropped one would make the
    # published hash unreproducible.
    payload = {
        "schema": "qairt-agent.multi-ar-sqnr-report.v1",
        "policy": "report_only",
        "coverage": {
            "mode": "all_requested_ars",
            "requested_ars": [1],
            "executed_ars": [1],
            "missing_ars": [],
            "complete": True,
            "context_lengths": [4096],
        },
        "results_by_ar": {},
        "reference_source": "provided",
        "reference_sources_by_ar": {},
        "first_teacher_error": None,
        "first_teacher_error_ar": None,
        "first_chain_error": None,
        "first_chain_error_ar": None,
        "divergence_observed": False,
        "a_field_added_later": {"nested": [1, 2, 3]},
    }

    rendered = MultiArSqnrReport.model_validate(payload).to_payload()

    assert rendered["a_field_added_later"] == {"nested": [1, 2, 3]}
    assert json.dumps(rendered, sort_keys=True) == json.dumps(payload, sort_keys=True)


def test_a_wrong_schema_string_is_refused() -> None:
    payload = {
        "schema": "qairt-agent.single-ar-sqnr-report.v1",
        "policy": "report_only",
        "coverage": {
            "mode": "all_requested_ars",
            "requested_ars": [1],
            "executed_ars": [1],
            "complete": True,
        },
        "results_by_ar": {},
        "reference_source": "provided",
        "reference_sources_by_ar": {},
        "divergence_observed": False,
    }

    with pytest.raises(Exception):
        MultiArSqnrReport.model_validate(payload)


def test_a_degraded_device_block_is_not_read_as_measured() -> None:
    block = DeviceExecutionBlock.model_validate(
        {
            "schema": "qairt-agent.device-execution/2",
            "policy": "report_only",
            "available": False,
            "reason": "the QAIRT adapter does not expose capture_device_execution",
        }
    )

    assert block.measured is False
    assert block.production_latency_us is None


def test_a_measured_device_block_reports_its_dispersion(tmp_path: Path) -> None:
    manifest = _multi_ar_run(tmp_path)
    latency = _published(manifest, "latency_report")
    ar1 = latency["results_by_ar"]["1"]["report"]

    block = DeviceExecutionBlock.model_validate(ar1["device_execution"])

    assert block.measured is True
    assert block.meter == "qnn_accelerator"
    assert block.production_latency_source == "accelerator_compute_us"
    assert block.production_latency_cv_percent is not None
