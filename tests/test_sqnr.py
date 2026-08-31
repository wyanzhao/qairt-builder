from __future__ import annotations

import math

import numpy as np
import pytest

from qairt_agent.diagnostics.sqnr import (
    QualityDiagnoser,
    compute_sqnr,
    compute_tensor_quality,
)


def test_sqnr_reference_energy_boundary_rules() -> None:
    exact_nonzero = compute_sqnr(np.array([1.0, -1.0]), np.array([1.0, -1.0]))
    assert exact_nonzero is not None and math.isinf(exact_nonzero)

    assert compute_sqnr(np.zeros(4), np.zeros(4)) is None
    assert compute_sqnr(np.zeros(4), np.ones(4)) is None
    assert compute_sqnr(np.ones(2), np.zeros(2)) == pytest.approx(0.0)

    floored = compute_sqnr(
        np.array([1e-9]),
        np.array([0.0]),
        reference_energy_floor=1e-17,
    )
    assert floored is None


def test_tensor_quality_reports_zero_reference_with_other_metrics() -> None:
    quality = compute_tensor_quality(np.zeros(2), np.array([3.0, 4.0]))

    assert quality.status == "undefined_reference_energy"
    assert quality.sqnr_db is None
    assert quality.rmse == pytest.approx(math.sqrt(12.5))
    assert quality.max_abs_error == 4.0
    assert quality.cosine_similarity == 0.0
    assert quality.normalized_rmse is None


def test_quality_diagnoser_attributes_local_and_propagated_error_without_gate() -> None:
    diagnoser = QualityDiagnoser()
    references = {
        "slice0": {"hidden": np.array([1.0, 2.0])},
        "slice1": {"logits": np.array([2.0, 4.0])},
    }
    teacher = {
        "slice0": {"hidden": np.array([1.0, 2.0])},
        "slice1": {"logits": np.array([2.0, 3.0])},
    }
    chain = {
        "slice0": {"hidden": np.array([1.0, 1.5])},
        "slice1": {"logits": np.array([1.0, 2.0])},
    }

    report = diagnoser.diagnose_slices(
        references,
        teacher_forced_outputs=teacher,
        device_chain_outputs=chain,
        lineage={"slice1": {"logits": {"layer": 7, "op": "MatMul"}}},
    )

    assert report.first_teacher_error == ("slice1", "logits")
    assert report.first_chain_error == ("slice0", "hidden")
    assert report.observations[0].attribution == "propagated_only"
    assert report.observations[1].attribution == "local_plus_propagated"
    assert report.observations[1].lineage == {"layer": 7, "op": "MatMul"}
    assert report.to_dict()["policy"] == "report_only"
    assert "pass" not in report.to_dict()


def test_trace_reports_first_observed_divergence_not_root_cause() -> None:
    diagnoser = QualityDiagnoser()
    references = {
        "layer0": np.array([1.0]),
        "layer1": np.array([2.0]),
        "op_matmul": np.array([3.0]),
    }
    actuals = {
        "layer0": np.array([1.0]),
        "layer1": np.array([1.5]),
        "op_matmul": np.array([2.0]),
    }
    report = diagnoser.diagnose_trace(
        references,
        actuals,
        lineage={"op_matmul": {"source_op": "/decoder/MatMul"}},
    )

    assert report.first_observed_error == "layer1"
    assert report.observations[-1].lineage == {"source_op": "/decoder/MatMul"}
    assert report.to_dict()["claim_scope"] == "first_observed_divergence_not_root_cause"


def test_sqnr_rejects_shape_and_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_sqnr(np.zeros((1, 2)), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="NaN or infinity"):
        compute_sqnr(np.array([1.0]), np.array([np.nan]))
    with pytest.raises(ValueError, match="finite and non-negative"):
        compute_sqnr(np.array([1.0]), np.array([1.0]), reference_energy_floor=np.inf)
    with pytest.raises(TypeError, match="Complex tensors"):
        compute_sqnr(np.array([1.0j]), np.array([1.0j]))


def test_diagnoser_rejects_missing_requested_outputs() -> None:
    with pytest.raises(KeyError, match="missing slice0.hidden"):
        QualityDiagnoser().diagnose_slices(
            {"slice0": {"hidden": np.ones(1)}},
            teacher_forced_outputs={"slice0": {}},
        )


# --------------------------------------------------------------------------- #
# A non-finite device tensor localizes instead of aborting blind (T16)
# --------------------------------------------------------------------------- #


def test_a_non_finite_device_tensor_names_the_slice_and_tensor() -> None:
    from qairt_agent.errors import InvalidSpecError

    diagnoser = QualityDiagnoser()
    references = {"decoder_00": {"hidden": np.array([1.0, 2.0], dtype=np.float32)}}
    chain = {
        "decoder_00": {"hidden": np.array([1.0, np.nan], dtype=np.float32)}
    }

    with pytest.raises(InvalidSpecError) as error:
        diagnoser.diagnose_slices(references, device_chain_outputs=chain)

    assert "decoder_00.hidden" in str(error.value)
    details = error.value.details
    assert details["slice_id"] == "decoder_00"
    assert details["tensor_name"] == "hidden"
    assert details["source"] == "device_chain"
    assert details["non_finite_elements"] == 1
    assert details["element_count"] == 2


def test_a_non_finite_teacher_tensor_says_which_side_it_was() -> None:
    from qairt_agent.errors import InvalidSpecError

    diagnoser = QualityDiagnoser()
    references = {"decoder_00": {"hidden": np.array([1.0], dtype=np.float32)}}
    teacher = {"decoder_00": {"hidden": np.array([np.inf], dtype=np.float32)}}

    with pytest.raises(InvalidSpecError) as error:
        diagnoser.diagnose_slices(references, teacher_forced_outputs=teacher)

    assert error.value.details["source"] == "teacher_forced"
