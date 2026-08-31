"""Report-only numerical quality metrics and attribution helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from qairt_agent.errors import InvalidSpecError


def _validated_pair(reference: Any, actual: Any) -> tuple[np.ndarray, np.ndarray]:
    reference_array = np.asarray(reference)
    actual_array = np.asarray(actual)
    if reference_array.shape != actual_array.shape:
        raise ValueError(
            f"Tensor shape mismatch: reference={reference_array.shape}, actual={actual_array.shape}"
        )
    if reference_array.size == 0:
        raise ValueError("Quality metrics are undefined for empty tensors")
    if reference_array.dtype.hasobject or actual_array.dtype.hasobject:
        raise TypeError("Object arrays are not valid numerical tensors")
    if np.iscomplexobj(reference_array) or np.iscomplexobj(actual_array):
        raise TypeError("Complex tensors are not supported by these real-valued quality metrics")

    reference_f64 = reference_array.astype(np.float64, copy=False)
    actual_f64 = actual_array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(reference_f64)):
        raise ValueError("Reference tensor contains NaN or infinity")
    if not np.all(np.isfinite(actual_f64)):
        raise ValueError("Actual tensor contains NaN or infinity")
    return reference_f64, actual_f64


def compute_sqnr(
    reference: Any,
    actual: Any,
    *,
    reference_energy_floor: float = 0.0,
) -> float | None:
    """Compute reference-energy SQNR in dB.

    Boundary rules are explicit:

    * reference energy less than or equal to ``reference_energy_floor``:
      return ``None`` because SQNR has no useful denominator;
    * positive reference energy and zero noise: return ``+inf``;
    * otherwise return ``10 * log10(reference_energy / noise_energy)``.

    Accumulation is always performed in float64. A negative energy floor is
    rejected rather than silently changing the zero-reference rule.
    """

    if not math.isfinite(reference_energy_floor) or reference_energy_floor < 0:
        raise ValueError("reference_energy_floor must be finite and non-negative")
    reference_f64, actual_f64 = _validated_pair(reference, actual)
    reference_energy = float(np.dot(reference_f64.ravel(), reference_f64.ravel()))
    if not math.isfinite(reference_energy):
        raise ValueError("Reference-energy accumulation overflowed float64")
    if reference_energy <= reference_energy_floor:
        return None
    difference = reference_f64 - actual_f64
    noise_energy = float(np.dot(difference.ravel(), difference.ravel()))
    if not math.isfinite(noise_energy):
        raise ValueError("Noise-energy accumulation overflowed float64")
    if noise_energy == 0.0:
        return math.inf
    return 10.0 * math.log10(reference_energy / noise_energy)


def _json_number(value: float | None) -> float | str | None:
    if value is None:
        return None
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return value


@dataclass(frozen=True)
class TensorQuality:
    """Descriptive metrics for one reference/actual tensor pair."""

    sqnr_db: float | None
    status: str
    numel: int
    reference_energy: float
    noise_energy: float
    rmse: float
    max_abs_error: float
    cosine_similarity: float
    normalized_rmse: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sqnr_db": _json_number(self.sqnr_db),
            "status": self.status,
            "numel": self.numel,
            "reference_energy": self.reference_energy,
            "noise_energy": self.noise_energy,
            "rmse": self.rmse,
            "max_abs_error": self.max_abs_error,
            "cosine_similarity": self.cosine_similarity,
            "normalized_rmse": self.normalized_rmse,
        }


def compute_tensor_quality(
    reference: Any,
    actual: Any,
    *,
    reference_energy_floor: float = 0.0,
) -> TensorQuality:
    reference_f64, actual_f64 = _validated_pair(reference, actual)
    difference = reference_f64 - actual_f64
    reference_flat = reference_f64.ravel()
    actual_flat = actual_f64.ravel()
    difference_flat = difference.ravel()

    reference_energy = float(np.dot(reference_flat, reference_flat))
    actual_energy = float(np.dot(actual_flat, actual_flat))
    noise_energy = float(np.dot(difference_flat, difference_flat))
    if not all(math.isfinite(value) for value in (reference_energy, actual_energy, noise_energy)):
        raise ValueError("Quality-energy accumulation overflowed float64")
    sqnr_db = compute_sqnr(
        reference_f64,
        actual_f64,
        reference_energy_floor=reference_energy_floor,
    )

    if reference_energy <= reference_energy_floor:
        status = "undefined_reference_energy"
        normalized_rmse = None
    elif noise_energy == 0.0:
        status = "exact"
        normalized_rmse = 0.0
    else:
        status = "measured"
        normalized_rmse = math.sqrt(noise_energy / reference_energy)

    if reference_energy == 0.0 and actual_energy == 0.0:
        cosine_similarity = 1.0
    elif reference_energy == 0.0 or actual_energy == 0.0:
        cosine_similarity = 0.0
    else:
        cosine_similarity = float(
            np.dot(reference_flat, actual_flat) / math.sqrt(reference_energy * actual_energy)
        )
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

    return TensorQuality(
        sqnr_db=sqnr_db,
        status=status,
        numel=int(reference_f64.size),
        reference_energy=reference_energy,
        noise_energy=noise_energy,
        rmse=math.sqrt(noise_energy / reference_f64.size),
        max_abs_error=float(np.max(np.abs(difference))),
        cosine_similarity=cosine_similarity,
        normalized_rmse=normalized_rmse,
    )


@dataclass(frozen=True)
class QualityObservation:
    slice_id: str
    tensor_name: str
    teacher_forced: TensorQuality | None
    device_chain: TensorQuality | None
    attribution: str
    propagated_noise_delta: float | None
    lineage: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "tensor_name": self.tensor_name,
            "teacher_forced": self.teacher_forced.to_dict() if self.teacher_forced else None,
            "device_chain": self.device_chain.to_dict() if self.device_chain else None,
            "attribution": self.attribution,
            "propagated_noise_delta": self.propagated_noise_delta,
            "lineage": dict(self.lineage),
        }


@dataclass(frozen=True)
class QualityReport:
    """A descriptive report. It intentionally has no pass/fail field."""

    observations: tuple[QualityObservation, ...]
    first_teacher_error: tuple[str, str] | None
    first_chain_error: tuple[str, str] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": [item.to_dict() for item in self.observations],
            "first_teacher_error": list(self.first_teacher_error) if self.first_teacher_error else None,
            "first_chain_error": list(self.first_chain_error) if self.first_chain_error else None,
            "policy": "report_only",
        }


@dataclass(frozen=True)
class TraceObservation:
    tap_name: str
    quality: TensorQuality
    sqnr_change_from_previous_db: float | None
    noise_energy_change_from_previous: float | None
    lineage: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tap_name": self.tap_name,
            "quality": self.quality.to_dict(),
            "sqnr_change_from_previous_db": _json_number(self.sqnr_change_from_previous_db),
            "noise_energy_change_from_previous": self.noise_energy_change_from_previous,
            "lineage": dict(self.lineage),
        }


@dataclass(frozen=True)
class TraceQualityReport:
    observations: tuple[TraceObservation, ...]
    first_observed_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": [item.to_dict() for item in self.observations],
            "first_observed_error": self.first_observed_error,
            "claim_scope": "first_observed_divergence_not_root_cause",
            "policy": "report_only",
        }


class QualityDiagnoser:
    """Generate teacher-forced, chain, layer, and op quality reports."""

    def __init__(self, *, reference_energy_floor: float = 0.0) -> None:
        if not math.isfinite(reference_energy_floor) or reference_energy_floor < 0:
            raise ValueError("reference_energy_floor must be finite and non-negative")
        self.reference_energy_floor = float(reference_energy_floor)

    def compare(self, reference: Any, actual: Any) -> TensorQuality:
        return compute_tensor_quality(
            reference,
            actual,
            reference_energy_floor=self.reference_energy_floor,
        )

    @staticmethod
    def _attribution(
        teacher: TensorQuality | None,
        chain: TensorQuality | None,
    ) -> tuple[str, float | None]:
        if teacher is None and chain is None:
            return "not_observed", None
        if teacher is None:
            return "chain_only_observation", None
        if chain is None:
            return "local_only_observation", None

        propagated_delta = chain.noise_energy - teacher.noise_energy
        teacher_exact = teacher.noise_energy == 0.0
        chain_exact = chain.noise_energy == 0.0
        if teacher_exact and chain_exact:
            label = "exact"
        elif teacher_exact:
            label = "propagated_only"
        elif chain_exact:
            label = "local_error_not_present_in_chain"
        elif propagated_delta > 0.0:
            label = "local_plus_propagated"
        elif propagated_delta < 0.0:
            label = "local_error_chain_reduced"
        else:
            label = "local_error_chain_unchanged"
        return label, propagated_delta

    def _compare_localized(
        self,
        reference: Any,
        actual: Any,
        *,
        slice_id: str,
        tensor_name: str,
        source: str,
    ) -> Any:
        """Compare one tap, naming it if the comparison is impossible.

        A device tensor full of NaN used to abort the whole validation with
        "Actual tensor contains NaN or infinity" and nothing else -- the reader
        learned that something diverged but not where, which is precisely what a
        validation stage exists to tell them. The failure is still a failure;
        it just localizes now.
        """

        try:
            return self.compare(reference, actual)
        except (ValueError, TypeError) as error:
            array = np.asarray(actual)
            non_finite = (
                int(np.count_nonzero(~np.isfinite(array)))
                if array.size and not array.dtype.hasobject
                and np.issubdtype(array.dtype, np.number)
                and not np.iscomplexobj(array)
                else None
            )
            raise InvalidSpecError(
                f"quality comparison failed at {slice_id}.{tensor_name} "
                f"({source}): {error}",
                stage="validate",
                details={
                    "slice_id": slice_id,
                    "tensor_name": tensor_name,
                    "source": source,
                    "reason": str(error),
                    **(
                        {
                            "non_finite_elements": non_finite,
                            "element_count": int(array.size),
                        }
                        if non_finite is not None
                        else {}
                    ),
                },
            ) from error

    def diagnose_slices(
        self,
        references: Mapping[str, Mapping[str, Any]],
        *,
        teacher_forced_outputs: Mapping[str, Mapping[str, Any]] | None = None,
        device_chain_outputs: Mapping[str, Mapping[str, Any]] | None = None,
        lineage: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    ) -> QualityReport:
        """Compare the same slice taps in teacher-forced and chained execution."""

        teacher_outputs = teacher_forced_outputs or {}
        chain_outputs = device_chain_outputs or {}
        teacher_enabled = teacher_forced_outputs is not None
        chain_enabled = device_chain_outputs is not None
        lineage_by_slice = lineage or {}
        observations: list[QualityObservation] = []
        first_teacher_error: tuple[str, str] | None = None
        first_chain_error: tuple[str, str] | None = None

        for slice_id, tensor_references in references.items():
            teacher_slice = teacher_outputs.get(slice_id, {})
            chain_slice = chain_outputs.get(slice_id, {})
            for tensor_name, reference in tensor_references.items():
                if teacher_enabled and tensor_name not in teacher_slice:
                    raise KeyError(
                        f"Teacher-forced output is missing {slice_id}.{tensor_name}"
                    )
                if chain_enabled and tensor_name not in chain_slice:
                    raise KeyError(f"Device-chain output is missing {slice_id}.{tensor_name}")
                teacher = (
                    self._compare_localized(
                        reference,
                        teacher_slice[tensor_name],
                        slice_id=str(slice_id),
                        tensor_name=str(tensor_name),
                        source="teacher_forced",
                    )
                    if tensor_name in teacher_slice
                    else None
                )
                chain = (
                    self._compare_localized(
                        reference,
                        chain_slice[tensor_name],
                        slice_id=str(slice_id),
                        tensor_name=str(tensor_name),
                        source="device_chain",
                    )
                    if tensor_name in chain_slice
                    else None
                )
                if teacher is not None and teacher.noise_energy > 0.0 and first_teacher_error is None:
                    first_teacher_error = (str(slice_id), str(tensor_name))
                if chain is not None and chain.noise_energy > 0.0 and first_chain_error is None:
                    first_chain_error = (str(slice_id), str(tensor_name))
                attribution, propagated_delta = self._attribution(teacher, chain)
                observations.append(
                    QualityObservation(
                        slice_id=str(slice_id),
                        tensor_name=str(tensor_name),
                        teacher_forced=teacher,
                        device_chain=chain,
                        attribution=attribution,
                        propagated_noise_delta=propagated_delta,
                        lineage=dict(
                            lineage_by_slice.get(slice_id, {}).get(tensor_name, {})
                        ),
                    )
                )

        return QualityReport(
            observations=tuple(observations),
            first_teacher_error=first_teacher_error,
            first_chain_error=first_chain_error,
        )

    def diagnose_trace(
        self,
        references: Mapping[str, Any],
        actuals: Mapping[str, Any],
        *,
        lineage: Mapping[str, Mapping[str, Any]] | None = None,
        order: Sequence[str] | None = None,
    ) -> TraceQualityReport:
        """Describe the first observed layer/op divergence in topological order."""

        tap_order = tuple(order) if order is not None else tuple(references)
        missing_references = set(tap_order) - set(references)
        missing_actuals = set(tap_order) - set(actuals)
        if missing_references or missing_actuals:
            raise KeyError(
                f"Trace taps missing; references={sorted(missing_references)}, "
                f"actuals={sorted(missing_actuals)}"
            )

        observations: list[TraceObservation] = []
        first_observed_error: str | None = None
        previous: TensorQuality | None = None
        for tap_name in tap_order:
            quality = self.compare(references[tap_name], actuals[tap_name])
            if quality.noise_energy > 0.0 and first_observed_error is None:
                first_observed_error = tap_name

            sqnr_change: float | None = None
            noise_change: float | None = None
            if previous is not None:
                noise_change = quality.noise_energy - previous.noise_energy
                if (
                    previous.sqnr_db is not None
                    and quality.sqnr_db is not None
                    and math.isfinite(previous.sqnr_db)
                    and math.isfinite(quality.sqnr_db)
                ):
                    sqnr_change = quality.sqnr_db - previous.sqnr_db
            observations.append(
                TraceObservation(
                    tap_name=tap_name,
                    quality=quality,
                    sqnr_change_from_previous_db=sqnr_change,
                    noise_energy_change_from_previous=noise_change,
                    lineage=dict((lineage or {}).get(tap_name, {})),
                )
            )
            previous = quality

        return TraceQualityReport(
            observations=tuple(observations),
            first_observed_error=first_observed_error,
        )


__all__ = [
    "QualityDiagnoser",
    "QualityObservation",
    "QualityReport",
    "TensorQuality",
    "TraceObservation",
    "TraceQualityReport",
    "compute_sqnr",
    "compute_tensor_quality",
]
