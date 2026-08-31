"""Cross-run comparison of the headline metrics, report-only.

The program contract already says a latency change must be read against
``production_latency_cv_percent`` -- on SM8750 that dispersion is 8-17% -- which
is exactly the sort of arithmetic a person should not be doing by eye at the end
of a long run. This module loads two hash-verified runs and emits the deltas.

Two rules shape everything here:

* **Fail closed on non-comparable pairs.** A delta between two runs that used
  different presets, targets, AR sets, context lengths, SQNR modes, or latency
  meters is not a measurement of anything. Each mismatch is named.
* **No verdicts.** Deltas are published with their dispersion context and
  nothing else; the program applies no pass/fail threshold anywhere, and a
  comparison is not the place to start.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qairt_agent.artifacts import ManifestStore, verify_artifact
from qairt_agent.contracts import ArtifactKind, ArtifactRef, RunManifest
from qairt_agent.contracts_reports import DeviceExecutionBlock
from qairt_agent.errors import InvalidSpecError

COMPARE_SCHEMA = "qairt-agent.run-comparison/1"

#: Spec identity that must agree before a delta means anything.
_IDENTITY_FIELDS = (
    "preset",
    "family",
    "target",
    "ars",
    "context_lengths",
    "sqnr_modes",
)


#: Published manifests are named ``manifest-r000003-<sha256>.json``.
_MANIFEST_NAME = re.compile(r"^manifest-r\d{6}-([0-9a-f]{64})\.json$")


def _load_manifest(reference: ArtifactRef | str | Path) -> tuple[ArtifactRef, RunManifest]:
    """Load a manifest, always against an independently recorded hash.

    ``ManifestStore`` refuses a bare path with no expectation, and rightly so:
    hashing a file and comparing it to itself is not verification. The
    publisher writes the digest into the filename, so that name *is* the
    recorded expectation for a path-addressed manifest, and the file is
    re-hashed against it here.
    """

    if isinstance(reference, ArtifactRef):
        path = Path(reference.path).expanduser().resolve()
        expected: str | None = reference.sha256
    else:
        path = Path(reference).expanduser().resolve()
        match = _MANIFEST_NAME.match(path.name)
        if match is None:
            raise InvalidSpecError(
                f"cannot verify manifest {path.name}: a path-addressed manifest "
                "must carry its sha256 in its published filename "
                "(manifest-rNNNNNN-<sha256>.json); pass a job id instead",
                stage="compare",
                details={"path": str(path)},
            )
        expected = match.group(1)
    store = ManifestStore(path.parent.parent)
    manifest = store.load(path, expected)
    ref = (
        reference
        if isinstance(reference, ArtifactRef)
        else ArtifactRef.from_path(path, kind=ArtifactKind.MANIFEST)
    )
    return ref, manifest


def _artifact(manifest: RunManifest, logical_name: str) -> tuple[ArtifactRef, dict[str, Any]] | None:
    """One verified JSON artifact, or ``None`` when the run has none."""

    candidates = [
        artifact
        for artifact in manifest.artifacts
        if artifact.logical_name == logical_name
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        raise InvalidSpecError(
            f"run {manifest.run_id} carries {len(candidates)} {logical_name} "
            "artifacts; a comparison cannot choose between them",
            stage="compare",
            details={"logical_name": logical_name, "run_id": str(manifest.run_id)},
        )
    ref = candidates[0]
    verify_artifact(ref)
    try:
        payload = json.loads(ref.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidSpecError(
            f"{logical_name} is not readable JSON",
            stage="compare",
            details={"path": str(ref.path), "reason": str(error)},
        ) from error
    if not isinstance(payload, Mapping):
        raise InvalidSpecError(
            f"{logical_name} JSON root must be an object",
            stage="compare",
            details={"path": str(ref.path)},
        )
    return ref, dict(payload)


def _identity(manifest: RunManifest) -> dict[str, Any]:
    spec = manifest.build_spec
    return {
        "preset": manifest.metadata.get("preset"),
        "family": spec.family.value,
        "target": {
            "name": spec.target.name,
            "chipset": spec.target.chipset,
            "dsp_arch": spec.target.dsp_arch,
            "soc_model": spec.target.soc_model,
        },
        "ars": sorted(int(value) for value in spec.sequence.ars),
        "context_lengths": sorted(
            int(value) for value in spec.sequence.context_lengths
        ),
        "sqnr_modes": sorted(str(mode) for mode in _sqnr_modes(spec)),
    }


def _sqnr_modes(spec: Any) -> Sequence[str]:
    quality = getattr(spec, "quality", None)
    modes = getattr(quality, "sqnr_modes", ()) if quality is not None else ()
    return [str(getattr(mode, "value", mode)) for mode in modes]


def _latency_meter(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {"latency_metric": None, "meter": None, "lane": None}
    block = report.get("device_execution")
    block = block if isinstance(block, Mapping) else {}
    return {
        "latency_metric": report.get("latency_metric"),
        "meter": block.get("meter"),
        "lane": block.get("lane"),
    }


def _per_ar_latency(report: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """``production_latency_us`` and its CV per AR, however the run reported it."""

    if report is None:
        return {}
    results = report.get("results_by_ar")
    if isinstance(results, Mapping) and results:
        collected: dict[str, dict[str, Any]] = {}
        for ar_key, entry in results.items():
            child = entry.get("report") if isinstance(entry, Mapping) else None
            if isinstance(child, Mapping):
                measurement = _single_latency(child)
                if measurement is not None:
                    collected[str(ar_key)] = measurement
        return collected
    measurement = _single_latency(report)
    if measurement is None:
        return {}
    binding = report.get("runtime_binding")
    ar = binding.get("ar") if isinstance(binding, Mapping) else None
    return {str(ar) if ar is not None else "default": measurement}


def _single_latency(report: Mapping[str, Any]) -> dict[str, Any] | None:
    """The device number one AR's report carries, parsed rather than walked."""

    raw = report.get("device_execution")
    if not isinstance(raw, Mapping):
        return None
    block = DeviceExecutionBlock.model_validate(raw)
    if not block.measured:
        return None
    return {
        "production_latency_us": block.production_latency_us,
        "production_latency_source": block.production_latency_source,
        "production_latency_cv_percent": block.production_latency_cv_percent,
        "scope": block.scope,
    }


def _per_tensor_quality(report: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Flatten SQNR observations to ``ar/scope/slice/tensor`` keys."""

    if report is None:
        return {}
    flattened: dict[str, dict[str, Any]] = {}
    results = report.get("results_by_ar")
    if isinstance(results, Mapping) and results:
        for ar_key, entry in results.items():
            child = entry.get("report") if isinstance(entry, Mapping) else None
            if isinstance(child, Mapping):
                for key, value in _per_tensor_quality(child).items():
                    flattened[f"ar{ar_key}/{key}"] = value
        return flattened

    blocks: list[tuple[str, Mapping[str, Any]]] = [("default", report)]
    modes = report.get("mode_reports")
    if isinstance(modes, Mapping):
        blocks.extend(
            (str(name), block)
            for name, block in modes.items()
            if isinstance(block, Mapping)
        )
    for scope, block in blocks:
        observations = block.get("observations")
        if not isinstance(observations, Sequence):
            continue
        for observation in observations:
            if not isinstance(observation, Mapping):
                continue
            slice_id = str(observation.get("slice_id", ""))
            tensor = str(observation.get("tensor_name", ""))
            for mode in ("teacher_forced", "device_chain"):
                quality = observation.get(mode)
                if not isinstance(quality, Mapping):
                    continue
                flattened[f"{scope}/{mode}/{slice_id}/{tensor}"] = {
                    key: (
                        float(quality[key])
                        if isinstance(quality.get(key), (int, float))
                        else None
                    )
                    for key in ("sqnr_db", "rmse", "cosine_similarity")
                }
    return flattened


def _delta(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None:
        return None
    return candidate - baseline


def compare_runs(
    baseline: ArtifactRef | str | Path,
    candidate: ArtifactRef | str | Path,
) -> dict[str, Any]:
    """Compare two verified runs' headline metrics and return the delta.

    Report-only: every number is a difference plus the dispersion it should be
    read against. No threshold is applied and no verdict is produced.
    """

    baseline_ref, baseline_manifest = _load_manifest(baseline)
    candidate_ref, candidate_manifest = _load_manifest(candidate)

    baseline_identity = _identity(baseline_manifest)
    candidate_identity = _identity(candidate_manifest)
    baseline_latency = _artifact(baseline_manifest, "latency_report")
    candidate_latency = _artifact(candidate_manifest, "latency_report")
    baseline_meter = _latency_meter(baseline_latency[1] if baseline_latency else None)
    candidate_meter = _latency_meter(candidate_latency[1] if candidate_latency else None)

    mismatches: list[dict[str, Any]] = []
    for field in _IDENTITY_FIELDS:
        if baseline_identity[field] != candidate_identity[field]:
            mismatches.append(
                {
                    "field": field,
                    "baseline": baseline_identity[field],
                    "candidate": candidate_identity[field],
                }
            )
    for field in ("meter", "lane"):
        if (
            baseline_meter[field] is not None
            and candidate_meter[field] is not None
            and baseline_meter[field] != candidate_meter[field]
        ):
            mismatches.append(
                {
                    "field": f"latency_{field}",
                    "baseline": baseline_meter[field],
                    "candidate": candidate_meter[field],
                }
            )
    if mismatches:
        raise InvalidSpecError(
            "runs are not comparable: "
            + ", ".join(str(item["field"]) for item in mismatches)
            + " differ",
            stage="compare",
            details={"mismatches": mismatches},
        )

    baseline_ars = _per_ar_latency(baseline_latency[1] if baseline_latency else None)
    candidate_ars = _per_ar_latency(candidate_latency[1] if candidate_latency else None)
    latency_rows: list[dict[str, Any]] = []
    for ar_key in sorted(set(baseline_ars) | set(candidate_ars)):
        before = baseline_ars.get(ar_key)
        after = candidate_ars.get(ar_key)
        row: dict[str, Any] = {
            "ar": ar_key,
            "baseline": before,
            "candidate": after,
        }
        if before is None or after is None:
            row["delta_us"] = None
            row["comparable"] = False
            row["reason"] = (
                "the baseline published no device meter for this AR"
                if before is None
                else "the candidate published no device meter for this AR"
            )
            latency_rows.append(row)
            continue
        delta = after["production_latency_us"] - before["production_latency_us"]
        # The dispersion the contract says to read a change against: pool the
        # two runs' published CVs rather than picking one arbitrarily.
        cvs = [
            value["production_latency_cv_percent"]
            for value in (before, after)
            if value["production_latency_cv_percent"] is not None
        ]
        pooled_cv = sum(cvs) / len(cvs) if cvs else None
        noise_us = (
            pooled_cv / 100.0 * before["production_latency_us"]
            if pooled_cv is not None
            else None
        )
        row.update(
            {
                "comparable": True,
                "delta_us": delta,
                "delta_percent": (
                    100.0 * delta / before["production_latency_us"]
                    if before["production_latency_us"]
                    else None
                ),
                "pooled_cv_percent": pooled_cv,
                "delta_in_pooled_cv": (
                    delta / noise_us if noise_us else None
                ),
                "dispersion_note": (
                    "production latency is the most dispersed metric in the "
                    "block; read delta_in_pooled_cv, not delta_us alone"
                ),
            }
        )
        latency_rows.append(row)

    baseline_quality = _artifact(baseline_manifest, "sqnr_report")
    candidate_quality = _artifact(candidate_manifest, "sqnr_report")
    baseline_taps = _per_tensor_quality(baseline_quality[1] if baseline_quality else None)
    candidate_taps = _per_tensor_quality(
        candidate_quality[1] if candidate_quality else None
    )
    quality_rows: list[dict[str, Any]] = []
    for key in sorted(set(baseline_taps) | set(candidate_taps)):
        before = baseline_taps.get(key)
        after = candidate_taps.get(key)
        row = {
            "tap": key,
            "baseline": before,
            "candidate": after,
            "delta_sqnr_db": _delta(
                (before or {}).get("sqnr_db"), (after or {}).get("sqnr_db")
            ),
            "delta_rmse": _delta(
                (before or {}).get("rmse"), (after or {}).get("rmse")
            ),
            "delta_cosine_similarity": _delta(
                (before or {}).get("cosine_similarity"),
                (after or {}).get("cosine_similarity"),
            ),
        }
        quality_rows.append(row)
    # Worst movers first: the largest SQNR drop leads. Taps whose delta could
    # not be computed sort last rather than being dropped.
    quality_rows.sort(
        key=lambda row: (
            row["delta_sqnr_db"] is None,
            row["delta_sqnr_db"] if row["delta_sqnr_db"] is not None else 0.0,
        )
    )

    return {
        "schema": COMPARE_SCHEMA,
        "policy": "report_only",
        "claim_scope": "measured_delta_not_a_verdict",
        "identity": baseline_identity,
        "latency": {
            "meter": baseline_meter,
            "by_ar": latency_rows,
            "covered_ars": [row["ar"] for row in latency_rows if row["comparable"]],
            "uncomparable_ars": [
                row["ar"] for row in latency_rows if not row["comparable"]
            ],
        },
        "quality": {
            "by_tap": quality_rows,
            "tap_count": len(quality_rows),
            "worst_mover": quality_rows[0] if quality_rows else None,
        },
        "provenance": {
            "baseline": _provenance(baseline_ref, baseline_manifest, baseline_latency, baseline_quality),
            "candidate": _provenance(candidate_ref, candidate_manifest, candidate_latency, candidate_quality),
        },
    }


def _provenance(
    ref: ArtifactRef,
    manifest: RunManifest,
    latency: tuple[ArtifactRef, dict[str, Any]] | None,
    quality: tuple[ArtifactRef, dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "run_id": str(manifest.run_id),
        "revision": manifest.revision,
        "manifest_sha256": ref.sha256,
        "manifest_path": str(ref.path),
        # Every report read here was verified against its recorded hash before
        # a single number was taken out of it.
        "evidence_verified": True,
        "latency_report_sha256": latency[0].sha256 if latency else None,
        "sqnr_report_sha256": quality[0].sha256 if quality else None,
    }


__all__ = ["COMPARE_SCHEMA", "compare_runs"]
