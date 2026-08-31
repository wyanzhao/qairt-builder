"""The diagnose stage: which path the evidence implicates, and why.

Selection is explicit or measured, never heuristic. `kind` runs one path and
fails closed when that path has no evidence; without a kind both run and the
report records what each found. The old selector keyed on "some observation has
nonzero noise", which is the steady state of every healthy quantized run, so the
latency path was unreachable after any validate stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from qairt_agent.artifacts import (
    ManifestStore,
    atomic_publish_json,
    canonical_json_bytes,
    verify_artifact,
)
from qairt_agent.contracts import (
    ArtifactKind,
    ArtifactRef,
    BuildSpec,
    EmbeddingMode,
    ModelFamily,
    PipelineKind,
    QuantizationMode,
    RunManifest,
    SqnrMode,
    StageExecutionContext,
    StageRecord,
    StageStatus,
    ToolResult,
    VectorMode,
    preset_id_for_family,
    utc_now,
)
from qairt_agent.diagnostics.device_metrics import (
    DEVICE_EXECUTION_METER,
    DEVICE_EXECUTION_SCHEMA,
    aggregate_device_executions,
)
from qairt_agent.diagnostics.latency import LatencyDiagnoser
from qairt_agent.diagnostics.sqnr import QualityDiagnoser, compute_tensor_quality
from qairt_agent.device import DeviceRuntime
from qairt_agent.errors import ErrorCode, InvalidSpecError, ToolErrorData
from qairt_agent.errors import ManifestConflictError
from qairt_agent.families import (
    FamilyConfigGenerator,
    FamilyCrossCheck,
    GeneratedFamilyConfig,
    OnnxInspector,
    apply_lane_benchmark_defaults,
    cross_check_declared_family,
    effective_benchmark_policy,
)
from qairt_agent.compare import compare_runs
from qairt_agent.contracts_reports import (
    MultiArLatencyReport,
    MultiArSqnrReport,
)
from qairt_agent.harness import load_harness_constraints, resolve_target
from qairt_agent.qairt_adapter import (
    LIVE_SDK_FIELDS,
    NativeKvGraphExpectation,
    QairtAdapterFactory,
    QairtAdapterProtocol,
    QairtSdkAdapter,
    Qwen35ValidationEvidence,
    Qwen35RuntimeValidationResult,
    require_preflight,
)
from qairt_agent.qairt_adapter.errors import (
    ExperimentalFeatureError,
    QairtAdapterError,
    QairtConfigurationError,
    QairtPreflightError,
    QairtSdkImportError,
)
from qairt_agent.runtime.chain import SliceChainRunner, SliceRoute
from qairt_agent.runtime.index import (
    load_runtime_index,
    make_runtime_index,
    select_runtime_binding,
)
from qairt_agent.vector_retarget import (
    VectorRetargetError,
    retarget_vector_manifest,
    validate_provided_ar_manifest,
)
from qairt_agent.vectors import TensorSource, VectorPreparer, sha256_file

# Shared helpers live in pipeline_support so the stage modules can use
# them without importing this facade. Re-exported here because existing
# call sites and tests import them from `qairt_agent.pipeline`.
from qairt_agent.pipeline_support import (  # noqa: F401
    run_directory,
    DEVICE_EXECUTION_METER,
    DEVICE_EXECUTION_SAMPLES,
    DEVICE_EXECUTION_SCHEMA,
    LIVE_SDK_FIELDS,
    _AUTOMATIC_DIAGNOSE_KEYS,
    _DEPLOYABLE_FOOTPRINT_ROLES,
    _DEVICE_EXECUTION_UNAVAILABLE,
    _EXECUTION_ATTEMPT_METADATA,
    _LIVE_SDK_FIELDS,
    _LOW_LEVEL_CHAIN_CONFIG_FIELDS,
    _OUTPUT_ONLY_CONFIG_FIELDS,
    _SDK_GENERATED_TOKEN_COUNT_KEYS,
    _STATIC_FOOTPRINT_SCHEMA,
    _artifact_kind,
    _config_input_artifacts,
    _jsonable,
    _layer_float_reference,
    _output_mapping,
    _path_artifacts,
    _sdk_generated_token_count,
    _stage_key_value,
    _static_footprint,
    _unique_artifacts,
    hashlib,
    json,
    np,
    os,
    re,
)



class DiagnoseStage:
    """DiagnoseStage — see the module docstring."""

    @staticmethod
    def _lineage_value(
        lineage: Mapping[str, Any],
        keys: Sequence[str],
    ) -> Any | None:
        for key in keys:
            value = lineage.get(key)
            if value is not None and str(value).strip():
                return value
        return None


    @classmethod
    def _quality_divergence_attributions(
        cls,
        report: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Extract only numerical divergences already proven by validate."""

        if report.get("schema") == "qairt-agent.multi-ar-sqnr-report.v1":
            raw_results = report.get("results_by_ar")
            if not isinstance(raw_results, Mapping) or not raw_results:
                raise InvalidSpecError(
                    "multi-AR sqnr_report.results_by_ar must be a non-empty object",
                    stage="diagnose",
                )
            coverage = report.get("coverage")
            if not isinstance(coverage, Mapping):
                raise InvalidSpecError(
                    "multi-AR sqnr_report is missing coverage",
                    stage="diagnose",
                )
            executed = coverage.get("executed_ars")
            if not isinstance(executed, Sequence) or isinstance(
                executed,
                (str, bytes, bytearray),
            ):
                raise InvalidSpecError(
                    "multi-AR sqnr_report coverage.executed_ars must be an array",
                    stage="diagnose",
                )
            attributions: list[dict[str, Any]] = []
            for ar in executed:
                ar_key = str(int(ar))
                result = raw_results.get(ar_key)
                if not isinstance(result, Mapping):
                    raise InvalidSpecError(
                        f"multi-AR sqnr_report is missing AR{ar_key} result",
                        stage="diagnose",
                    )
                child_report = result.get("report")
                if not isinstance(child_report, Mapping):
                    raise InvalidSpecError(
                        f"multi-AR sqnr_report AR{ar_key} report is invalid",
                        stage="diagnose",
                    )
                for attribution in cls._quality_divergence_attributions(
                    child_report
                ):
                    report_scope = str(
                        attribution.get("report_scope", "default")
                    )
                    attributions.append(
                        {
                            **attribution,
                            "ar": int(ar_key),
                            "report_scope": (
                                f"ar{ar_key}/{report_scope}"
                            ),
                        }
                    )
            return attributions

        report_blocks: list[tuple[str, Mapping[str, Any]]] = [
            ("default", report)
        ]
        raw_modes = report.get("mode_reports", {})
        if raw_modes is not None and not isinstance(raw_modes, Mapping):
            raise InvalidSpecError(
                "sqnr_report.mode_reports must be an object",
                stage="diagnose",
            )
        if isinstance(raw_modes, Mapping):
            requested_order = [
                str(value) for value in report.get("executed_modes", ())
            ]
            for mode in requested_order:
                block = raw_modes.get(mode)
                if isinstance(block, Mapping):
                    report_blocks.append((mode, block))
            for mode, block in raw_modes.items():
                if (
                    str(mode) not in requested_order
                    and isinstance(block, Mapping)
                ):
                    report_blocks.append((str(mode), block))

        attributions: list[dict[str, Any]] = []
        for report_scope, block in report_blocks:
            observations = block.get("observations", ())
            if not isinstance(observations, Sequence) or isinstance(
                observations,
                (str, bytes, bytearray),
            ):
                raise InvalidSpecError(
                    "sqnr_report observations must be an array",
                    stage="diagnose",
                    details={"report_scope": report_scope},
                )
            for observation_index, observation in enumerate(observations):
                if not isinstance(observation, Mapping):
                    raise InvalidSpecError(
                        "sqnr_report observation must be an object",
                        stage="diagnose",
                        details={
                            "report_scope": report_scope,
                            "observation_index": observation_index,
                        },
                    )
                raw_lineage = observation.get("lineage", {})
                if not isinstance(raw_lineage, Mapping):
                    raise InvalidSpecError(
                        "sqnr_report observation lineage must be an object",
                        stage="diagnose",
                        details={
                            "report_scope": report_scope,
                            "observation_index": observation_index,
                        },
                    )
                divergent_modes: dict[str, Any] = {}
                for mode in ("teacher_forced", "device_chain"):
                    quality = observation.get(mode)
                    if quality is None:
                        continue
                    if not isinstance(quality, Mapping):
                        raise InvalidSpecError(
                            f"sqnr_report {mode} quality must be an object",
                            stage="diagnose",
                        )
                    raw_noise = quality.get("noise_energy")
                    if raw_noise is None:
                        continue
                    noise_energy = float(raw_noise)
                    if not np.isfinite(noise_energy) or noise_energy < 0.0:
                        raise InvalidSpecError(
                            "sqnr_report contains invalid noise energy",
                            stage="diagnose",
                            details={
                                "report_scope": report_scope,
                                "mode": mode,
                                "noise_energy": noise_energy,
                            },
                        )
                    if noise_energy > 0.0:
                        divergent_modes[mode] = dict(quality)
                if not divergent_modes:
                    continue
                lineage = dict(raw_lineage)
                layer = cls._lineage_value(
                    lineage,
                    (
                        "layer_name",
                        "layer_id",
                        "layer_index",
                        "layer",
                    ),
                )
                op = cls._lineage_value(
                    lineage,
                    (
                        "op_name",
                        "op_id",
                        "source_op_id",
                        "op_type",
                    ),
                )
                attributions.append(
                    {
                        "report_scope": report_scope,
                        "observation_index": observation_index,
                        "slice_id": str(observation.get("slice_id", "")),
                        "tensor_name": str(
                            observation.get("tensor_name", "")
                        ),
                        "divergent_modes": divergent_modes,
                        "attribution": observation.get("attribution"),
                        "propagated_noise_delta": observation.get(
                            "propagated_noise_delta"
                        ),
                        "layer": _jsonable(layer),
                        "op": _jsonable(op),
                        "lineage": _jsonable(lineage),
                        "claim_scope": (
                            "first_observed_divergence_not_root_cause"
                        ),
                    }
                )
        has_reported_first_error = any(
            block.get(field) is not None
            for _, block in report_blocks
            for field in ("first_teacher_error", "first_chain_error")
        )
        if has_reported_first_error and not attributions:
            raise InvalidSpecError(
                "sqnr_report declares a first error without a corresponding "
                "positive-noise observation",
                stage="diagnose",
            )
        return attributions


    @classmethod
    def _optrace_baseline_from_history(
        cls,
        manifest: RunManifest,
        candidate_ref: ArtifactRef,
        candidate: Mapping[str, Any],
    ) -> tuple[
        tuple[ArtifactRef, ArtifactRef, dict[str, Any]] | None,
        list[dict[str, Any]],
    ]:
        """Find the nearest compatible immutable profile in parent/fork history."""

        queue: list[ArtifactRef] = []
        if manifest.parent_manifest is not None:
            queue.append(manifest.parent_manifest)

        def add_fork_ref(source: RunManifest) -> None:
            raw = source.metadata.get("forked_from_manifest")
            if raw is None:
                return
            try:
                ref = ArtifactRef.model_validate(raw)
            except ValidationError as exc:
                raise InvalidSpecError(
                    "forked_from_manifest metadata is invalid",
                    stage="diagnose",
                    details={
                        "validation_errors": exc.errors(
                            include_url=False
                        )
                    },
                ) from exc
            if ref.kind is not ArtifactKind.MANIFEST:
                raise InvalidSpecError(
                    "forked_from_manifest must reference a manifest",
                    stage="diagnose",
                )
            queue.append(ref)

        add_fork_ref(manifest)
        seen: set[tuple[Path, str]] = set()
        rejected: list[dict[str, Any]] = []
        while queue:
            manifest_ref = queue.pop(0)
            key = (
                manifest_ref.path.expanduser().resolve(),
                manifest_ref.sha256,
            )
            if key in seen:
                continue
            seen.add(key)
            historical = cls._store_for_manifest(
                manifest_ref.path
            ).load(manifest_ref)
            candidates = [
                artifact
                for artifact in historical.artifacts
                if artifact.logical_name == "optrace_evidence"
            ]
            if len(candidates) > 1:
                raise InvalidSpecError(
                    "historical manifest has ambiguous optrace evidence",
                    stage="diagnose",
                    details={
                        "manifest": _jsonable(manifest_ref),
                        "artifact_count": len(candidates),
                    },
                )
            if candidates and candidates[0].sha256 != candidate_ref.sha256:
                evidence_ref, evidence = cls._manifest_artifact_payload(
                    historical,
                    "optrace_evidence",
                    stage="diagnose",
                )
                mismatches = cls._optrace_compatibility_mismatches(
                    evidence,
                    candidate,
                )
                if not mismatches:
                    return (
                        (manifest_ref, evidence_ref, evidence),
                        rejected,
                    )
                rejected.append(
                    {
                        "manifest": _jsonable(manifest_ref),
                        "optrace_evidence": _jsonable(evidence_ref),
                        "mismatches": mismatches,
                    }
                )
            if historical.parent_manifest is not None:
                queue.append(historical.parent_manifest)
            add_fork_ref(historical)
        return None, rejected


    @classmethod
    def _latency_dimension_attributions(
        cls,
        attributions: Sequence[Mapping[str, Any]],
        *,
        score_field: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        """Choose representative ops per lineage dimension without summing."""

        layer_records: dict[str, dict[str, Any]] = {}
        tensor_records: dict[str, dict[str, Any]] = {}
        for attribution in attributions:
            raw_lineage = attribution.get("lineage", {})
            if not isinstance(raw_lineage, Mapping):
                continue
            layer = cls._lineage_value(
                raw_lineage,
                ("layer_name", "layer_id", "layer_index", "layer"),
            )
            tensor = cls._lineage_value(
                raw_lineage,
                (
                    "tensor_name",
                    "output_tensor",
                    "tensor",
                    "input_tensor",
                ),
            )
            raw_score = attribution.get(score_field)
            score = (
                float(raw_score)
                if raw_score is not None
                else float("-inf")
            )
            summary = {
                "representative_op_id": attribution.get("op_id"),
                "candidate_cycles": attribution.get("candidate_cycles"),
                "delta_cycles": attribution.get("delta_cycles"),
                "selection": (
                    f"maximum_{score_field}; op cycles are not summed"
                ),
            }
            for label, records, field in (
                (layer, layer_records, "layer"),
                (tensor, tensor_records, "tensor"),
            ):
                if label is None:
                    continue
                key = str(label)
                prior = records.get(key)
                if prior is None or score > float(prior["_score"]):
                    records[key] = {
                        field: _jsonable(label),
                        **summary,
                        "_score": score,
                    }

        def finalized(
            records: Mapping[str, Mapping[str, Any]],
        ) -> list[dict[str, Any]]:
            ordered = sorted(
                records.values(),
                key=lambda item: float(item["_score"]),
                reverse=True,
            )
            return [
                {
                    key: value
                    for key, value in item.items()
                    if key != "_score"
                }
                for item in ordered
            ]

        layers = finalized(layer_records)
        tensors = finalized(tensor_records)
        unavailable = []
        if not layers:
            unavailable.append("layer")
        if not tensors:
            unavailable.append("tensor")
        return layers, tensors, unavailable


    def _automatic_diagnosis(
        self,
        manifest: RunManifest,
        output_dir: Path,
        *,
        kind: str | None = None,
        baseline: str | Path | None = None,
        current: tuple[str | Path, str] | None = None,
    ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
        """Diagnose from verified published evidence, honouring an explicit kind.

        ``kind`` selects one path and fails closed when that path has no
        evidence -- previously a requested ``latency`` diagnosis silently ran
        the quality path instead. Without a kind both paths run and the report
        says what each found: selecting quality on "some observation has
        nonzero noise" made the latency path unreachable, because nonzero noise
        is the steady state of every healthy quantized run, not a regression
        signal. Nonzero noise remains what the quality path *attributes*; it is
        no longer what selects it.
        """

        runtime_ref, _runtime_payload = self._manifest_artifact_payload(
            manifest,
            "runtime_index",
            stage="diagnose",
        )
        # Schema/semantic validation is stricter than generic JSON parsing.
        load_runtime_index(runtime_ref.path)

        comparison: dict[str, Any] | None = None
        implicated: dict[str, Any] | None = None
        if baseline is not None:
            if current is None:
                raise InvalidSpecError(
                    "a baseline comparison needs this run's own manifest",
                    stage="diagnose",
                )
            comparison = compare_runs(Path(baseline), Path(current[0]))
            implicated = self._implicated_paths(comparison)

        quality = (
            None
            if kind == "latency"
            else self._automatic_quality_diagnosis(
                manifest, output_dir, runtime_ref
            )
        )
        latency = (
            None
            if kind == "quality"
            else self._automatic_latency_diagnosis(
                manifest, output_dir, runtime_ref
            )
        )
        considered = {
            "quality": (
                "skipped_by_kind"
                if kind == "latency"
                else ("found" if quality is not None else "no_evidence")
            ),
            "latency": (
                "skipped_by_kind"
                if kind == "quality"
                else ("found" if latency is not None else "no_evidence")
            ),
        }

        if kind == "quality" and quality is None:
            raise InvalidSpecError(
                "diagnose kind 'quality' found no provable quality divergence "
                "in the published sqnr_report",
                stage="diagnose",
                details={"considered": considered},
            )
        if kind == "latency" and latency is None:
            raise InvalidSpecError(
                "diagnose kind 'latency' found no benchmark optrace evidence",
                stage="diagnose",
                details={
                    "considered": considered,
                    "hint": (
                        "run benchmark with benchmark.optrace=true or provide "
                        "explicit diagnosis traces"
                    ),
                },
            )
        if quality is None and latency is None:
            raise InvalidSpecError(
                "automatic diagnose found no provable quality divergence and "
                "no benchmark optrace evidence",
                stage="diagnose",
                details={
                    "considered": considered,
                    "hint": (
                        "run benchmark with benchmark.optrace=true or provide "
                        "explicit diagnosis traces"
                    ),
                },
            )
        if quality is not None and latency is not None:
            quality_payload, quality_refs, quality_metrics = quality
            latency_payload, latency_refs, latency_metrics = latency
            payload = {
                "schema": "qairt-agent.automatic-diagnosis.v1",
                "diagnosis_kind": "quality_and_latency",
                "selection_reason": "both_paths_had_verified_evidence",
                "considered": considered,
                "quality": quality_payload,
                "latency": latency_payload,
                "policy": "report_only",
            }
            report_ref = atomic_publish_json(
                output_dir / "automatic_diagnosis.json",
                payload,
                kind=ArtifactKind.REPORT,
                logical_name="automatic_diagnosis",
            )
            return payload, (*quality_refs, *latency_refs, report_ref), {
                "diagnosis_kind": "quality_and_latency",
                "quality": quality_metrics,
                "latency": latency_metrics,
                "policy": "report_only",
            }

        selected = quality if quality is not None else latency
        assert selected is not None
        payload, refs, metrics = selected
        enriched = {**payload, "considered": considered}
        if comparison is not None:
            enriched["comparison"] = comparison
            enriched["implicated"] = implicated
        return enriched, refs, metrics


    @staticmethod
    def _implicated_paths(comparison: Mapping[str, Any]) -> dict[str, Any]:
        """Which path a measured delta points at, with the rule stated.

        This is a routing hint for where to drill, not a verdict on the model:
        both paths still run and both reports are published. It replaces the
        old selector, which keyed on "some observation has nonzero noise" --
        true of every healthy quantized run, so it never pointed anywhere.

        Latency is read against its own published dispersion, as the program
        contract requires; quality is read as any tap whose SQNR fell.
        """

        latency_rows = comparison.get("latency", {}).get("by_ar", [])
        moved_ars = [
            {
                "ar": row.get("ar"),
                "delta_us": row.get("delta_us"),
                "delta_in_pooled_cv": row.get("delta_in_pooled_cv"),
            }
            for row in latency_rows
            if isinstance(row, Mapping)
            and isinstance(row.get("delta_in_pooled_cv"), (int, float))
            and abs(float(row["delta_in_pooled_cv"])) >= 1.0
        ]
        quality_rows = comparison.get("quality", {}).get("by_tap", [])
        dropped_taps = [
            {"tap": row.get("tap"), "delta_sqnr_db": row.get("delta_sqnr_db")}
            for row in quality_rows
            if isinstance(row, Mapping)
            and isinstance(row.get("delta_sqnr_db"), (int, float))
            and float(row["delta_sqnr_db"]) < 0.0
        ]
        return {
            "quality": bool(dropped_taps),
            "latency": bool(moved_ars),
            "quality_evidence": dropped_taps[:5],
            "latency_evidence": moved_ars,
            "rule": (
                "quality is implicated by any tap whose SQNR fell; latency by "
                "any AR whose production latency moved at least one pooled CV. "
                "Both paths are reported regardless -- this only says where the "
                "measured change points."
            ),
        }


    def _automatic_quality_diagnosis(
        self,
        manifest: RunManifest,
        output_dir: Path,
        runtime_ref: ArtifactRef,
    ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]] | None:
        """Quality attribution, or ``None`` when validate proved no divergence."""

        sqnr_refs = [
            artifact
            for artifact in manifest.artifacts
            if artifact.logical_name == "sqnr_report"
        ]
        if len(sqnr_refs) > 1:
            raise InvalidSpecError(
                "automatic diagnose found ambiguous sqnr_report artifacts",
                stage="diagnose",
            )
        if sqnr_refs:
            sqnr_ref, sqnr_report = self._manifest_artifact_payload(
                manifest,
                "sqnr_report",
                stage="diagnose",
            )
            quality_attributions = (
                self._quality_divergence_attributions(sqnr_report)
            )
            if quality_attributions:
                layer_attributions = [
                    {
                        "layer": item["layer"],
                        "slice_id": item["slice_id"],
                        "tensor_name": item["tensor_name"],
                        "op": item["op"],
                        "report_scope": item["report_scope"],
                        **(
                            {"ar": item["ar"]}
                            if item.get("ar") is not None
                            else {}
                        ),
                    }
                    for item in quality_attributions
                    if item["layer"] is not None
                ]
                op_attributions = [
                    {
                        "op": item["op"],
                        "layer": item["layer"],
                        "slice_id": item["slice_id"],
                        "tensor_name": item["tensor_name"],
                        "report_scope": item["report_scope"],
                        **(
                            {"ar": item["ar"]}
                            if item.get("ar") is not None
                            else {}
                        ),
                    }
                    for item in quality_attributions
                    if item["op"] is not None
                ]
                if op_attributions and len(op_attributions) == len(
                    quality_attributions
                ):
                    attribution_scope = "slice_tensor_layer_op"
                elif op_attributions:
                    attribution_scope = "mixed_slice_tensor_and_op_lineage"
                else:
                    attribution_scope = "slice_tensor_only"
                limitations: list[str] = []
                if not layer_attributions:
                    limitations.append(
                        "validate published no explicit layer lineage"
                    )
                if not op_attributions:
                    limitations.append(
                        "validate published no operator intermediate trace; "
                        "operator-level root cause is not provable"
                    )
                diagnostic_evidence = sqnr_report.get(
                    "diagnostic_evidence"
                )
                if (
                    isinstance(diagnostic_evidence, Mapping)
                    and diagnostic_evidence.get("op_level_dump_available")
                    and not op_attributions
                ):
                    limitations.append(
                        "diagnostic contexts are available, but this validate "
                        "artifact contains no executed per-op tensor dump"
                    )
                payload = {
                    "schema": "qairt-agent.automatic-quality-diagnosis.v1",
                    "diagnosis_kind": "quality",
                    "selection_reason": (
                        "validated_sqnr_positive_noise_observation"
                    ),
                    "attribution_scope": attribution_scope,
                    "first_observed": quality_attributions[0],
                    "attributions": quality_attributions,
                    "layer_attributions": layer_attributions,
                    "tensor_attributions": [
                        {
                            "slice_id": item["slice_id"],
                            "tensor_name": item["tensor_name"],
                            "report_scope": item["report_scope"],
                            "divergent_modes": item["divergent_modes"],
                            **(
                                {"ar": item["ar"]}
                                if item.get("ar") is not None
                                else {}
                            ),
                        }
                        for item in quality_attributions
                    ],
                    "op_attributions": op_attributions,
                    "op_attribution_supported": bool(op_attributions),
                    "limitations": limitations,
                    "sources": {
                        "runtime_index": _jsonable(runtime_ref),
                        "sqnr_report": _jsonable(sqnr_ref),
                        "reference_source": sqnr_report.get(
                            "reference_source"
                        ),
                        "slice_reference_evidence": sqnr_report.get(
                            "slice_reference_evidence", []
                        ),
                        "diagnostic_evidence": diagnostic_evidence,
                    },
                    "policy": "report_only",
                    "claim_scope": (
                        "first_observed_divergence_not_root_cause"
                    ),
                }
                report_ref = atomic_publish_json(
                    output_dir / "quality_diagnosis.json",
                    payload,
                    kind=ArtifactKind.REPORT,
                    logical_name="quality_diagnosis",
                )
                return payload, (report_ref,), {
                    "diagnosis_kind": "quality",
                    "attribution_count": len(quality_attributions),
                    "op_attribution_supported": bool(op_attributions),
                    "policy": "report_only",
                }
        return None


    def _automatic_latency_diagnosis(
        self,
        manifest: RunManifest,
        output_dir: Path,
        runtime_ref: ArtifactRef,
    ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]] | None:
        """Op-cycle attribution, or ``None`` when no optrace evidence exists."""

        optrace_refs = [
            artifact
            for artifact in manifest.artifacts
            if artifact.logical_name == "optrace_evidence"
        ]
        if len(optrace_refs) > 1:
            raise InvalidSpecError(
                "automatic diagnose found ambiguous optrace evidence",
                stage="diagnose",
            )
        if not optrace_refs:
            # Absence is not an error here: the caller decides whether a
            # missing path is fatal, based on the requested kind.
            return None

        optrace_ref, optrace = self._manifest_artifact_payload(
            manifest,
            "optrace_evidence",
            stage="diagnose",
        )
        if optrace.get("schema") != "qairt-agent.optrace-evidence.v1":
            raise InvalidSpecError(
                "unsupported optrace evidence schema",
                stage="diagnose",
                details={"schema": optrace.get("schema")},
            )
        embedded_runtime = optrace.get("runtime_index")
        if (
            not isinstance(embedded_runtime, Mapping)
            or embedded_runtime.get("sha256") != runtime_ref.sha256
        ):
            raise InvalidSpecError(
                "optrace evidence is not bound to the current runtime_index",
                stage="diagnose",
            )
        candidate_ops = optrace.get("ops")
        if not isinstance(candidate_ops, Sequence) or isinstance(
            candidate_ops,
            (str, bytes, bytearray),
        ) or not candidate_ops:
            raise InvalidSpecError(
                "optrace evidence contains no reusable per-op records",
                stage="diagnose",
            )

        latency_ref, latency_report = self._manifest_artifact_payload(
            manifest,
            "latency_report",
            stage="diagnose",
        )
        embedded_optrace = latency_report.get("optrace_evidence")
        if (
            not isinstance(embedded_optrace, Mapping)
            or embedded_optrace.get("sha256") != optrace_ref.sha256
        ):
            raise InvalidSpecError(
                "latency_report is not bound to current optrace evidence",
                stage="diagnose",
            )

        historical, rejected_baselines = (
            self._optrace_baseline_from_history(
                manifest,
                optrace_ref,
                optrace,
            )
        )
        baseline_source: dict[str, Any] | None = None
        limitations = [
            "op cycles represent reported work and are not additive wall latency"
        ]
        if historical is not None:
            historical_manifest_ref, baseline_ref, baseline = historical
            baseline_ops = baseline.get("ops")
            if not isinstance(baseline_ops, Sequence) or isinstance(
                baseline_ops,
                (str, bytes, bytearray),
            ):
                raise InvalidSpecError(
                    "historical optrace evidence has invalid per-op records",
                    stage="diagnose",
                )
            attributed = LatencyDiagnoser.attribute_ops(
                baseline_ops,
                candidate_ops,
            )
            matched = [item for item in attributed if item.status == "matched"]
            if matched:
                comparison_mode = "parent_profile_delta"
                regression_attribution_supported = True
                score_field = "delta_cycles"
                baseline_source = {
                    "manifest": _jsonable(historical_manifest_ref),
                    "optrace_evidence": _jsonable(baseline_ref),
                }
            else:
                comparison_mode = "candidate_hotspot_only"
                regression_attribution_supported = False
                score_field = "candidate_cycles"
                limitations.append(
                    "compatible historical profile had no stable matching op "
                    "IDs; no latency delta is claimed"
                )
                attributed = LatencyDiagnoser.attribute_ops(
                    (),
                    candidate_ops,
                )
        else:
            comparison_mode = "candidate_hotspot_only"
            regression_attribution_supported = False
            score_field = "candidate_cycles"
            limitations.append(
                "no compatible parent/rerun profile baseline was found; "
                "candidate hotspots are not latency regressions"
            )
            attributed = LatencyDiagnoser.attribute_ops((), candidate_ops)

        attribution_payloads = [
            _jsonable(item) for item in attributed
        ]
        attribution_payloads.sort(
            key=lambda item: (
                float(item.get(score_field))
                if item.get(score_field) is not None
                else float("-inf")
            ),
            reverse=True,
        )
        layer_attributions, tensor_attributions, unavailable = (
            self._latency_dimension_attributions(
                attribution_payloads,
                score_field=score_field,
            )
        )
        if unavailable:
            limitations.append(
                "optrace published no explicit "
                + "/".join(unavailable)
                + " lineage for those attribution dimensions"
            )
        positive_regressions = [
            item
            for item in attribution_payloads
            if item.get("delta_cycles") is not None
            and float(item["delta_cycles"]) > 0.0
        ]
        payload = {
            "schema": "qairt-agent.automatic-latency-diagnosis.v1",
            "diagnosis_kind": "latency",
            "selection_reason": "benchmark_optrace_evidence_available",
            "comparison_mode": comparison_mode,
            "regression_attribution_supported": (
                regression_attribution_supported
            ),
            "attributions": attribution_payloads,
            "op_attributions": attribution_payloads,
            "layer_attributions": layer_attributions,
            "tensor_attributions": tensor_attributions,
            "unavailable_dimensions": unavailable,
            "positive_regression_count": len(positive_regressions),
            "first_problem": (
                positive_regressions[0]
                if positive_regressions
                else (
                    attribution_payloads[0]
                    if attribution_payloads
                    else None
                )
            ),
            "limitations": limitations,
            "sources": {
                "runtime_index": _jsonable(runtime_ref),
                "latency_report": _jsonable(latency_ref),
                "candidate_optrace_evidence": _jsonable(optrace_ref),
                "baseline": baseline_source,
                "rejected_baselines": rejected_baselines,
            },
            "policy": "report_only",
            "claim_scope": "op_work_not_additive_wall_latency",
        }
        report_ref = atomic_publish_json(
            output_dir / "latency_diagnosis.json",
            payload,
            kind=ArtifactKind.REPORT,
            logical_name="latency_diagnosis",
        )
        return payload, (report_ref,), {
            "diagnosis_kind": "latency",
            "comparison_mode": comparison_mode,
            "attribution_count": len(attribution_payloads),
            "positive_regression_count": len(positive_regressions),
            "policy": "report_only",
        }


    def diagnose_quality(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        config: Mapping[str, Any] | None = None,
        execution_context: StageExecutionContext | None = None,
    ) -> ToolResult[dict[str, Any]]:
        """Localize the first observable numerical drop to a tap and lineage."""

        selected = dict(config or {})

        def operation(
            manifest: RunManifest, _adapter: Any, output_dir: Path
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            if not selected or set(selected) <= _AUTOMATIC_DIAGNOSE_KEYS:
                return self._automatic_diagnosis(
                    manifest,
                    output_dir,
                    kind=selected.get("kind"),
                    baseline=selected.get("baseline_manifest"),
                    current=(manifest_uri, manifest_sha256),
                )
            if "reference_trace" not in selected or "actual_trace" not in selected:
                raise ValueError(
                    "diagnose_quality requires reference_trace and actual_trace"
                )
            references = self._tensor_mapping(
                selected["reference_trace"],
                section=str(selected.get("reference_section", "goldens")),
            )
            actuals = self._tensor_mapping(
                selected["actual_trace"],
                section=str(selected.get("actual_section", "inputs")),
            )
            report = QualityDiagnoser(
                reference_energy_floor=float(selected.get("reference_energy_floor", 0.0))
            ).diagnose_trace(
                references,
                actuals,
                lineage=selected.get("lineage"),
                order=selected.get("order"),
            )
            payload = report.to_dict()
            payload["policy"] = "report_only"
            report_ref = atomic_publish_json(
                output_dir / "quality_diagnosis.json",
                payload,
                kind=ArtifactKind.REPORT,
                logical_name="quality_diagnosis",
            )
            return payload, (report_ref,), {
                "first_observed_error": report.first_observed_error,
                "tap_count": len(report.observations),
            }

        return self._continuation_operation(
            "diagnose_quality",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
            execution_context=execution_context,
        )


    def diagnose_latency(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        config: Mapping[str, Any] | None = None,
        execution_context: StageExecutionContext | None = None,
    ) -> ToolResult[dict[str, Any]]:
        """Attribute op-cycle changes without treating their sum as wall time."""

        selected = dict(config or {})

        def operation(
            manifest: RunManifest, _adapter: Any, output_dir: Path
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            if not selected or set(selected) <= _AUTOMATIC_DIAGNOSE_KEYS:
                # An explicitly requested latency diagnosis runs the latency
                # path. It used to share the quality-first automatic selector,
                # so a manifest carrying any SQNR observation -- every healthy
                # quantized run -- silently produced a quality report instead.
                return self._automatic_diagnosis(
                    manifest,
                    output_dir,
                    kind=str(selected.get("kind", "latency")),
                    baseline=selected.get("baseline_manifest"),
                    current=(manifest_uri, manifest_sha256),
                )
            if "baseline_ops" not in selected or "candidate_ops" not in selected:
                raise ValueError(
                    "diagnose_latency requires baseline_ops and candidate_ops "
                    "from QAIRT detailed/optrace reports"
                )
            attributions = LatencyDiagnoser.attribute_ops(
                selected["baseline_ops"],
                selected["candidate_ops"],
            )
            payload = {
                "attributions": [_jsonable(item) for item in attributions],
                "policy": "report_only",
                "claim_scope": "op_work_not_additive_wall_latency",
            }
            report_ref = atomic_publish_json(
                output_dir / "latency_diagnosis.json",
                payload,
                kind=ArtifactKind.REPORT,
                logical_name="latency_diagnosis",
            )
            regressions = [
                item
                for item in attributions
                if item.delta_cycles is not None and item.delta_cycles > 0
            ]
            return payload, (report_ref,), {
                "attribution_count": len(attributions),
                "positive_regression_count": len(regressions),
            }

        return self._continuation_operation(
            "diagnose_latency",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
            execution_context=execution_context,
        )
