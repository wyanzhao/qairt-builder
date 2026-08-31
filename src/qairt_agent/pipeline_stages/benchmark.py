"""The benchmark stage: warmed samples, and the device meter beside them.

Sampling is lane-aware -- the low-level lane keeps 10 warmup and 50 measured
graph invocations, a GenAI sample is a whole ``generate()`` call -- and every
report names its metric in ``latency_metric``, pointing at the
``device_execution`` block or at the string ``unavailable``, never at a host
number.
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
from qairt_agent.pipeline_stages.diagnose import DiagnoseStage
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



class BenchmarkStage:
    """BenchmarkStage — see the module docstring."""

    def benchmark(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        config: Mapping[str, Any] | None = None,
        execution_context: StageExecutionContext | None = None,
    ) -> ToolResult[dict[str, Any]]:
        """Measure warmed wall latency with optional A/A calibration."""

        selected = dict(config or {})

        def operation(
            manifest: RunManifest, adapter: Any, output_dir: Path
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            explicit_chain = (
                selected.get("routes") is not None
                or selected.get("steps") is not None
            )
            explicit_graph = all(
                selected.get(key) is not None
                for key in ("context_path", "graph_name", "vector_manifest")
            )
            explicit_genai = selected.get("container_path") is not None
            automatic_binding = (
                not explicit_chain
                and not explicit_graph
                and not explicit_genai
            )
            requested_ars = tuple(
                int(value) for value in manifest.build_spec.sequence.ars
            )
            if (
                not automatic_binding
                or selected.get("ar") is not None
                or len(requested_ars) <= 1
            ):
                return self._benchmark_one(
                    manifest,
                    adapter,
                    output_dir,
                    selected,
                    manifest_sha256=manifest_sha256,
                )

            runtime_index = self._runtime_index_for_manifest(manifest)
            lane = str(runtime_index.get("lane", ""))
            optrace_requested = bool(
                selected.get(
                    "optrace",
                    manifest.build_spec.benchmark.optrace,
                )
            )
            if lane == "genai_builder":
                if optrace_requested:
                    raise InvalidSpecError(
                        "automatic multi-AR GenAI optrace is unavailable: the "
                        "public generation executor measures end-to-end "
                        "prefill/decode latency, while raw CompiledModel profiling "
                        "selects one AR at a time",
                        stage="benchmark",
                        details={
                            "requested_ars": list(requested_ars),
                            "hint": (
                                "run explicit per-AR raw tensor benchmarks with "
                                "stage_configs.benchmark.ar, or disable optrace "
                                "for executor-managed generation latency"
                            ),
                        },
                    )
                # One public generate() call is the production GenAI wall-time
                # unit. run_one publishes an explicit non-claim about internal
                # AR graph selection rather than pretending it profiled each AR.
                return self._benchmark_one(
                    manifest,
                    adapter,
                    output_dir,
                    selected,
                    manifest_sha256=manifest_sha256,
                )
            if lane != "low_level":
                raise InvalidSpecError(
                    "runtime index has an unsupported benchmark lane",
                    stage="benchmark",
                    details={"lane": lane},
                )

            vectors = runtime_index.get("vectors") or {}
            exact_vectors = (
                vectors.get("validation_manifests_by_ar") or {}
                if isinstance(vectors, Mapping)
                else {}
            )
            missing_vector_ars = [
                ar for ar in requested_ars if not exact_vectors.get(str(ar))
            ]
            if missing_vector_ars:
                raise InvalidSpecError(
                    "automatic multi-AR benchmark requires one exact vector "
                    "manifest for every requested AR",
                    stage="benchmark",
                    details={
                        "requested_ars": list(requested_ars),
                        "missing_vector_ars": missing_vector_ars,
                        "hint": (
                            "build must publish runtime_index.vectors."
                            "validation_manifests_by_ar for every AR"
                        ),
                    },
                )

            results_by_ar: dict[str, Any] = {}
            output_refs: list[ArtifactRef] = []
            per_ar_metrics: dict[str, Any] = {}
            per_ar_optrace: dict[str, tuple[ArtifactRef, dict[str, Any]]] = {}
            for ar in requested_ars:
                ar_key = str(ar)
                payload, refs, metrics = self._benchmark_one(
                    manifest,
                    adapter,
                    output_dir / f"ar{ar}",
                    {**selected, "ar": ar},
                    manifest_sha256=manifest_sha256,
                    report_suffix=f"_ar{ar}",
                )
                report_refs = [
                    ref
                    for ref in refs
                    if ref.logical_name == f"latency_report_ar{ar}"
                ]
                if len(report_refs) != 1:
                    raise InvalidSpecError(
                        "multi-AR benchmark did not publish exactly one "
                        f"AR{ar} latency report",
                        stage="benchmark",
                    )
                results_by_ar[ar_key] = {
                    "report": payload,
                    "report_artifact": _jsonable(report_refs[0]),
                }
                per_ar_metrics[ar_key] = _jsonable(metrics)
                output_refs.extend(refs)
                if optrace_requested:
                    optrace_refs = [
                        ref
                        for ref in refs
                        if ref.logical_name == f"optrace_evidence_ar{ar}"
                    ]
                    if len(optrace_refs) != 1:
                        raise InvalidSpecError(
                            "multi-AR benchmark did not publish exactly one "
                            f"AR{ar} optrace report",
                            stage="benchmark",
                        )
                    verify_artifact(optrace_refs[0])
                    try:
                        optrace_payload = json.loads(
                            optrace_refs[0].path.read_text(encoding="utf-8")
                        )
                    except (
                        OSError,
                        UnicodeError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise InvalidSpecError(
                            f"AR{ar} optrace report is not readable JSON",
                            stage="benchmark",
                        ) from exc
                    if not isinstance(optrace_payload, Mapping):
                        raise InvalidSpecError(
                            f"AR{ar} optrace report must be an object",
                            stage="benchmark",
                        )
                    per_ar_optrace[ar_key] = (
                        optrace_refs[0],
                        dict(optrace_payload),
                    )

            # The aggregate label is derived from what the per-AR captures
            # actually produced. Hardcoding "device_execution" claimed a device
            # meter for the whole set even when an AR degraded to
            # available=false, which is the one thing a latency label must
            # never do.
            unmetered: dict[str, Any] = {}
            for ar_key, entry in results_by_ar.items():
                report = entry.get("report")
                block = (
                    report.get("device_execution")
                    if isinstance(report, Mapping)
                    else None
                )
                if not isinstance(block, Mapping) or block.get("available") is False:
                    unmetered[ar_key] = (
                        block.get("reason")
                        if isinstance(block, Mapping)
                        else "no device_execution block was published"
                    )
            coverage = {
                "mode": "all_requested_ars",
                "requested_ars": list(requested_ars),
                "executed_ars": list(requested_ars),
                "missing_ars": [],
                "complete": True,
                "context_lengths": [
                    int(value)
                    for value in manifest.build_spec.sequence.context_lengths
                ],
                "metered_ars": [
                    int(ar) for ar in requested_ars if str(ar) not in unmetered
                ],
                "unmetered_ars": [
                    int(ar) for ar in requested_ars if str(ar) in unmetered
                ],
                "device_meter_complete": not unmetered,
            }
            if unmetered:
                coverage["unmetered_ar_reasons"] = unmetered
            aggregate_optrace_ref: ArtifactRef | None = None
            if optrace_requested:
                runtime_indexes = [
                    artifact
                    for artifact in manifest.artifacts
                    if artifact.logical_name == "runtime_index"
                ]
                if len(runtime_indexes) != 1:
                    raise InvalidSpecError(
                        "multi-AR optrace requires exactly one runtime_index",
                        stage="benchmark",
                    )
                all_profiles: list[dict[str, Any]] = []
                all_ops: list[dict[str, Any]] = []
                seen_op_ids: set[str] = set()
                device_identifiers: list[Any] = []
                profile_levels: set[str] = set()
                profile_options: set[str] = set()
                for ar_key, (ref, evidence) in per_ar_optrace.items():
                    if evidence.get("schema") != (
                        "qairt-agent.optrace-evidence.v1"
                    ):
                        raise InvalidSpecError(
                            f"AR{ar_key} optrace has an unsupported schema",
                            stage="benchmark",
                        )
                    device_identifiers.append(
                        evidence.get("device_identifier")
                    )
                    profile_levels.add(str(evidence.get("profile_level")))
                    profile_options.add(str(evidence.get("profile_option")))
                    for profile in evidence.get("profiles", ()):
                        if not isinstance(profile, Mapping):
                            raise InvalidSpecError(
                                f"AR{ar_key} optrace profile is invalid",
                                stage="benchmark",
                            )
                        all_profiles.append(
                            {**dict(profile), "ar": int(ar_key)}
                        )
                    for op in evidence.get("ops", ()):
                        if not isinstance(op, Mapping):
                            raise InvalidSpecError(
                                f"AR{ar_key} optrace op is invalid",
                                stage="benchmark",
                            )
                        op_id = str(op.get("op_id", ""))
                        if not op_id or op_id in seen_op_ids:
                            raise InvalidSpecError(
                                "multi-AR optrace requires stable unique op IDs "
                                "across all AR graphs",
                                stage="benchmark",
                                details={
                                    "ar": int(ar_key),
                                    "op_id": op_id or None,
                                },
                            )
                        seen_op_ids.add(op_id)
                        lineage = op.get("lineage", {})
                        all_ops.append(
                            {
                                **dict(op),
                                "lineage": {
                                    **(
                                        dict(lineage)
                                        if isinstance(lineage, Mapping)
                                        else {}
                                    ),
                                    "ar": int(ar_key),
                                },
                            }
                        )
                if len(profile_levels) != 1 or len(profile_options) != 1:
                    raise InvalidSpecError(
                        "multi-AR optrace profile settings are inconsistent",
                        stage="benchmark",
                        details={
                            "profile_levels": sorted(profile_levels),
                            "profile_options": sorted(profile_options),
                        },
                    )
                unique_devices = list(
                    dict.fromkeys(
                        json.dumps(value, sort_keys=True, default=str)
                        for value in device_identifiers
                    )
                )
                device_identity: Any = (
                    device_identifiers[0]
                    if len(unique_devices) == 1
                    else device_identifiers
                )
                aggregate_optrace_ref = atomic_publish_json(
                    output_dir / "optrace_evidence.json",
                    {
                        "schema": "qairt-agent.optrace-evidence.v1",
                        "source_manifest_sha256": manifest_sha256,
                        "runtime_index": _jsonable(runtime_indexes[0]),
                        "runtime_binding": {
                            "lane": "low_level",
                            "family": runtime_index.get("family"),
                            "scope": "multi_ar",
                            "ars": list(requested_ars),
                            "context_lengths": coverage[
                                "context_lengths"
                            ],
                        },
                        "coverage": coverage,
                        "device_identifier": device_identity,
                        "profile_level": next(iter(profile_levels)),
                        "profile_option": next(iter(profile_options)),
                        "profiles": all_profiles,
                        "ops": all_ops,
                        "per_ar_evidence": {
                            ar: _jsonable(ref)
                            for ar, (ref, _) in per_ar_optrace.items()
                        },
                        "profile_scope": (
                            "multi_ar_production_runtime_optrace"
                        ),
                        "claim_scope": (
                            "reported_op_work_not_additive_wall_latency"
                        ),
                    },
                    kind=ArtifactKind.REPORT,
                    logical_name="optrace_evidence",
                )
                output_refs.append(aggregate_optrace_ref)

            aggregate_payload: dict[str, Any] = {
                "schema": "qairt-agent.multi-ar-latency-report.v1",
                "policy": "report_only",
                "latency_metric": (
                    "device_execution" if not unmetered else "partial"
                ),
                "harness_diagnostics": {
                    "metric_name": "host_orchestrated_call_latency",
                    "not_latency": True,
                    "harness_setup_excluded": True,
                },
                "coverage": coverage,
                "results_by_ar": results_by_ar,
            }
            if aggregate_optrace_ref is not None:
                aggregate_payload["optrace_evidence"] = _jsonable(
                    aggregate_optrace_ref
                )
            aggregate_payload = MultiArLatencyReport.model_validate(
                aggregate_payload
            ).to_payload()
            aggregate_ref = atomic_publish_json(
                output_dir / "latency_report.json",
                aggregate_payload,
                kind=ArtifactKind.REPORT,
                logical_name="latency_report",
            )
            output_refs.append(aggregate_ref)
            return aggregate_payload, tuple(output_refs), {
                "policy": "report_only",
                "coverage": coverage,
                "optrace": optrace_requested,
                "results_by_ar": per_ar_metrics,
            }

        return self._continuation_operation(
            "benchmark",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
            execution_context=execution_context,
        )


