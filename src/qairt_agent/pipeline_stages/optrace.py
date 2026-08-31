"""Normalizing QAIRT's per-operator records into reusable evidence.

`option="optrace"` is *not* how per-op cycles are obtained -- it needs a
schematic binary this program's compile does not emit. These records come from
`level="detailed"` alone, and they describe reported op work, which is never
additive wall latency.
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




class OptraceStage:
    """OptraceStage — see the module docstring."""

    @staticmethod
    def _optrace_record_collections(value: Any) -> list[Any]:
        """Find only explicitly named op-record collections."""

        collections: list[Any] = []
        if isinstance(value, Mapping):
            for key in (
                "ops",
                "operations",
                "op_records",
                "operator_records",
                "nodes",
                "events",
            ):
                candidate = value.get(key)
                if isinstance(candidate, Mapping) or (
                    isinstance(candidate, Sequence)
                    and not isinstance(candidate, (str, bytes, bytearray))
                ):
                    collections.append(candidate)
            for key in ("data", "report", "profile", "summary"):
                nested = value.get(key)
                if isinstance(nested, (Mapping, list, tuple)):
                    collections.extend(
                        OptraceStage._optrace_record_collections(nested)
                    )
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for item in value:
                if isinstance(item, (Mapping, list, tuple)):
                    collections.extend(
                        OptraceStage._optrace_record_collections(item)
                    )
        return collections



    @staticmethod
    def _normalize_optrace_records(
        reports: Sequence[Any],
        *,
        slice_id: str,
        graph_name: str,
        step_index: int,
    ) -> list[dict[str, Any]]:
        """Normalize provable cycle records while retaining raw report data."""

        normalized: list[dict[str, Any]] = []
        for report_index, report in enumerate(reports):
            for collection in OptraceStage._optrace_record_collections(report):
                if isinstance(collection, Mapping):
                    items = collection.items()
                else:
                    items = enumerate(collection)
                for record_index, raw in items:
                    fallback_id = str(record_index)
                    if isinstance(raw, (int, float, np.integer, np.floating)):
                        record: Mapping[str, Any] = {
                            "op_id": fallback_id,
                            "cycles": raw,
                        }
                    elif isinstance(raw, Mapping):
                        record = raw
                    else:
                        continue
                    source_op_id = str(
                        record.get("op_id")
                        or record.get("source_id")
                        or record.get("name")
                        or record.get("op_name")
                        or fallback_id
                    )
                    cycle_basis = str(
                        record.get("cycle_basis", "reported")
                    )
                    if record.get("cycles") is not None:
                        cycles = float(record["cycles"])
                    elif record.get("thread_cycles") is not None:
                        thread_value = record["thread_cycles"]
                        if isinstance(thread_value, Mapping):
                            threads = [
                                float(value)
                                for value in thread_value.values()
                            ]
                        elif isinstance(thread_value, Sequence) and not isinstance(
                            thread_value,
                            (str, bytes, bytearray),
                        ):
                            threads = [float(value) for value in thread_value]
                        else:
                            continue
                        if not threads:
                            continue
                        # Threads overlap; a sum would not be wall latency.
                        cycles = max(threads)
                        cycle_basis = "max_thread"
                    else:
                        continue
                    if not np.isfinite(cycles) or cycles < 0.0:
                        raise InvalidSpecError(
                            "QAIRT optrace contained invalid cycle data",
                            stage="benchmark",
                            details={
                                "graph_name": graph_name,
                                "op_id": source_op_id,
                                "cycles": cycles,
                            },
                        )
                    raw_lineage = record.get("lineage", {})
                    if not isinstance(raw_lineage, Mapping):
                        raise InvalidSpecError(
                            "QAIRT optrace lineage must be a mapping",
                            stage="benchmark",
                            details={
                                "graph_name": graph_name,
                                "op_id": source_op_id,
                            },
                        )
                    lineage = dict(raw_lineage)
                    for key in (
                        "layer",
                        "layer_id",
                        "layer_index",
                        "layer_name",
                        "op_id",
                        "op_name",
                        "op_type",
                        "tensor",
                        "tensor_name",
                        "input_tensor",
                        "output_tensor",
                    ):
                        if key in record and key not in lineage:
                            lineage[key] = record[key]
                    lineage.setdefault("slice_id", slice_id)
                    lineage.setdefault("graph_name", graph_name)
                    evidence_id = (
                        f"{slice_id}:{graph_name}:step{step_index}:"
                        f"report{report_index}:record{record_index}:"
                        f"{source_op_id}"
                    )
                    normalized.append(
                        {
                            "op_id": evidence_id,
                            "source_op_id": source_op_id,
                            "report_index": report_index,
                            "record_index": _jsonable(record_index),
                            "cycles": cycles,
                            "cycle_basis": cycle_basis,
                            "critical_path": bool(
                                record.get("critical_path", False)
                            ),
                            "lineage": _jsonable(
                                {
                                    **lineage,
                                    "source_op_id": source_op_id,
                                }
                            ),
                        }
                    )
        unique: dict[str, dict[str, Any]] = {}
        for record in normalized:
            op_id = str(record["op_id"])
            if op_id in unique and unique[op_id] != record:
                raise InvalidSpecError(
                    "QAIRT optrace produced conflicting records for one op",
                    stage="benchmark",
                    details={"op_id": op_id},
                )
            unique[op_id] = record
        return list(unique.values())



    @staticmethod
    def _optrace_profile_signature(
        evidence: Mapping[str, Any],
    ) -> tuple[tuple[str, str, int, int], ...]:
        profiles = evidence.get("profiles", ())
        if not isinstance(profiles, Sequence) or isinstance(
            profiles,
            (str, bytes, bytearray),
        ):
            return ()
        signatures: list[tuple[str, str, int, int]] = []
        for profile in profiles:
            if not isinstance(profile, Mapping):
                return ()
            signatures.append(
                (
                    str(profile.get("slice_id", "")),
                    str(profile.get("graph_name", "")),
                    int(profile.get("step_index", 0)),
                    int(profile.get("normalized_op_count", 0)),
                )
            )
        return tuple(signatures)

