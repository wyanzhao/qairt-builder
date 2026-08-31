"""The Qwen3.5 multi-AR derivation check.

Deriving several ARs from one Qwen3.5 export is fail-closed and experimental:
production supplies independent per-AR exports. When it is used, the adapter
mints evidence only after AR rewrite, state IO, MHA2SHA and initializer
compatibility all pass, and this runs the device-backed standalone, joint and
golden comparisons that evidence is contingent on.
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
from qairt_agent.pipeline_stages.benchmark import BenchmarkStage
from qairt_agent.pipeline_stages.diagnose import DiagnoseStage
from qairt_agent.pipeline_stages.execution import ExecutionStage
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




class Qwen35DerivationStage:
    """Qwen35DerivationStage — see the module docstring."""

    def _qwen35_validator(
        self,
        adapter: Any,
        validation_config: Mapping[str, Any],
        report_root: Path,
        *,
        manifest: RunManifest,
        input_manifest_sha256: str,
    ) -> tuple[
        Callable[[Any], Qwen35RuntimeValidationResult],
        tuple[ArtifactRef, ...],
    ]:
        """Create the device-backed validator used to mint Qwen3.5 evidence.

        Validation cases are keyed by graph name or AR and each point to a
        content-verified vector manifest containing graph inputs and goldens.
        The boolean result is computed here; callers cannot attest it.
        """

        cases = validation_config.get("cases")
        if not isinstance(cases, Mapping):
            raise InvalidSpecError(
                "metadata.qwen35_runtime_validation.cases must map graph name or AR "
                "to a vector-manifest descriptor",
                stage="build",
            )
        rtol = float(validation_config.get("rtol", 0.0))
        atol = float(validation_config.get("atol", 0.0))
        if rtol < 0.0 or atol < 0.0:
            raise InvalidSpecError("Qwen3.5 validation tolerances cannot be negative", stage="build")
        native_io = bool(validation_config.get("native_io", False))
        execution_options = self._execution_options(validation_config)
        case_refs: list[ArtifactRef] = []
        seen_case_paths: set[Path] = set()

        def collect_case_refs(value: Any) -> None:
            if isinstance(value, (str, Path)):
                path = Path(value).expanduser().resolve()
                expected = None
            elif isinstance(value, Mapping) and value.get("manifest_path") is not None:
                path = Path(str(value["manifest_path"])).expanduser().resolve()
                expected = (
                    str(value["manifest_sha256"])
                    if value.get("manifest_sha256") is not None
                    else None
                )
            elif isinstance(value, Mapping):
                for nested in value.values():
                    collect_case_refs(nested)
                return
            else:
                raise InvalidSpecError(
                    "Qwen3.5 cases must contain vector-manifest paths or descriptors",
                    stage="build",
                )
            VectorPreparer.load_manifest(path, expected_sha256=expected)
            if path not in seen_case_paths:
                seen_case_paths.add(path)
                case_refs.append(
                    ArtifactRef.from_path(
                        path,
                        kind=ArtifactKind.GOLDEN_VECTORS,
                        logical_name=f"qwen35_golden_{len(case_refs):03d}",
                    )
                )

        collect_case_refs(cases)

        def case_for(
            slice_name: str,
            graph_name: str,
            ar: int,
        ) -> tuple[Path, str | None]:
            slice_cases = cases.get(slice_name, {})
            if not isinstance(slice_cases, Mapping):
                slice_cases = {}
            raw = cases.get(
                graph_name,
                slice_cases.get(
                    graph_name,
                    slice_cases.get(str(ar), slice_cases.get(ar)),
                ),
            )
            if raw is None:
                raw = cases.get(str(ar), cases.get(ar))
            if raw is None:
                raise ExperimentalFeatureError(
                    f"no Qwen3.5 runtime validation case for {graph_name} / AR{ar}"
                )
            if isinstance(raw, (str, Path)):
                return Path(raw), None
            if not isinstance(raw, Mapping) or raw.get("manifest_path") is None:
                raise InvalidSpecError(
                    f"Qwen3.5 case {graph_name!r} must provide manifest_path",
                    stage="build",
                )
            return Path(str(raw["manifest_path"])), (
                str(raw["manifest_sha256"])
                if raw.get("manifest_sha256") is not None
                else None
            )

        def compare(
            expected: Mapping[str, np.ndarray],
            actual: Mapping[str, np.ndarray],
            *,
            label: str,
        ) -> tuple[bool, dict[str, Any]]:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            if missing:
                return False, {"label": label, "missing": missing, "extra": extra}
            observations: dict[str, Any] = {}
            passed = not extra
            diagnoser = QualityDiagnoser()
            for name, reference in expected.items():
                candidate = np.asarray(actual[name])
                reference_array = np.asarray(reference)
                if reference_array.shape != candidate.shape:
                    passed = False
                    observations[name] = {
                        "shape_match": False,
                        "reference_shape": list(reference_array.shape),
                        "actual_shape": list(candidate.shape),
                    }
                    continue
                finite = bool(
                    np.all(np.isfinite(reference_array))
                    and np.all(np.isfinite(candidate))
                )
                equivalent = finite and bool(
                    np.allclose(
                        reference_array,
                        candidate,
                        rtol=rtol,
                        atol=atol,
                        equal_nan=False,
                    )
                )
                passed = passed and equivalent
                quality = diagnoser.compare(reference_array, candidate)
                observations[name] = {
                    "shape_match": True,
                    "finite": finite,
                    "equivalent": equivalent,
                    "quality": quality.to_dict(),
                }
            return passed, {
                "label": label,
                "missing": missing,
                "extra": extra,
                "tensors": observations,
            }

        def validate(request: Any) -> Qwen35RuntimeValidationResult:
            standalone_ok = True
            joint_ok = True
            pair_ok = True
            executed: list[str] = []
            golden_ids: list[str] = []
            graph_reports: list[dict[str, Any]] = []
            resolved_cases = [
                (
                    int(ar),
                    str(graph_name),
                    standalone_context,
                    *case_for(request.slice_name, graph_name, ar),
                )
                for ar, graph_name, standalone_context in zip(
                    request.ar_values,
                    request.graph_names,
                    request.standalone_contexts,
                )
            ]
            push_files = self._device_stage_files(
                report_root,
                contexts=(
                    *tuple(request.standalone_contexts),
                    request.joint_context,
                ),
                vector_manifests=tuple(
                    manifest_path
                    for (
                        _ar,
                        _graph_name,
                        _standalone_context,
                        manifest_path,
                        _expected_sha,
                    ) in resolved_cases
                ),
            )

            with self._device_stage(
                manifest,
                adapter,
                stage_name="build",
                input_manifest_sha256=input_manifest_sha256,
                stage_config={},
                push_files=push_files,
            ) as device_stage:
                for (
                    ar,
                    graph_name,
                    standalone_context,
                    manifest_path,
                    expected_sha,
                ) in resolved_cases:
                    vector = VectorPreparer.load_manifest(
                        manifest_path,
                        expected_sha256=expected_sha,
                    )
                    inputs = VectorPreparer.load_tensors(
                        manifest_path,
                        section="inputs",
                        expected_manifest_sha256=expected_sha,
                    )
                    goldens = VectorPreparer.load_tensors(
                        manifest_path,
                        section="goldens",
                        expected_manifest_sha256=expected_sha,
                    )
                    if not goldens:
                        raise ExperimentalFeatureError(
                            "Qwen3.5 validation case "
                            f"{vector.case_id!r} has no goldens"
                        )
                    standalone_outputs = _output_mapping(
                        adapter.run_graph(
                            standalone_context,
                            inputs,
                            graph_name=graph_name,
                            device=device_stage.device,
                            native_io=native_io,
                            **execution_options,
                        ),
                        graph_name=graph_name,
                    )
                    joint_outputs = _output_mapping(
                        adapter.run_graph(
                            request.joint_context,
                            inputs,
                            graph_name=graph_name,
                            device=device_stage.device,
                            native_io=native_io,
                            **execution_options,
                        ),
                        graph_name=graph_name,
                    )
                    standalone_passed, standalone_report = compare(
                        goldens,
                        standalone_outputs,
                        label="standalone_vs_golden",
                    )
                    joint_passed, joint_report = compare(
                        goldens,
                        joint_outputs,
                        label="joint_vs_golden",
                    )
                    pair_passed, pair_report = compare(
                        standalone_outputs,
                        joint_outputs,
                        label="standalone_vs_joint",
                    )
                    standalone_ok = standalone_ok and standalone_passed
                    joint_ok = joint_ok and joint_passed
                    pair_ok = pair_ok and pair_passed
                    executed.append(str(graph_name))
                    golden_ids.append(
                        f"{vector.case_id}:"
                        f"{VectorPreparer.manifest_sha256(manifest_path)}"
                    )
                    graph_reports.append(
                        {
                            "ar": int(ar),
                            "graph_name": str(graph_name),
                            "golden_vector_id": golden_ids[-1],
                            "standalone_vs_golden": standalone_report,
                            "joint_vs_golden": joint_report,
                            "standalone_vs_joint": pair_report,
                        }
                    )
                device_identifier = device_stage.identifier
                device_soc = device_stage.soc_verification
                remote_attempt_dir = device_stage.adb.attempt_dir

            report_payload = {
                "schema": "qairt-agent.qwen35-runtime-validation",
                "slice_name": request.slice_name,
                "rtol": rtol,
                "atol": atol,
                "graphs": graph_reports,
                "gates": {
                    "standalone_vs_golden": standalone_ok,
                    "joint_vs_golden": joint_ok,
                    "standalone_vs_joint": pair_ok,
                },
                "quality_policy": "SQNR values are report_only",
                "equivalence_policy": "allclose is a structural transform gate",
                "device_identifier": device_identifier,
                "device_soc": device_soc,
                "remote_attempt_dir": remote_attempt_dir,
                "remote_cleanup": "confirmed",
            }
            report_ref = atomic_publish_json(
                report_root
                / request.slice_name
                / "qwen35_runtime_validation.json",
                report_payload,
                kind=ArtifactKind.REPORT,
                logical_name=f"qwen35_{request.slice_name}_runtime_validation",
            )
            return Qwen35RuntimeValidationResult(
                standalone_vs_golden_passed=standalone_ok,
                joint_vs_golden_passed=joint_ok,
                standalone_vs_joint_passed=pair_ok,
                executed_graph_names=tuple(executed),
                golden_vector_ids=tuple(golden_ids),
                report_paths=(report_ref.path,),
                details=(
                    "device-backed standalone, joint, and golden comparisons "
                    f"with rtol={rtol}, atol={atol}"
                ),
            )

        return validate, tuple(case_refs)


