"""Binding the vectors each SQNR mode is entitled to.

`teacher_forced` feeds every slice inputs from its own golden boundary, which
must come from `slice_vector_manifests` or an exact ONNX reference run over the
transformed slice models -- device boundary outputs are never accepted as
teacher inputs, because that would compare a slice against its own error.
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




class QualityVectorStage:
    """QualityVectorStage — see the module docstring."""

    @classmethod
    def _slice_tensor_tree(
        cls,
        value: Any,
        *,
        section: str,
    ) -> dict[str, dict[str, np.ndarray]]:
        if not isinstance(value, Mapping):
            raise TypeError("slice tensor tree must map slice name to tensors or manifest")
        return {
            str(slice_name): cls._tensor_mapping(slice_value, section=section)
            for slice_name, slice_value in value.items()
        }



    @staticmethod
    def _requested_sqnr_modes(
        spec: BuildSpec,
        config: Mapping[str, Any],
    ) -> tuple[SqnrMode, ...]:
        raw_modes = config.get("sqnr_modes")
        if raw_modes is None:
            return tuple(spec.quality.sqnr_modes)
        if isinstance(raw_modes, (str, bytes)) or not isinstance(
            raw_modes, Sequence
        ):
            raise InvalidSpecError(
                "sqnr_modes must be a list of full_reference, "
                "teacher_forced, and/or chain",
                stage="validate",
            )
        try:
            modes = tuple(
                value
                if isinstance(value, SqnrMode)
                else SqnrMode(str(getattr(value, "value", value)))
                for value in raw_modes
            )
        except ValueError as exc:
            raise InvalidSpecError(
                "sqnr_modes contains an unsupported mode",
                stage="validate",
                details={"sqnr_modes": [str(value) for value in raw_modes]},
            ) from exc
        if len(modes) != len(set(modes)):
            raise InvalidSpecError(
                "sqnr_modes must not contain duplicates",
                stage="validate",
                details={"sqnr_modes": [mode.value for mode in modes]},
            )
        return modes



    @staticmethod
    def _validate_slice_quality_contract(
        routes: Sequence[SliceRoute],
        references: Mapping[str, Mapping[str, np.ndarray]],
        teacher_inputs: Mapping[str, Mapping[str, np.ndarray]],
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, dict[str, np.ndarray]],
    ]:
        """Require explicit golden inputs/outputs for every routed slice."""

        missing: list[dict[str, Any]] = []
        normalized_references: dict[str, dict[str, np.ndarray]] = {}
        normalized_inputs: dict[str, dict[str, np.ndarray]] = {}
        for route in routes:
            slice_references = dict(references.get(route.slice_id, {}))
            slice_inputs = dict(teacher_inputs.get(route.slice_id, {}))
            missing_inputs = sorted(set(route.input_names) - set(slice_inputs))
            missing_outputs = sorted(
                set(route.output_names) - set(slice_references)
            )
            if missing_inputs or missing_outputs:
                missing.append(
                    {
                        "slice": route.slice_id,
                        "inputs": missing_inputs,
                        "outputs": missing_outputs,
                    }
                )
                continue
            normalized_inputs[route.slice_id] = {
                name: np.asarray(slice_inputs[name])
                for name in route.input_names
            }
            normalized_references[route.slice_id] = {
                name: np.asarray(slice_references[name])
                for name in route.output_names
            }
        if missing:
            raise InvalidSpecError(
                "teacher-forced SQNR requires golden inputs and outputs for "
                "every routed slice",
                stage="validate",
                details={"missing_slice_tensors": missing},
            )
        return normalized_references, normalized_inputs



    @classmethod
    def _prepare_slice_quality_vectors(
        cls,
        manifest: RunManifest,
        effective: Mapping[str, Any],
        routes: Sequence[SliceRoute],
        initial_inputs: Mapping[str, np.ndarray],
        output_dir: Path,
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, dict[str, np.ndarray]],
        tuple[Path, ...],
        tuple[ArtifactRef, ...],
        list[dict[str, Any]],
    ]:
        """Resolve explicit per-slice goldens or capture a reference ONNX chain.

        Device-chain outputs are never reused as teacher inputs.  An explicit
        ``slice_vector_manifests`` mapping wins.  Otherwise the exact transformed
        ONNX slice selected by the build runtime index is executed in route order
        and its raw inputs/outputs are published as immutable golden manifests.
        """

        explicit_manifests = effective.get("slice_vector_manifests")
        explicit_references = effective.get("slice_references")
        explicit_teacher_inputs = effective.get("teacher_inputs")
        explicit_count = sum(
            value is not None
            for value in (
                explicit_manifests,
                explicit_references,
                explicit_teacher_inputs,
            )
        )
        if explicit_manifests is not None and explicit_count != 1:
            raise InvalidSpecError(
                "slice_vector_manifests cannot be combined with "
                "slice_references or teacher_inputs",
                stage="validate",
            )
        if (explicit_references is None) != (explicit_teacher_inputs is None):
            raise InvalidSpecError(
                "slice_references and teacher_inputs must be supplied together",
                stage="validate",
            )

        if explicit_manifests is not None:
            if not isinstance(explicit_manifests, Mapping):
                raise InvalidSpecError(
                    "slice_vector_manifests must map slice IDs to vector manifests",
                    stage="validate",
                )
            missing_slices = [
                route.slice_id
                for route in routes
                if route.slice_id not in explicit_manifests
            ]
            if missing_slices:
                raise InvalidSpecError(
                    "slice_vector_manifests is incomplete",
                    stage="validate",
                    details={"missing_slices": missing_slices},
                )
            references: dict[str, dict[str, np.ndarray]] = {}
            teacher_inputs: dict[str, dict[str, np.ndarray]] = {}
            paths: list[Path] = []
            audit: list[dict[str, Any]] = []
            for route in routes:
                path = Path(
                    str(explicit_manifests[route.slice_id])
                ).expanduser().resolve()
                vector = VectorPreparer.load_manifest(path)
                teacher_inputs[route.slice_id] = VectorPreparer.load_tensors(
                    path,
                    section="inputs",
                )
                references[route.slice_id] = VectorPreparer.load_tensors(
                    path,
                    section="goldens",
                )
                paths.append(path)
                audit.append(
                    {
                        "slice": route.slice_id,
                        "reference_source": "provided_per_slice_golden",
                        "manifest_path": os.fspath(path),
                        "manifest_sha256": sha256_file(path),
                        "case_id": vector.case_id,
                    }
                )
            normalized_refs, normalized_inputs = (
                cls._validate_slice_quality_contract(
                    routes,
                    references,
                    teacher_inputs,
                )
            )
            return (
                normalized_refs,
                normalized_inputs,
                tuple(paths),
                (),
                audit,
            )

        if explicit_references is not None:
            references = cls._slice_tensor_tree(
                explicit_references,
                section=str(effective.get("slice_reference_section", "goldens")),
            )
            teacher_inputs = cls._slice_tensor_tree(
                explicit_teacher_inputs,
                section=str(effective.get("teacher_input_section", "inputs")),
            )
            normalized_refs, normalized_inputs = (
                cls._validate_slice_quality_contract(
                    routes,
                    references,
                    teacher_inputs,
                )
            )
            return (
                normalized_refs,
                normalized_inputs,
                (),
                (),
                [
                    {
                        "slice": route.slice_id,
                        "reference_source": "provided_per_slice_golden",
                    }
                    for route in routes
                ],
            )

        index = cls._runtime_index_for_manifest(manifest)
        cl_key = str(int(effective["context_length"]))
        ar_key = str(int(effective["ar"]))
        models = (
            ((index.get("transformed_slices") or {}).get(cl_key) or {}).get(
                ar_key
            )
            or {}
        )
        missing_models = [
            route.slice_id
            for route in routes
            if not isinstance(models.get(route.slice_id), Mapping)
            or not models[route.slice_id].get("onnx_path")
        ]
        if missing_models:
            raise InvalidSpecError(
                "automatic teacher-forced SQNR cannot capture a reference "
                "ONNX slice chain",
                stage="validate",
                details={
                    "missing_slice_models": missing_models,
                    "hint": (
                        "provide stage_configs.validation."
                        "slice_vector_manifests with one golden manifest per slice"
                    ),
                },
            )

        available = {
            str(name): np.asarray(value)
            for name, value in initial_inputs.items()
        }
        native_state: dict[str, np.ndarray] = {}
        for route in routes:
            for input_name, state_slot in route.state_inputs.items():
                if input_name in initial_inputs:
                    native_state[state_slot] = np.asarray(
                        initial_inputs[input_name]
                    )

        references: dict[str, dict[str, np.ndarray]] = {}
        teacher_inputs: dict[str, dict[str, np.ndarray]] = {}
        paths: list[Path] = []
        refs: list[ArtifactRef] = []
        audit: list[dict[str, Any]] = []
        for ordinal, route in enumerate(routes):
            forced: dict[str, np.ndarray] = {}
            missing_inputs: list[str] = []
            for input_name in route.input_names:
                if input_name in route.state_inputs:
                    state_slot = route.state_inputs[input_name]
                    if state_slot in native_state:
                        forced[input_name] = native_state[state_slot]
                    else:
                        missing_inputs.append(input_name)
                    continue
                if input_name in route.from_previous:
                    source_name = route.from_previous[input_name]
                    if source_name in available:
                        forced[input_name] = available[source_name]
                    else:
                        missing_inputs.append(input_name)
                    continue
                if input_name in available:
                    forced[input_name] = available[input_name]
                else:
                    missing_inputs.append(input_name)
            if missing_inputs:
                raise InvalidSpecError(
                    "reference ONNX slice chain is missing golden boundary inputs",
                    stage="validate",
                    details={
                        "missing_slice_tensors": [
                            {
                                "slice": route.slice_id,
                                "inputs": sorted(missing_inputs),
                                "outputs": [],
                            }
                        ]
                    },
                )

            model_path = Path(
                str(models[route.slice_id]["onnx_path"])
            ).expanduser().resolve()
            preparer = VectorPreparer(
                output_dir
                / "slice-reference"
                / f"{ordinal:03d}-{route.slice_id}"
            )
            seed = preparer.prepare_case(
                f"reference-{route.slice_id}-ar{ar_key}",
                forced,
                metadata={
                    "reference_request": "teacher_forced_slice_boundary",
                    "slice": route.slice_id,
                    "ar": int(ar_key),
                    "context_length": int(cl_key),
                    "reference_model_path": os.fspath(model_path),
                    "reference_model_sha256": sha256_file(model_path),
                },
            )
            try:
                captured = preparer.capture_onnx(
                    seed,
                    model_path,
                    output_names=route.output_names,
                    destination_name="vector_manifest.slice-reference.json",
                )
            except Exception as exc:
                raise InvalidSpecError(
                    "failed to capture a golden reference for transformed "
                    f"slice {route.slice_id!r}",
                    stage="validate",
                    details={
                        "slice": route.slice_id,
                        "model_path": os.fspath(model_path),
                        "missing_inputs": [],
                    },
                ) from exc
            outputs = VectorPreparer.load_tensors(
                captured,
                section="goldens",
            )
            teacher_inputs[route.slice_id] = forced
            references[route.slice_id] = {
                name: outputs[name] for name in route.output_names
            }
            available.update(references[route.slice_id])
            available.update(
                {
                    f"{route.slice_id}.{name}": value
                    for name, value in references[route.slice_id].items()
                }
            )
            for output_name, state_slot in route.state_outputs.items():
                native_state[state_slot] = references[route.slice_id][
                    output_name
                ]
            paths.append(Path(captured).resolve())
            ref = ArtifactRef.from_path(
                captured,
                kind=ArtifactKind.GOLDEN_VECTORS,
                logical_name=f"slice_reference_{route.slice_id}",
            )
            refs.append(ref)
            audit.append(
                {
                    "slice": route.slice_id,
                    "reference_source": "onnxruntime_slice_chain",
                    "manifest_path": os.fspath(ref.path),
                    "manifest_sha256": ref.sha256,
                    "model_path": os.fspath(model_path),
                    "model_sha256": sha256_file(model_path),
                    "input_names": list(route.input_names),
                    "output_names": list(route.output_names),
                }
            )

        normalized_refs, normalized_inputs = (
            cls._validate_slice_quality_contract(
                routes,
                references,
                teacher_inputs,
            )
        )
        return (
            normalized_refs,
            normalized_inputs,
            tuple(paths),
            tuple(refs),
            audit,
        )


