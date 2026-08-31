"""The validate stage: supplied goldens first, and every gap declared.

Supplied AIMET goldens are the production reference. An ONNX Runtime capture is
a recorded fallback for a manifest with executable inputs and no goldens, never
a replacement; a manifest with neither fails closed. The float-graph reference
is a separate debug-only mode that publishes its own artifact and never touches
the golden comparison.
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



class ValidateStage:
    """ValidateStage — see the module docstring."""

    @staticmethod
    def _enforce_qwen3_vl_runtime_scope(
        manifest: RunManifest,
        selected: Mapping[str, Any],
        *,
        stage: str,
    ) -> dict[str, Any]:
        """Forbid an unlabelled text-only run from posing as VL end-to-end."""

        effective = dict(selected)
        if manifest.build_spec.family is not ModelFamily.QWEN3_VL:
            return effective

        component = str(effective.get("component", "")).strip().lower()
        if component not in {"text", "vision"}:
            raise InvalidSpecError(
                "Qwen3-VL compiled runtime is component-scoped: declare "
                "component='text' for a text-only run or component='vision' "
                "for a vision-only graph. End-to-end multimodal execution "
                "is unavailable until the vision-to-text boundary is audited.",
                stage=stage,
                details={
                    "requested_component": component or None,
                    "supported_components": ["text", "vision"],
                    "end_to_end_supported": False,
                },
            )

        expected_coverage = (
            "text_only" if component == "text" else "vision_only"
        )
        supplied_coverage = effective.get("coverage")
        if (
            supplied_coverage is not None
            and str(supplied_coverage) != expected_coverage
        ):
            raise InvalidSpecError(
                "Qwen3-VL runtime coverage contradicts the selected component",
                stage=stage,
                details={
                    "component": component,
                    "expected_coverage": expected_coverage,
                    "supplied_coverage": supplied_coverage,
                },
            )
        if component == "vision" and effective.get("routes") is not None:
            raise InvalidSpecError(
                "Qwen3-VL vision-only execution requires one explicit graph; "
                "a text slice route cannot be reused for the vision component",
                stage=stage,
            )

        effective["component"] = component
        effective["coverage"] = expected_coverage
        return effective


    @staticmethod
    def _reject_genai_chain_keys(
        effective: Mapping[str, Any],
        *,
        stage: str,
    ) -> None:
        """Fail closed when a GenAI run carries low-level chain keys.

        The GenAI branch executes the saved container and reads raw slices from
        ``tensor_runtime``; a top-level ``routes``/``steps`` pair would be
        silently ignored or crash the optrace path instead of measuring what the
        caller asked for.
        """

        if not (
            effective.get("lane") == "genai_builder"
            or effective.get("container_path") is not None
        ):
            return
        conflicting = [
            key
            for key in _LOW_LEVEL_CHAIN_CONFIG_FIELDS
            if effective.get(key) is not None
        ]
        if not conflicting:
            return
        raise InvalidSpecError(
            "GenAI container execution cannot be combined with low-level "
            "slice-chain configuration; drop these keys or select the "
            "low-level lane explicitly",
            stage=stage,
            details={
                "conflicting_keys": conflicting,
                "container_path": effective.get("container_path"),
                "lane": effective.get("lane"),
                "raw_slice_profiling": "stage_configs.benchmark.tensor_runtime",
            },
        )


    @staticmethod
    def _initial_native_state_from_routes(
        routes: Sequence[Mapping[str, Any]],
        inputs: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Bind manifest KV/state tensors to the route's invocation-local slots."""

        state: dict[str, np.ndarray] = {}
        for route in routes:
            state_inputs = (
                route.state_inputs
                if isinstance(route, SliceRoute)
                else route.get("state_inputs")
            )
            for input_name, slot in dict(state_inputs or {}).items():
                if input_name not in inputs:
                    continue
                value = np.asarray(inputs[input_name])
                if slot in state and not np.array_equal(state[slot], value):
                    raise ValueError(
                        f"native state slot {slot!r} has conflicting inputs"
                    )
                state[str(slot)] = value
        return state


    @staticmethod
    def _manifest_inputs(
        path: str | Path,
        *,
        section: str = "inputs",
        sha256: str | None = None,
    ) -> dict[str, np.ndarray]:
        return VectorPreparer.load_tensors(
            path,
            section=section,
            expected_manifest_sha256=sha256,
        )


    @staticmethod
    def _tensor_value(value: Any) -> np.ndarray:
        if isinstance(value, Mapping) and "path" in value:
            source = TensorSource.from_value(value)
            path = source.path.expanduser().resolve()
            if path.suffix == ".npy":
                return np.asarray(np.load(path, allow_pickle=False))
            if source.dtype is None or source.shape is None:
                raise ValueError(f"raw tensor {path} requires dtype and shape")
            return np.fromfile(path, dtype=np.dtype(source.dtype)).reshape(source.shape)
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser().resolve()
            if path.suffix != ".npy":
                raise ValueError(f"standalone tensor path must be .npy: {path}")
            return np.asarray(np.load(path, allow_pickle=False))
        return np.asarray(value)


    @classmethod
    def _tensor_mapping(
        cls,
        value: Any,
        *,
        section: str = "inputs",
        sha256: str | None = None,
    ) -> dict[str, np.ndarray]:
        if isinstance(value, (str, Path)):
            return cls._manifest_inputs(value, section=section, sha256=sha256)
        if not isinstance(value, Mapping):
            raise TypeError("tensor mapping must be a vector-manifest path or mapping")
        return {str(name): cls._tensor_value(tensor) for name, tensor in value.items()}


    def validate(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        vector_manifest: str | None = None,
        config: Mapping[str, Any] | None = None,
        execution_context: StageExecutionContext | None = None,
    ) -> ToolResult[dict[str, Any]]:
        """Compare full/teacher/device-chain tensors; SQNR remains report-only."""

        selected = dict(config or {})
        if vector_manifest is not None:
            selected.setdefault("vector_manifest", vector_manifest)

        def run_one(
            manifest: RunManifest,
            adapter: Any,
            output_dir: Path,
            selected_config: Mapping[str, Any],
            *,
            report_suffix: str = "",
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            effective = dict(selected_config)
            explicit_outputs = any(
                key in effective
                for key in (
                    "references",
                    "teacher_forced_outputs",
                    "device_chain_outputs",
                    "actual_manifest",
                )
            )
            explicit_graph = all(
                effective.get(key) is not None
                for key in ("context_path", "graph_name", "vector_manifest")
            )
            explicit_chain = (
                effective.get("routes") is not None
                and isinstance(effective.get("contexts"), Mapping)
                and effective.get("vector_manifest") is not None
            )
            if not explicit_outputs and not explicit_graph and not explicit_chain:
                effective = self._automatic_runtime_binding(manifest, effective)
                if effective["lane"] == "genai_builder":
                    if not effective.get("runtime_supported", False):
                        raise InvalidSpecError(
                            "the GenAI container runtime is unsupported by this "
                            "QAIRT SDK workflow; validation fails closed",
                            stage="validate",
                            details={
                                "family": effective.get("family"),
                                "container_path": effective.get("container_path"),
                            },
                        )
                    tensor_runtime = effective.get("tensor_runtime")
                    if not isinstance(tensor_runtime, Mapping):
                        raise InvalidSpecError(
                            "this saved GenAI container did not expose an "
                            "auditable public CompiledModel split route for "
                            "raw-tensor SQNR; provide explicit actual/reference "
                            "manifests or a low-level diagnostic context",
                            stage="validate",
                        )
                    effective.update(
                        {
                            "routes": tensor_runtime["routes"],
                            "contexts": tensor_runtime["contexts"],
                            "scope": tensor_runtime["scope"],
                        }
                    )
            if not explicit_outputs:
                effective = self._enforce_qwen3_vl_runtime_scope(
                    manifest,
                    effective,
                    stage="validate",
                )
            requested_modes = self._requested_sqnr_modes(
                manifest.build_spec,
                effective,
            )
            automatic_mode_execution = bool(requested_modes) and not explicit_outputs

            reference_refs: list[ArtifactRef] = []
            reference_source = "provided"
            if "references" in effective:
                references = self._slice_tensor_tree(
                    effective["references"],
                    section=str(effective.get("reference_section", "goldens")),
                )
            elif effective.get("vector_manifest") is not None:
                vector_path = effective["vector_manifest"]
                vector = VectorPreparer.load_manifest(
                    vector_path,
                    expected_sha256=effective.get("vector_manifest_sha256"),
                )
                if not vector.goldens:
                    reference_model = (
                        effective.get("reference_model_path")
                        or effective.get("model_path")
                        or manifest.build_spec.sources.text.onnx_path
                    )
                    vector_path = self._capture_onnx_reference(
                        vector_path,
                        reference_model,
                        output_dir,
                        expected_manifest_sha256=effective.get(
                            "vector_manifest_sha256"
                        ),
                    )
                    reference_refs.append(
                        ArtifactRef.from_path(
                            vector_path,
                            kind=ArtifactKind.GOLDEN_VECTORS,
                            logical_name=(
                                "onnxruntime_reference_manifest"
                                f"{report_suffix}"
                            ),
                        )
                    )
                    reference_source = "onnxruntime"
                references = {
                    "model": self._manifest_inputs(
                        vector_path,
                        section="goldens",
                        sha256=(
                            None
                            if reference_source == "onnxruntime"
                            else effective.get("vector_manifest_sha256")
                        ),
                    )
                }
            else:
                raise ValueError("validate requires references or vector_manifest")
            if not references or not any(references.values()):
                raise ValueError(
                    "SQNR validation requires at least one reference tensor; "
                    "empty reports are forbidden"
                )
            full_references = references
            full_reference_actual: dict[str, dict[str, np.ndarray]] | None = None
            slice_reference_refs: tuple[ArtifactRef, ...] = ()
            slice_reference_audit: list[dict[str, Any]] = []
            device_identifier: Any | None = None
            device_soc: Any | None = None
            remote_attempt_dir: Any | None = None
            diagnostic_outputs: dict[str, dict[str, Any]] | None = None
            diagnostic_evidence: dict[str, Any] | None = None

            teacher = (
                self._slice_tensor_tree(
                    effective["teacher_forced_outputs"],
                    section=str(effective.get("actual_section", "inputs")),
                )
                if "teacher_forced_outputs" in effective
                else None
            )
            chain = (
                self._slice_tensor_tree(
                    effective["device_chain_outputs"],
                    section=str(effective.get("actual_section", "inputs")),
                )
                if "device_chain_outputs" in effective
                else None
            )
            if teacher is None and chain is None:
                actual = effective.get("actual_manifest")
                if actual is not None:
                    chain = {
                        "model": self._manifest_inputs(
                            actual,
                            section=str(effective.get("actual_section", "goldens")),
                            sha256=effective.get("actual_manifest_sha256"),
                        )
                    }
                elif effective.get("routes") is not None:
                    self._preflight(adapter, manifest.build_spec)
                    inputs = self._manifest_inputs(
                        effective["vector_manifest"],
                        section=str(effective.get("input_section", "inputs")),
                        sha256=effective.get("vector_manifest_sha256"),
                    )
                    contexts = effective.get("contexts")
                    if not isinstance(contexts, Mapping) or not contexts:
                        raise ValueError(
                            "chain validation requires contexts mapped by slice"
                        )
                    routes = tuple(
                        SliceRoute.from_object(route)
                        for route in effective["routes"]
                    )
                    initial_native_state = self._tensor_mapping(
                        effective.get("initial_native_state", {})
                    )
                    if not initial_native_state:
                        initial_native_state = self._initial_native_state_from_routes(
                            routes,
                            inputs,
                        )
                    slice_vector_paths: tuple[Path, ...] = ()
                    teacher_inputs: dict[
                        str, dict[str, np.ndarray]
                    ] | None = None
                    if automatic_mode_execution and any(
                        mode in requested_modes
                        for mode in (SqnrMode.TEACHER_FORCED, SqnrMode.CHAIN)
                    ):
                        (
                            slice_references,
                            teacher_inputs,
                            slice_vector_paths,
                            slice_reference_refs,
                            slice_reference_audit,
                        ) = self._prepare_slice_quality_vectors(
                            manifest,
                            effective,
                            routes,
                            inputs,
                            output_dir,
                        )
                        references = slice_references
                    inline_cases: list[Mapping[str, Any]] = []
                    if initial_native_state:
                        inline_cases.append(initial_native_state)
                    if teacher_inputs is not None:
                        inline_cases.extend(teacher_inputs.values())
                    push_files = self._device_stage_files(
                        output_dir,
                        contexts=tuple(contexts.values()),
                        vector_manifests=(
                            effective["vector_manifest"],
                            *slice_vector_paths,
                        ),
                        inline_cases=tuple(inline_cases),
                    )
                    with self._device_stage(
                        manifest,
                        adapter,
                        stage_name="validate",
                        input_manifest_sha256=manifest_sha256,
                        stage_config=effective,
                        push_files=push_files,
                    ) as device_stage:
                        runner = SliceChainRunner(
                            routes,
                            self._chain_executors(
                                adapter,
                                contexts,
                                device=device_stage.device,
                                native_io=bool(effective.get("native_io", False)),
                                execution_options=self._execution_options(effective),
                            ),
                        )
                        selected_ar = int(
                            effective.get(
                                "ar",
                                manifest.build_spec.sequence.ars[0],
                            )
                        )
                        needs_chain = (
                            not automatic_mode_execution
                            or SqnrMode.FULL_REFERENCE in requested_modes
                            or SqnrMode.CHAIN in requested_modes
                        )
                        if needs_chain:
                            chain_result = runner.run_device_chain(
                                inputs,
                                ar=selected_ar,
                                initial_native_state=initial_native_state,
                            )
                            full_reference_actual = {
                                "model": dict(chain_result.final_outputs)
                            }
                            if automatic_mode_execution:
                                chain = (
                                    {
                                        str(name): dict(values)
                                        for name, values
                                        in chain_result.outputs_by_slice().items()
                                    }
                                    if SqnrMode.CHAIN in requested_modes
                                    else None
                                )
                            else:
                                chain = (
                                    full_reference_actual
                                    if set(references) == {"model"}
                                    else {
                                        str(name): dict(values)
                                        for name, values
                                        in chain_result.outputs_by_slice().items()
                                    }
                                )
                        if (
                            automatic_mode_execution
                            and SqnrMode.TEACHER_FORCED in requested_modes
                        ):
                            assert teacher_inputs is not None
                            teacher_result = runner.run_teacher_forced(
                                inputs,
                                teacher_inputs,
                                ar=selected_ar,
                                initial_native_state=initial_native_state,
                            )
                            teacher = {
                                str(name): dict(values)
                                for name, values
                                in teacher_result.outputs_by_slice().items()
                            }
                        if _layer_float_reference(effective):
                            (
                                diagnostic_outputs,
                                diagnostic_evidence,
                            ) = self._diagnostic_device_outputs(
                                manifest,
                                effective,
                                adapter,
                                device=device_stage.device,
                                ar=selected_ar,
                                inputs=inputs,
                                initial_native_state=initial_native_state,
                            )
                        device_identifier = device_stage.identifier
                        device_soc = device_stage.soc_verification
                        remote_attempt_dir = device_stage.adb.attempt_dir
                elif all(
                    effective.get(key) is not None
                    for key in ("context_path", "graph_name", "vector_manifest")
                ):
                    self._preflight(adapter, manifest.build_spec)
                    inputs = self._manifest_inputs(
                        effective["vector_manifest"],
                        section=str(effective.get("input_section", "inputs")),
                        sha256=effective.get("vector_manifest_sha256"),
                    )
                    push_files = self._device_stage_files(
                        output_dir,
                        contexts=(effective["context_path"],),
                        vector_manifests=(effective["vector_manifest"],),
                    )
                    with self._device_stage(
                        manifest,
                        adapter,
                        stage_name="validate",
                        input_manifest_sha256=manifest_sha256,
                        stage_config=effective,
                        push_files=push_files,
                    ) as device_stage:
                        raw_result = adapter.run_graph(
                            effective["context_path"],
                            inputs,
                            graph_name=str(effective["graph_name"]),
                            device=device_stage.device,
                            native_io=bool(effective.get("native_io", False)),
                            **self._execution_options(effective),
                        )
                        chain = {
                            "model": _output_mapping(
                                raw_result,
                                graph_name=str(effective["graph_name"]),
                            )
                        }
                        full_reference_actual = chain
                        if automatic_mode_execution:
                            teacher = (
                                chain
                                if SqnrMode.TEACHER_FORCED in requested_modes
                                else None
                            )
                            chain = (
                                chain
                                if SqnrMode.CHAIN in requested_modes
                                else None
                            )
                        if _layer_float_reference(effective):
                            (
                                diagnostic_outputs,
                                diagnostic_evidence,
                            ) = self._diagnostic_device_outputs(
                                manifest,
                                effective,
                                adapter,
                                device=device_stage.device,
                                ar=int(
                                    effective.get("ar")
                                    or manifest.build_spec.sequence.ars[0]
                                ),
                                inputs=inputs,
                            )
                        device_identifier = device_stage.identifier
                        device_soc = device_stage.soc_verification
                        remote_attempt_dir = device_stage.adb.attempt_dir
                else:
                    raise ValueError(
                        "validate requires teacher_forced_outputs, "
                        "device_chain_outputs, actual_manifest, or "
                        "context_path+graph_name+vector_manifest"
                    )
            diagnoser = QualityDiagnoser(
                reference_energy_floor=float(
                    effective.get("reference_energy_floor", 0.0)
                )
            )
            report = diagnoser.diagnose_slices(
                references,
                teacher_forced_outputs=teacher,
                device_chain_outputs=chain,
                lineage=effective.get("lineage"),
            )
            if not report.observations:
                raise ValueError("SQNR validation produced no observations")
            mode_reports: dict[str, Any] = {}
            if requested_modes:
                if SqnrMode.FULL_REFERENCE in requested_modes:
                    full_actual = full_reference_actual
                    if full_actual is None and set(full_references) == {"model"}:
                        full_actual = chain
                    if full_actual is None:
                        raise InvalidSpecError(
                            "full_reference SQNR requested without end-to-end "
                            "device outputs",
                            stage="validate",
                        )
                    full_report = diagnoser.diagnose_slices(
                        full_references,
                        device_chain_outputs=full_actual,
                        lineage=effective.get("lineage"),
                    )
                    mode_reports[SqnrMode.FULL_REFERENCE.value] = (
                        full_report.to_dict()
                    )
                if SqnrMode.TEACHER_FORCED in requested_modes:
                    if teacher is None:
                        raise InvalidSpecError(
                            "teacher_forced SQNR requested without "
                            "teacher-forced device outputs",
                            stage="validate",
                        )
                    mode_reports[SqnrMode.TEACHER_FORCED.value] = (
                        diagnoser.diagnose_slices(
                            references,
                            teacher_forced_outputs=teacher,
                            lineage=effective.get("lineage"),
                        ).to_dict()
                    )
                if SqnrMode.CHAIN in requested_modes:
                    if chain is None:
                        raise InvalidSpecError(
                            "chain SQNR requested without device-chain outputs",
                            stage="validate",
                        )
                    mode_reports[SqnrMode.CHAIN.value] = (
                        diagnoser.diagnose_slices(
                            references,
                            device_chain_outputs=chain,
                            lineage=effective.get("lineage"),
                        ).to_dict()
                    )
            payload = report.to_dict()
            payload["policy"] = "report_only"
            payload["reference_source"] = reference_source
            payload["requested_modes"] = [
                mode.value for mode in requested_modes
            ]
            payload["executed_modes"] = list(mode_reports)
            payload["mode_reports"] = mode_reports
            float_reference, float_reference_refs = self._float_reference_report(
                manifest,
                effective,
                device_outputs=chain,
                output_dir=output_dir,
                report_suffix=report_suffix,
                diagnostic_outputs=diagnostic_outputs,
                diagnostic_evidence=diagnostic_evidence,
            )
            if float_reference:
                payload["float_reference"] = float_reference
            if slice_reference_audit:
                payload["slice_reference_evidence"] = slice_reference_audit
            divergence_observed = any(
                mode_report.get("first_teacher_error") is not None
                or mode_report.get("first_chain_error") is not None
                for mode_report in mode_reports.values()
            ) if mode_reports else (
                report.first_teacher_error is not None
                or report.first_chain_error is not None
            )
            dump_intermediates = bool(
                effective.get(
                    "dump_intermediates_on_failure",
                    manifest.build_spec.quality.dump_intermediates_on_failure,
                )
            )
            if automatic_mode_execution:
                payload["diagnostic_evidence"] = (
                    self._diagnostic_context_evidence(
                        manifest,
                        effective,
                        requested=dump_intermediates,
                        divergence_observed=divergence_observed,
                        slice_reference_refs=slice_reference_refs,
                    )
                )
            payload["runtime_binding"] = {
                key: _jsonable(effective.get(key))
                for key in (
                    "lane",
                    "family",
                    "ar",
                    "context_length",
                    "scope",
                    "route_manifest",
                    "context_path",
                    "graph_name",
                    "reference_model_path",
                    "component",
                    "coverage",
                    "excluded_components",
                    "graph_ar",
                )
                if effective.get(key) is not None
            }
            requested_ar_values = [
                int(value)
                for value in manifest.build_spec.sequence.ars
            ]
            if effective.get("ar") is not None:
                executed_ar = int(effective["ar"])
                payload["coverage"] = {
                    "mode": (
                        "single_ar_override"
                        if selected_config.get("ar") is not None
                        and not report_suffix
                        else "single_ar"
                    ),
                    "requested_ars": requested_ar_values,
                    "executed_ars": [executed_ar],
                    "missing_ars": [
                        ar
                        for ar in requested_ar_values
                        if ar != executed_ar
                    ],
                    "complete": requested_ar_values == [executed_ar],
                }
            else:
                payload["coverage"] = {
                    "mode": "explicit_custom_runtime",
                    "requested_ars": requested_ar_values,
                    "executed_ars": "caller_defined",
                    "complete": None,
                }
            report_ref = atomic_publish_json(
                output_dir / f"sqnr_report{report_suffix}.json",
                payload,
                kind=ArtifactKind.REPORT,
                logical_name=f"sqnr_report{report_suffix}",
            )
            metrics: dict[str, Any] = {
                "observation_count": len(report.observations),
                "policy": "report_only",
                "reference_source": reference_source,
                "requested_modes": [
                    mode.value for mode in requested_modes
                ],
                "executed_modes": list(mode_reports),
                "divergence_observed": divergence_observed,
                "float_reference_debug": bool(float_reference),
            }
            if device_identifier is not None:
                metrics.update(
                    {
                        "device_identifier": device_identifier,
                        "device_soc": device_soc,
                        "remote_attempt_dir": remote_attempt_dir,
                        "remote_cleanup": "confirmed",
                    }
                )
            return (
                payload,
                tuple(reference_refs)
                + slice_reference_refs
                + float_reference_refs
                + (report_ref,),
                metrics,
            )

        def operation(
            manifest: RunManifest, adapter: Any, output_dir: Path
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            explicit_outputs = any(
                key in selected
                for key in (
                    "references",
                    "teacher_forced_outputs",
                    "device_chain_outputs",
                    "actual_manifest",
                )
            )
            explicit_graph = all(
                selected.get(key) is not None
                for key in ("context_path", "graph_name", "vector_manifest")
            )
            explicit_chain = (
                selected.get("routes") is not None
                and isinstance(selected.get("contexts"), Mapping)
                and selected.get("vector_manifest") is not None
            )
            automatic_binding = (
                not explicit_outputs
                and not explicit_graph
                and not explicit_chain
            )
            requested_ars = tuple(
                int(value) for value in manifest.build_spec.sequence.ars
            )
            if (
                not automatic_binding
                or selected.get("ar") is not None
                or len(requested_ars) <= 1
            ):
                return run_one(
                    manifest,
                    adapter,
                    output_dir,
                    selected,
                )

            runtime_index = self._runtime_index_for_manifest(manifest)
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
                    "automatic multi-AR validation requires one exact vector "
                    "manifest for every requested AR",
                    stage="validate",
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
            for ar in requested_ars:
                ar_key = str(ar)
                payload, refs, metrics = run_one(
                    manifest,
                    adapter,
                    output_dir / f"ar{ar}",
                    {**selected, "ar": ar},
                    report_suffix=f"_ar{ar}",
                )
                report_refs = [
                    ref
                    for ref in refs
                    if ref.logical_name == f"sqnr_report_ar{ar}"
                ]
                if len(report_refs) != 1:
                    raise InvalidSpecError(
                        "multi-AR validation did not publish exactly one "
                        f"AR{ar} SQNR report",
                        stage="validate",
                    )
                results_by_ar[ar_key] = {
                    "report": payload,
                    "report_artifact": _jsonable(report_refs[0]),
                }
                per_ar_metrics[ar_key] = _jsonable(metrics)
                output_refs.extend(refs)

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
            }
            reference_sources_by_ar = {
                ar: result["report"].get("reference_source")
                for ar, result in results_by_ar.items()
            }
            distinct_reference_sources = {
                str(value)
                for value in reference_sources_by_ar.values()
                if value is not None
            }
            first_teacher_error_ar = next(
                (
                    ar
                    for ar, result in results_by_ar.items()
                    if result["report"].get("first_teacher_error") is not None
                ),
                None,
            )
            first_chain_error_ar = next(
                (
                    ar
                    for ar, result in results_by_ar.items()
                    if result["report"].get("first_chain_error") is not None
                ),
                None,
            )
            aggregate_payload = {
                "schema": "qairt-agent.multi-ar-sqnr-report.v1",
                "policy": "report_only",
                "coverage": coverage,
                "results_by_ar": results_by_ar,
                # Preserve the most frequently consumed single-report fields
                # while keeping the AR-scoped reports authoritative.
                "reference_source": (
                    next(iter(distinct_reference_sources))
                    if len(distinct_reference_sources) == 1
                    else "mixed"
                ),
                "reference_sources_by_ar": reference_sources_by_ar,
                "first_teacher_error": (
                    results_by_ar[first_teacher_error_ar]["report"].get(
                        "first_teacher_error"
                    )
                    if first_teacher_error_ar is not None
                    else None
                ),
                "first_teacher_error_ar": (
                    int(first_teacher_error_ar)
                    if first_teacher_error_ar is not None
                    else None
                ),
                "first_chain_error": (
                    results_by_ar[first_chain_error_ar]["report"].get(
                        "first_chain_error"
                    )
                    if first_chain_error_ar is not None
                    else None
                ),
                "first_chain_error_ar": (
                    int(first_chain_error_ar)
                    if first_chain_error_ar is not None
                    else None
                ),
                "divergence_observed": (
                    first_teacher_error_ar is not None
                    or first_chain_error_ar is not None
                ),
            }
            # Build it, validate it, publish what the model says: a missing or
            # mistyped field fails here rather than in whatever reads the
            # report weeks later. The dump is byte-identical to the dict above
            # -- asserted by tests/test_contracts_reports.py.
            aggregate_payload = MultiArSqnrReport.model_validate(
                aggregate_payload
            ).to_payload()
            aggregate_ref = atomic_publish_json(
                output_dir / "sqnr_report.json",
                aggregate_payload,
                kind=ArtifactKind.REPORT,
                logical_name="sqnr_report",
            )
            output_refs.append(aggregate_ref)
            return aggregate_payload, tuple(output_refs), {
                "policy": "report_only",
                "coverage": coverage,
                "results_by_ar": per_ar_metrics,
                "reference_source": aggregate_payload["reference_source"],
                "divergence_observed": aggregate_payload[
                    "divergence_observed"
                ],
            }

        return self._continuation_operation(
            "validate",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
            execution_context=execution_context,
        )

