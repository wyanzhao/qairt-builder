"""The build stage: both lanes, and the evidence each publishes.

The low-level lane chains inspection, per-AR/CL conversion, split+MHA2SHA,
convert with AIMET encodings, and weight-shared compile per semantic slice. The
GenAI lane drives the SDK's pinned family builder with fail-closed per-AR
attached models and encodings. Neither lane lets a production context carry
diagnostic outputs.
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
from qairt_agent.family_registry import (
    has_audio_component,
    has_decoder_lane,
    has_vision_component,
    requires_derivation_evidence,
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



class BuildStage:
    """BuildStage — see the module docstring."""

    @staticmethod
    def _standalone_vit_config_path(spec: BuildSpec) -> Path | None:
        config_path = spec.sources.text.config_path
        if config_path is None:
            return None
        resolved = config_path.expanduser().resolve()
        if resolved.is_dir():
            resolved = resolved / "config.json"
        if not resolved.is_file():
            raise FileNotFoundError(f"model config does not exist: {resolved}")
        return resolved


    @staticmethod
    def _standalone_vit_effective_payload(spec: BuildSpec) -> dict[str, Any]:
        return {
            "schema": "qairt-agent.effective-config",
            "family": ModelFamily.VIT.value,
            "pipeline": "low_level_python_api",
            "component": "vit",
            "sources": spec.sources.model_dump(mode="json", exclude_none=True),
            "sequence": spec.sequence.model_dump(mode="json"),
            "split": spec.split.model_dump(mode="json"),
            "transforms": spec.transforms.model_dump(mode="json"),
            "quantization": spec.quantization.model_dump(mode="json"),
            "compile": spec.compile.model_dump(mode="json"),
            "benchmark": effective_benchmark_policy(spec),
            "target": spec.target.model_dump(mode="json"),
            "stages": ["qairt.convert", "qairt.compile"],
            "context_binary_grouping": "one_standalone_vit_context",
            "runtime_supported": True,
            "quality_policy": "report_only",
        }


    @staticmethod
    def _build_static_footprint(manifest: RunManifest) -> dict[str, Any] | None:
        """Return the footprint the build stage measured for this run.

        The block is copied from the hash-verified manifest, never re-measured,
        so a latency report answers "how big is what I just measured" with the
        same numbers the build published.
        """

        for stage in reversed(manifest.stages):
            footprint = stage.metrics.get("static_footprint")
            if isinstance(footprint, Mapping):
                return {
                    **dict(footprint),
                    "source": "build_receipt",
                    "measured_by_stage": stage.name,
                    "measured_by_attempt": stage.attempt,
                }
        return None


    @classmethod
    def _prepare_low_level_validation_vectors(
        cls,
        manifest: RunManifest,
        spec: BuildSpec,
        result: Any,
    ) -> tuple[dict[int, Path], tuple[ArtifactRef, ...], list[dict[str, Any]]]:
        """Bind or derive exact per-AR vectors for low-level build outputs."""

        base_manifest = spec.vectors.validation_manifest
        provided_by_ar = {
            int(ar): Path(path).expanduser().resolve()
            for ar, path in spec.vectors.validation_manifests_by_ar.items()
        }
        if base_manifest is None and not provided_by_ar:
            return {}, (), []

        context_lengths = tuple(int(value) for value in spec.sequence.context_lengths)
        if len(context_lengths) != 1:
            raise InvalidSpecError(
                "automatic validation-vector binding currently requires exactly "
                "one context length; use one workflow per CL",
                stage="build",
                details={"context_lengths": list(context_lengths)},
            )
        context_length = context_lengths[0]
        variants = {
            (int(item.ar), int(item.context_length)): Path(item.model_path)
            for item in tuple(getattr(result, "variants", ()))
        }
        if not has_decoder_lane(spec.family):
            variants[(1, context_length)] = spec.sources.text.onnx_path

        output_paths: dict[int, Path] = {}
        references: list[ArtifactRef] = []
        audit: list[dict[str, Any]] = []
        source_path = (
            Path(base_manifest).expanduser().resolve()
            if base_manifest is not None
            else None
        )
        for ar in (int(value) for value in spec.sequence.ars):
            target = variants.get((ar, context_length))
            if target is None:
                raise InvalidSpecError(
                    "low-level build did not produce the ONNX variant required "
                    "to bind validation vectors",
                    stage="build",
                    details={"ar": ar, "context_length": context_length},
                )

            provided = provided_by_ar.get(ar)
            if provided is not None:
                binding = validate_provided_ar_manifest(
                    provided,
                    target,
                    family=spec.family.value,
                    ar=ar,
                    cl=context_length,
                )
                selected_path = binding.manifest_path
                record = {
                    **binding.to_dict(),
                    "binding": "provided_per_ar",
                }
            elif source_path is not None:
                try:
                    binding = validate_provided_ar_manifest(
                        source_path,
                        target,
                        family=spec.family.value,
                        ar=ar,
                        cl=context_length,
                    )
                except VectorRetargetError:
                    selected_path = retarget_vector_manifest(
                        source_path,
                        target,
                        family=spec.family.value,
                        ar=ar,
                        cl=context_length,
                        output_dir=(
                            cls._run_dir(manifest)
                            / "vectors"
                            / f"cl{context_length}"
                            / f"ar{ar}"
                        ),
                        preserve_extra_inputs=(
                            has_vision_component(spec.family)
                        ),
                    )
                    generated = VectorPreparer.load_manifest(selected_path)
                    record = {
                        "binding": "derived_from_source_manifest",
                        "manifest_path": os.fspath(selected_path),
                        "manifest_sha256": sha256_file(selected_path),
                        "target_onnx_path": os.fspath(
                            Path(target).expanduser().resolve()
                        ),
                        "target_onnx_sha256": sha256_file(target),
                        "family": spec.family.value,
                        "ar": ar,
                        "cl": context_length,
                        "reference_source": generated.metadata.get(
                            "reference_source",
                            "onnxruntime",
                        ),
                    }
                else:
                    selected_path = binding.manifest_path
                    record = {
                        **binding.to_dict(),
                        "binding": "provided_base_exact_abi",
                    }
            else:
                raise InvalidSpecError(
                    f"no validation vector manifest is available for AR{ar}",
                    stage="build",
                )

            output_paths[ar] = selected_path
            vector = VectorPreparer.load_manifest(selected_path)
            kind = (
                ArtifactKind.GOLDEN_VECTORS
                if vector.goldens
                else ArtifactKind.TEST_VECTORS
            )
            references.append(
                ArtifactRef.from_path(
                    selected_path,
                    kind=kind,
                    logical_name=f"validation_vector_manifest_ar{ar}",
                )
            )
            audit.append(record)

        return output_paths, tuple(references), audit


    @classmethod
    def _prepare_genai_validation_vectors(
        cls,
        manifest: RunManifest,
        spec: BuildSpec,
        attached_models_by_ar: Mapping[Any, Any] | None,
    ) -> tuple[dict[int, Path], tuple[ArtifactRef, ...], list[dict[str, Any]]]:
        """Validate independent Qwen3.5 ONNX/vector pairs before packaging."""

        provided = {
            int(ar): Path(path).expanduser().resolve()
            for ar, path in spec.vectors.validation_manifests_by_ar.items()
        }
        if not provided:
            if spec.vectors.validation_manifest is not None:
                raise InvalidSpecError(
                    "GenAI Qwen3.5 workflows require "
                    "vectors.validation_manifests_by_ar; one shared manifest "
                    "cannot prove both independent AR exports",
                    stage="build_genai_container",
                )
            return {}, (), []
        if not isinstance(attached_models_by_ar, Mapping):
            raise InvalidSpecError(
                "GenAI vector binding requires metadata.attached_models_by_ar",
                stage="build_genai_container",
            )
        if len(spec.sequence.context_lengths) != 1:
            raise InvalidSpecError(
                "GenAI per-AR vector binding currently requires exactly one "
                "context length",
                stage="build_genai_container",
            )
        context_length = int(spec.sequence.context_lengths[0])
        missing = sorted(set(int(ar) for ar in spec.sequence.ars) - set(provided))
        if missing:
            raise InvalidSpecError(
                "GenAI validation vectors are missing requested ARs",
                stage="build_genai_container",
                details={"missing_ars": missing},
            )

        output_paths: dict[int, Path] = {}
        refs: list[ArtifactRef] = []
        audit: list[dict[str, Any]] = []
        for ar in (int(value) for value in spec.sequence.ars):
            source = attached_models_by_ar.get(
                str(ar),
                attached_models_by_ar.get(ar),
            )
            if not isinstance(source, Mapping) or source.get("model_path") is None:
                raise InvalidSpecError(
                    f"attached_models_by_ar has no model_path for AR{ar}",
                    stage="build_genai_container",
                )
            target_model = Path(source["model_path"]).expanduser().resolve()
            binding = validate_provided_ar_manifest(
                provided[ar],
                target_model,
                family=spec.family.value,
                ar=ar,
                cl=context_length,
            )
            selected_path = binding.manifest_path
            record = {
                **binding.to_dict(),
                "binding": "provided_independent_ar",
            }
            if binding.needs_onnxruntime_capture:
                selected_path = cls._capture_onnx_reference(
                    binding.manifest_path,
                    target_model,
                    cls._run_dir(manifest)
                    / "vectors"
                    / f"cl{context_length}"
                    / f"ar{ar}",
                    expected_manifest_sha256=binding.manifest_sha256,
                )
                record.update(
                    {
                        "manifest_path": os.fspath(selected_path),
                        "manifest_sha256": sha256_file(selected_path),
                        "reference_source": "onnxruntime",
                        "binding": "provided_independent_ar_with_ort_fallback",
                    }
                )
            output_paths[ar] = selected_path
            refs.append(
                ArtifactRef.from_path(
                    selected_path,
                    kind=ArtifactKind.GOLDEN_VECTORS,
                    logical_name=f"validation_vector_manifest_ar{ar}",
                )
            )
            audit.append(record)
        return output_paths, tuple(refs), audit


    @staticmethod
    def _create_calibration_config(
        adapter: Any,
        spec: BuildSpec,
        output_dir: Path,
    ) -> tuple[Any, tuple[ArtifactRef, ...]]:
        manifest_path = spec.vectors.calibration_manifest
        if manifest_path is None:
            raise InvalidSpecError(
                "calibration requires vectors.calibration_manifest in the production build; "
                "use qairt_prepare_vectors first, then provide its immutable manifest",
                stage="build",
            )
        VectorPreparer.load_manifest(manifest_path)
        input_list = VectorPreparer.write_input_list(
            [manifest_path],
            output_dir / "calibration_input_list.txt",
        )
        if hasattr(adapter, "create_calibration_config"):
            config = adapter.create_calibration_config(
                dataset=input_list,
                act_precision=spec.quantization.act_precision,
                bias_precision=spec.quantization.bias_precision,
                weights_precision=spec.quantization.weights_precision,
                act_calibration_method=spec.quantization.act_calibration_method,
                param_calibration_method=spec.quantization.param_calibration_method,
            )
        elif hasattr(adapter, "_load_module"):
            qairt = adapter._load_module("qairt")
            config = qairt.CalibrationConfig(
                dataset=os.fspath(input_list),
                act_precision=spec.quantization.act_precision,
                bias_precision=spec.quantization.bias_precision,
                weights_precision=spec.quantization.weights_precision,
                act_calibration_method=spec.quantization.act_calibration_method,
                param_calibration_method=spec.quantization.param_calibration_method,
            )
        else:
            raise QairtConfigurationError(
                "adapter does not expose create_calibration_config"
            )
        refs = (
            ArtifactRef.from_path(
                manifest_path,
                kind=ArtifactKind.TEST_VECTORS,
                logical_name="calibration_vector_manifest",
            ),
            ArtifactRef.from_path(
                input_list,
                kind=ArtifactKind.TEST_VECTORS,
                logical_name="calibration_input_list",
            ),
        )
        return config, refs


    def build(
        self,
        spec: BuildSpec | Mapping[str, Any],
        *,
        execution_context: StageExecutionContext | None = None,
    ) -> ToolResult[dict[str, Any]]:
        """Run the production Python API path through context generation."""

        def operation(
            parsed: BuildSpec, manifest: RunManifest, adapter: Any
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            if not has_decoder_lane(parsed.family):
                config_path = self._standalone_vit_config_path(parsed)
                effective = self._standalone_vit_effective_payload(parsed)
                config_ref = self._publish_effective_config(manifest, effective)
                sources = self._source_artifacts(
                    parsed, config_path, require_models=True
                )
                self._preflight(adapter, parsed)
                if parsed.vectors.mode is VectorMode.PROVIDED:
                    for vector_path in (
                        parsed.vectors.validation_manifest,
                        *parsed.vectors.validation_manifests_by_ar.values(),
                        parsed.vectors.calibration_manifest,
                    ):
                        if vector_path is None:
                            continue
                        VectorPreparer.load_tensors(vector_path, section="inputs")
                        vector = VectorPreparer.load_manifest(vector_path)
                        if vector.goldens:
                            VectorPreparer.load_tensors(
                                vector_path, section="goldens"
                            )
                calibration_config = None
                calibration_refs: tuple[ArtifactRef, ...] = ()
                if parsed.quantization.mode is QuantizationMode.CALIBRATE:
                    calibration_config, calibration_refs = (
                        self._create_calibration_config(
                            adapter,
                            parsed,
                            self._run_dir(manifest) / "vectors",
                        )
                    )
                runtime = {
                    "sources": parsed.sources,
                    "sequence": parsed.sequence,
                    "transforms": parsed.transforms,
                    "compile": parsed.compile,
                    "quantization": {
                        **parsed.quantization.model_dump(mode="python"),
                        **(
                            {"calibration_config": calibration_config}
                            if calibration_config is not None
                            else {}
                        ),
                    },
                }
                result = adapter.build_standalone_vit(
                    parsed,
                    runtime,
                    self._run_dir(manifest) / "build",
                )
                per_ar_vectors, vector_refs, vector_audit = (
                    self._prepare_low_level_validation_vectors(
                        manifest,
                        parsed,
                        result,
                    )
                )
                vector_binding_refs: tuple[ArtifactRef, ...] = ()
                if vector_audit:
                    vector_binding_refs = (
                        atomic_publish_json(
                            self._run_dir(manifest)
                            / "config"
                            / "validation_vector_bindings.json",
                            {
                                "schema": "qairt-agent.validation-vector-bindings",
                                "bindings": vector_audit,
                            },
                            kind=ArtifactKind.CONFIG,
                            logical_name="validation_vector_bindings",
                        ),
                    )
                route_refs, route_summaries = self._publish_slice_routes(
                    manifest,
                    result,
                    family=parsed.family,
                )
                runtime_index_ref = self._publish_runtime_index(
                    manifest,
                    parsed,
                    result,
                    lane="low_level",
                    route_refs=route_refs,
                    validation_manifests_by_ar=per_ar_vectors,
                )
                result_refs = _path_artifacts(
                    result, logical_prefix="build.vit."
                )
                artifacts = _unique_artifacts(
                    sources
                    + calibration_refs
                    + vector_refs
                    + vector_binding_refs
                    + (config_ref, runtime_index_ref)
                    + result_refs
                    + route_refs
                )
                footprint = _static_footprint(result, artifacts)
                return {
                    "lane": "standalone_vit_low_level",
                    "build": _jsonable(result),
                    "effective_config": effective,
                    "slice_routes": route_summaries,
                    "validation_vector_bindings": vector_audit,
                    "runtime_index": _jsonable(runtime_index_ref),
                    "static_footprint": footprint,
                    "artifact_count": len(artifacts),
                }, artifacts, {
                    "variant_count": 0,
                    "transformed_slice_count": 0,
                    "converted_model_count": len(result.converted_models),
                    "production_context_count": len(result.contexts),
                    "diagnostic_context_count": len(
                        result.diagnostic_contexts
                    ),
                    "static_footprint": footprint,
                }
            generated, config_path, cross_check = self._generate_family_config(parsed)
            effective = self._effective_payload(parsed, generated, cross_check)
            config_ref = self._publish_effective_config(manifest, effective)
            sources = self._source_artifacts(parsed, config_path, require_models=True)
            self._preflight(adapter, parsed)
            calibration_config = None
            calibration_refs: tuple[ArtifactRef, ...] = ()
            qwen35_refs: tuple[ArtifactRef, ...] = ()
            transform_refs: tuple[ArtifactRef, ...] = ()
            if parsed.vectors.mode is VectorMode.PROVIDED:
                for vector_path in (
                    parsed.vectors.validation_manifest,
                    *parsed.vectors.validation_manifests_by_ar.values(),
                    parsed.vectors.calibration_manifest,
                ):
                    if vector_path is None:
                        continue
                    VectorPreparer.load_tensors(vector_path, section="inputs")
                    vector = VectorPreparer.load_manifest(vector_path)
                    if vector.goldens:
                        VectorPreparer.load_tensors(vector_path, section="goldens")
            transform_input_list: Path | None = None
            transform_input_base_dir: Path | None = None
            if parsed.transforms.mha2sha_validate:
                if parsed.vectors.validation_manifest is None:
                    raise InvalidSpecError(
                        "mha2sha_validate requires vectors.validation_manifest",
                        stage="build",
                    )
                transform_input_list = VectorPreparer.write_input_list(
                    [parsed.vectors.validation_manifest],
                    self._run_dir(manifest) / "vectors" / "transform_input_list.txt",
                )
                transform_input_base_dir = (
                    parsed.vectors.validation_manifest.expanduser().resolve().parent
                )
                transform_refs = (
                    ArtifactRef.from_path(
                        transform_input_list,
                        kind=ArtifactKind.TEST_VECTORS,
                        logical_name="transform_validation_input_list",
                    ),
                )
            if parsed.quantization.mode is QuantizationMode.CALIBRATE:
                calibration_config, calibration_refs = self._create_calibration_config(
                    adapter,
                    parsed,
                    self._run_dir(manifest) / "vectors",
                )
            runtime = self._runtime_config(
                parsed,
                generated,
                calibration_config=calibration_config,
                transform_input_list_path=transform_input_list,
                transform_input_base_dir=transform_input_base_dir,
            )
            # Record each (context length, slice) publish point as it happens.
            # A build that dies partway can then say what completed, instead of
            # leaving a from-zero restart with nothing to report.
            build_progress_path = (
                self._run_dir(manifest) / "stages" / "build" / "progress.jsonl"
            )
            build_kwargs: dict[str, Any] = {
                "on_publish": self._build_progress_recorder(build_progress_path)
            }
            if requires_derivation_evidence(
                parsed.family, len(parsed.sequence.ars)
            ):
                validation_config = parsed.metadata.get("qwen35_runtime_validation")
                if not isinstance(validation_config, Mapping):
                    raise InvalidSpecError(
                        "Qwen3.5 multi-AR build requires "
                        "metadata.qwen35_runtime_validation with per-graph/AR golden cases",
                        stage="build",
                    )
                validator, qwen35_refs = self._qwen35_validator(
                    adapter,
                    validation_config,
                    self._run_dir(manifest) / "diagnostics" / "qwen35-runtime",
                    manifest=manifest,
                    input_manifest_sha256=hashlib.sha256(
                        canonical_json_bytes(manifest)
                    ).hexdigest(),
                )
                build_kwargs["qwen35_runtime_validator"] = validator
                build_kwargs["qwen35_validation_payload"] = _jsonable(validation_config)
            result = adapter.build(
                parsed,
                runtime,
                self._run_dir(manifest) / "build",
                **build_kwargs,
            )
            per_ar_vectors, vector_refs, vector_audit = (
                self._prepare_low_level_validation_vectors(
                    manifest,
                    parsed,
                    result,
                )
            )
            vector_binding_refs: tuple[ArtifactRef, ...] = ()
            if vector_audit:
                vector_binding_refs = (
                    atomic_publish_json(
                        self._run_dir(manifest)
                        / "config"
                        / "validation_vector_bindings.json",
                        {
                            "schema": "qairt-agent.validation-vector-bindings",
                            "bindings": vector_audit,
                        },
                        kind=ArtifactKind.CONFIG,
                        logical_name="validation_vector_bindings",
                    ),
                )
            route_refs, route_summaries = self._publish_slice_routes(
                manifest,
                result,
                family=parsed.family,
            )
            runtime_index_ref = self._publish_runtime_index(
                manifest,
                parsed,
                result,
                lane="low_level",
                route_refs=route_refs,
                validation_manifests_by_ar=per_ar_vectors,
            )
            result_refs = _path_artifacts(result, logical_prefix="build.")
            artifacts = _unique_artifacts(
                sources
                + calibration_refs
                + qwen35_refs
                + transform_refs
                + vector_refs
                + vector_binding_refs
                + (config_ref, runtime_index_ref)
                + result_refs
                + route_refs
            )
            footprint = _static_footprint(result, artifacts)
            data = {
                "build": _jsonable(result),
                "effective_config": effective,
                "slice_routes": route_summaries,
                "validation_vector_bindings": vector_audit,
                "runtime_index": _jsonable(runtime_index_ref),
                "static_footprint": footprint,
                "artifact_count": len(artifacts),
            }
            metrics = {
                "variant_count": len(getattr(result, "variants", ())),
                "transformed_slice_count": len(getattr(result, "transformed_slices", ())),
                "converted_model_count": len(getattr(result, "converted_models", ())),
                "production_context_count": len(getattr(result, "contexts", ())),
                "diagnostic_context_count": len(
                    getattr(result, "diagnostic_contexts", ())
                ),
                "static_footprint": footprint,
            }
            return data, artifacts, metrics

        return self._initial_operation(
            "build",
            spec,
            operation,
            execution_context=execution_context,
        )


    def build_genai_container(
        self,
        spec: BuildSpec | Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
        execution_context: StageExecutionContext | None = None,
    ) -> ToolResult[dict[str, Any]]:
        """Build a production container through QAIRT's GenAI Builder API.

        This lane is intentionally separate from :meth:`build`: invoking both
        APIs in one stage would transform, convert, and compile the same model
        twice.  A GenAI manifest can still be continued with the low-level
        vector, execution, profiling, and diagnosis tools.
        """

        selected = dict(config or {})

        def operation(
            parsed: BuildSpec,
            manifest: RunManifest,
            adapter: Any,
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            if parsed.quantization.mode is not QuantizationMode.APPLY_ENCODINGS:
                raise InvalidSpecError(
                    "the GenAI Builder lane currently requires precomputed AIMET "
                    "encodings; use qairt_build/qairt_quantize for calibration",
                    stage="build_genai_container",
                )
            if not parsed.transforms.mha2sha:
                raise InvalidSpecError(
                    "the GenAI Builder lane requires MHA2SHA; use the "
                    "low-level build lane when MHA2SHA is disabled",
                    stage="build_genai_container",
                )

            generated, config_path, cross_check = self._generate_family_config(parsed)
            source_config, resolved_source_config = self._load_model_config(parsed)
            if config_path != resolved_source_config:
                raise RuntimeError("model config resolution changed during one build")
            effective = self._effective_payload(parsed, generated, cross_check)
            config_ref = self._publish_effective_config(manifest, effective)
            sources = self._source_artifacts(
                parsed,
                config_path,
                require_models=True,
            )
            self._preflight(adapter, parsed)

            text = parsed.sources.text
            vision = parsed.sources.vision
            audio = parsed.sources.audio
            output_dir = (
                self._run_dir(manifest) / "genai" / "container"
                if execution_context is not None
                else Path(
                    selected.get(
                        "output_dir",
                        self._run_dir(manifest) / "genai" / "container",
                    )
                )
            )
            cache_root = (
                self._run_dir(manifest) / "genai" / "cache"
                if execution_context is not None
                else selected.get(
                    "cache_root",
                    self._run_dir(manifest) / "genai" / "cache",
                )
            )
            attached_models_by_ar = selected.get("attached_models_by_ar")
            if attached_models_by_ar is None:
                attached_models_by_ar = parsed.metadata.get(
                    "attached_models_by_ar"
                )
            attached_model_refs = _config_input_artifacts(
                {"attached_models_by_ar": attached_models_by_ar}
            )
            per_ar_vectors, vector_refs, vector_audit = (
                self._prepare_genai_validation_vectors(
                    manifest,
                    parsed,
                    attached_models_by_ar,
                )
            )
            vector_binding_refs: tuple[ArtifactRef, ...] = ()
            if vector_audit:
                vector_binding_refs = (
                    atomic_publish_json(
                        self._run_dir(manifest)
                        / "config"
                        / "validation_vector_bindings.json",
                        {
                            "schema": "qairt-agent.validation-vector-bindings",
                            "bindings": vector_audit,
                        },
                        kind=ArtifactKind.CONFIG,
                        logical_name="validation_vector_bindings",
                    ),
                )
            extra_config_refs: tuple[ArtifactRef, ...] = ()
            if has_audio_component(parsed.family):
                if audio is None or audio.encodings_path is None:
                    raise InvalidSpecError(
                        "qwen3_5_omni requires sources.audio ONNX and AIMET encodings",
                        stage="build_genai_container",
                    )
                text_config_path = config_path
                if text_config_path is None:
                    text_config_ref = atomic_publish_json(
                        self._run_dir(manifest)
                        / "config"
                        / "qwen3_5_omni_source_config.json",
                        source_config,
                        kind=ArtifactKind.CONFIG,
                        logical_name="qwen3_5_omni_source_config",
                    )
                    text_config_path = text_config_ref.path
                    extra_config_refs = (text_config_ref,)
                else:
                    text_config_path = Path(text_config_path).expanduser().resolve()
                    if text_config_path.is_dir():
                        text_config_path = text_config_path / "config.json"
                audio_config_path = audio.config_path or text_config_path
                audio_config_path = Path(audio_config_path).expanduser().resolve()
                if audio_config_path.is_dir():
                    audio_config_path = audio_config_path / "config.json"
                result = adapter.build_qwen35_omni_components(
                    text.onnx_path,
                    audio_model_path=audio.onnx_path,
                    output_dir=output_dir,
                    target=parsed.target.name,
                    split_plan=generated.split_plan,
                    text_encodings_path=text.encodings_path,
                    audio_encodings_path=audio.encodings_path,
                    text_config_path=text_config_path,
                    audio_config_path=audio_config_path,
                    tokenizer_path=text.tokenizer_path,
                    cache_root=cache_root,
                    ar_values=parsed.sequence.ars,
                    context_lengths=parsed.sequence.context_lengths,
                    native_kv=parsed.sequence.native_kv,
                    weight_sharing=parsed.sequence.weight_sharing,
                    attached_models_by_ar=attached_models_by_ar,
                    exist_ok=bool(selected.get("exist_ok", False)),
                )
                lane = "qwen3_5_omni_component_packaging"
            else:
                result = adapter.build_genai_container(
                    text.onnx_path,
                    output_dir=output_dir,
                    family=generated.profile,
                    target=parsed.target.name,
                    split_plan=generated.split_plan,
                    encodings_path=text.encodings_path,
                    vision_model_path=vision.onnx_path if vision is not None else None,
                    vision_encodings_path=(
                        vision.encodings_path if vision is not None else None
                    ),
                    vision_config_path=(
                        vision.config_path if vision is not None else None
                    ),
                    tokenizer_path=text.tokenizer_path,
                    config_path=config_path,
                    config_dict=source_config if config_path is None else None,
                    cache_root=cache_root,
                    ar_values=parsed.sequence.ars,
                    context_lengths=parsed.sequence.context_lengths,
                    native_kv=parsed.sequence.native_kv,
                    weight_sharing=parsed.sequence.weight_sharing,
                    attached_models_by_ar=attached_models_by_ar,
                    exist_ok=bool(selected.get("exist_ok", False)),
                )
                lane = "genai_builder_production_packaging"
            runtime_index_ref = self._publish_runtime_index(
                manifest,
                parsed,
                result,
                lane="genai_builder",
                runtime_supported=bool(result.runtime_supported),
                container_path=result.container_path,
                validation_manifests_by_ar=per_ar_vectors,
            )
            result_refs = _path_artifacts(result, logical_prefix="genai.")
            artifacts = _unique_artifacts(
                sources
                + attached_model_refs
                + vector_refs
                + vector_binding_refs
                + (config_ref, runtime_index_ref)
                + extra_config_refs
                + result_refs
            )
            footprint = _static_footprint(result, artifacts)
            data = {
                "lane": lane,
                "genai_container": _jsonable(result),
                "effective_config": effective,
                "validation_vector_bindings": vector_audit,
                "runtime_index": _jsonable(runtime_index_ref),
                "static_footprint": footprint,
                "artifact_count": len(artifacts),
            }
            metrics = {
                "factory_support": result.factory_support,
                "compatibility_mode": result.compatibility_mode,
                "runtime_supported": result.runtime_supported,
                "container_file_count": len(result_refs),
                "static_footprint": footprint,
            }
            return data, artifacts, metrics

        return self._initial_operation(
            "build_genai_container",
            spec,
            operation,
            operation_config=selected,
            execution_context=execution_context,
        )

