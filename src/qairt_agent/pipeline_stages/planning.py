"""Resolving a spec into the effective build, and publishing what resolved.

`plan` is the contract check before a build: it renders the pipeline, the AR and
native-KV policy, the resolved target and whether hardware has proven it, the
effective compile, and the effective benchmark. The preset routes, but the
supplied config cross-checks it -- an architecture belonging to a different
family fails here rather than silently bypassing every family gate downstream.
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
from qairt_agent.pipeline_stages.build import BuildStage
from qairt_agent.pipeline_stages.diagnose import DiagnoseStage
from qairt_agent.pipeline_stages.execution import ExecutionStage
from qairt_agent.pipeline_stages.validate import ValidateStage
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



class PlanningStage:
    """PlanningStage — see the module docstring."""

    @staticmethod
    def _load_model_config(spec: BuildSpec) -> tuple[dict[str, Any], Path | None]:
        inline = spec.metadata.get("model_config")
        if isinstance(inline, Mapping):
            return dict(inline), None

        config_path = spec.sources.text.config_path
        if config_path is None:
            metadata_path = spec.metadata.get("model_config_path")
            config_path = Path(str(metadata_path)) if metadata_path is not None else None
        if config_path is None:
            raise InvalidSpecError(
                "family config generation requires sources.text.config_path "
                "or metadata.model_config",
                stage="generate_config",
            )
        resolved = config_path.expanduser().resolve()
        if resolved.is_dir():
            resolved = resolved / "config.json"
        if not resolved.is_file():
            raise FileNotFoundError(f"model config does not exist: {resolved}")
        try:
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidSpecError(
                f"cannot parse model config: {resolved}",
                stage="generate_config",
                details={"reason": str(exc)},
            ) from exc
        if not isinstance(value, dict):
            raise InvalidSpecError("model config root must be a JSON object", stage="generate_config")
        return value, resolved


    @classmethod
    def _cross_check_declared_family(
        cls,
        spec: BuildSpec,
        source_config: Mapping[str, Any],
        source_path: Path | None,
    ) -> FamilyCrossCheck:
        """Confront the declared preset with what the supplied config says.

        The preset stays the routing authority -- that is a program decision --
        but authority that is never checked lets a mis-declared export bypass
        every family gate that follows. A config naming an architecture this
        table maps to a *different* family is a contradiction and fails here,
        before any SDK call; anything weaker is carried as a warning so an
        incomplete table cannot block a genuinely new family.
        """

        check = cross_check_declared_family(
            source_config,
            spec.family.value,
            config_path=str(source_path) if source_path is not None else None,
        )
        if check.contradicts:
            raise InvalidSpecError(
                f"declared preset {spec.family.value!r} contradicts the supplied "
                f"model config, which declares architecture(s) "
                f"{', '.join(check.architectures)} belonging to "
                f"{check.implied_family!r}",
                stage="generate_config",
                details=check.to_dict(),
            )
        return check


    @classmethod
    def _generate_family_config(
        cls, spec: BuildSpec
    ) -> tuple[GeneratedFamilyConfig, Path | None, FamilyCrossCheck]:
        source_config, source_path = cls._load_model_config(spec)
        cross_check = cls._cross_check_declared_family(spec, source_config, source_path)
        generated = FamilyConfigGenerator().generate(
            source_config,
            family=spec.family.value,
            ar_values=spec.sequence.ars,
            context_length=spec.sequence.context_lengths[0],
            decoder_slices=spec.split.decoder_slice_count,
            split_embedding=True,
            split_lm_head=spec.split.split_lm_head,
        )
        for context_length in spec.sequence.context_lengths:
            if max(spec.sequence.ars) > context_length:
                raise InvalidSpecError(
                    f"AR{max(spec.sequence.ars)} does not fit CL{context_length}",
                    stage="generate_config",
                )
        cls._validate_vision_boundary(spec, generated)
        return generated, source_path, cross_check


    @staticmethod
    def _validate_vision_boundary(
        spec: BuildSpec, generated: GeneratedFamilyConfig
    ) -> None:
        if spec.family is not ModelFamily.QWEN3_VL:
            return
        assert spec.sources.vision is not None
        vision_path = spec.sources.vision.onnx_path.expanduser().resolve()
        if not vision_path.is_file():
            return
        info = OnnxInspector().inspect(vision_path)
        numeric_widths = {
            int(output.shape[-1])
            for output in info.outputs
            if output.shape and isinstance(output.shape[-1], int)
        }
        if numeric_widths and generated.hidden_size not in numeric_widths:
            raise InvalidSpecError(
                "Qwen3-VL vision/projector output width does not match text hidden_size",
                stage="generate_config",
                details={
                    "vision_output_widths": sorted(numeric_widths),
                    "text_hidden_size": generated.hidden_size,
                },
            )


    @staticmethod
    def _effective_payload(
        spec: BuildSpec,
        generated: GeneratedFamilyConfig,
        cross_check: FamilyCrossCheck | None = None,
    ) -> dict[str, Any]:
        payload = generated.to_dict()
        if cross_check is not None:
            payload["family_cross_check"] = cross_check.to_dict()
        payload.update(
            {
                "schema": "qairt-agent.effective-config",
                "family": spec.family.value,
                "sequence": spec.sequence.model_dump(mode="json"),
                "split": spec.split.model_dump(mode="json"),
                "transforms": {
                    **spec.transforms.model_dump(mode="json"),
                    "permute_kv_cache_io": (
                        spec.sequence.native_kv
                        or spec.transforms.permute_kv_cache_io
                    ),
                },
                "quantization": spec.quantization.model_dump(mode="json"),
                "compile": spec.compile.model_dump(mode="json"),
                "benchmark": effective_benchmark_policy(spec),
                "target": spec.target.model_dump(mode="json"),
                "sources": spec.sources.model_dump(mode="json", exclude_none=True),
                "embedding_packaging": {
                    "mode": spec.split.embedding_mode.value,
                    "semantic_slice": True,
                    "compile_context": spec.split.embedding_mode is EmbeddingMode.COMPILED,
                },
                "context_binary_grouping": "one_per_semantic_slice_and_context_length",
                "quality_policy": "report_only",
            }
        )
        if has_vision_component(spec.family):
            payload["workflow"] = {
                "api": "qairt.gen_ai_api.builders.WorkflowBuilder.from_workflow",
                "nodes": [
                    {
                        "name": "vision_encoder",
                        "role": "IMAGE_ENCODER",
                        "projector": "inside_vision_onnx",
                    },
                    {"name": "text_generator", "role": "TEXT_GENERATOR"},
                ],
                "connections": [["vision_encoder", "text_generator"]],
            }
        elif has_audio_component(spec.family):
            payload["workflow"] = {
                "api": (
                    "qairt.gen_ai_api.builders.workflow_builder."
                    "WorkflowBuilder.from_builders"
                ),
                "nodes": [
                    {"name": "audioEncoder", "role": "AUDIO_ENCODER"},
                    {"name": "textGenerator", "role": "TEXT_GENERATOR"},
                ],
                "connections": [["audioEncoder", "textGenerator"]],
                "runtime_supported": False,
                "runtime_boundary": (
                    "QAIRT 2.49 builds both components but does not provide "
                    "validated end-to-end audio workflow execution"
                ),
            }
        return payload


    @staticmethod
    def _runtime_config(
        spec: BuildSpec,
        generated: GeneratedFamilyConfig,
        *,
        calibration_config: Any = None,
        transform_input_list_path: Path | None = None,
        transform_input_base_dir: Path | None = None,
    ) -> dict[str, Any]:
        quantization = spec.quantization.model_dump(mode="python")
        if calibration_config is not None:
            quantization["calibration_config"] = calibration_config
        transforms = spec.transforms.model_dump(mode="python")
        if transform_input_list_path is not None:
            transforms["input_raw_list_path"] = transform_input_list_path
        if transform_input_base_dir is not None:
            transforms["input_raw_base_dir"] = transform_input_base_dir
        return {
            "profile": generated.profile,
            "family": generated.family,
            "sources": spec.sources,
            "sequence": spec.sequence,
            "split": spec.split,
            "split_plan": generated.split_plan,
            "num_hidden_layers": generated.num_hidden_layers,
            "quantization": quantization,
            "transforms": transforms,
            "compile": spec.compile,
        }


    @staticmethod
    def _state_slot(tensor_name: str) -> str | None:
        lowered = tensor_name.lower()
        state_tokens = (
            "past_key",
            "past_value",
            "present_key",
            "present_value",
            "key_cache",
            "value_cache",
            "kv_cache",
            "recurrent_state",
            "conv_state",
        )
        if not any(token in lowered for token in state_tokens):
            return None
        normalized = lowered
        for old, new in (
            ("present_key", "key"),
            ("past_key", "key"),
            ("present_value", "value"),
            ("past_value", "value"),
            ("_input", ""),
            ("_output", ""),
            ("_in", ""),
            ("_out", ""),
        ):
            normalized = normalized.replace(old, new)
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
        return normalized


    @classmethod
    def _publish_slice_routes(
        cls,
        manifest: RunManifest,
        result: Any,
        *,
        family: ModelFamily | str | None = None,
    ) -> tuple[tuple[ArtifactRef, ...], list[dict[str, Any]]]:
        """Publish executable route skeletons derived from transformed ONNX I/O."""

        transformed = tuple(getattr(result, "transformed_slices", ()))
        family_name = str(getattr(family, "value", family) or "")
        all_contexts = tuple(getattr(result, "contexts", ()))
        contexts = tuple(
            item
            for item in all_contexts
            if getattr(item, "context_length", None) is not None
            and not (
                family_name == ModelFamily.QWEN3_VL.value
                and getattr(item, "slice_name", None) == "vision_projector"
            )
        )
        if not transformed or not contexts:
            return (), []

        inspector = OnnxInspector()
        route_refs: list[ArtifactRef] = []
        route_summaries: list[dict[str, Any]] = []
        context_lengths = sorted(
            {
                int(item.context_length)
                for item in contexts
                if item.context_length is not None
            }
        )
        for context_length in context_lengths:
            contexts_by_slice = {
                str(item.slice_name): item
                for item in contexts
                if item.context_length == context_length and item.slice_name is not None
            }
            candidates: dict[str, list[Any]] = {}
            for item in transformed:
                if item.context_length == context_length and item.slice_name in contexts_by_slice:
                    candidates.setdefault(str(item.slice_name), []).append(item)
            ordered_slices = sorted(
                candidates,
                key=lambda name: min(item.split_index for item in candidates[name]),
            )
            routes: list[dict[str, Any]] = []
            available_outputs: set[str] = set()
            for slice_name in ordered_slices:
                representative = min(
                    candidates[slice_name],
                    key=lambda item: int(item.ar or 0),
                )
                info = inspector.inspect(representative.model_path)
                input_names = tuple(item.name for item in info.inputs)
                output_names = tuple(item.name for item in info.outputs)
                state_inputs = {
                    name: slot
                    for name in input_names
                    if (slot := cls._state_slot(name)) is not None
                }
                state_outputs = {
                    name: slot
                    for name in output_names
                    if (slot := cls._state_slot(name)) is not None
                }
                from_previous = {
                    name: name
                    for name in input_names
                    if name in available_outputs and name not in state_inputs
                }
                context = contexts_by_slice[slice_name]
                graph_names = {
                    str(ar): graph_name
                    for ar, graph_name in zip(
                        context.ar_values,
                        context.graph_names,
                    )
                }
                unresolved = [
                    name
                    for name in input_names
                    if name not in from_previous and name not in state_inputs
                ]
                routes.append(
                    {
                        "slice_id": slice_name,
                        "input_names": list(input_names),
                        "output_names": list(output_names),
                        "graph_names": graph_names,
                        "from_previous": from_previous,
                        "state_inputs": state_inputs,
                        "state_outputs": state_outputs,
                        "unresolved_external_inputs": unresolved,
                    }
                )
                available_outputs.update(output_names)
            routed_slice_ids = {
                str(route["slice_id"])
                for route in routes
            }
            excluded_components = sorted(
                {
                    str(item.slice_name)
                    for item in all_contexts
                    if item.slice_name is not None
                    and str(item.slice_name) not in routed_slice_ids
                }
            )
            is_vl_text_component = (
                family_name == ModelFamily.QWEN3_VL.value
            )
            payload = {
                "schema": "qairt-agent.slice-routes",
                "context_length": context_length,
                "component": (
                    "text" if is_vl_text_component else "model"
                ),
                "coverage": (
                    "text_only" if is_vl_text_component else "full_model"
                ),
                "excluded_components": excluded_components,
                "contexts": {
                    slice_name: os.fspath(context.context_binary_path)
                    for slice_name, context in contexts_by_slice.items()
                },
                "routes": routes,
                "routing_policy": (
                    "exact tensor names are auto-routed; unresolved inputs must "
                    "be supplied explicitly or the route artifact must be reviewed"
                ),
            }
            route_ref = atomic_publish_json(
                cls._run_dir(manifest)
                / "config"
                / f"slice_routes_cl{context_length}.json",
                payload,
                kind=ArtifactKind.CONFIG,
                logical_name=f"slice_routes_cl{context_length}",
            )
            route_refs.append(route_ref)
            route_summaries.append(
                {
                    "context_length": context_length,
                    "artifact": _jsonable(route_ref),
                    "slice_count": len(routes),
                    "component": payload["component"],
                    "coverage": payload["coverage"],
                    "excluded_components": excluded_components,
                }
            )
        return tuple(route_refs), route_summaries


    @classmethod
    def _publish_runtime_index(
        cls,
        manifest: RunManifest,
        spec: BuildSpec,
        result: Any,
        *,
        lane: str,
        route_refs: Sequence[ArtifactRef] = (),
        runtime_supported: bool = True,
        container_path: str | Path | None = None,
        validation_manifests_by_ar: Mapping[int | str, str | Path] | None = None,
    ) -> ArtifactRef:
        """Publish the exact build outputs consumed by later workflow stages."""

        per_ar = (
            dict(spec.vectors.validation_manifests_by_ar)
            if validation_manifests_by_ar is None
            else dict(validation_manifests_by_ar)
        )
        payload = make_runtime_index(
            result=result,
            lane=lane,
            family=spec.family.value,
            default_ar=int(spec.sequence.ars[0]),
            default_context_length=int(spec.sequence.context_lengths[0]),
            route_artifacts=tuple(route_refs),
            validation_manifest=spec.vectors.validation_manifest,
            validation_manifests_by_ar=per_ar,
            runtime_supported=runtime_supported,
            container_path=container_path,
        )
        return atomic_publish_json(
            cls._run_dir(manifest) / "config" / "runtime_index.json",
            payload,
            kind=ArtifactKind.CONFIG,
            logical_name="runtime_index",
        )


    @staticmethod
    def _runtime_index_for_manifest(manifest: RunManifest) -> dict[str, Any]:
        candidates = [
            artifact
            for artifact in manifest.artifacts
            if artifact.logical_name == "runtime_index"
        ]
        if len(candidates) != 1:
            raise InvalidSpecError(
                "automatic execution requires exactly one build runtime_index artifact",
                stage="runtime_binding",
                details={
                    "runtime_index_count": len(candidates),
                    "hint": (
                        "rerun build with the current qairt-agent or provide "
                        "explicit stage_configs"
                    ),
                },
            )
        verify_artifact(candidates[0])
        return load_runtime_index(candidates[0].path)


    @classmethod
    def _automatic_runtime_binding(
        cls,
        manifest: RunManifest,
        selected: Mapping[str, Any],
    ) -> dict[str, Any]:
        index = cls._runtime_index_for_manifest(manifest)
        binding = select_runtime_binding(
            index,
            ar=(
                int(selected["ar"])
                if selected.get("ar") is not None
                else None
            ),
            context_length=(
                int(selected["context_length"])
                if selected.get("context_length") is not None
                else None
            ),
            component=(
                str(selected["component"])
                if selected.get("component") is not None
                else None
            ),
        )
        return {**binding, **dict(selected)}


    @staticmethod
    def _publish_effective_config(
        manifest: RunManifest,
        payload: Mapping[str, Any],
    ) -> ArtifactRef:
        path = run_directory(manifest) / "config" / "effective_config.json"
        return atomic_publish_json(
            path,
            dict(payload),
            kind=ArtifactKind.CONFIG,
            logical_name="effective_config",
        )


    @classmethod
    def _source_artifacts(
        cls,
        spec: BuildSpec,
        config_path: Path | None,
        *,
        require_models: bool,
    ) -> tuple[ArtifactRef, ...]:
        paths: list[tuple[Path, ArtifactKind, str]] = []
        onnx_sources: list[tuple[str, Path]] = []
        if config_path is not None:
            paths.append((config_path, ArtifactKind.CONFIG, "model_config"))
        for source_name, source in (
            ("text", spec.sources.text),
            ("vision", spec.sources.vision),
            ("audio", spec.sources.audio),
        ):
            if source is None:
                continue
            paths.append((source.onnx_path, ArtifactKind.ONNX, f"{source_name}_onnx"))
            onnx_sources.append((source_name, source.onnx_path))
            if source.encodings_path is not None:
                paths.append(
                    (
                        source.encodings_path,
                        ArtifactKind.AIMET_ENCODINGS,
                        f"{source_name}_encodings",
                    )
                )
            if source_name != "text" and source.config_path is not None:
                source_config_path = source.config_path.expanduser()
                if source_config_path.is_dir():
                    source_config_path = source_config_path / "config.json"
                paths.append(
                    (
                        source_config_path,
                        ArtifactKind.CONFIG,
                        f"{source_name}_model_config",
                    )
                )
            if source.tokenizer_path is not None:
                tokenizer_path = source.tokenizer_path.expanduser()
                if tokenizer_path.is_dir():
                    tokenizer_path = tokenizer_path / "tokenizer.json"
                paths.append(
                    (
                        tokenizer_path,
                        ArtifactKind.CONFIG,
                        f"{source_name}_tokenizer",
                    )
                )
            if source.aimet_config_path is not None:
                paths.append(
                    (
                        source.aimet_config_path,
                        ArtifactKind.CONFIG,
                        f"{source_name}_aimet_config",
                    )
                )
        if spec.vectors.validation_manifest is not None:
            paths.append(
                (
                    spec.vectors.validation_manifest,
                    ArtifactKind.GOLDEN_VECTORS,
                    "validation_vector_manifest",
                )
            )
        for ar, vector_path in sorted(
            spec.vectors.validation_manifests_by_ar.items()
        ):
            paths.append(
                (
                    vector_path,
                    ArtifactKind.GOLDEN_VECTORS,
                    f"validation_vector_manifest_ar{ar}",
                )
            )
        if spec.vectors.calibration_manifest is not None:
            paths.append(
                (
                    spec.vectors.calibration_manifest,
                    ArtifactKind.TEST_VECTORS,
                    "calibration_vector_manifest",
                )
            )
        if require_models:
            inspector = OnnxInspector()
            for source_name, onnx_path in onnx_sources:
                resolved_onnx = onnx_path.expanduser().resolve()
                if not resolved_onnx.is_file():
                    continue
                for index, external_path in enumerate(
                    inspector.external_data_paths(resolved_onnx)
                ):
                    paths.append(
                        (
                            external_path,
                            ArtifactKind.OTHER,
                            f"{source_name}_onnx_external_data_{index:03d}",
                        )
                    )
        refs: list[ArtifactRef] = []
        for path, kind, name in paths:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                if require_models or kind is ArtifactKind.CONFIG:
                    raise FileNotFoundError(f"source artifact does not exist: {resolved}")
                continue
            refs.append(ArtifactRef.from_path(resolved, kind=kind, logical_name=name))
        return tuple(refs)


    def generate_config(
        self, spec: BuildSpec | Mapping[str, Any]
    ) -> ToolResult[dict[str, Any]]:
        """Generate family, shape, split, embedding, and workflow configuration."""

        def operation(
            parsed: BuildSpec, manifest: RunManifest, _adapter: Any
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            if not has_decoder_lane(parsed.family):
                config_path = self._standalone_vit_config_path(parsed)
                payload = self._standalone_vit_effective_payload(parsed)
                config_ref = self._publish_effective_config(manifest, payload)
                source_refs = self._source_artifacts(
                    parsed, config_path, require_models=False
                )
                return payload, source_refs + (config_ref,), {
                    "decoder_layers": 0,
                    "decoder_slices": 0,
                    "semantic_slices": 1,
                }
            generated, config_path, cross_check = self._generate_family_config(parsed)
            payload = self._effective_payload(parsed, generated, cross_check)
            config_ref = self._publish_effective_config(manifest, payload)
            source_refs = self._source_artifacts(parsed, config_path, require_models=False)
            artifacts = source_refs + (config_ref,)
            return payload, artifacts, {
                "decoder_layers": generated.num_hidden_layers,
                "decoder_slices": len(generated.split_plan.decoder_slices),
                "semantic_slices": generated.split_plan.num_splits,
            }

        return self._initial_operation("generate_config", spec, operation)


    def plan(
        self,
        spec: BuildSpec | Mapping[str, Any],
        *,
        offline: bool = False,
    ) -> ToolResult[dict[str, Any]]:
        """Publish a deterministic plan; online mode also enforces SDK preflight."""

        def operation(
            parsed: BuildSpec, manifest: RunManifest, adapter: Any
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            if not has_decoder_lane(parsed.family):
                config_path = self._standalone_vit_config_path(parsed)
                payload = self._standalone_vit_effective_payload(parsed)
                config_ref = self._publish_effective_config(manifest, payload)
                source_refs = self._source_artifacts(
                    parsed, config_path, require_models=False
                )
                preflight = None if offline else self._preflight(adapter, parsed)
                plan_data = {
                    "offline": offline,
                    "effective_config": payload,
                    "preflight": _jsonable(preflight),
                    "production_lanes": {
                        "low_level_contexts": "qairt_build",
                        "genai_container": None,
                        "mutually_composed": False,
                        "reason": "standalone ViT uses the low-level converter/compiler lane",
                    },
                    "stages": [
                        "prepare_vectors",
                        "convert",
                        "quantize_or_apply_encodings",
                        "compile_context",
                        "validate",
                        "benchmark",
                    ],
                    "contexts": [
                        {
                            "context_length": None,
                            "slice": "vit",
                            "ars": [1],
                            "weight_sharing": False,
                        }
                    ],
                }
                return plan_data, source_refs + (config_ref,), {
                    "offline": offline,
                    "context_count": 1,
                }
            generated, config_path, cross_check = self._generate_family_config(parsed)
            payload = self._effective_payload(parsed, generated, cross_check)
            config_ref = self._publish_effective_config(manifest, payload)
            source_refs = self._source_artifacts(parsed, config_path, require_models=False)
            preflight = None if offline else self._preflight(adapter, parsed)
            plan_data = {
                "offline": offline,
                "effective_config": payload,
                "preflight": _jsonable(preflight),
                "production_lanes": {
                    "low_level_contexts": "qairt_build",
                    "genai_container": "qairt_build_genai_container",
                    "mutually_composed": False,
                    "reason": (
                        "GenAIBuilderHTP.build already transforms, converts, "
                        "quantizes, and compiles"
                    ),
                },
                "stages": [
                    "prepare_vectors",
                    "ar_convert",
                    "split+mha2sha",
                    "convert",
                    "quantize_or_apply_encodings",
                    "compile_context",
                    "validate",
                    "benchmark",
                ],
                "contexts": [
                    {
                        "context_length": context_length,
                        "slice": slice_spec.name,
                        "ars": list(parsed.sequence.ars),
                        "weight_sharing": parsed.sequence.weight_sharing,
                    }
                    for context_length in parsed.sequence.context_lengths
                    for slice_spec in generated.split_plan.slices
                    if not (
                        slice_spec.name == "embedding"
                        and parsed.split.embedding_mode is not EmbeddingMode.COMPILED
                    )
                ],
            }
            return plan_data, source_refs + (config_ref,), {
                "offline": offline,
                "context_count": len(plan_data["contexts"]),
            }

        return self._initial_operation(
            "plan",
            spec,
            operation,
            operation_config={"offline": offline},
        )

