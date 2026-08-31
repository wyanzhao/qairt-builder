"""The per-stage low-level tools.

A deprecated compatibility and debugging surface: normal long transform,
compile and profiling work goes through the CLI's detached job worker and its
resumable journal. Each tool is a continuation operation over a verified
manifest, so the evidence discipline is the same as a full workflow's.
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



class StageToolsStage:
    """StageToolsStage — see the module docstring."""

    def prepare_vectors(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> ToolResult[dict[str, Any]]:
        """Import explicit vectors and optionally capture ONNX golden outputs."""

        selected = dict(config or {})

        def operation(
            manifest: RunManifest, _adapter: Any, output_dir: Path
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            preparer = VectorPreparer(output_dir)
            outputs: list[Path] = []
            existing = selected.get("manifest_path")
            if existing is not None:
                expected = selected.get("manifest_sha256")
                VectorPreparer.load_manifest(existing, expected_sha256=expected)
                outputs.append(Path(str(existing)).expanduser().resolve())

            cases = selected.get("cases", ())
            if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
                raise ValueError("prepare_vectors config.cases must be a sequence")
            for case in cases:
                if not isinstance(case, Mapping):
                    raise TypeError("each vector case must be an object")
                path = preparer.prepare_case(
                    str(case["case_id"]),
                    dict(case["inputs"]),
                    goldens=dict(case.get("goldens", {})),
                    roles=dict(case.get("roles", {})),
                    metadata=dict(case.get("metadata", {})),
                )
                if bool(case.get("capture_onnx", False)):
                    path = preparer.capture_onnx(
                        path,
                        manifest.build_spec.sources.text.onnx_path,
                        output_names=case.get("output_names"),
                    )
                outputs.append(path)
            if not outputs:
                raise ValueError("provide config.manifest_path or at least one config.cases entry")
            refs = tuple(
                ArtifactRef.from_path(
                    path,
                    kind=ArtifactKind.TEST_VECTORS,
                    logical_name=f"vector_manifest_{index}",
                )
                for index, path in enumerate(outputs)
            )
            return {
                "vector_manifests": [_jsonable(ref) for ref in refs],
            }, refs, {"case_count": len(refs)}

        return self._continuation_operation(
            "prepare_vectors",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
        )


    def ar_convert(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> ToolResult[dict[str, Any]]:
        selected = dict(config or {})

        def operation(
            manifest: RunManifest, adapter: Any, output_dir: Path
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            self._preflight(adapter, manifest.build_spec)
            spec = manifest.build_spec
            ar = int(selected.get("ar", spec.sequence.ars[0]))
            context_length = int(
                selected.get("context_length", spec.sequence.context_lengths[0])
            )
            artifact = adapter.ar_convert(
                selected.get("model_path", spec.sources.text.onnx_path),
                ar=ar,
                context_length=context_length,
                output_dir=output_dir,
                encodings_path=selected.get(
                    "encodings_path", spec.sources.text.encodings_path
                ),
                family=spec.family.value,
                source_kind=str(selected.get("source_kind", "derived")),
                allow_experimental_qwen35=spec.sequence.qwen35_experimental_auto_ar,
                prefix=selected.get("prefix"),
            )
            refs = _path_artifacts(artifact, logical_prefix="ar_convert.")
            return _jsonable(artifact), refs, {"ar": ar, "context_length": context_length}

        return self._continuation_operation(
            "ar_convert",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
        )


    def _transform(
        self,
        stage_name: str,
        manifest_uri: str | Path,
        manifest_sha256: str,
        config: Mapping[str, Any] | None,
    ) -> ToolResult[dict[str, Any]]:
        selected = dict(config or {})

        def operation(
            manifest: RunManifest, adapter: Any, output_dir: Path
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            spec = manifest.build_spec
            self._preflight(adapter, spec)
            generated, _, _ = self._generate_family_config(spec)
            default_mha2sha = stage_name == "mha2sha"
            run_mha2sha = bool(selected.get("mha2sha", default_mha2sha))
            artifacts = adapter.transform(
                selected.get("model_path", spec.sources.text.onnx_path),
                split_plan=generated.split_plan,
                family=spec.family.value,
                output_dir=output_dir,
                encodings_path=selected.get(
                    "encodings_path", spec.sources.text.encodings_path
                ),
                mha2sha=run_mha2sha,
                native_kv=bool(
                    selected.get(
                        "permute_kv_cache_io",
                        spec.sequence.native_kv
                        or spec.transforms.permute_kv_cache_io,
                    )
                ),
                validate=bool(
                    selected.get("validate", spec.transforms.mha2sha_validate)
                ),
                input_raw_list_path=selected.get("input_raw_list_path"),
                input_raw_base_dir=selected.get("input_raw_base_dir"),
                m2s_head_split_map=selected.get("m2s_head_split_map"),
                adapt_moe=selected.get("adapt_moe"),
            )
            refs = _path_artifacts(artifacts, logical_prefix=f"{stage_name}.")
            return {
                "transform": "split+mha2sha" if run_mha2sha else "split",
                "artifacts": _jsonable(artifacts),
            }, refs, {"slice_count": len(artifacts)}

        return self._continuation_operation(
            stage_name,
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
        )


    def split(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> ToolResult[dict[str, Any]]:
        return self._transform("split", manifest_uri, manifest_sha256, config)


    def mha2sha(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> ToolResult[dict[str, Any]]:
        return self._transform("mha2sha", manifest_uri, manifest_sha256, config)


    def convert(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> ToolResult[dict[str, Any]]:
        selected = dict(config or {})

        def operation(
            manifest: RunManifest, adapter: Any, output_dir: Path
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            spec = manifest.build_spec
            self._preflight(adapter, spec)
            artifact = adapter.convert(
                selected.get("model_path", spec.sources.text.onnx_path),
                encodings_path=selected.get(
                    "encodings_path", spec.sources.text.encodings_path
                ),
                output_path=selected.get("output_path", output_dir / "model.dlc"),
                **dict(selected.get("options", {})),
            )
            refs = _path_artifacts(artifact, logical_prefix="convert.")
            return _jsonable(artifact), refs, {"quantization_mode": artifact.quantization_mode}

        return self._continuation_operation(
            "convert",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
        )


    def quantize(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> ToolResult[dict[str, Any]]:
        selected = dict(config or {})

        def operation(
            manifest: RunManifest, adapter: Any, output_dir: Path
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            self._preflight(adapter, manifest.build_spec)
            if "input_dlc" not in selected:
                raise ValueError("quantize requires config.input_dlc")
            artifact = adapter.quantize(
                selected["input_dlc"],
                output_dlc=selected.get("output_dlc", output_dir / "quantized.dlc"),
                input_list=selected.get("input_list"),
                **dict(selected.get("options", {})),
            )
            refs = _path_artifacts(artifact, logical_prefix="quantize.")
            return _jsonable(artifact), refs, {}

        return self._continuation_operation(
            "quantize",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
        )


    def compile_context(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> ToolResult[dict[str, Any]]:
        selected = dict(config or {})

        def operation(
            manifest: RunManifest, adapter: Any, output_dir: Path
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            spec = manifest.build_spec
            self._preflight(adapter, spec)
            models = tuple(selected.get("models", ()))
            if not models:
                raise ValueError("compile_context requires config.models")
            ars = tuple(int(value) for value in selected.get("ar_values", spec.sequence.ars))
            graph_names = tuple(
                str(value)
                for value in selected.get(
                    "graph_names", tuple(f"graph_ar{ar}" for ar in ars)
                )
            )
            source_kinds = tuple(
                str(value)
                for value in selected.get("source_kinds", ("derived",) * len(models))
            )
            raw_expectations = selected.get("native_kv_expectations", ())
            if not isinstance(raw_expectations, Sequence) or isinstance(
                raw_expectations,
                (str, bytes, bytearray),
            ):
                raise ValueError("native_kv_expectations must be a sequence")
            native_kv_expectations: list[NativeKvGraphExpectation] = []
            for index, value in enumerate(raw_expectations):
                if isinstance(value, NativeKvGraphExpectation):
                    native_kv_expectations.append(value)
                    continue
                if not isinstance(value, Mapping):
                    raise TypeError(
                        f"native_kv_expectations[{index}] must be an object"
                    )
                try:
                    native_kv_expectations.append(
                        NativeKvGraphExpectation.from_dict(value)
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid native_kv_expectations[{index}]: {exc}"
                    ) from exc
            evidence = selected.get("qwen35_validation_evidence")
            if isinstance(evidence, Mapping):
                raise InvalidSpecError(
                    "qwen35_validation_evidence cannot be reconstructed from JSON; "
                    "it must be minted and consumed by the same QairtSdkAdapter "
                    "invocation, or use qairt_build with runtime validation",
                    stage="compile_context",
                )
            if evidence is not None and not isinstance(
                evidence,
                Qwen35ValidationEvidence,
            ):
                raise TypeError(
                    "qwen35_validation_evidence must be Qwen35ValidationEvidence"
                )
            artifact = adapter.compile_context(
                models,
                output_path=selected.get("output_path", output_dir / "context.bin"),
                graph_names=graph_names,
                ar_values=ars,
                source_kinds=source_kinds,
                target_soc=spec.target.chipset,
                dsp_arch=spec.target.dsp_arch,
                soc_model=spec.target.soc_model,
                family=spec.family.value,
                slice_name=selected.get("slice_name"),
                weight_sharing=bool(
                    selected.get("weight_sharing", spec.sequence.weight_sharing)
                ),
                native_kv_config=selected.get("native_kv_config"),
                native_kv_expectations=tuple(native_kv_expectations),
                expect_native_kv=bool(
                    selected.get("expect_native_kv", spec.sequence.native_kv)
                ),
                context_length=int(
                    selected.get(
                        "context_length", spec.sequence.context_lengths[0]
                    )
                ),
                qwen35_validation_evidence=evidence,
                compile_config_options=dict(
                    selected.get("compile_config_options", spec.compile.compiler_options)
                ),
            )
            refs = _path_artifacts(artifact, logical_prefix="compile_context.")
            return _jsonable(artifact), refs, {
                "graph_count": len(graph_names),
                "weight_sharing": artifact.weight_sharing,
            }

        return self._continuation_operation(
            "compile_context",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
        )

