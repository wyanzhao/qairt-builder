"""Manifest-driven orchestration for the Python-only QAIRT agent.

The public facade deliberately stores no current model, compiled context,
device handle, tensor, or native-KV state.  A continuation call must identify
an immutable manifest by both path and SHA256.
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
from qairt_agent.pipeline_stages.benchmark_one import BenchmarkOneStage
from qairt_agent.pipeline_stages.build import BuildStage
from qairt_agent.pipeline_stages.diagnose import DiagnoseStage
from qairt_agent.pipeline_stages.execution import ExecutionStage
from qairt_agent.pipeline_stages.optrace import OptraceStage
from qairt_agent.pipeline_stages.float_reference import FloatReferenceStage
from qairt_agent.pipeline_stages.planning import PlanningStage
from qairt_agent.pipeline_stages.qwen35_derivation import Qwen35DerivationStage
from qairt_agent.pipeline_stages.stage_tools import StageToolsStage
from qairt_agent.pipeline_stages.validate import ValidateStage
from qairt_agent.pipeline_stages.vectors_for_quality import QualityVectorStage
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


class QairtAgent(
    PlanningStage,
    BuildStage,
    Qwen35DerivationStage,
    ValidateStage,
    QualityVectorStage,
    FloatReferenceStage,
    BenchmarkStage,
    BenchmarkOneStage,
    OptraceStage,
    ExecutionStage,
    DiagnoseStage,
    StageToolsStage,
):
    """Synchronous Python facade used directly and through MCP."""

    def __init__(
        self,
        *,
        adapter: QairtAdapterProtocol | None = None,
        adapter_factory: QairtAdapterFactory | None = None,
        device_runtime: DeviceRuntime | Any | None = None,
    ) -> None:
        if adapter is not None and adapter_factory is not None:
            raise ValueError("provide adapter or adapter_factory, not both")
        self._adapter_override = adapter
        self._adapter_factory = adapter_factory
        self._device_runtime = device_runtime or DeviceRuntime()

    def _new_adapter(self) -> QairtAdapterProtocol:
        if self._adapter_factory is not None:
            return self._adapter_factory()
        if self._adapter_override is not None:
            return self._adapter_override
        return QairtSdkAdapter()

    @staticmethod
    def _parse_spec(spec: BuildSpec | Mapping[str, Any]) -> BuildSpec:
        try:
            parsed = (
                spec
                if isinstance(spec, BuildSpec)
                else BuildSpec.model_validate(spec)
            )
            # A requested intermediate dump must never contaminate the
            # production context.  The low-level adapter interprets this flag
            # by compiling an additional diagnostic context while leaving the
            # production context unchanged.  Materialize the effective policy
            # in BuildSpec itself so the manifest and cache identity record it.
            if (
                parsed.quality.sqnr_modes
                and parsed.quality.dump_intermediates_on_failure
                and not parsed.compile.enable_intermediate_outputs
                and not parsed.compile.output_tensors
            ):
                parsed = parsed.model_copy(
                    update={
                        "compile": parsed.compile.model_copy(
                            update={"enable_intermediate_outputs": True}
                        )
                    }
                )
            # Same discipline for the GenAI lane's benchmark sampling policy:
            # resolve it once, here, where "the caller did not set this" is
            # still knowable, so plan output, the manifest, and every later
            # stage read the numbers that will actually run.  `to_build_spec`
            # applies the identical rule for the workflow-spec entry point.
            benchmark = apply_lane_benchmark_defaults(
                parsed.family, parsed.benchmark
            )
            if benchmark != parsed.benchmark:
                parsed = parsed.model_copy(update={"benchmark": benchmark})
            return parsed
        except ValidationError as exc:
            raise InvalidSpecError(
                "invalid BuildSpec",
                stage="spec",
                details={"validation_errors": exc.errors(include_url=False)},
            ) from exc

    @staticmethod
    def _error(exc: BaseException, stage: str) -> ToolErrorData:
        if isinstance(exc, InvalidSpecError):
            return ToolErrorData.from_exception(exc, stage=stage)
        if isinstance(exc, (ValidationError, ValueError, TypeError, KeyError)):
            code = ErrorCode.INVALID_SPEC
        elif isinstance(exc, QairtSdkImportError):
            code = ErrorCode.QAIRT_UNAVAILABLE
        elif isinstance(exc, QairtPreflightError):
            code = ErrorCode.PREFLIGHT_FAILED
        elif isinstance(exc, (ExperimentalFeatureError, QairtConfigurationError)):
            code = ErrorCode.STAGE_FAILED
        elif isinstance(exc, QairtAdapterError):
            code = ErrorCode.STAGE_FAILED
        elif isinstance(exc, FileNotFoundError):
            code = ErrorCode.ARTIFACT_NOT_FOUND
        else:
            code = ErrorCode.INTERNAL_ERROR
        return ToolErrorData.from_exception(exc, code=code, stage=stage)

    @staticmethod
    def _store_for_spec(spec: BuildSpec) -> ManifestStore:
        return ManifestStore(spec.output_root / "manifests")

    @staticmethod
    def _store_for_manifest(manifest_path: str | Path) -> ManifestStore:
        path = Path(manifest_path).expanduser().resolve()
        if len(path.parents) < 2:
            raise ValueError(f"manifest path has no store root: {path}")
        return ManifestStore(path.parent.parent)

    @staticmethod
    def _run_dir(manifest: RunManifest) -> Path:
        return manifest.build_spec.output_root.expanduser().resolve() / "runs" / str(manifest.run_id)

    @staticmethod
    def _attempt(manifest: RunManifest, name: str) -> int:
        execution_attempt = manifest.metadata.get(_EXECUTION_ATTEMPT_METADATA)
        if isinstance(execution_attempt, int) and execution_attempt > 0:
            return execution_attempt
        return 1 + max(
            (stage.attempt for stage in manifest.stages if stage.name == name),
            default=0,
        )

    @staticmethod
    def _for_execution(
        manifest: RunManifest,
        execution_context: StageExecutionContext | None,
    ) -> RunManifest:
        if execution_context is None:
            return manifest
        return manifest.model_copy(
            update={
                "metadata": {
                    **manifest.metadata,
                    _EXECUTION_ATTEMPT_METADATA: execution_context.attempt,
                }
            }
        )

    @staticmethod
    def _stage_inputs(
        manifest_ref: ArtifactRef,
        stage_config: Mapping[str, Any] | None,
    ) -> tuple[ArtifactRef, ...]:
        return _unique_artifacts(
            (manifest_ref,) + _config_input_artifacts(dict(stage_config or {}))
        )

    @staticmethod
    def _add_vector_stage_files(
        push_files: dict[str, Path],
        vector_manifest: str | Path,
        *,
        prefix: str,
    ) -> None:
        manifest_path = Path(vector_manifest).expanduser().resolve()
        manifest = VectorPreparer.load_manifest(manifest_path)
        push_files[f"{prefix}-manifest.json"] = manifest_path
        for section_name, records in (
            ("input", manifest.inputs),
            ("golden", manifest.goldens),
        ):
            for index, (_name, record) in enumerate(sorted(records.items())):
                tensor_path = Path(record.path)
                if not tensor_path.is_absolute():
                    tensor_path = manifest_path.parent / tensor_path
                tensor_path = tensor_path.resolve()
                actual_sha256 = sha256_file(tensor_path)
                if actual_sha256 != record.sha256:
                    raise ValueError(
                        f"Vector tensor {record.name!r} SHA256 mismatch: "
                        f"expected {record.sha256}, got {actual_sha256}"
                    )
                push_files[
                    f"{prefix}-{section_name}-{index:04d}.raw"
                ] = tensor_path

    @classmethod
    def _device_stage_files(
        cls,
        output_dir: Path,
        *,
        contexts: Sequence[Any],
        vector_manifests: Sequence[str | Path] = (),
        inline_cases: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Path]:
        """Materialize the exact context and vector inputs staged for a run."""

        push_files: dict[str, Path] = {}
        for index, context in enumerate(contexts):
            if hasattr(context, "context_binary_path"):
                context = getattr(context, "context_binary_path")
            if not isinstance(context, (str, os.PathLike)):
                raise TypeError(
                    "device context must be a local context-binary path before loading"
                )
            push_files[f"context-{index:04d}.bin"] = (
                Path(context).expanduser().resolve()
            )
        for index, manifest_path in enumerate(vector_manifests):
            cls._add_vector_stage_files(
                push_files,
                manifest_path,
                prefix=f"vectors-{index:04d}",
            )
        for index, tensors in enumerate(inline_cases):
            if not tensors:
                continue
            manifest_path = VectorPreparer(
                output_dir / "device-staging" / f"inline-{index:04d}"
            ).prepare_case(
                f"device-inline-{index:04d}",
                tensors,
                metadata={"role": "device_staging_input"},
            )
            cls._add_vector_stage_files(
                push_files,
                manifest_path,
                prefix=f"inline-{index:04d}",
            )
        return push_files

    def _device_stage(
        self,
        manifest: RunManifest,
        adapter: Any,
        *,
        stage_name: str,
        input_manifest_sha256: str,
        stage_config: Mapping[str, Any],
        push_files: Mapping[str, Path],
    ) -> Any:
        return self._device_runtime.stage(
            adapter,
            output_root=manifest.build_spec.output_root,
            job_id=str(manifest.run_id),
            stage_key=self._stage_key(
                stage_name,
                input_manifest_sha256,
                stage_config,
            ),
            attempt_id=f"attempt-{self._attempt(manifest, stage_name):03d}",
            push_files=push_files,
            expected_target=self._resolved_target_entry(manifest.build_spec),
        )

    @staticmethod
    def _build_progress_recorder(
        path: Path,
    ) -> Callable[[Mapping[str, Any]], None]:
        """Append one JSON line per published (context length, slice).

        Deliberately append-only and outside the manifest: this is a crash
        breadcrumb, not evidence. Nothing reads it to decide what to reuse --
        reuse still goes through verified receipts -- and a write failure must
        never take the build down with it.
        """

        def record(entry: Mapping[str, Any]) -> None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"published_at": utc_now().isoformat(), **dict(entry)},
                            sort_keys=True,
                            default=str,
                        )
                        + "\n"
                    )
            except OSError:
                return

        return record

    @staticmethod
    def _resolved_target_entry(spec: BuildSpec) -> Any:
        """The registry entry the spec's target resolves to.

        Spec-time validation already refused an unregistered name or tuple, so
        this only re-reads what was accepted; it is needed here because the
        Android ``soc_id`` list lives on the registry entry, not on the spec.
        """

        return resolve_target(spec.target.name)

    @staticmethod
    def _initialize_execution(
        adapter: Any,
        compiled: Any,
        *,
        device: Any,
    ) -> Any:
        """Establish QAIRT's persistent execution context, when supported."""

        initialize = getattr(adapter, "initialize_execution", None)
        release = getattr(adapter, "release_execution", None)
        if not callable(initialize) or not callable(release):
            return None
        return initialize(compiled, device=device)

    @staticmethod
    def _profile_report_payload(
        report: Any,
    ) -> tuple[Any, ArtifactRef | None]:
        """Materialize a profiler report without inventing a parser."""

        if isinstance(report, (str, Path)):
            candidate = Path(report).expanduser()
            if candidate.is_file():
                resolved = candidate.resolve()
                ref = ArtifactRef.from_path(
                    resolved,
                    kind=ArtifactKind.REPORT,
                    logical_name=None,
                )
                if resolved.suffix.lower() == ".json":
                    try:
                        return (
                            json.loads(resolved.read_text(encoding="utf-8")),
                            ref,
                        )
                    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                        raise InvalidSpecError(
                            "QAIRT profiler returned an unreadable JSON report",
                            stage="benchmark",
                            details={
                                "path": os.fspath(resolved),
                                "reason": str(exc),
                            },
                        ) from exc
                return (
                    {
                        "artifact": _jsonable(ref),
                        "structured": False,
                    },
                    ref,
                )
        return _jsonable(report), None

    @staticmethod
    def _manifest_artifact_payload(
        manifest: RunManifest,
        logical_name: str,
        *,
        stage: str,
    ) -> tuple[ArtifactRef, dict[str, Any]]:
        """Load one verified JSON artifact from the cumulative manifest."""

        candidates = [
            artifact
            for artifact in manifest.artifacts
            if artifact.logical_name == logical_name
        ]
        if len(candidates) != 1:
            raise InvalidSpecError(
                f"automatic {stage} requires exactly one {logical_name} artifact",
                stage=stage,
                details={
                    "logical_name": logical_name,
                    "artifact_count": len(candidates),
                },
            )
        ref = candidates[0]
        verify_artifact(ref)
        try:
            payload = json.loads(ref.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InvalidSpecError(
                f"{logical_name} is not readable JSON",
                stage=stage,
                details={"artifact": _jsonable(ref), "reason": str(exc)},
            ) from exc
        if not isinstance(payload, Mapping):
            raise InvalidSpecError(
                f"{logical_name} JSON root must be an object",
                stage=stage,
                details={"artifact": _jsonable(ref)},
            )
        return ref, dict(payload)

    @staticmethod
    def _verify_manifest_artifacts(manifest: RunManifest) -> None:
        stage_artifacts = tuple(
            artifact
            for stage in manifest.stages
            for artifact in (*stage.inputs, *stage.outputs)
        )
        for artifact in _unique_artifacts(manifest.artifacts + stage_artifacts):
            verify_artifact(artifact)

    @staticmethod
    def _published_child(
        store: ManifestStore,
        manifest: RunManifest,
        manifest_ref: ArtifactRef,
    ) -> tuple[RunManifest, ArtifactRef] | None:
        pattern = f"manifest-r{manifest.revision + 1:06d}-*.json"
        children: list[tuple[RunManifest, ArtifactRef]] = []
        for path in sorted(manifest_ref.path.parent.glob(pattern)):
            child_ref = ArtifactRef.from_path(
                path,
                kind=ArtifactKind.MANIFEST,
                logical_name=f"run-manifest-r{manifest.revision + 1}",
            )
            child = store.load(child_ref)
            parent = child.parent_manifest
            if (
                parent is not None
                and parent.path == manifest_ref.path
                and parent.sha256 == manifest_ref.sha256
            ):
                children.append((child, child_ref))
        if len(children) > 1:
            raise ManifestConflictError(
                "manifest parent has multiple published child revisions",
                details={
                    "parent": os.fspath(manifest_ref.path),
                    "children": [os.fspath(ref.path) for _, ref in children],
                },
            )
        return children[0] if children else None

    @classmethod
    def _reuse_child_result(
        cls,
        child: RunManifest,
        child_ref: ArtifactRef,
        *,
        stage_name: str,
        stage_key: str,
    ) -> ToolResult[dict[str, Any]] | None:
        if not child.stages:
            return None
        stage = child.stages[-1]
        if (
            stage.name != stage_name
            or stage.metrics.get("stage_key") != stage_key
        ):
            return None
        cls._verify_manifest_artifacts(child)
        if stage.status is StageStatus.FAILED:
            assert stage.error is not None
            return ToolResult.failure(stage.error, manifest=child_ref)
        if stage.status is not StageStatus.SUCCEEDED:
            return None
        return ToolResult.success(
            {
                "reused": True,
                "stage": stage_name,
                "stage_key": stage_key,
                "artifacts": [_jsonable(ref) for ref in stage.outputs],
            },
            manifest=child_ref,
        )

    @staticmethod
    def _stage_key(
        stage_name: str,
        manifest_sha256: str,
        config: Mapping[str, Any] | None,
    ) -> str:
        constraints = load_harness_constraints()
        payload = {
            "stage": stage_name,
            "input_manifest_sha256": manifest_sha256,
            "config": _stage_key_value(dict(config or {})),
            "qairt_version": constraints.qairt_version,
            "qairt_build_id": constraints.qairt_build_id,
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    @staticmethod
    def _preflight(adapter: Any, spec: BuildSpec) -> Any:
        report = adapter.preflight(spec)
        require_preflight(report)
        return report

    @staticmethod
    def _failure_revision(
        store: ManifestStore,
        manifest_ref: ArtifactRef,
        manifest: RunManifest,
        *,
        stage_name: str,
        stage_key: str,
        inputs: tuple[ArtifactRef, ...],
        started_at: Any,
        exc: BaseException,
        outputs: tuple[ArtifactRef, ...] = (),
        attempt: int | None = None,
    ) -> tuple[ToolErrorData, ArtifactRef]:
        error = QairtAgent._error(exc, stage_name)
        stage = StageRecord(
            name=stage_name,
            attempt=attempt or QairtAgent._attempt(manifest, stage_name),
            status=StageStatus.FAILED,
            started_at=started_at,
            completed_at=utc_now(),
            inputs=inputs,
            outputs=outputs,
            metrics={"stage_key": stage_key},
            error=error,
        )
        _, failed_ref = store.revise(manifest_ref, stage=stage)
        return error, failed_ref

    def _initial_operation(
        self,
        stage_name: str,
        spec_value: BuildSpec | Mapping[str, Any],
        operation: Callable[
            [BuildSpec, RunManifest, Any],
            tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]],
        ],
        *,
        operation_config: Mapping[str, Any] | None = None,
        execution_context: StageExecutionContext | None = None,
    ) -> ToolResult[dict[str, Any]]:
        try:
            spec = self._parse_spec(spec_value)
        except Exception as exc:
            return ToolResult.failure(self._error(exc, stage_name))

        try:
            store = self._store_for_spec(spec)
            manifest, manifest_ref = store.create(
                spec,
                metadata={"entrypoint": stage_name, "state_model": "stateless"},
            )
            started_at = utc_now()
            stage_key = self._stage_key(
                stage_name,
                manifest_ref.sha256,
                operation_config,
            )
            stage_inputs = self._stage_inputs(manifest_ref, operation_config)
        except Exception as exc:
            return ToolResult.failure(self._error(exc, stage_name))
        try:
            execution_manifest = self._for_execution(manifest, execution_context)
            data, artifacts, metrics = operation(
                spec,
                execution_manifest,
                self._new_adapter(),
            )
            artifacts = _unique_artifacts(artifacts)
            metrics = {**metrics, "stage_key": stage_key}
            stage = StageRecord(
                name=stage_name,
                attempt=execution_context.attempt if execution_context is not None else 1,
                status=StageStatus.SUCCEEDED,
                started_at=started_at,
                completed_at=utc_now(),
                inputs=stage_inputs,
                outputs=artifacts,
                metrics=_jsonable(metrics),
            )
            _, final_ref = store.revise(
                manifest_ref,
                stage=stage,
                artifacts=artifacts,
            )
            return ToolResult.success(data, manifest=final_ref)
        except Exception as exc:
            try:
                error, failed_ref = self._failure_revision(
                    store,
                    manifest_ref,
                    manifest,
                    stage_name=stage_name,
                    stage_key=stage_key,
                    inputs=stage_inputs,
                    started_at=started_at,
                    exc=exc,
                    attempt=(
                        execution_context.attempt
                        if execution_context is not None
                        else None
                    ),
                )
            except Exception:
                return ToolResult.failure(self._error(exc, stage_name), manifest=manifest_ref)
            return ToolResult.failure(error, manifest=failed_ref)

    def _continuation_operation(
        self,
        stage_name: str,
        manifest_uri: str | Path,
        manifest_sha256: str,
        operation: Callable[
            [RunManifest, Any, Path],
            tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]],
        ],
        *,
        stage_config: Mapping[str, Any] | None = None,
        execution_context: StageExecutionContext | None = None,
    ) -> ToolResult[dict[str, Any]]:
        manifest_ref: ArtifactRef | None = None
        try:
            store = self._store_for_manifest(manifest_uri)
            manifest = store.load(manifest_uri, expected_sha256=manifest_sha256)
            manifest_ref = ArtifactRef.from_path(
                manifest_uri,
                kind=ArtifactKind.MANIFEST,
                logical_name=f"run-manifest-r{manifest.revision}",
            )
            if manifest_ref.sha256 != manifest_sha256.lower():
                raise ValueError("loaded manifest hash changed during verification")
            self._verify_manifest_artifacts(manifest)
            stage_key = self._stage_key(
                stage_name,
                manifest_ref.sha256,
                stage_config,
            )
            stage_inputs = self._stage_inputs(manifest_ref, stage_config)
            published_child = self._published_child(store, manifest, manifest_ref)
        except Exception as exc:
            return ToolResult.failure(
                self._error(exc, stage_name),
                manifest=manifest_ref,
            )

        if published_child is not None:
            child_manifest, child_ref = published_child
            try:
                reused = self._reuse_child_result(
                    child_manifest,
                    child_ref,
                    stage_name=stage_name,
                    stage_key=stage_key,
                )
            except Exception as exc:
                return ToolResult.failure(self._error(exc, stage_name), manifest=child_ref)
            if reused is not None:
                return reused
            conflict = ManifestConflictError(
                "manifest was already continued by a different stage or configuration",
                stage=stage_name,
                details={
                    "parent_manifest": os.fspath(manifest_ref.path),
                    "published_child": os.fspath(child_ref.path),
                },
            )
            return ToolResult.failure(conflict, manifest=child_ref)

        started_at = utc_now()
        attempt = (
            execution_context.attempt
            if execution_context is not None
            else self._attempt(manifest, stage_name)
        )
        execution_manifest = self._for_execution(manifest, execution_context)
        output_dir = (
            self._run_dir(execution_manifest)
            / "stages"
            / stage_name
            / stage_key[:16]
            / f"attempt-{attempt:03d}"
        )
        try:
            data, artifacts, metrics = operation(
                execution_manifest,
                self._new_adapter(),
                output_dir,
            )
            artifacts = _unique_artifacts(artifacts)
            metrics = {**metrics, "stage_key": stage_key}
            stage = StageRecord(
                name=stage_name,
                attempt=attempt,
                status=StageStatus.SUCCEEDED,
                started_at=started_at,
                completed_at=utc_now(),
                inputs=stage_inputs,
                outputs=artifacts,
                metrics=_jsonable(metrics),
            )
            try:
                _, final_ref = store.revise(
                    manifest_ref,
                    stage=stage,
                    artifacts=artifacts,
                )
            except ManifestConflictError as exc:
                published_child = self._published_child(store, manifest, manifest_ref)
                if published_child is None:
                    return ToolResult.failure(self._error(exc, stage_name), manifest=manifest_ref)
                child_manifest, child_ref = published_child
                reused = self._reuse_child_result(
                    child_manifest,
                    child_ref,
                    stage_name=stage_name,
                    stage_key=stage_key,
                )
                if reused is not None and reused.ok:
                    return ToolResult.success(data, manifest=child_ref)
                return ToolResult.failure(self._error(exc, stage_name), manifest=child_ref)
            return ToolResult.success(data, manifest=final_ref)
        except Exception as exc:
            try:
                error, failed_ref = self._failure_revision(
                    store,
                    manifest_ref,
                    manifest,
                    stage_name=stage_name,
                    stage_key=stage_key,
                    inputs=stage_inputs,
                    started_at=started_at,
                    exc=exc,
                    attempt=attempt,
                )
            except ManifestConflictError as conflict:
                try:
                    published_child = self._published_child(store, manifest, manifest_ref)
                except Exception as child_exc:
                    return ToolResult.failure(
                        self._error(child_exc, stage_name),
                        manifest=manifest_ref,
                    )
                if published_child is not None:
                    child_manifest, child_ref = published_child
                    reused = self._reuse_child_result(
                        child_manifest,
                        child_ref,
                        stage_name=stage_name,
                        stage_key=stage_key,
                    )
                    if reused is not None:
                        return reused
                    return ToolResult.failure(
                        self._error(conflict, stage_name),
                        manifest=child_ref,
                    )
                return ToolResult.failure(
                    self._error(conflict, stage_name),
                    manifest=manifest_ref,
                )
            except Exception as revision_exc:
                return ToolResult.failure(
                    self._error(revision_exc, stage_name),
                    manifest=manifest_ref,
                )
            return ToolResult.failure(error, manifest=failed_ref)

    @classmethod
    def _optrace_compatibility_mismatches(
        cls,
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> list[str]:
        mismatches: list[str] = []
        if baseline.get("schema") != "qairt-agent.optrace-evidence.v1":
            mismatches.append("baseline.schema")
        for key in (
            "device_identifier",
            "profile_level",
            "profile_option",
            "profile_scope",
            "claim_scope",
        ):
            if baseline.get(key) != candidate.get(key):
                mismatches.append(key)
        baseline_binding = baseline.get("runtime_binding", {})
        candidate_binding = candidate.get("runtime_binding", {})
        if not isinstance(baseline_binding, Mapping) or not isinstance(
            candidate_binding,
            Mapping,
        ):
            mismatches.append("runtime_binding")
        else:
            for key in (
                "lane",
                "family",
                "ar",
                "context_length",
                "scope",
                "graph_name",
                "component",
                "coverage",
                "graph_ar",
            ):
                if baseline_binding.get(key) != candidate_binding.get(key):
                    mismatches.append(f"runtime_binding.{key}")
        if cls._optrace_profile_signature(
            baseline
        ) != cls._optrace_profile_signature(candidate):
            mismatches.append("profile_signature")
        return mismatches


__all__ = ["QairtAgent"]
