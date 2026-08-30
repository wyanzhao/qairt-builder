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
    GeneratedFamilyConfig,
    OnnxInspector,
    apply_lane_benchmark_defaults,
    effective_benchmark_policy,
)
from qairt_agent.harness import load_harness_constraints
from qairt_agent.qairt_adapter import (
    NativeKvGraphExpectation,
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

_LIVE_SDK_FIELDS = {
    "execution_result",
    "graph_context",
    "reports",
    "sdk_container",
    "sdk_compiled_model",
    "sdk_model",
    "sdk_output",
}

_OUTPUT_ONLY_CONFIG_FIELDS = {
    "destination",
    "output_dir",
    "output_dlc",
    "output_file",
    "output_path",
    "output_root",
    "report_path",
}
_EXECUTION_ATTEMPT_METADATA = "_qairt_agent_execution_attempt"

# Low-level slice-chain keys. The GenAI lane drives its container through the
# public executor and exposes raw slices only under ``tensor_runtime``, so these
# top-level keys are never executable there.
_LOW_LEVEL_CHAIN_CONFIG_FIELDS = (
    "routes",
    "contexts",
    "steps",
    "initial_native_state",
)


def _jsonable(value: Any) -> Any:
    """Convert public values to JSON without traversing live SDK objects."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "nbytes": value.nbytes,
        }
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
            if field.name not in _LIVE_SDK_FIELDS
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


def _stage_key_value(value: Any) -> Any:
    """Canonicalize stage inputs while retaining their content identity."""

    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("object arrays cannot participate in a stage key")
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
        }
    if isinstance(value, Path):
        path = value.expanduser().resolve()
        if path.is_file():
            ref = ArtifactRef.from_path(path, kind=_artifact_kind(path))
            return {
                "path": os.fspath(path),
                "sha256": ref.sha256,
                "size_bytes": ref.size_bytes,
            }
        return {"path": os.fspath(path), "exists": False}
    if isinstance(value, str):
        try:
            path = Path(value).expanduser()
            if path.is_file():
                return _stage_key_value(path)
        except (OSError, ValueError):
            pass
        return value
    if hasattr(value, "model_dump"):
        return _stage_key_value(value.model_dump(mode="python", exclude_none=True))
    if hasattr(value, "to_dict"):
        return _stage_key_value(value.to_dict())
    if is_dataclass(value):
        return {
            field.name: _stage_key_value(getattr(value, field.name))
            for field in fields(value)
            if field.name not in _LIVE_SDK_FIELDS
        }
    if isinstance(value, Mapping):
        return {
            str(key): _stage_key_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower() not in _OUTPUT_ONLY_CONFIG_FIELDS
        }
    if isinstance(value, (tuple, list)):
        return [_stage_key_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_stage_key_value(item) for item in value]
        return sorted(normalized, key=lambda item: canonical_json_bytes(item))
    return str(value)


def _artifact_kind(path: Path, field_name: str = "") -> ArtifactKind:
    lowered = path.name.lower()
    if lowered.endswith(".onnx"):
        return ArtifactKind.ONNX
    if "encoding" in lowered or "encoding" in field_name:
        return ArtifactKind.AIMET_ENCODINGS
    if lowered.endswith(".dlc"):
        return ArtifactKind.DLC
    if lowered.endswith((".bin", ".serialized")) or "context_binary" in field_name:
        return ArtifactKind.CONTEXT_BINARY
    if "manifest" in lowered:
        return ArtifactKind.TEST_VECTORS
    if lowered.endswith(".json"):
        return ArtifactKind.CONFIG
    return ArtifactKind.OTHER


def _path_artifacts(value: Any, *, logical_prefix: str = "") -> tuple[ArtifactRef, ...]:
    """Collect and hash materialized paths from an adapter result."""

    collected: list[ArtifactRef] = []
    seen: set[Path] = set()

    def visit(item: Any, field_name: str = "", prefix: str = "") -> None:
        if item is None or field_name in _LIVE_SDK_FIELDS:
            return
        if isinstance(item, Path):
            resolved = item.expanduser().resolve()
            if resolved.is_dir():
                children = sorted(path for path in resolved.rglob("*") if path.is_file())
                if not children:
                    raise FileNotFoundError(
                        f"adapter returned an empty artifact directory: {resolved}"
                    )
                for child in children:
                    if child.is_symlink():
                        raise ValueError(
                            "adapter artifact directories cannot contain symlinks: "
                            f"{child}"
                        )
                    relative = child.relative_to(resolved).as_posix()
                    visit(
                        child,
                        f"{field_name}.{relative}".strip("."),
                        prefix,
                    )
                return
            if not resolved.is_file():
                raise FileNotFoundError(f"adapter returned a missing artifact: {resolved}")
            if resolved not in seen:
                seen.add(resolved)
                collected.append(
                    ArtifactRef.from_path(
                        resolved,
                        kind=_artifact_kind(resolved, field_name),
                        logical_name=f"{prefix}{field_name}".strip(".") or resolved.name,
                    )
                )
            return
        if isinstance(item, str):
            return
        if is_dataclass(item):
            for field in fields(item):
                visit(
                    getattr(item, field.name),
                    field.name,
                    f"{prefix}{field.name}." if not isinstance(getattr(item, field.name), Path) else prefix,
                )
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                visit(nested, str(key), f"{prefix}{key}.")
            return
        if isinstance(item, (tuple, list)):
            for index, nested in enumerate(item):
                visit(nested, field_name, f"{prefix}{index}.")

    visit(value, prefix=logical_prefix)
    return tuple(collected)


# Keys under which an SDK might *report* a generated-token count. QAIRT 2.49's
# public GenerationMetrics exposes token_generation_rate and
# token_generation_time but no count, so on this SDK the caller's explicit
# token_count is the only source; multiplying a rate by a duration would
# manufacture a number the SDK never reported.
_SDK_GENERATED_TOKEN_COUNT_KEYS = (
    "num_generated_tokens",
    "generated_token_count",
    "num-generated-tokens",
)


def _sdk_generated_token_count(metrics: Any) -> int | None:
    """Return a generated-token count the SDK reported, never a derived one."""

    if not isinstance(metrics, Mapping):
        return None
    for key in _SDK_GENERATED_TOKEN_COUNT_KEYS:
        value = metrics.get(key)
        if isinstance(value, Mapping):
            value = value.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if float(value).is_integer() and int(value) > 0:
            return int(value)
    return None


# Device latency is the reported metric, so it is sampled rather than measured
# once (maintainer decision 2026-08-30: ten executes, averaged).
DEVICE_EXECUTION_SAMPLES = 10

# Why a scope has no device meter. Stated in the report so an absent device
# number is a declared gap rather than an omission the reader must notice.
_DEVICE_EXECUTION_UNAVAILABLE = {
    "genai_generation": (
        "generate() reaches Genie as GenieDialog_query rather than "
        "CompiledModel.__call__, so qairt.Profiler observes nothing; the GenAI "
        "lane needs its own genie_execution meter (T11)"
    ),
    "chain": (
        "no chain slice executed, so no per-slice inputs were recorded to "
        "profile with"
    ),
    "chain_sequence": (
        "no chain slice executed, so no per-slice inputs were recorded to "
        "profile with"
    ),
}

_STATIC_FOOTPRINT_SCHEMA = "qairt-agent.static-footprint/1"

# Roles summed into the headline total: what a device actually has to hold.
# Converted DLCs are build intermediates and are reported but never summed in.
_DEPLOYABLE_FOOTPRINT_ROLES = ("context", "genai_container")


def _static_footprint(
    result: Any,
    artifacts: Sequence[ArtifactRef],
) -> dict[str, Any]:
    """Summarize the on-disk size of one build's outputs.

    Sizes are read from the published content-addressed references, so nothing
    here is estimated: an output that was not published is absent from the
    report rather than reported as zero.  Diagnostic contexts are kept in their
    own section and never counted in the production totals, mirroring the rule
    that diagnostic-context latency is not production latency.
    """

    by_path: dict[Path, ArtifactRef] = {}
    for ref in artifacts:
        by_path.setdefault(ref.path.expanduser().resolve(), ref)

    def entry(
        path: Any,
        role: str,
        **details: Any,
    ) -> dict[str, Any] | None:
        if path is None:
            return None
        ref = by_path.get(Path(path).expanduser().resolve())
        if ref is None:
            return None
        record = {
            "role": role,
            "logical_name": ref.logical_name,
            "path": os.fspath(ref.path),
            "sha256": ref.sha256,
            "bytes": ref.size_bytes,
        }
        record.update({key: value for key, value in details.items() if value is not None})
        return record

    counted: set[Path] = set()

    def directory_entries(directory: Any, role: str) -> list[dict[str, Any]]:
        # Omni packaging nests the audio/text component directories inside the
        # container directory, so a file must be counted for the first (widest)
        # root that claims it and never again.
        if directory is None:
            return []
        root = Path(directory).expanduser().resolve()
        found: list[dict[str, Any]] = []
        for path in sorted(by_path):
            if path in counted or not (path == root or path.is_relative_to(root)):
                continue
            record = entry(path, role, container=os.fspath(root))
            if record is not None:
                counted.add(path)
                found.append(record)
        return found

    production: list[dict[str, Any]] = []
    diagnostic: list[dict[str, Any]] = []

    for context in getattr(result, "contexts", ()) or ():
        record = entry(
            getattr(context, "context_binary_path", None),
            "context",
            slice_name=getattr(context, "slice_name", None),
            context_length=getattr(context, "context_length", None),
            ar_values=list(getattr(context, "ar_values", ()) or ()) or None,
            weight_sharing=getattr(context, "weight_sharing", None),
        )
        if record is not None:
            production.append(record)

    for context in getattr(result, "diagnostic_contexts", ()) or ():
        record = entry(
            getattr(context, "context_binary_path", None),
            "diagnostic_context",
            slice_name=getattr(context, "slice_name", None),
            context_length=getattr(context, "context_length", None),
        )
        if record is not None:
            diagnostic.append(record)

    for converted in getattr(result, "converted_models", ()) or ():
        record = entry(
            getattr(converted, "model_path", None),
            "converted_model",
            slice_name=getattr(converted, "slice_name", None),
            ar=getattr(converted, "ar", None),
            context_length=getattr(converted, "context_length", None),
        )
        if record is not None:
            production.append(record)

    for attribute in ("container_path", "text_container_path", "audio_container_path"):
        production.extend(
            directory_entries(getattr(result, attribute, None), "genai_container")
        )

    for raw_slice in getattr(result, "raw_slices", ()) or ():
        # A raw slice normally lives inside the container directory and is
        # already counted there.
        slice_path = getattr(raw_slice, "context_binary_path", None)
        if slice_path is None:
            continue
        resolved_slice = Path(slice_path).expanduser().resolve()
        if resolved_slice in counted:
            continue
        record = entry(
            slice_path,
            "genai_raw_slice",
            slice_name=getattr(raw_slice, "slice_id", None),
        )
        if record is not None:
            counted.add(resolved_slice)
            production.append(record)

    footprint: dict[str, Any] = {
        "schema": _STATIC_FOOTPRINT_SCHEMA,
        "policy": "report_only",
        "unit": "bytes",
        "artifacts": production,
    }
    role_totals: dict[str, int] = {}
    for record in production:
        role_totals[record["role"]] = role_totals.get(record["role"], 0) + record["bytes"]
    for role, key in (
        ("context", "contexts_total_bytes"),
        ("converted_model", "converted_models_total_bytes"),
        ("genai_container", "genai_container_total_bytes"),
        ("genai_raw_slice", "genai_raw_slices_total_bytes"),
    ):
        if role in role_totals:
            footprint[key] = role_totals[role]
    summed_roles = [role for role in _DEPLOYABLE_FOOTPRINT_ROLES if role in role_totals]
    if summed_roles:
        footprint["total_bytes"] = sum(role_totals[role] for role in summed_roles)
        footprint["total_includes"] = summed_roles
    if diagnostic:
        footprint["diagnostic"] = {
            "artifacts": diagnostic,
            "total_bytes": sum(record["bytes"] for record in diagnostic),
            "counted_in_totals": False,
        }
    return footprint


def _config_input_artifacts(value: Any) -> tuple[ArtifactRef, ...]:
    """Collect existing file paths supplied directly in a stage config."""

    refs: list[ArtifactRef] = []
    seen: set[Path] = set()

    def visit(item: Any, logical_name: str) -> None:
        path: Path | None = None
        if isinstance(item, Path):
            path = item
        elif isinstance(item, str):
            try:
                candidate = Path(item).expanduser()
                if candidate.is_file():
                    path = candidate
            except (OSError, ValueError):
                path = None
        if path is not None:
            resolved = path.expanduser().resolve()
            if resolved.is_file() and resolved not in seen:
                seen.add(resolved)
                refs.append(
                    ArtifactRef.from_path(
                        resolved,
                        kind=_artifact_kind(resolved, logical_name),
                        logical_name=f"stage_input.{logical_name}".strip("."),
                    )
                )
                if resolved.suffix.lower() == ".onnx":
                    for index, external_path in enumerate(
                        OnnxInspector().external_data_paths(resolved)
                    ):
                        if external_path in seen:
                            continue
                        seen.add(external_path)
                        refs.append(
                            ArtifactRef.from_path(
                                external_path,
                                kind=ArtifactKind.OTHER,
                                logical_name=(
                                    f"stage_input.{logical_name}."
                                    f"external_data_{index:03d}"
                                ).strip("."),
                            )
                        )
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if str(key).lower() in _OUTPUT_ONLY_CONFIG_FIELDS:
                    continue
                visit(nested, f"{logical_name}.{key}".strip("."))
        elif isinstance(item, (tuple, list)):
            for index, nested in enumerate(item):
                visit(nested, f"{logical_name}.{index}".strip("."))

    visit(value, "config")
    return tuple(refs)


def _unique_artifacts(
    artifacts: Sequence[ArtifactRef],
) -> tuple[ArtifactRef, ...]:
    """Keep one reference per content/path pair while preserving order."""

    unique: list[ArtifactRef] = []
    seen: set[tuple[Path, str]] = set()
    for artifact in artifacts:
        key = (artifact.path, artifact.sha256)
        if key not in seen:
            seen.add(key)
            unique.append(artifact)
    return tuple(unique)


def _layer_float_reference(effective: Mapping[str, Any]) -> bool:
    """Whether this validation was asked for layer-level float comparison."""

    config = effective.get("float_reference")
    if not isinstance(config, Mapping):
        return False
    return str(config.get("granularity", "slice_boundary")) == "layer"


def _output_mapping(
    value: Any,
    *,
    graph_name: str | None = None,
) -> dict[str, np.ndarray]:
    """Normalize a QAIRT execution result to a named tensor mapping."""

    candidate = value
    if not isinstance(candidate, Mapping):
        candidate = getattr(value, "outputs", None)
    if callable(candidate):
        candidate = candidate()
    if candidate is None and hasattr(value, "get_outputs"):
        candidate = value.get_outputs()
    if candidate is None:
        candidate = getattr(value, "data", None)
    if isinstance(candidate, Sequence) and not isinstance(
        candidate,
        (str, bytes, bytearray),
    ):
        if len(candidate) != 1 or not isinstance(candidate[0], Mapping):
            raise TypeError(
                "QAIRT execution returned multiple inference results; "
                "one named graph invocation was expected"
            )
        candidate = candidate[0]
    if not isinstance(candidate, Mapping):
        raise TypeError(
            "QAIRT execution result must expose a mapping through itself, "
            "`.data`, `.outputs`, or `.get_outputs()`"
        )
    if graph_name is not None and graph_name in candidate:
        graph_candidate = candidate[graph_name]
        if isinstance(graph_candidate, Mapping) or (
            isinstance(graph_candidate, Sequence)
            and not isinstance(graph_candidate, (str, bytes, bytearray))
            and graph_candidate
            and isinstance(graph_candidate[0], Mapping)
        ):
            candidate = graph_candidate
    if (
        isinstance(candidate, Mapping)
        and candidate
        and all(isinstance(item, Mapping) for item in candidate.values())
    ):
        if graph_name is not None and graph_name in candidate:
            candidate = candidate[graph_name]
        elif len(candidate) == 1:
            candidate = next(iter(candidate.values()))
        else:
            raise TypeError(
                "QAIRT execution returned multiple graph outputs without "
                "an exact graph_name match"
            )
    if (
        isinstance(candidate, Sequence)
        and not isinstance(candidate, (str, bytes, bytearray))
    ):
        if len(candidate) != 1 or not isinstance(candidate[0], Mapping):
            raise TypeError("QAIRT graph output contains multiple inference batches")
        candidate = candidate[0]
    if not isinstance(candidate, Mapping):
        raise TypeError("QAIRT selected graph output is not a tensor mapping")
    return {str(name): np.asarray(tensor) for name, tensor in candidate.items()}


class QairtAgent:
    """Synchronous Python facade used directly and through MCP."""

    def __init__(
        self,
        *,
        adapter: Any | None = None,
        adapter_factory: Callable[[], Any] | None = None,
        device_runtime: DeviceRuntime | Any | None = None,
    ) -> None:
        if adapter is not None and adapter_factory is not None:
            raise ValueError("provide adapter or adapter_factory, not both")
        self._adapter_override = adapter
        self._adapter_factory = adapter_factory
        self._device_runtime = device_runtime or DeviceRuntime()

    def _new_adapter(self) -> Any:
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
        )

    @staticmethod
    def _execution_options(config: Mapping[str, Any]) -> dict[str, Any]:
        options = dict(config.get("execution_options", {}))
        if "device" in options:
            raise ValueError(
                "execution_options.device is controlled by "
                "QAIRT_AGENT_ADB_SERIAL/QAIRT_AGENT_ADB_SERVER"
            )
        return options

    @staticmethod
    def _device_execution_block(
        adapter: Any,
        compiled: Any,
        inputs: Mapping[str, np.ndarray],
        *,
        graph_name: str,
        device: Any,
        native_io: bool,
        execution_options: Mapping[str, Any],
        working_dir: Path,
        repeats: int = DEVICE_EXECUTION_SAMPLES,
    ) -> dict[str, Any] | None:
        """Device-side execute evidence, or ``None`` when unavailable.

        The profiled execute is repeated and averaged, because this is the
        latency metric and one sample cannot show a regression.  An adapter
        that cannot profile still produces a valid benchmark, so a failed
        capture degrades the report rather than failing the stage -- what it
        must never do is publish a device claim it did not measure.
        """

        capture = getattr(adapter, "capture_device_execution", None)
        if not callable(capture):
            return {
                "schema": DEVICE_EXECUTION_SCHEMA,
                "policy": "report_only",
                "available": False,
                "reason": (
                    "the QAIRT adapter does not expose capture_device_execution"
                ),
            }
        try:
            blocks = [
                capture(
                    compiled,
                    inputs,
                    graph_name=graph_name,
                    device=device,
                    native_io=native_io,
                    working_dir=working_dir,
                    **dict(execution_options),
                )
                for _ in range(max(1, int(repeats)))
            ]
            return aggregate_device_executions(blocks)
        except Exception as error:  # report-only: never fail the benchmark
            return {
                "schema": DEVICE_EXECUTION_SCHEMA,
                "policy": "report_only",
                "available": False,
                "reason": f"{type(error).__name__}: {error}",
            }

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
                        QairtAgent._optrace_record_collections(nested)
                    )
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for item in value:
                if isinstance(item, (Mapping, list, tuple)):
                    collections.extend(
                        QairtAgent._optrace_record_collections(item)
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
            for collection in QairtAgent._optrace_record_collections(report):
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
    def _generate_family_config(
        cls, spec: BuildSpec
    ) -> tuple[GeneratedFamilyConfig, Path | None]:
        source_config, source_path = cls._load_model_config(spec)
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
        return generated, source_path

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
        spec: BuildSpec, generated: GeneratedFamilyConfig
    ) -> dict[str, Any]:
        payload = generated.to_dict()
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
        if spec.family is ModelFamily.QWEN3_VL:
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
        elif spec.family is ModelFamily.QWEN3_5_OMNI:
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

    def _diagnostic_device_outputs(
        self,
        manifest: RunManifest,
        effective: Mapping[str, Any],
        adapter: Any,
        *,
        device: Any,
        ar: int,
        inputs: Mapping[str, np.ndarray],
        initial_native_state: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Execute the diagnostic contexts and collect their tapped tensors.

        The build has always compiled these contexts and verified their hashes,
        but nothing ever ran them: an ``op_level_dump_available`` claim that
        rests on a context's *existence* is not operator evidence. This is the
        step that turns it into evidence.

        Fails closed rather than degrading -- a layer-level report that quietly
        fell back to slice boundaries would be the same overclaim in a new
        place.
        """

        index = self._runtime_index_for_manifest(manifest)
        cl_key = str(int(effective["context_length"]))
        entries = list((index.get("diagnostic_contexts") or {}).get(cl_key) or ())
        if not entries:
            raise InvalidSpecError(
                "layer-level float reference requires diagnostic contexts, and "
                "this build produced none; set quality."
                "dump_intermediates_on_failure or compile."
                "enable_intermediate_outputs and rebuild",
                stage="validate",
                details={"context_length": cl_key},
            )

        artifacts_by_path = {
            artifact.path.expanduser().resolve(): artifact
            for artifact in manifest.artifacts
        }
        loaded: dict[str, Any] = {}
        graphs: dict[str, str] = {}
        executed: list[dict[str, Any]] = []
        for entry in entries:
            path = Path(str(entry["context_path"])).expanduser().resolve()
            artifact = artifacts_by_path.get(path)
            if artifact is None:
                raise InvalidSpecError(
                    "runtime_index references a diagnostic context that is not "
                    "a verified build artifact",
                    stage="validate",
                    details={"context_path": os.fspath(path)},
                )
            verify_artifact(artifact)
            graph_name = (entry.get("graphs_by_ar") or {}).get(str(int(ar)))
            if graph_name is None:
                raise InvalidSpecError(
                    "the diagnostic context carries no graph for the bound AR",
                    stage="validate",
                    details={
                        "context_path": os.fspath(path),
                        "ar": int(ar),
                        "available_ars": sorted(
                            (entry.get("graphs_by_ar") or {})
                        ),
                    },
                )
            slice_name = str(entry.get("slice") or "model")
            loaded[slice_name] = (
                adapter.load_compiled(path)
                if hasattr(adapter, "load_compiled")
                else path
            )
            graphs[slice_name] = str(graph_name)
            executed.append(
                {
                    "slice": slice_name,
                    "graph_name": str(graph_name),
                    "artifact": _jsonable(artifact),
                }
            )

        routes = effective.get("routes")
        execution_options = self._execution_options(effective)
        native_io = bool(effective.get("native_io", False))
        if routes:
            # Reuse the production chain wiring with the diagnostic contexts
            # substituted, so each slice is fed exactly what it is fed in a
            # real run instead of a guess at its inputs.
            runner = SliceChainRunner(
                routes,
                self._chain_executors(
                    adapter,
                    loaded,
                    device=device,
                    native_io=native_io,
                    execution_options=execution_options,
                ),
            )
            result = runner.run_device_chain(
                dict(inputs),
                ar=int(ar),
                initial_native_state=dict(initial_native_state or {}),
            )
            outputs = {
                str(name): dict(values)
                for name, values in result.outputs_by_slice().items()
            }
        elif len(loaded) == 1:
            slice_name, compiled = next(iter(loaded.items()))
            raw = adapter.run_graph(
                compiled,
                dict(inputs),
                graph_name=graphs[slice_name],
                device=device,
                native_io=native_io,
                **execution_options,
            )
            outputs = {
                slice_name: dict(
                    _output_mapping(raw, graph_name=graphs[slice_name])
                )
            }
        else:
            raise InvalidSpecError(
                "several diagnostic contexts but no routes: each slice's "
                "diagnostic inputs come from the previous slice, so a chain "
                "definition is required rather than guessed inputs",
                stage="validate",
                details={"diagnostic_slices": sorted(loaded)},
            )

        evidence = {
            "executed_contexts": executed,
            "context_count": len(executed),
            "execution": "device_chain" if routes else "single_graph",
            "tensor_counts": {
                name: len(values) for name, values in outputs.items()
            },
        }
        return outputs, evidence

    def _float_reference_report(
        self,
        manifest: RunManifest,
        effective: Mapping[str, Any],
        *,
        device_outputs: Mapping[str, Mapping[str, Any]] | None,
        output_dir: Path,
        report_suffix: str,
        diagnostic_outputs: Mapping[str, Mapping[str, Any]] | None = None,
        diagnostic_evidence: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...]]:
        """Compare device slice boundaries against an ONNX Runtime float run.

        Debug-only. Nothing here runs unless
        ``stage_configs.validation.float_reference`` is present, and the result
        is published beside the production report rather than replacing any
        part of it: the AIMET golden comparison remains the production
        reference.
        """

        config = effective.get("float_reference")
        if config is None:
            return {}, ()
        if not isinstance(config, Mapping):
            raise InvalidSpecError(
                "stage_configs.validation.float_reference must be an object",
                stage="validate",
            )

        granularity = str(config.get("granularity", "slice_boundary"))
        if granularity not in {"slice_boundary", "layer"}:
            raise InvalidSpecError(
                "float reference granularity must be 'slice_boundary' or "
                "'layer'",
                stage="validate",
                details={"requested_granularity": granularity},
            )
        if granularity == "layer" and not diagnostic_outputs:
            # Degrading to slice boundaries here would republish the exact
            # overclaim this granularity exists to retire.
            raise InvalidSpecError(
                "layer granularity requires executed diagnostic contexts; none "
                "were collected for this run",
                stage="validate",
                details={"requested_granularity": granularity},
            )
        if config.get("ar") is None:
            raise InvalidSpecError(
                "the float reference is a single-AR debug mode: set an "
                "explicit float_reference.ar",
                stage="validate",
            )
        requested_ar = int(config["ar"])
        bound_ar = effective.get("ar")
        if bound_ar is not None and int(bound_ar) != requested_ar:
            raise InvalidSpecError(
                "float_reference.ar must match the AR the validation run is "
                "bound to",
                stage="validate",
                details={
                    "float_reference_ar": requested_ar,
                    "runtime_binding_ar": int(bound_ar),
                },
            )
        if not device_outputs:
            raise InvalidSpecError(
                "the float reference compares device slice boundaries, so it "
                "requires a device chain run; supply routes/contexts or "
                "device_chain_outputs",
                stage="validate",
            )
        if effective.get("vector_manifest") is None:
            raise InvalidSpecError(
                "the float reference must be fed the same inputs as the "
                "device run; no vector manifest is bound",
                stage="validate",
            )

        model_path = (
            config.get("model_path")
            or effective.get("reference_model_path")
            or manifest.build_spec.sources.text.onnx_path
        )
        resolved_model = Path(str(model_path)).expanduser().resolve()
        if not resolved_model.is_file():
            raise InvalidSpecError(
                "the float reference model does not exist",
                stage="validate",
                details={"model_path": os.fspath(resolved_model)},
            )

        supplied_map = config.get("tensor_map") or {}
        if not isinstance(supplied_map, Mapping):
            raise InvalidSpecError(
                "float_reference.tensor_map must be an object",
                stage="validate",
            )

        def mapped_name(slice_name: str, tensor_name: str) -> str | None:
            nested = supplied_map.get(slice_name)
            if isinstance(nested, Mapping) and tensor_name in nested:
                return str(nested[tensor_name])
            direct = supplied_map.get(tensor_name)
            if isinstance(direct, str):
                return direct
            return None

        # Layer granularity compares the tapped intermediates; the boundary
        # tensors stay in the same comparison so a run shows both.
        compared_outputs: dict[str, dict[str, Any]] = {
            str(name): dict(values) for name, values in device_outputs.items()
        }
        if diagnostic_outputs:
            for name, values in diagnostic_outputs.items():
                compared_outputs.setdefault(str(name), {}).update(dict(values))

        producible = VectorPreparer.onnx_producible_tensor_names(resolved_model)
        pairs: list[tuple[str, str, str]] = []
        unmapped: list[dict[str, Any]] = []
        for slice_name, tensors in compared_outputs.items():
            for tensor_name in tensors:
                explicit = mapped_name(str(slice_name), str(tensor_name))
                candidate = explicit if explicit is not None else str(tensor_name)
                if candidate in producible:
                    pairs.append((str(slice_name), str(tensor_name), candidate))
                    continue
                unmapped.append(
                    {
                        "slice": str(slice_name),
                        "tensor": str(tensor_name),
                        "attempted_float_tensor": candidate,
                        "reason": (
                            "explicit tensor_map entry is not produced by the "
                            "float graph"
                            if explicit is not None
                            else "no exact name match in the float graph and no "
                            "tensor_map entry"
                        ),
                    }
                )
        if not pairs:
            raise InvalidSpecError(
                "no device boundary tensor could be bound to the float graph; "
                "supply float_reference.tensor_map — boundary names are never "
                "guessed",
                stage="validate",
                details={"unmapped_tensors": unmapped},
            )

        inputs = self._manifest_inputs(
            effective["vector_manifest"],
            section=str(effective.get("input_section", "inputs")),
            sha256=effective.get("vector_manifest_sha256"),
        )
        providers = tuple(
            str(item) for item in config.get("providers", ("CPUExecutionProvider",))
        )
        captured, provenance = VectorPreparer.capture_onnx_float_activations(
            resolved_model,
            inputs,
            [float_name for _, _, float_name in pairs],
            providers=providers,
        )

        floor = float(effective.get("reference_energy_floor", 0.0))
        observations: list[dict[str, Any]] = []
        for slice_name, tensor_name, float_name in pairs:
            quality = compute_tensor_quality(
                captured[float_name],
                compared_outputs[slice_name][tensor_name],
                reference_energy_floor=floor,
            )
            observations.append(
                {
                    "slice": slice_name,
                    "tensor": tensor_name,
                    "float_tensor": float_name,
                    "quality": quality.to_dict(),
                }
            )

        # Ordered by the float graph's own topology so the first divergence is
        # the first row, not something the reader has to search for.
        topology = {name: order for order, name in enumerate(producible)}
        observations.sort(
            key=lambda item: topology.get(item["float_tensor"], len(topology))
        )

        payload = {
            "schema": "qairt-agent.float-reference-report/1",
            "mode": "debug_only",
            "policy": "report_only",
            "granularity": granularity,
            "ar": requested_ar,
            "claim_scope": "first_observed_divergence_not_root_cause",
            "comparison": "device_chain_vs_onnxruntime_float_graph",
            "ordered_by": "float_graph_topology",
            "op_level_dump_available": bool(diagnostic_outputs),
            "tensor_map": {
                slice_name: {tensor_name: float_name}
                for slice_name, tensor_name, float_name in pairs
            },
            "unmapped_tensors": unmapped,
            "observations": observations,
            **provenance,
        }
        if diagnostic_evidence is not None:
            payload["diagnostic_contexts"] = dict(diagnostic_evidence)
        report_ref = atomic_publish_json(
            output_dir / f"float_reference_report{report_suffix}.json",
            payload,
            kind=ArtifactKind.REPORT,
            logical_name=f"float_reference_report{report_suffix}",
        )
        return payload, (report_ref,)

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
        if spec.family is ModelFamily.VIT:
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
                            spec.family is ModelFamily.QWEN3_VL
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
    def _preflight(adapter: Any, spec: BuildSpec) -> Any:
        report = adapter.preflight(spec)
        require_preflight(report)
        return report

    @staticmethod
    def _publish_effective_config(
        manifest: RunManifest,
        payload: Mapping[str, Any],
    ) -> ArtifactRef:
        path = QairtAgent._run_dir(manifest) / "config" / "effective_config.json"
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

    def generate_config(
        self, spec: BuildSpec | Mapping[str, Any]
    ) -> ToolResult[dict[str, Any]]:
        """Generate family, shape, split, embedding, and workflow configuration."""

        def operation(
            parsed: BuildSpec, manifest: RunManifest, _adapter: Any
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            if parsed.family is ModelFamily.VIT:
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
            generated, config_path = self._generate_family_config(parsed)
            payload = self._effective_payload(parsed, generated)
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
            if parsed.family is ModelFamily.VIT:
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
            generated, config_path = self._generate_family_config(parsed)
            payload = self._effective_payload(parsed, generated)
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
            if parsed.family is ModelFamily.VIT:
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
            generated, config_path = self._generate_family_config(parsed)
            effective = self._effective_payload(parsed, generated)
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
            build_kwargs: dict[str, Any] = {}
            if parsed.family is ModelFamily.QWEN3_5 and len(parsed.sequence.ars) > 1:
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

            generated, config_path = self._generate_family_config(parsed)
            source_config, resolved_source_config = self._load_model_config(parsed)
            if config_path != resolved_source_config:
                raise RuntimeError("model config resolution changed during one build")
            effective = self._effective_payload(parsed, generated)
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
            if parsed.family is ModelFamily.QWEN3_5_OMNI:
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

    @classmethod
    def _diagnostic_context_evidence(
        cls,
        manifest: RunManifest,
        effective: Mapping[str, Any],
        *,
        requested: bool,
        divergence_observed: bool,
        slice_reference_refs: Sequence[ArtifactRef],
    ) -> dict[str, Any]:
        """Describe verified diagnostic evidence without overstating its scope."""

        evidence: dict[str, Any] = {
            "requested": bool(requested),
            "triggered": bool(requested and divergence_observed),
            "reason": (
                "observed_numerical_divergence"
                if requested and divergence_observed
                else "not_triggered"
            ),
        }
        if not requested or not divergence_observed:
            return evidence

        index = cls._runtime_index_for_manifest(manifest)
        cl_key = str(int(effective["context_length"]))
        entries = list(
            (index.get("diagnostic_contexts") or {}).get(cl_key) or ()
        )
        artifacts_by_path = {
            artifact.path.expanduser().resolve(): artifact
            for artifact in manifest.artifacts
        }
        contexts: list[dict[str, Any]] = []
        for entry in entries:
            path = Path(str(entry["context_path"])).expanduser().resolve()
            artifact = artifacts_by_path.get(path)
            if artifact is None:
                raise InvalidSpecError(
                    "runtime_index references a diagnostic context that is "
                    "not a verified build artifact",
                    stage="validate",
                    details={"context_path": os.fspath(path)},
                )
            verify_artifact(artifact)
            contexts.append(
                {
                    **dict(entry),
                    "artifact": _jsonable(artifact),
                }
            )
        if contexts:
            evidence.update(
                {
                    "status": "ready",
                    "evidence_scope": "op_intermediate_contexts",
                    "op_level_dump_available": True,
                    "contexts": contexts,
                }
            )
            return evidence

        evidence.update(
            {
                "status": "slice_tensor_only",
                "evidence_scope": "verified_slice_tensor_boundaries",
                "op_level_dump_available": False,
                "contexts": [],
                "slice_reference_manifests": [
                    _jsonable(ref) for ref in slice_reference_refs
                ],
                "limitation": (
                    "the build produced no diagnostic context; the report may "
                    "localize a slice/tensor boundary but cannot claim an "
                    "operator-level intermediate dump"
                ),
            }
        )
        return evidence

    @staticmethod
    def _capture_onnx_reference(
        vector_manifest: str | Path,
        model_path: str | Path,
        output_dir: Path,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> Path:
        """Copy inputs into the run tree and capture an auditable ORT reference."""

        source_path = Path(vector_manifest).expanduser().resolve()
        source = VectorPreparer.load_manifest(
            source_path,
            expected_sha256=expected_manifest_sha256,
        )
        inputs = VectorPreparer.load_tensors(
            source_path,
            section="inputs",
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if not inputs:
            raise ValueError(
                "ONNX Runtime fallback requires a vector manifest with model inputs"
            )
        preparer = VectorPreparer(output_dir / "onnxruntime-reference")
        copied = preparer.prepare_case(
            source.case_id,
            inputs,
            metadata={
                **dict(source.metadata),
                "reference_request": "golden_missing_fallback",
                "source_manifest_path": os.fspath(source_path),
                "source_manifest_sha256": sha256_file(source_path),
            },
        )
        return preparer.capture_onnx(
            copied,
            model_path,
            destination_name="vector_manifest.onnx-reference.json",
        )

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
            generated, _ = self._generate_family_config(spec)
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
                        NativeKvGraphExpectation(
                            graph_name=str(value["graph_name"]),
                            ar=int(value["ar"]),
                            input_names=tuple(
                                str(name) for name in value["input_names"]
                            ),
                            output_names=tuple(
                                str(name) for name in value["output_names"]
                            ),
                        )
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

    def run_graph(
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
            context_path = selected.get("context_path")
            graph_name = selected.get("graph_name")
            vector_manifest = selected.get("vector_manifest")
            if context_path is None or not graph_name or vector_manifest is None:
                raise ValueError(
                    "run_graph requires context_path, graph_name, and vector_manifest"
                )
            inputs = self._manifest_inputs(
                vector_manifest,
                section=str(selected.get("input_section", "inputs")),
                sha256=selected.get("vector_manifest_sha256"),
            )
            push_files = self._device_stage_files(
                output_dir,
                contexts=(context_path,),
                vector_manifests=(vector_manifest,),
            )
            with self._device_stage(
                manifest,
                adapter,
                stage_name="run_graph",
                input_manifest_sha256=manifest_sha256,
                stage_config=selected,
                push_files=push_files,
            ) as device_stage:
                raw_result = adapter.run_graph(
                    context_path,
                    inputs,
                    graph_name=str(graph_name),
                    device=device_stage.device,
                    native_io=bool(selected.get("native_io", False)),
                    **self._execution_options(selected),
                )
                outputs = _output_mapping(raw_result, graph_name=str(graph_name))
                device_identifier = device_stage.identifier
                remote_attempt_dir = device_stage.adb.attempt_dir
            result_manifest = VectorPreparer(output_dir).prepare_case(
                "run_graph",
                inputs,
                goldens=outputs,
                metadata={"graph_name": str(graph_name), "role": "device_outputs"},
            )
            refs = (
                ArtifactRef.from_path(
                    result_manifest,
                    kind=ArtifactKind.TEST_VECTORS,
                    logical_name="run_graph_outputs",
                ),
            )
            return {
                "graph_name": str(graph_name),
                "output_manifest": _jsonable(refs[0]),
                "outputs": _jsonable(outputs),
            }, refs, {
                "output_count": len(outputs),
                "device_identifier": device_identifier,
                "remote_attempt_dir": remote_attempt_dir,
                "remote_cleanup": "confirmed",
            }

        return self._continuation_operation(
            "run_graph",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
        )

    @staticmethod
    def _chain_executors(
        adapter: Any,
        contexts: Mapping[str, Any],
        *,
        device: Any,
        native_io: bool,
        execution_options: Mapping[str, Any],
    ) -> dict[str, Callable[[Mapping[str, np.ndarray], Any], Mapping[str, np.ndarray]]]:
        executors: dict[
            str, Callable[[Mapping[str, np.ndarray], Any], Mapping[str, np.ndarray]]
        ] = {}
        for slice_name, context in contexts.items():
            def execute(
                inputs: Mapping[str, np.ndarray],
                invocation: Any,
                *,
                compiled_context: Any = context,
            ) -> Mapping[str, np.ndarray]:
                result = adapter.run_graph(
                    compiled_context,
                    inputs,
                    graph_name=invocation.graph_name,
                    device=device,
                    native_io=native_io,
                    **dict(execution_options),
                )
                return _output_mapping(result, graph_name=invocation.graph_name)

            executors[str(slice_name)] = execute
        return executors

    @staticmethod
    def _recording_chain_executors(
        base: Mapping[str, Callable[[Mapping[str, np.ndarray], Any], Mapping[str, np.ndarray]]],
        recorded: dict[str, dict[str, Any]],
    ) -> dict[str, Callable[[Mapping[str, np.ndarray], Any], Mapping[str, np.ndarray]]]:
        """Wrap chain executors so each slice's exact inputs are captured.

        A profiled per-slice execute needs the inputs that slice is really fed,
        and in a chain those come from the previous slice at run time. Recording
        them during one ordinary pass is what lets the device capture replay
        each slice faithfully instead of guessing.
        """

        wrapped: dict[
            str, Callable[[Mapping[str, np.ndarray], Any], Mapping[str, np.ndarray]]
        ] = {}
        for slice_name, executor in base.items():
            def record(
                inputs: Mapping[str, np.ndarray],
                invocation: Any,
                *,
                bound: str = str(slice_name),
                inner: Any = executor,
            ) -> Mapping[str, np.ndarray]:
                recorded[bound] = {
                    "inputs": {
                        str(name): np.asarray(value)
                        for name, value in inputs.items()
                    },
                    "graph_name": str(invocation.graph_name),
                }
                return inner(inputs, invocation)

            wrapped[str(slice_name)] = record
        return wrapped

    def _chain_device_execution(
        self,
        adapter: Any,
        recorded: Mapping[str, Mapping[str, Any]],
        contexts: Mapping[str, Any],
        *,
        device: Any,
        native_io: bool,
        execution_options: Mapping[str, Any],
        working_dir: Path,
    ) -> dict[str, Any]:
        """Per-slice device execute time for a chain run.

        Each slice is profiled with the inputs it was actually fed, recorded
        during the preceding ordinary chain pass.
        """

        by_slice: dict[str, Any] = {}
        for slice_name, entry in recorded.items():
            context = contexts.get(slice_name)
            if context is None:
                continue
            by_slice[str(slice_name)] = self._device_execution_block(
                adapter,
                context,
                entry["inputs"],
                graph_name=str(entry["graph_name"]),
                device=device,
                native_io=native_io,
                execution_options=execution_options,
                working_dir=working_dir / str(slice_name),
            )

        measured = {
            name: block
            for name, block in by_slice.items()
            if isinstance(block, Mapping) and block.get("available") is not False
        }
        block: dict[str, Any] = {
            "schema": DEVICE_EXECUTION_SCHEMA,
            "meter": DEVICE_EXECUTION_METER,
            "lane": "low_level",
            "policy": "report_only",
            "scope": "chain",
            "statistic": "mean",
            "slice_count": len(by_slice),
            "measured_slice_count": len(measured),
            "by_slice": by_slice,
        }
        if measured and len(measured) == len(by_slice):
            # Chain slices run sequentially, so their device execute times add.
            # This is a sum of per-slice means, not a measured end-to-end
            # number, and says so.
            block["totals"] = {
                key: sum(
                    float(item[key])
                    for item in measured.values()
                    if isinstance(item.get(key), (int, float))
                )
                for key in (
                    "accelerator_compute_us",
                    "accelerator_execute_us",
                    "qnn_execute_us",
                )
                if all(
                    isinstance(item.get(key), (int, float))
                    for item in measured.values()
                )
            }
            block["totals_basis"] = "sum_of_per_slice_means_slices_run_sequentially"
        else:
            block["available"] = False
            block["reason"] = (
                "not every chain slice produced device evidence; a partial "
                "chain total would understate the work"
            )
        return block

    def run_chain(
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
            routes = selected.get("routes")
            contexts = selected.get("contexts")
            if not routes or not isinstance(contexts, Mapping):
                raise ValueError("run_chain requires config.routes and config.contexts")
            mode = str(selected.get("mode", "device_chain"))
            inline_cases: list[Mapping[str, Any]] = []
            if "steps" in selected:
                inline_cases.extend(
                    dict(step.get("inputs", {}))
                    for step in selected["steps"]
                    if isinstance(step, Mapping)
                )
            initial_native_state = self._tensor_mapping(
                selected.get("initial_native_state", {})
            )
            if initial_native_state:
                inline_cases.append(initial_native_state)
            vector_manifests = (
                (selected["vector_manifest"],)
                if selected.get("vector_manifest") is not None
                else ()
            )
            push_files = self._device_stage_files(
                output_dir,
                contexts=tuple(contexts.values()),
                vector_manifests=vector_manifests,
                inline_cases=tuple(inline_cases),
            )
            execution_options = self._execution_options(selected)
            with self._device_stage(
                manifest,
                adapter,
                stage_name="run_chain",
                input_manifest_sha256=manifest_sha256,
                stage_config=selected,
                push_files=push_files,
            ) as device_stage:
                runner = SliceChainRunner(
                    routes,
                    self._chain_executors(
                        adapter,
                        contexts,
                        device=device_stage.device,
                        native_io=bool(selected.get("native_io", False)),
                        execution_options=execution_options,
                    ),
                )
                device_identifier = device_stage.identifier
                remote_attempt_dir = device_stage.adb.attempt_dir
                if "steps" in selected:
                    result = runner.run_sequence(
                        selected["steps"],
                        mode=mode,
                        initial_native_state=initial_native_state,
                    )
                    last = result.steps[-1]
                else:
                    vector_manifest = selected.get("vector_manifest")
                    if vector_manifest is None:
                        raise ValueError(
                            "run_chain requires vector_manifest when steps are absent"
                        )
                    inputs = self._manifest_inputs(
                        vector_manifest,
                        sha256=selected.get("vector_manifest_sha256"),
                    )
                    ar = int(
                        selected.get("ar", manifest.build_spec.sequence.ars[0])
                    )
                    if mode == "teacher_forced":
                        teacher_inputs = self._slice_tensor_tree(
                            selected.get("teacher_inputs", {}),
                            section="inputs",
                        )
                        result = runner.run_teacher_forced(
                            inputs,
                            teacher_inputs,
                            ar=ar,
                            initial_native_state=initial_native_state,
                        )
                    else:
                        result = runner.run_device_chain(
                            inputs,
                            ar=ar,
                            initial_native_state=initial_native_state,
                        )
                    last = result

            refs: list[ArtifactRef] = []
            output_manifests: dict[str, Any] = {}
            for slice_name, outputs in last.outputs_by_slice().items():
                path = VectorPreparer(output_dir / slice_name).prepare_case(
                    f"{mode}_{slice_name}",
                    outputs,
                    roles={name: "device_output" for name in outputs},
                    metadata={"slice": slice_name, "mode": mode},
                )
                ref = ArtifactRef.from_path(
                    path,
                    kind=ArtifactKind.TEST_VECTORS,
                    logical_name=f"{mode}_{slice_name}_outputs",
                )
                refs.append(ref)
                output_manifests[slice_name] = _jsonable(ref)
            return {
                "mode": mode,
                "output_manifests": output_manifests,
                "final_outputs": _jsonable(last.final_outputs),
                "native_state_slots": list(last.native_state_slots),
            }, tuple(refs), {
                "slice_count": len(refs),
                "device_identifier": device_identifier,
                "remote_attempt_dir": remote_attempt_dir,
                "remote_cleanup": "confirmed",
            }

        return self._continuation_operation(
            "run_chain",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
        )

    def profile(
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
            required = ("context_path", "graph_name", "vector_manifest")
            missing = [key for key in required if selected.get(key) is None]
            if missing:
                raise ValueError(f"profile missing config fields: {missing}")
            inputs = self._manifest_inputs(
                selected["vector_manifest"],
                sha256=selected.get("vector_manifest_sha256"),
            )
            push_files = self._device_stage_files(
                output_dir,
                contexts=(selected["context_path"],),
                vector_manifests=(selected["vector_manifest"],),
            )
            with self._device_stage(
                manifest,
                adapter,
                stage_name="profile",
                input_manifest_sha256=manifest_sha256,
                stage_config=selected,
                push_files=push_files,
            ) as device_stage:
                result = adapter.profile(
                    selected["context_path"],
                    inputs,
                    graph_name=str(selected["graph_name"]),
                    device=device_stage.device,
                    native_io=bool(selected.get("native_io", False)),
                    level=str(selected.get("level", "detailed")),
                    option=str(selected.get("option", "optrace")),
                    **self._execution_options(selected),
                )
                device_identifier = device_stage.identifier
                remote_attempt_dir = device_stage.adb.attempt_dir
            report_payload = {
                "graph_name": result.graph_name,
                "level": result.level,
                "option": result.option,
                "reports": [_jsonable(report) for report in result.reports],
            }
            report_ref = atomic_publish_json(
                output_dir / "profile_report.json",
                report_payload,
                kind=ArtifactKind.REPORT,
                logical_name="profile_report",
            )
            return report_payload, (report_ref,), {
                "report_count": len(result.reports),
                "device_identifier": device_identifier,
                "remote_attempt_dir": remote_attempt_dir,
                "remote_cleanup": "confirmed",
            }

        return self._continuation_operation(
            "profile",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
        )

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

        def run_one(
            manifest: RunManifest,
            adapter: Any,
            output_dir: Path,
            selected_config: Mapping[str, Any],
            *,
            report_suffix: str = "",
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            effective = dict(selected_config)
            explicit_chain = effective.get("routes") is not None
            explicit_graph = all(
                effective.get(key) is not None
                for key in ("context_path", "graph_name", "vector_manifest")
            )
            explicit_genai = effective.get("container_path") is not None
            if not explicit_chain and not explicit_graph and not explicit_genai:
                effective = self._automatic_runtime_binding(
                    manifest,
                    effective,
                )
            effective = self._enforce_qwen3_vl_runtime_scope(
                manifest,
                effective,
                stage="benchmark",
            )
            self._reject_genai_chain_keys(effective, stage="benchmark")

            spec = manifest.build_spec
            optrace_enabled = bool(
                effective.get("optrace", spec.benchmark.optrace)
            )
            profile_level = str(
                effective.get("profile_level", "detailed")
            )
            profile_option = str(
                effective.get("profile_option", "optrace")
            )
            if optrace_enabled and profile_option.lower() != "optrace":
                raise InvalidSpecError(
                    "benchmark.optrace requires profile_option='optrace'",
                    stage="benchmark",
                    details={"profile_option": profile_option},
                )
            profile_entries: list[dict[str, Any]] = []
            normalized_profile_ops: list[dict[str, Any]] = []
            profile_source_refs: list[ArtifactRef] = []
            self._preflight(adapter, spec)
            execution_options = self._execution_options(effective)
            native_io = bool(effective.get("native_io", False))
            scope = "graph"
            generation_metrics: dict[str, Any] | None = None
            generated_text_sha256: str | None = None
            generated_text_length: int | None = None
            profile_routes: Sequence[Mapping[str, Any]] | None = None
            profile_contexts: Mapping[str, Any] | None = None
            profile_inputs: dict[str, np.ndarray] | None = None
            profile_ar: int | None = None
            profile_initial_native_state: dict[str, np.ndarray] = {}
            device_execution: dict[str, Any] | None = None
            execution_owner: Any = None
            chain_slice_inputs: dict[str, dict[str, Any]] = {}
            loaded_contexts: dict[str, Any] = {}
            profile_claim_scope = (
                "production_runtime_optrace"
            )
            if effective.get("lane") == "genai_builder" or explicit_genai:
                if not effective.get("runtime_supported", True):
                    raise InvalidSpecError(
                        "the saved GenAI container is marked runtime unsupported "
                        "by the selected QAIRT workflow",
                        stage="benchmark",
                        details={
                            "family": effective.get("family"),
                            "container_path": effective.get("container_path"),
                        },
                    )
                if effective.get("container_path") is None:
                    raise ValueError(
                        "GenAI benchmark requires container_path"
                    )
                if effective.get("prompt") is not None:
                    prompt: Any = effective["prompt"]
                elif effective.get("prompt_path") is not None:
                    prompt = Path(effective["prompt_path"]).expanduser().resolve()
                    if not prompt.is_file():
                        raise FileNotFoundError(
                            f"GenAI prompt_path does not exist: {prompt}"
                        )
                else:
                    raise InvalidSpecError(
                        "GenAI benchmark requires stage_configs.benchmark.prompt "
                        "or prompt_path so the measured workload is explicit",
                        stage="benchmark",
                    )
                tensor_runtime = effective.get("tensor_runtime")
                if optrace_enabled:
                    if not isinstance(tensor_runtime, Mapping):
                        raise InvalidSpecError(
                            "GenAI benchmark.optrace requires an auditable "
                            "public raw CompiledModel tensor runtime",
                            stage="benchmark",
                            details={
                                "family": effective.get("family"),
                                "container_path": effective.get(
                                    "container_path"
                                ),
                            },
                        )
                    raw_routes = tensor_runtime.get("routes")
                    raw_contexts = tensor_runtime.get("contexts")
                    if (
                        not isinstance(raw_routes, Sequence)
                        or isinstance(
                            raw_routes,
                            (str, bytes, bytearray),
                        )
                        or not raw_routes
                        or not isinstance(raw_contexts, Mapping)
                        or not raw_contexts
                    ):
                        raise InvalidSpecError(
                            "GenAI tensor runtime is missing raw routes or "
                            "compiled contexts required for optrace",
                            stage="benchmark",
                        )
                    if effective.get("vector_manifest") is None:
                        raise InvalidSpecError(
                            "GenAI benchmark.optrace requires the exact per-AR "
                            "vector manifest selected by runtime_index",
                            stage="benchmark",
                        )
                    profile_routes = list(raw_routes)
                    profile_contexts = dict(raw_contexts)
                    profile_inputs = self._manifest_inputs(
                        effective["vector_manifest"],
                        sha256=effective.get("vector_manifest_sha256"),
                    )
                    profile_initial_native_state = (
                        self._initial_native_state_from_routes(
                            profile_routes,
                            profile_inputs,
                        )
                    )
                    profile_ar = int(
                        effective.get("ar", spec.sequence.ars[0])
                    )
                    profile_claim_scope = (
                        "raw_compiled_slices_not_generation_wall_latency"
                    )
                context_paths = (
                    tuple(profile_contexts.values())
                    if profile_contexts is not None
                    else ()
                )
                vector_manifests = (
                    (effective["vector_manifest"],)
                    if profile_inputs is not None
                    else ()
                )
                inline_cases: list[Mapping[str, Any]] = []
                if profile_initial_native_state:
                    inline_cases.append(profile_initial_native_state)
                scope = "genai_generation"
            elif effective.get("routes") is not None:
                contexts = effective.get("contexts")
                if not isinstance(contexts, Mapping) or not contexts:
                    raise ValueError(
                        "chain benchmark requires config.contexts mapped by slice"
                    )
                context_paths = tuple(contexts.values())
                vector_manifests = (
                    (effective["vector_manifest"],)
                    if effective.get("vector_manifest") is not None
                    else ()
                )
                inline_cases = [
                    dict(step.get("inputs", {}))
                    for step in effective.get("steps", ())
                    if isinstance(step, Mapping)
                ]
                initial_native_state = self._tensor_mapping(
                    effective.get("initial_native_state", {})
                )
                if (
                    not initial_native_state
                    and effective.get("vector_manifest") is not None
                ):
                    manifest_inputs = self._manifest_inputs(
                        effective["vector_manifest"],
                        sha256=effective.get("vector_manifest_sha256"),
                    )
                    initial_native_state = self._initial_native_state_from_routes(
                        effective["routes"],
                        manifest_inputs,
                    )
                if initial_native_state:
                    inline_cases.append(initial_native_state)
                profile_routes = list(effective["routes"])
                profile_contexts = dict(contexts)
                profile_initial_native_state = dict(initial_native_state)

            else:
                required = ("context_path", "graph_name", "vector_manifest")
                missing = [key for key in required if effective.get(key) is None]
                if missing:
                    raise ValueError(f"benchmark missing config fields: {missing}")
                context_paths = (effective["context_path"],)
                vector_manifests = (effective["vector_manifest"],)
                inline_cases = []

            push_files = self._device_stage_files(
                output_dir,
                contexts=context_paths,
                vector_manifests=vector_manifests,
                inline_cases=tuple(inline_cases),
            )

            warmup = int(effective.get("warmup_runs", spec.benchmark.warmup_runs))
            repeats = int(
                effective.get("measured_runs", spec.benchmark.measured_runs)
            )
            diagnoser = LatencyDiagnoser()
            with self._device_stage(
                manifest,
                adapter,
                stage_name="benchmark",
                input_manifest_sha256=manifest_sha256,
                stage_config=effective,
                push_files=push_files,
            ) as device_stage:
                if effective.get("lane") == "genai_builder" or explicit_genai:
                    executor = adapter.create_genai_executor(
                        effective["container_path"],
                        device=device_stage.device,
                    )
                    last_generation: dict[str, Any] = {}

                    def invoke() -> Any:
                        result = executor.generate(prompt)
                        last_generation["result"] = result
                        return result

                    try:
                        measurement = diagnoser.measure(
                            invoke,
                            warmup=warmup,
                            repeats=repeats,
                        )
                        aa_measurement = (
                            diagnoser.calibrate_aa(
                                invoke,
                                warmup=warmup,
                                repeats=repeats,
                            )
                            if bool(effective.get("aa_calibration", True))
                            else None
                        )
                        result = last_generation.get("result")
                        metrics_value = getattr(result, "metrics", None)
                        if metrics_value is not None:
                            generation_metrics = _jsonable(metrics_value)
                        generated_text = getattr(result, "generated_text", None)
                        if isinstance(generated_text, str):
                            generated_text_length = len(generated_text)
                            generated_text_sha256 = hashlib.sha256(
                                generated_text.encode("utf-8")
                            ).hexdigest()
                    finally:
                        adapter.clean_genai_executor(executor)
                elif effective.get("routes") is not None:
                    for slice_name, context_path in contexts.items():
                        if hasattr(adapter, "load_compiled"):
                            loaded_contexts[str(slice_name)] = (
                                adapter.load_compiled(context_path)
                            )
                        elif hasattr(adapter, "_compiled_model"):
                            loaded_contexts[str(slice_name)] = (
                                adapter._compiled_model(context_path)
                            )
                        else:
                            loaded_contexts[str(slice_name)] = context_path
                    runner = SliceChainRunner(
                        effective["routes"],
                        self._recording_chain_executors(
                            self._chain_executors(
                                adapter,
                                loaded_contexts,
                                device=device_stage.device,
                                native_io=native_io,
                                execution_options=execution_options,
                            ),
                            chain_slice_inputs,
                        ),
                    )
                    scope = (
                        "chain_sequence" if "steps" in effective else "chain"
                    )
                    if "steps" in effective:
                        steps = effective["steps"]

                        def invoke() -> Any:
                            return runner.run_sequence(
                                steps,
                                mode="device_chain",
                                initial_native_state=initial_native_state,
                            )

                    else:
                        if effective.get("vector_manifest") is None:
                            raise ValueError(
                                "chain benchmark requires vector_manifest "
                                "when steps are absent"
                            )
                        inputs = self._manifest_inputs(
                            effective["vector_manifest"],
                            sha256=effective.get("vector_manifest_sha256"),
                        )
                        ar = int(
                            effective.get("ar", spec.sequence.ars[0])
                        )
                        profile_inputs = dict(inputs)
                        profile_ar = ar

                        def invoke() -> Any:
                            return runner.run_device_chain(
                                inputs,
                                ar=ar,
                                initial_native_state=initial_native_state,
                            )

                else:
                    inputs = self._manifest_inputs(
                        effective["vector_manifest"],
                        sha256=effective.get("vector_manifest_sha256"),
                    )
                    if hasattr(adapter, "load_compiled"):
                        compiled = adapter.load_compiled(
                            effective["context_path"]
                        )
                    elif hasattr(adapter, "_compiled_model"):
                        compiled = adapter._compiled_model(
                            effective["context_path"]
                        )
                    else:
                        compiled = effective["context_path"]

                    def invoke() -> Any:
                        return adapter.run_graph(
                            compiled,
                            inputs,
                            graph_name=str(effective["graph_name"]),
                            device=device_stage.device,
                            native_io=native_io,
                            **execution_options,
                        )
                    profile_inputs = dict(inputs)
                    profile_ar = int(
                        effective.get("ar", spec.sequence.ars[0])
                    )
                    # Ask the device what it actually spent, before the model
                    # is initialized: an initialized model carries an execution
                    # context created with profiling disabled, and QAIRT would
                    # then report no profiling data at all.
                    device_execution = self._device_execution_block(
                        adapter,
                        compiled,
                        inputs,
                        graph_name=str(effective["graph_name"]),
                        device=device_stage.device,
                        native_io=native_io,
                        execution_options=execution_options,
                        working_dir=output_dir / "device_profiling",
                    )
                    # Without initialize(), QAIRT rebuilds the backend and the
                    # inferencer on every call.  This is its documented way to
                    # execute repeatedly against one model/backend/device.
                    execution_owner = self._initialize_execution(
                        adapter,
                        compiled,
                        device=device_stage.device,
                    )

                if scope != "genai_generation":
                    # Our own setup -- context loading, Device construction, ADB
                    # staging, graph-runner setup -- is complete before the
                    # timer.  What QAIRT does inside one call is not: on the
                    # low-level lane it relaunches qnn-net-run per call.
                    try:
                        measurement = diagnoser.measure(
                            invoke,
                            warmup=warmup,
                            repeats=repeats,
                        )
                        aa_measurement = (
                            diagnoser.calibrate_aa(
                                invoke,
                                warmup=warmup,
                                repeats=repeats,
                            )
                            if bool(effective.get("aa_calibration", True))
                            else None
                        )
                    finally:
                        if execution_owner is not None:
                            adapter.release_execution(execution_owner)
                            execution_owner = None

                if device_execution is None and chain_slice_inputs:
                    # The measured chain pass recorded what each slice was
                    # actually fed, so each one can now be profiled with its
                    # real inputs rather than a guess.
                    device_execution = self._chain_device_execution(
                        adapter,
                        chain_slice_inputs,
                        loaded_contexts,
                        device=device_stage.device,
                        native_io=native_io,
                        execution_options=execution_options,
                        working_dir=output_dir / "device_profiling",
                    )
                if device_execution is None:
                    # Latency is device time, so a scope with no device meter
                    # says so with a cause rather than quietly omitting the
                    # block and leaving the wall number to look like latency.
                    device_execution = {
                        "schema": DEVICE_EXECUTION_SCHEMA,
                        "policy": "report_only",
                        "available": False,
                        "reason": _DEVICE_EXECUTION_UNAVAILABLE.get(
                            scope, "no device meter is wired for this scope"
                        ),
                    }
                if optrace_enabled:
                    if not hasattr(adapter, "profile"):
                        raise InvalidSpecError(
                            "the selected QAIRT adapter does not expose the "
                            "public profile API required by benchmark.optrace",
                            stage="benchmark",
                        )

                    def capture_profile(
                        compiled_context: Any,
                        profile_inputs: Mapping[str, np.ndarray],
                        *,
                        slice_id: str,
                        graph_name: str,
                        step_index: int,
                    ) -> dict[str, np.ndarray]:
                        profiled = adapter.profile(
                            compiled_context,
                            profile_inputs,
                            graph_name=graph_name,
                            device=device_stage.device,
                            native_io=native_io,
                            level=profile_level,
                            option=profile_option,
                            **execution_options,
                        )
                        raw_reports: list[Any] = []
                        source_refs: list[ArtifactRef] = []
                        profile_call_index = len(profile_entries)
                        for report_index, raw_report in enumerate(
                            tuple(getattr(profiled, "reports", ()) or ())
                        ):
                            report_payload, source_ref = (
                                self._profile_report_payload(raw_report)
                            )
                            raw_reports.append(report_payload)
                            captured_ref = atomic_publish_json(
                                output_dir
                                / "optrace"
                                / (
                                    f"profile-{profile_call_index:04d}-"
                                    f"report-{report_index:04d}.json"
                                ),
                                {
                                    "schema": (
                                        "qairt-agent.captured-profile-report.v1"
                                    ),
                                    "source_artifact": (
                                        _jsonable(source_ref)
                                        if source_ref is not None
                                        else None
                                    ),
                                    "report": report_payload,
                                },
                                kind=ArtifactKind.REPORT,
                            )
                            source_refs.append(captured_ref)
                        records = self._normalize_optrace_records(
                            raw_reports,
                            slice_id=slice_id,
                            graph_name=graph_name,
                            step_index=step_index,
                        )
                        profile_entries.append(
                            {
                                "slice_id": slice_id,
                                "graph_name": graph_name,
                                "step_index": step_index,
                                "level": str(
                                    getattr(
                                        profiled,
                                        "level",
                                        profile_level,
                                    )
                                ),
                                "option": str(
                                    getattr(
                                        profiled,
                                        "option",
                                        profile_option,
                                    )
                                ),
                                "reports": raw_reports,
                                "report_artifacts": [
                                    _jsonable(ref) for ref in source_refs
                                ],
                                "normalized_op_count": len(records),
                            }
                        )
                        normalized_profile_ops.extend(records)
                        profile_source_refs.extend(source_refs)
                        return _output_mapping(
                            getattr(profiled, "execution_result", None),
                            graph_name=graph_name,
                        )

                    if profile_routes is not None:
                        assert profile_contexts is not None
                        loaded_profile_contexts: dict[str, Any] = {}
                        for (
                            slice_name,
                            context_path,
                        ) in profile_contexts.items():
                            if hasattr(adapter, "load_compiled"):
                                loaded_profile_contexts[str(slice_name)] = (
                                    adapter.load_compiled(context_path)
                                )
                            elif hasattr(adapter, "_compiled_model"):
                                loaded_profile_contexts[str(slice_name)] = (
                                    adapter._compiled_model(context_path)
                                )
                            else:
                                loaded_profile_contexts[str(slice_name)] = (
                                    context_path
                                )
                        profile_executors: dict[
                            str,
                            Callable[
                                [Mapping[str, np.ndarray], Any],
                                Mapping[str, np.ndarray],
                            ],
                        ] = {}
                        for (
                            slice_name,
                            compiled_context,
                        ) in loaded_profile_contexts.items():
                            def execute_profile(
                                slice_inputs: Mapping[str, np.ndarray],
                                invocation: Any,
                                *,
                                context: Any = compiled_context,
                                bound_slice: str = str(slice_name),
                            ) -> Mapping[str, np.ndarray]:
                                return capture_profile(
                                    context,
                                    slice_inputs,
                                    slice_id=bound_slice,
                                    graph_name=invocation.graph_name,
                                    step_index=int(invocation.step_index),
                                )

                            profile_executors[str(slice_name)] = (
                                execute_profile
                            )
                        profile_runner = SliceChainRunner(
                            profile_routes,
                            profile_executors,
                        )
                        if "steps" in effective:
                            profile_runner.run_sequence(
                                steps,
                                mode="device_chain",
                                initial_native_state=initial_native_state,
                            )
                        else:
                            if profile_inputs is None or profile_ar is None:
                                raise InvalidSpecError(
                                    "optrace chain execution is missing exact "
                                    "inputs or AR selection",
                                    stage="benchmark",
                                )
                            profile_runner.run_device_chain(
                                profile_inputs,
                                ar=profile_ar,
                                initial_native_state=(
                                    profile_initial_native_state
                                ),
                            )
                    else:
                        capture_profile(
                            compiled,
                            inputs,
                            slice_id=str(
                                effective.get("slice_id") or "model"
                            ),
                            graph_name=str(effective["graph_name"]),
                            step_index=0,
                        )
                    if not normalized_profile_ops:
                        raise InvalidSpecError(
                            "benchmark.optrace produced no structured per-op "
                            "cycle records; raw profiler reports cannot support "
                            "automatic latency attribution",
                            stage="benchmark",
                            details={
                                "profile_count": len(profile_entries),
                                "required_fields": [
                                    "op_id/name",
                                    "cycles or thread_cycles",
                                ],
                            },
                        )
                device_identifier = device_stage.identifier
                remote_attempt_dir = device_stage.adb.attempt_dir
            optrace_ref: ArtifactRef | None = None
            if optrace_enabled:
                runtime_indexes = [
                    artifact
                    for artifact in manifest.artifacts
                    if artifact.logical_name == "runtime_index"
                ]
                if len(runtime_indexes) != 1:
                    raise InvalidSpecError(
                        "benchmark.optrace requires exactly one build "
                        "runtime_index artifact",
                        stage="benchmark",
                        details={
                            "runtime_index_count": len(runtime_indexes)
                        },
                    )
                verify_artifact(runtime_indexes[0])
                optrace_ref = atomic_publish_json(
                    output_dir / f"optrace_evidence{report_suffix}.json",
                    {
                        "schema": "qairt-agent.optrace-evidence.v1",
                        "source_manifest_sha256": manifest_sha256,
                        "runtime_index": _jsonable(runtime_indexes[0]),
                        "runtime_binding": {
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
                                "component",
                                "coverage",
                            )
                            if effective.get(key) is not None
                        },
                        "device_identifier": device_identifier,
                        "profile_level": profile_level,
                        "profile_option": profile_option,
                        "profiles": profile_entries,
                        "ops": normalized_profile_ops,
                        "profile_scope": profile_claim_scope,
                        "claim_scope": (
                            "reported_op_work_not_additive_wall_latency"
                        ),
                    },
                    kind=ArtifactKind.REPORT,
                    logical_name=f"optrace_evidence{report_suffix}",
                )
            payload: dict[str, Any] = {
                "policy": "report_only",
                "scope": scope,
                # Latency means device time. The host wall number is kept for
                # diagnosing the harness -- it still detects ADB, container and
                # transport degradation -- but it is not this report's latency
                # and never grounds a regression verdict.
                "latency_metric": (
                    "device_execution"
                    if device_execution is not None
                    and device_execution.get("available") is not False
                    else "unavailable"
                ),
                "harness_diagnostics": {
                    "metric_name": "host_orchestrated_call_latency",
                    "not_latency": True,
                    "note": (
                        "host wall time around one SDK call, kept to detect "
                        "harness and transport degradation; the reported "
                        "latency is the block named by 'latency_metric'"
                    ),
                    "measurement": measurement.to_dict(),
                    # Our own setup is outside the timer; what the SDK does
                    # inside one call is not, and on the low-level lane that
                    # includes a fresh qnn-net-run process per call.  Reporting
                    # a single "setup_excluded" flag conflated the two.
                    "harness_setup_excluded": True,
                    "sdk_per_call_setup_included": (
                        "unverified"
                        if scope == "genai_generation"
                        else True
                    ),
                    "measurement_scope": {
                        "clock": "host_perf_counter_ns",
                        "includes": "host_to_sdk_to_device_round_trip",
                        "device_side_sync_barrier": False,
                        "note": (
                            "the QAIRT Python API exposes no device-side "
                            "synchronization barrier, so each sample is the warmed "
                            "host wall time around one call, including whatever "
                            "per-call setup the SDK performs inside it; this is not "
                            "device execution time -- see 'device_execution'"
                        ),
                        "excluded_from_timer": [
                            "context_loading",
                            "device_construction",
                            "adb_staging",
                            "graph_runner_setup",
                        ],
                        "included_in_sample": (
                            []
                            if scope == "genai_generation"
                            else [
                                "qnn_net_run_process_launch",
                                "per_call_context_load",
                                "hvx_hmx_power_on_and_acquire",
                                "per_call_deinit",
                                "adb_input_push_and_output_pull",
                            ]
                        ),
                        "sample_unit": (
                            "generate_call"
                            if scope == "genai_generation"
                            else "graph_invocation"
                        ),
                    },
                },
                "runtime_binding": {
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
                        "container_path",
                        "component",
                        "coverage",
                        "excluded_components",
                        "graph_ar",
                    )
                    if effective.get(key) is not None
                },
            }
            requested_ar_values = [
                int(value) for value in spec.sequence.ars
            ]
            if scope == "genai_generation":
                payload["coverage"] = {
                    "mode": "executor_managed_generation",
                    "requested_ars": requested_ar_values,
                    "executed_ars": "not_observable_via_public_generation_api",
                    "complete": None,
                    "prefill_decode_scope": True,
                    "graph_ar_coverage_proven": False,
                    "limitation": (
                        "the public GenAI generation executor reports end-to-end "
                        "generation latency but does not expose which AR graph "
                        "served each prefill/decode step"
                    ),
                }
            elif effective.get("ar") is not None:
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
            if device_execution is not None:
                payload["device_execution"] = device_execution
            if generation_metrics is not None:
                payload["generation_metrics"] = generation_metrics
            if generated_text_sha256 is not None:
                payload["generated_text"] = {
                    "sha256": generated_text_sha256,
                    "character_count": generated_text_length,
                }
            token_count = effective.get("token_count")
            token_source: str | None = None
            if token_count is not None:
                normalized_tokens = int(token_count)
                if normalized_tokens <= 0:
                    raise ValueError("benchmark token_count must be positive")
                token_source = "caller"
            else:
                reported = _sdk_generated_token_count(generation_metrics)
                if reported is not None:
                    normalized_tokens = reported
                    token_source = "sdk_metrics"
            if token_source is not None:
                payload["token_count"] = normalized_tokens
                payload["ms_per_token_source"] = token_source
                # Derived from the host wall samples, so it lives with them:
                # it inherits everything that makes them not device latency.
                payload["harness_diagnostics"]["p50_ms_per_token"] = (
                    measurement.summary.p50_ms / normalized_tokens
                )
            if aa_measurement is not None:
                # A/A calibrated host noise, which is no longer the metric.
                payload["harness_diagnostics"]["aa_calibration"] = (
                    aa_measurement.to_dict()
                )
            if optrace_ref is not None:
                payload["optrace_evidence"] = _jsonable(optrace_ref)
            footprint = self._build_static_footprint(manifest)
            if footprint is not None:
                payload["static_footprint"] = footprint
            report_ref = atomic_publish_json(
                output_dir / f"latency_report{report_suffix}.json",
                payload,
                kind=ArtifactKind.REPORT,
                logical_name=f"latency_report{report_suffix}",
            )
            output_refs = (
                tuple(profile_source_refs)
                + ((optrace_ref,) if optrace_ref is not None else ())
                + (report_ref,)
            )
            return payload, output_refs, {
                "warmup_runs": warmup,
                "measured_runs": repeats,
                "optrace": optrace_enabled,
                "optrace_profile_count": len(profile_entries),
                "optrace_op_count": len(normalized_profile_ops),
                "policy": "report_only",
                "device_identifier": device_identifier,
                "remote_attempt_dir": remote_attempt_dir,
                "remote_cleanup": "confirmed",
            }

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
                return run_one(
                    manifest,
                    adapter,
                    output_dir,
                    selected,
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
                return run_one(
                    manifest,
                    adapter,
                    output_dir,
                    selected,
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
                "latency_metric": "device_execution",
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
    ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
        """Select quality or latency only from verified published evidence."""

        runtime_ref, _runtime_payload = self._manifest_artifact_payload(
            manifest,
            "runtime_index",
            stage="diagnose",
        )
        # Schema/semantic validation is stricter than generic JSON parsing.
        load_runtime_index(runtime_ref.path)

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
            raise InvalidSpecError(
                "automatic diagnose found no provable quality divergence and "
                "no benchmark optrace evidence",
                stage="diagnose",
                details={
                    "sqnr_report_present": bool(sqnr_refs),
                    "hint": (
                        "run benchmark with benchmark.optrace=true or provide "
                        "explicit diagnosis traces"
                    ),
                },
            )

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
            if not selected:
                return self._automatic_diagnosis(manifest, output_dir)
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
            if not selected:
                return self._automatic_diagnosis(manifest, output_dir)
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


__all__ = ["QairtAgent"]
