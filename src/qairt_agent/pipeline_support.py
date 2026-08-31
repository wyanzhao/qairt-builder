"""Shared helpers for the pipeline facade and its stage modules.

Extracted from ``pipeline.py`` so the stage modules can use them without
importing the facade -- which would be a cycle, since the facade imports the
stages. Nothing here changed in the move; ``pipeline`` re-exports every name so
existing imports keep working.
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


#: One definition, in the adapter that owns these objects.
_LIVE_SDK_FIELDS = LIVE_SDK_FIELDS

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

#: Diagnose stage-config keys that still mean "automatic". Anything beyond
#: these switches the stage into explicit-trace mode.
_AUTOMATIC_DIAGNOSE_KEYS = frozenset({"kind", "baseline_manifest"})

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


def run_directory(manifest: Any) -> Path:
    """Where one run's artifacts live beneath its output root."""

    return (
        manifest.build_spec.output_root.expanduser().resolve()
        / "runs"
        / str(manifest.run_id)
    )
