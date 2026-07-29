"""Family-aware AR/CL retargeting for content-verified test vectors.

This module is deliberately independent from the pipeline and CLI layers.  It
turns one existing :class:`~qairt_agent.vectors.VectorManifest` into inputs for
an already-produced target ONNX graph, then captures that graph's outputs with
ONNX Runtime.  It never mutates the source manifest or guesses an unknown graph
input.

The transformation contract is intentionally small and auditable:

* input names come from the target ONNX graph;
* target dtypes and concrete shapes come from the graph ABI;
* recognized dynamic dimensions are resolved from ``AR``/``CL``;
* provided tensors are prefix-cropped and/or zero-padded per axis;
* position ids are rebuilt as a contiguous sequence from the provided first
  position (or zero when the graph input is synthesized);
* missing KV/cache state may be initialized to zero;
* every other missing input, unknown dynamic dimension, or unsafe dtype cast
  fails closed.

Qwen3.5 and Qwen3.5-Omni Thinker production graphs are independent per-AR
exports.  They must use :func:`validate_provided_ar_manifest`; automatic
single-source retargeting is rejected for those families.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import onnx
from onnx import helper

from qairt_agent.vectors import VectorManifest, VectorPreparer, sha256_file


ReferenceSource = Literal["provided_golden", "onnxruntime"]


class VectorRetargetError(ValueError):
    """The vector case cannot be proven compatible with the target graph."""


@dataclass(frozen=True)
class OnnxTensorAbi:
    """One tensor entry in an ONNX graph's public input or output ABI."""

    name: str
    dtype: np.dtype[Any]
    shape: tuple[int | str | None, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype.str,
            "shape": [item for item in self.shape],
        }


@dataclass(frozen=True)
class ValidatedArManifest:
    """Read-only binding of a provided per-AR manifest to one ONNX graph."""

    manifest_path: Path
    manifest_sha256: str
    manifest: VectorManifest
    target_onnx_path: Path
    target_onnx_sha256: str
    family: str
    ar: int
    cl: int
    reference_source: ReferenceSource
    input_abi: tuple[OnnxTensorAbi, ...]
    golden_names: tuple[str, ...]

    @property
    def needs_onnxruntime_capture(self) -> bool:
        """Whether no supplied golden exists and ORT should be used as fallback."""

        return self.reference_source == "onnxruntime"

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": os.fspath(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "target_onnx_path": os.fspath(self.target_onnx_path),
            "target_onnx_sha256": self.target_onnx_sha256,
            "family": self.family,
            "ar": self.ar,
            "cl": self.cl,
            "reference_source": self.reference_source,
            "needs_onnxruntime_capture": self.needs_onnxruntime_capture,
            "input_abi": [item.to_dict() for item in self.input_abi],
            "golden_names": list(self.golden_names),
        }


@dataclass(frozen=True)
class _FamilyPolicy:
    canonical_name: str
    retarget_allowed: bool
    kv_state: bool
    allowed_ars: frozenset[int] | None = None


_FAMILY_ALIASES: Mapping[str, _FamilyPolicy] = {}


def _normalized_family_name(value: Any) -> str:
    raw = str(getattr(value, "value", value)).strip().lower()
    return re.sub(r"[-_.\s]+", "", raw)


def _register_family(
    canonical_name: str,
    *aliases: str,
    retarget_allowed: bool,
    kv_state: bool,
    allowed_ars: frozenset[int] | None = None,
) -> None:
    policy = _FamilyPolicy(
        canonical_name=canonical_name,
        retarget_allowed=retarget_allowed,
        kv_state=kv_state,
        allowed_ars=allowed_ars,
    )
    for alias in (canonical_name, *aliases):
        key = _normalized_family_name(alias)
        if key in _FAMILY_ALIASES and _FAMILY_ALIASES[key] != policy:
            raise RuntimeError(f"duplicate vector-retarget family alias {alias!r}")
        _FAMILY_ALIASES[key] = policy


_register_family(
    "qwen3",
    "qwen3_dense",
    "qwen3-dense",
    "qwen3-4b",
    retarget_allowed=True,
    kv_state=True,
)
_register_family(
    "qwen3-moe",
    "qwen3_moe",
    "qwen3moe",
    retarget_allowed=True,
    kv_state=True,
)
_register_family(
    "qwen3-vl",
    "qwen3_vl",
    "qwen3vl",
    retarget_allowed=True,
    kv_state=True,
)
_register_family(
    "vit",
    "vision-transformer",
    retarget_allowed=True,
    kv_state=False,
    allowed_ars=frozenset({1}),
)
_register_family(
    "qwen3.5",
    "qwen3_5",
    "qwen3-5",
    "qwen35",
    retarget_allowed=False,
    kv_state=True,
    allowed_ars=frozenset({1, 128}),
)
_register_family(
    "qwen3.5-omni-thinker",
    "qwen3_5_omni_thinker",
    "qwen3-5-omni-thinker",
    "qwen35-omni-thinker",
    retarget_allowed=False,
    kv_state=True,
    allowed_ars=frozenset({1, 128}),
)
_register_family(
    "qwen3.5-omni",
    "qwen3_5_omni",
    "qwen3-5-omni",
    "qwen35-omni",
    retarget_allowed=False,
    kv_state=True,
    allowed_ars=frozenset({1, 128}),
)


def _family_policy(family: Any, ar: int, cl: int) -> _FamilyPolicy:
    if ar <= 0 or cl <= 0:
        raise VectorRetargetError("AR and CL must be positive integers")
    if ar > cl:
        raise VectorRetargetError(f"AR={ar} cannot exceed CL={cl}")
    key = _normalized_family_name(family)
    policy = _FAMILY_ALIASES.get(key)
    if policy is None:
        supported = sorted({item.canonical_name for item in _FAMILY_ALIASES.values()})
        raise VectorRetargetError(
            f"unsupported vector-retarget family {family!r}; expected one of {supported}"
        )
    if policy.allowed_ars is not None and ar not in policy.allowed_ars:
        raise VectorRetargetError(
            f"{policy.canonical_name} requires AR in {sorted(policy.allowed_ars)}, got {ar}"
        )
    return policy


def _path_from_value(value: Any) -> Path:
    if isinstance(value, (str, os.PathLike)):
        return Path(value)
    if hasattr(value, "path"):
        return Path(getattr(value, "path"))
    raise TypeError(f"expected a path or artifact-like object, got {type(value).__name__}")


def _tensor_abi(value_info: Any) -> OnnxTensorAbi:
    value_type = value_info.type
    if not value_type.HasField("tensor_type"):
        raise VectorRetargetError(
            f"ONNX value {value_info.name!r} is not a tensor; sequence/map/optional inputs "
            "are not supported"
        )
    tensor_type = value_type.tensor_type
    if not tensor_type.HasField("shape"):
        raise VectorRetargetError(f"ONNX tensor {value_info.name!r} has no declared rank")
    try:
        dtype = np.dtype(helper.tensor_dtype_to_np_dtype(tensor_type.elem_type))
    except Exception as exc:
        raise VectorRetargetError(
            f"ONNX tensor {value_info.name!r} has unsupported element type "
            f"{tensor_type.elem_type}"
        ) from exc
    if dtype.hasobject or dtype.kind in {"O", "S", "U", "V", "c"}:
        raise VectorRetargetError(
            f"ONNX tensor {value_info.name!r} has unsupported dtype {dtype}"
        )

    dimensions: list[int | str | None] = []
    for axis, dimension in enumerate(tensor_type.shape.dim):
        if dimension.HasField("dim_value"):
            size = int(dimension.dim_value)
            if size <= 0:
                raise VectorRetargetError(
                    f"ONNX tensor {value_info.name!r} axis {axis} has non-positive "
                    f"static size {size}"
                )
            dimensions.append(size)
        elif dimension.HasField("dim_param") and dimension.dim_param:
            dimensions.append(str(dimension.dim_param))
        else:
            dimensions.append(None)
    return OnnxTensorAbi(
        name=str(value_info.name),
        dtype=dtype,
        shape=tuple(dimensions),
    )


def inspect_onnx_abi(
    model_path: str | os.PathLike[str],
) -> tuple[tuple[OnnxTensorAbi, ...], tuple[OnnxTensorAbi, ...]]:
    """Return public tensor inputs and outputs from an ONNX graph.

    Initializers that old exporters also list in ``graph.input`` are excluded
    because callers must not provide them at runtime.
    """

    path = Path(model_path)
    try:
        model = onnx.load_model(path, load_external_data=False)
    except Exception as exc:
        raise VectorRetargetError(f"cannot load target ONNX {path}: {exc}") from exc

    initializer_names = {item.name for item in model.graph.initializer}
    initializer_names.update(item.values.name for item in model.graph.sparse_initializer)
    inputs = tuple(
        _tensor_abi(item)
        for item in model.graph.input
        if item.name not in initializer_names
    )
    outputs = tuple(_tensor_abi(item) for item in model.graph.output)
    if not inputs:
        raise VectorRetargetError(f"target ONNX {path} has no public tensor inputs")
    if not outputs:
        raise VectorRetargetError(f"target ONNX {path} has no public tensor outputs")
    input_names = [item.name for item in inputs]
    output_names = [item.name for item in outputs]
    if len(set(input_names)) != len(input_names):
        raise VectorRetargetError("target ONNX contains duplicate public input names")
    if len(set(output_names)) != len(output_names):
        raise VectorRetargetError("target ONNX contains duplicate public output names")
    return inputs, outputs


def _semantic_role(name: str) -> str:
    normalized = name.lower()
    if (
        any(token in normalized for token in ("position", "rotary", "rope"))
        and any(
            token in normalized
            for token in (
                "_cos",
                "_sin",
                "cos_",
                "sin_",
            )
        )
    ):
        # Qualcomm-style Qwen exports commonly expose precomputed floating
        # RoPE tables as position_ids_cos/position_ids_sin with shape
        # [B, 1, AR, head_dim].  They are not integer position IDs.
        return "rotary_position"
    if "position_id" in normalized or "pos_id" in normalized:
        return "position_ids"
    if normalized in {"input_ids", "token_ids", "tokens"} or normalized.endswith(
        ("_input_ids", "_token_ids")
    ):
        return "tokens"
    if "attention_mask" in normalized or "attn_mask" in normalized or "causal_mask" in normalized:
        return "attention_mask"
    cache_markers = (
        "past_key",
        "past_value",
        "key_cache",
        "value_cache",
        "kv_cache",
        "k_cache",
        "v_cache",
        "recurrent_state",
        "conv_state",
    )
    if any(marker in normalized for marker in cache_markers):
        return "kv_cache"
    if "input_embed" in normalized or "hidden_state" in normalized:
        return "sequence_features"
    if (
        "pixel" in normalized
        or "image" in normalized
        or "vision" in normalized
        or normalized.startswith("vit_")
    ):
        return "vision"
    return "unknown"


def _sequence_axis(role: str, rank: int) -> int | None:
    if rank == 0:
        return None
    if role in {"tokens", "position_ids"}:
        return rank - 1
    if role == "sequence_features":
        return rank - 2 if rank >= 3 else rank - 1
    if role == "rotary_position":
        return rank - 2 if rank >= 3 else rank - 1
    if role == "attention_mask" and rank >= 3:
        return rank - 2
    return None


def _symbol_kind(symbol: str | None) -> str | None:
    if symbol is None:
        return None
    value = re.sub(r"[^a-z0-9]+", "", symbol.lower())
    if any(token in value for token in ("batch", "batchsize")) or value in {"b", "n"}:
        return "batch"
    if any(token in value for token in ("context", "cache", "past", "kvlen", "keylen")):
        return "cl"
    if any(token in value for token in ("sequence", "seqlen", "token", "querylen")):
        return "ar"
    if value in {"ar", "seq", "s", "q", "qlen"}:
        return "ar"
    if value in {"cl", "ctx", "c", "k", "klen"}:
        return "cl"
    return None


def _resolve_target_shape(
    abi: OnnxTensorAbi,
    *,
    source_shape: tuple[int, ...] | None,
    role: str,
    ar: int,
    cl: int,
) -> tuple[tuple[int, ...], tuple[dict[str, Any], ...]]:
    if source_shape is not None and len(source_shape) != len(abi.shape):
        raise VectorRetargetError(
            f"input {abi.name!r} rank mismatch: source={source_shape}, "
            f"target_rank={len(abi.shape)}"
        )
    sequence_axis = _sequence_axis(role, len(abi.shape))
    resolved: list[int] = []
    evidence: list[dict[str, Any]] = []
    for axis, dimension in enumerate(abi.shape):
        if isinstance(dimension, int):
            size = dimension
            source = "static_onnx"
        else:
            symbol_kind = _symbol_kind(dimension)
            if symbol_kind == "ar":
                size = ar
                source = "symbolic_ar"
            elif symbol_kind == "cl":
                size = cl
                source = "symbolic_cl"
            elif symbol_kind == "batch":
                size = source_shape[axis] if source_shape is not None else 1
                source = "source_batch" if source_shape is not None else "default_batch_1"
            elif axis == sequence_axis and role in {
                "tokens",
                "position_ids",
                "sequence_features",
            }:
                size = ar
                source = "semantic_ar"
            elif axis == 0 and source_shape is not None:
                size = source_shape[axis]
                source = "source_batch"
            else:
                rendered = dimension if dimension is not None else "<anonymous>"
                raise VectorRetargetError(
                    f"input {abi.name!r} axis {axis} has unresolved dynamic dimension "
                    f"{rendered!r}; name it as batch/sequence/context in the target ONNX"
                )
        if size <= 0:
            raise VectorRetargetError(
                f"input {abi.name!r} axis {axis} resolved to non-positive size {size}"
            )
        resolved.append(int(size))
        evidence.append(
            {
                "axis": axis,
                "onnx_dimension": dimension,
                "resolved": int(size),
                "resolution": source,
            }
        )

    # Token and position inputs are the strongest available proof that this is
    # the requested AR graph.  A static ONNX that disagrees must never be
    # silently cropped to a different requested AR.
    if sequence_axis is not None and role in {
        "tokens",
        "position_ids",
        "rotary_position",
        "sequence_features",
    }:
        if resolved[sequence_axis] != ar:
            raise VectorRetargetError(
                f"target ONNX input {abi.name!r} proves AR={resolved[sequence_axis]}, "
                f"but AR={ar} was requested"
            )
    return tuple(resolved), tuple(evidence)


def _safe_cast(
    name: str,
    value: np.ndarray,
    target_dtype: np.dtype[Any],
) -> tuple[np.ndarray, str]:
    source_dtype = np.dtype(value.dtype)
    if source_dtype == target_dtype:
        return np.ascontiguousarray(value), "identity"
    if not np.can_cast(source_dtype, target_dtype, casting="safe"):
        raise VectorRetargetError(
            f"input {name!r} cannot be safely cast from {source_dtype} to {target_dtype}"
        )
    return np.ascontiguousarray(value.astype(target_dtype, copy=False)), "safe_cast"


def _prefix_crop_zero_pad(
    value: np.ndarray,
    target_shape: tuple[int, ...],
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    source_shape = tuple(int(item) for item in value.shape)
    destination = np.zeros(target_shape, dtype=value.dtype)
    overlap = tuple(min(source, target) for source, target in zip(source_shape, target_shape))
    slices = tuple(slice(0, size) for size in overlap)
    destination[slices] = value[slices]
    transforms: list[dict[str, Any]] = []
    for axis, (source, target) in enumerate(zip(source_shape, target_shape)):
        operation = "unchanged"
        if source > target:
            operation = "prefix_crop"
        elif source < target:
            operation = "zero_pad"
        transforms.append(
            {
                "axis": axis,
                "source_size": source,
                "target_size": target,
                "operation": operation,
            }
        )
    return np.ascontiguousarray(destination), tuple(transforms)


def _contiguous_position_ids(
    target_shape: tuple[int, ...],
    dtype: np.dtype[Any],
    *,
    start: int,
) -> tuple[np.ndarray, int]:
    if dtype.kind not in {"i", "u"}:
        raise VectorRetargetError(f"position_ids target dtype must be integral, got {dtype}")
    axis = len(target_shape) - 1
    if axis < 0:
        raise VectorRetargetError("position_ids must have rank >= 1")
    maximum = start + target_shape[axis] - 1
    limits = np.iinfo(dtype)
    if start < limits.min or maximum > limits.max:
        raise VectorRetargetError(
            f"position_ids range [{start}, {maximum}] does not fit target dtype {dtype}"
        )
    sequence = np.arange(start, start + target_shape[axis], dtype=dtype)
    reshape = [1] * len(target_shape)
    reshape[axis] = target_shape[axis]
    return np.ascontiguousarray(np.broadcast_to(sequence.reshape(reshape), target_shape)), axis


def _validate_manifest_metadata(
    manifest: VectorManifest,
    policy: _FamilyPolicy,
    *,
    ar: int,
    cl: int,
) -> None:
    metadata = manifest.metadata
    if "family" in metadata:
        declared = _family_policy(metadata["family"], ar, cl)
        if declared.canonical_name != policy.canonical_name:
            raise VectorRetargetError(
                f"manifest family {metadata['family']!r} does not match requested "
                f"{policy.canonical_name!r}"
            )
    for key, expected in (("ar", ar), ("cl", cl), ("context_length", cl)):
        if key in metadata and int(metadata[key]) != expected:
            raise VectorRetargetError(
                f"manifest metadata {key}={metadata[key]!r} does not match requested {expected}"
            )


def _validate_golden_abi(
    manifest_path: Path,
    manifest: VectorManifest,
    output_abi: Sequence[OnnxTensorAbi],
    *,
    ar: int,
    cl: int,
) -> tuple[str, ...]:
    if not manifest.goldens:
        return ()
    outputs_by_name = {item.name: item for item in output_abi}
    unknown = sorted(set(manifest.goldens) - set(outputs_by_name))
    if unknown:
        raise VectorRetargetError(
            f"provided golden tensors are not target ONNX outputs: {unknown}"
        )
    values = VectorPreparer.load_tensors(manifest_path, section="goldens")
    for name, value in values.items():
        abi = outputs_by_name[name]
        if np.dtype(value.dtype) != abi.dtype:
            raise VectorRetargetError(
                f"golden {name!r} dtype mismatch: manifest={value.dtype}, target={abi.dtype}"
            )
        if value.ndim != len(abi.shape):
            raise VectorRetargetError(
                f"golden {name!r} rank mismatch: manifest={value.ndim}, "
                f"target={len(abi.shape)}"
            )
        for axis, (actual, declared) in enumerate(zip(value.shape, abi.shape)):
            if isinstance(declared, int) and actual != declared:
                raise VectorRetargetError(
                    f"golden {name!r} axis {axis} mismatch: manifest={actual}, "
                    f"target={declared}"
                )
            kind = _symbol_kind(declared if isinstance(declared, str) else None)
            if kind == "ar" and actual != ar:
                raise VectorRetargetError(
                    f"golden {name!r} axis {axis} proves AR={actual}, expected {ar}"
                )
            if kind == "cl" and actual != cl:
                raise VectorRetargetError(
                    f"golden {name!r} axis {axis} proves CL={actual}, expected {cl}"
                )
    return tuple(sorted(values))


def validate_provided_ar_manifest(
    manifest_path: str | os.PathLike[str] | Any,
    target_onnx_path: str | os.PathLike[str],
    *,
    family: Any,
    ar: int,
    cl: int,
    expected_manifest_sha256: str | None = None,
    allow_extra_inputs: bool = False,
) -> ValidatedArManifest:
    """Validate and bind an immutable, already-per-AR vector manifest.

    No tensor is rewritten.  Supplied goldens win when present; otherwise the
    returned binding explicitly selects ``onnxruntime`` as the required
    fallback reference source.
    """

    policy = _family_policy(family, ar, cl)
    source_path = _path_from_value(manifest_path).resolve()
    target_path = Path(target_onnx_path).resolve()
    manifest = VectorPreparer.load_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
    )
    _validate_manifest_metadata(manifest, policy, ar=ar, cl=cl)
    inputs_abi, outputs_abi = inspect_onnx_abi(target_path)
    values = VectorPreparer.load_tensors(
        source_path,
        section="inputs",
        expected_manifest_sha256=expected_manifest_sha256,
    )

    target_names = {item.name for item in inputs_abi}
    provided_names = set(values)
    missing = sorted(target_names - provided_names)
    if missing:
        raise VectorRetargetError(
            f"provided per-AR manifest is missing target ONNX inputs: {missing}"
        )
    extra = sorted(provided_names - target_names)
    if extra and not allow_extra_inputs:
        raise VectorRetargetError(
            f"provided per-AR manifest has inputs absent from target ONNX: {extra}"
        )

    for abi in inputs_abi:
        value = values[abi.name]
        role = _semantic_role(abi.name)
        target_shape, _ = _resolve_target_shape(
            abi,
            source_shape=tuple(int(item) for item in value.shape),
            role=role,
            ar=ar,
            cl=cl,
        )
        if tuple(value.shape) != target_shape:
            raise VectorRetargetError(
                f"provided per-AR input {abi.name!r} shape mismatch: "
                f"manifest={tuple(value.shape)}, target={target_shape}"
            )
        if np.dtype(value.dtype) != abi.dtype:
            raise VectorRetargetError(
                f"provided per-AR input {abi.name!r} dtype mismatch: "
                f"manifest={value.dtype}, target={abi.dtype}"
            )

    golden_names = _validate_golden_abi(
        source_path,
        manifest,
        outputs_abi,
        ar=ar,
        cl=cl,
    )
    return ValidatedArManifest(
        manifest_path=source_path,
        manifest_sha256=sha256_file(source_path),
        manifest=manifest,
        target_onnx_path=target_path,
        target_onnx_sha256=sha256_file(target_path),
        family=policy.canonical_name,
        ar=ar,
        cl=cl,
        reference_source="provided_golden" if golden_names else "onnxruntime",
        input_abi=inputs_abi,
        golden_names=golden_names,
    )


def retarget_vector_manifest(
    source_manifest_path: str | os.PathLike[str] | Any,
    target_onnx_path: str | os.PathLike[str],
    *,
    family: Any,
    ar: int,
    cl: int,
    output_dir: str | os.PathLike[str],
    expected_source_manifest_sha256: str | None = None,
    providers: Sequence[str] = ("CPUExecutionProvider",),
    preserve_extra_inputs: bool = False,
) -> Path:
    """Retarget one source vector case and capture all target ONNX outputs.

    The returned path is a new immutable :class:`VectorManifest`.  Its inputs
    have the exact target ONNX ABI and its goldens are always produced by ONNX
    Runtime.  The original source goldens are deliberately not copied because
    an AR/CL shape change invalidates them.
    """

    policy = _family_policy(family, ar, cl)
    if not policy.retarget_allowed:
        raise VectorRetargetError(
            f"{policy.canonical_name} requires independent per-AR ONNX/encodings/vectors; "
            "use validate_provided_ar_manifest instead of single-source retargeting"
        )

    source_path = _path_from_value(source_manifest_path).resolve()
    target_path = Path(target_onnx_path).resolve()
    source = VectorPreparer.load_manifest(
        source_manifest_path,
        expected_sha256=expected_source_manifest_sha256,
    )
    source_manifest_sha256 = sha256_file(source_path)
    target_onnx_sha256 = sha256_file(target_path)
    source_inputs = VectorPreparer.load_tensors(
        source_path,
        section="inputs",
        expected_manifest_sha256=expected_source_manifest_sha256,
    )
    inputs_abi, outputs_abi = inspect_onnx_abi(target_path)

    transformed: dict[str, np.ndarray] = {}
    roles: dict[str, str] = {}
    audit: dict[str, dict[str, Any]] = {}
    for abi in inputs_abi:
        role = _semantic_role(abi.name)
        roles[abi.name] = role
        source_value = source_inputs.get(abi.name)
        target_shape, shape_resolution = _resolve_target_shape(
            abi,
            source_shape=(
                tuple(int(item) for item in source_value.shape)
                if source_value is not None
                else None
            ),
            role=role,
            ar=ar,
            cl=cl,
        )

        if source_value is None:
            if role == "position_ids":
                value, position_axis = _contiguous_position_ids(
                    target_shape,
                    abi.dtype,
                    start=0,
                )
                transformed[abi.name] = value
                audit[abi.name] = {
                    "semantic_role": role,
                    "source": "synthesized",
                    "source_shape": None,
                    "target_shape": list(target_shape),
                    "source_dtype": None,
                    "target_dtype": abi.dtype.str,
                    "dtype_transform": "synthesized_exact",
                    "shape_resolution": list(shape_resolution),
                    "axis_transforms": [],
                    "value_transform": "contiguous_position_ids",
                    "position_axis": position_axis,
                    "position_start": 0,
                }
                continue
            if role == "kv_cache" and policy.kv_state:
                transformed[abi.name] = np.zeros(target_shape, dtype=abi.dtype)
                audit[abi.name] = {
                    "semantic_role": role,
                    "source": "synthesized",
                    "source_shape": None,
                    "target_shape": list(target_shape),
                    "source_dtype": None,
                    "target_dtype": abi.dtype.str,
                    "dtype_transform": "synthesized_exact",
                    "shape_resolution": list(shape_resolution),
                    "axis_transforms": [],
                    "value_transform": "zero_initialized_state",
                }
                continue
            raise VectorRetargetError(
                f"target ONNX input {abi.name!r} is missing from the source manifest "
                f"and has no safe family-aware synthesis rule (role={role})"
            )

        cast_value, dtype_transform = _safe_cast(abi.name, source_value, abi.dtype)
        source_shape = tuple(int(item) for item in source_value.shape)
        if role == "position_ids":
            if source_value.size == 0 or source_value.dtype.kind not in {"i", "u"}:
                raise VectorRetargetError(
                    f"source position_ids {abi.name!r} must be a non-empty integral tensor"
                )
            start = int(source_value.reshape(-1)[0])
            value, position_axis = _contiguous_position_ids(
                target_shape,
                abi.dtype,
                start=start,
            )
            axis_transforms = tuple(
                {
                    "axis": axis,
                    "source_size": source_size,
                    "target_size": target_size,
                    "operation": (
                        "prefix_crop"
                        if source_size > target_size
                        else "zero_pad"
                        if source_size < target_size
                        else "unchanged"
                    ),
                }
                for axis, (source_size, target_size) in enumerate(
                    zip(source_shape, target_shape)
                )
            )
            value_transform = "contiguous_position_ids"
        else:
            value, axis_transforms = _prefix_crop_zero_pad(cast_value, target_shape)
            position_axis = None
            start = None
            if role == "rotary_position":
                sequence_axis = _sequence_axis(role, len(target_shape))
                assert sequence_axis is not None
                if source_shape[sequence_axis] < target_shape[sequence_axis]:
                    raise VectorRetargetError(
                        f"rotary position tensor {abi.name!r} cannot be extended "
                        "with zero padding; provide a source vector covering the "
                        f"requested AR={ar}"
                    )
                value_transform = "prefix_crop_rotary_table"
            else:
                value_transform = "prefix_crop_zero_pad"
        transformed[abi.name] = value
        record: dict[str, Any] = {
            "semantic_role": role,
            "source": "provided",
            "source_shape": list(source_shape),
            "target_shape": list(target_shape),
            "source_dtype": np.dtype(source_value.dtype).str,
            "target_dtype": abi.dtype.str,
            "dtype_transform": dtype_transform,
            "shape_resolution": list(shape_resolution),
            "axis_transforms": list(axis_transforms),
            "value_transform": value_transform,
        }
        if position_axis is not None:
            record["position_axis"] = position_axis
            record["position_start"] = start
        audit[abi.name] = record

    extra_input_names = sorted(
        set(source_inputs) - {item.name for item in inputs_abi}
    )
    if preserve_extra_inputs:
        for name in extra_input_names:
            transformed[name] = np.ascontiguousarray(source_inputs[name])
            roles[name] = _semantic_role(name)
    dropped = [] if preserve_extra_inputs else extra_input_names

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(os.fspath(target_path), providers=list(providers))
    except Exception as exc:
        raise VectorRetargetError(
            f"ONNX Runtime could not load target graph {target_path}: {exc}"
        ) from exc
    runtime_inputs = {item.name for item in session.get_inputs()}
    expected_inputs = {item.name for item in inputs_abi}
    if runtime_inputs != expected_inputs:
        raise VectorRetargetError(
            "ONNX Runtime input ABI differs from serialized graph ABI: "
            f"runtime={sorted(runtime_inputs)}, graph={sorted(expected_inputs)}"
        )
    output_names = tuple(item.name for item in outputs_abi)
    try:
        output_values = session.run(
            list(output_names),
            {
                name: transformed[name]
                for name in sorted(expected_inputs)
            },
        )
    except Exception as exc:
        raise VectorRetargetError(
            f"ONNX Runtime golden capture failed for {target_path}: {exc}"
        ) from exc
    if len(output_values) != len(output_names):
        raise VectorRetargetError(
            f"ONNX Runtime returned {len(output_values)} outputs for "
            f"{len(output_names)} requested names"
        )
    goldens = dict(zip(output_names, output_values))

    metadata = copy.deepcopy(dict(source.metadata))
    metadata.update(
        {
            "family": policy.canonical_name,
            "ar": ar,
            "cl": cl,
            "context_length": cl,
            "reference_source": "onnxruntime",
            "reference_priority": "onnxruntime_fallback",
            "source_vector_manifest": {
                "path": os.fspath(source_path),
                "sha256": source_manifest_sha256,
                "case_id": source.case_id,
                "supplied_goldens_ignored": sorted(source.goldens),
            },
            "target_onnx": {
                "path": os.fspath(target_path),
                "sha256": target_onnx_sha256,
            },
            "retarget": {
                "policy": "family_prefix_crop_zero_pad_v1",
                "input_transforms": audit,
                "dropped_source_inputs": dropped,
                "preserved_extra_inputs": (
                    extra_input_names if preserve_extra_inputs else []
                ),
            },
            "onnxruntime": {
                "requested_providers": list(providers),
                "active_providers": list(session.get_providers()),
                "outputs": list(output_names),
            },
        }
    )
    case_id = (
        f"{source.case_id}/{policy.canonical_name}/"
        f"ar{ar}-cl{cl}-{source_manifest_sha256[:12]}-{target_onnx_sha256[:12]}"
    )
    preparer = VectorPreparer(output_dir)
    return preparer.prepare_case(
        case_id,
        transformed,
        goldens=goldens,
        roles=roles,
        metadata=metadata,
        manifest_name="vector_manifest.onnx-reference.json",
    )


__all__ = [
    "OnnxTensorAbi",
    "ReferenceSource",
    "ValidatedArManifest",
    "VectorRetargetError",
    "inspect_onnx_abi",
    "retarget_vector_manifest",
    "validate_provided_ar_manifest",
]
