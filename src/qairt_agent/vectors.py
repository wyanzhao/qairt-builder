"""Explicit, content-verified test-vector artifacts.

The vector layer deliberately has no dependency on the orchestration contracts.
It can therefore be imported on hosts that do not have QAIRT installed, and it
can consume paths or contract-like objects through their public attributes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

VECTOR_MANIFEST_SCHEMA = "qairt-agent.vector-manifest"


def _path_from_value(value: Any) -> Path:
    if isinstance(value, (str, os.PathLike)):
        return Path(value)
    if hasattr(value, "path"):
        return Path(getattr(value, "path"))
    raise TypeError(f"Expected a path or artifact-like object, got {type(value).__name__}")


def sha256_file(path: str | os.PathLike[str] | Any) -> str:
    """Return the SHA256 of *path* without loading the whole file in memory."""

    digest = hashlib.sha256()
    with _path_from_value(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "tensor"
    suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"{stem[:72]}-{suffix}"


def _canonical_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Object arrays cannot be stored as QAIRT test vectors")
    if array.dtype.kind in {"U", "S", "V"}:
        raise TypeError(f"Unsupported tensor dtype: {array.dtype}")

    byte_order = array.dtype.byteorder
    if byte_order == ">" or (byte_order == "=" and sys.byteorder == "big"):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    elif byte_order == "=" and array.dtype.itemsize > 1:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class TensorSource:
    """Description of an existing ``.raw`` or ``.npy`` tensor."""

    path: Path
    dtype: str | np.dtype[Any] | None = None
    shape: tuple[int, ...] | None = None
    role: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> "TensorSource":
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, os.PathLike)):
            return cls(Path(value))
        if hasattr(value, "path"):
            raw_shape = getattr(value, "shape", None)
            return cls(
                path=Path(getattr(value, "path")),
                dtype=getattr(value, "dtype", None),
                shape=tuple(int(item) for item in raw_shape) if raw_shape is not None else None,
                role=getattr(value, "role", None),
            )
        if isinstance(value, Mapping) and "path" in value:
            raw_shape = value.get("shape")
            return cls(
                path=Path(value["path"]),
                dtype=value.get("dtype"),
                shape=tuple(int(item) for item in raw_shape) if raw_shape is not None else None,
                role=value.get("role"),
            )
        raise TypeError(f"Cannot interpret {type(value).__name__} as a tensor source")


@dataclass(frozen=True)
class TensorRecord:
    """A tensor file plus enough metadata to read and verify it."""

    name: str
    path: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    nbytes: int
    storage: str = "raw"
    byte_order: str = "little"
    role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "path": self.path,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "sha256": self.sha256,
            "nbytes": self.nbytes,
            "storage": self.storage,
            "byte_order": self.byte_order,
        }
        if self.role is not None:
            data["role"] = self.role
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorRecord":
        return cls(
            name=str(data["name"]),
            path=str(data["path"]),
            dtype=str(data["dtype"]),
            shape=tuple(int(item) for item in data["shape"]),
            sha256=str(data["sha256"]),
            nbytes=int(data["nbytes"]),
            storage=str(data.get("storage", "raw")),
            byte_order=str(data.get("byte_order", "little")),
            role=str(data["role"]) if data.get("role") is not None else None,
        )


@dataclass(frozen=True)
class VectorManifest:
    """Manifest for one fully explicit vector case."""

    case_id: str
    inputs: Mapping[str, TensorRecord]
    goldens: Mapping[str, TensorRecord] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = VECTOR_MANIFEST_SCHEMA
    source_manifest_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema": self.schema,
            "case_id": self.case_id,
            "inputs": {name: record.to_dict() for name, record in sorted(self.inputs.items())},
            "goldens": {name: record.to_dict() for name, record in sorted(self.goldens.items())},
            "metadata": copy.deepcopy(dict(self.metadata)),
        }
        if self.source_manifest_sha256 is not None:
            data["source_manifest_sha256"] = self.source_manifest_sha256
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VectorManifest":
        schema = str(data.get("schema", ""))
        if schema != VECTOR_MANIFEST_SCHEMA:
            raise ValueError(
                f"Unsupported vector manifest schema {schema!r}; expected {VECTOR_MANIFEST_SCHEMA!r}"
            )
        return cls(
            case_id=str(data["case_id"]),
            inputs={name: TensorRecord.from_dict(record) for name, record in data["inputs"].items()},
            goldens={
                name: TensorRecord.from_dict(record)
                for name, record in data.get("goldens", {}).items()
            },
            metadata=copy.deepcopy(dict(data.get("metadata", {}))),
            schema=schema,
            source_manifest_sha256=(
                str(data["source_manifest_sha256"])
                if data.get("source_manifest_sha256") is not None
                else None
            ),
        )


def _external_data_paths(model: Any, model_path: Path) -> tuple[Path, ...]:
    """Return the external initializer files a loaded model proto references."""

    resolved: set[Path] = set()
    for tensor in model.graph.initializer:
        metadata = {
            str(item.key): str(item.value)
            for item in getattr(tensor, "external_data", ())
        }
        location = metadata.get("location")
        if not location:
            continue
        external = Path(location)
        if not external.is_absolute():
            external = model_path.parent / external
        resolved.add(external.resolve())
    return tuple(sorted(resolved))


@contextmanager
def _instrumented_model_source(
    model: Any,
    model_path: Path,
    external_data: Sequence[str],
) -> Iterator[Any]:
    """Yield something ``InferenceSession`` accepts for an instrumented model.

    A self-contained model is handed over as bytes.  A model with external data
    must be materialized *beside the original* so its relative initializer
    locations still resolve; that temporary file is always removed. If the
    model's directory is not writable the mode fails closed rather than
    silently running a graph with missing weights.
    """

    payload = model.SerializeToString()
    if not external_data:
        yield payload
        return

    try:
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{model_path.stem}.float-reference.",
            suffix=".onnx",
            dir=model_path.parent,
        )
    except OSError as error:
        raise PermissionError(
            "the float reference model uses external data, so an instrumented "
            "copy must be written next to it, but "
            f"{model_path.parent} is not writable: {error}"
        ) from error
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        yield os.fspath(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)


class VectorPreparer:
    """Prepare, import, capture, and verify explicit test-vector artifacts."""

    def __init__(self, output_dir: str | os.PathLike[str]) -> None:
        self.output_dir = Path(output_dir)

    @staticmethod
    def manifest_sha256(path: str | os.PathLike[str]) -> str:
        return sha256_file(path)

    @staticmethod
    def load_manifest(
        path: str | os.PathLike[str] | Any, *, expected_sha256: str | None = None
    ) -> VectorManifest:
        manifest_path = _path_from_value(path)
        if expected_sha256 is None and hasattr(path, "sha256"):
            expected_sha256 = str(getattr(path, "sha256"))
        if expected_sha256 is not None:
            actual_sha256 = sha256_file(manifest_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Vector manifest SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
                )
        with manifest_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        return VectorManifest.from_dict(data)

    @staticmethod
    def _resolve_tensor_path(manifest_path: Path, record: TensorRecord) -> Path:
        tensor_path = Path(record.path)
        if not tensor_path.is_absolute():
            tensor_path = manifest_path.parent / tensor_path
        return tensor_path.resolve()

    @classmethod
    def load_tensors(
        cls,
        manifest_path: str | os.PathLike[str],
        *,
        section: str = "inputs",
        expected_manifest_sha256: str | None = None,
        verify: bool = True,
    ) -> dict[str, np.ndarray]:
        """Load one manifest section, checking content hashes and tensor sizes."""

        path = _path_from_value(manifest_path)
        manifest = cls.load_manifest(path, expected_sha256=expected_manifest_sha256)
        if section not in {"inputs", "goldens"}:
            raise ValueError("section must be 'inputs' or 'goldens'")
        records = manifest.inputs if section == "inputs" else manifest.goldens

        tensors: dict[str, np.ndarray] = {}
        for name, record in records.items():
            tensor_path = cls._resolve_tensor_path(path, record)
            if verify:
                actual_sha256 = sha256_file(tensor_path)
                if actual_sha256 != record.sha256:
                    raise ValueError(
                        f"Tensor {name!r} SHA256 mismatch: expected {record.sha256}, got {actual_sha256}"
                    )

            dtype = np.dtype(record.dtype)
            expected_items = int(np.prod(record.shape, dtype=np.int64))
            if record.storage == "raw":
                tensor = np.fromfile(tensor_path, dtype=dtype)
            elif record.storage == "npy":
                tensor = np.load(tensor_path, allow_pickle=False)
                if tensor.dtype != dtype:
                    raise ValueError(
                        f"Tensor {name!r} dtype mismatch: manifest={dtype}, file={tensor.dtype}"
                    )
            else:
                raise ValueError(f"Unsupported tensor storage format {record.storage!r}")

            if tensor.size != expected_items:
                raise ValueError(
                    f"Tensor {name!r} size mismatch: expected {expected_items} items, got {tensor.size}"
                )
            tensor = np.asarray(tensor).reshape(record.shape)
            if tensor.nbytes != record.nbytes:
                raise ValueError(
                    f"Tensor {name!r} byte-size mismatch: expected {record.nbytes}, got {tensor.nbytes}"
                )
            tensors[name] = tensor
        return tensors

    def _source_to_array(self, value: Any) -> tuple[np.ndarray, str | None]:
        source_like = isinstance(value, (TensorSource, str, os.PathLike, Mapping)) or hasattr(
            value, "path"
        )
        if isinstance(value, np.ndarray) or not source_like:
            return _canonical_array(value), None

        source = TensorSource.from_value(value)
        path = source.path
        if path.suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            if source.dtype is not None and array.dtype != np.dtype(source.dtype):
                raise ValueError(
                    f"Tensor source {path} dtype mismatch: "
                    f"expected {np.dtype(source.dtype)}, got {array.dtype}"
                )
            if source.shape is not None and tuple(array.shape) != source.shape:
                raise ValueError(
                    f"Tensor source {path} shape mismatch: expected {source.shape}, got {tuple(array.shape)}"
                )
            return _canonical_array(array), source.role

        if source.dtype is None or source.shape is None:
            raise ValueError(f"Raw tensor source {path} requires explicit dtype and shape")
        dtype = np.dtype(source.dtype)
        array = np.fromfile(path, dtype=dtype)
        expected_items = int(np.prod(source.shape, dtype=np.int64))
        if array.size != expected_items:
            raise ValueError(
                f"Raw tensor source {path} has {array.size} items; expected {expected_items}"
            )
        return _canonical_array(array.reshape(source.shape)), source.role

    @staticmethod
    def _store_tensor(
        case_dir: Path,
        section: str,
        name: str,
        value: Any,
        *,
        role: str | None = None,
    ) -> TensorRecord:
        array = _canonical_array(value)
        raw_bytes = array.tobytes(order="C")
        content_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        section_dir = case_dir / section
        section_dir.mkdir(parents=True, exist_ok=True)
        destination = section_dir / f"{_safe_name(name)}-{content_sha256[:12]}.raw"

        if destination.exists():
            if sha256_file(destination) != content_sha256:
                raise FileExistsError(f"Refusing to replace conflicting tensor artifact {destination}")
        else:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(raw_bytes)
            os.replace(temporary, destination)

        return TensorRecord(
            name=name,
            path=destination.relative_to(case_dir).as_posix(),
            dtype=array.dtype.str,
            shape=tuple(int(item) for item in array.shape),
            sha256=content_sha256,
            nbytes=array.nbytes,
            storage="raw",
            byte_order="not-applicable" if array.dtype.itemsize == 1 else "little",
            role=role,
        )

    @staticmethod
    def _write_manifest(path: Path, manifest: VectorManifest) -> Path:
        payload = _json_bytes(manifest.to_dict())
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() == payload:
                return path
            raise FileExistsError(f"Refusing to replace immutable vector manifest {path}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
        return path

    def prepare_case(
        self,
        case_id: str,
        inputs: Mapping[str, Any],
        *,
        goldens: Mapping[str, Any] | None = None,
        roles: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        manifest_name: str = "vector_manifest.json",
    ) -> Path:
        """Materialize one vector case and return its immutable manifest path."""

        if not case_id or (not inputs and not goldens):
            raise ValueError(
                "case_id and at least one input or golden tensor are required"
            )
        # Validate metadata before writing any artifacts.
        json.dumps(dict(metadata or {}), allow_nan=False)

        case_dir = self.output_dir / "cases" / _safe_name(case_id)
        input_records: dict[str, TensorRecord] = {}
        golden_records: dict[str, TensorRecord] = {}
        tensor_roles = dict(roles or {})

        for name, source_value in inputs.items():
            array, source_role = self._source_to_array(source_value)
            input_records[name] = self._store_tensor(
                case_dir,
                "inputs",
                name,
                array,
                role=tensor_roles.get(name, source_role),
            )

        for name, source_value in (goldens or {}).items():
            array, source_role = self._source_to_array(source_value)
            golden_records[name] = self._store_tensor(
                case_dir,
                "goldens",
                name,
                array,
                role=tensor_roles.get(name, source_role),
            )

        manifest = VectorManifest(
            case_id=case_id,
            inputs=input_records,
            goldens=golden_records,
            metadata=copy.deepcopy(dict(metadata or {})),
        )
        return self._write_manifest(case_dir / manifest_name, manifest)

    def import_case(
        self,
        case_id: str,
        inputs: Mapping[str, Any],
        *,
        goldens: Mapping[str, Any] | None = None,
        roles: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Alias with import-oriented naming for existing raw/NumPy vectors."""

        return self.prepare_case(
            case_id,
            inputs,
            goldens=goldens,
            roles=roles,
            metadata=metadata,
        )

    def capture_reference(
        self,
        manifest_path: str | os.PathLike[str],
        runner: Callable[..., Mapping[str, Any]],
        *,
        output_names: Sequence[str] | None = None,
        destination_name: str = "vector_manifest.reference.json",
        expected_manifest_sha256: str | None = None,
        metadata_updates: Mapping[str, Any] | None = None,
    ) -> Path:
        """Execute a reference runner and publish a new manifest with captured goldens.

        ``runner`` is called as ``runner(inputs)`` when ``output_names`` is
        ``None`` and as ``runner(inputs, tuple(output_names))`` otherwise.
        """

        source_path = _path_from_value(manifest_path)
        if expected_manifest_sha256 is None and hasattr(manifest_path, "sha256"):
            expected_manifest_sha256 = str(getattr(manifest_path, "sha256"))
        source = self.load_manifest(source_path, expected_sha256=expected_manifest_sha256)
        inputs = self.load_tensors(
            source_path,
            section="inputs",
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if output_names is None:
            outputs = runner(inputs)
        else:
            outputs = runner(inputs, tuple(output_names))
        if not isinstance(outputs, Mapping):
            raise TypeError("Reference runner must return a mapping of tensor name to array")
        if output_names is not None:
            missing = set(output_names) - set(outputs)
            if missing:
                raise KeyError(f"Reference runner did not return requested tensors: {sorted(missing)}")
            outputs = {name: outputs[name] for name in output_names}

        case_dir = source_path.parent
        golden_records = dict(source.goldens)
        for name, value in outputs.items():
            golden_records[name] = self._store_tensor(case_dir, "goldens", name, value, role="golden")

        captured = VectorManifest(
            case_id=source.case_id,
            inputs=dict(source.inputs),
            goldens=golden_records,
            metadata={
                **dict(source.metadata),
                **copy.deepcopy(dict(metadata_updates or {})),
            },
            source_manifest_sha256=sha256_file(source_path),
        )
        return self._write_manifest(case_dir / destination_name, captured)

    def capture_onnx(
        self,
        manifest_path: str | os.PathLike[str],
        model_path: str | os.PathLike[str],
        *,
        output_names: Sequence[str] | None = None,
        destination_name: str = "vector_manifest.onnx-reference.json",
        expected_manifest_sha256: str | None = None,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> Path:
        """Capture ONNX graph outputs using ONNX Runtime's Python API."""

        import onnxruntime as ort

        session = ort.InferenceSession(os.fspath(model_path), providers=list(providers))
        available_outputs = {item.name for item in session.get_outputs()}
        selected = tuple(output_names) if output_names is not None else tuple(sorted(available_outputs))
        missing = set(selected) - available_outputs
        if missing:
            raise KeyError(
                "Requested ONNX taps are not graph outputs. Export an instrumented ONNX first: "
                f"{sorted(missing)}"
            )

        def run(inputs: Mapping[str, np.ndarray], names: Sequence[str]) -> Mapping[str, np.ndarray]:
            values = session.run(list(names), dict(inputs))
            return dict(zip(names, values))

        return self.capture_reference(
            manifest_path,
            run,
            output_names=selected,
            destination_name=destination_name,
            expected_manifest_sha256=expected_manifest_sha256,
            metadata_updates={
                "reference_source": "onnxruntime",
                "reference_model_path": os.fspath(
                    Path(model_path).expanduser().resolve()
                ),
                "reference_model_sha256": sha256_file(model_path),
                "onnxruntime_version": str(getattr(ort, "__version__", "unknown")),
                "onnxruntime_providers": list(providers),
                "reference_output_names": list(selected),
            },
        )

    @staticmethod
    def onnx_producible_tensor_names(
        model_path: str | os.PathLike[str],
    ) -> frozenset[str]:
        """Return every tensor name a graph can expose as an output."""

        import onnx

        model = onnx.load(os.fspath(Path(model_path).expanduser().resolve()), load_external_data=False)
        graph = model.graph
        names = {
            *(item.name for item in graph.input),
            *(item.name for item in graph.output),
            *(item.name for item in graph.initializer),
        }
        for node in graph.node:
            names.update(str(name) for name in node.output if name)
        return frozenset(names)

    @staticmethod
    def capture_onnx_float_activations(
        model_path: str | os.PathLike[str],
        inputs: Mapping[str, np.ndarray],
        tensor_names: Sequence[str],
        *,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Run a float ONNX graph and capture named internal activations.

        The requested names are promoted to graph outputs in an in-memory copy
        of the model; the file on disk is never modified.  A name that no node
        in the graph produces is an error, not a silent omission: the caller is
        expected to have resolved the name mapping already and to report
        anything it could not map.

        Returns the captured values and the provenance an auditable report
        needs (model hash, ORT version, providers, promoted names).
        """

        import onnx
        import onnxruntime as ort

        resolved_model = Path(model_path).expanduser().resolve()
        requested = tuple(dict.fromkeys(str(name) for name in tensor_names))
        if not requested:
            raise ValueError("float activation capture requires at least one tensor name")

        model = onnx.load(os.fspath(resolved_model), load_external_data=False)
        graph = model.graph
        existing_outputs = {item.name for item in graph.output}
        producible = {
            *(item.name for item in graph.input),
            *(item.name for item in graph.initializer),
        }
        for node in graph.node:
            producible.update(str(name) for name in node.output if name)
        unknown = [name for name in requested if name not in producible | existing_outputs]
        if unknown:
            raise KeyError(
                "float reference tensors are not present in the graph: " f"{sorted(unknown)}"
            )

        promoted = [name for name in requested if name not in existing_outputs]
        for name in promoted:
            graph.output.append(onnx.helper.make_empty_tensor_value_info(name))

        external_data = [
            os.fspath(item)
            for item in _external_data_paths(model, resolved_model)
        ]
        with _instrumented_model_source(model, resolved_model, external_data) as source:
            session = ort.InferenceSession(source, providers=list(providers))
            session_inputs = {item.name for item in session.get_inputs()}
            missing_inputs = sorted(session_inputs - set(inputs))
            if missing_inputs:
                raise KeyError(
                    "float reference run is missing graph inputs: " f"{missing_inputs}"
                )
            feeds = {
                name: np.asarray(value)
                for name, value in inputs.items()
                if name in session_inputs
            }
            values = session.run(list(requested), feeds)

        provenance = {
            "reference_source": "onnxruntime_float",
            "reference_model_path": os.fspath(resolved_model),
            "reference_model_sha256": sha256_file(resolved_model),
            "reference_model_external_data": external_data,
            "onnxruntime_version": str(getattr(ort, "__version__", "unknown")),
            "onnxruntime_providers": list(providers),
            "captured_tensors": list(requested),
            "promoted_tensors": promoted,
        }
        return dict(zip(requested, values)), provenance

    @classmethod
    def write_input_list(
        cls,
        manifest_paths: Iterable[str | os.PathLike[str]],
        destination: str | os.PathLike[str],
        *,
        expected_manifest_sha256: Mapping[str, str] | None = None,
    ) -> Path:
        """Write a QAIRT-compatible named input list from verified manifests."""

        lines: list[str] = []
        expected_hashes = dict(expected_manifest_sha256 or {})
        for manifest_value in manifest_paths:
            manifest_path = _path_from_value(manifest_value)
            declared_sha = (
                str(getattr(manifest_value, "sha256"))
                if hasattr(manifest_value, "sha256")
                else None
            )
            manifest = cls.load_manifest(
                manifest_path,
                expected_sha256=expected_hashes.get(os.fspath(manifest_path), declared_sha),
            )
            tokens: list[str] = []
            for name, record in sorted(manifest.inputs.items()):
                tensor_path = cls._resolve_tensor_path(manifest_path, record)
                if any(character.isspace() for character in os.fspath(tensor_path)):
                    raise ValueError(f"QAIRT input-list path contains whitespace: {tensor_path}")
                if sha256_file(tensor_path) != record.sha256:
                    raise ValueError(f"Tensor {name!r} failed SHA256 verification")
                tokens.append(f"{name}:={tensor_path}")
            lines.append(" ".join(tokens))

        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
        if destination_path.exists() and destination_path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace input-list artifact {destination_path}")
        if not destination_path.exists():
            temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, destination_path)
        return destination_path


__all__ = [
    "TensorRecord",
    "TensorSource",
    "VectorManifest",
    "VectorPreparer",
    "sha256_file",
]
