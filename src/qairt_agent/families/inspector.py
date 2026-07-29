"""Lazy ONNX metadata inspection.

Importing :mod:`qairt_agent.families` must work on orchestration hosts without
ONNX.  The dependency is therefore loaded only when ``inspect`` is called.
"""

from __future__ import annotations

import importlib
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int | str | None, ...]
    dtype: str


@dataclass(frozen=True)
class NodeInfo:
    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    domain: str = ""


@dataclass(frozen=True)
class InitializerInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    num_elements: int
    content_sha256: str | None
    external_location: str | None = None


@dataclass(frozen=True)
class OnnxModelInfo:
    path: Path
    graph_name: str
    inputs: tuple[TensorInfo, ...]
    outputs: tuple[TensorInfo, ...]
    nodes: tuple[NodeInfo, ...]
    initializer_names: tuple[str, ...]
    initializers: tuple[InitializerInfo, ...]
    metadata: dict[str, str]

    @property
    def op_types(self) -> tuple[str, ...]:
        return tuple(node.op_type for node in self.nodes)


def _shape_from_value_info(value_info: Any) -> tuple[int | str | None, ...]:
    tensor_type = value_info.type.tensor_type
    dimensions: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if getattr(dimension, "dim_param", ""):
            dimensions.append(str(dimension.dim_param))
        elif hasattr(dimension, "HasField") and dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        else:
            value = getattr(dimension, "dim_value", None)
            dimensions.append(int(value) if value not in (None, 0) else None)
    return tuple(dimensions)


def _dtype_name(onnx_module: Any, value_info: Any) -> str:
    elem_type = value_info.type.tensor_type.elem_type
    tensor_proto = getattr(onnx_module, "TensorProto", None)
    data_type = getattr(tensor_proto, "DataType", None)
    name_method = getattr(data_type, "Name", None)
    if callable(name_method):
        try:
            return str(name_method(elem_type))
        except (TypeError, ValueError):
            pass
    return str(elem_type)


class OnnxInspector:
    """Inspect graph metadata without loading external tensor data."""

    def __init__(self, module_loader: Callable[[str], Any] | None = None) -> None:
        self._module_loader = module_loader or importlib.import_module

    def _tensor_infos(self, onnx_module: Any, values: Iterable[Any]) -> tuple[TensorInfo, ...]:
        return tuple(
            TensorInfo(
                name=str(value.name),
                shape=_shape_from_value_info(value),
                dtype=_dtype_name(onnx_module, value),
            )
            for value in values
        )

    @staticmethod
    def _external_digest(model_path: Path, tensor: Any) -> tuple[str | None, str | None]:
        metadata = {
            str(item.key): str(item.value)
            for item in getattr(tensor, "external_data", ())
        }
        location = metadata.get("location")
        if not location:
            return None, None
        external_path = (model_path.parent / location).resolve()
        try:
            offset = int(metadata.get("offset", "0"))
            length_value = metadata.get("length")
            remaining = int(length_value) if length_value else None
            digest = hashlib.sha256()
            with external_path.open("rb") as stream:
                stream.seek(offset)
                while remaining is None or remaining > 0:
                    chunk_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    digest.update(chunk)
                    if remaining is not None:
                        remaining -= len(chunk)
            if remaining not in (None, 0):
                return None, location
            return digest.hexdigest(), location
        except (OSError, TypeError, ValueError):
            return None, location

    def _initializer_info(
        self,
        onnx_module: Any,
        model_path: Path,
        tensor: Any,
    ) -> InitializerInfo:
        shape = tuple(int(value) for value in getattr(tensor, "dims", ()))
        num_elements = 1
        for dimension in shape:
            num_elements *= dimension
        digest, location = self._external_digest(model_path, tensor)
        if digest is None and location is None:
            serializer = getattr(tensor, "SerializeToString", None)
            if callable(serializer):
                digest = hashlib.sha256(serializer()).hexdigest()
            else:
                raw_data = bytes(getattr(tensor, "raw_data", b""))
                digest = hashlib.sha256(raw_data).hexdigest() if raw_data else None
        tensor_proto = getattr(onnx_module, "TensorProto", None)
        data_type = getattr(tensor_proto, "DataType", None)
        name_method = getattr(data_type, "Name", None)
        raw_dtype = getattr(tensor, "data_type", "")
        try:
            dtype = str(name_method(raw_dtype)) if callable(name_method) else str(raw_dtype)
        except (KeyError, TypeError, ValueError):
            dtype = str(raw_dtype)
        return InitializerInfo(
            name=str(tensor.name),
            shape=shape,
            dtype=dtype,
            num_elements=num_elements,
            content_sha256=digest,
            external_location=location,
        )

    def external_data_paths(self, model_path: str | Path) -> tuple[Path, ...]:
        """Return referenced external initializer files without hashing weights."""

        onnx_module = self._module_loader("onnx")
        path = Path(model_path).expanduser().resolve()
        model = onnx_module.load(str(path), load_external_data=False)
        resolved: set[Path] = set()
        for tensor in model.graph.initializer:
            metadata = {
                str(item.key): str(item.value)
                for item in getattr(tensor, "external_data", ())
            }
            location = metadata.get("location")
            if location:
                external_path = Path(location)
                if not external_path.is_absolute():
                    external_path = path.parent / external_path
                resolved.add(external_path.resolve())
        return tuple(sorted(resolved))

    def inspect(self, model_path: str | Path) -> OnnxModelInfo:
        onnx_module = self._module_loader("onnx")
        path = Path(model_path)
        model = onnx_module.load(str(path), load_external_data=False)
        graph = model.graph
        initializers = tuple(str(item.name) for item in graph.initializer)
        initializer_set = set(initializers)

        # ONNX graph inputs can include initializer-backed weights.  Only expose
        # true runtime inputs in the inspection contract.
        graph_inputs = tuple(item for item in graph.input if item.name not in initializer_set)
        nodes = tuple(
            NodeInfo(
                name=str(node.name),
                op_type=str(node.op_type),
                inputs=tuple(str(item) for item in node.input),
                outputs=tuple(str(item) for item in node.output),
                domain=str(getattr(node, "domain", "")),
            )
            for node in graph.node
        )
        metadata = {
            str(item.key): str(item.value)
            for item in getattr(model, "metadata_props", ())
        }
        return OnnxModelInfo(
            path=path,
            graph_name=str(graph.name),
            inputs=self._tensor_infos(onnx_module, graph_inputs),
            outputs=self._tensor_infos(onnx_module, graph.output),
            nodes=nodes,
            initializer_names=initializers,
            initializers=tuple(
                self._initializer_info(onnx_module, path, tensor)
                for tensor in graph.initializer
            ),
            metadata=metadata,
        )
