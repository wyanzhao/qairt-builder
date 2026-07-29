"""Restricted, opt-in pickle import for trusted local test vectors.

Pickle is disabled everywhere else in the agent: normal builds never call
``pickle.load`` implicitly.  This module is the *only* sanctioned path that can
reconstruct a pickle, and it is deliberately narrow:

* :class:`RestrictedUnpickler` resolves globals through an explicit
  ``(module, name)`` allowlist containing just the NumPy entry points required
  to rebuild arrays/scalars.  Dotted attribute paths in the ``name`` argument
  are rejected unconditionally as defense-in-depth against CPython's
  ``pickle._getattribute`` traversal.  Anything else (``os.system``,
  ``posix.*``, ``torch.*``, user classes, arbitrary callables) is rejected with
  :class:`PickleRejectedError`.
* :func:`safe_load_pickle` layers a size cap, a recursive tree validator, and an
  optional subprocess+``resource.setrlimit`` isolation sandbox on top of that
  unpickler.  When isolation is active, the child serializes the validated
  tree through a non-pickle binary protocol (JSON header + raw array bytes) so
  the parent never calls pickle on child-produced data.
* :func:`import_pickle_artifacts` is the explicit, ``--trusted-local``-gated
  operation that converts a validated container tree into a
  :class:`~qairt_agent.contracts.VectorBundle`.

Security notes:

* The rlimit sandbox constrains CPU, file size, and address space but does
  **not** provide filesystem isolation.  The allowlist is the authoritative
  gate; rlimits are a secondary mitigation.
* ``--trusted-local`` is an operator's declaration about the *source* of the
  pickle file, not a security proof.  The restricted unpickler remains the
  enforcement boundary regardless of this flag.

The module has no QAIRT SDK dependency and never opens a file for writing except
the raw tensor artifacts that :func:`import_pickle_artifacts` is asked to emit.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import struct
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from qairt_agent.contracts import TensorRepresentation, VectorBundle, VectorTensor
from qairt_agent.errors import PickleRejectedError
from qairt_agent.vectors import (
    TensorRecord,
    VectorManifest,
    VectorPreparer,
    _atomic_write,
    _canonical_array,
    _safe_name,
)

_DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024
_MAX_BYTES_ENV = "QAIRT_PICKLE_MAX_INPUT_BYTES"
_TIMEOUT_ENV = "QAIRT_PICKLE_TIMEOUT_SECONDS"
_ISOLATION_CHILD_CODE = (
    "from qairt_agent.vectors_pickle import _isolation_child_main as _main; _main()"
)
_TORCH_ISOLATION_CHILD_CODE = (
    "from qairt_agent.vectors_pickle import _torch_isolation_child_main as _main; _main()"
)
_SOURCE_FORMATS = frozenset({"auto", "numpy-pickle", "torch"})
_SECTIONS = frozenset({"auto", "inputs", "goldens"})

# Explicit global allowlist.  NumPy 1.26 rebuilds arrays through
# ``numpy.core.multiarray._reconstruct`` and scalars through
# ``numpy.core.multiarray.scalar``; NumPy 2.x moved these to ``numpy._core``.
# ``numpy.ndarray`` and ``numpy.dtype`` are themselves referenced as globals
# during reconstruction, so they must be admitted too.  ``numpy.dtypes.*`` holds
# the scalar dtype classes used by newer NumPy pickles; each is listed
# explicitly so that a dotted attribute path (e.g.
# ``numpy.dtypes.__loader__.set_data``) can never sneak through a prefix match.
# Basic containers (dict/list/tuple) and primitives are encoded with dedicated
# pickle opcodes and therefore need no globals at all -- in particular
# ``builtins.getattr`` is never admitted.
_ALLOWED_GLOBALS: dict[str, frozenset[str]] = {
    "numpy.core.multiarray": frozenset({"_reconstruct", "scalar"}),
    "numpy._core.multiarray": frozenset({"_reconstruct", "scalar"}),
    "numpy": frozenset({"dtype", "ndarray"}),
    "numpy.dtypes": frozenset({
        "BoolDType",
        "ByteDType",
        "UByteDType",
        "ShortDType",
        "UShortDType",
        "IntDType",
        "UIntDType",
        "LongDType",
        "ULongDType",
        "LongLongDType",
        "ULongLongDType",
        "Int8DType",
        "Int16DType",
        "Int32DType",
        "Int64DType",
        "UInt8DType",
        "UInt16DType",
        "UInt32DType",
        "UInt64DType",
        "Float16DType",
        "Float32DType",
        "Float64DType",
        "LongDoubleDType",
        "Complex64DType",
        "Complex128DType",
        "CLongDoubleDType",
    }),
}


class RestrictedUnpickler(pickle.Unpickler):
    """An unpickler that only resolves an explicit NumPy-only global allowlist."""

    def find_class(self, module: str, name: str) -> Any:
        # Defense-in-depth: CPython's pickle._getattribute traverses dotted
        # attribute paths in protocol >= 4, so a name like "__loader__.set_data"
        # would resolve through intermediate objects.  Reject unconditionally.
        if "." in name:
            raise PickleRejectedError(
                f"Pickle global {module}.{name} contains a dotted name and is "
                f"not permitted by the restricted unpickler",
                stage="unpickle",
                details={"module": module, "name": name},
            )
        allowed_names = _ALLOWED_GLOBALS.get(module)
        if allowed_names is not None and name in allowed_names:
            return super().find_class(module, name)
        raise PickleRejectedError(
            f"Pickle global {module}.{name} is not permitted by the restricted unpickler",
            stage="unpickle",
            details={"module": module, "name": name},
        )

    def persistent_load(self, pid: Any) -> Any:  # pragma: no cover - defensive
        raise PickleRejectedError(
            "Pickle persistent ids are not permitted by the restricted unpickler",
            stage="unpickle",
            details={"pid": str(pid)},
        )


def _read_source_bytes(source: bytes | os.PathLike[str] | str) -> bytes:
    """Return the raw pickle bytes for *source* without ever writing."""

    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if isinstance(source, (str, os.PathLike)):
        return Path(source).read_bytes()
    raise PickleRejectedError(
        f"Cannot read pickle source of type {type(source).__name__}",
        stage="read",
        details={"type": type(source).__name__},
    )


def _validated_choice(value: str, *, choices: frozenset[str], label: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise PickleRejectedError(
            f"Unsupported {label} {value!r}; expected one of {sorted(choices)}",
            stage="read",
            details={label: value, "supported": sorted(choices)},
        )
    return normalized


def _looks_like_torch_archive(data: bytes) -> bool:
    """Recognize the zip container emitted by modern ``torch.save``.

    Auto-detection deliberately does not inspect or execute the archive's
    pickle payload. Legacy non-zip torch archives and direct tensor pickles are
    unsupported in every mode.
    """

    if not data.startswith(b"PK"):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
            names = tuple(archive.namelist())
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return False
    has_pickle = any(name == "data.pkl" or name.endswith("/data.pkl") for name in names)
    has_version = any(name == "version" or name.endswith("/version") for name in names)
    return has_pickle and has_version


def _effective_source_format(data: bytes, source_format: str) -> str:
    selected = _validated_choice(
        source_format,
        choices=_SOURCE_FORMATS,
        label="source_format",
    )
    if selected != "auto":
        return selected
    return "torch" if _looks_like_torch_archive(data) else "numpy-pickle"


def detect_pickle_source_format(
    source: bytes | os.PathLike[str] | str,
    *,
    source_format: str = "auto",
    max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES,
) -> str:
    """Resolve ``auto`` without reconstructing any pickle object."""

    data = _read_source_bytes(source)
    if len(data) > max_input_bytes:
        raise PickleRejectedError(
            "Pickle input exceeds the maximum allowed size",
            stage="read",
            details={"nbytes": len(data), "max_input_bytes": max_input_bytes},
        )
    return _effective_source_format(data, source_format)


def _validate_tree(obj: Any) -> None:
    """Recursively admit only containers, NumPy arrays/scalars, and primitives.

    Object-dtype arrays are rejected explicitly even if the unpickler allowed
    them through, and any other type (callables, custom classes, torch tensors,
    sets, arbitrary bytes blobs, ...) is rejected.
    """

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "O" or obj.dtype.hasobject:
            raise PickleRejectedError(
                "Object-dtype numpy arrays are not permitted in pickle imports",
                stage="validate",
                details={"dtype": str(obj.dtype)},
            )
        return
    if isinstance(obj, np.generic):
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            _validate_tree(key)
            _validate_tree(value)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _validate_tree(item)
        return
    raise PickleRejectedError(
        f"Type {type(obj).__name__} is not permitted in pickle imports",
        stage="validate",
        details={"type": type(obj).__name__},
    )


# ---------------------------------------------------------------------------
# Non-pickle isolation channel
#
# The rlimit child serializes the validated tree as a length-prefixed JSON
# header followed by raw array bytes.  The parent reconstructs the tree from
# this binary protocol *without* ever calling pickle, so a compromised child
# cannot re-attack the parent through the same unpickler.
#
# Wire format:
#   [4 bytes: uint32-LE header length][JSON header (UTF-8)][raw array bytes]
# ---------------------------------------------------------------------------

def _serialize_validated_tree(obj: Any) -> bytes:
    """Encode a validated tree as JSON header + raw array bytes (no pickle)."""

    raw_parts: list[bytes] = []
    current_offset = 0

    def encode(node: Any) -> Any:
        nonlocal current_offset
        if node is None:
            return {"__type__": "none"}
        if isinstance(node, np.generic):
            raw = node.tobytes()
            offset = current_offset
            current_offset += len(raw)
            raw_parts.append(raw)
            return {
                "__type__": "npscalar",
                "dtype": node.dtype.str,
                "offset": offset,
                "nbytes": len(raw),
            }
        if isinstance(node, bool):
            return {"__type__": "primitive", "value": node}
        if isinstance(node, (str, int, float)):
            return {"__type__": "primitive", "value": node}
        if isinstance(node, np.ndarray):
            raw = node.tobytes(order="C")
            offset = current_offset
            current_offset += len(raw)
            raw_parts.append(raw)
            return {
                "__type__": "ndarray",
                "dtype": node.dtype.str,
                "shape": list(node.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
        if isinstance(node, dict):
            return {
                "__type__": "dict",
                "items": [[encode(k), encode(v)] for k, v in node.items()],
            }
        if isinstance(node, list):
            return {"__type__": "list", "items": [encode(item) for item in node]}
        if isinstance(node, tuple):
            return {"__type__": "tuple", "items": [encode(item) for item in node]}
        raise PickleRejectedError(
            f"Type {type(node).__name__} cannot be serialized through the "
            f"isolation channel",
            stage="isolate",
            details={"type": type(node).__name__},
        )

    header = json.dumps(encode(obj), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return struct.pack("<I", len(header)) + header + b"".join(raw_parts)


def _deserialize_validated_tree(data: bytes) -> Any:
    """Reconstruct a validated tree from the non-pickle isolation channel."""

    if len(data) < 4:
        raise PickleRejectedError(
            "Truncated isolation channel data",
            stage="isolate",
            details={"nbytes": len(data)},
        )
    (header_len,) = struct.unpack("<I", data[:4])
    if len(data) < 4 + header_len:
        raise PickleRejectedError(
            "Truncated isolation channel header",
            stage="isolate",
            details={"header_len": header_len, "nbytes": len(data)},
        )
    try:
        header = json.loads(data[4 : 4 + header_len].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PickleRejectedError(
            f"Invalid isolation channel header: {exc}",
            stage="isolate",
            details={"error": str(exc)},
        ) from exc
    raw_section = data[4 + header_len :]

    def decode(node: Any) -> Any:
        if not isinstance(node, dict) or "__type__" not in node:
            raise PickleRejectedError(
                "Invalid isolation channel node",
                stage="isolate",
                details={"node": repr(node)[:200]},
            )
        tag = node["__type__"]
        if tag == "none":
            return None
        if tag == "primitive":
            value = node["value"]
            if not isinstance(value, (str, int, float, bool)) and value is not None:
                raise PickleRejectedError(
                    f"Invalid primitive type in isolation channel: {type(value).__name__}",
                    stage="isolate",
                )
            return value
        if tag == "ndarray":
            dtype = np.dtype(node["dtype"])
            shape = tuple(int(d) for d in node["shape"])
            nbytes = int(node["nbytes"])
            offset = int(node["offset"])
            if offset + nbytes > len(raw_section):
                raise PickleRejectedError(
                    "Isolation channel array data out of bounds",
                    stage="isolate",
                    details={"offset": offset, "nbytes": nbytes, "available": len(raw_section)},
                )
            expected = dtype.itemsize * int(np.prod(shape)) if shape else dtype.itemsize
            if nbytes != expected:
                raise PickleRejectedError(
                    "Isolation channel array size mismatch",
                    stage="isolate",
                    details={"nbytes": nbytes, "expected": expected},
                )
            return np.frombuffer(raw_section[offset : offset + nbytes], dtype=dtype).reshape(shape).copy()
        if tag == "npscalar":
            dtype = np.dtype(node["dtype"])
            nbytes = int(node["nbytes"])
            offset = int(node["offset"])
            if (
                nbytes != dtype.itemsize
                or offset < 0
                or offset + nbytes > len(raw_section)
            ):
                raise PickleRejectedError(
                    "Isolation channel scalar data is invalid",
                    stage="isolate",
                    details={
                        "offset": offset,
                        "nbytes": nbytes,
                        "expected": dtype.itemsize,
                        "available": len(raw_section),
                    },
                )
            return np.frombuffer(
                raw_section[offset : offset + nbytes],
                dtype=dtype,
                count=1,
            )[0]
        if tag == "dict":
            return {decode(k): decode(v) for k, v in node["items"]}
        if tag == "list":
            return [decode(item) for item in node["items"]]
        if tag == "tuple":
            return tuple(decode(item) for item in node["items"])
        raise PickleRejectedError(
            f"Unknown isolation channel tag: {tag!r}",
            stage="isolate",
            details={"tag": tag},
        )

    return decode(header)


def _load_in_process(data: bytes) -> Any:
    try:
        return RestrictedUnpickler(io.BytesIO(data)).load()
    except PickleRejectedError:
        raise
    except Exception as exc:  # noqa: BLE001 - any unpickle failure fails closed
        raise PickleRejectedError(
            f"Failed to unpickle input: {exc}",
            stage="unpickle",
            details={"error": str(exc)},
        ) from exc


def _subprocess_isolation_available() -> bool:
    """Subprocess isolation requires the POSIX ``resource`` module."""

    try:
        import resource  # noqa: F401
    except ImportError:
        return False
    return True


def _apply_resource_limits() -> None:  # pragma: no cover - runs in the child
    """Best-effort rlimit caps applied inside the isolation child."""

    try:
        import resource
    except ImportError:
        return

    try:
        timeout = float(os.environ.get(_TIMEOUT_ENV, "30"))
    except ValueError:
        timeout = 30.0
    try:
        max_bytes = int(os.environ.get(_MAX_BYTES_ENV, str(_DEFAULT_MAX_INPUT_BYTES)))
    except ValueError:
        max_bytes = _DEFAULT_MAX_INPUT_BYTES

    cpu_seconds = max(1, int(timeout) + 1)
    file_size = max(max_bytes * 2, 64 * 1024 * 1024)
    address_space = 4 * 1024 * 1024 * 1024
    candidates = (
        ("RLIMIT_CPU", cpu_seconds),
        ("RLIMIT_FSIZE", file_size),
        ("RLIMIT_NOFILE", 128),
        ("RLIMIT_AS", address_space),
    )
    for name, soft in candidates:
        constant = getattr(resource, name, None)
        if constant is None:
            continue
        try:
            resource.setrlimit(constant, (soft, soft))
        except (ValueError, OSError):
            # Some limits (notably RLIMIT_AS on macOS) cannot be lowered; the
            # allowlist remains the authoritative gate when a cap is refused.
            continue


def _isolation_child_main() -> None:  # pragma: no cover - runs in the child
    """Child entry point: load + validate under rlimits, emit via safe channel.

    The validated tree is serialized through a non-pickle binary protocol
    (JSON header + raw array bytes) so the parent never calls pickle on
    child-produced data.
    """

    _apply_resource_limits()
    raw = sys.stdin.buffer.read()
    try:
        max_bytes = int(os.environ.get(_MAX_BYTES_ENV, str(_DEFAULT_MAX_INPUT_BYTES)))
    except ValueError:
        max_bytes = _DEFAULT_MAX_INPUT_BYTES
    if len(raw) > max_bytes:
        raise PickleRejectedError(
            "Pickle input exceeds the maximum allowed size",
            stage="read",
            details={"nbytes": len(raw), "max_input_bytes": max_bytes},
        )
    obj = RestrictedUnpickler(io.BytesIO(raw)).load()
    _validate_tree(obj)
    sys.stdout.buffer.write(_serialize_validated_tree(obj))
    sys.stdout.buffer.flush()


def _normalize_torch_tree(obj: Any, torch_module: Any) -> Any:
    """Convert a weights-only torch object tree into the NumPy-safe tree.

    Only dense tensors and the same plain containers/primitives admitted by the
    restricted NumPy loader are supported.  Quantized, bfloat16, and float8
    tensors are decoded to logical float32 because NumPy 1.26 cannot represent
    all of those physical dtypes portably.
    """

    tensor_type = getattr(torch_module, "Tensor", None)
    if tensor_type is not None and isinstance(obj, tensor_type):
        tensor = obj.detach().cpu()
        layout = getattr(tensor, "layout", None)
        strided = getattr(torch_module, "strided", None)
        if layout is not None and layout != strided and str(layout) != "torch.strided":
            raise PickleRejectedError(
                f"Torch tensor layout {layout!s} is not supported",
                stage="torch_normalize",
                details={"layout": str(layout)},
            )
        if bool(getattr(tensor, "is_quantized", False)):
            tensor = tensor.dequantize()
        dtype_name = str(getattr(tensor, "dtype", ""))
        if dtype_name == "torch.bfloat16" or dtype_name.startswith("torch.float8"):
            tensor = tensor.float()
        try:
            return np.asarray(tensor.numpy())
        except Exception as exc:  # noqa: BLE001 - conversion must fail closed
            raise PickleRejectedError(
                f"Torch tensor dtype {dtype_name or '<unknown>'} cannot be converted "
                f"to a NumPy SQNR tensor: {exc}",
                stage="torch_normalize",
                details={"dtype": dtype_name, "error": str(exc)},
            ) from exc
    if obj is None or isinstance(obj, (str, int, float, bool, np.ndarray, np.generic)):
        return obj
    if isinstance(obj, Mapping):
        normalized: dict[Any, Any] = {}
        for key, value in obj.items():
            normalized_key = _normalize_torch_tree(key, torch_module)
            try:
                duplicate = normalized_key in normalized
            except TypeError as exc:
                raise PickleRejectedError(
                    "Torch archive contains an unsupported mapping key",
                    stage="torch_normalize",
                    details={"key_type": type(normalized_key).__name__},
                ) from exc
            if duplicate:
                raise PickleRejectedError(
                    f"Torch archive mapping key collision after normalization: "
                    f"{normalized_key!r}",
                    stage="torch_normalize",
                )
            normalized[normalized_key] = _normalize_torch_tree(value, torch_module)
        return normalized
    if isinstance(obj, list):
        return [_normalize_torch_tree(item, torch_module) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_normalize_torch_tree(item, torch_module) for item in obj)
    raise PickleRejectedError(
        f"Type {type(obj).__name__} is not supported in a weights-only torch archive",
        stage="torch_normalize",
        details={"type": type(obj).__name__},
    )


def _torch_load_archive(data: bytes, *, torch_module: Any | None = None) -> Any:
    """Load one torch archive with the pinned safe loader contract."""

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise PickleRejectedError(
                "Torch pickle import requires the pinned torch dependency inside "
                "the Ubuntu worker",
                stage="torch_load",
                details={"dependency": "torch"},
            ) from exc
    try:
        loaded = torch_module.load(
            io.BytesIO(data),
            weights_only=True,
            map_location="cpu",
        )
    except TypeError as exc:
        raise PickleRejectedError(
            "Installed torch does not support the required weights_only loader",
            stage="torch_load",
            details={"error": str(exc)},
        ) from exc
    except Exception as exc:  # noqa: BLE001 - torch rejects fail closed
        raise PickleRejectedError(
            f"Safe torch archive load failed: {exc}",
            stage="torch_load",
            details={"error": str(exc)},
        ) from exc
    normalized = _normalize_torch_tree(loaded, torch_module)
    _validate_tree(normalized)
    return normalized


def _torch_isolation_child_main() -> None:  # pragma: no cover - runs in the child
    """Load a torch archive in the rlimit child and emit via safe channel."""

    _apply_resource_limits()
    raw = sys.stdin.buffer.read()
    try:
        max_bytes = int(os.environ.get(_MAX_BYTES_ENV, str(_DEFAULT_MAX_INPUT_BYTES)))
    except ValueError:
        max_bytes = _DEFAULT_MAX_INPUT_BYTES
    if len(raw) > max_bytes:
        raise PickleRejectedError(
            "Pickle input exceeds the maximum allowed size",
            stage="read",
            details={"nbytes": len(raw), "max_input_bytes": max_bytes},
        )
    obj = _torch_load_archive(raw)
    sys.stdout.buffer.write(_serialize_validated_tree(obj))
    sys.stdout.buffer.flush()


def _load_isolated(data: bytes, *, timeout: float, max_input_bytes: int) -> Any:
    env = dict(os.environ)
    paths = [os.path.abspath(entry) for entry in sys.path if entry]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([*paths, existing] if existing else paths)
    env[_MAX_BYTES_ENV] = str(max_input_bytes)
    env[_TIMEOUT_ENV] = str(timeout)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _ISOLATION_CHILD_CODE],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PickleRejectedError(
            "Isolated pickle load timed out",
            stage="isolate",
            details={"timeout": timeout},
        ) from exc
    except (OSError, ValueError) as exc:
        raise PickleRejectedError(
            f"Isolated pickle load failed to start: {exc}",
            stage="isolate",
            details={"error": str(exc)},
        ) from exc

    if proc.returncode != 0:
        stderr_tail = proc.stderr.decode("utf-8", errors="replace")[-2000:]
        raise PickleRejectedError(
            "Isolated pickle load was rejected",
            stage="isolate",
            details={"returncode": proc.returncode, "stderr": stderr_tail},
        )

    obj = _deserialize_validated_tree(proc.stdout)
    _validate_tree(obj)
    return obj


def _load_torch_isolated(data: bytes, *, timeout: float, max_input_bytes: int) -> Any:
    """Load torch only in a child process; never fall back to in-process."""

    env = dict(os.environ)
    paths = [os.path.abspath(entry) for entry in sys.path if entry]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([*paths, existing] if existing else paths)
    env[_MAX_BYTES_ENV] = str(max_input_bytes)
    env[_TIMEOUT_ENV] = str(timeout)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _TORCH_ISOLATION_CHILD_CODE],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PickleRejectedError(
            "Isolated torch archive load timed out",
            stage="torch_isolate",
            details={"timeout": timeout},
        ) from exc
    except (OSError, ValueError) as exc:
        raise PickleRejectedError(
            f"Isolated torch archive load failed to start: {exc}",
            stage="torch_isolate",
            details={"error": str(exc)},
        ) from exc
    if proc.returncode != 0:
        stderr_tail = proc.stderr.decode("utf-8", errors="replace")[-2000:]
        raise PickleRejectedError(
            "Isolated torch archive load was rejected",
            stage="torch_isolate",
            details={"returncode": proc.returncode, "stderr": stderr_tail},
        )
    obj = _deserialize_validated_tree(proc.stdout)
    _validate_tree(obj)
    return obj


def safe_load_pickle(
    source: bytes | os.PathLike[str] | str,
    *,
    max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES,
    isolate: bool = False,
    timeout: float = 30.0,
    source_format: str = "auto",
) -> object:
    """Load a pickle through the restricted unpickler and validate the result.

    ``source`` is read-only (``bytes`` or a ``Path``); inputs larger than
    ``max_input_bytes`` are rejected before any parsing.  When ``isolate=True``
    the load+validate runs in a subprocess capped with ``resource.setrlimit``
    and the validated tree is returned to the parent by re-serializing only the
    safe NumPy/container structure.
    """

    data = _read_source_bytes(source)
    if len(data) > max_input_bytes:
        raise PickleRejectedError(
            "Pickle input exceeds the maximum allowed size",
            stage="read",
            details={"nbytes": len(data), "max_input_bytes": max_input_bytes},
        )

    effective_format = _effective_source_format(data, source_format)
    if effective_format == "torch":
        if not _looks_like_torch_archive(data):
            raise PickleRejectedError(
                "Torch format accepts modern torch.save zip archives only; "
                "direct pickle.dump(torch.Tensor) and legacy non-zip archives "
                "are not supported",
                stage="read",
                details={"source_format": "torch"},
            )
        return _load_torch_isolated(
            data,
            timeout=timeout,
            max_input_bytes=max_input_bytes,
        )

    if isolate and _subprocess_isolation_available():
        return _load_isolated(data, timeout=timeout, max_input_bytes=max_input_bytes)

    # isolate=False, or the ``resource`` module is unavailable (e.g. Windows):
    # fall back to in-process loading.  The RestrictedUnpickler allowlist and the
    # tree validator still apply, but no rlimit/subprocess sandbox is used.
    obj = _load_in_process(data)
    _validate_tree(obj)
    return obj


def _flatten_leaves(obj: Any, prefix: str, leaves: list[tuple[str, np.ndarray]]) -> None:
    """Flatten a validated container tree into ``(name, ndarray)`` leaf pairs.

    Names are derived from dict keys and list indices (``"logits"``,
    ``"layers.0.hidden"``).  NumPy scalars become 0-d arrays; plain Python
    primitives are not arrays and are skipped.
    """

    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten_leaves(value, child, leaves)
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            child = f"{prefix}.{index}" if prefix else str(index)
            _flatten_leaves(value, child, leaves)
    elif isinstance(obj, np.ndarray):
        leaves.append((prefix or "tensor", obj))
    elif isinstance(obj, np.generic):
        leaves.append((prefix or "tensor", np.asarray(obj)))
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return
    else:  # pragma: no cover - unreachable after _validate_tree
        raise PickleRejectedError(
            f"Type {type(obj).__name__} cannot be flattened into a vector leaf",
            stage="flatten",
            details={"type": type(obj).__name__},
        )


def _unique_leaves(
    obj: Any,
    *,
    section: str,
) -> dict[str, np.ndarray]:
    leaves: list[tuple[str, np.ndarray]] = []
    _flatten_leaves(obj, "", leaves)
    result: dict[str, np.ndarray] = {}
    for name, value in leaves:
        if name in result:
            raise PickleRejectedError(
                f"Flattened tensor name collision in {section}: {name!r}",
                stage="flatten",
                details={"section": section, "name": name},
            )
        result[name] = value
    return result


def import_pickle_vectors(
    source: bytes | os.PathLike[str] | str,
    *,
    output_dir: str | os.PathLike[str],
    bundle_id: str | None = None,
    trusted_local: bool = False,
    isolate: bool = False,
    source_format: str = "auto",
) -> VectorBundle:
    """Deprecated compatibility wrapper over the canonical artifact importer."""

    imported = import_pickle_artifacts(
        source,
        output_dir=output_dir,
        bundle_id=bundle_id,
        case_id=bundle_id or "pickle-import",
        trusted_local=trusted_local,
        isolate=isolate,
        source_format=source_format,
        section="goldens",
        _raw_subdir=None,
    )
    root = Path(output_dir).expanduser().resolve()
    return imported.bundle.model_copy(
        update={
            "tensors": tuple(
                tensor.model_copy(update={"path": (root / tensor.path).resolve()})
                for tensor in imported.bundle.tensors
            )
        }
    )


@dataclass(frozen=True)
class ImportedPickleArtifacts:
    """Persistent outputs of a trusted pickle import.

    ``manifest_path`` is the artifact consumed by ``VectorSpec`` and the
    validation/benchmark pipeline.  ``bundle_path`` preserves the richer
    per-tensor provenance for inspection.  A legacy pickle without an
    explicit top-level ``inputs`` section remains useful as a golden-only
    SQNR reference, but is not executable on a device by itself.
    """

    bundle: VectorBundle
    bundle_path: Path
    manifest_path: Path
    execution_ready: bool
    source_format: str
    section: str


def _section_arrays(
    obj: object,
    *,
    section: str = "auto",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Interpret a validated pickle as ``inputs``/``goldens`` tensor sections."""

    selected = _validated_choice(section, choices=_SECTIONS, label="section")
    if selected != "auto":
        values = _unique_leaves(obj, section=selected)
        if not values:
            raise PickleRejectedError(
                f"Selected pickle section {selected!r} contains no tensor leaves",
                stage="flatten",
                details={"section": selected},
            )
        return (values, {}) if selected == "inputs" else ({}, values)

    if isinstance(obj, Mapping) and (
        "inputs" in obj or "goldens" in obj
    ):
        sections: dict[str, dict[str, np.ndarray]] = {}
        for name in ("inputs", "goldens"):
            if name not in obj:
                sections[name] = {}
                continue
            values = _unique_leaves(obj[name], section=name)
            if not values:
                raise PickleRejectedError(
                    f"Explicit pickle section {name!r} contains no tensor leaves",
                    stage="flatten",
                    details={"section": name},
                )
            sections[name] = values
        return sections["inputs"], sections["goldens"]

    # Backward-compatible shape: an unsectioned pickle is a collection of
    # golden outputs.  It can be paired with separately captured device
    # outputs for SQNR, but cannot drive execution without model inputs.
    goldens = _unique_leaves(obj, section="goldens")
    if not goldens:
        raise PickleRejectedError(
            "Pickle contains no tensor leaves",
            stage="flatten",
        )
    return {}, goldens


def _record_from_tensor(tensor: VectorTensor, *, role: str) -> TensorRecord:
    if tensor.path is None or tensor.sha256 is None or tensor.nbytes is None:
        raise PickleRejectedError(
            f"Imported tensor {tensor.name!r} is missing persistent storage metadata",
            stage="materialize",
        )
    return TensorRecord(
        name=tensor.name,
        path=os.fspath(tensor.path),
        dtype=tensor.dtype,
        shape=tensor.shape,
        sha256=tensor.sha256,
        nbytes=tensor.nbytes,
        storage="raw",
        byte_order=tensor.byte_order,
        role=role,
    )


def import_pickle_artifacts(
    source: bytes | os.PathLike[str] | str,
    *,
    output_dir: str | os.PathLike[str],
    bundle_id: str | None = None,
    case_id: str = "pickle-import",
    trusted_local: bool = False,
    isolate: bool = False,
    source_format: str = "auto",
    section: str = "auto",
    source_key: str | None = None,
    expected_source_sha256: str | None = None,
    _raw_subdir: str | None = "raw",
) -> ImportedPickleArtifacts:
    """Convert a trusted pickle into raw tensors plus a usable VectorManifest.

    Preferred pickle shape::

        {
          "inputs": {"input_ids": np.ndarray, ...},
          "goldens": {"logits": np.ndarray, ...}
        }

    An unsectioned mapping is treated as golden-only for compatibility.  The
    returned manifest can then be used for offline SQNR against an
    ``actual_manifest``; device execution additionally requires an ``inputs``
    section.
    """

    if not trusted_local:
        raise PickleRejectedError(
            "Pickle import is disabled unless explicitly enabled with trusted_local=True "
            "(--trusted-local)",
            stage="gate",
        )
    if not case_id.strip():
        raise PickleRejectedError("case_id cannot be blank", stage="materialize")

    data = _read_source_bytes(source)
    source_sha256 = hashlib.sha256(data).hexdigest()
    if (
        expected_source_sha256 is not None
        and source_sha256 != expected_source_sha256
    ):
        raise PickleRejectedError(
            "Pickle source hash does not match its declared provenance",
            stage="read",
            details={
                "expected_sha256": expected_source_sha256,
                "actual_sha256": source_sha256,
            },
        )
    resolved_source_key = source_key
    if resolved_source_key is None and isinstance(source, (str, os.PathLike)):
        resolved_source_key = str(Path(source).expanduser().resolve())
    effective_format = _effective_source_format(data, source_format)
    selected_section = _validated_choice(
        section,
        choices=_SECTIONS,
        label="section",
    )
    obj = safe_load_pickle(
        data,
        isolate=isolate,
        source_format=effective_format,
    )
    inputs, goldens = _section_arrays(obj, section=selected_section)
    if not inputs and not goldens:
        raise PickleRejectedError(
            "Pickle contains no NumPy tensor leaves",
            stage="flatten",
        )

    output = Path(output_dir).expanduser().resolve()
    # The legacy ``import_pickle_vectors`` API placed raw files directly in
    # ``output_dir``.  Preserve that layout while sharing this implementation.
    raw_root = output / _raw_subdir if _raw_subdir is not None else output
    raw_root.mkdir(parents=True, exist_ok=True)
    resolved_bundle_id = bundle_id or uuid4().hex
    tensors: list[VectorTensor] = []
    records_by_section: dict[str, dict[str, TensorRecord]] = {
        "inputs": {},
        "goldens": {},
    }

    for section, values, role in (
        ("inputs", inputs, "input"),
        ("goldens", goldens, "golden"),
    ):
        for name, array in values.items():
            canonical = _canonical_array(array)
            raw_bytes = canonical.tobytes(order="C")
            digest = hashlib.sha256(raw_bytes).hexdigest()
            destination = raw_root / f"{_safe_name(name)}-{digest[:12]}.raw"
            if destination.exists():
                if destination.read_bytes() != raw_bytes:
                    raise PickleRejectedError(
                        f"Refusing to replace conflicting tensor {destination}",
                        stage="materialize",
                    )
            else:
                _atomic_write(destination, raw_bytes)
            tensor = VectorTensor(
                name=name,
                path=destination.relative_to(output),
                representation=TensorRepresentation.LOGICAL_FP,
                dtype=str(canonical.dtype),
                shape=tuple(int(dim) for dim in canonical.shape),
                layout="C",
                byte_order=(
                    "not-applicable" if canonical.dtype.itemsize == 1 else "little"
                ),
                sha256=digest,
                nbytes=canonical.nbytes,
                role=role,
            )
            tensors.append(tensor)
            records_by_section[section][name] = _record_from_tensor(
                tensor,
                role=role,
            )

    bundle = VectorBundle(
        bundle_id=resolved_bundle_id,
        tensors=tuple(tensors),
        source_key=resolved_source_key,
        source_sha256=source_sha256,
        metadata={
            "schema": "qairt-agent.pickle-import",
            "case_id": case_id,
            "execution_ready": bool(inputs),
            "source_format": effective_format,
            "section": selected_section,
            **(
                {"source_path": resolved_source_key}
                if resolved_source_key is not None
                else {}
            ),
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = output / "vector_bundle.json"
    bundle_payload = (
        json.dumps(
            bundle.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    if bundle_path.exists() and bundle_path.read_bytes() != bundle_payload:
        raise PickleRejectedError(
            f"Refusing to replace immutable bundle {bundle_path}",
            stage="materialize",
        )
    if not bundle_path.exists():
        _atomic_write(bundle_path, bundle_payload)

    manifest = VectorManifest(
        case_id=case_id,
        inputs=records_by_section["inputs"],
        goldens=records_by_section["goldens"],
        metadata={
            "source": "trusted_local_pickle",
            "source_sha256": source_sha256,
            "bundle_id": resolved_bundle_id,
            "execution_ready": bool(inputs),
            "reference_priority": (
                "provided_golden"
                if goldens
                else "onnxruntime_fallback_required"
            ),
            "source_format": effective_format,
            "section": selected_section,
            **(
                {"source_path": resolved_source_key}
                if resolved_source_key is not None
                else {}
            ),
        },
    )
    manifest_path = VectorPreparer._write_manifest(  # noqa: SLF001
        output / "vector_manifest.json",
        manifest,
    )
    return ImportedPickleArtifacts(
        bundle=bundle,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        execution_ready=bool(inputs),
        source_format=effective_format,
        section=selected_section,
    )


__all__ = [
    "ImportedPickleArtifacts",
    "RestrictedUnpickler",
    "detect_pickle_source_format",
    "import_pickle_artifacts",
    "safe_load_pickle",
]
