"""Stage-key derivation for journal deduplication and reuse.

A stage key captures everything that determines a stage's outputs: its inputs,
the resolved preset, the SDK build, the adapter capability, the Docker image
digest, and the platform ABI.  Device stages additionally fold in a
device/runtime fingerprint.  A rerun may reuse a prior stage's outputs only
when the key matches exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from qairt_agent.artifacts import canonical_json_bytes, sha256_file
from qairt_agent.contracts import ArtifactRef
from qairt_agent.errors import ArtifactIntegrityError, ArtifactNotFoundError
from qairt_agent.families.inspector import OnnxInspector

_OUTPUT_ONLY_FIELDS = {
    "artifact_root",
    "cache_dir",
    "destination",
    "jobs_root",
    "output_dir",
    "output_dlc",
    "output_file",
    "output_path",
    "output_root",
    "report_path",
    "work_dir",
    "workdir",
}
_SDK_FIELDS = {
    "qairt_root",
    "qairt_sdk_root",
    "qnn_root",
    "qnn_sdk_root",
    "sdk_dir",
    "sdk_path",
    "sdk_root",
}
_PATH_FIELDS = {
    "actual_manifest",
    "actual_trace",
    "aimet_config_path",
    "baseline_trace",
    "calibration_manifest",
    "config_path",
    "context_path",
    "encodings_path",
    "model_path",
    "onnx_path",
    "path",
    "reference_manifest",
    "reference_trace",
    "tokenizer_path",
    "validation_manifest",
    "vector_manifest",
}
_IGNORED_DIRECTORY_NAMES = {".git", ".mypy_cache", ".pytest_cache", "__pycache__"}
_CONFIG_DIRECTORY_SUFFIXES = {
    ".json",
    ".model",
    ".textproto",
    ".tiktoken",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def hash_inputs(refs: Iterable[ArtifactRef | str]) -> str:
    """Stable hash over a set of input artifact SHAs (or raw sha strings)."""

    shas: list[str] = []
    for ref in refs:
        shas.append(ref.sha256 if isinstance(ref, ArtifactRef) else str(ref))
    payload = {"inputs_sha256": sorted(shas)}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _field_name(key_path: tuple[str, ...]) -> str:
    return key_path[-1].lower() if key_path else ""


def _is_output_field(key_path: tuple[str, ...]) -> bool:
    return _field_name(key_path) in _OUTPUT_ONLY_FIELDS


def _is_sdk_field(key_path: tuple[str, ...]) -> bool:
    field = _field_name(key_path)
    return field in _SDK_FIELDS or field.endswith("_sdk_root")


def _is_path_field(key_path: tuple[str, ...]) -> bool:
    field = _field_name(key_path)
    return (
        field in _PATH_FIELDS
        or field.endswith(("_file", "_manifest", "_path", "_trace"))
    )


def _looks_like_sdk_directory(path: Path) -> bool:
    """Recognize SDK roots without recursively walking their very large trees."""

    if (path / "sdk.yaml").is_file():
        return True
    qairt_dir = path / "qairt"
    if not qairt_dir.is_dir():
        return False
    try:
        return any((version / "sdk.yaml").is_file() for version in qairt_dir.iterdir())
    except OSError:
        return False


def _resolve_path(value: str | os.PathLike[str], base_dir: Path | None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _missing_path_identity(path: Path) -> dict[str, Any]:
    return {"path": os.fspath(path), "exists": False}


def _raise_missing(path: Path) -> None:
    raise ArtifactNotFoundError(
        f"referenced input does not exist: {path}",
        details={"path": os.fspath(path)},
    )


def _verify_expected_sha(path: Path, actual_sha256: str, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    expected = expected_sha256.lower()
    if actual_sha256 != expected:
        raise ArtifactIntegrityError(
            f"referenced input hash does not match its manifest: {path}",
            details={
                "path": os.fspath(path),
                "expected_sha256": expected,
                "actual_sha256": actual_sha256,
            },
        )


def _directory_identity(
    path: Path,
    *,
    key_path: tuple[str, ...],
    active_paths: set[Path],
) -> dict[str, Any]:
    """Hash a directory deterministically, excluding SDK and generated caches."""

    if _is_sdk_field(key_path) or _looks_like_sdk_directory(path):
        return {
            "path": os.fspath(path),
            "type": "sdk_directory",
            "content_identity": "sdk_build_provenance",
        }

    field = _field_name(key_path)
    config_only = field in {"config_path", "tokenizer_path", "aimet_config_path"}
    entries: list[dict[str, Any]] = []
    for root, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if name not in _IGNORED_DIRECTORY_NAMES
        )
        root_path = Path(root)
        for filename in sorted(filenames):
            candidate = root_path / filename
            if config_only and candidate.suffix.lower() not in _CONFIG_DIRECTORY_SUFFIXES:
                continue
            relative = candidate.relative_to(path).as_posix()
            if candidate.is_symlink():
                entries.append(
                    {
                        "relative_path": relative,
                        "type": "symlink",
                        "target": os.readlink(candidate),
                    }
                )
                continue
            if not candidate.is_file():
                continue
            sha256, size_bytes = sha256_file(candidate)
            entries.append(
                {
                    "relative_path": relative,
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                }
            )

    payload = {"entries": entries}
    return {
        "path": os.fspath(path),
        "type": "directory",
        "sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        "file_count": len(entries),
        "size_bytes": sum(int(entry.get("size_bytes", 0)) for entry in entries),
    }


def _is_manifest_path(path: Path, key_path: tuple[str, ...]) -> bool:
    field = _field_name(key_path)
    return "manifest" in field or ".manifest." in path.name.lower()


def _expected_sha_for(mapping: Mapping[Any, Any], key: str) -> str | None:
    if key == "path":
        value = mapping.get("sha256")
    else:
        value = mapping.get(f"{key}_sha256")
        if value is None and key.endswith("_path"):
            value = mapping.get(f"{key[:-5]}_sha256")
    if not isinstance(value, str) or len(value) != 64:
        return None
    try:
        int(value, 16)
    except ValueError:
        return None
    return value.lower()


def _file_identity(
    path: Path,
    *,
    key_path: tuple[str, ...],
    expected_sha256: str | None,
    active_paths: set[Path],
) -> dict[str, Any]:
    if path in active_paths:
        return {"path": os.fspath(path), "cycle": True}

    sha256, size_bytes = sha256_file(path)
    _verify_expected_sha(path, sha256, expected_sha256)
    identity: dict[str, Any] = {
        "path": os.fspath(path),
        "sha256": sha256,
        "size_bytes": size_bytes,
    }

    active_paths.add(path)
    try:
        if path.suffix.lower() == ".onnx":
            try:
                external_paths = OnnxInspector().external_data_paths(path)
            except Exception as exc:  # noqa: BLE001 - malformed ONNX must fail closed
                raise ArtifactIntegrityError(
                    f"failed to inspect ONNX external data: {path}",
                    details={"path": os.fspath(path), "reason": str(exc)},
                ) from exc
            identity["external_data"] = [
                _path_identity(
                    _require_external_data(external_path),
                    key_path=key_path + (f"external_data_{index:03d}",),
                    expected_sha256=None,
                    active_paths=active_paths,
                )
                for index, external_path in enumerate(external_paths)
            ]

        if _is_manifest_path(path, key_path):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ArtifactIntegrityError(
                    f"referenced manifest is not valid JSON: {path}",
                    details={"path": os.fspath(path), "reason": str(exc)},
                ) from exc
            identity["referenced_content"] = _content_identity(
                payload,
                key_path=key_path + ("manifest_content",),
                base_dir=path.parent,
                expected_sha256=None,
                active_paths=active_paths,
            )
    finally:
        active_paths.remove(path)
    return identity


def _require_external_data(path: Path) -> Path:
    if not path.is_file():
        _raise_missing(path)
    return path


def _path_identity(
    path: Path,
    *,
    key_path: tuple[str, ...],
    expected_sha256: str | None,
    active_paths: set[Path],
) -> dict[str, Any]:
    if _is_output_field(key_path):
        return {"path": os.fspath(path), "content_identity": "output_only"}
    if not path.exists():
        if expected_sha256 is not None:
            _raise_missing(path)
        return _missing_path_identity(path)
    if path.is_dir():
        identity = _directory_identity(
            path,
            key_path=key_path,
            active_paths=active_paths,
        )
        if expected_sha256 is not None and "sha256" in identity:
            _verify_expected_sha(path, str(identity["sha256"]), expected_sha256)
        return identity
    if not path.is_file():
        _raise_missing(path)
    return _file_identity(
        path,
        key_path=key_path,
        expected_sha256=expected_sha256,
        active_paths=active_paths,
    )


def _content_identity(
    value: Any,
    *,
    key_path: tuple[str, ...],
    base_dir: Path | None,
    expected_sha256: str | None,
    active_paths: set[Path],
) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _content_identity(
            value.model_dump(mode="python", exclude_none=True),
            key_path=key_path,
            base_dir=base_dir,
            expected_sha256=expected_sha256,
            active_paths=active_paths,
        )
    if isinstance(value, os.PathLike):
        path = _resolve_path(value, base_dir)
        return _path_identity(
            path,
            key_path=key_path,
            expected_sha256=expected_sha256,
            active_paths=active_paths,
        )
    if isinstance(value, str):
        if _is_output_field(key_path) or _is_sdk_field(key_path):
            if _is_sdk_field(key_path):
                try:
                    sdk_path = _resolve_path(value, base_dir)
                    if sdk_path.is_dir():
                        return _path_identity(
                            sdk_path,
                            key_path=key_path,
                            expected_sha256=expected_sha256,
                            active_paths=active_paths,
                        )
                except (OSError, ValueError):
                    pass
            return value
        if _is_path_field(key_path):
            try:
                path = _resolve_path(value, base_dir)
                return _path_identity(
                    path,
                    key_path=key_path,
                    expected_sha256=expected_sha256,
                    active_paths=active_paths,
                )
            except (OSError, ValueError):
                pass
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            key = str(raw_key)
            normalized[key] = _content_identity(
                item,
                key_path=key_path + (key,),
                base_dir=base_dir,
                expected_sha256=_expected_sha_for(value, key),
                active_paths=active_paths,
            )
        return normalized
    if isinstance(value, (tuple, list)):
        return [
            _content_identity(
                item,
                key_path=key_path + (str(index),),
                base_dir=base_dir,
                expected_sha256=None,
                active_paths=active_paths,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        normalized = [
            _content_identity(
                item,
                key_path=key_path + ("set_item",),
                base_dir=base_dir,
                expected_sha256=None,
                active_paths=active_paths,
            )
            for item in value
        ]
        return sorted(normalized, key=canonical_json_bytes)
    return str(value)


def content_identity(
    value: Any,
    *,
    project_root: str | os.PathLike[str] | None = None,
) -> Any:
    """Canonicalize config/spec data and replace input paths with byte identities.

    Existing files are hashed from their bytes. ONNX files additionally include
    every referenced external-data file, and JSON manifests include the files
    they reference. Generated output roots and QAIRT SDK directories are never
    recursively hashed.
    """

    base_dir = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else None
    )
    return _content_identity(
        value,
        key_path=(),
        base_dir=base_dir,
        expected_sha256=None,
        active_paths=set(),
    )


def compute_stage_key(
    *,
    stage_name: str,
    inputs_sha256: str,
    resolved_preset_sha256: str,
    sdk_build: str,
    adapter_capability: str,
    platform_abi: str,
    image_digest: str | None = None,
    device_fingerprint: str | None = None,
) -> str:
    """Derive the deterministic stage key used for reuse and provenance."""

    payload = {
        "stage": stage_name,
        "inputs_sha256": inputs_sha256,
        "resolved_preset_sha256": resolved_preset_sha256,
        "sdk_build": sdk_build,
        "adapter_capability": adapter_capability,
        "image_digest": image_digest,
        "platform_abi": platform_abi,
        "device_fingerprint": device_fingerprint,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


__all__ = ["compute_stage_key", "content_identity", "hash_inputs"]
