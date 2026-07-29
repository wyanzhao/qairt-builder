"""Versioned worker-harness constraints.

All host/worker compatibility pins live in ``harness/constraints.json``.
Keeping these values in data, rather than scattered Python and Dockerfile
constants, makes a QAIRT or runtime upgrade one reviewable change.  A project
may select another file with ``QAIRT_AGENT_HARNESS_CONSTRAINTS``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from qairt_agent.errors import InvalidSpecError


DEFAULT_CONSTRAINTS_LOGICAL_PATH = "harness/constraints.json"
ENV_HARNESS_CONSTRAINTS = "QAIRT_AGENT_HARNESS_CONSTRAINTS"
_VERSION_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)")
_ACTIVE_CONSTRAINTS_PATH: ContextVar[Path | None] = ContextVar(
    "qairt_agent_active_constraints_path",
    default=None,
)


class HarnessConstraintsError(InvalidSpecError):
    """Raised when the harness constraints are missing or malformed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, stage="harness")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessConstraintsError(f"{name} must be a JSON object")
    return value


def _require_string(section: Mapping[str, Any], key: str, name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HarnessConstraintsError(f"{name}.{key} must be a non-empty string")
    return value.strip()


def _require_int(section: Mapping[str, Any], key: str, name: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise HarnessConstraintsError(f"{name}.{key} must be an integer")
    return value


@dataclass(frozen=True)
class HarnessConstraints:
    """Validated compatibility contract shared by every worker backend."""

    source_path: Path
    schema_version: int
    qairt_version: str
    qairt_build_id: str
    ubuntu_version: str
    python_version: str
    platform: str
    worker_image: str
    dockerfile: str
    dependencies_file: str
    torch_version: str
    torch_index_url: str
    apple_container_version: str
    apple_container_host_alias: str
    docker_minimum_version: str
    docker_host_alias: str
    target_chipset: str
    target_dsp_arch: str
    target_soc_model: int

    @property
    def python_version_tuple(self) -> tuple[int, int]:
        parts = self.python_version.split(".")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise HarnessConstraintsError(
                "worker.python_version must use major.minor form"
            )
        return int(parts[0]), int(parts[1])

    @property
    def platform_arch(self) -> str:
        os_name, separator, arch = self.platform.partition("/")
        if os_name != "linux" or not separator or not arch:
            raise HarnessConstraintsError(
                "worker.platform must use linux/<architecture> form"
            )
        return arch

    @classmethod
    def from_dict(
        cls,
        document: Mapping[str, Any],
        *,
        source_path: Path,
    ) -> "HarnessConstraints":
        schema_version = document.get("schema_version")
        if schema_version != 1:
            raise HarnessConstraintsError(
                f"unsupported harness schema_version {schema_version!r}; expected 1"
            )
        qairt = _require_mapping(document.get("qairt"), "qairt")
        worker = _require_mapping(document.get("worker"), "worker")
        runtime_cli = _require_mapping(document.get("runtime_cli"), "runtime_cli")
        apple = _require_mapping(
            runtime_cli.get("apple_container"), "runtime_cli.apple_container"
        )
        docker = _require_mapping(runtime_cli.get("docker"), "runtime_cli.docker")
        target = _require_mapping(document.get("target"), "target")
        constraints = cls(
            source_path=source_path,
            schema_version=1,
            qairt_version=_require_string(qairt, "version", "qairt"),
            qairt_build_id=_require_string(qairt, "build_id", "qairt"),
            ubuntu_version=_require_string(
                worker, "ubuntu_version", "worker"
            ),
            python_version=_require_string(
                worker, "python_version", "worker"
            ),
            platform=_require_string(worker, "platform", "worker"),
            worker_image=_require_string(worker, "image", "worker"),
            dockerfile=_require_string(worker, "dockerfile", "worker"),
            dependencies_file=_require_string(
                worker, "dependencies_file", "worker"
            ),
            torch_version=_require_string(worker, "torch_version", "worker"),
            torch_index_url=_require_string(
                worker, "torch_index_url", "worker"
            ),
            apple_container_version=_require_string(
                apple, "version", "runtime_cli.apple_container"
            ),
            apple_container_host_alias=_require_string(
                apple, "host_alias", "runtime_cli.apple_container"
            ),
            docker_minimum_version=_require_string(
                docker, "minimum_version", "runtime_cli.docker"
            ),
            docker_host_alias=_require_string(
                docker, "host_alias", "runtime_cli.docker"
            ),
            target_chipset=_require_string(target, "chipset", "target"),
            target_dsp_arch=_require_string(target, "dsp_arch", "target"),
            target_soc_model=_require_int(target, "soc_model", "target"),
        )
        constraints.python_version_tuple
        constraints.platform_arch
        for name, value in (
            ("qairt.version", constraints.qairt_version),
            ("runtime_cli.apple_container.version", constraints.apple_container_version),
            ("runtime_cli.docker.minimum_version", constraints.docker_minimum_version),
            ("worker.torch_version", constraints.torch_version),
        ):
            if parse_version(value) is None:
                raise HarnessConstraintsError(
                    f"{name} must contain a semantic x.y.z version"
                )
        return constraints


def parse_version(value: str) -> tuple[int, int, int] | None:
    """Extract the first semantic ``major.minor.patch`` tuple from text."""

    match = _VERSION_RE.search(value)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _source_tree_constraints_path() -> Path:
    return Path(__file__).resolve().parents[2] / DEFAULT_CONSTRAINTS_LOGICAL_PATH


def default_constraints_path() -> Path:
    """Return the editable source-tree file or the bundled wheel resource."""

    source_tree = _source_tree_constraints_path()
    if source_tree.is_file():
        return source_tree
    return Path(__file__).resolve().parent / "_data" / "harness_constraints.json"


def load_harness_constraints(
    path: str | Path | None = None,
) -> HarnessConstraints:
    """Load and validate one harness constraints document."""

    selected = (
        path
        or _ACTIVE_CONSTRAINTS_PATH.get()
        or os.environ.get(ENV_HARNESS_CONSTRAINTS)
    )
    resolved = (
        Path(selected).expanduser().resolve()
        if selected
        else default_constraints_path().resolve()
    )
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HarnessConstraintsError(
            f"harness constraints not found at {resolved}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise HarnessConstraintsError(
            f"invalid harness constraints JSON at {resolved}: {exc}"
        ) from exc
    return HarnessConstraints.from_dict(
        _require_mapping(document, "constraints"),
        source_path=resolved,
    )


@contextmanager
def use_harness_constraints(
    path: str | Path,
) -> Iterator[HarnessConstraints]:
    """Activate one project harness for the current context and child tasks."""

    resolved = Path(path).expanduser().resolve()
    constraints = load_harness_constraints(resolved)
    token = _ACTIVE_CONSTRAINTS_PATH.set(resolved)
    try:
        yield constraints
    finally:
        _ACTIVE_CONSTRAINTS_PATH.reset(token)


def install_default_constraints(target: str | Path) -> Path:
    """Copy the checked-in defaults into a newly initialized project."""

    destination = Path(target).expanduser().resolve()
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(default_constraints_path(), destination)
    return destination


DEFAULT_CONSTRAINTS = load_harness_constraints()


__all__ = [
    "DEFAULT_CONSTRAINTS",
    "DEFAULT_CONSTRAINTS_LOGICAL_PATH",
    "ENV_HARNESS_CONSTRAINTS",
    "HarnessConstraints",
    "HarnessConstraintsError",
    "default_constraints_path",
    "install_default_constraints",
    "load_harness_constraints",
    "parse_version",
    "use_harness_constraints",
]
