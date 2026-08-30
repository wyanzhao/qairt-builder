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
TARGET_REGISTRY_DIRNAME = "targets"
ENV_HARNESS_CONSTRAINTS = "QAIRT_AGENT_HARNESS_CONSTRAINTS"
ENV_TARGET_ACCEPTANCE = "QAIRT_AGENT_TARGET_ACCEPTANCE"
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
    target_name: str

    @property
    def targets_dir(self) -> Path:
        """The reviewed target registry, resolved beside the constraints file."""

        return self.source_path.parent / TARGET_REGISTRY_DIRNAME

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
            target_name=_require_string(target, "name", "target"),
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


def default_targets_dir() -> Path:
    """Return the checked-in target registry beside the default constraints."""

    return default_constraints_path().parent / TARGET_REGISTRY_DIRNAME


def install_default_constraints(target: str | Path) -> Path:
    """Copy the checked-in defaults into a newly initialized project.

    The target registry travels with the constraints file: a project whose
    constraints name a target but whose registry is missing cannot resolve it,
    so both are installed together.
    """

    destination = Path(target).expanduser().resolve()
    registry_source = default_targets_dir()
    registry_destination = destination.parent / TARGET_REGISTRY_DIRNAME
    if registry_source.is_dir():
        registry_destination.mkdir(parents=True, exist_ok=True)
        for entry in sorted(registry_source.glob("*.json")):
            installed = registry_destination / entry.name
            if not installed.exists():
                shutil.copyfile(entry, installed)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(default_constraints_path(), destination)
    return destination


@dataclass(frozen=True)
class TargetEntry:
    """One reviewed HTP target from ``harness/targets/<name>.json``.

    ``soc_model`` is the ``Qnn_SocModel_t`` value the compiler consumes.
    ``soc_id`` is the Android SoC ID a device reports at
    ``/sys/devices/soc0/soc_id`` and is carried only so the two numbering
    schemes can never be conflated again -- it is never passed to the SDK.
    """

    source_path: Path
    name: str
    chipset: str
    dsp_arch: str
    soc_model: int
    soc_id: tuple[int, ...]
    notes: tuple[str, ...]
    verified: Mapping[str, Any] | None

    @property
    def tuple_text(self) -> str:
        return f"{self.chipset}/{self.dsp_arch}/soc_model {self.soc_model}"

    def matches(self, chipset: str, dsp_arch: str, soc_model: int) -> bool:
        return (
            str(chipset).upper() == self.chipset.upper()
            and str(dsp_arch).lower() == self.dsp_arch.lower()
            and int(soc_model) == self.soc_model
        )

    @classmethod
    def from_dict(
        cls,
        document: Mapping[str, Any],
        *,
        source_path: Path,
    ) -> "TargetEntry":
        if document.get("schema_version") != 1:
            raise HarnessConstraintsError(
                f"unsupported target schema_version in {source_path}; expected 1"
            )
        raw_soc_id = document.get("soc_id", ())
        if isinstance(raw_soc_id, int):
            raw_soc_id = (raw_soc_id,)
        if not isinstance(raw_soc_id, (list, tuple)):
            raise HarnessConstraintsError(
                f"target.soc_id in {source_path} must be an integer or a list"
            )
        verified = document.get("verified")
        if verified is not None and not isinstance(verified, Mapping):
            raise HarnessConstraintsError(
                f"target.verified in {source_path} must be an object or null"
            )
        return cls(
            source_path=source_path,
            name=_require_string(document, "name", "target"),
            chipset=_require_string(document, "chipset", "target"),
            dsp_arch=_require_string(document, "dsp_arch", "target"),
            soc_model=_require_int(document, "soc_model", "target"),
            soc_id=tuple(int(value) for value in raw_soc_id),
            notes=tuple(str(item) for item in document.get("notes", ())),
            verified=dict(verified) if verified is not None else None,
        )


def load_target_registry(
    constraints: "HarnessConstraints | None" = None,
) -> dict[str, TargetEntry]:
    """Load every reviewed target beside the active constraints file."""

    active = constraints or load_harness_constraints()
    directory = active.targets_dir
    if not directory.is_dir():
        raise HarnessConstraintsError(
            f"target registry directory not found at {directory}"
        )
    registry: dict[str, TargetEntry] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessConstraintsError(
                f"invalid target registry entry {path}: {exc}"
            ) from exc
        entry = TargetEntry.from_dict(
            _require_mapping(document, "target"), source_path=path
        )
        if entry.name != path.stem:
            raise HarnessConstraintsError(
                f"target registry entry {path} declares name {entry.name!r}; "
                "the file name and the declared name must agree"
            )
        if entry.name in registry:
            raise HarnessConstraintsError(f"duplicate target entry {entry.name!r}")
        registry[entry.name] = entry
    if not registry:
        raise HarnessConstraintsError(
            f"target registry at {directory} contains no entries"
        )
    return registry


def resolve_target(
    name: str | None = None,
    *,
    constraints: "HarnessConstraints | None" = None,
) -> TargetEntry:
    """Resolve one registered target by name; there is no implicit fallback."""

    active = constraints or load_harness_constraints()
    registry = load_target_registry(active)
    selected = name or active.target_name
    entry = registry.get(str(selected).strip().lower())
    if entry is None:
        raise HarnessConstraintsError(
            f"unregistered target {selected!r}; reviewed targets are "
            f"{sorted(registry)}"
        )
    return entry


def resolve_target_tuple(
    chipset: str,
    dsp_arch: str,
    soc_model: int,
    *,
    constraints: "HarnessConstraints | None" = None,
) -> TargetEntry:
    """Resolve an inline tuple, which must match a registered target exactly."""

    registry = load_target_registry(constraints)
    for entry in registry.values():
        if entry.matches(chipset, dsp_arch, soc_model):
            return entry
    raise HarnessConstraintsError(
        f"target {chipset}/{dsp_arch}/soc_model {soc_model} is not a reviewed "
        "target; registered targets are "
        + ", ".join(entry.tuple_text for entry in registry.values())
    )


def acceptance_target_name() -> str | None:
    """The target an explicitly declared acceptance run is qualifying, if any."""

    value = os.environ.get(ENV_TARGET_ACCEPTANCE, "").strip().lower()
    return value or None


def require_verified_target(entry: TargetEntry) -> TargetEntry:
    """Refuse a target that has never been proven on real hardware.

    A target cannot become verified without a run, and a run is refused while
    the target is unverified, so the acceptance run itself is the one explicit
    exception: set ``QAIRT_AGENT_TARGET_ACCEPTANCE`` to the exact target name.
    Naming it is deliberate -- it cannot be switched on by accident, it applies
    to one target only, and preflight records that the run was qualifying.
    """

    if entry.verified is not None:
        return entry
    if acceptance_target_name() == entry.name:
        return entry
    raise HarnessConstraintsError(
        f"target {entry.name!r} ({entry.tuple_text}) has no verified block: "
        "it has never completed a real-device acceptance run, so build and "
        f"device stages are refused. Record a completed run in "
        f"{entry.source_path}, or qualify the target now by setting "
        f"{ENV_TARGET_ACCEPTANCE}={entry.name}."
    )


DEFAULT_CONSTRAINTS = load_harness_constraints()


__all__ = [
    "DEFAULT_CONSTRAINTS",
    "DEFAULT_CONSTRAINTS_LOGICAL_PATH",
    "ENV_HARNESS_CONSTRAINTS",
    "ENV_TARGET_ACCEPTANCE",
    "TARGET_REGISTRY_DIRNAME",
    "HarnessConstraints",
    "HarnessConstraintsError",
    "TargetEntry",
    "acceptance_target_name",
    "default_constraints_path",
    "default_targets_dir",
    "install_default_constraints",
    "load_harness_constraints",
    "load_target_registry",
    "parse_version",
    "require_verified_target",
    "resolve_target",
    "resolve_target_tuple",
    "use_harness_constraints",
]
