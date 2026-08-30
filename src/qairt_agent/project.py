"""Project configuration, initialization, doctor, and logical URIs.

A project is rooted at a directory containing ``qairt-agent.toml``.  The SDK
installation root defaults to ``./qnn/qnn`` and may either be an SDK itself or
contain a versioned ``qairt/<release>`` SDK.  Discovery is deterministic and
never moves the user's SDK.

Worker execution is explicit.  ``auto`` selects Apple ``container`` on macOS
and Docker on Linux.  Native execution is an explicit opt-in and doctor still
requires the pinned Ubuntu/amd64/Python ABI.
"""

from __future__ import annotations

import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qairt_agent.apple_container import AppleContainerRunner
from qairt_agent.docker import DockerRunner, WorkerImageConfig
from qairt_agent.docker.image import DEFAULT_IMAGE_REF, DEFAULT_PLATFORM
from qairt_agent.errors import ProjectNotInitializedError
from qairt_agent.harness import (
    DEFAULT_CONSTRAINTS,
    DEFAULT_CONSTRAINTS_LOGICAL_PATH,
    HarnessConstraints,
    HarnessConstraintsError,
    TargetEntry,
    install_default_constraints,
    load_harness_constraints,
    resolve_target,
)
from qairt_agent.qairt_adapter.preflight import (
    ACTIVE_TARGET,
    PINNED_QAIRT_BUILD_ID,
)
from qairt_agent.worker_scaffold import (
    ensure_worker_build_context,
    worker_build_context_issues,
)

def _target_fields(entry: TargetEntry) -> dict[str, Any]:
    """Project-config fields for one registered target."""

    return {
        "target_name": entry.name,
        "target_chipset": entry.chipset,
        "target_dsp_arch": entry.dsp_arch,
        "target_soc_model": entry.soc_model,
    }


CONFIG_FILENAME = "qairt-agent.toml"
DEFAULT_SDK_ROOT = "./qnn/qnn"
DEFAULT_WORKER_BACKEND = "auto"
DEFAULT_WORKER_DOCKERFILE = DEFAULT_CONSTRAINTS.dockerfile
WORKER_BACKENDS = frozenset({"apple_container", "auto", "docker", "native"})


# --------------------------------------------------------------------------- #
# Minimal TOML subset (flat tables of str/int/bool/array) — no dependency.
# --------------------------------------------------------------------------- #


def _dump_toml(sections: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for section, values in sections.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _load_toml(text: str) -> dict[str, dict[str, Any]]:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _load_toml_minimal(text)
    parsed = tomllib.loads(text)
    return {section: dict(values) for section, values in parsed.items()}


def _load_toml_minimal(text: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = {}
            result[line[1:-1].strip()] = current
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        current[key.strip()] = _parse_toml_scalar(value.strip())
    return result


def _parse_toml_scalar(token: str) -> Any:
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_toml_scalar(part.strip()) for part in inner.split(",")]
    if token in {"true", "false"}:
        return token == "true"
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    try:
        return int(token)
    except ValueError:
        return token


# --------------------------------------------------------------------------- #
# Project config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved project configuration (paths are logical, relative to root)."""

    project_root: Path
    harness_constraints: str = DEFAULT_CONSTRAINTS_LOGICAL_PATH
    sdk_root: str = DEFAULT_SDK_ROOT
    jobs_dir: str = ".qairt-agent/jobs"
    state_dir: str = ".qairt-agent/state"
    artifacts_dir: str = "artifacts"
    cache_dir: str = ".qairt-agent/cache"
    worker_backend: str = DEFAULT_WORKER_BACKEND
    docker_image: str = DEFAULT_IMAGE_REF
    docker_platform: str = DEFAULT_PLATFORM
    dockerfile: str = DEFAULT_WORKER_DOCKERFILE
    target_name: str = ACTIVE_TARGET.name
    target_chipset: str = ACTIVE_TARGET.chipset
    target_dsp_arch: str = ACTIVE_TARGET.dsp_arch
    target_soc_model: int = ACTIVE_TARGET.soc_model

    def abs(self, logical: str) -> Path:
        return (self.project_root / logical).resolve()

    @property
    def sdk_path(self) -> Path:
        return discover_sdk_path(self.abs(self.sdk_root), constraints=self.harness)

    @property
    def harness_path(self) -> Path:
        resolved = self.abs(self.harness_constraints)
        if not resolved.is_relative_to(self.project_root.resolve()):
            raise HarnessConstraintsError(
                "harness.constraints must resolve inside the project root: "
                f"{self.harness_constraints!r}"
            )
        return resolved

    @property
    def harness(self) -> HarnessConstraints:
        # A configured compatibility contract is part of the build identity.
        # Never fall back to repository defaults when it is missing: doing so
        # could silently run a job with different QAIRT/runtime pins.
        return load_harness_constraints(self.harness_path)

    @property
    def jobs_path(self) -> Path:
        return self.abs(self.jobs_dir)

    @property
    def state_path(self) -> Path:
        return self.abs(self.state_dir)

    @property
    def artifacts_path(self) -> Path:
        return self.abs(self.artifacts_dir)

    @property
    def cache_path(self) -> Path:
        return self.abs(self.cache_dir)

    @property
    def models_path(self) -> Path:
        return self.abs("models")

    @property
    def dockerfile_path(self) -> Path:
        return self.abs(self.dockerfile)

    @property
    def effective_worker_backend(self) -> str:
        return select_worker_backend(self.worker_backend)

    def to_toml(self) -> str:
        sections: dict[str, dict[str, Any]] = {
            "project": {"sdk_root": self.sdk_root},
            "harness": {"constraints": self.harness_constraints},
            "worker": {"backend": self.worker_backend},
            "state": {
                "jobs_dir": self.jobs_dir,
                "state_dir": self.state_dir,
                "artifacts_dir": self.artifacts_dir,
                "cache_dir": self.cache_dir,
            },
            "target": {
                "name": self.target_name,
            },
        }
        constraints = self.harness
        docker_overrides: dict[str, Any] = {}
        if self.docker_image != constraints.worker_image:
            docker_overrides["image"] = self.docker_image
        if self.docker_platform != constraints.platform:
            docker_overrides["platform"] = self.docker_platform
        if self.dockerfile != constraints.dockerfile:
            docker_overrides["dockerfile"] = self.dockerfile
        if docker_overrides:
            # Legacy table name retained for backwards compatibility.  New
            # projects keep all default pins solely in harness constraints.
            sections["docker"] = docker_overrides
        return _dump_toml(sections)

    @classmethod
    def from_toml(cls, project_root: Path, text: str) -> "ProjectConfig":
        data = _load_toml(text)
        project = data.get("project", {})
        harness_section = data.get("harness", {})
        worker = data.get("worker", {})
        state = data.get("state", {})
        docker = data.get("docker", {})
        target = data.get("target", {})
        backend = str(worker.get("backend", DEFAULT_WORKER_BACKEND)).strip().lower()
        if backend not in WORKER_BACKENDS:
            raise ValueError(
                f"worker.backend must be one of {sorted(WORKER_BACKENDS)}; got {backend!r}"
            )
        harness_logical = str(
            harness_section.get(
                "constraints", DEFAULT_CONSTRAINTS_LOGICAL_PATH
            )
        )
        harness_path = (project_root / harness_logical).resolve()
        if not harness_path.is_relative_to(project_root.resolve()):
            raise HarnessConstraintsError(
                "harness.constraints must resolve inside the project root: "
                f"{harness_logical!r}"
            )
        constraints = load_harness_constraints(harness_path)
        return cls(
            project_root=project_root,
            harness_constraints=harness_logical,
            sdk_root=str(project.get("sdk_root", DEFAULT_SDK_ROOT)),
            worker_backend=backend,
            jobs_dir=str(state.get("jobs_dir", ".qairt-agent/jobs")),
            state_dir=str(state.get("state_dir", ".qairt-agent/state")),
            artifacts_dir=str(state.get("artifacts_dir", "artifacts")),
            cache_dir=str(state.get("cache_dir", ".qairt-agent/cache")),
            docker_image=str(docker.get("image", constraints.worker_image)),
            docker_platform=str(docker.get("platform", constraints.platform)),
            dockerfile=str(docker.get("dockerfile", constraints.dockerfile)),
            **_target_fields(
                resolve_target(
                    target.get("name", constraints.target_name),
                    constraints=constraints,
                )
            ),
        )


def config_path(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / CONFIG_FILENAME


def find_project_root(start: str | Path) -> Path | None:
    """Find the nearest initialized project at or above ``start``."""

    candidate = Path(start).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / CONFIG_FILENAME).is_file():
            return directory
    return None


def select_container_backend(*, system_name: str | None = None) -> str:
    """Select the supported host container runtime without mutating the host."""

    host_system = (system_name or platform.system()).strip().lower()
    if host_system == "darwin":
        return "apple_container"
    if host_system == "linux":
        return "docker"
    raise ValueError(
        f"no automatic worker container backend for host {host_system!r}; "
        "configure worker.backend explicitly"
    )


def select_worker_backend(
    configured: str,
    *,
    native_abi: bool | None = None,
    system_name: str | None = None,
) -> str:
    """Resolve ``auto`` to the host container runtime.

    Native execution is opt-in.  This keeps Linux development reproducible
    even when a developer machine happens to match the pinned ABI.
    ``native_abi`` remains accepted for API compatibility but does not change
    automatic selection.
    """

    normalized = configured.strip().lower()
    if normalized not in WORKER_BACKENDS:
        raise ValueError(
            f"worker backend must be one of {sorted(WORKER_BACKENDS)}; got {configured!r}"
        )
    if normalized != "auto":
        return normalized
    del native_abi
    return select_container_backend(system_name=system_name)


def init(project_root: str | Path, *, exist_ok: bool = True) -> ProjectConfig:
    """Initialize a project and its self-contained worker image context.

    Never moves an existing SDK and never installs a container runtime.  The
    SDK is expected at ``./qnn/qnn``; if it is absent, doctor reports it but
    init still succeeds so the control plane can be prepared ahead of the SDK.
    Worker assets come from the exact editable checkout or installed wheel
    running this command; arbitrary project ``pyproject.toml``/``src`` files
    are never treated as qairt-agent image inputs.
    """

    root = Path(project_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    constraints_path = (
        root / DEFAULT_CONSTRAINTS_LOGICAL_PATH
    ).resolve()
    install_default_constraints(constraints_path)
    constraints = load_harness_constraints(constraints_path)
    config = ProjectConfig(
        project_root=root,
        docker_image=constraints.worker_image,
        docker_platform=constraints.platform,
        dockerfile=constraints.dockerfile,
        **_target_fields(resolve_target(constraints=constraints)),
    )

    target = config_path(root)
    if target.exists() and not exist_ok:
        raise FileExistsError(f"{CONFIG_FILENAME} already exists at {target}")
    if target.exists():
        config = ProjectConfig.from_toml(root, target.read_text(encoding="utf-8"))
    else:
        target.write_text(config.to_toml(), encoding="utf-8")

    for logical in (
        config.jobs_dir,
        config.state_dir,
        config.cache_dir,
        config.artifacts_dir,
        "models",
    ):
        (root / logical).mkdir(parents=True, exist_ok=True)
    ensure_worker_build_context(root, config.harness)
    return config


def load(project_root: str | Path) -> ProjectConfig:
    root = Path(project_root).expanduser().resolve()
    target = root / CONFIG_FILENAME
    if not target.exists():
        raise ProjectNotInitializedError(
            f"project not initialized: {target} not found; run 'qairt-agent init'",
            stage="project",
            details={"project_root": str(root)},
        )
    return ProjectConfig.from_toml(root, target.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# logical URIs
# --------------------------------------------------------------------------- #


def to_logical_uri(path: str | Path, project_root: str | Path) -> str:
    """Return a project-relative logical URI for ``path``.

    Refuses to emit a host absolute path: ``path`` must live inside the project
    root.  The result is a posix relative path (e.g. ``models/qwen3/model.onnx``).
    """

    root = Path(project_root).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"path {resolved} is outside the project root {root}; cannot form a logical URI"
        ) from exc
    return relative.as_posix()


def resolve_logical_uri(uri: str, project_root: str | Path) -> Path:
    """Resolve a logical URI back to an absolute path under the project root."""

    root = Path(project_root).expanduser().resolve()
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    candidate = (root / uri).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"logical URI '{uri}' escapes the project root")
    return candidate


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*([^#]+?)\s*$", line)
        if match:
            values[match.group(1)] = match.group(2).strip().strip("'\"")
    return values


def discover_sdk_path(
    installation_root: str | Path,
    *,
    constraints: HarnessConstraints | None = None,
) -> Path:
    """Resolve a direct or versioned QAIRT SDK root.

    Qualcomm SDK archives commonly unpack as
    ``<installation_root>/qairt/<version>/sdk.yaml``.  Prefer an exact pinned
    build, then an exact pinned version, then the only discovered SDK.  The
    fallback is the installation root itself so doctor can report a precise
    missing-metadata error.
    """

    active = constraints or DEFAULT_CONSTRAINTS
    root = Path(installation_root).expanduser().resolve()
    if (root / "sdk.yaml").is_file():
        return root

    metadata_files = {
        *root.glob("qairt/*/sdk.yaml"),
        *root.glob("*/sdk.yaml"),
    }
    candidates: list[tuple[Path, dict[str, str]]] = []
    for metadata in sorted(metadata_files):
        try:
            candidates.append((metadata.parent, _parse_simple_yaml(metadata)))
        except OSError:
            continue

    exact_build = [
        path
        for path, meta in candidates
        if meta.get("version") == active.qairt_version
        and meta.get("build_id") == active.qairt_build_id
    ]
    if exact_build:
        return sorted(exact_build)[-1]

    exact_version = [
        path for path, meta in candidates if meta.get("version") == active.qairt_version
    ]
    if exact_version:
        return sorted(exact_version)[-1]
    if len(candidates) == 1:
        return candidates[0][0]
    return root


def _read_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip("'\"")
    return values


def _probe_docker(
    image_ref: str,
    *,
    platform_name: str = DEFAULT_PLATFORM,
    constraints: HarnessConstraints | None = None,
) -> tuple[bool, bool, str]:
    """Return daemon/image availability without mutating Docker state."""

    active = constraints or DEFAULT_CONSTRAINTS
    runner = DockerRunner(
        image=WorkerImageConfig(
            image_ref=image_ref,
            platform=platform_name,
            ubuntu_version=active.ubuntu_version,
            python_version=active.python_version,
        ),
        constraints=active,
    )
    try:
        runner.require_available()
    except Exception as exc:  # noqa: BLE001 - doctor reports a check
        return False, False, str(exc)
    try:
        runner.require_image()
    except Exception as exc:  # noqa: BLE001 - doctor reports a check
        return True, False, str(exc)
    return True, True, f"Docker daemon reachable; image {image_ref!r} found"


def _probe_apple_container(
    image_ref: str,
    *,
    platform_name: str = DEFAULT_PLATFORM,
    constraints: HarnessConstraints | None = None,
) -> tuple[bool, bool, str]:
    """Probe Apple ``container`` without starting services or pulling images."""

    active = constraints or DEFAULT_CONSTRAINTS
    runner = AppleContainerRunner(
        image=WorkerImageConfig(
            image_ref=image_ref,
            platform=platform_name,
            ubuntu_version=active.ubuntu_version,
            python_version=active.python_version,
        ),
        constraints=active,
    )
    try:
        runner.require_available()
    except Exception as exc:  # noqa: BLE001 - doctor reports a check
        return False, False, str(exc)
    try:
        runner.require_image()
    except Exception as exc:  # noqa: BLE001 - doctor reports a check
        return True, False, str(exc)
    return (
        True,
        True,
        f"Apple container services reachable; image {image_ref!r} found",
    )


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    message: str
    critical: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "message": self.message, "critical": self.critical}


def doctor(project_root: str | Path) -> dict[str, Any]:
    """Verify SDK metadata, QAIRT 2.49 capability, host ABI, and target.

    Honest by design: on a host without ``./qnn/qnn`` or the selected worker
    runtime the relevant checks report ``ok=False`` with guidance rather than
    failing silently. Live runtime/device acceptance starts only after those
    dependencies are prepared.
    """

    config = load(project_root)
    constraints = config.harness
    checks: list[DoctorCheck] = []
    checks.append(
        DoctorCheck(
            "harness_constraints",
            config.harness_path.is_file(),
            (
                f"harness constraints loaded from {config.harness_path}"
                if config.harness_path.is_file()
                else f"harness constraints missing at {config.harness_path}"
            ),
        )
    )

    sdk_path = config.sdk_path
    sdk_yaml = sdk_path / "sdk.yaml"
    if not config.abs(config.sdk_root).exists():
        checks.append(
            DoctorCheck(
                "sdk_present",
                False,
                f"SDK root not found at {config.sdk_root}; unpack QAIRT "
                f"{constraints.qairt_version} there "
                "(the framework never moves an existing SDK)",
            )
        )
    elif not sdk_yaml.exists():
        checks.append(
            DoctorCheck(
                "sdk_metadata",
                False,
                f"no supported sdk.yaml found under installation root {config.sdk_root}",
            )
        )
    else:
        meta = _parse_simple_yaml(sdk_yaml)
        version = meta.get("version", "")
        build_id = meta.get("build_id", "")
        version_ok = version.startswith(constraints.qairt_version)
        build_ok = build_id == constraints.qairt_build_id
        checks.append(
            DoctorCheck(
                "sdk_metadata",
                version_ok and build_ok,
                f"resolved {sdk_path}; sdk.yaml version={version or '?'} build_id={build_id or '?'} "
                f"(expected {constraints.qairt_version} / "
                f"{constraints.qairt_build_id})",
            )
        )
        python_api = sdk_path / "lib" / "python"
        checks.append(
            DoctorCheck(
                "qairt_capability",
                python_api.exists(),
                f"QAIRT Python API {'found' if python_api.exists() else 'missing'} at lib/python",
            )
        )

    os_release = _read_os_release()
    machine = platform.machine().lower()
    host_is_ubuntu = (
        os_release.get("ID") == "ubuntu"
        and os_release.get("VERSION_ID") == constraints.ubuntu_version
    )
    host_is_x86 = machine in {"x86_64", "amd64"}
    host_python = (sys.version_info.major, sys.version_info.minor)
    host_is_pinned_python = host_python == constraints.python_version_tuple
    backend = config.effective_worker_backend
    checks.append(
        DoctorCheck(
            "worker_backend",
            True,
            f"configured={config.worker_backend} resolved={backend}",
        )
    )
    checks.append(
        DoctorCheck(
            "host_abi",
            host_is_ubuntu and host_is_x86 and host_is_pinned_python,
            f"host os={os_release.get('ID', platform.system().lower())}/"
            f"{os_release.get('VERSION_ID', '?')} arch={machine} "
            f"python={host_python[0]}.{host_python[1]} "
            f"(worker target: ubuntu{constraints.ubuntu_version}/x86_64/"
            f"python{constraints.python_version}; incompatible macOS hosts use "
            "Apple container and incompatible Linux hosts use Docker)",
            critical=backend == "native",
        )
    )

    active_target = resolve_target(constraints=constraints)
    target_ok = (
        config.target_name == active_target.name
        and config.target_chipset == active_target.chipset
        and config.target_dsp_arch == active_target.dsp_arch
        and config.target_soc_model == active_target.soc_model
    )
    checks.append(
        DoctorCheck(
            "target",
            target_ok,
            f"target {config.target_name} "
            f"{config.target_chipset}/{config.target_dsp_arch}/soc_model "
            f"{config.target_soc_model} (expected {active_target.name} "
            f"{active_target.tuple_text})",
        )
    )

    if backend == "docker":
        build_context_issues = worker_build_context_issues(
            config.project_root,
            constraints,
        )
        docker_ok, image_ok, docker_message = _probe_docker(
            config.docker_image,
            platform_name=config.docker_platform,
            constraints=constraints,
        )
        checks.extend(
            (
                DoctorCheck(
                    "docker",
                    docker_ok,
                    docker_message,
                ),
                DoctorCheck(
                    "docker_image",
                    image_ok,
                    docker_message
                    if image_ok
                    else f"{docker_message}; run 'qairt-agent image build --root {config.project_root}'",
                ),
                DoctorCheck(
                    "dockerfile",
                    config.dockerfile_path.is_file(),
                    f"worker Dockerfile "
                    f"{'found' if config.dockerfile_path.is_file() else 'missing'} at "
                    f"{config.dockerfile_path}",
                ),
                DoctorCheck(
                    "worker_build_context",
                    not build_context_issues,
                    (
                        "worker image build context is complete"
                        if not build_context_issues
                        else "; ".join(build_context_issues)
                    ),
                ),
            )
        )
    elif backend == "apple_container":
        build_context_issues = worker_build_context_issues(
            config.project_root,
            constraints,
        )
        runtime_ok, image_ok, runtime_message = _probe_apple_container(
            config.docker_image,
            platform_name=config.docker_platform,
            constraints=constraints,
        )
        checks.extend(
            (
                DoctorCheck(
                    "apple_container",
                    runtime_ok,
                    runtime_message,
                ),
                DoctorCheck(
                    "apple_container_image",
                    image_ok,
                    runtime_message
                    if image_ok
                    else f"{runtime_message}; run 'qairt-agent image build "
                    f"--root {config.project_root}'",
                ),
                DoctorCheck(
                    "dockerfile",
                    config.dockerfile_path.is_file(),
                    f"worker Dockerfile "
                    f"{'found' if config.dockerfile_path.is_file() else 'missing'} "
                    f"at {config.dockerfile_path}",
                ),
                DoctorCheck(
                    "worker_build_context",
                    not build_context_issues,
                    (
                        "worker image build context is complete"
                        if not build_context_issues
                        else "; ".join(build_context_issues)
                    ),
                ),
            )
        )

    critical_ok = all(check.ok for check in checks if check.critical)
    return {
        "ok": critical_ok,
        "project_root": str(config.project_root),
        "sdk_root": str(sdk_path),
        "harness_constraints": str(config.harness_path),
        "worker_backend": backend,
        "checks": [check.to_dict() for check in checks],
    }


__all__ = [
    "CONFIG_FILENAME",
    "DEFAULT_SDK_ROOT",
    "DEFAULT_WORKER_BACKEND",
    "DEFAULT_WORKER_DOCKERFILE",
    "DoctorCheck",
    "ProjectConfig",
    "WORKER_BACKENDS",
    "discover_sdk_path",
    "doctor",
    "ensure_worker_build_context",
    "find_project_root",
    "init",
    "load",
    "resolve_logical_uri",
    "select_container_backend",
    "select_worker_backend",
    "to_logical_uri",
]
