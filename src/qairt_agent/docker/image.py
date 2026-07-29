"""Worker image and runtime-mount contracts for the QAIRT agent.

macOS Apple ``container`` and Linux Docker share one pinned worker platform.
The concrete versions come from ``harness/constraints.json``.

Two invariants shape this module:

* The framework **fails closed** when Docker is unavailable.  It never installs
  Docker and never moves the SDK; the user owns the Docker toolchain.
* The SDK, models, and artifacts are **mounted at runtime**, never baked into an
  image.  :func:`build_context_excludes` lists the patterns that keep them out of
  any build context, and :class:`RuntimeMounts` describes the read-only SDK /
  workspace mounts plus the caller-provided read-write volumes.
"""

from __future__ import annotations

from pydantic import field_validator

from qairt_agent.contracts import FrozenContract
from qairt_agent.contracts import StageProvenance
from qairt_agent.errors import DockerUnavailableError
from qairt_agent.harness import DEFAULT_CONSTRAINTS

#: Project-built worker image.  Unlike a bare Ubuntu image, this contains
#: Harness-selected Python, adb, and the qairt-agent package.
DEFAULT_IMAGE_REF = DEFAULT_CONSTRAINTS.worker_image

#: The pinned Ubuntu release shared by the macOS and Linux workers.
PINNED_UBUNTU = DEFAULT_CONSTRAINTS.ubuntu_version

#: The pinned Python interpreter version inside the worker image.
PINNED_PYTHON = DEFAULT_CONSTRAINTS.python_version

#: The single worker platform.  Every host targets this; non-amd64 hosts
#: emulate it.
DEFAULT_PLATFORM = DEFAULT_CONSTRAINTS.platform

# Container-side mount targets.  Sources are logical (relative paths or named
# volumes) so the plan is host-independent; targets are fixed by the worker.
SDK_MOUNT_TARGET = "/opt/qairt"
WORKSPACE_MOUNT_TARGET = "/workspace"
MODELS_MOUNT_TARGET = "/models"
STATE_MOUNT_TARGET = "/state"
ARTIFACTS_MOUNT_TARGET = "/artifacts"
CACHE_MOUNT_TARGET = "/cache"
WORKER_AGENT_SOURCE_PATH = "/opt/qairt-agent/qairt-agent-src.zip"
WORKER_PYTHONPATH = (
    f"{WORKER_AGENT_SOURCE_PATH}:"
    "/opt/qairt/lib/python:/opt/qairt/benchmarks/QNN"
)

#: Host architectures that run the amd64 worker natively (no emulation).
_NATIVE_AMD64_ARCHES = frozenset({"x86_64", "amd64"})


class WorkerImageConfig(FrozenContract):
    """An immutable reference to the pinned worker image.

    The platform is constrained to :data:`DEFAULT_PLATFORM`: the worker always
    targets ``linux/amd64`` and relies on emulation elsewhere.
    """

    image_ref: str
    platform: str = DEFAULT_PLATFORM
    ubuntu_version: str = PINNED_UBUNTU
    python_version: str = PINNED_PYTHON
    digest: str | None = None

    @field_validator("image_ref")
    @classmethod
    def validate_image_ref(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("image_ref cannot be blank")
        return stripped

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, value: str) -> str:
        if value != DEFAULT_PLATFORM:
            raise ValueError(
                f"platform must be {DEFAULT_PLATFORM!r} (the pinned worker platform); got {value!r}"
            )
        return value


class RuntimeMounts(FrozenContract):
    """The runtime mount plan for one container invocation.

    Sources are logical (relative host paths or named volumes), never
    host-absolute, so the plan stays portable.  The SDK and workspace are
    mounted read-only; the state, artifacts, and cache volumes are read-write
    and are provided by the caller.
    """

    sdk_root: str = "./qnn/qnn"
    workspace: str = "./models"
    state_volume: str
    artifacts_volume: str
    cache_volume: str
    jobs_volume: str | None = None
    models_root: str | None = None
    compatibility_mounts: tuple["BindMount", ...] = ()

    def to_docker_args(self) -> list[str]:
        """Render the mount plan as ``docker run -v`` arguments.

        The SDK and workspace are read-only (``:ro``); the three caller-provided
        volumes are read-write.
        """

        args = [
            "-v",
            f"{self.sdk_root}:{SDK_MOUNT_TARGET}:ro",
            "-v",
            f"{self.workspace}:{WORKSPACE_MOUNT_TARGET}:ro",
            "-v",
            f"{self.state_volume}:{STATE_MOUNT_TARGET}",
            "-v",
            f"{self.artifacts_volume}:{ARTIFACTS_MOUNT_TARGET}",
            "-v",
            f"{self.cache_volume}:{CACHE_MOUNT_TARGET}",
        ]
        if self.jobs_volume is not None:
            args += ["-v", f"{self.jobs_volume}:{STATE_MOUNT_TARGET}/jobs"]
        if self.models_root is not None:
            args += ["-v", f"{self.models_root}:{MODELS_MOUNT_TARGET}:ro"]
        for mount in self.compatibility_mounts:
            suffix = ":ro" if mount.read_only else ""
            args += ["-v", f"{mount.source}:{mount.target}{suffix}"]
        return args

    def to_apple_container_args(self) -> list[str]:
        """Render mounts with Apple ``container``'s documented mount syntax."""

        args: list[str] = []

        def append(source: str, target: str, *, read_only: bool) -> None:
            mount_type = (
                "bind"
                if source.startswith(("/", "./", "../"))
                else "volume"
            )
            value = f"type={mount_type},source={source},target={target}"
            if read_only:
                value += ",readonly"
            args.extend(("--mount", value))

        append(self.sdk_root, SDK_MOUNT_TARGET, read_only=True)
        append(self.workspace, WORKSPACE_MOUNT_TARGET, read_only=True)
        append(self.state_volume, STATE_MOUNT_TARGET, read_only=False)
        append(self.artifacts_volume, ARTIFACTS_MOUNT_TARGET, read_only=False)
        append(self.cache_volume, CACHE_MOUNT_TARGET, read_only=False)
        if self.jobs_volume is not None:
            append(
                self.jobs_volume,
                f"{STATE_MOUNT_TARGET}/jobs",
                read_only=False,
            )
        if self.models_root is not None:
            append(self.models_root, MODELS_MOUNT_TARGET, read_only=True)
        for mount in self.compatibility_mounts:
            append(
                mount.source,
                mount.target,
                read_only=mount.read_only,
            )
        return args


class BindMount(FrozenContract):
    """One additional bind mount, used for host-path compatibility aliases."""

    source: str
    target: str
    read_only: bool = False

    @field_validator("target")
    @classmethod
    def target_is_absolute(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("container mount target must be absolute")
        return value


def default_mounts(
    state_volume: str,
    artifacts_volume: str,
    cache_volume: str,
    *,
    sdk_root: str = "./qnn/qnn",
    workspace: str = "./models",
    jobs_volume: str | None = None,
    models_root: str | None = None,
    compatibility_mounts: tuple[BindMount, ...] = (),
) -> RuntimeMounts:
    """Build the standard mount plan from the three caller-provided volumes."""

    return RuntimeMounts(
        sdk_root=sdk_root,
        workspace=workspace,
        state_volume=state_volume,
        artifacts_volume=artifacts_volume,
        cache_volume=cache_volume,
        jobs_volume=jobs_volume,
        models_root=models_root,
        compatibility_mounts=compatibility_mounts,
    )


def build_context_excludes() -> tuple[str, ...]:
    """Patterns that must never enter a Docker build context.

    The SDK, models, and artifacts are mounted at runtime, so they are excluded
    from any image build to keep them from being copied into a layer.
    """

    return (
        ".git/",
        ".venv/",
        ".qairt-agent/",
        "__pycache__/",
        ".pytest_cache/",
        ".coverage",
        "qnn/",
        "qairt/",
        "models/",
        "artifacts/",
        "*.onnx",
        "*.bin",
        "*.raw",
        "*.pkl",
        "*.pickle",
    )


def resolve_image_digest(image_ref: str, *, runner) -> str:
    """Resolve the ``sha256:...`` repo digest for an image via an injected runner.

    ``runner`` is a callable ``list[str] -> CompletedProcess-like``.  Raises
    :class:`DockerUnavailableError` when the probe fails or yields no digest so
    the framework fails closed rather than caching against an unpinned image.
    """

    result = runner(
        ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image_ref]
    )
    stdout = (getattr(result, "stdout", "") or "").strip()
    stderr = (getattr(result, "stderr", "") or "").strip()
    if getattr(result, "returncode", 1) != 0 or "sha256:" not in stdout:
        raise DockerUnavailableError(
            f"could not resolve a sha256 digest for image '{image_ref}'",
            stage="docker",
            retryable=True,
            details={"image_ref": image_ref, "stderr": stderr},
        )
    # ``{{index .RepoDigests 0}}`` prints e.g. ``ubuntu@sha256:<hex>``; return
    # just the ``sha256:<hex>`` token.
    return stdout[stdout.index("sha256:") :].split()[0]


def image_provenance(
    config: WorkerImageConfig,
    *,
    host_arch: str,
    sdk_build: str,
    adapter_capability: str,
) -> StageProvenance:
    """Derive the stage provenance contributed by the worker image and host.

    ``emulation`` is true when the host is not a native amd64 architecture
    (e.g. ``arm64`` on Apple Silicon), since the amd64 worker then runs under
    emulation.
    """

    return StageProvenance(
        sdk_build=sdk_build,
        adapter_capability=adapter_capability,
        platform_abi=f"ubuntu{config.ubuntu_version}-amd64",
        image_digest=config.digest,
        host_arch=host_arch,
        emulation=host_arch not in _NATIVE_AMD64_ARCHES,
    )


# Backwards-compatible public name for existing callers.
DockerImageConfig = WorkerImageConfig


__all__ = [
    "ARTIFACTS_MOUNT_TARGET",
    "CACHE_MOUNT_TARGET",
    "DEFAULT_IMAGE_REF",
    "DEFAULT_PLATFORM",
    "BindMount",
    "DockerImageConfig",
    "WorkerImageConfig",
    "MODELS_MOUNT_TARGET",
    "PINNED_PYTHON",
    "PINNED_UBUNTU",
    "RuntimeMounts",
    "SDK_MOUNT_TARGET",
    "STATE_MOUNT_TARGET",
    "WORKSPACE_MOUNT_TARGET",
    "WORKER_AGENT_SOURCE_PATH",
    "WORKER_PYTHONPATH",
    "build_context_excludes",
    "default_mounts",
    "image_provenance",
    "resolve_image_digest",
]
