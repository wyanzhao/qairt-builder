"""Thin, injectable wrapper around ``docker run``.

The framework never installs Docker and never relocates the SDK: if Docker is
unavailable it fails closed with :class:`DockerUnavailableError`.  The SDK,
models, and artifacts are mounted at runtime (see
:class:`~qairt_agent.docker.image.RuntimeMounts`), and build-only /
pickle-import jobs disable the network.

All subprocess interaction flows through an injectable ``command_executor``
callable so tests can record argv without a real ``docker`` binary.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from qairt_agent.docker.image import (
    DEFAULT_IMAGE_REF,
    DEFAULT_PLATFORM,
    DockerImageConfig,
    RuntimeMounts,
    WORKER_PYTHONPATH,
)
from qairt_agent.container_runtime import (
    SDK_MOUNT_TARGET,
    SMOKE_COMMAND,
    WORKER_WORKDIR,
    flatten_env,
    image_build_args,
    resolve_harness_build_path,
    worker_smoke_env,
)
from qairt_agent.errors import DockerUnavailableError
from qairt_agent.harness import (
    DEFAULT_CONSTRAINTS,
    DEFAULT_CONSTRAINTS_LOGICAL_PATH,
    HarnessConstraints,
    parse_version,
)

CommandExecutor = Callable[[list[str]], Any]


def _default_executor(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing output as text and never raising on status."""

    return subprocess.run(argv, capture_output=True, text=True, check=False)


class DockerRunner:
    """Build and dispatch ``docker run`` invocations against a pinned image.

    The ``command_executor`` injection point is what makes the runner testable:
    pass a fake callable to record argv instead of shelling out to Docker.
    """

    def __init__(
        self,
        *,
        command_executor: CommandExecutor | None = None,
        image: DockerImageConfig | None = None,
        constraints: HarnessConstraints | None = None,
    ) -> None:
        self._executor: CommandExecutor = command_executor or _default_executor
        self.image = image
        self.constraints = constraints or DEFAULT_CONSTRAINTS

    def is_available(self) -> bool:
        """Return True iff ``docker`` is on PATH and a ``docker version`` probe succeeds."""

        if shutil.which("docker") is None:
            return False
        try:
            result = self._executor(
                ["docker", "version", "--format", "{{json .}}"]
            )
        except Exception:  # noqa: BLE001 - any probe failure means unavailable
            return False
        try:
            document = json.loads(
                (getattr(result, "stdout", "") or "").strip()
            )
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(document, dict):
            return False
        client = document.get("Client")
        server = document.get("Server")
        client_version = (
            parse_version(str(client.get("Version", "")))
            if isinstance(client, dict)
            else None
        )
        server_version = (
            parse_version(str(server.get("Version", "")))
            if isinstance(server, dict)
            else None
        )
        minimum = parse_version(self.constraints.docker_minimum_version)
        return (
            getattr(result, "returncode", 1) == 0
            and client_version is not None
            and server_version is not None
            and minimum is not None
            and client_version >= minimum
            and server_version >= minimum
        )

    def require_available(self) -> None:
        """Raise :class:`DockerUnavailableError` if Docker cannot be used.

        The message directs the user to install Docker themselves; the framework
        never installs or manages the Docker toolchain.
        """

        if not self.is_available():
            raise DockerUnavailableError(
                "Docker is not available. The framework never installs Docker; "
                "install Docker yourself, then ensure the 'docker' binary is on "
                "PATH, the daemon is reachable, and the CLI satisfies the "
                f"harness minimum {self.constraints.docker_minimum_version}.",
                stage="docker",
                retryable=True,
                details={
                    "minimum_version": self.constraints.docker_minimum_version,
                    "constraints": str(self.constraints.source_path),
                },
            )

    def require_image(self) -> str:
        """Return the immutable image ID, failing closed if it is unavailable."""

        image_ref = self.image.image_ref if self.image else DEFAULT_IMAGE_REF
        result = self._executor(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image_ref]
        )
        image_id = (getattr(result, "stdout", "") or "").strip()
        if getattr(result, "returncode", 1) != 0 or not image_id.startswith("sha256:"):
            raise DockerUnavailableError(
                f"Docker worker image '{image_ref}' is unavailable; build it with "
                "'qairt-agent image build'",
                stage="docker",
                retryable=True,
                details={"image_ref": image_ref, "stderr": getattr(result, "stderr", "") or ""},
            )
        return image_id

    def build_run_argv(
        self,
        *,
        mounts: RuntimeMounts,
        command: list[str],
        network: bool = True,
        platform: str | None = None,
        workdir: str = "/workspace",
        env: dict[str, str] | None = None,
        add_host_gateway: bool = False,
        user: str | None = None,
    ) -> list[str]:
        """Render a docker-run argv without executing it."""

        effective_platform = platform or (self.image.platform if self.image else None) or DEFAULT_PLATFORM
        image_ref = self.image.image_ref if self.image else DEFAULT_IMAGE_REF
        argv: list[str] = ["docker", "run", "--rm", "--platform", effective_platform]
        if not network:
            argv += ["--network", "none"]
        if add_host_gateway:
            argv += [
                "--add-host",
                f"{self.constraints.docker_host_alias}:host-gateway",
            ]
        if user:
            argv += ["--user", user]
        argv += ["--workdir", workdir]
        for key, value in sorted((env or {}).items()):
            argv += ["-e", f"{key}={value}"]
        argv += mounts.to_docker_args()
        argv.append(image_ref)
        argv += list(command)
        return argv

    def build_image(self, *, context: str | Path, dockerfile: str | Path) -> Any:
        """Build the pinned worker image from the checked-in Dockerfile."""

        self.require_available()
        context_path = Path(context).expanduser().resolve()
        harness_build_path = resolve_harness_build_path(
            self.constraints,
            context_path,
            on_outside_context=lambda constraints_path, resolved_context: (
                DockerUnavailableError(
                    "selected harness constraints must be inside the image "
                    "build context",
                    stage="docker-build",
                    details={
                        "constraints": str(constraints_path),
                        "context": str(resolved_context),
                    },
                )
            ),
        )
        image_ref = self.image.image_ref if self.image else DEFAULT_IMAGE_REF
        platform_name = self.image.platform if self.image else DEFAULT_PLATFORM
        argv = [
            "docker",
            "build",
            "--platform",
            platform_name,
            "--file",
            str(Path(dockerfile)),
            "--tag",
            image_ref,
            *image_build_args(self.constraints, harness_build_path),
            str(context_path),
        ]
        result = self._executor(argv)
        if getattr(result, "returncode", 1) != 0:
            raise DockerUnavailableError(
                f"failed to build Docker worker image '{image_ref}'",
                stage="docker",
                retryable=True,
                details={"stderr": getattr(result, "stderr", "") or ""},
            )
        return result

    def build_sdk_smoke_argv(self, *, sdk_root: str | Path) -> list[str]:
        """Render the post-build SDK/API smoke-test invocation.

        The SDK remains outside the image and is mounted read-only for this
        acceptance test.  Network access is disabled: every dependency must
        already be present in the worker image.
        """

        resolved_sdk = Path(sdk_root).expanduser().resolve()
        image_ref = self.image.image_ref if self.image else DEFAULT_IMAGE_REF
        platform_name = self.image.platform if self.image else DEFAULT_PLATFORM
        return [
            "docker",
            "run",
            "--rm",
            "--platform",
            platform_name,
            # Docker gives real IP-egress isolation here; Apple `container`
            # only offers --no-dns, and says so at its own call site.
            "--network",
            "none",
            "--workdir",
            WORKER_WORKDIR,
            *flatten_env(worker_smoke_env(), flag="-e"),
            "-v",
            f"{resolved_sdk}:{SDK_MOUNT_TARGET}:ro",
            image_ref,
            *SMOKE_COMMAND,
        ]

    def smoke_test_sdk(self, *, sdk_root: str | Path) -> Any:
        """Accept an image only if both pinned QAIRT Python API lanes import."""

        resolved_sdk = Path(sdk_root).expanduser().resolve()
        if not (resolved_sdk / "sdk.yaml").is_file():
            raise DockerUnavailableError(
                f"cannot smoke-test worker image: QAIRT SDK not found at "
                f"'{resolved_sdk}'",
                stage="image-smoke",
                details={"sdk_root": str(resolved_sdk)},
            )
        self.require_available()
        self.require_image()
        result = self._executor(self.build_sdk_smoke_argv(sdk_root=resolved_sdk))
        if getattr(result, "returncode", 1) != 0:
            raise DockerUnavailableError(
                "Docker worker image failed the mounted QAIRT Python API smoke test",
                stage="image-smoke",
                details={
                    "sdk_root": str(resolved_sdk),
                    "stdout": getattr(result, "stdout", "") or "",
                    "stderr": getattr(result, "stderr", "") or "",
                },
            )
        return result

    def run(
        self,
        *,
        mounts: RuntimeMounts,
        command: list[str],
        network: bool = True,
        platform: str | None = None,
        workdir: str = "/workspace",
        env: dict[str, str] | None = None,
        user: str | None = None,
    ) -> Any:
        """Assemble and execute one ``docker run`` invocation.

        The platform always defaults to the pinned worker platform; passing
        ``network=False`` isolates the container (``--network none``) for
        build-only and pickle-import jobs.
        """

        self.require_available()

        argv = self.build_run_argv(
            mounts=mounts,
            command=command,
            network=network,
            platform=platform,
            workdir=workdir,
            env=env,
            user=user,
        )
        result = self._executor(argv)
        if getattr(result, "returncode", 1) != 0:
            raise DockerUnavailableError(
                "Docker worker command failed",
                stage="docker-run",
                retryable=True,
                details={
                    "returncode": getattr(result, "returncode", None),
                    "stdout": getattr(result, "stdout", "") or "",
                    "stderr": getattr(result, "stderr", "") or "",
                },
            )
        return result

    def run_build_isolated(
        self,
        *,
        mounts: RuntimeMounts,
        command: list[str],
        platform: str | None = None,
        workdir: str = "/workspace",
        env: dict[str, str] | None = None,
        user: str | None = None,
    ) -> Any:
        """Run a build-only / pickle-import job with the network disabled."""

        return self.run(
            mounts=mounts,
            command=command,
            network=False,
            platform=platform,
            workdir=workdir,
            env=env,
            user=user,
        )


__all__ = ["CommandExecutor", "DockerRunner"]
