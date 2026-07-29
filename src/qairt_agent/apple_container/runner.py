"""Injectable Apple ``container`` CLI worker backend.

Apple's CLI runs each Linux container in a lightweight VM.  On Apple Silicon
the QAIRT x86_64 worker is selected with ``--platform linux/amd64`` and
``--rosetta``.  The backend never starts the service, installs a kernel, pulls
an image, or changes privileged DNS settings on the user's behalf.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from qairt_agent.container_runner import (
    WorkerRunnerMixin,
    execute_command,
    harness_build_args,
    harness_build_path,
)
from qairt_agent.docker.image import (
    BindMount,
    DEFAULT_IMAGE_REF,
    DEFAULT_PLATFORM,
    RuntimeMounts,
    WORKER_PYTHONPATH,
    WorkerImageConfig,
)
from qairt_agent.errors import (
    AppleContainerUnavailableError,
)
from qairt_agent.harness import (
    DEFAULT_CONSTRAINTS,
    HarnessConstraints,
    parse_version,
)

CommandExecutor = Callable[[list[str]], Any]
_PROBE_TIMEOUT_SECONDS = 10.0
_COMMAND_TIMEOUT_SECONDS = 7200.0


def _default_executor(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        timeout=_COMMAND_TIMEOUT_SECONDS,
    )


def _default_probe_executor(
    argv: list[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROBE_TIMEOUT_SECONDS,
    )


def _platform_present(document: Any, platform_name: str) -> bool:
    expected_os, _, expected_arch = platform_name.partition("/")
    if isinstance(document, dict):
        platform_value = document.get("platform")
        if isinstance(platform_value, dict):
            if (
                platform_value.get("os") == expected_os
                and platform_value.get("architecture") == expected_arch
            ):
                return True
        return any(
            _platform_present(value, platform_name)
            for value in document.values()
        )
    if isinstance(document, list):
        return any(_platform_present(value, platform_name) for value in document)
    return False


def _find_string_field(document: Any, field_name: str) -> str | None:
    normalized = field_name.lower()
    if isinstance(document, dict):
        for key, value in document.items():
            if str(key).lower() == normalized and isinstance(value, str):
                return value
        for value in document.values():
            found = _find_string_field(value, field_name)
            if found is not None:
                return found
    elif isinstance(document, list):
        for value in document:
            found = _find_string_field(value, field_name)
            if found is not None:
                return found
    return None


def _has_dns_alias(document: Any, alias: str) -> bool:
    if isinstance(document, dict):
        for key, value in document.items():
            if (
                str(key).lower() in {"domain", "name"}
                and isinstance(value, str)
                and value.rstrip(".").lower()
                == alias.rstrip(".").lower()
            ):
                return True
        return any(_has_dns_alias(value, alias) for value in document.values())
    if isinstance(document, list):
        return any(_has_dns_alias(value, alias) for value in document)
    return False


class AppleContainerRunner(WorkerRunnerMixin):
    """Build and dispatch QAIRT workers with Apple ``container`` 1.x."""

    worker_backend_name = "Apple container"
    worker_run_stage = "apple-container-run"

    def __init__(
        self,
        *,
        command_executor: CommandExecutor | None = None,
        image: WorkerImageConfig | None = None,
        constraints: HarnessConstraints | None = None,
        host_arch: Callable[[], str] = platform.machine,
    ) -> None:
        self._executor: CommandExecutor = command_executor or _default_executor
        self._probe_executor: CommandExecutor = (
            command_executor or _default_probe_executor
        )
        self.image = image
        self.constraints = constraints or DEFAULT_CONSTRAINTS
        self._host_arch = host_arch

    @property
    def image_ref(self) -> str:
        return self.image.image_ref if self.image else DEFAULT_IMAGE_REF

    @property
    def platform_name(self) -> str:
        return (
            self.image.platform
            if self.image
            else self.constraints.platform
        )

    def _availability_error(self) -> AppleContainerUnavailableError | None:
        if shutil.which("container") is None:
            return AppleContainerUnavailableError(
                "Apple container CLI is not installed; install the pinned CLI "
                "version yourself before running macOS workers.",
                stage="apple-container",
                retryable=True,
                details={
                    "required_version": self.constraints.apple_container_version,
                    "constraints": str(self.constraints.source_path),
                },
            )
        try:
            version_result = self._probe_executor(["container", "--version"])
        except Exception as exc:  # noqa: BLE001 - normalized below
            return AppleContainerUnavailableError(
                "Apple container CLI version probe failed",
                stage="apple-container",
                retryable=True,
                details={"reason": str(exc)},
            )
        version_text = (
            (getattr(version_result, "stdout", "") or "")
            + "\n"
            + (getattr(version_result, "stderr", "") or "")
        ).strip()
        actual_version = parse_version(version_text)
        required_version = parse_version(
            self.constraints.apple_container_version
        )
        if (
            getattr(version_result, "returncode", 1) != 0
            or actual_version is None
            or actual_version != required_version
        ):
            return AppleContainerUnavailableError(
                "Apple container CLI version does not match the harness",
                stage="apple-container",
                retryable=False,
                details={
                    "required_version": self.constraints.apple_container_version,
                    "actual_version": (
                        ".".join(str(part) for part in actual_version)
                        if actual_version
                        else version_text
                    ),
                    "constraints": str(self.constraints.source_path),
                },
            )
        try:
            status = self._probe_executor(
                ["container", "system", "status", "--format", "json"]
            )
        except Exception as exc:  # noqa: BLE001 - normalized below
            return AppleContainerUnavailableError(
                "Apple container service status probe failed",
                stage="apple-container",
                retryable=True,
                details={"reason": str(exc)},
            )
        if getattr(status, "returncode", 1) != 0:
            return AppleContainerUnavailableError(
                "Apple container services are not running; start them explicitly "
                "with 'container system start'.",
                stage="apple-container",
                retryable=True,
                details={
                    "stderr": getattr(status, "stderr", "") or "",
                    "stdout": getattr(status, "stdout", "") or "",
                },
            )
        status_text = (getattr(status, "stdout", "") or "").strip()
        try:
            status_document = json.loads(status_text)
        except json.JSONDecodeError as exc:
            return AppleContainerUnavailableError(
                "Apple container service status returned invalid JSON",
                stage="apple-container",
                retryable=True,
                details={"reason": str(exc), "stdout": status_text},
            )
        server_version_text = _find_string_field(
            status_document,
            "apiServerVersion",
        )
        server_version = (
            parse_version(server_version_text)
            if server_version_text is not None
            else None
        )
        if server_version != required_version:
            return AppleContainerUnavailableError(
                "Apple container API server version does not match the "
                "harness; restart or reinstall the pinned service",
                stage="apple-container",
                retryable=True,
                details={
                    "required_version": self.constraints.apple_container_version,
                    "api_server_version": (
                        ".".join(str(part) for part in server_version)
                        if server_version is not None
                        else server_version_text or "<missing>"
                    ),
                },
            )
        return None

    def is_available(self) -> bool:
        return self._availability_error() is None

    def require_available(self) -> None:
        error = self._availability_error()
        if error is not None:
            raise error

    def require_image(self) -> str:
        """Return the content digest exposed by the selected image platform."""

        result = execute_command(
            self._probe_executor,
            ["container", "image", "inspect", self.image_ref],
            backend_name="Apple container",
            stage="apple-container",
        )
        stdout = (getattr(result, "stdout", "") or "").strip()
        if getattr(result, "returncode", 1) != 0 or not stdout:
            raise AppleContainerUnavailableError(
                f"Apple container worker image '{self.image_ref}' is unavailable; "
                "build it with 'qairt-agent image build'.",
                stage="apple-container",
                retryable=True,
                details={
                    "image_ref": self.image_ref,
                    "stderr": getattr(result, "stderr", "") or "",
                },
            )
        try:
            document = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise AppleContainerUnavailableError(
                "Apple container image inspect returned invalid JSON",
                stage="apple-container",
                details={"image_ref": self.image_ref, "reason": str(exc)},
            ) from exc
        if not _platform_present(document, self.platform_name):
            raise AppleContainerUnavailableError(
                f"worker image '{self.image_ref}' has no "
                f"{self.platform_name} variant",
                stage="apple-container",
                details={
                    "image_ref": self.image_ref,
                    "platform": self.platform_name,
                },
            )
        image_digest = _find_string_field(document, "digest")
        if (
            image_digest is None
            or not image_digest.startswith("sha256:")
            or len(image_digest) != len("sha256:") + 64
        ):
            raise AppleContainerUnavailableError(
                "Apple container image inspect did not expose a content "
                "digest for the selected platform",
                stage="apple-container",
                retryable=False,
                details={"image_ref": self.image_ref},
            )
        return image_digest

    def require_host_alias(self, alias: str) -> None:
        """Require the privileged localhost DNS bridge used for host ADB."""

        result = execute_command(
            self._probe_executor,
            ["container", "system", "dns", "list", "--format", "json"],
            backend_name="Apple container",
            stage="apple-container-adb",
        )
        stdout = (getattr(result, "stdout", "") or "").strip()
        if getattr(result, "returncode", 1) == 0:
            try:
                document = json.loads(stdout)
            except json.JSONDecodeError:
                document = None
            if document is not None and _has_dns_alias(document, alias):
                return
        raise AppleContainerUnavailableError(
            f"Apple container host alias '{alias}' is not configured for ADB. "
            "Configure it explicitly before launching device jobs.",
            stage="apple-container-adb",
            retryable=True,
            details={
                "alias": alias,
                "setup_command": (
                    f"sudo container system dns create {alias} "
                    "--localhost 203.0.113.113"
                ),
                "stderr": getattr(result, "stderr", "") or "",
            },
        )

    def build_run_argv(
        self,
        *,
        mounts: RuntimeMounts,
        command: list[str],
        network: bool = True,
        platform: str | None = None,
        workdir: str = "/workspace",
        env: dict[str, str] | None = None,
        user: str | None = None,
        memory: str | None = None,
        cpus: int | None = None,
    ) -> list[str]:
        """Render an Apple ``container run`` invocation without executing it."""

        effective_platform = platform or self.platform_name or DEFAULT_PLATFORM
        argv = [
            "container",
            "run",
            "--rm",
            "--platform",
            effective_platform,
        ]
        if (
            self._host_arch().lower() in {"arm64", "aarch64"}
            and effective_platform.endswith("/amd64")
        ):
            argv.append("--rosetta")
        if not network:
            # Apple container 1.0 has no Docker-equivalent ``--network none``.
            # ``--no-dns`` makes offline smoke jobs deterministic with respect
            # to names, without pretending that IP egress is hard-isolated.
            argv.append("--no-dns")
        if memory:
            argv += ["-m", memory]
        if cpus is not None:
            argv += ["-c", str(cpus)]
        if user:
            argv += ["--user", user]
        argv += ["--workdir", workdir]
        for key, value in sorted((env or {}).items()):
            argv += ["--env", f"{key}={value}"]
        argv += mounts.to_apple_container_args()
        argv.append(self.image_ref)
        argv += list(command)
        return argv

    def build_image(self, *, context: str | Path, dockerfile: str | Path) -> Any:
        self.require_available()
        context_path = Path(context).expanduser().resolve()
        visible_constraints = harness_build_path(
            context_path,
            self.constraints,
            error_type=AppleContainerUnavailableError,
            stage="apple-container-build",
        )
        argv = [
            "container",
            "build",
            "--platform",
            self.platform_name,
            "--file",
            str(Path(dockerfile)),
            "--tag",
            self.image_ref,
            "--progress",
            "plain",
            *harness_build_args(self.constraints, visible_constraints),
            str(context_path),
        ]
        result = execute_command(
            self._executor,
            argv,
            backend_name="Apple container",
            stage="apple-container-build",
        )
        if getattr(result, "returncode", 1) != 0:
            raise AppleContainerUnavailableError(
                f"failed to build Apple container worker image "
                f"'{self.image_ref}'",
                stage="apple-container-build",
                retryable=True,
                details={
                    "stdout": getattr(result, "stdout", "") or "",
                    "stderr": getattr(result, "stderr", "") or "",
                },
            )
        return result

    def build_sdk_smoke_argv(self, *, sdk_root: str | Path) -> list[str]:
        resolved_sdk = Path(sdk_root).expanduser().resolve()
        sdk_mount = BindMount(
            source=str(resolved_sdk),
            target="/opt/qairt",
            read_only=True,
        )
        # Smoke only needs the SDK mount; avoid exposing unrelated writable
        # aliases by constructing this short argv directly.
        argv = [
            "container",
            "run",
            "--rm",
            "--platform",
            self.platform_name,
        ]
        if (
            self._host_arch().lower() in {"arm64", "aarch64"}
            and self.platform_name.endswith("/amd64")
        ):
            argv.append("--rosetta")
        argv += [
            "--no-dns",
            "--workdir",
            "/opt/qairt-agent",
            "--env",
            "QAIRT_SDK_ROOT=/opt/qairt",
            "--env",
            "QNN_SDK_ROOT=/opt/qairt",
            "--env",
            "QAIRT_AGENT_HARNESS_CONSTRAINTS=/opt/qairt-agent/harness/constraints.json",
            "--env",
            f"PYTHONPATH={WORKER_PYTHONPATH}",
            "--env",
            "LD_LIBRARY_PATH=/opt/qairt/lib/x86_64-linux-clang",
            "--mount",
            (
                f"type=bind,source={sdk_mount.source},"
                f"target={sdk_mount.target},readonly"
            ),
            self.image_ref,
            "/opt/venv/bin/python",
            "-m",
            "qairt_agent.docker.smoke",
        ]
        return argv

    def smoke_test_sdk(self, *, sdk_root: str | Path) -> Any:
        resolved_sdk = Path(sdk_root).expanduser().resolve()
        if not (resolved_sdk / "sdk.yaml").is_file():
            raise AppleContainerUnavailableError(
                f"cannot smoke-test worker image: QAIRT SDK not found at "
                f"'{resolved_sdk}'",
                stage="image-smoke",
                details={"sdk_root": str(resolved_sdk)},
            )
        self.require_available()
        self.require_image()
        result = execute_command(
            self._executor,
            self.build_sdk_smoke_argv(sdk_root=resolved_sdk),
            backend_name="Apple container",
            stage="image-smoke",
        )
        if getattr(result, "returncode", 1) != 0:
            raise AppleContainerUnavailableError(
                "Apple container worker image failed the mounted QAIRT Python "
                "API smoke test",
                stage="image-smoke",
                details={
                    "sdk_root": str(resolved_sdk),
                    "stdout": getattr(result, "stdout", "") or "",
                    "stderr": getattr(result, "stderr", "") or "",
                },
            )
        return result

__all__ = ["AppleContainerRunner", "CommandExecutor"]
