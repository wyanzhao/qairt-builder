"""Shared mechanics for the Docker and Apple container worker backends."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

from qairt_agent.errors import WorkerCommandError
from qairt_agent.harness import (
    DEFAULT_CONSTRAINTS,
    DEFAULT_CONSTRAINTS_LOGICAL_PATH,
    HarnessConstraints,
)


def execute_command(
    executor: Any,
    argv: list[str],
    *,
    backend_name: str,
    stage: str,
) -> Any:
    """Execute an injected command and normalize timeout failures."""

    try:
        return executor(argv)
    except subprocess.TimeoutExpired as exc:
        raise WorkerCommandError(
            f"{backend_name} command timed out",
            stage=stage,
            retryable=True,
            details={
                "argv": list(argv),
                "timeout_seconds": exc.timeout,
            },
        ) from exc


def harness_build_path(
    context: Path,
    constraints: HarnessConstraints,
    *,
    error_type: type[Exception],
    stage: str,
) -> str:
    """Return the constraints path visible inside an image build context."""

    constraints_path = constraints.source_path.expanduser().resolve()
    try:
        return constraints_path.relative_to(context).as_posix()
    except ValueError:
        if constraints_path == DEFAULT_CONSTRAINTS.source_path.resolve():
            return DEFAULT_CONSTRAINTS_LOGICAL_PATH
        raise error_type(
            "selected harness constraints must be inside the image build context",
            stage=stage,
            details={
                "constraints": str(constraints_path),
                "context": str(context),
            },
        )


def harness_build_args(
    constraints: HarnessConstraints,
    visible_constraints_path: str,
) -> list[str]:
    """Render the backend-neutral pinned image build arguments."""

    return [
        "--build-arg",
        f"UBUNTU_VERSION={constraints.ubuntu_version}",
        "--build-arg",
        f"PYTHON_VERSION={constraints.python_version}",
        "--build-arg",
        f"QAIRT_DEPENDENCIES_FILE={constraints.dependencies_file}",
        "--build-arg",
        f"HARNESS_CONSTRAINTS_FILE={visible_constraints_path}",
        "--build-arg",
        f"TORCH_VERSION={constraints.torch_version}",
        "--build-arg",
        f"TORCH_INDEX_URL={constraints.torch_index_url}",
    ]


class WorkerRunnerMixin:
    """Common execution/error boundary for worker container backends."""

    worker_backend_name = "container"
    worker_run_stage = "container-run"

    def run(
        self,
        *,
        mounts: Any,
        command: list[str],
        network: bool = True,
        platform: str | None = None,
        workdir: str = "/workspace",
        env: Mapping[str, str] | None = None,
        user: str | None = None,
        memory: str | None = None,
        cpus: int | None = None,
    ) -> Any:
        self.require_available()
        argv = self.build_run_argv(
            mounts=mounts,
            command=command,
            network=network,
            platform=platform,
            workdir=workdir,
            env=dict(env or {}),
            user=user,
            memory=memory,
            cpus=cpus,
        )
        result = execute_command(
            self._executor,
            argv,
            backend_name=self.worker_backend_name,
            stage=self.worker_run_stage,
        )
        if getattr(result, "returncode", 1) != 0:
            raise WorkerCommandError(
                f"{self.worker_backend_name} worker command failed",
                stage=self.worker_run_stage,
                retryable=False,
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
        mounts: Any,
        command: list[str],
        platform: str | None = None,
        workdir: str = "/workspace",
        env: Mapping[str, str] | None = None,
        user: str | None = None,
        memory: str | None = None,
        cpus: int | None = None,
    ) -> Any:
        """Run a build-only or pickle-import job without network access."""

        return self.run(
            mounts=mounts,
            command=command,
            network=False,
            platform=platform,
            workdir=workdir,
            env=dict(env or {}),
            user=user,
            memory=memory,
            cpus=cpus,
        )


__all__ = [
    "WorkerRunnerMixin",
    "execute_command",
    "harness_build_args",
    "harness_build_path",
]
