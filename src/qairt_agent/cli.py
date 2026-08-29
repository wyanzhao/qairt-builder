"""Command-line interface for the QAIRT agent native workflow.

Long-running commands (``build``/``validate``/``benchmark``/``diagnose``/
``workflow``/``rerun``) submit a persistent journal job and, by default, spawn a
detached worker process so the job survives the CLI exiting.  They print a
single JSON line ``{job_id, state, status_path}`` and return.  ``--follow``
streams JSONL events until the job reaches a terminal state; ``--inline`` runs
the worker in the current process (used by tests and short jobs).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, TextIO

from qairt_agent import project
from qairt_agent.agent import QairtAgentClient
from qairt_agent.apple_container import AppleContainerRunner
from qairt_agent.artifacts import sha256_file
from qairt_agent.contracts import JobState, StageProvenance
from qairt_agent.device.adb import canonicalize_adb_server
from qairt_agent.docker import (
    BindMount,
    DockerRunner,
    WORKER_PYTHONPATH,
    WorkerImageConfig,
    default_mounts,
)
from qairt_agent.errors import (
    ErrorCode,
    InvalidSpecError,
    PickleRejectedError,
    ProjectNotInitializedError,
    QairtAgentError,
    ToolError,
    ToolErrorData,
)
from qairt_agent.families.presets import (
    effective_benchmark_policy,
    family_for_preset,
    resolve_workflow,
    to_build_spec,
)
from qairt_agent.harness import DEFAULT_CONSTRAINTS, load_harness_constraints
from qairt_agent.jobs.journal import JobJournal
from qairt_agent.jobs.worker import DEFAULT_WORKFLOW_STAGES

_STAGE_COMMANDS = {
    "build": ("build",),
    "validate": ("validate",),
    "benchmark": ("benchmark",),
    "diagnose": ("diagnose",),
}
_REQUIRES_FROM_JOB = {"validate", "benchmark", "diagnose"}
_DEFAULT_WORKER_STARTUP_TIMEOUT = 30.0
_WORKER_STARTUP_POLL_INTERVAL = 0.05
_WORKER_LOG_TAIL_BYTES = 8192
_PICKLE_IMPORT_LOCAL_ENV = "QAIRT_AGENT_PICKLE_IMPORT_LOCAL"
_PICKLE_SOURCE_PATH_ENV = "QAIRT_AGENT_PICKLE_SOURCE_PATH"
_PICKLE_SOURCE_SHA256_ENV = "QAIRT_AGENT_PICKLE_SOURCE_SHA256"
_PICKLE_WORKER_SOURCE = "/qairt-agent-input/archive.pt"
_PICKLE_WORKER_SOURCE_ROOT = "/qairt-agent-input"
_PICKLE_WORKER_OUTPUT = "/qairt-agent-output"


def _verified_pickle_sha256(
    path: Path,
    *,
    expected_sha256: str | None = None,
    label: str,
) -> str:
    """Hash one pickle source and fail closed on mutation or read failure."""

    try:
        digest, _ = sha256_file(path)
    except (OSError, QairtAgentError) as exc:
        raise PickleRejectedError(
            f"Cannot hash {label} {path}: {exc}",
            stage="pickle-worker",
            details={"path": str(path), "label": label, "error": str(exc)},
        ) from exc
    if expected_sha256 is not None and digest != expected_sha256:
        raise PickleRejectedError(
            f"{label} changed while preparing the isolated Torch import",
            stage="pickle-worker",
            details={
                "path": str(path),
                "label": label,
                "expected_sha256": expected_sha256,
                "actual_sha256": digest,
            },
        )
    return digest


@contextlib.contextmanager
def _pickle_worker_source_mount(
    source: Path,
    *,
    backend: str,
    source_sha256: str,
):
    """Yield a read-only source mount accepted by the selected runtime.

    Apple container 1.0 cannot bind a regular file.  Its source is therefore
    a mode-0700 temporary directory containing only a content-verified
    ``archive.pt`` copy.  Docker retains its narrower regular-file bind.
    Both paths re-hash the original source after the worker returns.
    """

    if backend != "apple_container":
        try:
            yield BindMount(
                source=str(source),
                target=_PICKLE_WORKER_SOURCE,
                read_only=True,
            )
        finally:
            _verified_pickle_sha256(
                source,
                expected_sha256=source_sha256,
                label="original Torch archive",
            )
        return

    with tempfile.TemporaryDirectory(
        prefix="qairt-agent-pickle-source-"
    ) as temporary:
        staging_root = Path(temporary).resolve()
        os.chmod(staging_root, 0o700)
        staged_archive = staging_root / "archive.pt"
        try:
            shutil.copyfile(source, staged_archive)
            os.chmod(staged_archive, 0o400)
        except OSError as exc:
            raise PickleRejectedError(
                "Cannot stage the Torch archive for Apple container",
                stage="pickle-worker",
                details={
                    "source": str(source),
                    "staging_root": str(staging_root),
                    "error": str(exc),
                },
            ) from exc
        _verified_pickle_sha256(
            source,
            expected_sha256=source_sha256,
            label="original Torch archive",
        )
        _verified_pickle_sha256(
            staged_archive,
            expected_sha256=source_sha256,
            label="staged Torch archive",
        )
        if tuple(path.name for path in staging_root.iterdir()) != (
            "archive.pt",
        ):
            raise PickleRejectedError(
                "Apple container Torch staging directory is not private",
                stage="pickle-worker",
                details={"staging_root": str(staging_root)},
            )
        try:
            yield BindMount(
                source=str(staging_root),
                target=_PICKLE_WORKER_SOURCE_ROOT,
                read_only=True,
            )
        finally:
            _verified_pickle_sha256(
                source,
                expected_sha256=source_sha256,
                label="original Torch archive",
            )
            _verified_pickle_sha256(
                staged_archive,
                expected_sha256=source_sha256,
                label="staged Torch archive",
            )
            if tuple(path.name for path in staging_root.iterdir()) != (
                "archive.pt",
            ):
                raise PickleRejectedError(
                    "Apple container modified the read-only Torch staging "
                    "directory",
                    stage="pickle-worker",
                    details={"staging_root": str(staging_root)},
                )


def _docker_adb_server(
    server: str,
    *,
    host_alias: str | None = None,
) -> str:
    """Map every host-loopback spelling to Docker's host gateway alias."""

    canonical = canonicalize_adb_server(server)
    host, _, port = canonical.rpartition(":")
    if host == "localhost":
        return f"{host_alias or DEFAULT_CONSTRAINTS.docker_host_alias}:{port}"
    return server


def _apple_container_adb_server(
    server: str,
    *,
    host_alias: str | None = None,
) -> str:
    """Map host loopback to Apple container's configured localhost domain."""

    canonical = canonicalize_adb_server(server)
    host, _, port = canonical.rpartition(":")
    if host == "localhost":
        return (
            f"{host_alias or DEFAULT_CONSTRAINTS.apple_container_host_alias}:"
            f"{port}"
        )
    return server


def _emit(out: TextIO, payload: Any) -> None:
    out.write(json.dumps(payload, default=str) + "\n")
    out.flush()


def _worker_startup_timeout() -> float:
    value = os.environ.get(
        "QAIRT_AGENT_WORKER_STARTUP_TIMEOUT",
        str(_DEFAULT_WORKER_STARTUP_TIMEOUT),
    )
    try:
        timeout = float(value)
    except ValueError as exc:
        raise InvalidSpecError(
            "QAIRT_AGENT_WORKER_STARTUP_TIMEOUT must be a positive number",
            stage="worker_startup",
            details={"value": value},
        ) from exc
    if timeout <= 0:
        raise InvalidSpecError(
            "QAIRT_AGENT_WORKER_STARTUP_TIMEOUT must be a positive number",
            stage="worker_startup",
            details={"value": value},
        )
    return timeout


def _worker_log_tail(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError:
        return ""
    return payload[-_WORKER_LOG_TAIL_BYTES:].decode(
        "utf-8",
        errors="replace",
    )


def _stop_worker_process(process: Any) -> None:
    """Best-effort bounded stop of a worker that never claimed its journal."""

    with contextlib.suppress(Exception):
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


def _fail_worker_startup(
    *,
    journal: JobJournal,
    backend: str,
    process: Any,
    log_path: Path,
    message: str,
    returncode: int | None,
) -> None:
    if backend == "apple_container":
        code = ErrorCode.APPLE_CONTAINER_UNAVAILABLE
    elif backend == "docker":
        code = ErrorCode.DOCKER_UNAVAILABLE
    else:
        code = ErrorCode.STAGE_FAILED
    error = ToolErrorData(
        code=code,
        message=message,
        stage="worker_startup",
        retryable=True,
        details={
            "backend": backend,
            "pid": getattr(process, "pid", None),
            "returncode": returncode,
            "log_path": str(log_path),
            "log_tail": _worker_log_tail(log_path),
        },
    )
    current = journal.state()
    if not current.state.terminal:
        journal.set_state(
            JobState.FAILED,
            error=error,
            event_payload={
                "backend": backend,
                "worker_pid": getattr(process, "pid", None),
                "launch_log": str(log_path),
            },
        )
    raise ToolError(error)


def _await_worker_startup(
    *,
    journal: JobJournal,
    backend: str,
    process: Any,
    log_path: Path,
    timeout: float,
) -> None:
    """Require a detached worker to claim its journal within ``timeout``."""

    deadline = time.monotonic() + timeout
    while True:
        state = journal.state()
        if state.state is not JobState.QUEUED:
            return

        returncode = process.poll()
        if returncode is not None:
            _fail_worker_startup(
                journal=journal,
                backend=backend,
                process=process,
                log_path=log_path,
                message=(
                    f"{backend} worker exited before claiming job "
                    f"{journal.job_id}"
                ),
                returncode=int(returncode),
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _stop_worker_process(process)
            _fail_worker_startup(
                journal=journal,
                backend=backend,
                process=process,
                log_path=log_path,
                message=(
                    f"{backend} worker did not claim job {journal.job_id} "
                    f"within {timeout:g}s"
                ),
                returncode=process.poll(),
            )
        time.sleep(min(_WORKER_STARTUP_POLL_INTERVAL, remaining))


def _default_client(jobs_root: str | None) -> QairtAgentClient:
    project_root = project.find_project_root(os.getcwd())
    config = project.load(project_root) if project_root is not None else None
    constraints = (
        config.harness
        if config is not None
        else load_harness_constraints()
    )
    provenance = None
    if os.environ.get("QAIRT_AGENT_WORKER_PROVENANCE") == "1":
        provenance = StageProvenance(
            sdk_build=os.environ.get(
                "QAIRT_AGENT_SDK_BUILD", constraints.qairt_build_id
            ),
            adapter_capability=os.environ.get(
                "QAIRT_AGENT_ADAPTER_CAPABILITY", "explicit_factory"
            ),
            platform_abi=os.environ.get(
                "QAIRT_AGENT_PLATFORM_ABI",
                f"ubuntu{constraints.ubuntu_version}-"
                f"{constraints.platform_arch}",
            ),
            image_digest=os.environ.get("QAIRT_AGENT_IMAGE_DIGEST") or None,
            host_arch=os.environ.get("QAIRT_AGENT_HOST_ARCH"),
            emulation=os.environ.get("QAIRT_AGENT_EMULATION", "false").lower()
            in {"1", "true", "yes"},
            device_fingerprint=os.environ.get("QAIRT_AGENT_DEVICE_FINGERPRINT"),
        )
    if project_root is not None and config is not None:
        return QairtAgentClient.from_project(
            project_root,
            jobs_root=jobs_root or config.jobs_path,
            background=False,
            provenance=provenance,
        )
    return QairtAgentClient(
        jobs_root=jobs_root or ".qairt-agent/jobs",
        background=False,
        provenance=provenance,
    )


def _spawn_worker_impl(job_id: str, jobs_root: Path) -> int:
    """Launch a detached worker process after its journal is prepared.

    ``start_new_session`` detaches the worker so the calling CLI can exit (or be
    killed) without terminating the job.
    """

    actual_jobs_root = Path(jobs_root).expanduser().resolve()
    project_root = project.find_project_root(actual_jobs_root) or project.find_project_root(os.getcwd())
    if project_root is None:
        raise ProjectNotInitializedError(
            "detached workers require an initialized project; run 'qairt-agent init'",
            stage="worker",
        )
    config = project.load(project_root)
    constraints = config.harness
    backend = config.effective_worker_backend
    worker_args = [
        "-m",
        "qairt_agent.cli",
        "--jobs-root",
        str(actual_jobs_root),
        "_worker",
        "--job-id",
        job_id,
    ]
    env = os.environ.copy()
    cwd = str(config.project_root)
    worker_provenance_env = {
        "QAIRT_AGENT_WORKER_PROVENANCE": "1",
        "QAIRT_AGENT_SDK_BUILD": constraints.qairt_build_id,
        "QAIRT_AGENT_ADAPTER_CAPABILITY": "explicit_factory",
        "QAIRT_AGENT_HOST_ARCH": platform.machine(),
        "QAIRT_AGENT_PROJECT_ROOT": str(config.project_root),
        "QAIRT_AGENT_LEASES_DIR": str(config.state_path / "leases"),
        "QAIRT_AGENT_WORKER_BACKEND": backend,
    }
    adb_serial = env.get("QAIRT_AGENT_ADB_SERIAL")
    adb_server = env.get("QAIRT_AGENT_ADB_SERVER")
    if adb_serial and adb_server:
        canonical_adb_server = canonicalize_adb_server(adb_server)
        worker_provenance_env["QAIRT_AGENT_ADB_CANONICAL_SERVER"] = (
            canonical_adb_server
        )
        worker_provenance_env["QAIRT_AGENT_DEVICE_FINGERPRINT"] = (
            f"{adb_serial}@{canonical_adb_server}"
        )

    if backend == "native":
        env.update(worker_provenance_env)
        env["QAIRT_AGENT_PLATFORM_ABI"] = (
            f"ubuntu{constraints.ubuntu_version}-"
            f"{platform.machine().lower()}"
        )
        env["QAIRT_AGENT_EMULATION"] = "false"
        env["QAIRT_AGENT_HARNESS_CONSTRAINTS"] = str(
            config.harness_path
        )
        env["QAIRT_SDK_ROOT"] = str(config.sdk_path)
        env["QNN_SDK_ROOT"] = str(config.sdk_path)
        sdk_python = (
            str(config.sdk_path / "lib" / "python"),
            str(config.sdk_path / "benchmarks" / "QNN"),
        )
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (*sdk_python, env.get("PYTHONPATH")) if value
        )
        env["PATH"] = os.pathsep.join(
            value
            for value in (
                str(config.sdk_path / "bin" / "x86_64-linux-clang"),
                env.get("PATH"),
            )
            if value
        )
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            value
            for value in (
                str(config.sdk_path / "lib" / "x86_64-linux-clang"),
                env.get("LD_LIBRARY_PATH"),
            )
            if value
        )
        cmd = [sys.executable, *worker_args]
    elif backend in {"apple_container", "docker"}:
        image = WorkerImageConfig(
            image_ref=config.docker_image,
            platform=config.docker_platform,
            ubuntu_version=constraints.ubuntu_version,
            python_version=constraints.python_version,
        )
        runner: Any
        if backend == "apple_container":
            runner = AppleContainerRunner(
                image=image,
                constraints=constraints,
            )
        else:
            runner = DockerRunner(image=image, constraints=constraints)
        runner.require_available()
        image_identity = runner.require_image()
        if (
            backend == "apple_container"
            and adb_server
            and canonicalize_adb_server(adb_server).rpartition(":")[0]
            == "localhost"
        ):
            runner.require_host_alias(
                constraints.apple_container_host_alias
            )

        aliases = (
            BindMount(
                source=str(config.project_root),
                target=str(config.project_root),
                read_only=True,
            ),
            BindMount(source=str(config.state_path), target=str(config.state_path)),
            BindMount(source=str(actual_jobs_root), target=str(actual_jobs_root)),
            BindMount(source=str(config.artifacts_path), target=str(config.artifacts_path)),
            BindMount(source=str(config.cache_path), target=str(config.cache_path)),
        )
        mounts = default_mounts(
            str(config.state_path),
            str(config.artifacts_path),
            str(config.cache_path),
            sdk_root=str(config.sdk_path),
            workspace=str(config.project_root),
            jobs_volume=str(actual_jobs_root),
            models_root=str(config.models_path),
            compatibility_mounts=aliases,
        )
        container_env = {
            **worker_provenance_env,
            "QAIRT_SDK_ROOT": "/opt/qairt",
            "QNN_SDK_ROOT": "/opt/qairt",
            "PYTHONPATH": WORKER_PYTHONPATH,
            "QAIRT_AGENT_PROJECT_ROOT": "/workspace",
            "QAIRT_AGENT_LEASES_DIR": "/state/leases",
            "QAIRT_AGENT_IMAGE_DIGEST": image_identity,
            "QAIRT_AGENT_PLATFORM_ABI": (
                f"ubuntu{constraints.ubuntu_version}-"
                f"{constraints.platform_arch}"
            ),
            "QAIRT_AGENT_EMULATION": str(
                platform.machine().lower() not in {"x86_64", "amd64"}
            ).lower(),
            "HOME": "/tmp/qairt-agent-home",
            "XDG_CACHE_HOME": "/tmp/qairt-agent-cache",
            "QAIRT_AGENT_HARNESS_CONSTRAINTS": (
                f"/workspace/{config.harness_constraints}"
            ),
        }
        for name in ("QAIRT_AGENT_ADB_SERIAL", "QAIRT_AGENT_ADB_SERVER"):
            if value := env.get(name):
                container_env[name] = value
        server = container_env.get("QAIRT_AGENT_ADB_SERVER")
        if server:
            container_env["QAIRT_AGENT_ADB_SERVER"] = (
                _apple_container_adb_server(
                    server,
                    host_alias=constraints.apple_container_host_alias,
                )
                if backend == "apple_container"
                else _docker_adb_server(
                    server,
                    host_alias=constraints.docker_host_alias,
                )
            )

        run_kwargs = {
            "mounts": mounts,
            "command": [
                "/opt/venv/bin/python",
                *worker_args[:-5],
                "--jobs-root",
                str(actual_jobs_root),
                *worker_args[-3:],
            ],
            "platform": config.docker_platform,
            "workdir": str(config.project_root),
            "env": container_env,
            "user": (
                f"{os.getuid()}:{os.getgid()}"
                if hasattr(os, "getuid") and hasattr(os, "getgid")
                else None
            ),
        }
        if backend == "docker":
            run_kwargs["add_host_gateway"] = True
        cmd = runner.build_run_argv(**run_kwargs)
        env = None
    else:
        raise ValueError(f"unsupported worker backend {backend!r}")

    journal = JobJournal.open(actual_jobs_root, job_id)
    launch_log = journal.dir / "logs" / "worker-launch.log"
    launch_log.parent.mkdir(parents=True, exist_ok=True)
    startup_timeout = _worker_startup_timeout()
    with launch_log.open("ab", buffering=0) as log_stream:
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            cmd,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=cwd,
            env=env,
        )
    _await_worker_startup(
        journal=journal,
        backend=backend,
        process=process,
        log_path=launch_log,
        timeout=startup_timeout,
    )
    return process.pid


def _spawn_worker(job_id: str, jobs_root: Path) -> int:
    """Start a detached worker and never leave a failed launch queued."""

    actual_jobs_root = Path(jobs_root).expanduser().resolve()
    journal = JobJournal.open(actual_jobs_root, job_id)
    launch_log = journal.dir / "logs" / "worker-launch.log"
    try:
        return _spawn_worker_impl(job_id, actual_jobs_root)
    except Exception as exc:
        if isinstance(exc, ToolError):
            error = exc.to_tool_error()
        else:
            error = ToolErrorData.from_exception(
                exc,
                code=(
                    ErrorCode.INTERNAL_ERROR
                    if isinstance(exc, QairtAgentError)
                    else ErrorCode.STAGE_FAILED
                ),
                stage="worker_startup",
                retryable=True,
                details={
                    "job_id": job_id,
                    "log_path": str(launch_log),
                    "log_tail": _worker_log_tail(launch_log),
                },
            )
        current = journal.state()
        if not current.state.terminal:
            journal.set_state(
                JobState.FAILED,
                error=error,
                event_payload={
                    "launch_log": str(launch_log),
                    "startup_error": error.code.value,
                },
            )
        if isinstance(exc, ToolError) and exc.data == error:
            raise
        raise ToolError(error) from exc


def _follow(client: QairtAgentClient, job_id: str, out: TextIO, *, poll: float = 0.05) -> dict[str, Any]:
    """Stream events as JSONL until the job is terminal; return final status."""

    seen = 0
    while True:
        handle = client.job(job_id)
        for event in handle.events(after_seq=seen):
            _emit(out, event)
            seen = max(seen, int(event.get("seq", seen)))
        status = handle.status()
        if status.state.terminal:
            final = status.model_dump(mode="json")
            _emit(out, {"type": "final", "status": final})
            return final
        time.sleep(poll)


def _load_spec_dict(client: QairtAgentClient, spec_arg: str | None, from_job: str | None) -> Any:
    if spec_arg is not None:
        return spec_arg
    if from_job is not None:
        return client.job(from_job).journal.spec_original()
    return None


def _run_stage_command(
    client: QairtAgentClient,
    args: argparse.Namespace,
    stages: tuple[str, ...],
    out: TextIO,
    spawner: Any,
) -> int:
    command = args.command
    from_job = getattr(args, "from_job", None)
    if command in _REQUIRES_FROM_JOB and from_job is None:
        _emit(out, {"ok": False, "error": f"'{command}' requires --from-job <build job id>"})
        return 2

    spec = _load_spec_dict(client, args.spec, from_job)
    if spec is None:
        _emit(out, {"ok": False, "error": "a --spec is required"})
        return 2

    handle = client.prepare(spec, stages=stages, initial_manifest_job=from_job)

    if args.inline:
        status = client.execute(handle.job_id)
        _emit(out, {"job_id": handle.job_id, "state": status.state.value, "status_path": handle.status_path})
        if args.follow:
            _follow(client, handle.job_id, out)
        return 0 if status.state == JobState.SUCCEEDED else 1

    pid = spawner(handle.job_id, client.jobs_root)
    _emit(out, {**handle.submission(), "worker_pid": pid})
    if args.follow:
        _follow(client, handle.job_id, out)
    return 0


def _run_rerun(args: argparse.Namespace, client: QairtAgentClient, out: TextIO, spawner: Any) -> int:
    from_job = args.from_job
    spec = _load_spec_dict(client, args.spec, from_job)
    parent = client.job(from_job).journal
    parent_stages = tuple(parent.launcher().get("stages") or DEFAULT_WORKFLOW_STAGES)
    handle = client.prepare(
        spec,
        stages=parent_stages,
        parent_job_id=from_job,
        reuse_from_job=from_job,
    )
    if args.inline:
        status = client.execute(handle.job_id)
        _emit(out, {"job_id": handle.job_id, "state": status.state.value, "status_path": handle.status_path})
        if args.follow:
            _follow(client, handle.job_id, out)
        return 0 if status.state == JobState.SUCCEEDED else 1
    pid = spawner(handle.job_id, client.jobs_root)
    _emit(out, {**handle.submission(), "worker_pid": pid})
    if args.follow:
        _follow(client, handle.job_id, out)
    return 0


# --------------------------------------------------------------------------- #
# command handlers
# --------------------------------------------------------------------------- #


def _cmd_init(args: argparse.Namespace, client: QairtAgentClient, out: TextIO, spawner: Any) -> int:
    config = project.init(args.root)
    _emit(
        out,
        {
            "ok": True,
            "project_root": str(config.project_root),
            "config_path": str(project.config_path(args.root)),
            "sdk_root": config.sdk_root,
            "resolved_sdk_root": str(config.sdk_path),
            "worker_backend": config.effective_worker_backend,
            "harness_constraints": str(config.harness_path),
            "note": (
                "SDK discovery never moves an SDK. macOS uses Apple container; "
                "Linux uses Docker. Native execution is an explicit opt-in."
            ),
        },
    )
    return 0


def _cmd_doctor(args: argparse.Namespace, client: QairtAgentClient, out: TextIO, spawner: Any) -> int:
    report = project.doctor(args.root)
    _emit(out, report)
    return 0 if report["ok"] else 1


def _cmd_image(args: argparse.Namespace, client: QairtAgentClient, out: TextIO, spawner: Any) -> int:
    config = project.load(args.root)
    constraints = config.harness
    backend = config.effective_worker_backend
    if backend == "native":
        backend = project.select_container_backend()
    image = WorkerImageConfig(
        image_ref=config.docker_image,
        platform=config.docker_platform,
        ubuntu_version=constraints.ubuntu_version,
        python_version=constraints.python_version,
    )
    if backend == "apple_container":
        runner: Any = AppleContainerRunner(
            image=image,
            constraints=constraints,
        )
    elif backend == "docker":
        runner = DockerRunner(image=image, constraints=constraints)
    else:
        raise ValueError(f"image command does not support backend {backend!r}")
    if args.image_command == "build":
        project.ensure_worker_build_context(config.project_root, constraints)
        runner.build_image(
            context=config.project_root,
            dockerfile=config.dockerfile_path,
        )
    else:
        runner.require_available()
    image_id = runner.require_image()
    runner.smoke_test_sdk(sdk_root=config.sdk_path)
    _emit(
        out,
        {
            "ok": True,
            "action": args.image_command,
            "runtime": backend,
            "image": config.docker_image,
            "image_id": image_id,
            "platform": config.docker_platform,
            "dockerfile": str(config.dockerfile_path),
            "sdk_root": str(config.sdk_path),
            "smoke": "passed",
        },
    )
    return 0


def _cmd_plan(args: argparse.Namespace, client: QairtAgentClient, out: TextIO, spawner: Any) -> int:
    workflow_spec = client._normalize_spec(args.spec)  # noqa: SLF001 - CLI uses the normalizer
    resolved = resolve_workflow(workflow_spec)
    family = family_for_preset(workflow_spec.preset)
    effective_build = to_build_spec(workflow_spec)
    _emit(
        out,
        {
            "ok": True,
            "preset": workflow_spec.preset,
            "family": family.value if family else None,
            "resolved": resolved.to_dict(),
            "quality": workflow_spec.quality.model_dump(mode="json"),
            "effective_compile": effective_build.compile.model_dump(mode="json"),
            "effective_benchmark": effective_benchmark_policy(effective_build),
            "workflow_stages": list(DEFAULT_WORKFLOW_STAGES),
        },
    )
    return 0


def _cmd_job(args: argparse.Namespace, client: QairtAgentClient, out: TextIO, spawner: Any) -> int:
    action = args.job_command
    if action == "list":
        _emit(out, {"jobs": client.list_jobs()})
        return 0
    if action == "status":
        _emit(out, client.job(args.job_id).status().model_dump(mode="json"))
        return 0
    if action == "watch":
        if args.follow:
            _follow(client, args.job_id, out)
        else:
            for event in client.job(args.job_id).events(after_seq=args.after_seq):
                _emit(out, event)
        return 0
    if action == "cancel":
        client.job(args.job_id).cancel()
        _emit(out, {"ok": True, "job_id": args.job_id, "cancel_requested": True})
        return 0
    if action == "resume":
        if args.inline:
            status = client.execute(args.job_id)
            _emit(out, {"job_id": args.job_id, "state": status.state.value})
            return 0 if status.state == JobState.SUCCEEDED else 1
        handle = client.prepare_resume(args.job_id)
        if handle.status().state == JobState.SUCCEEDED:
            _emit(out, handle.submission())
            return 0
        pid = spawner(handle.job_id, client.jobs_root)
        _emit(out, {**handle.submission(), "worker_pid": pid})
        return 0
    _emit(out, {"ok": False, "error": f"unknown job action '{action}'"})
    return 2


def _cmd_vectors(args: argparse.Namespace, client: QairtAgentClient, out: TextIO, spawner: Any) -> int:
    if args.vectors_command != "import-pickle":
        _emit(out, {"ok": False, "error": f"unknown vectors action '{args.vectors_command}'"})
        return 2
    from qairt_agent.vectors_pickle import (
        detect_pickle_source_format,
        import_pickle_artifacts,
    )

    if (
        args.trusted_local
        and os.environ.get(_PICKLE_IMPORT_LOCAL_ENV) != "1"
        and detect_pickle_source_format(
            args.pickle_path,
            source_format=args.source_format,
        )
        == "torch"
    ):
        dispatched = _dispatch_torch_pickle_import(args)
        if dispatched is not None:
            _emit(out, dispatched)
            return 0

    imported = import_pickle_artifacts(
        args.pickle_path,
        output_dir=args.output_dir,
        trusted_local=args.trusted_local,
        bundle_id=args.bundle_id,
        case_id=args.case_id,
        isolate=args.isolate,
        source_format=args.source_format,
        section=args.section,
        source_key=(
            os.environ.get(_PICKLE_SOURCE_PATH_ENV)
            or str(Path(args.pickle_path).expanduser().resolve())
        ),
        expected_source_sha256=os.environ.get(
            _PICKLE_SOURCE_SHA256_ENV
        ),
    )
    _emit(
        out,
        {
            "ok": True,
            "bundle": imported.bundle.model_dump(mode="json"),
            "bundle_path": str(imported.bundle_path),
            "manifest_path": str(imported.manifest_path),
            "execution_ready": imported.execution_ready,
            "source_format": imported.source_format,
            "section": imported.section,
        },
    )
    return 0


def _dispatch_torch_pickle_import(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Run a Torch archive import in the configured Ubuntu worker.

    NumPy imports never use this path. Docker mounts the exact source file
    read-only. Apple container, which cannot bind a regular file, receives a
    private content-verified directory containing only ``archive.pt``. The
    exact output directory is mounted read-write, and the internal environment
    marker prevents the worker CLI from recursively dispatching itself.
    """

    project_root = project.find_project_root(os.getcwd())
    if project_root is None:
        raise ProjectNotInitializedError(
            "Torch archive import requires an initialized project so the pinned "
            "Ubuntu worker can be selected; run 'qairt-agent init --root .' and "
            "invoke the command from that project",
            stage="pickle-worker",
        )
    config = project.load(project_root)
    backend = config.effective_worker_backend
    if backend == "native":
        return None
    if backend not in {"apple_container", "docker"}:
        raise PickleRejectedError(
            f"Torch archive import does not support worker backend {backend!r}",
            stage="pickle-worker",
            details={"backend": backend},
        )

    source = Path(args.pickle_path).expanduser().resolve()
    if not source.is_file():
        raise PickleRejectedError(
            f"Torch archive source is not a file: {source}",
            stage="read",
            details={"path": str(source)},
        )
    source_sha256 = _verified_pickle_sha256(
        source,
        label="original Torch archive",
    )
    output = Path(args.output_dir).expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PickleRejectedError(
            f"Cannot create Torch import output directory {output}: {exc}",
            stage="materialize",
            details={"path": str(output), "error": str(exc)},
        ) from exc
    if not output.is_dir():
        raise PickleRejectedError(
            f"Torch import output is not a directory: {output}",
            stage="materialize",
            details={"path": str(output)},
        )

    constraints = config.harness
    image = WorkerImageConfig(
        image_ref=config.docker_image,
        platform=config.docker_platform,
        ubuntu_version=constraints.ubuntu_version,
        python_version=constraints.python_version,
    )
    runner: Any
    if backend == "apple_container":
        runner = AppleContainerRunner(image=image, constraints=constraints)
    else:
        runner = DockerRunner(image=image, constraints=constraints)
    runner.require_available()
    image_identity = runner.require_image()

    command = [
        "/opt/venv/bin/python",
        "-m",
        "qairt_agent.cli",
        "vectors",
        "import-pickle",
        _PICKLE_WORKER_SOURCE,
        "--output-dir",
        _PICKLE_WORKER_OUTPUT,
        "--trusted-local",
        "--format",
        "torch",
        "--section",
        str(args.section),
        "--case-id",
        str(args.case_id),
    ]
    if args.bundle_id is not None:
        command += ["--bundle-id", str(args.bundle_id)]
    if args.isolate:
        command.append("--isolate")

    env = {
        _PICKLE_IMPORT_LOCAL_ENV: "1",
        _PICKLE_SOURCE_PATH_ENV: str(source),
        _PICKLE_SOURCE_SHA256_ENV: source_sha256,
        "HOME": "/tmp/qairt-agent-home",
        "XDG_CACHE_HOME": "/tmp/qairt-agent-cache",
        "PYTHONPATH": WORKER_PYTHONPATH,
        "QAIRT_SDK_ROOT": "/opt/qairt",
        "QNN_SDK_ROOT": "/opt/qairt",
        "QAIRT_AGENT_HARNESS_CONSTRAINTS": (
            f"/workspace/{config.harness_constraints}"
        ),
    }
    with _pickle_worker_source_mount(
        source,
        backend=backend,
        source_sha256=source_sha256,
    ) as source_mount:
        mounts = default_mounts(
            str(config.state_path),
            str(config.artifacts_path),
            str(config.cache_path),
            sdk_root=str(config.sdk_path),
            workspace=str(config.project_root),
            models_root=str(config.models_path),
            compatibility_mounts=(
                BindMount(
                    source=str(output),
                    target=_PICKLE_WORKER_OUTPUT,
                    read_only=False,
                ),
                source_mount,
            ),
        )
        result = runner.run_build_isolated(
            mounts=mounts,
            command=command,
            platform=config.docker_platform,
            workdir="/workspace",
            env=env,
            user=(
                f"{os.getuid()}:{os.getgid()}"
                if hasattr(os, "getuid") and hasattr(os, "getgid")
                else None
            ),
        )
    stdout = (getattr(result, "stdout", "") or "").strip()
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise PickleRejectedError(
            "Torch import worker did not return exactly one structured JSON line",
            stage="pickle-worker",
            details={
                "backend": backend,
                "stdout": stdout[-4000:],
                "stderr": (getattr(result, "stderr", "") or "")[-4000:],
            },
        )
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise PickleRejectedError(
            "Torch import worker returned invalid JSON",
            stage="pickle-worker",
            details={"backend": backend, "stdout": lines[0][-4000:]},
        ) from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise PickleRejectedError(
            "Torch import worker returned an unsuccessful payload",
            stage="pickle-worker",
            details={"backend": backend, "payload": payload},
        )
    bundle_path = output / "vector_bundle.json"
    manifest_path = output / "vector_manifest.json"
    try:
        with bundle_path.open("r", encoding="utf-8") as stream:
            bundle_payload = json.load(stream)
        from qairt_agent.vectors import VectorPreparer

        manifest = VectorPreparer.load_manifest(manifest_path)
        VectorPreparer.load_tensors(manifest_path, section="inputs")
        VectorPreparer.load_tensors(manifest_path, section="goldens")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise PickleRejectedError(
            "Torch import worker result could not be reopened on the host",
            stage="pickle-worker",
            details={
                "backend": backend,
                "output_dir": str(output),
                "error": str(exc),
            },
        ) from exc
    bundle_source_sha256 = bundle_payload.get("source_sha256")
    manifest_source_sha256 = manifest.metadata.get("source_sha256")
    bundle_source_key = bundle_payload.get("source_key")
    manifest_source_path = manifest.metadata.get("source_path")
    if (
        bundle_source_sha256 != source_sha256
        or manifest_source_sha256 != source_sha256
        or bundle_source_key != str(source)
        or manifest_source_path != str(source)
    ):
        raise PickleRejectedError(
            "Torch import worker result is not bound to the verified host "
            "source archive",
            stage="pickle-worker",
            details={
                "host_source_path": str(source),
                "verified_source_sha256": source_sha256,
                "bundle_source_path": bundle_source_key,
                "bundle_source_sha256": bundle_source_sha256,
                "manifest_source_path": manifest_source_path,
                "manifest_source_sha256": manifest_source_sha256,
            },
        )
    return {
        **payload,
        "bundle": bundle_payload,
        "bundle_path": str(bundle_path),
        "manifest_path": str(manifest_path),
        "execution_backend": backend,
        "worker_image": config.docker_image,
        "worker_image_identity": image_identity,
        "source_path": str(source),
        "source_sha256": source_sha256,
    }


def _cmd_device(args: argparse.Namespace, client: QairtAgentClient, out: TextIO, spawner: Any) -> int:
    from qairt_agent.device import AdbClient, AdbConfig, device_doctor, device_gc

    if args.device_command == "doctor":
        config = AdbConfig.from_env()
        report = device_doctor(config, AdbClient(config))
        _emit(out, report)
        return 0 if report.get("ok") else 1
    if args.device_command == "gc":
        configured = args.leases_dir or os.environ.get("QAIRT_AGENT_LEASES_DIR")
        if configured is not None:
            leases_dir = Path(configured).expanduser().resolve()
        else:
            project_root = project.find_project_root(os.getcwd())
            leases_dir = (
                project.load(project_root).state_path / "leases"
                if project_root is not None
                else Path(".qairt-agent/leases").resolve()
            )
        client = None
        if not args.dry_run:
            config = AdbConfig.from_env()
            client = AdbClient(config)
        report = device_gc(leases_dir, client=client, dry_run=args.dry_run)
        _emit(out, report)
        return 0
    _emit(out, {"ok": False, "error": f"unknown device action '{args.device_command}'"})
    return 2


def _cmd_artifact(args: argparse.Namespace, client: QairtAgentClient, out: TextIO, spawner: Any) -> int:
    from qairt_agent.artifacts import verify_artifact
    from qairt_agent.contracts import ArtifactRef

    if args.artifact_command != "verify":
        _emit(out, {"ok": False, "error": f"unknown artifact action '{args.artifact_command}'"})
        return 2
    ref = ArtifactRef.from_path(args.path)
    if args.sha256 and ref.sha256 != args.sha256.lower():
        _emit(out, {"ok": False, "error": "sha256 mismatch", "expected": args.sha256, "actual": ref.sha256})
        return 1
    verify_artifact(ref)
    _emit(out, {"ok": True, "path": str(ref.path), "sha256": ref.sha256, "size_bytes": ref.size_bytes})
    return 0


def _cmd_worker(args: argparse.Namespace, client: QairtAgentClient, out: TextIO, spawner: Any) -> int:
    status = client.execute(args.job_id)
    _emit(out, {"job_id": args.job_id, "state": status.state.value})
    return 0 if status.state == JobState.SUCCEEDED else 1


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qairt-agent", description="QAIRT agent native workflow CLI")
    parser.add_argument(
        "--jobs-root", default=None, help="job journal root (default: project or .qairt-agent/jobs)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="initialize qairt-agent.toml and state dirs")
    p_init.add_argument("--root", default=os.getcwd())

    p_doctor = sub.add_parser("doctor", help="verify SDK metadata, ABI, and target")
    p_doctor.add_argument("--root", default=os.getcwd())

    p_image = sub.add_parser(
        "image",
        help="build or smoke-test the pinned worker image",
    )
    image_sub = p_image.add_subparsers(dest="image_command", required=True)
    p_image_build = image_sub.add_parser("build")
    p_image_build.add_argument("--root", default=os.getcwd())
    p_image_smoke = image_sub.add_parser("smoke")
    p_image_smoke.add_argument("--root", default=os.getcwd())

    p_plan = sub.add_parser("plan", help="resolve a spec into a preset/workflow plan")
    p_plan.add_argument("--spec", required=True)

    for command, stages in _STAGE_COMMANDS.items():
        stage_parser = sub.add_parser(command, help=f"submit a {command} job")
        stage_parser.add_argument("--spec", default=None)
        stage_parser.add_argument("--from-job", dest="from_job", default=None)
        stage_parser.add_argument("--follow", action="store_true")
        stage_parser.add_argument("--inline", action="store_true")

    p_workflow = sub.add_parser(
        "workflow",
        help="submit build->validate->benchmark; run diagnose from the failed job when needed",
    )
    p_workflow.add_argument("--spec", required=True)
    p_workflow.add_argument("--from-job", dest="from_job", default=None)
    p_workflow.add_argument("--follow", action="store_true")
    p_workflow.add_argument("--inline", action="store_true")

    p_rerun = sub.add_parser("rerun", help="rerun a job, reusing unchanged stages")
    p_rerun.add_argument("--from-job", dest="from_job", required=True)
    p_rerun.add_argument("--spec", default=None)
    p_rerun.add_argument("--follow", action="store_true")
    p_rerun.add_argument("--inline", action="store_true")

    p_job = sub.add_parser("job", help="inspect/control jobs")
    job_sub = p_job.add_subparsers(dest="job_command", required=True)
    job_sub.add_parser("list")
    j_status = job_sub.add_parser("status")
    j_status.add_argument("job_id")
    j_watch = job_sub.add_parser("watch")
    j_watch.add_argument("job_id")
    j_watch.add_argument("--after-seq", dest="after_seq", type=int, default=0)
    j_watch.add_argument("--follow", action="store_true")
    j_cancel = job_sub.add_parser("cancel")
    j_cancel.add_argument("job_id")
    j_resume = job_sub.add_parser("resume")
    j_resume.add_argument("job_id")
    j_resume.add_argument("--inline", action="store_true")

    p_vectors = sub.add_parser("vectors", help="vector utilities")
    vectors_sub = p_vectors.add_subparsers(dest="vectors_command", required=True)
    p_pickle = vectors_sub.add_parser("import-pickle", help="restricted pickle import (opt-in)")
    p_pickle.add_argument("pickle_path")
    p_pickle.add_argument("--output-dir", dest="output_dir", required=True)
    p_pickle.add_argument("--trusted-local", dest="trusted_local", action="store_true")
    p_pickle.add_argument("--bundle-id", dest="bundle_id", default=None)
    p_pickle.add_argument("--case-id", dest="case_id", default="pickle-import")
    p_pickle.add_argument("--isolate", action="store_true")
    p_pickle.add_argument(
        "--format",
        dest="source_format",
        choices=("auto", "numpy-pickle", "torch"),
        default="auto",
        help=(
            "input serialization; auto recognizes modern torch.save zip archives "
            "and otherwise uses the restricted NumPy pickle loader"
        ),
    )
    p_pickle.add_argument(
        "--section",
        choices=("auto", "inputs", "goldens"),
        default="auto",
        help=(
            "section assignment; explicit inputs/goldens treats the whole pickle "
            "as that section"
        ),
    )

    p_device = sub.add_parser("device", help="ADB device lifecycle")
    device_sub = p_device.add_subparsers(dest="device_command", required=True)
    device_sub.add_parser("doctor")
    p_gc = device_sub.add_parser("gc")
    p_gc.add_argument("--leases-dir", dest="leases_dir", default=None)
    p_gc.add_argument("--dry-run", dest="dry_run", action="store_true")

    p_artifact = sub.add_parser("artifact", help="artifact utilities")
    artifact_sub = p_artifact.add_subparsers(dest="artifact_command", required=True)
    p_verify = artifact_sub.add_parser("verify")
    p_verify.add_argument("path")
    p_verify.add_argument("--sha256", default=None)

    sub.add_parser("_worker", help=argparse.SUPPRESS).add_argument("--job-id", dest="job_id", required=True)
    return parser


_HANDLERS = {
    "init": _cmd_init,
    "doctor": _cmd_doctor,
    "image": _cmd_image,
    "plan": _cmd_plan,
    "workflow": lambda a, c, o, s: _run_stage_command(c, a, DEFAULT_WORKFLOW_STAGES, o, s),
    "rerun": _run_rerun,
    "job": _cmd_job,
    "vectors": _cmd_vectors,
    "device": _cmd_device,
    "artifact": _cmd_artifact,
    "_worker": _cmd_worker,
}
for _command, _stages in _STAGE_COMMANDS.items():
    _HANDLERS[_command] = (lambda stages: lambda a, c, o, s: _run_stage_command(c, a, stages, o, s))(_stages)


def main(
    argv: list[str] | None = None,
    *,
    client: QairtAgentClient | None = None,
    out: TextIO | None = None,
    spawner: Any = _spawn_worker,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = out or sys.stdout
    try:
        client = client or _default_client(args.jobs_root)
        handler = _HANDLERS[args.command]
        return handler(args, client, out, spawner)
    except QairtAgentError as exc:
        _emit(out, {"ok": False, "error": exc.to_tool_error().as_dict()})
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
