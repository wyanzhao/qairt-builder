"""Exclusive, content-verified staging for QAIRT Python device execution."""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from qairt_agent.device.adb import (
    AdbClient,
    AdbConfig,
    AttemptSession,
    canonicalize_adb_server,
    remote_attempt_dir,
)
from qairt_agent.device.lease import DeviceLease
from qairt_agent.errors import DeviceUnavailableError

ENV_LEASES_DIR = "QAIRT_AGENT_LEASES_DIR"
ENV_ADB_CANONICAL_SERVER = "QAIRT_AGENT_ADB_CANONICAL_SERVER"

__all__ = [
    "DeviceRuntime",
    "DeviceStageSession",
    "ENV_ADB_CANONICAL_SERVER",
    "ENV_LEASES_DIR",
]


@dataclass(frozen=True)
class DeviceStageSession:
    """One exclusive QAIRT device stage and its exact remote sandbox."""

    device: Any
    adb: AttemptSession
    identifier: str


class DeviceRuntime:
    """Build a QAIRT ``Device`` and guard one staged execution attempt.

    All dependencies are injectable so tests can exercise lifecycle semantics
    without an ADB daemon or QAIRT installation.  The default configuration is
    intentionally fail-closed through :meth:`AdbConfig.from_env`.
    """

    def __init__(
        self,
        *,
        config_factory: Callable[[], AdbConfig] | None = None,
        adb_client_factory: Callable[[AdbConfig], AdbClient] | None = None,
        lease_factory: Callable[[Path, str, str, str], DeviceLease] | None = None,
        device_factory: Callable[[Any, AdbConfig], Any] | None = None,
        leases_dir: str | Path | None = None,
    ) -> None:
        self._config_factory = config_factory or AdbConfig.from_env
        self._adb_client_factory = adb_client_factory or AdbClient
        self._lease_factory = lease_factory or (
            lambda root, server, serial, owner: DeviceLease(
                root, server, serial, owner
            )
        )
        self._device_factory = device_factory or self._default_device_factory
        self._leases_dir = (
            Path(leases_dir).expanduser().resolve()
            if leases_dir is not None
            else None
        )

    @staticmethod
    def _default_device_factory(adapter: Any, config: AdbConfig) -> Any:
        creator = getattr(adapter, "create_device", None)
        if not callable(creator):
            raise TypeError(
                "QAIRT adapter must implement create_device(serial=..., server=...)"
            )
        return creator(serial=config.serial, server=config.server)

    @contextlib.contextmanager
    def stage(
        self,
        adapter: Any,
        *,
        output_root: str | Path,
        job_id: str,
        stage_key: str,
        attempt_id: str,
        push_files: Mapping[str, str | Path],
    ) -> Iterator[DeviceStageSession]:
        """Acquire, record, stage, yield, and exactly clean one attempt.

        The attempt path is recorded in the lease before ``mkdir``/``push``.
        It is forgotten only after :meth:`AdbClient.remove_exact` succeeds.
        A cleanup failure intentionally leaves the lease record intact for
        ``device gc`` rather than losing the only recovery pointer.
        """

        config = self._config_factory()
        client = self._adb_client_factory(config)
        attempt_dir = remote_attempt_dir(job_id, stage_key, attempt_id)
        configured_lease_root = (os.environ.get(ENV_LEASES_DIR) or "").strip()
        project_root = (
            os.environ.get("QAIRT_AGENT_PROJECT_ROOT") or ""
        ).strip()
        lease_root = (
            self._leases_dir
            or (
                Path(configured_lease_root).expanduser().resolve()
                if configured_lease_root
                else (
                    Path(project_root).expanduser().resolve()
                    if project_root
                    else Path.cwd().resolve()
                )
                / ".qairt-agent"
                / "leases"
            )
        )
        owner = f"{job_id}:{stage_key}:{attempt_id}"
        actual_lease_server = canonicalize_adb_server(config.server)
        configured_canonical_server = (
            os.environ.get(ENV_ADB_CANONICAL_SERVER) or ""
        ).strip()
        if configured_canonical_server:
            try:
                declared_lease_server = canonicalize_adb_server(
                    configured_canonical_server
                )
            except ValueError as exc:
                raise DeviceUnavailableError(
                    "canonical ADB server identity is invalid",
                    stage="device",
                    retryable=False,
                    details={"canonical_server": configured_canonical_server},
                ) from exc
            if declared_lease_server != actual_lease_server:
                raise DeviceUnavailableError(
                    "canonical ADB server identity does not match the actual "
                    "ADB connection address",
                    stage="device",
                    retryable=False,
                    details={
                        "canonical_server": declared_lease_server,
                        "actual_server": actual_lease_server,
                    },
                )
        lease_server = actual_lease_server
        lease = self._lease_factory(
            lease_root,
            lease_server,
            config.serial,
            owner,
        )
        lease.acquire()
        try:
            lease.record_attempt_dir(attempt_dir)
        except BaseException:
            # record_attempt_dir is atomic.  If it failed before publishing an
            # attempt pointer, release the otherwise empty lease; if a custom
            # implementation did publish one, release() deliberately retains
            # it for GC.
            lease.release()
            raise

        pending: tuple[
            BaseException,
            TracebackType | None,
        ] | None = None

        def cleanup_confirmed(cleaned_dir: str) -> None:
            lease.forget_attempt_dir(cleaned_dir)

        normalized_files = {
            str(name): str(Path(path).expanduser().resolve())
            for name, path in push_files.items()
        }
        try:
            with client.stage_attempt(
                job_id,
                stage_key,
                attempt_id,
                push_files=normalized_files,
                cleanup_callback=cleanup_confirmed,
            ) as adb_session:
                device = self._device_factory(adapter, config)
                try:
                    yield DeviceStageSession(
                        device=device,
                        adb=adb_session,
                        identifier=config.device_identifier,
                    )
                except BaseException as exc:
                    # Let stage_attempt finish its exact cleanup first, then
                    # re-raise the original operation failure with traceback.
                    pending = (exc, sys.exc_info()[2])
        finally:
            # release() always stops the independent heartbeat.  It removes a
            # clean lease, but deliberately retains one whose exact attempt
            # pointer survived a cleanup failure so GC can recover it.
            lease.release()

        if pending is not None:
            exc, traceback = pending
            raise exc.with_traceback(traceback)
