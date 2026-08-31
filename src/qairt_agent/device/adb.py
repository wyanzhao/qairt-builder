"""ADB device transfer and lifecycle primitives.

ADB is used *only* for moving files to/from a device and for device lifecycle
(probing, staging, cleanup).  All QAIRT transform/build/runtime work happens
through the Python API elsewhere in this package.

Safety invariants enforced here:

* A device is never auto-selected.  ``AdbConfig.from_env`` fails closed unless
  both ``QAIRT_AGENT_ADB_SERIAL`` and ``QAIRT_AGENT_ADB_SERVER`` are set.
* Remote work lives under ``/data/local/tmp/qairt-agent/<job>/<stage>/<attempt>/``
  and cleanup only ever deletes one exact attempt directory (``remove_exact``);
  broad recursive deletes are refused.
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import os
import re
import shlex
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from pydantic import field_validator

from qairt_agent.contracts import FrozenContract
from qairt_agent.errors import ArtifactIntegrityError, DeviceUnavailableError
from qairt_agent.harness import load_harness_constraints

__all__ = [
    "REMOTE_ROOT",
    "AdbConfig",
    "AdbClient",
    "AttemptSession",
    "canonicalize_adb_server",
    "parse_remote_attempt_dir",
    "remote_attempt_dir",
]

#: Root under which every remote attempt directory must live.
REMOTE_ROOT = "/data/local/tmp/qairt-agent"

ENV_SERIAL = "QAIRT_AGENT_ADB_SERIAL"
ENV_SERVER = "QAIRT_AGENT_ADB_SERVER"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
def canonicalize_adb_server(server: str) -> str:
    """Return the stable lease identity for an ADB ``host:port`` address.

    This does not change the address used by :class:`AdbClient`.  It only
    collapses host-loopback spellings (including Docker and Apple container
    host aliases) so native and container workers contend on the same lease.
    """

    text = (server or "").strip()
    host, separator, port = text.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise ValueError("server must be in host:port form")
    normalized_host = host.strip().lower().rstrip(".")
    if normalized_host.startswith("[") and normalized_host.endswith("]"):
        normalized_host = normalized_host[1:-1]
    constraints = load_harness_constraints()
    container_host_aliases = {
        constraints.docker_host_alias.lower().rstrip("."),
        constraints.apple_container_host_alias.lower().rstrip("."),
    }
    loopback = normalized_host == "localhost" or (
        normalized_host in container_host_aliases
    )
    if not loopback:
        try:
            loopback = ipaddress.ip_address(normalized_host).is_loopback
        except ValueError:
            pass
    if loopback:
        normalized_host = "localhost"
    return f"{normalized_host}:{int(port)}"


def _validate_component(name: str, value: Any) -> str:
    """Validate one path component: non-empty and free of ``/`` and ``..``."""

    if value is None:
        raise ValueError(f"{name} cannot be empty")
    original = str(value)
    text = original.strip()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    if text != original:
        raise ValueError(f"{name} must not contain leading or trailing whitespace")
    if text == "." or ".." in text or not _SAFE_COMPONENT_RE.fullmatch(text):
        raise ValueError(
            f"{name} must match [A-Za-z0-9._-]+ and cannot be '.' or contain '..'"
        )
    return text


def remote_attempt_dir(job_id: Any, stage_key: Any, attempt_id: Any) -> str:
    """Return the exact remote attempt directory for one stage attempt.

    The layout is fixed::

        /data/local/tmp/qairt-agent/<job-id>/<stage-key>/<attempt-id>/

    Each component must be non-empty and contain neither ``/`` nor ``..`` so the
    resulting path can never escape ``REMOTE_ROOT``.
    """

    job = _validate_component("job_id", job_id)
    stage = _validate_component("stage_key", stage_key)
    attempt = _validate_component("attempt_id", attempt_id)
    return f"{REMOTE_ROOT}/{job}/{stage}/{attempt}/"


def parse_remote_attempt_dir(remote_dir: str) -> tuple[str, str, str]:
    """Validate and split one exact three-level attempt directory.

    Accepted paths have exactly the form
    ``REMOTE_ROOT/<job>/<stage>/<attempt>/``.  This single parser is shared by
    lease bookkeeping and destructive ADB cleanup.
    """

    if not isinstance(remote_dir, str):
        raise ValueError("remote attempt directory must be a string")
    prefix = REMOTE_ROOT + "/"
    if not remote_dir.startswith(prefix) or not remote_dir.endswith("/"):
        raise ValueError(
            "remote attempt directory must exactly match "
            f"{REMOTE_ROOT}/<job>/<stage>/<attempt>/: {remote_dir!r}"
        )
    parts = remote_dir[len(prefix) : -1].split("/")
    if len(parts) != 3:
        raise ValueError(
            "remote attempt directory must contain exactly job/stage/attempt: "
            f"{remote_dir!r}"
        )
    job, stage, attempt = (
        _validate_component(name, value)
        for name, value in zip(("job_id", "stage_key", "attempt_id"), parts)
    )
    expected = remote_attempt_dir(job, stage, attempt)
    if remote_dir != expected:
        raise ValueError(f"non-canonical remote attempt directory: {remote_dir!r}")
    return job, stage, attempt


def _local_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class AdbConfig(FrozenContract):
    """An explicitly-selected target device.

    ``server`` is a ``host:port`` ADB server address; ``serial`` is the exact
    device serial.  Nothing here is inferred from the connected device list.
    """

    serial: str
    server: str

    @field_validator("serial")
    @classmethod
    def _validate_serial(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("serial cannot be blank")
        return value

    @field_validator("server")
    @classmethod
    def _validate_server(cls, value: str) -> str:
        host, sep, port = (value or "").rpartition(":")
        if not sep or not host or not port.isdigit():
            raise ValueError("server must be in host:port form")
        return value

    @classmethod
    def from_env(cls, environ: Any | None = None) -> "AdbConfig":
        """Build a config from the fixed env vars, failing closed if absent.

        A device is never auto-selected: if either variable is missing or blank
        a :class:`DeviceUnavailableError` is raised rather than guessing the
        "only" attached device.
        """

        env = os.environ if environ is None else environ
        serial = (env.get(ENV_SERIAL) or "").strip()
        server = (env.get(ENV_SERVER) or "").strip()
        missing = [name for name, value in ((ENV_SERIAL, serial), (ENV_SERVER, server)) if not value]
        if missing:
            raise DeviceUnavailableError(
                "no QAIRT device is auto-selected; the environment variables "
                f"{ENV_SERIAL} and {ENV_SERVER} (host:port) are required to target a device",
                stage="device",
                retryable=False,
                details={"missing_env": missing},
            )
        return cls(serial=serial, server=server)

    def _host_port(self) -> tuple[str, str]:
        host, _, port = self.server.rpartition(":")
        return host, port

    @property
    def device_identifier(self) -> str:
        """The QAIRT device fingerprint form ``<serial>@<host:port>``."""

        return f"{self.serial}@{self.server}"

    @property
    def adb_server_arg(self) -> str:
        host, port = self._host_port()
        return f"-H {host} -P {port}"

    def adb_server_argv(self) -> list[str]:
        """argv prefix targeting the ADB server only (no ``-s`` serial)."""

        host, port = self._host_port()
        return ["adb", "-H", host, "-P", port]

    def adb_base_argv(self) -> list[str]:
        """argv prefix targeting the server *and* the exact serial."""

        return [*self.adb_server_argv(), "-s", self.serial]


@dataclass(frozen=True)
class AttemptSession:
    """Handle yielded by :meth:`AdbClient.stage_attempt`."""

    attempt_dir: str
    incoming_dir: str
    ready_dir: str
    files: tuple[str, ...]


class AdbClient:
    """Thin, injectable wrapper over the ``adb`` CLI.

    ``command_executor`` is a ``list[str] -> CompletedProcess-like`` callable;
    tests inject a fake so no real device or ``adb`` binary is ever required.
    """

    def __init__(
        self,
        config: AdbConfig,
        *,
        command_executor: Callable[[list[str]], Any] | None = None,
    ) -> None:
        self._config = config
        self._executor = command_executor or self._default_executor

    @property
    def config(self) -> AdbConfig:
        return self._config

    @staticmethod
    def _default_executor(argv: list[str]) -> Any:
        return subprocess.run(argv, capture_output=True, text=True, check=False)

    # ------------------------------------------------------------------ #
    # execution helpers
    # ------------------------------------------------------------------ #

    def _execute(self, argv: list[str]) -> Any:
        return self._executor(list(argv))

    def _run(self, argv: list[str], *, stage: str = "device") -> Any:
        result = self._execute(argv)
        returncode = getattr(result, "returncode", 0)
        if returncode != 0:
            raise DeviceUnavailableError(
                f"adb command failed with exit code {returncode}: {shlex.join(list(argv))}",
                stage=stage,
                retryable=True,
                details={
                    "argv": list(argv),
                    "returncode": returncode,
                    "stdout": getattr(result, "stdout", "") or "",
                    "stderr": getattr(result, "stderr", "") or "",
                },
            )
        return result

    # ------------------------------------------------------------------ #
    # device operations
    # ------------------------------------------------------------------ #

    def devices(self) -> list[str]:
        """List serials known to the ADB server (runs without ``-s``)."""

        result = self._run([*self._config.adb_server_argv(), "devices"])
        serials: list[str] = []
        for raw_line in (getattr(result, "stdout", "") or "").splitlines():
            line = raw_line.strip()
            if not line or line.lower().startswith("list of devices"):
                continue
            parts = line.split()
            if parts:
                serials.append(parts[0])
        return serials

    def device_state(self) -> str:
        """Return the state of the configured serial (e.g. ``device``)."""

        result = self._run([*self._config.adb_base_argv(), "get-state"])
        return (getattr(result, "stdout", "") or "").strip()

    def shell(self, command: str) -> Any:
        argv = [*self._config.adb_base_argv(), "shell", *shlex.split(command)]
        return self._run(argv)

    def push(self, local: str, remote: str) -> None:
        self._run([*self._config.adb_base_argv(), "push", local, remote])

    def pull(self, remote: str, local: str) -> None:
        self._run([*self._config.adb_base_argv(), "pull", remote, local])

    def remote_sha256(self, remote_path: str) -> str:
        result = self.shell(f"sha256sum {remote_path}")
        tokens = (getattr(result, "stdout", "") or "").split()
        if not tokens:
            raise DeviceUnavailableError(
                f"no sha256sum output for {remote_path}",
                stage="device",
                details={"remote_path": remote_path},
            )
        digest = tokens[0].lower()
        if not _SHA256_RE.fullmatch(digest):
            raise DeviceUnavailableError(
                f"invalid sha256sum output for {remote_path}: {tokens[0]!r}",
                stage="device",
                details={"remote_path": remote_path, "stdout": getattr(result, "stdout", "") or ""},
            )
        return digest

    def read_soc_id(self) -> dict[str, Any]:
        """Read the Android SoC ID the handset reports.

        The registry records each target's ``soc_id`` list precisely so a
        report can never publish under a target identity the hardware
        contradicts. This is the read half: a pure adb property read, no SDK
        involvement. ``ro.soc.id`` is the modern property; the sysfs node is
        the fallback for handsets that do not export it.

        Returns the parsed ``soc_id`` when one could be read, plus every raw
        source output so an unreadable value stays inspectable instead of
        becoming an unexplained absence.
        """

        sources: list[dict[str, Any]] = []
        soc_id: int | None = None
        for kind, command in (
            ("getprop ro.soc.id", "getprop ro.soc.id"),
            ("getprop ro.soc.model", "getprop ro.soc.model"),
            ("sysfs", "cat /sys/devices/soc0/soc_id"),
        ):
            try:
                result = self.shell(command)
            except Exception as error:  # noqa: BLE001 - absence is a warning
                sources.append({"source": kind, "error": str(error)})
                continue
            raw = (getattr(result, "stdout", "") or "").strip()
            sources.append({"source": kind, "raw": raw})
            if soc_id is None and raw.isdigit():
                soc_id = int(raw)
        return {"soc_id": soc_id, "sources": sources}

    def remote_exists(self, remote_path: str) -> bool:
        result = self._execute([*self._config.adb_base_argv(), "shell", "test", "-e", remote_path])
        return getattr(result, "returncode", 1) == 0

    def remove_exact(self, remote_dir: str) -> None:
        """``rm -rf`` exactly one attempt directory under ``REMOTE_ROOT``.

        Refuses anything except the exact three-level attempt layout so a buggy
        or malicious caller can never trigger a broad delete.
        """

        parse_remote_attempt_dir(remote_dir)
        self._run([*self._config.adb_base_argv(), "shell", "rm", "-rf", remote_dir])

    # ------------------------------------------------------------------ #
    # staging
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def stage_attempt(
        self,
        job_id: Any,
        stage_key: Any,
        attempt_id: Any,
        *,
        push_files: dict[str, str],
        cleanup_callback: Callable[[str], None] | None = None,
    ) -> Iterator[AttemptSession]:
        """Stage local files into a fresh attempt directory, then always clean up.

        Files are pushed to ``<dir>/incoming/<name>``, each verified by sha256
        against its local source, then atomically marked ready by moving
        ``incoming`` to ``ready``.  In *all* terminal states (success, failure,
        cancel) the ``finally`` removes only the exact attempt directory.
        """

        attempt_dir = remote_attempt_dir(job_id, stage_key, attempt_id)
        incoming_dir = attempt_dir + "incoming"
        ready_dir = attempt_dir + "ready"
        names = tuple(sorted(push_files))
        for name in names:
            _validate_component("file name", name)
        try:
            self.shell(f"mkdir -p {incoming_dir}")
            for name in names:
                local_path = push_files[name]
                remote_path = f"{incoming_dir}/{name}"
                self.push(local_path, remote_path)
                local_sha = _local_sha256(local_path)
                remote_sha = self.remote_sha256(remote_path)
                if remote_sha != local_sha:
                    raise ArtifactIntegrityError(
                        f"remote sha256 mismatch while staging {name}",
                        stage="device",
                        details={
                            "remote_path": remote_path,
                            "expected_sha256": local_sha,
                            "actual_sha256": remote_sha,
                        },
                    )
            self.shell(f"mv {incoming_dir} {ready_dir}")
            yield AttemptSession(
                attempt_dir=attempt_dir,
                incoming_dir=incoming_dir,
                ready_dir=ready_dir,
                files=names,
            )
        finally:
            self.remove_exact(attempt_dir)
            if cleanup_callback is not None:
                cleanup_callback(attempt_dir)
