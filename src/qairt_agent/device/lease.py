"""Crash-safe, cross-process device leases.

Each server+serial pair has one atomically-created owner record.  A separate
Python sidecar process refreshes a unique heartbeat file, so a QAIRT extension
holding the worker's GIL cannot make a live lease look stale.  GC uses the
heartbeat rather than a container PID, then revalidates the owner token and
record CAS while holding the same per-device lock used by acquisition.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qairt_agent.contracts import utc_now
from qairt_agent.device.adb import (
    canonicalize_adb_server,
    parse_remote_attempt_dir,
)
from qairt_agent.errors import LeaseConflictError

LEASE_HEARTBEAT_INTERVAL_SECONDS = 5.0
LEASE_STALE_AFTER_SECONDS = 30.0
INVALID_LEASE_GRACE_SECONDS = 30.0
MAX_LEASE_HEARTBEAT_INTERVAL_SECONDS = 60.0

__all__ = [
    "DeviceLease",
    "INVALID_LEASE_GRACE_SECONDS",
    "LEASE_HEARTBEAT_INTERVAL_SECONDS",
    "LEASE_STALE_AFTER_SECONDS",
    "LeaseSnapshot",
    "lease_file_lock",
    "lease_snapshot",
    "list_stale_leases",
    "scan_stale_lease_snapshots",
]


def _default_alive(pid: int) -> bool:
    """Return True if ``pid`` is alive; treat PermissionError as alive."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_path(owner_path: Path) -> Path:
    return owner_path.with_name(f".{owner_path.stem}.gc.lock")


def _fsync_directory(directory: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(directory, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError as exc:
        unsupported = {
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if exc.errno not in unsupported:
            raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _publish_noclobber(path: Path, data: bytes) -> None:
    """Publish complete bytes atomically without ever exposing a partial file."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard-link publication is atomic and never replaces an
        # existing owner.  The temporary inode is complete before it is visible
        # at the canonical lease path.
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


@contextlib.contextmanager
def lease_file_lock(owner_path: str | Path) -> Iterator[None]:
    """Serialize acquisition, owner mutation, and GC for one device."""

    path = Path(owner_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(_lock_path(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _heartbeat_path(owner_path: Path, data: dict[str, Any]) -> Path | None:
    owner_token = data.get("owner_token")
    heartbeat_file = data.get("heartbeat_file")
    if (
        not isinstance(owner_token, str)
        or len(owner_token) != 32
        or any(character not in "0123456789abcdef" for character in owner_token)
        or not isinstance(heartbeat_file, str)
    ):
        return None
    expected = f".{owner_path.stem}.{owner_token}.heartbeat"
    if heartbeat_file != expected or Path(heartbeat_file).name != heartbeat_file:
        return None
    return owner_path.parent / heartbeat_file


def _valid_base_record(data: dict[str, Any]) -> bool:
    if not all(
        isinstance(data.get(field), str) and bool(data[field])
        for field in ("owner", "server", "serial")
    ):
        return False
    try:
        canonicalize_adb_server(data["server"])
    except ValueError:
        return False
    attempt_dirs = data.get("attempt_dirs")
    if not isinstance(attempt_dirs, list):
        return False
    try:
        for attempt_dir in attempt_dirs:
            parse_remote_attempt_dir(attempt_dir)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class LeaseSnapshot:
    """One immutable owner-file observation used for stale-scan CAS."""

    path: Path
    data: dict[str, Any] | None
    cas_token: str
    owner_token: str | None
    heartbeat_path: Path | None
    stale: bool
    stale_reason: str | None


def lease_snapshot(
    path: str | Path,
    *,
    alive: Callable[[int], bool] | None = None,
    now: float | None = None,
    stale_after: float = LEASE_STALE_AFTER_SECONDS,
    invalid_grace_after: float = INVALID_LEASE_GRACE_SECONDS,
) -> LeaseSnapshot | None:
    """Read and classify one lease without mutating it."""

    if stale_after <= 0:
        raise ValueError("stale_after must be positive")
    if invalid_grace_after <= 0:
        raise ValueError("invalid_grace_after must be positive")
    owner_path = Path(path)
    try:
        owner_stat = owner_path.stat()
        raw = owner_path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        return None

    observed_at = time.time() if now is None else float(now)
    cas_digest = hashlib.sha256(raw).hexdigest()
    cas_token = (
        f"{owner_stat.st_ino}:{owner_stat.st_mtime_ns}:"
        f"{owner_stat.st_size}:{cas_digest}"
    )
    try:
        parsed = json.loads(raw.decode("utf-8"))
        data = parsed if isinstance(parsed, dict) else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        data = None

    owner_age = max(0.0, observed_at - owner_stat.st_mtime)
    if data is None:
        stale = owner_age > invalid_grace_after
        return LeaseSnapshot(
            path=owner_path,
            data=None,
            cas_token=cas_token,
            owner_token=None,
            heartbeat_path=None,
            stale=stale,
            stale_reason="invalid_owner_record" if stale else None,
        )

    if not _valid_base_record(data):
        stale = owner_age > invalid_grace_after
        return LeaseSnapshot(
            path=owner_path,
            data=None,
            cas_token=cas_token,
            owner_token=None,
            heartbeat_path=None,
            stale=stale,
            stale_reason="invalid_owner_record" if stale else None,
        )

    owner_token = data.get("owner_token")
    heartbeat = _heartbeat_path(owner_path, data)
    if heartbeat is not None:
        recorded_interval = data.get("heartbeat_interval_seconds")
        effective_stale_after = stale_after
        if (
            isinstance(recorded_interval, (int, float))
            and not isinstance(recorded_interval, bool)
            and math.isfinite(float(recorded_interval))
            and recorded_interval > 0
        ):
            effective_stale_after = max(
                stale_after,
                min(
                    float(recorded_interval),
                    MAX_LEASE_HEARTBEAT_INTERVAL_SECONDS,
                )
                * 3.0,
            )
        try:
            heartbeat_age = max(0.0, observed_at - heartbeat.stat().st_mtime)
            stale = heartbeat_age > effective_stale_after
            reason = "heartbeat_stale" if stale else None
        except OSError:
            stale = owner_age > invalid_grace_after
            reason = "heartbeat_missing" if stale else None
        return LeaseSnapshot(
            path=owner_path,
            data=data,
            cas_token=cas_token,
            owner_token=owner_token if isinstance(owner_token, str) else None,
            heartbeat_path=heartbeat,
            stale=stale,
            stale_reason=reason,
        )

    # Backward compatibility for pre-heartbeat records.  New leases never use
    # this host-PID check because a PID from a Docker namespace is ambiguous.
    pid = data.get("pid")
    has_partial_heartbeat_schema = (
        "owner_token" in data
        or "heartbeat_file" in data
        or "heartbeat_mode" in data
    )
    if has_partial_heartbeat_schema:
        stale = owner_age > invalid_grace_after
        reason = "invalid_owner_record" if stale else None
    elif pid == 1:
        # PID 1 is the usual worker PID in a Docker namespace.  Probing host
        # PID 1 would keep an abandoned legacy container lease forever.
        stale = owner_age > invalid_grace_after
        reason = "legacy_container_owner_expired" if stale else None
    elif isinstance(pid, int):
        stale = not (alive or _default_alive)(pid)
        reason = "legacy_owner_pid_dead" if stale else None
    else:
        stale = owner_age > invalid_grace_after
        reason = "invalid_owner_record" if stale else None
    return LeaseSnapshot(
        path=owner_path,
        data=data,
        cas_token=cas_token,
        owner_token=owner_token if isinstance(owner_token, str) else None,
        heartbeat_path=None,
        stale=stale,
        stale_reason=reason,
    )


def scan_stale_lease_snapshots(
    leases_dir: str | Path,
    *,
    alive: Callable[[int], bool] | None = None,
    now: float | None = None,
    stale_after: float = LEASE_STALE_AFTER_SECONDS,
    invalid_grace_after: float = INVALID_LEASE_GRACE_SECONDS,
) -> list[LeaseSnapshot]:
    """Return stale owner snapshots, including old malformed records."""

    root = Path(leases_dir).expanduser()
    if not root.exists():
        return []
    snapshots: list[LeaseSnapshot] = []
    for path in sorted(root.glob("*.json")):
        snapshot = lease_snapshot(
            path,
            alive=alive,
            now=now,
            stale_after=stale_after,
            invalid_grace_after=invalid_grace_after,
        )
        if snapshot is not None and snapshot.stale:
            snapshots.append(snapshot)
    return snapshots


def list_stale_leases(
    leases_dir: str | Path,
    *,
    alive: Callable[[int], bool] | None = None,
    now: float | None = None,
    stale_after: float = LEASE_STALE_AFTER_SECONDS,
    invalid_grace_after: float = INVALID_LEASE_GRACE_SECONDS,
) -> list[Path]:
    """Return paths classified stale by heartbeat (or legacy PID fallback)."""

    return [
        snapshot.path
        for snapshot in scan_stale_lease_snapshots(
            leases_dir,
            alive=alive,
            now=now,
            stale_after=stale_after,
            invalid_grace_after=invalid_grace_after,
        )
    ]


class DeviceLease:
    """An exclusive, heartbeat-backed lease on one ADB server + serial."""

    def __init__(
        self,
        leases_dir: str | Path,
        server: str,
        serial: str,
        owner: str,
        *,
        heartbeat_interval: float = LEASE_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        if (
            not math.isfinite(float(heartbeat_interval))
            or heartbeat_interval <= 0
            or heartbeat_interval > MAX_LEASE_HEARTBEAT_INTERVAL_SECONDS
        ):
            raise ValueError(
                "heartbeat_interval must be finite, positive, and at most "
                f"{MAX_LEASE_HEARTBEAT_INTERVAL_SECONDS:g} seconds"
            )
        self._leases_dir = Path(leases_dir).expanduser()
        self._server = canonicalize_adb_server(server)
        self._serial = serial
        self._owner = owner
        self._heartbeat_interval = float(heartbeat_interval)
        digest = hashlib.sha256(
            f"{self._server}\x00{serial}".encode("utf-8")
        ).hexdigest()
        self._path = self._leases_dir / f"{digest}.json"
        self._owner_token: str | None = None
        self._heartbeat_path: Path | None = None
        self._heartbeat_process: subprocess.Popen[bytes] | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def owner(self) -> str:
        return self._owner

    def _stop_heartbeat(self) -> None:
        process = self._heartbeat_process
        self._heartbeat_process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)

    def acquire(self) -> None:
        """Atomically publish the owner and start its heartbeat sidecar."""

        self._leases_dir.mkdir(parents=True, exist_ok=True)
        owner_token = secrets.token_hex(16)
        heartbeat_path = self._path.with_name(
            f".{self._path.stem}.{owner_token}.heartbeat"
        )
        payload = {
            "owner": self._owner,
            "owner_token": owner_token,
            "server": self._server,
            "serial": self._serial,
            "acquired_at": utc_now().isoformat(),
            "pid": os.getpid(),
            "pid_scope": "diagnostic-only",
            "heartbeat_mode": "process-sidecar-v1",
            "heartbeat_file": heartbeat_path.name,
            "heartbeat_interval_seconds": self._heartbeat_interval,
            "attempt_dirs": [],
        }
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        process: subprocess.Popen[bytes] | None = None
        with lease_file_lock(self._path):
            try:
                _publish_noclobber(self._path, data)
            except FileExistsError:
                raise LeaseConflictError(
                    f"device '{self._serial}@{self._server}' is already leased",
                    stage="device",
                    retryable=True,
                    details={
                        "current_owner": self.read_owner(),
                        "path": str(self._path),
                    },
                ) from None
            try:
                _atomic_write(heartbeat_path, b"pending\n")
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "qairt_agent.device.heartbeat",
                        str(heartbeat_path),
                        str(os.getpid()),
                        owner_token,
                        str(self._heartbeat_interval),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                )
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("device lease heartbeat sidecar exited")
                    try:
                        if heartbeat_path.read_text(encoding="utf-8").startswith(
                            owner_token + ":"
                        ):
                            break
                    except OSError:
                        pass
                    time.sleep(0.01)
                else:
                    raise RuntimeError(
                        "device lease heartbeat sidecar did not become ready"
                    )
            except BaseException:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1.0)
                changed = False
                with contextlib.suppress(FileNotFoundError):
                    heartbeat_path.unlink()
                    changed = True
                with contextlib.suppress(FileNotFoundError):
                    self._path.unlink()
                    changed = True
                if changed:
                    _fsync_directory(self._leases_dir)
                raise
        self._owner_token = owner_token
        self._heartbeat_path = heartbeat_path
        self._heartbeat_process = process

    def release(self) -> None:
        """Stop heartbeating and remove an empty lease owned by this instance.

        If exact remote attempt directories remain, the owner record and last
        heartbeat are retained for GC recovery and become stale after the
        normal heartbeat timeout.
        """

        self._stop_heartbeat()
        with lease_file_lock(self._path):
            current = self.read_owner()
            owns_current = (
                current is not None
                and self._owner_token is not None
                and current.get("owner_token") == self._owner_token
            )
            if owns_current and not current.get("attempt_dirs"):
                changed = False
                if self._heartbeat_path is not None:
                    with contextlib.suppress(FileNotFoundError):
                        self._heartbeat_path.unlink()
                        changed = True
                # The owner is the discoverable lock.  Remove it last so a
                # hard interruption cannot leave an unreferenced heartbeat.
                with contextlib.suppress(FileNotFoundError):
                    self._path.unlink()
                    changed = True
                if changed:
                    _fsync_directory(self._leases_dir)
            elif not owns_current and self._heartbeat_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    self._heartbeat_path.unlink()
                    _fsync_directory(self._leases_dir)

    def _replace_owner(self, payload: dict[str, Any]) -> None:
        """Atomically CAS-replace this instance's owner record."""

        with lease_file_lock(self._path):
            current = self.read_owner()
            if (
                current is None
                or self._owner_token is None
                or current.get("owner_token") != self._owner_token
            ):
                raise LeaseConflictError(
                    "cannot update a device lease owned by another process",
                    stage="device",
                    retryable=True,
                    details={"current_owner": current, "path": str(self._path)},
                )
            self._leases_dir.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self._path.stem}.",
                suffix=".tmp",
                dir=self._leases_dir,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self._path)
                _fsync_directory(self._leases_dir)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    temporary.unlink()

    @staticmethod
    def _validate_attempt_dir(attempt_dir: str) -> str:
        parse_remote_attempt_dir(attempt_dir)
        return attempt_dir

    def record_attempt_dir(self, attempt_dir: str) -> None:
        """Persist one exact remote attempt before any files are pushed."""

        normalized = self._validate_attempt_dir(attempt_dir)
        payload = self.read_owner()
        if (
            payload is None
            or self._owner_token is None
            or payload.get("owner_token") != self._owner_token
        ):
            raise LeaseConflictError(
                "cannot record an attempt on an unowned device lease",
                stage="device",
                retryable=True,
                details={"current_owner": payload, "path": str(self._path)},
            )
        attempts = [
            str(item)
            for item in payload.get("attempt_dirs", [])
            if isinstance(item, str)
        ]
        if normalized not in attempts:
            attempts.append(normalized)
        payload["attempt_dirs"] = sorted(attempts)
        self._replace_owner(payload)

    def forget_attempt_dir(self, attempt_dir: str) -> None:
        """Remove one attempt only after exact remote cleanup succeeded."""

        normalized = self._validate_attempt_dir(attempt_dir)
        payload = self.read_owner()
        if (
            payload is None
            or self._owner_token is None
            or payload.get("owner_token") != self._owner_token
        ):
            raise LeaseConflictError(
                "cannot clear an attempt on an unowned device lease",
                stage="device",
                retryable=True,
                details={"current_owner": payload, "path": str(self._path)},
            )
        payload["attempt_dirs"] = [
            str(item)
            for item in payload.get("attempt_dirs", [])
            if isinstance(item, str) and str(item) != normalized
        ]
        self._replace_owner(payload)

    def is_held(self) -> bool:
        return self._path.exists()

    def read_owner(self) -> dict[str, Any] | None:
        try:
            parsed = json.loads(self._path.read_text(encoding="utf-8"))
            return parsed if isinstance(parsed, dict) else None
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def __enter__(self) -> "DeviceLease":
        self.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()
