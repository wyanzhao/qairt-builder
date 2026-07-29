"""Device health checks and stale-lease garbage collection.

``device_doctor`` runs a sequence of independent checks and captures each
failure into a structured report instead of raising, so a caller sees every
problem at once.  ``device_gc`` reclaims leases whose process-sidecar heartbeat
is stale and removes *only* exact recorded attempt directories after a
per-device lock/CAS recheck.
"""

from __future__ import annotations

import contextlib
import errno
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qairt_agent.device.adb import (
    AdbClient,
    AdbConfig,
    canonicalize_adb_server,
)
from qairt_agent.device.lease import (
    INVALID_LEASE_GRACE_SECONDS,
    LEASE_STALE_AFTER_SECONDS,
    _default_alive,
    lease_file_lock,
    lease_snapshot,
    scan_stale_lease_snapshots,
)
from qairt_agent.errors import DeviceUnavailableError
from qairt_agent.harness import (
    DEFAULT_CONSTRAINTS,
    HarnessConstraints,
    load_harness_constraints,
)

__all__ = ["device_doctor", "require_healthy", "device_gc"]

#: The QAIRT build id this agent is pinned to (see contracts.TargetSpec).
EXPECTED_SDK_BUILD = DEFAULT_CONSTRAINTS.qairt_build_id
DEFAULT_TARGET = (
    f"{DEFAULT_CONSTRAINTS.target_chipset}/"
    f"{DEFAULT_CONSTRAINTS.target_dsp_arch}/"
    f"{DEFAULT_CONSTRAINTS.target_soc_model}"
)


def _check(ok: bool, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": ok, "message": message, **extra}


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


def _parse_size_kb(token: str) -> int | None:
    """Parse a ``df`` size token like ``90G``/``1024M``/``4096`` into KB."""

    token = token.strip()
    if not token:
        return None
    multipliers = {"K": 1, "M": 1024, "G": 1024 * 1024, "T": 1024 * 1024 * 1024}
    suffix = token[-1].upper()
    if suffix in multipliers:
        number, factor = token[:-1], multipliers[suffix]
    else:
        number, factor = token, 1
    try:
        return int(float(number) * factor)
    except ValueError:
        return None


def _free_space_kb(client: AdbClient) -> int | None:
    """Best-effort parse of ``df /data/local/tmp`` available space (KB).

    Handles both the Android (``Filesystem Size Used Free Blksize``) and toybox
    (``Filesystem 1K-blocks Used Available Use% Mounted``) layouts, where the
    available/free value is the 4th whitespace-separated field of the data line.
    """

    result = client.shell("df /data/local/tmp")
    for raw_line in (getattr(result, "stdout", "") or "").splitlines():
        if "/data/local/tmp" not in raw_line:
            continue
        fields = raw_line.split()
        if len(fields) >= 4:
            parsed = _parse_size_kb(fields[3])
            if parsed is not None:
                return parsed
    return None


def device_doctor(
    config: AdbConfig,
    client: AdbClient,
    *,
    sdk_build: str | None = None,
    target: str | None = None,
    min_free_kb: int = 100 * 1024,
    constraints: HarnessConstraints | None = None,
) -> dict[str, Any]:
    """Run device health checks and return a structured report.

    Each check is captured (never raised) so the report lists every problem.
    The overall ``ok`` is False if any check fails — in particular if the device
    is entirely unreachable.
    """

    active = constraints or load_harness_constraints()
    expected_sdk_build = active.qairt_build_id
    resolved_sdk_build = (
        expected_sdk_build if sdk_build is None else sdk_build
    )
    resolved_target = (
        (
            f"{active.target_chipset}/"
            f"{active.target_dsp_arch}/"
            f"{active.target_soc_model}"
        )
        if target is None
        else target
    )
    checks: dict[str, dict[str, Any]] = {}

    # 1. server reachable (devices() works) + 2. target device present.
    serials: list[str] | None = None
    try:
        serials = client.devices()
        checks["server_reachable"] = _check(True, f"adb server reachable; {len(serials)} device(s)")
    except Exception as exc:  # noqa: BLE001 - doctor captures, never raises
        checks["server_reachable"] = _check(False, f"adb server unreachable: {exc}")

    if serials is None:
        checks["device_present"] = _check(False, "skipped: adb server unreachable")
    elif config.serial in serials:
        checks["device_present"] = _check(True, f"serial {config.serial} present")
    else:
        checks["device_present"] = _check(
            False,
            f"serial {config.serial} not attached",
            known_serials=serials,
        )

    # 3. target device in the "device" state.
    try:
        state = client.device_state()
        ok = state == "device"
        checks["device_state"] = _check(ok, f"device state is '{state}'", state=state)
    except Exception as exc:  # noqa: BLE001
        checks["device_state"] = _check(False, f"could not read device state: {exc}")

    # 4. target triple resolvable (recorded; acceptance is policy elsewhere).
    checks["target_resolved"] = _check(
        True,
        f"target triple recorded: {resolved_target}",
        target=resolved_target,
    )

    # 5. remote free space.
    try:
        free_kb = _free_space_kb(client)
        if free_kb is None:
            checks["remote_free_space"] = _check(False, "could not parse remote free space")
        else:
            ok = free_kb >= min_free_kb
            checks["remote_free_space"] = _check(
                ok,
                f"remote free space {free_kb} KB (need >= {min_free_kb} KB)",
                available_kb=free_kb,
            )
    except Exception as exc:  # noqa: BLE001
        checks["remote_free_space"] = _check(False, f"could not read remote free space: {exc}")

    # 6. SDK compatibility.
    sdk_ok = resolved_sdk_build == expected_sdk_build
    checks["sdk_compatible"] = _check(
        sdk_ok,
        f"sdk build {resolved_sdk_build} "
        f"{'matches' if sdk_ok else 'does not match'} expected "
        f"{expected_sdk_build}",
        sdk_build=resolved_sdk_build,
        expected_sdk_build=expected_sdk_build,
    )

    return {
        "ok": all(entry["ok"] for entry in checks.values()),
        "device_identifier": config.device_identifier,
        "target": resolved_target,
        "checks": checks,
    }


def require_healthy(report: dict[str, Any]) -> None:
    """Raise :class:`DeviceUnavailableError` if the report is not healthy."""

    if report.get("ok"):
        return
    checks = report.get("checks", {})
    failed = {name: entry.get("message", "") for name, entry in checks.items() if not entry.get("ok")}
    summary = ", ".join(sorted(failed)) or "unknown device failure"
    raise DeviceUnavailableError(
        f"device is not healthy: {summary}",
        stage="device",
        retryable=True,
        details={"failed_checks": failed, "device_identifier": report.get("device_identifier")},
    )


def device_gc(
    leases_dir: str | Path,
    client: AdbClient | None = None,
    *,
    alive: Callable[[int], bool] | None = None,
    dry_run: bool = False,
    stale_after: float = LEASE_STALE_AFTER_SECONDS,
    invalid_grace_after: float = INVALID_LEASE_GRACE_SECONDS,
) -> dict[str, Any]:
    """Reclaim stale leases and their recorded remote attempt directories.

    For each stale candidate, re-read the owner token and content CAS while
    holding the per-device lock, then remove ONLY exact attempt directories
    listed in ``owner.json``.  Invalid old records are reclaimed locally after
    a creation grace and supply no trusted remote cleanup pointer.  With
    ``dry_run=True`` nothing is deleted.
    """

    if not dry_run and client is not None:
        actual_server = canonicalize_adb_server(client.config.server)
        declared_server_value = (
            os.environ.get("QAIRT_AGENT_ADB_CANONICAL_SERVER") or ""
        ).strip()
        if declared_server_value:
            try:
                declared_server = canonicalize_adb_server(declared_server_value)
            except ValueError as exc:
                raise DeviceUnavailableError(
                    "canonical ADB server identity is invalid",
                    stage="device",
                    retryable=False,
                    details={"canonical_server": declared_server_value},
                ) from exc
            if declared_server != actual_server:
                raise DeviceUnavailableError(
                    "canonical ADB server identity does not match the configured "
                    "ADB client",
                    stage="device",
                    retryable=False,
                    details={
                        "canonical_server": declared_server,
                        "actual_server": actual_server,
                    },
                )

    stale = scan_stale_lease_snapshots(
        leases_dir,
        alive=alive,
        stale_after=stale_after,
        invalid_grace_after=invalid_grace_after,
    )
    if not dry_run and client is None:
        raise DeviceUnavailableError(
            "device_gc requires an explicitly configured AdbClient for "
            "non-dry-run cleanup",
            stage="device",
            retryable=False,
            details={"leases_dir": str(Path(leases_dir).expanduser().resolve())},
        )
    cleaned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    owner_alive = alive or _default_alive

    for candidate in stale:
        lease_path = candidate.path
        with lease_file_lock(lease_path):
            current = lease_snapshot(
                lease_path,
                alive=alive,
                stale_after=stale_after,
                invalid_grace_after=invalid_grace_after,
            )
            if current is None:
                skipped.append(
                    {
                        "lease": str(lease_path),
                        "reason": "lease_disappeared_after_scan",
                    }
                )
                continue
            if (
                current.cas_token != candidate.cas_token
                or current.owner_token != candidate.owner_token
            ):
                skipped.append(
                    {
                        "lease": str(lease_path),
                        "owner": (current.data or {}).get("owner"),
                        "reason": "lease_changed_after_scan",
                    }
                )
                continue
            if not current.stale:
                skipped.append(
                    {
                        "lease": str(lease_path),
                        "owner": (current.data or {}).get("owner"),
                        "reason": "lease_no_longer_stale",
                    }
                )
                continue

            data = current.data or {}
            owner_pid = data.get("pid")
            if (
                not dry_run
                and isinstance(owner_pid, int)
                and owner_pid > 0
                and owner_alive(owner_pid)
            ):
                skipped.append(
                    {
                        "lease": str(lease_path),
                        "owner": data.get("owner"),
                        "pid": owner_pid,
                        "reason": "owner_process_alive",
                    }
                )
                continue
            expected_server_value = (
                client.config.server if client is not None else ""
            )
            expected_server = (
                canonicalize_adb_server(expected_server_value)
                if expected_server_value
                else ""
            )
            recorded_server = data.get("server")
            if isinstance(recorded_server, str):
                try:
                    recorded_server = canonicalize_adb_server(recorded_server)
                except ValueError:
                    recorded_server = None
            if (
                not dry_run
                and client is not None
                and data
                and (
                    recorded_server != expected_server
                    or data.get("serial") != client.config.serial
                )
            ):
                skipped.append(
                    {
                        "lease": str(lease_path),
                        "owner": data.get("owner"),
                        "server": data.get("server"),
                        "serial": data.get("serial"),
                        "reason": "configured_device_mismatch",
                    }
                )
                continue
            attempt_dirs = [
                entry
                for entry in data.get("attempt_dirs", [])
                if isinstance(entry, str)
            ]
            removed: list[str] = []
            for attempt_dir in attempt_dirs:
                # remove_exact shares the strict three-component parser used by
                # DeviceLease bookkeeping; malformed entries fail closed.
                if not dry_run:
                    assert client is not None
                    client.remove_exact(attempt_dir)
                removed.append(attempt_dir)
            if not dry_run:
                changed = False
                if current.heartbeat_path is not None:
                    with contextlib.suppress(FileNotFoundError):
                        current.heartbeat_path.unlink()
                        changed = True
                with contextlib.suppress(FileNotFoundError):
                    lease_path.unlink()
                    changed = True
                if changed:
                    _fsync_directory(lease_path.parent)
            cleaned.append(
                {
                    "lease": str(lease_path),
                    "owner": data.get("owner"),
                    "attempt_dirs": removed,
                    "released": not dry_run,
                    "stale_reason": current.stale_reason,
                }
            )

    return {
        "ok": True,
        "dry_run": dry_run,
        "stale_leases": len(stale),
        "cleaned": cleaned,
        "skipped": skipped,
    }
