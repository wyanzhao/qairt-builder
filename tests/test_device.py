from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from qairt_agent.device import (
    AdbClient,
    AdbConfig,
    DeviceLease,
    DeviceRuntime,
    canonicalize_adb_server,
    device_doctor,
    device_gc,
    list_stale_leases,
    parse_remote_attempt_dir,
    remote_attempt_dir,
    require_healthy,
)
from qairt_agent.errors import (
    ArtifactIntegrityError,
    DeviceUnavailableError,
    LeaseConflictError,
)
from qairt_agent.harness import DEFAULT_CONSTRAINTS

CONFIG = AdbConfig(serial="ABC123", server="localhost:5037")
BASE = ["adb", "-H", "localhost", "-P", "5037", "-s", "ABC123"]
ATTEMPT_DIR = "/data/local/tmp/qairt-agent/job1/build/3/"


class FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeExecutor:
    """Records argv and returns canned results; never touches a real device."""

    def __init__(self, handler=None) -> None:
        self.calls: list[list[str]] = []
        self.handler = handler

    def __call__(self, argv: list[str]) -> FakeCompleted:
        self.calls.append(list(argv))
        if self.handler is not None:
            return self.handler(list(argv))
        return FakeCompleted()


# --------------------------------------------------------------------------- #
# AdbConfig
# --------------------------------------------------------------------------- #


def test_from_env_fails_closed_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("QAIRT_AGENT_ADB_SERIAL", raising=False)
    monkeypatch.delenv("QAIRT_AGENT_ADB_SERVER", raising=False)
    with pytest.raises(DeviceUnavailableError, match="auto-selected"):
        AdbConfig.from_env()

    monkeypatch.setenv("QAIRT_AGENT_ADB_SERVER", "localhost:5037")
    with pytest.raises(DeviceUnavailableError):
        AdbConfig.from_env()  # serial still missing

    monkeypatch.delenv("QAIRT_AGENT_ADB_SERVER", raising=False)
    monkeypatch.setenv("QAIRT_AGENT_ADB_SERIAL", "ABC123")
    with pytest.raises(DeviceUnavailableError):
        AdbConfig.from_env()  # server still missing


def test_from_env_success(monkeypatch) -> None:
    monkeypatch.setenv("QAIRT_AGENT_ADB_SERIAL", "ABC123")
    monkeypatch.setenv("QAIRT_AGENT_ADB_SERVER", "localhost:5037")
    config = AdbConfig.from_env()
    assert config.serial == "ABC123"
    assert config.server == "localhost:5037"
    assert config.device_identifier == "ABC123@localhost:5037"


def test_from_env_explicit_environ() -> None:
    config = AdbConfig.from_env(
        {"QAIRT_AGENT_ADB_SERIAL": "S1", "QAIRT_AGENT_ADB_SERVER": "host:9999"}
    )
    assert config.device_identifier == "S1@host:9999"


def test_adb_base_argv_contains_server_and_serial() -> None:
    assert CONFIG.adb_base_argv() == BASE
    assert CONFIG.adb_server_arg == "-H localhost -P 5037"
    assert CONFIG.adb_server_argv() == ["adb", "-H", "localhost", "-P", "5037"]


def test_config_rejects_blank_or_malformed() -> None:
    with pytest.raises(ValueError):
        AdbConfig(serial="  ", server="localhost:5037")
    with pytest.raises(ValueError):
        AdbConfig(serial="ABC123", server="no-port-here")


@pytest.mark.parametrize(
    "server",
    [
        "localhost:5037",
        "LOCALHOST.:5037",
        "127.0.0.1:5037",
        "::1:5037",
        "[::1]:5037",
        "host.docker.internal:5037",
        "host.container.internal:5037",
    ],
)
def test_adb_server_canonicalization_collapses_host_loopback_aliases(server) -> None:
    assert canonicalize_adb_server(server) == "localhost:5037"


def test_adb_server_canonicalization_does_not_change_connection_config() -> None:
    config = AdbConfig(serial="ABC123", server="127.0.0.1:5037")
    assert canonicalize_adb_server(config.server) == "localhost:5037"
    assert config.adb_server_argv() == ["adb", "-H", "127.0.0.1", "-P", "5037"]


# --------------------------------------------------------------------------- #
# AdbClient argv + failure handling
# --------------------------------------------------------------------------- #


def test_push_pull_shell_build_correct_argv() -> None:
    executor = FakeExecutor()
    client = AdbClient(CONFIG, command_executor=executor)

    client.push("/local/a.bin", "/remote/a.bin")
    assert executor.calls[-1] == [*BASE, "push", "/local/a.bin", "/remote/a.bin"]

    client.pull("/remote/a.bin", "/local/a.bin")
    assert executor.calls[-1] == [*BASE, "pull", "/remote/a.bin", "/local/a.bin"]

    client.shell("ls -la /data")
    assert executor.calls[-1] == [*BASE, "shell", "ls -la /data"]


def test_devices_runs_without_serial_and_parses() -> None:
    stdout = "List of devices attached\nABC123\tdevice\nXYZ\toffline\n"
    executor = FakeExecutor(handler=lambda argv: FakeCompleted(stdout=stdout))
    client = AdbClient(CONFIG, command_executor=executor)
    assert client.devices() == ["ABC123", "XYZ"]
    assert executor.calls[-1] == ["adb", "-H", "localhost", "-P", "5037", "devices"]


def test_remote_sha256_parses_digest() -> None:
    digest = "a" * 64
    executor = FakeExecutor(handler=lambda argv: FakeCompleted(stdout=f"{digest}  /remote/x\n"))
    client = AdbClient(CONFIG, command_executor=executor)
    assert client.remote_sha256("/remote/x") == digest
    assert executor.calls[-1] == [*BASE, "shell", "sha256sum /remote/x"]


def test_remote_helpers_quote_exported_api_paths() -> None:
    digest = "b" * 64
    executor = FakeExecutor(
        handler=lambda argv: FakeCompleted(
            stdout=f"{digest}  /remote/unsafe path;reboot\n"
        )
    )
    client = AdbClient(CONFIG, command_executor=executor)
    unsafe = "/remote/unsafe path;reboot"

    assert client.remote_sha256(unsafe) == digest
    assert executor.calls[-1] == [
        *BASE,
        "shell",
        "sha256sum '/remote/unsafe path;reboot'",
    ]
    assert client.remote_exists(unsafe) is True
    assert executor.calls[-1] == [
        *BASE,
        "shell",
        "test -e '/remote/unsafe path;reboot'",
    ]


def test_nonzero_return_raises_device_unavailable() -> None:
    executor = FakeExecutor(handler=lambda argv: FakeCompleted(returncode=1, stderr="boom"))
    client = AdbClient(CONFIG, command_executor=executor)
    with pytest.raises(DeviceUnavailableError) as excinfo:
        client.push("/l", "/r")
    assert excinfo.value.details["returncode"] == 1
    assert excinfo.value.details["stderr"] == "boom"

    with pytest.raises(DeviceUnavailableError):
        client.shell("reboot")


def test_remote_exists_uses_returncode_without_raising() -> None:
    executor = FakeExecutor(handler=lambda argv: FakeCompleted(returncode=1))
    client = AdbClient(CONFIG, command_executor=executor)
    assert client.remote_exists("/remote/missing") is False
    assert executor.calls[-1] == [*BASE, "shell", "test -e /remote/missing"]


# --------------------------------------------------------------------------- #
# remote_attempt_dir + remove_exact guards
# --------------------------------------------------------------------------- #


def test_remote_attempt_dir_exact_layout() -> None:
    assert remote_attempt_dir("job1", "build", "3") == ATTEMPT_DIR


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "  ",
        "../x",
        "a/b",
        "..",
        "x;rm",
        "x $(id)",
        "x`id`",
        "x&whoami",
        "x|whoami",
        ".",
        " job",
        "job ",
    ],
)
def test_remote_attempt_dir_rejects_unsafe_components(bad) -> None:
    with pytest.raises(ValueError):
        remote_attempt_dir(bad, "build", "3")
    with pytest.raises(ValueError):
        remote_attempt_dir("job1", bad, "3")
    with pytest.raises(ValueError):
        remote_attempt_dir("job1", "build", bad)


def test_remove_exact_refuses_paths_outside_root() -> None:
    executor = FakeExecutor()
    client = AdbClient(CONFIG, command_executor=executor)
    for bad in (
        "/sdcard",
        "/data/local/tmp/other",
        "/data/local/tmp/qairt-agent/../x",
        "/data/local/tmp/qairt-agent/job/stage/",
        "/data/local/tmp/qairt-agent/job/stage/attempt",
        "/data/local/tmp/qairt-agent/job/stage/attempt/extra/",
        "/data/local/tmp/qairt-agent/job/stage/attempt//",
    ):
        with pytest.raises(ValueError):
            client.remove_exact(bad)
    assert executor.calls == []  # nothing was ever executed


def test_attempt_path_parser_accepts_only_exact_generated_layout() -> None:
    path = remote_attempt_dir("job-1", "benchmark_2", "attempt.003")
    assert parse_remote_attempt_dir(path) == (
        "job-1",
        "benchmark_2",
        "attempt.003",
    )


def test_remove_exact_issues_rm_rf_for_valid_dir() -> None:
    executor = FakeExecutor()
    client = AdbClient(CONFIG, command_executor=executor)
    client.remove_exact(ATTEMPT_DIR)
    assert executor.calls[-1] == [*BASE, "shell", "rm", "-rf", ATTEMPT_DIR]


def test_stage_attempt_rejects_noncanonical_filename_before_adb() -> None:
    executor = FakeExecutor()
    client = AdbClient(CONFIG, command_executor=executor)
    with pytest.raises(ValueError, match="whitespace"):
        with client.stage_attempt(
            "job1",
            "build",
            "3",
            push_files={" model.bin ": "/unused"},
        ):
            pass
    assert executor.calls == []


# --------------------------------------------------------------------------- #
# staging session
# --------------------------------------------------------------------------- #


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_stage_attempt_pushes_verifies_marks_ready_and_cleans_up(tmp_path) -> None:
    local = tmp_path / "model.bin"
    local.write_bytes(b"weights")
    digest = _sha(b"weights")

    def handler(argv):
        if any("sha256sum" in str(a) for a in argv):
            return FakeCompleted(stdout=f"{digest}  /remote\n")
        return FakeCompleted()

    executor = FakeExecutor(handler=handler)
    client = AdbClient(CONFIG, command_executor=executor)

    with client.stage_attempt("job1", "build", "3", push_files={"model.bin": str(local)}) as session:
        assert session.attempt_dir == ATTEMPT_DIR
        incoming = ATTEMPT_DIR + "incoming"
        # pushed to incoming/, verified, then incoming moved to ready
        assert [*BASE, "push", str(local), f"{incoming}/model.bin"] in executor.calls
        assert [*BASE, "shell", f"mkdir -p {incoming}"] in executor.calls
        assert [*BASE, "shell", f"mv {incoming} {ATTEMPT_DIR}ready"] in executor.calls

    # cleanup is the final action: rm -rf the exact attempt dir
    assert executor.calls[-1] == [*BASE, "shell", "rm", "-rf", ATTEMPT_DIR]


def test_stage_attempt_cleans_up_when_body_raises(tmp_path) -> None:
    local = tmp_path / "model.bin"
    local.write_bytes(b"weights")
    digest = _sha(b"weights")

    def handler(argv):
        if any("sha256sum" in str(a) for a in argv):
            return FakeCompleted(stdout=f"{digest}  /remote\n")
        return FakeCompleted()

    executor = FakeExecutor(handler=handler)
    client = AdbClient(CONFIG, command_executor=executor)

    with pytest.raises(RuntimeError, match="boom"):
        with client.stage_attempt("job1", "build", "3", push_files={"model.bin": str(local)}):
            raise RuntimeError("boom")

    assert executor.calls[-1] == [*BASE, "shell", "rm", "-rf", ATTEMPT_DIR]


def test_stage_attempt_sha_mismatch_fails_and_cleans_up(tmp_path) -> None:
    local = tmp_path / "model.bin"
    local.write_bytes(b"weights")

    executor = FakeExecutor(
        handler=lambda argv: FakeCompleted(stdout=f"{'b' * 64}  /remote\n")
        if any("sha256sum" in str(a) for a in argv)
        else FakeCompleted()
    )
    client = AdbClient(CONFIG, command_executor=executor)

    with pytest.raises(ArtifactIntegrityError):
        with client.stage_attempt("job1", "build", "3", push_files={"model.bin": str(local)}):
            pass

    assert executor.calls[-1] == [*BASE, "shell", "rm", "-rf", ATTEMPT_DIR]


# --------------------------------------------------------------------------- #
# DeviceLease
# --------------------------------------------------------------------------- #


def test_lease_acquire_creates_owner_json(tmp_path) -> None:
    lease = DeviceLease(tmp_path, "localhost:5037", "ABC123", owner="job-1")
    assert not lease.is_held()
    lease.acquire()
    try:
        assert lease.is_held()
        owner = lease.read_owner()
        assert owner["owner"] == "job-1"
        assert owner["server"] == "localhost:5037"
        assert owner["serial"] == "ABC123"
        assert owner["pid"] == os.getpid()
        assert owner["attempt_dirs"] == []
        assert "acquired_at" in owner
    finally:
        lease.release()


def test_lease_second_owner_conflicts(tmp_path) -> None:
    first = DeviceLease(tmp_path, "localhost:5037", "ABC123", owner="job-1")
    first.acquire()
    second = DeviceLease(tmp_path, "localhost:5037", "ABC123", owner="job-2")
    try:
        with pytest.raises(LeaseConflictError) as excinfo:
            second.acquire()
        assert excinfo.value.details["current_owner"]["owner"] == "job-1"
    finally:
        first.release()


def test_lease_release_removes_and_is_idempotent(tmp_path) -> None:
    lease = DeviceLease(tmp_path, "h:1", "S", owner="job-1")
    lease.acquire()
    lease.release()
    assert not lease.is_held()
    lease.release()  # already gone -> no error


def test_lease_release_does_not_steal_another_owner(tmp_path) -> None:
    first = DeviceLease(tmp_path, "h:1", "S", owner="job-1")
    first.acquire()
    try:
        DeviceLease(tmp_path, "h:1", "S", owner="job-2").release()
        assert first.is_held()
    finally:
        first.release()


def test_lease_context_manager_releases_on_exit(tmp_path) -> None:
    lease = DeviceLease(tmp_path, "h:1", "S", owner="job-1")
    with lease:
        assert lease.is_held()
    assert not lease.is_held()


def test_lease_attempt_bookkeeping_is_atomic_and_blocks_early_release(tmp_path) -> None:
    lease = DeviceLease(tmp_path, "h:1", "S", owner="job-1")
    lease.acquire()
    lease.record_attempt_dir(ATTEMPT_DIR)
    assert lease.read_owner()["attempt_dirs"] == [ATTEMPT_DIR]
    lease.release()
    assert lease.is_held()  # recovery pointer cannot be discarded
    lease.forget_attempt_dir(ATTEMPT_DIR)
    lease.release()
    assert not lease.is_held()


def test_lease_attempt_bookkeeping_rejects_broad_or_foreign_paths(tmp_path) -> None:
    lease = DeviceLease(tmp_path, "h:1", "S", owner="job-1")
    lease.acquire()
    for bad in (
        "/data/local/tmp/qairt-agent/",
        "/data/local/tmp/qairt-agent/../other",
        "/data/local/tmp/other/job/stage/attempt/",
    ):
        with pytest.raises(ValueError):
            lease.record_attempt_dir(bad)
    lease.release()


def test_lease_path_is_derived_from_server_and_serial(tmp_path) -> None:
    a = DeviceLease(tmp_path, "h:1", "S1", owner="o")
    b = DeviceLease(tmp_path, "h:1", "S2", owner="o")
    assert a.path != b.path
    assert a.path.suffix == ".json"


def test_lease_path_canonicalizes_native_and_container_loopback(tmp_path) -> None:
    native = DeviceLease(tmp_path, "127.0.0.1:5037", "S", owner="native")
    docker = DeviceLease(
        tmp_path,
        "host.docker.internal:5037",
        "S",
        owner="docker",
    )
    ipv6 = DeviceLease(tmp_path, "::1:5037", "S", owner="ipv6")
    apple = DeviceLease(
        tmp_path,
        "host.container.internal:5037",
        "S",
        owner="apple",
    )
    assert native.path == docker.path == apple.path == ipv6.path


@pytest.mark.parametrize("interval", [0.0, float("inf"), 60.1])
def test_lease_rejects_unsafe_heartbeat_interval(tmp_path, interval) -> None:
    with pytest.raises(ValueError, match="at most 60"):
        DeviceLease(
            tmp_path,
            "localhost:5037",
            "S",
            owner="job",
            heartbeat_interval=interval,
        )


def test_lease_heartbeat_is_a_process_sidecar_and_remains_fresh(tmp_path) -> None:
    lease = DeviceLease(
        tmp_path,
        "localhost:5037",
        "S",
        owner="job",
        heartbeat_interval=0.03,
    )
    lease.acquire()
    try:
        owner = lease.read_owner()
        assert owner["heartbeat_mode"] == "process-sidecar-v1"
        assert owner["pid_scope"] == "diagnostic-only"
        heartbeat = tmp_path / owner["heartbeat_file"]
        first = heartbeat.read_text(encoding="utf-8")
        time.sleep(0.10)
        second = heartbeat.read_text(encoding="utf-8")
        assert first != second
        # The heartbeat is authoritative even when the observer cannot resolve
        # the worker PID in its own namespace.
        assert list_stale_leases(
            tmp_path,
            alive=lambda _pid: False,
            stale_after=0.08,
        ) == []
    finally:
        lease.release()


def test_hard_killed_owner_stops_sidecar_and_becomes_stale(tmp_path) -> None:
    child_code = (
        "import sys,time;"
        "from qairt_agent.device import DeviceLease;"
        "lease=DeviceLease(sys.argv[1],'localhost:5037','S','hard-kill',"
        "heartbeat_interval=0.03);"
        "lease.acquire();"
        "lease.record_attempt_dir("
        "'/data/local/tmp/qairt-agent/job/stage/attempt/');"
        "time.sleep(60)"
    )
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (source_root, environment.get("PYTHONPATH"))
        if value
    )
    expected_path = DeviceLease(
        tmp_path,
        "localhost:5037",
        "S",
        owner="probe",
    ).path
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(tmp_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=environment,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not expected_path.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert process.poll() is None
        owner = json.loads(expected_path.read_text(encoding="utf-8"))
        heartbeat = tmp_path / owner["heartbeat_file"]
        deadline = time.monotonic() + 3.0
        while not heartbeat.exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert heartbeat.exists()
        deadline = time.monotonic() + 3.0
        while not owner.get("attempt_dirs"):
            if process.poll() is not None or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
            owner = json.loads(expected_path.read_text(encoding="utf-8"))
        assert owner["attempt_dirs"] == [
            "/data/local/tmp/qairt-agent/job/stage/attempt/"
        ]
        first = heartbeat.stat().st_mtime_ns
        deadline = time.monotonic() + 1.0
        while heartbeat.stat().st_mtime_ns == first:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert heartbeat.stat().st_mtime_ns != first
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2.0)

    # The sidecar observes re-parenting and exits independently of the worker's
    # GIL.  Use an injected observation time to exercise 30-second semantics
    # without making the test sleep for 30 seconds.
    time.sleep(0.10)
    stopped_at = heartbeat.stat().st_mtime_ns
    time.sleep(0.10)
    assert heartbeat.stat().st_mtime_ns == stopped_at
    assert list_stale_leases(
        tmp_path,
        alive=lambda _pid: True,
        now=heartbeat.stat().st_mtime + 31.0,
        stale_after=30.0,
    ) == [expected_path]


def test_lease_owner_is_fully_formed_before_noclobber_publication(
    tmp_path,
    monkeypatch,
) -> None:
    from qairt_agent.device import lease as lease_module

    observed = {}
    original_link = lease_module.os.link

    def inspect_then_link(source, destination):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        observed.update(payload)
        assert not os.path.exists(destination)
        return original_link(source, destination)

    monkeypatch.setattr(lease_module.os, "link", inspect_then_link)
    lease = DeviceLease(tmp_path, "localhost:5037", "S", owner="job")
    lease.acquire()
    try:
        assert observed["owner"] == "job"
        assert observed["owner_token"]
        assert observed["attempt_dirs"] == []
    finally:
        lease.release()


@pytest.mark.parametrize(
    "module_name",
    [
        "qairt_agent.device.lease",
        "qairt_agent.device.doctor",
    ],
)
def test_directory_fsync_tolerates_unsupported_virtual_filesystem(
    tmp_path,
    monkeypatch,
    module_name,
) -> None:
    import importlib

    module = importlib.import_module(module_name)

    def unsupported(_descriptor):
        raise OSError(errno.EINVAL, "directory fsync unsupported")

    monkeypatch.setattr(module.os, "fsync", unsupported)
    module._fsync_directory(tmp_path)


def test_directory_fsync_propagates_real_io_failure(tmp_path, monkeypatch) -> None:
    from qairt_agent.device import lease as lease_module

    def io_failure(_descriptor):
        raise OSError(errno.EIO, "disk error")

    monkeypatch.setattr(lease_module.os, "fsync", io_failure)
    with pytest.raises(OSError) as excinfo:
        lease_module._fsync_directory(tmp_path)
    assert excinfo.value.errno == errno.EIO


def test_release_retains_recovery_pointer_but_stops_heartbeat(tmp_path) -> None:
    lease = DeviceLease(
        tmp_path,
        "localhost:5037",
        "S",
        owner="job",
        heartbeat_interval=0.03,
    )
    lease.acquire()
    lease.record_attempt_dir(ATTEMPT_DIR)
    owner = lease.read_owner()
    heartbeat = tmp_path / owner["heartbeat_file"]
    lease.release()
    modified = heartbeat.stat().st_mtime
    time.sleep(0.10)
    assert heartbeat.stat().st_mtime == modified
    assert list_stale_leases(
        tmp_path,
        alive=lambda _pid: True,
        now=modified + 1.0,
        stale_after=0.5,
    ) == [lease.path]


# --------------------------------------------------------------------------- #
# list_stale_leases
# --------------------------------------------------------------------------- #


def _write_lease(
    tmp_path,
    serial,
    *,
    pid,
    owner="o",
    attempt_dirs=None,
    server="h:1",
) -> DeviceLease:
    """Write a lease file directly so tests control pid/attempt_dirs precisely."""

    lease = DeviceLease(tmp_path, server, serial, owner=owner)
    payload = {
        "owner": owner,
        "server": server,
        "serial": serial,
        "acquired_at": "2026-07-28T00:00:00+00:00",
        "pid": pid,
        "attempt_dirs": attempt_dirs or [],
    }
    lease.path.parent.mkdir(parents=True, exist_ok=True)
    lease.path.write_text(json.dumps(payload), encoding="utf-8")
    return lease


def test_list_stale_leases_uses_injected_alive_predicate(tmp_path) -> None:
    live = _write_lease(tmp_path, "LIVE", pid=1111)
    dead = _write_lease(tmp_path, "DEAD", pid=2222)
    stale = list_stale_leases(tmp_path, alive=lambda pid: pid == 1111)
    assert stale == [dead.path]
    assert live.path not in stale


def test_list_stale_leases_missing_dir_is_empty(tmp_path) -> None:
    assert list_stale_leases(tmp_path / "nope", alive=lambda pid: False) == []


def test_heartbeat_staleness_ignores_colliding_host_pid(tmp_path) -> None:
    lease = _write_lease(tmp_path, "DOCKER", pid=1)
    token = "a" * 32
    heartbeat = tmp_path / f".{lease.path.stem}.{token}.heartbeat"
    heartbeat.write_text("old\n", encoding="utf-8")
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    payload.update(
        {
            "owner_token": token,
            "heartbeat_mode": "process-sidecar-v1",
            "heartbeat_file": heartbeat.name,
        }
    )
    lease.path.write_text(json.dumps(payload), encoding="utf-8")
    observed = time.time()
    os.utime(heartbeat, (observed - 100.0, observed - 100.0))
    assert list_stale_leases(
        tmp_path,
        alive=lambda _pid: True,
        now=observed,
        stale_after=30.0,
    ) == [lease.path]


def test_heartbeat_interval_extends_effective_stale_threshold(tmp_path) -> None:
    lease = _write_lease(tmp_path, "SLOW-HEARTBEAT", pid=1)
    token = "b" * 32
    heartbeat = tmp_path / f".{lease.path.stem}.{token}.heartbeat"
    heartbeat.write_text("old\n", encoding="utf-8")
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    payload.update(
        {
            "owner_token": token,
            "heartbeat_mode": "process-sidecar-v1",
            "heartbeat_file": heartbeat.name,
            "heartbeat_interval_seconds": 60.0,
        }
    )
    lease.path.write_text(json.dumps(payload), encoding="utf-8")
    observed = time.time()
    os.utime(heartbeat, (observed - 40.0, observed - 40.0))
    assert list_stale_leases(
        tmp_path,
        alive=lambda _pid: False,
        now=observed,
        stale_after=30.0,
    ) == []
    assert list_stale_leases(
        tmp_path,
        alive=lambda _pid: True,
        now=observed + 181.0,
        stale_after=30.0,
    ) == [lease.path]


def test_invalid_owner_record_is_protected_by_creation_grace(tmp_path) -> None:
    broken = tmp_path / ("b" * 64 + ".json")
    broken.write_bytes(b"")
    observed = time.time()
    os.utime(broken, (observed, observed))
    assert list_stale_leases(
        tmp_path,
        now=observed + 10.0,
        invalid_grace_after=30.0,
    ) == []
    assert list_stale_leases(
        tmp_path,
        now=observed + 31.0,
        invalid_grace_after=30.0,
    ) == [broken]


def test_legacy_container_pid_one_expires_by_age_not_host_liveness(tmp_path) -> None:
    lease = _write_lease(tmp_path, "DOCKER-OLD", pid=1)
    observed = time.time()
    os.utime(lease.path, (observed - 60.0, observed - 60.0))
    assert list_stale_leases(
        tmp_path,
        alive=lambda _pid: True,
        now=observed,
        invalid_grace_after=30.0,
    ) == [lease.path]


# --------------------------------------------------------------------------- #
# device_gc
# --------------------------------------------------------------------------- #


def test_device_gc_dry_run_reports_without_deleting(tmp_path) -> None:
    lease = _write_lease(
        tmp_path, "DEAD", pid=2222, owner="dead-job", attempt_dirs=[ATTEMPT_DIR]
    )
    report = device_gc(tmp_path, client=None, alive=lambda pid: False, dry_run=True)
    assert report["dry_run"] is True
    assert report["stale_leases"] == 1
    assert report["cleaned"][0]["attempt_dirs"] == [ATTEMPT_DIR]
    assert report["cleaned"][0]["released"] is False
    assert lease.path.exists()  # untouched in dry run


def test_device_gc_removes_only_recorded_attempt_dirs(tmp_path) -> None:
    _write_lease(
        tmp_path,
        CONFIG.serial,
        server=CONFIG.server,
        pid=2222,
        owner="dead-job",
        attempt_dirs=[ATTEMPT_DIR],
    )
    executor = FakeExecutor()
    client = AdbClient(CONFIG, command_executor=executor)

    report = device_gc(tmp_path, client=client, alive=lambda pid: False)

    assert report["stale_leases"] == 1
    # remove_exact issued ONLY for the recorded attempt dir, then lease released
    assert executor.calls == [[*BASE, "shell", "rm", "-rf", ATTEMPT_DIR]]
    assert list(tmp_path.glob("*.json")) == []


def test_device_gc_skips_live_leases(tmp_path) -> None:
    _write_lease(tmp_path, "LIVE", pid=1111, attempt_dirs=["/data/local/tmp/qairt-agent/x/y/1/"])
    executor = FakeExecutor()
    client = AdbClient(CONFIG, command_executor=executor)

    report = device_gc(tmp_path, client=client, alive=lambda pid: True)

    assert report["stale_leases"] == 0
    assert executor.calls == []
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_device_gc_treats_cross_user_permission_error_as_alive(
    tmp_path,
    monkeypatch,
) -> None:
    lease = _write_lease(
        tmp_path,
        CONFIG.serial,
        server=CONFIG.server,
        pid=4242,
        owner="other-user-job",
        attempt_dirs=[ATTEMPT_DIR],
    )
    token = "c" * 32
    heartbeat = tmp_path / f".{lease.path.stem}.{token}.heartbeat"
    heartbeat.write_text("old\n", encoding="utf-8")
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    payload.update(
        {
            "owner_token": token,
            "heartbeat_mode": "process-sidecar-v1",
            "heartbeat_file": heartbeat.name,
        }
    )
    lease.path.write_text(json.dumps(payload), encoding="utf-8")
    old = time.time() - 100.0
    os.utime(heartbeat, (old, old))

    def permission_denied(_pid: int, _signal: int) -> None:
        raise PermissionError("owned by another uid")

    monkeypatch.setattr("qairt_agent.device.lease.os.kill", permission_denied)
    executor = FakeExecutor()
    report = device_gc(
        tmp_path,
        client=AdbClient(CONFIG, command_executor=executor),
        stale_after=30.0,
    )

    assert report["cleaned"] == []
    assert report["skipped"][0]["reason"] == "owner_process_alive"
    assert lease.path.exists()
    assert executor.calls == []


def test_device_gc_skips_stale_lease_for_another_device(tmp_path) -> None:
    foreign = _write_lease(
        tmp_path,
        "OTHER",
        server="other-host:5037",
        pid=2222,
        owner="other-job",
        attempt_dirs=[ATTEMPT_DIR],
    )
    executor = FakeExecutor()
    report = device_gc(
        tmp_path,
        client=AdbClient(CONFIG, command_executor=executor),
        alive=lambda pid: False,
    )

    assert report["stale_leases"] == 1
    assert report["cleaned"] == []
    assert report["skipped"][0]["reason"] == "configured_device_mismatch"
    assert foreign.path.exists()
    assert executor.calls == []


def test_device_gc_non_dry_run_requires_explicit_client(tmp_path) -> None:
    _write_lease(
        tmp_path,
        "DEAD",
        pid=2222,
        owner="dead-job",
        attempt_dirs=[ATTEMPT_DIR],
    )
    with pytest.raises(DeviceUnavailableError, match="explicitly configured"):
        device_gc(tmp_path, client=None, alive=lambda pid: False)
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_device_gc_reclaims_old_malformed_owner_without_remote_delete(tmp_path) -> None:
    broken = tmp_path / ("c" * 64 + ".json")
    broken.write_bytes(b"{")
    old = time.time() - 60.0
    os.utime(broken, (old, old))
    executor = FakeExecutor()
    report = device_gc(
        tmp_path,
        client=AdbClient(CONFIG, command_executor=executor),
        invalid_grace_after=30.0,
    )
    assert report["stale_leases"] == 1
    assert report["cleaned"][0]["stale_reason"] == "invalid_owner_record"
    assert executor.calls == []
    assert not broken.exists()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"owner": "partial", "pid": 123},
        {
            "owner": "partial",
            "server": CONFIG.server,
            "serial": CONFIG.serial,
            "pid": 123,
        },
        {
            "owner": "bad-new-format",
            "server": CONFIG.server,
            "serial": CONFIG.serial,
            "pid": 1,
            "attempt_dirs": [],
            "owner_token": "not-a-token",
            "heartbeat_file": "../escape",
            "heartbeat_mode": "process-sidecar-v1",
        },
    ],
)
def test_device_gc_reclaims_json_schema_invalid_owner_locally(
    tmp_path,
    payload,
) -> None:
    broken = tmp_path / ("d" * 64 + ".json")
    broken.write_text(json.dumps(payload), encoding="utf-8")
    old = time.time() - 60.0
    os.utime(broken, (old, old))
    executor = FakeExecutor()
    report = device_gc(
        tmp_path,
        client=AdbClient(CONFIG, command_executor=executor),
        alive=lambda _pid: False,
        invalid_grace_after=30.0,
    )
    assert report["cleaned"][0]["stale_reason"] == "invalid_owner_record"
    assert report["cleaned"][0]["attempt_dirs"] == []
    assert executor.calls == []
    assert not broken.exists()


def test_device_gc_cas_does_not_delete_replaced_owner(
    tmp_path,
    monkeypatch,
) -> None:
    from qairt_agent.device import doctor as doctor_module

    old = _write_lease(
        tmp_path,
        CONFIG.serial,
        server=CONFIG.server,
        pid=2222,
        owner="old",
        attempt_dirs=[ATTEMPT_DIR],
    )
    original_scan = doctor_module.scan_stale_lease_snapshots

    def scan_then_replace(*args, **kwargs):
        candidates = original_scan(*args, **kwargs)
        payload = {
            "owner": "new",
            "server": CONFIG.server,
            "serial": CONFIG.serial,
            "acquired_at": "2026-07-28T00:00:00+00:00",
            "pid": 1111,
            "attempt_dirs": [],
        }
        old.path.write_text(json.dumps(payload), encoding="utf-8")
        return candidates

    monkeypatch.setattr(
        doctor_module,
        "scan_stale_lease_snapshots",
        scan_then_replace,
    )
    executor = FakeExecutor()
    report = device_gc(
        tmp_path,
        client=AdbClient(CONFIG, command_executor=executor),
        alive=lambda pid: pid == 1111,
    )
    assert report["cleaned"] == []
    assert report["skipped"][0]["reason"] == "lease_changed_after_scan"
    assert json.loads(old.path.read_text(encoding="utf-8"))["owner"] == "new"
    assert executor.calls == []


def test_device_gc_matches_loopback_alias_without_changing_adb_target(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("QAIRT_AGENT_ADB_CANONICAL_SERVER", raising=False)
    _write_lease(
        tmp_path,
        CONFIG.serial,
        server="localhost:5037",
        pid=2222,
        attempt_dirs=[ATTEMPT_DIR],
    )
    alias_config = AdbConfig(serial=CONFIG.serial, server="127.0.0.1:5037")
    executor = FakeExecutor()
    report = device_gc(
        tmp_path,
        client=AdbClient(alias_config, command_executor=executor),
        alive=lambda _pid: False,
    )
    assert report["cleaned"][0]["attempt_dirs"] == [ATTEMPT_DIR]
    assert executor.calls[0][:5] == [
        "adb",
        "-H",
        "127.0.0.1",
        "-P",
        "5037",
    ]


def test_device_gc_rejects_canonical_server_env_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    _write_lease(
        tmp_path,
        CONFIG.serial,
        server=CONFIG.server,
        pid=2222,
        attempt_dirs=[ATTEMPT_DIR],
    )
    monkeypatch.setenv(
        "QAIRT_AGENT_ADB_CANONICAL_SERVER",
        "different-host:5037",
    )
    executor = FakeExecutor()
    with pytest.raises(DeviceUnavailableError, match="does not match"):
        device_gc(
            tmp_path,
            client=AdbClient(CONFIG, command_executor=executor),
            alive=lambda _pid: False,
        )
    assert executor.calls == []


def test_device_gc_wraps_invalid_canonical_server_env(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("QAIRT_AGENT_ADB_CANONICAL_SERVER", "invalid")
    with pytest.raises(DeviceUnavailableError, match="identity is invalid"):
        device_gc(
            tmp_path,
            client=AdbClient(CONFIG, command_executor=FakeExecutor()),
        )


def test_device_runtime_rejects_canonical_server_env_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "QAIRT_AGENT_ADB_CANONICAL_SERVER",
        "different-host:5037",
    )
    runtime = DeviceRuntime(
        config_factory=lambda: CONFIG,
        leases_dir=tmp_path / "leases",
    )
    with pytest.raises(DeviceUnavailableError, match="does not match"):
        with runtime.stage(
            FakeDeviceAdapter(),
            output_root=tmp_path / "artifacts",
            job_id="job1",
            stage_key="stage-key",
            attempt_id="attempt-001",
            push_files={},
        ):
            pass
    assert not (tmp_path / "leases").exists()


# --------------------------------------------------------------------------- #
# DeviceRuntime
# --------------------------------------------------------------------------- #


class FakeDeviceAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.device = object()

    def create_device(self, *, serial: str, server: str):
        self.calls.append((serial, server))
        return self.device


def test_device_runtime_stages_records_device_and_cleans(tmp_path) -> None:
    local = tmp_path / "context.bin"
    local.write_bytes(b"context")
    digest = _sha(b"context")

    def handler(argv):
        if any("sha256sum" in str(a) for a in argv):
            return FakeCompleted(stdout=f"{digest}  /remote/context.bin\n")
        return FakeCompleted()

    executor = FakeExecutor(handler=handler)
    adapter = FakeDeviceAdapter()
    leases = tmp_path / "leases"
    runtime = DeviceRuntime(
        config_factory=lambda: CONFIG,
        adb_client_factory=lambda config: AdbClient(
            config,
            command_executor=executor,
        ),
        leases_dir=leases,
    )

    with runtime.stage(
        adapter,
        output_root=tmp_path / "artifacts",
        job_id="job1",
        stage_key="stage-key",
        attempt_id="attempt-001",
        push_files={"context.bin": local},
    ) as session:
        assert session.device is adapter.device
        assert session.identifier == CONFIG.device_identifier
        assert session.adb.attempt_dir == (
            "/data/local/tmp/qairt-agent/job1/stage-key/attempt-001/"
        )
        owner_path = next(leases.glob("*.json"))
        assert json.loads(owner_path.read_text())["attempt_dirs"] == [
            session.adb.attempt_dir
        ]

    assert adapter.calls == [("ABC123", "localhost:5037")]
    assert executor.calls[-1] == [
        *BASE,
        "shell",
        "rm",
        "-rf",
        "/data/local/tmp/qairt-agent/job1/stage-key/attempt-001/",
    ]
    assert list(leases.glob("*.json")) == []


def test_device_runtime_cleanup_failure_retains_gc_pointer(tmp_path) -> None:
    local = tmp_path / "context.bin"
    local.write_bytes(b"context")
    digest = _sha(b"context")

    def handler(argv):
        if any("sha256sum" in str(a) for a in argv):
            return FakeCompleted(stdout=f"{digest}  /remote/context.bin\n")
        if "rm" in argv:
            return FakeCompleted(returncode=1, stderr="cleanup failed")
        return FakeCompleted()

    leases = tmp_path / "leases"
    runtime = DeviceRuntime(
        config_factory=lambda: CONFIG,
        adb_client_factory=lambda config: AdbClient(
            config,
            command_executor=FakeExecutor(handler=handler),
        ),
        leases_dir=leases,
    )
    with pytest.raises(DeviceUnavailableError, match="adb command failed"):
        with runtime.stage(
            FakeDeviceAdapter(),
            output_root=tmp_path / "artifacts",
            job_id="job1",
            stage_key="stage-key",
            attempt_id="attempt-001",
            push_files={"context.bin": local},
        ):
            pass

    owner_path = next(leases.glob("*.json"))
    assert json.loads(owner_path.read_text())["attempt_dirs"] == [
        "/data/local/tmp/qairt-agent/job1/stage-key/attempt-001/"
    ]


def test_device_cleanup_failure_preserves_stage_exception_as_cause(
    tmp_path,
) -> None:
    def handler(argv):
        if "rm" in argv:
            return FakeCompleted(returncode=1, stderr="cleanup failed")
        return FakeCompleted()

    runtime = DeviceRuntime(
        config_factory=lambda: CONFIG,
        adb_client_factory=lambda config: AdbClient(
            config,
            command_executor=FakeExecutor(handler=handler),
        ),
        leases_dir=tmp_path / "leases",
    )
    with pytest.raises(
        DeviceUnavailableError,
        match="cleanup failed after the stage operation failed",
    ) as caught:
        with runtime.stage(
            FakeDeviceAdapter(),
            output_root=tmp_path / "artifacts",
            job_id="job1",
            stage_key="stage-key",
            attempt_id="attempt-001",
            push_files={},
        ):
            raise RuntimeError("original stage failure")

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "original stage failure"


def test_device_runtime_releases_empty_lease_when_record_fails(tmp_path) -> None:
    class FailingLease:
        def __init__(self) -> None:
            self.acquired = False
            self.released = False

        def acquire(self) -> None:
            self.acquired = True

        def record_attempt_dir(self, _attempt_dir: str) -> None:
            raise OSError("owner record failed")

        def release(self) -> None:
            self.released = True

    lease = FailingLease()
    runtime = DeviceRuntime(
        config_factory=lambda: CONFIG,
        lease_factory=lambda *_args: lease,
        leases_dir=tmp_path / "leases",
    )
    with pytest.raises(OSError, match="owner record failed"):
        with runtime.stage(
            FakeDeviceAdapter(),
            output_root=tmp_path / "artifacts",
            job_id="job1",
            stage_key="stage-key",
            attempt_id="attempt-001",
            push_files={},
        ):
            pass
    assert lease.acquired is True
    assert lease.released is True


# --------------------------------------------------------------------------- #
# device_doctor
# --------------------------------------------------------------------------- #

DF_OK = (
    "Filesystem 1K-blocks Used Available Use% Mounted on\n"
    "/dev/block/x 100G 10G 90G 10% /data/local/tmp\n"
)


def _doctor_handler(devices_stdout, *, state="device", state_rc=0, df_stdout=DF_OK):
    def handler(argv):
        if "devices" in argv:
            return FakeCompleted(stdout=devices_stdout)
        if "get-state" in argv:
            return FakeCompleted(returncode=state_rc, stdout=state, stderr="state-err")
        if any(str(a).startswith("df ") for a in argv):
            return FakeCompleted(stdout=df_stdout)
        return FakeCompleted()

    return handler


def test_device_doctor_healthy_and_require_healthy_passes() -> None:
    executor = FakeExecutor(handler=_doctor_handler("List of devices attached\nABC123\tdevice\n"))
    report = device_doctor(CONFIG, AdbClient(CONFIG, command_executor=executor))
    assert report["ok"] is True
    assert report["device_identifier"] == "ABC123@localhost:5037"
    assert report["checks"]["device_present"]["ok"] is True
    assert report["checks"]["remote_free_space"]["ok"] is True
    require_healthy(report)  # does not raise


def test_device_doctor_reports_missing_device() -> None:
    executor = FakeExecutor(
        handler=_doctor_handler("List of devices attached\nOTHER\tdevice\n", state="offline")
    )
    report = device_doctor(CONFIG, AdbClient(CONFIG, command_executor=executor))
    assert report["ok"] is False
    assert report["checks"]["server_reachable"]["ok"] is True
    assert report["checks"]["device_present"]["ok"] is False
    assert report["checks"]["device_state"]["ok"] is False
    with pytest.raises(DeviceUnavailableError):
        require_healthy(report)


def test_device_doctor_unreachable_server_is_not_ok() -> None:
    executor = FakeExecutor(handler=lambda argv: FakeCompleted(returncode=1, stderr="no server"))
    report = device_doctor(CONFIG, AdbClient(CONFIG, command_executor=executor))
    assert report["ok"] is False
    assert report["checks"]["server_reachable"]["ok"] is False
    with pytest.raises(DeviceUnavailableError):
        require_healthy(report)


def test_device_doctor_sdk_mismatch_flagged() -> None:
    executor = FakeExecutor(handler=_doctor_handler("List of devices attached\nABC123\tdevice\n"))
    report = device_doctor(CONFIG, AdbClient(CONFIG, command_executor=executor), sdk_build="999")
    assert report["checks"]["sdk_compatible"]["ok"] is False
    assert report["ok"] is False


def test_device_doctor_uses_injected_harness_identity() -> None:
    executor = FakeExecutor(
        handler=_doctor_handler(
            "List of devices attached\nABC123\tdevice\n"
        )
    )
    constraints = replace(
        DEFAULT_CONSTRAINTS,
        qairt_build_id="next-build",
    )

    report = device_doctor(
        CONFIG,
        AdbClient(CONFIG, command_executor=executor),
        sdk_build="next-build",
        constraints=constraints,
    )

    assert report["ok"] is True
    assert (
        report["checks"]["sdk_compatible"]["expected_sdk_build"]
        == "next-build"
    )
