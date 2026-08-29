from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from qairt_agent.agent import QairtAgentClient
from qairt_agent.cli import (
    _default_client,
    _apple_container_adb_server,
    _docker_adb_server,
    _spawn_worker,
    main,
)
from qairt_agent.contracts import (
    ArtifactKind,
    ArtifactRef,
    JobState,
    ToolResult,
)
from qairt_agent.errors import QairtAgentError
from qairt_agent.jobs.journal import JobJournal
from qairt_agent.vectors import VectorPreparer


def _queued_job(jobs_root: Path, job_id: str) -> JobJournal:
    return JobJournal.create(
        jobs_root,
        job_id,
        spec_original={},
        spec_resolved={},
        spec_sha256="0" * 64,
    )


def _claim_fake_job(jobs_root: Path, job_id: str) -> None:
    JobJournal.open(jobs_root, job_id).set_state(JobState.STAGING)


@pytest.mark.parametrize(
    "server",
    [
        "localhost:5037",
        "LOCALHOST.:5037",
        "127.0.0.2:5037",
        "::1:5037",
        "[::1]:5037",
    ],
)
def test_docker_adb_server_maps_all_loopback_spellings(server) -> None:
    assert _docker_adb_server(server) == "host.docker.internal:5037"


def test_docker_adb_server_preserves_non_loopback_connection() -> None:
    assert _docker_adb_server("adb.example.test:5037") == "adb.example.test:5037"


@pytest.mark.parametrize(
    "server",
    ["localhost:5037", "127.0.0.1:5037", "[::1]:5037"],
)
def test_apple_container_adb_server_maps_loopback(server) -> None:
    assert (
        _apple_container_adb_server(server)
        == "host.container.internal:5037"
    )


def spec_dict() -> dict:
    return {
        "family": "qwen3",
        "sources": {"text": {"onnx_path": "/m/model.onnx", "encodings_path": "/m/model.encodings"}},
        "output_root": "/artifacts/out",
        "vectors": {"mode": "provided", "validation_manifest": "/v/golden.json"},
    }


class FakeEngine:
    def __init__(self, workdir: Path) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.calls: list[str] = []
        self._n = 0

    def _result(self, stage: str) -> ToolResult:
        self.calls.append(stage)
        self._n += 1
        path = self.workdir / f"{stage}-{self._n}.json"
        path.write_text(json.dumps({"stage": stage}), encoding="utf-8")
        manifest = ArtifactRef.from_path(path, kind=ArtifactKind.MANIFEST)
        return ToolResult.success({"stage": stage}, manifest=manifest)

    def build(self, spec):
        return self._result("build")

    def build_genai_container(self, spec, config=None):
        return self._result("build")

    def validate(self, uri, sha, vector_manifest=None, config=None):
        return self._result("validate")

    def benchmark(self, uri, sha, config=None):
        return self._result("benchmark")

    def diagnose_quality(self, uri, sha, config=None):
        return self._result("diagnose")


def make_client(tmp_path) -> tuple[QairtAgentClient, FakeEngine]:
    fake = FakeEngine(tmp_path / "engine")
    client = QairtAgentClient(jobs_root=tmp_path / "jobs", engine_factory=lambda: fake, background=False)
    return client, fake


def write_spec(tmp_path) -> Path:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec_dict()), encoding="utf-8")
    return spec_path


def run(argv, client, *, spawner=None):
    out = io.StringIO()
    code = main(argv, client=client, out=out, spawner=spawner)
    lines = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    return code, lines


class FakeSpawner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, job_id, jobs_root):
        self.calls.append((job_id, jobs_root))
        return 4321


def test_init_and_doctor(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    code, lines = run(["init", "--root", str(tmp_path)], client)
    assert code == 0
    assert lines[0]["ok"] is True
    assert (tmp_path / "qairt-agent.toml").exists()

    code, lines = run(["doctor", "--root", str(tmp_path)], client)
    assert lines[0]["ok"] is False  # no SDK present
    names = {c["name"] for c in lines[0]["checks"]}
    assert "target" in names


def test_invalid_harness_is_one_structured_cli_error(tmp_path) -> None:
    from qairt_agent import project

    project.init(tmp_path)
    (tmp_path / "harness" / "constraints.json").write_text(
        "{not-json",
        encoding="utf-8",
    )
    client, _ = make_client(tmp_path)

    code, lines = run(
        ["doctor", "--root", str(tmp_path)],
        client,
    )

    assert code == 1
    assert len(lines) == 1
    assert lines[0]["ok"] is False
    assert lines[0]["error"]["code"] == "invalid_spec"
    assert lines[0]["error"]["stage"] == "harness"


def test_image_build_requires_mounted_sdk_api_smoke(tmp_path, monkeypatch) -> None:
    from qairt_agent import project
    from qairt_agent.docker import DockerRunner

    project.init(tmp_path)
    config_text = (tmp_path / project.CONFIG_FILENAME).read_text(
        encoding="utf-8"
    )
    (tmp_path / project.CONFIG_FILENAME).write_text(
        config_text.replace(
            'backend = "auto"', 'backend = "docker"'
        ),
        encoding="utf-8",
    )
    sdk = tmp_path / "qnn" / "qnn" / "qairt" / "2.49.0.260730"
    sdk.mkdir(parents=True)
    (sdk / "sdk.yaml").write_text(
        "version: 2.49.0\nbuild_id: 260730134355\n",
        encoding="utf-8",
    )
    calls: list[tuple[str, object]] = []

    def fake_build(self, *, context, dockerfile):
        calls.append(("build", (Path(context), Path(dockerfile))))

    def fake_smoke(self, *, sdk_root):
        calls.append(("smoke", Path(sdk_root)))

    image_id = "sha256:" + "1" * 64
    monkeypatch.setattr(DockerRunner, "build_image", fake_build)
    monkeypatch.setattr(DockerRunner, "require_image", lambda self: image_id)
    monkeypatch.setattr(DockerRunner, "smoke_test_sdk", fake_smoke)

    client, _ = make_client(tmp_path)
    code, lines = run(["image", "build", "--root", str(tmp_path)], client)

    assert code == 0
    assert calls == [
        ("build", (tmp_path.resolve(), tmp_path / "docker" / "worker.Dockerfile")),
        ("smoke", sdk),
    ]
    assert lines[0]["image_id"] == image_id
    assert lines[0]["sdk_root"] == str(sdk)
    assert lines[0]["smoke"] == "passed"


def test_image_smoke_does_not_rebuild_image(tmp_path, monkeypatch) -> None:
    from qairt_agent import project
    from qairt_agent.docker import DockerRunner

    project.init(tmp_path)
    config_text = (tmp_path / project.CONFIG_FILENAME).read_text(
        encoding="utf-8"
    )
    (tmp_path / project.CONFIG_FILENAME).write_text(
        config_text.replace(
            'backend = "auto"', 'backend = "docker"'
        ),
        encoding="utf-8",
    )
    sdk = tmp_path / "qnn" / "qnn" / "qairt" / "2.49.0.260730"
    sdk.mkdir(parents=True)
    (sdk / "sdk.yaml").write_text("version: 2.49.0\n", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(
        DockerRunner,
        "build_image",
        lambda self, **kwargs: (_ for _ in ()).throw(
            AssertionError("image smoke must not rebuild")
        ),
    )
    monkeypatch.setattr(
        DockerRunner,
        "require_available",
        lambda self: calls.append("available"),
    )
    monkeypatch.setattr(
        DockerRunner,
        "require_image",
        lambda self: calls.append("image") or "sha256:" + "2" * 64,
    )
    monkeypatch.setattr(
        DockerRunner,
        "smoke_test_sdk",
        lambda self, *, sdk_root: calls.append(f"smoke:{Path(sdk_root)}"),
    )

    client, _ = make_client(tmp_path)
    code, lines = run(["image", "smoke", "--root", str(tmp_path)], client)

    assert code == 0
    assert calls == ["available", "image", f"smoke:{sdk}"]
    assert lines[0]["action"] == "smoke"
    assert lines[0]["smoke"] == "passed"


def test_plan_resolves_preset(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    spec = spec_dict()
    spec["quality"] = {
        "sqnr_modes": ["full_reference", "teacher_forced", "chain"],
        "dump_intermediates_on_failure": True,
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    code, lines = run(["plan", "--spec", str(spec_path)], client)
    assert code == 0
    assert lines[0]["preset"] == "qwen3_dense"
    assert lines[0]["family"] == "qwen3"
    assert lines[0]["resolved"]["pipeline"] == "low_level"
    assert lines[0]["resolved"]["output_layout"]["contexts"] == (
        "/artifacts/out/runs/{run_id}/build/contexts"
    )
    assert lines[0]["quality"]["sqnr_modes"] == [
        "full_reference",
        "teacher_forced",
        "chain",
    ]
    assert lines[0]["effective_compile"]["enable_intermediate_outputs"] is True
    assert lines[0]["effective_benchmark"] == {
        "warmup_runs": 10,
        "measured_runs": 50,
        "optrace": False,
        "lane": "low_level",
        "sample_unit": "graph_invocation",
        "aa_calibration_doubles_runs": True,
    }


def test_plan_shows_the_genai_lane_benchmark_defaults(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    spec = spec_dict()
    del spec["family"]
    spec["preset"] = "qwen3_5"
    spec["metadata"] = {
        "attached_models_by_ar": {
            ar: {
                "model_path": f"/m/ar{ar}.onnx",
                "encodings_path": f"/m/ar{ar}.encodings",
            }
            for ar in ("1", "128")
        }
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    code, lines = run(["plan", "--spec", str(spec_path)], client)

    assert code == 0
    assert lines[0]["resolved"]["pipeline"] == "genai_builder"
    # `plan` must show what a run will actually execute, not the schema default.
    assert lines[0]["effective_benchmark"]["warmup_runs"] == 3
    assert lines[0]["effective_benchmark"]["measured_runs"] == 10
    assert lines[0]["effective_benchmark"]["sample_unit"] == "generate_call"


def test_build_inline_succeeds(tmp_path) -> None:
    client, fake = make_client(tmp_path)
    spec_path = write_spec(tmp_path)
    code, lines = run(["build", "--spec", str(spec_path), "--inline"], client)
    assert code == 0
    assert lines[0]["state"] == "succeeded"
    assert fake.calls == ["build"]


def test_workflow_inline_runs_core_stages(tmp_path) -> None:
    client, fake = make_client(tmp_path)
    spec_path = write_spec(tmp_path)
    code, lines = run(["workflow", "--spec", str(spec_path), "--inline"], client)
    assert code == 0
    assert fake.calls == ["build", "validate", "benchmark"]


def test_build_background_spawns_detached_worker(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    spec_path = write_spec(tmp_path)
    spawner = FakeSpawner()
    code, lines = run(["build", "--spec", str(spec_path)], client, spawner=spawner)
    assert code == 0
    assert lines[0]["worker_pid"] == 4321
    assert lines[0]["state"] == "queued"
    job_id, jobs_root = spawner.calls[0]
    assert job_id == lines[0]["job_id"]
    assert jobs_root == client.jobs_root


def test_detached_worker_immediate_exit_is_journaled_and_structured(
    tmp_path,
    monkeypatch,
) -> None:
    from qairt_agent import project
    from qairt_agent.docker import DockerRunner

    config = project.init(tmp_path)
    text = (tmp_path / project.CONFIG_FILENAME).read_text(
        encoding="utf-8"
    )
    (tmp_path / project.CONFIG_FILENAME).write_text(
        text.replace('backend = "auto"', 'backend = "docker"'),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        DockerRunner,
        "require_available",
        lambda self: None,
    )
    monkeypatch.setattr(
        DockerRunner,
        "require_image",
        lambda self: "sha256:" + "f" * 64,
    )
    journal = _queued_job(config.jobs_path, "launch-failed")

    class FailedProcess:
        pid = 7654

        def poll(self):
            return 125

    def fake_popen(argv, **kwargs):
        kwargs["stdout"].write(b"container runtime rejected mount\n")
        return FailedProcess()

    monkeypatch.setattr("qairt_agent.cli.subprocess.Popen", fake_popen)

    with pytest.raises(QairtAgentError) as raised:
        _spawn_worker("launch-failed", config.jobs_path)

    error = raised.value.to_tool_error()
    assert error.code.value == "docker_unavailable"
    assert error.stage == "worker_startup"
    status = journal.state()
    assert status.state is JobState.FAILED
    assert status.error is not None
    assert status.error.details["returncode"] == 125
    log_path = Path(str(status.error.details["log_path"]))
    assert log_path.is_file()
    assert "rejected mount" in log_path.read_text(encoding="utf-8")


def test_detached_worker_prelaunch_error_cannot_leave_job_queued(
    tmp_path,
    monkeypatch,
) -> None:
    from qairt_agent import project
    from qairt_agent.docker import DockerRunner
    from qairt_agent.errors import DockerUnavailableError

    config = project.init(tmp_path)
    text = (tmp_path / project.CONFIG_FILENAME).read_text(
        encoding="utf-8"
    )
    (tmp_path / project.CONFIG_FILENAME).write_text(
        text.replace('backend = "auto"', 'backend = "docker"'),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        DockerRunner,
        "require_available",
        lambda self: None,
    )
    monkeypatch.setattr(
        DockerRunner,
        "require_image",
        lambda self: (_ for _ in ()).throw(
            DockerUnavailableError(
                "image missing",
                stage="docker",
                retryable=True,
            )
        ),
    )
    journal = _queued_job(config.jobs_path, "prelaunch-failed")

    with pytest.raises(QairtAgentError) as raised:
        _spawn_worker("prelaunch-failed", config.jobs_path)

    assert raised.value.to_tool_error().code.value == "docker_unavailable"
    assert journal.state().state is JobState.FAILED


def test_detached_worker_startup_timeout_cannot_leave_job_queued(
    tmp_path,
    monkeypatch,
) -> None:
    from qairt_agent import project
    from qairt_agent.docker import DockerRunner

    config = project.init(tmp_path)
    text = (tmp_path / project.CONFIG_FILENAME).read_text(
        encoding="utf-8"
    )
    (tmp_path / project.CONFIG_FILENAME).write_text(
        text.replace('backend = "auto"', 'backend = "docker"'),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        DockerRunner,
        "require_available",
        lambda self: None,
    )
    monkeypatch.setattr(
        DockerRunner,
        "require_image",
        lambda self: "sha256:" + "e" * 64,
    )
    monkeypatch.setenv("QAIRT_AGENT_WORKER_STARTUP_TIMEOUT", "0.01")
    journal = _queued_job(config.jobs_path, "launch-timeout")

    class StuckProcess:
        pid = 7655

        def __init__(self) -> None:
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = StuckProcess()
    monkeypatch.setattr(
        "qairt_agent.cli.subprocess.Popen",
        lambda argv, **kwargs: process,
    )

    with pytest.raises(QairtAgentError) as raised:
        _spawn_worker("launch-timeout", config.jobs_path)

    assert raised.value.to_tool_error().stage == "worker_startup"
    assert process.terminated is True
    assert journal.state().state is JobState.FAILED


def test_default_spawner_uses_docker_with_sdk_and_adb_mapping(tmp_path, monkeypatch) -> None:
    from qairt_agent import project
    from qairt_agent.docker import DockerRunner

    config = project.init(tmp_path)
    config_text = (tmp_path / project.CONFIG_FILENAME).read_text(encoding="utf-8")
    (tmp_path / project.CONFIG_FILENAME).write_text(
        config_text.replace('backend = "auto"', 'backend = "docker"'),
        encoding="utf-8",
    )
    sdk = tmp_path / "qnn" / "qnn" / "qairt" / "2.49.0.260730"
    sdk.mkdir(parents=True)
    (sdk / "sdk.yaml").write_text(
        "version: 2.49.0\nbuild_id: 260730134355\n", encoding="utf-8"
    )
    monkeypatch.setenv("QAIRT_AGENT_ADB_SERIAL", "SERIAL")
    monkeypatch.setenv("QAIRT_AGENT_ADB_SERVER", "localhost:5037")
    monkeypatch.setattr(DockerRunner, "require_available", lambda self: None)
    image_id = "sha256:" + "a" * 64
    monkeypatch.setattr(DockerRunner, "require_image", lambda self: image_id)

    captured = {}
    _queued_job(config.jobs_path, "job-1")

    class FakeProcess:
        pid = 9876

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        _claim_fake_job(config.jobs_path, "job-1")
        return FakeProcess()

    monkeypatch.setattr("qairt_agent.cli.subprocess.Popen", fake_popen)

    pid = _spawn_worker("job-1", config.jobs_path)

    assert pid == 9876
    argv = captured["argv"]
    assert argv[:3] == ["docker", "run", "--rm"]
    assert str(sdk) + ":/opt/qairt:ro" in argv
    assert str(tmp_path.resolve()) + ":/workspace:ro" in argv
    assert str(config.models_path) + ":/models:ro" in argv
    assert str(config.jobs_path) + ":/state/jobs" in argv
    assert "QAIRT_SDK_ROOT=/opt/qairt" in argv
    assert "QNN_SDK_ROOT=/opt/qairt" in argv
    assert "QAIRT_AGENT_ADB_SERIAL=SERIAL" in argv
    assert "QAIRT_AGENT_ADB_SERVER=host.docker.internal:5037" in argv
    assert "QAIRT_AGENT_ADB_CANONICAL_SERVER=localhost:5037" in argv
    assert "QAIRT_AGENT_LEASES_DIR=/state/leases" in argv
    assert f"QAIRT_AGENT_IMAGE_DIGEST={image_id}" in argv
    assert "QAIRT_AGENT_DEVICE_FINGERPRINT=SERIAL@localhost:5037" in argv
    assert "HOME=/tmp/qairt-agent-home" in argv
    assert "XDG_CACHE_HOME=/tmp/qairt-agent-cache" in argv
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert all("HEXAGON_TOOLS_DIR=" not in item for item in argv)
    assert argv[-8:] == [
        "/opt/venv/bin/python",
        "-m",
        "qairt_agent.cli",
        "--jobs-root",
        str(config.jobs_path),
        "_worker",
        "--job-id",
        "job-1",
    ]


def test_default_spawner_uses_apple_container_on_macos_backend(
    tmp_path,
    monkeypatch,
) -> None:
    from qairt_agent import project
    from qairt_agent.apple_container import AppleContainerRunner

    config = project.init(tmp_path)
    config_text = (tmp_path / project.CONFIG_FILENAME).read_text(
        encoding="utf-8"
    )
    (tmp_path / project.CONFIG_FILENAME).write_text(
        config_text.replace(
            'backend = "auto"', 'backend = "apple_container"'
        ),
        encoding="utf-8",
    )
    sdk = tmp_path / "qnn" / "qnn" / "qairt" / "2.49.0.260730"
    sdk.mkdir(parents=True)
    (sdk / "sdk.yaml").write_text(
        "version: 2.49.0\nbuild_id: 260730134355\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QAIRT_AGENT_ADB_SERIAL", "SERIAL")
    monkeypatch.setenv("QAIRT_AGENT_ADB_SERVER", "localhost:5037")
    monkeypatch.setattr(
        AppleContainerRunner,
        "require_available",
        lambda self: None,
    )
    image_id = "sha256:" + "9" * 64
    monkeypatch.setattr(
        AppleContainerRunner,
        "require_image",
        lambda self: image_id,
    )
    aliases: list[str] = []
    monkeypatch.setattr(
        AppleContainerRunner,
        "require_host_alias",
        lambda self, alias: aliases.append(alias),
    )
    captured = {}
    _queued_job(config.jobs_path, "job-apple")

    class FakeProcess:
        pid = 9877

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        _claim_fake_job(config.jobs_path, "job-apple")
        return FakeProcess()

    monkeypatch.setattr("qairt_agent.cli.subprocess.Popen", fake_popen)

    pid = _spawn_worker("job-apple", config.jobs_path)

    assert pid == 9877
    assert aliases == ["host.container.internal"]
    argv = captured["argv"]
    assert argv[:3] == ["container", "run", "--rm"]
    assert argv[argv.index("--platform") + 1] == "linux/amd64"
    assert (
        f"type=bind,source={sdk},target=/opt/qairt,readonly"
        in argv
    )
    assert (
        f"type=bind,source={tmp_path.resolve()},target=/workspace,readonly"
        in argv
    )
    assert "QAIRT_AGENT_ADB_SERIAL=SERIAL" in argv
    assert (
        "QAIRT_AGENT_ADB_SERVER=host.container.internal:5037"
        in argv
    )
    assert (
        "QAIRT_AGENT_ADB_CANONICAL_SERVER=localhost:5037"
        in argv
    )
    assert "QAIRT_AGENT_WORKER_BACKEND=apple_container" in argv
    assert (
        "QAIRT_AGENT_HARNESS_CONSTRAINTS=/workspace/"
        "harness/constraints.json"
        in argv
    )
    assert argv[argv.index("--user") + 1] == (
        f"{os.getuid()}:{os.getgid()}"
    )
    assert "--add-host" not in argv
    assert argv[-8:] == [
        "/opt/venv/bin/python",
        "-m",
        "qairt_agent.cli",
        "--jobs-root",
        str(config.jobs_path),
        "_worker",
        "--job-id",
        "job-apple",
    ]


def test_default_spawner_mounts_and_uses_custom_jobs_root(tmp_path, monkeypatch) -> None:
    from qairt_agent import project
    from qairt_agent.docker import DockerRunner

    config = project.init(tmp_path)
    text = (tmp_path / project.CONFIG_FILENAME).read_text(encoding="utf-8")
    (tmp_path / project.CONFIG_FILENAME).write_text(
        text.replace('backend = "auto"', 'backend = "docker"'),
        encoding="utf-8",
    )
    custom_jobs = tmp_path / "custom-state" / "jobs"
    custom_jobs.mkdir(parents=True)
    monkeypatch.setattr(DockerRunner, "require_available", lambda self: None)
    monkeypatch.setattr(
        DockerRunner,
        "require_image",
        lambda self: "sha256:" + "b" * 64,
    )
    captured = {}
    _queued_job(custom_jobs, "custom-job")

    class FakeProcess:
        pid = 1234

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        _claim_fake_job(custom_jobs, "custom-job")
        return FakeProcess()

    monkeypatch.setattr("qairt_agent.cli.subprocess.Popen", fake_popen)

    assert _spawn_worker("custom-job", custom_jobs) == 1234
    argv = captured["argv"]
    assert f"{custom_jobs.resolve()}:/state/jobs" in argv
    assert f"{custom_jobs.resolve()}:{custom_jobs.resolve()}" in argv
    assert argv[-5:] == [
        "--jobs-root",
        str(custom_jobs.resolve()),
        "_worker",
        "--job-id",
        "custom-job",
    ]


def test_container_environment_populates_stage_provenance(tmp_path, monkeypatch) -> None:
    image_id = "sha256:" + "c" * 64
    monkeypatch.setenv("QAIRT_AGENT_WORKER_PROVENANCE", "1")
    monkeypatch.setenv("QAIRT_AGENT_IMAGE_DIGEST", image_id)
    monkeypatch.setenv("QAIRT_AGENT_PLATFORM_ABI", "ubuntu22.04-amd64")
    monkeypatch.setenv("QAIRT_AGENT_HOST_ARCH", "arm64")
    monkeypatch.setenv("QAIRT_AGENT_EMULATION", "true")
    monkeypatch.setenv(
        "QAIRT_AGENT_DEVICE_FINGERPRINT",
        "SERIAL@localhost:5037",
    )

    client = _default_client(str(tmp_path / "jobs"))
    provenance = client._provenance_for(  # noqa: SLF001
        SimpleNamespace(resolved_preset_sha256="d" * 64)
    )

    assert provenance.image_digest == image_id
    assert provenance.device_fingerprint == "SERIAL@localhost:5037"
    assert provenance.platform_abi == "ubuntu22.04-amd64"
    assert provenance.host_arch == "arm64"
    assert provenance.emulation is True
    assert provenance.resolved_preset_sha256 == "d" * 64


def test_native_spawner_injects_device_provenance_without_image(
    tmp_path,
    monkeypatch,
) -> None:
    from qairt_agent import project

    config = project.init(tmp_path)
    text = (tmp_path / project.CONFIG_FILENAME).read_text(encoding="utf-8")
    (tmp_path / project.CONFIG_FILENAME).write_text(
        text.replace('backend = "auto"', 'backend = "native"'),
        encoding="utf-8",
    )
    sdk = tmp_path / "qnn" / "qnn" / "qairt" / "2.49.0.260730"
    sdk.mkdir(parents=True)
    (sdk / "sdk.yaml").write_text(
        "version: 2.49.0\nbuild_id: 260730134355\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QAIRT_AGENT_ADB_SERIAL", "NATIVE-SERIAL")
    monkeypatch.setenv("QAIRT_AGENT_ADB_SERVER", "localhost:5037")
    captured = {}
    _queued_job(config.jobs_path, "native-job")

    class FakeProcess:
        pid = 2468

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["env"] = dict(kwargs["env"])
        _claim_fake_job(config.jobs_path, "native-job")
        return FakeProcess()

    monkeypatch.setattr("qairt_agent.cli.subprocess.Popen", fake_popen)

    assert _spawn_worker("native-job", config.jobs_path) == 2468
    assert captured["argv"][1:3] == ["-m", "qairt_agent.cli"]
    worker_env = captured["env"]
    assert worker_env["QAIRT_AGENT_WORKER_PROVENANCE"] == "1"
    assert worker_env["QAIRT_AGENT_SDK_BUILD"] == "260730134355"
    assert worker_env["QAIRT_AGENT_PLATFORM_ABI"].startswith("ubuntu22.04-")
    assert (
        worker_env["QAIRT_AGENT_DEVICE_FINGERPRINT"]
        == "NATIVE-SERIAL@localhost:5037"
    )
    assert worker_env["QAIRT_AGENT_ADB_CANONICAL_SERVER"] == "localhost:5037"
    assert "QAIRT_AGENT_IMAGE_DIGEST" not in worker_env
    assert worker_env["QAIRT_SDK_ROOT"] == str(sdk)
    assert worker_env["QNN_SDK_ROOT"] == str(sdk)
    assert worker_env["QAIRT_AGENT_LEASES_DIR"] == str(
        config.state_path / "leases"
    )

    for name, value in worker_env.items():
        if name.startswith("QAIRT_AGENT_"):
            monkeypatch.setenv(name, value)
    client = _default_client(str(tmp_path / "native-client-jobs"))
    provenance = client._provenance_for(  # noqa: SLF001
        SimpleNamespace(resolved_preset_sha256="e" * 64)
    )
    assert provenance.image_digest is None
    assert provenance.device_fingerprint == "NATIVE-SERIAL@localhost:5037"


def test_plain_default_client_does_not_manufacture_worker_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("QAIRT_AGENT_WORKER_PROVENANCE", raising=False)
    monkeypatch.delenv("QAIRT_AGENT_IMAGE_DIGEST", raising=False)
    monkeypatch.delenv("QAIRT_AGENT_DEVICE_FINGERPRINT", raising=False)

    client = _default_client(str(tmp_path / "jobs"))

    assert client._provenance is None  # noqa: SLF001


def test_worker_command_executes_prepared_job(tmp_path) -> None:
    client, fake = make_client(tmp_path)
    spec_path = write_spec(tmp_path)
    spawner = FakeSpawner()
    run(["build", "--spec", str(spec_path)], client, spawner=spawner)
    job_id = spawner.calls[0][0]

    code, lines = run(["--jobs-root", str(client.jobs_root), "_worker", "--job-id", job_id], client)
    assert code == 0
    assert lines[0]["state"] == "succeeded"
    assert fake.calls == ["build"]


def test_worker_command_returns_nonzero_for_failed_job(tmp_path) -> None:
    class FailingEngine(FakeEngine):
        def build(self, spec):
            raise RuntimeError("compile failed")

    fake = FailingEngine(tmp_path / "engine")
    client = QairtAgentClient(
        jobs_root=tmp_path / "jobs",
        engine_factory=lambda: fake,
        background=False,
    )
    spec_path = write_spec(tmp_path)
    spawner = FakeSpawner()
    run(["build", "--spec", str(spec_path)], client, spawner=spawner)
    job_id = spawner.calls[0][0]

    code, lines = run(
        ["--jobs-root", str(client.jobs_root), "_worker", "--job-id", job_id],
        client,
    )

    assert code == 1
    assert lines[0]["state"] == "failed"


def test_job_status_list_cancel(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    spec_path = write_spec(tmp_path)
    _, lines = run(["build", "--spec", str(spec_path), "--inline"], client)
    job_id = lines[0]["job_id"]

    _, status_lines = run(["job", "status", job_id], client)
    assert status_lines[0]["state"] == "succeeded"

    _, list_lines = run(["job", "list"], client)
    assert job_id in list_lines[0]["jobs"]

    _, cancel_lines = run(["job", "cancel", job_id], client)
    assert cancel_lines[0]["cancel_requested"] is True


def test_job_resume_detached_prepares_without_host_execution(tmp_path) -> None:
    client, fake = make_client(tmp_path)
    spec_path = write_spec(tmp_path)
    initial_spawner = FakeSpawner()
    _, lines = run(["build", "--spec", str(spec_path)], client, spawner=initial_spawner)
    job_id = lines[0]["job_id"]
    assert fake.calls == []

    resume_spawner = FakeSpawner()
    code, resume_lines = run(
        ["job", "resume", job_id],
        client,
        spawner=resume_spawner,
    )

    assert code == 0
    assert fake.calls == []
    assert resume_spawner.calls == [(job_id, client.jobs_root)]
    assert resume_lines[0]["worker_pid"] == 4321


def test_job_resume_succeeded_is_noop_without_spawning(tmp_path) -> None:
    client, fake = make_client(tmp_path)
    spec_path = write_spec(tmp_path)
    _, lines = run(["build", "--spec", str(spec_path), "--inline"], client)
    job_id = lines[0]["job_id"]
    assert fake.calls == ["build"]

    spawner = FakeSpawner()
    code, resume_lines = run(
        ["job", "resume", job_id],
        client,
        spawner=spawner,
    )

    assert code == 0
    assert spawner.calls == []
    assert resume_lines[0]["state"] == "succeeded"
    assert "worker_pid" not in resume_lines[0]


def test_validate_requires_from_job(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    spec_path = write_spec(tmp_path)
    code, lines = run(["validate", "--spec", str(spec_path), "--inline"], client)
    assert code == 2
    assert "requires --from-job" in lines[0]["error"]


def test_diagnose_from_job_submits_default_auto_evidence_stage(
    tmp_path,
) -> None:
    class DiagnoseCommandClient:
        def __init__(self) -> None:
            self.jobs_root = tmp_path / "jobs"
            self.prepared: list[dict[str, object]] = []

        def job(self, job_id):
            assert job_id == "parent-job"
            return SimpleNamespace(
                journal=SimpleNamespace(spec_original=lambda: spec_dict())
            )

        def prepare(
            self,
            spec,
            *,
            stages,
            initial_manifest_job=None,
        ):
            self.prepared.append(
                {
                    "spec": spec,
                    "stages": stages,
                    "initial_manifest_job": initial_manifest_job,
                }
            )
            return SimpleNamespace(
                job_id="diagnose-job",
                status_path=tmp_path / "jobs" / "diagnose-job" / "state.json",
            )

        def execute(self, job_id):
            assert job_id == "diagnose-job"
            return SimpleNamespace(state=JobState.SUCCEEDED)

    client = DiagnoseCommandClient()
    code, lines = run(
        [
            "diagnose",
            "--from-job",
            "parent-job",
            "--inline",
        ],
        client,
    )

    assert code == 0
    assert lines[0]["state"] == "succeeded"
    assert client.prepared == [
        {
            "spec": spec_dict(),
            "stages": ("diagnose",),
            "initial_manifest_job": "parent-job",
        }
    ]


def test_rerun_inline_reuses_stages(tmp_path) -> None:
    client, fake = make_client(tmp_path)
    spec_path = write_spec(tmp_path)
    _, lines = run(["workflow", "--spec", str(spec_path), "--inline"], client)
    first_job = lines[0]["job_id"]
    assert fake.calls == ["build", "validate", "benchmark"]

    fake.calls.clear()
    code, rerun_lines = run(["rerun", "--from-job", first_job, "--inline"], client)
    assert code == 0
    assert rerun_lines[0]["state"] == "succeeded"
    assert fake.calls == []  # all stages reused from the parent


def test_artifact_verify(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    target = tmp_path / "blob.bin"
    target.write_bytes(b"hello")
    import hashlib

    sha = hashlib.sha256(b"hello").hexdigest()
    code, lines = run(["artifact", "verify", str(target), "--sha256", sha], client)
    assert code == 0
    assert lines[0]["ok"] is True

    code, lines = run(["artifact", "verify", str(target), "--sha256", "0" * 64], client)
    assert code == 1
    assert lines[0]["ok"] is False


def test_vectors_import_pickle_via_cli(tmp_path) -> None:
    import pickle

    import numpy as np

    client, _ = make_client(tmp_path)
    pickle_path = tmp_path / "vectors.pkl"
    pickle_path.write_bytes(
        pickle.dumps(
            {
                "inputs": {
                    "input_ids": np.zeros((1, 2), dtype="int64"),
                },
                "goldens": {
                    "logits": np.zeros((1, 2, 4), dtype="float32"),
                },
            }
        )
    )
    out_dir = tmp_path / "bundle"

    # Without --trusted-local the import fails closed.
    code, lines = run(["vectors", "import-pickle", str(pickle_path), "--output-dir", str(out_dir)], client)
    assert code == 1
    assert lines[0]["ok"] is False

    code, lines = run(
        ["vectors", "import-pickle", str(pickle_path), "--output-dir", str(out_dir), "--trusted-local"],
        client,
    )
    assert code == 0
    assert lines[0]["ok"] is True
    assert lines[0]["execution_ready"] is True
    assert Path(lines[0]["manifest_path"]).is_file()
    assert Path(lines[0]["bundle_path"]).is_file()
    tensors = lines[0]["bundle"]["tensors"]
    assert [t["name"] for t in tensors] == ["input_ids", "logits"]
    assert any((out_dir / "raw").glob("input_ids-*.raw"))
    assert any((out_dir / "raw").glob("logits-*.raw"))


def test_vectors_import_pickle_cli_supports_explicit_format_and_section(
    tmp_path,
    monkeypatch,
) -> None:
    import pickle

    import numpy as np
    import qairt_agent.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "_dispatch_torch_pickle_import",
        lambda _args: pytest.fail("NumPy pickle must stay on the local path"),
    )

    client, _ = make_client(tmp_path)
    pickle_path = tmp_path / "inputs.pkl"
    pickle_path.write_bytes(
        pickle.dumps(
            {
                "input_ids": np.array([[1, 2]], dtype=np.int64),
                "attention_mask": np.ones((1, 2), dtype=np.int32),
            }
        )
    )
    out_dir = tmp_path / "input-bundle"

    code, lines = run(
        [
            "vectors",
            "import-pickle",
            str(pickle_path),
            "--output-dir",
            str(out_dir),
            "--trusted-local",
            "--format",
            "numpy-pickle",
            "--section",
            "inputs",
        ],
        client,
    )

    assert code == 0
    assert lines[0]["ok"] is True
    assert lines[0]["source_format"] == "numpy-pickle"
    assert lines[0]["section"] == "inputs"
    assert lines[0]["execution_ready"] is True
    manifest = VectorPreparer.load_manifest(lines[0]["manifest_path"])
    assert set(manifest.inputs) == {"input_ids", "attention_mask"}
    assert manifest.goldens == {}


@pytest.mark.parametrize(
    ("backend", "runner_name"),
    [
        ("apple_container", "AppleContainerRunner"),
        ("docker", "DockerRunner"),
    ],
)
def test_vectors_torch_archive_dispatches_to_configured_isolated_worker(
    tmp_path,
    monkeypatch,
    backend,
    runner_name,
) -> None:
    import pickle
    import zipfile

    import numpy as np
    import qairt_agent.cli as cli_module
    from qairt_agent.harness import load_harness_constraints
    from qairt_agent.vectors_pickle import import_pickle_artifacts

    source = tmp_path / "goldens.pt"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("archive/data.pkl", b"weights-only payload")
        stream.writestr("archive/version", b"3")
    source.write_bytes(archive.getvalue())
    output = tmp_path / "imported"
    for directory in (
        tmp_path / "state",
        tmp_path / "artifacts",
        tmp_path / "cache",
        tmp_path / "models",
        tmp_path / "qnn",
    ):
        directory.mkdir()

    constraints = load_harness_constraints()
    config = SimpleNamespace(
        effective_worker_backend=backend,
        harness=constraints,
        docker_image=constraints.worker_image,
        docker_platform=constraints.platform,
        state_path=tmp_path / "state",
        artifacts_path=tmp_path / "artifacts",
        cache_path=tmp_path / "cache",
        sdk_path=tmp_path / "qnn",
        models_path=tmp_path / "models",
        project_root=tmp_path,
        harness_constraints="harness/constraints.json",
    )
    calls: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def require_available(self):
            calls["available"] = True

        def require_image(self):
            calls["image"] = True
            return "sha256:" + "a" * 64

        def run_build_isolated(self, **kwargs):
            calls["isolated"] = kwargs
            source_mount = kwargs["mounts"].compatibility_mounts[1]
            mounted_source = Path(source_mount.source)
            calls["mounted_source_is_dir"] = mounted_source.is_dir()
            if mounted_source.is_dir():
                calls["mounted_source_names"] = tuple(
                    path.name for path in mounted_source.iterdir()
                )
                calls["mounted_source_bytes"] = (
                    mounted_source / "archive.pt"
                ).read_bytes()
            else:
                calls["mounted_source_bytes"] = mounted_source.read_bytes()
            imported = import_pickle_artifacts(
                pickle.dumps(
                    {"goldens": {"logits": np.ones(1, dtype=np.float32)}}
                ),
                output_dir=output,
                trusted_local=True,
                source_key=str(source.resolve()),
            )
            source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            bundle_payload = imported.bundle.model_dump(mode="json")
            bundle_payload["source_sha256"] = source_sha256
            bundle_payload["metadata"]["source_path"] = str(source.resolve())
            imported.bundle_path.write_text(
                json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest_payload = json.loads(
                imported.manifest_path.read_text(encoding="utf-8")
            )
            manifest_payload["metadata"]["source_sha256"] = source_sha256
            manifest_payload["metadata"]["source_path"] = str(
                source.resolve()
            )
            imported.manifest_path.write_text(
                json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "bundle": bundle_payload,
                        "bundle_path": "/qairt-agent-output/vector_bundle.json",
                        "manifest_path": "/qairt-agent-output/vector_manifest.json",
                        "execution_ready": False,
                        "source_format": "torch",
                        "section": "goldens",
                    }
                )
                + "\n",
                stderr="",
            )

    monkeypatch.delenv("QAIRT_AGENT_PICKLE_IMPORT_LOCAL", raising=False)
    monkeypatch.setattr(
        cli_module.project,
        "find_project_root",
        lambda _start: tmp_path,
    )
    monkeypatch.setattr(cli_module.project, "load", lambda _root: config)
    monkeypatch.setattr(cli_module, runner_name, FakeRunner)

    client, _ = make_client(tmp_path)
    code, lines = run(
        [
            "vectors",
            "import-pickle",
            str(source),
            "--output-dir",
            str(output),
            "--trusted-local",
            "--format",
            "auto",
            "--section",
            "goldens",
        ],
        client,
    )

    assert code == 0
    assert lines[0]["ok"] is True
    assert lines[0]["execution_backend"] == backend
    assert lines[0]["worker_image_identity"] == "sha256:" + "a" * 64
    invocation = calls["isolated"]
    assert isinstance(invocation, dict)
    assert invocation["env"]["QAIRT_AGENT_PICKLE_IMPORT_LOCAL"] == "1"
    assert invocation["env"]["QAIRT_AGENT_PICKLE_SOURCE_PATH"] == str(
        source.resolve()
    )
    assert invocation["env"]["QAIRT_AGENT_PICKLE_SOURCE_SHA256"] == (
        hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert invocation["env"]["PYTHONPATH"] == cli_module.WORKER_PYTHONPATH
    assert invocation["command"][:3] == [
        "/opt/venv/bin/python",
        "-m",
        "qairt_agent.cli",
    ]
    assert invocation["command"][
        invocation["command"].index("--format") + 1
    ] == "torch"
    assert invocation["workdir"] == "/workspace"
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        assert invocation["user"] == f"{os.getuid()}:{os.getgid()}"

    compatibility = invocation["mounts"].compatibility_mounts
    assert compatibility[0].source == str(output.resolve())
    assert compatibility[0].target == "/qairt-agent-output"
    assert compatibility[0].read_only is False
    assert compatibility[1].read_only is True
    assert calls["mounted_source_bytes"] == source.read_bytes()
    if backend == "apple_container":
        assert compatibility[1].target == "/qairt-agent-input"
        assert calls["mounted_source_is_dir"] is True
        assert calls["mounted_source_names"] == ("archive.pt",)
        assert not Path(compatibility[1].source).exists()
    else:
        assert compatibility[1].source == str(source.resolve())
        assert compatibility[1].target == "/qairt-agent-input/archive.pt"
        assert calls["mounted_source_is_dir"] is False
    assert str(output.resolve()) == lines[0]["bundle_path"].rsplit("/", 1)[0]
    assert str(output.resolve()) == lines[0]["manifest_path"].rsplit("/", 1)[0]


def test_device_doctor_fails_closed_without_env(tmp_path, monkeypatch) -> None:
    client, _ = make_client(tmp_path)
    monkeypatch.delenv("QAIRT_AGENT_ADB_SERIAL", raising=False)
    monkeypatch.delenv("QAIRT_AGENT_ADB_SERVER", raising=False)
    code, lines = run(["device", "doctor"], client)
    assert code == 1
    assert lines[0]["ok"] is False
    assert lines[0]["error"]["code"] == "device_unavailable"


def test_device_gc_non_dry_run_fails_closed_without_env(
    tmp_path,
    monkeypatch,
) -> None:
    client, _ = make_client(tmp_path)
    monkeypatch.delenv("QAIRT_AGENT_ADB_SERIAL", raising=False)
    monkeypatch.delenv("QAIRT_AGENT_ADB_SERVER", raising=False)
    code, lines = run(
        [
            "device",
            "gc",
            "--leases-dir",
            str(tmp_path / "leases"),
        ],
        client,
    )
    assert code == 1
    assert lines[0]["error"]["code"] == "device_unavailable"


def test_device_gc_dry_run_does_not_require_device_env(
    tmp_path,
    monkeypatch,
) -> None:
    client, _ = make_client(tmp_path)
    monkeypatch.delenv("QAIRT_AGENT_ADB_SERIAL", raising=False)
    monkeypatch.delenv("QAIRT_AGENT_ADB_SERVER", raising=False)
    code, lines = run(
        [
            "device",
            "gc",
            "--leases-dir",
            str(tmp_path / "leases"),
            "--dry-run",
        ],
        client,
    )
    assert code == 0
    assert lines[0]["dry_run"] is True
