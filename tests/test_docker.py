from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from qairt_agent.docker import (
    BindMount,
    DEFAULT_PLATFORM,
    DockerImageConfig,
    DockerRunner,
    RuntimeMounts,
    build_context_excludes,
    default_mounts,
    image_provenance,
    resolve_image_digest,
)
from qairt_agent.errors import DockerUnavailableError
from qairt_agent.harness import DEFAULT_CONSTRAINTS


def _docker_version(
    client: str = "26.1.0",
    server: str = "26.1.0",
) -> str:
    return (
        '{"Client":{"Version":"'
        + client
        + '"},"Server":{"Version":"'
        + server
        + '"}}\n'
    )


class FakeCompleted:
    """A duck-typed stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeExecutor:
    """Records every argv it receives and returns queued results.

    When the queue is exhausted it returns a default success result, so the
    ``docker version`` availability probe passes unless told otherwise.
    """

    def __init__(self, results: list[FakeCompleted] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._results = list(results or [])

    def __call__(self, argv: list[str]) -> FakeCompleted:
        self.calls.append(list(argv))
        if self._results:
            return self._results.pop(0)
        return FakeCompleted(stdout=_docker_version())

    @property
    def last_argv(self) -> list[str]:
        return self.calls[-1]


@pytest.fixture
def docker_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the docker binary is on PATH so availability hinges on the probe."""

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")


def _mounts() -> RuntimeMounts:
    return default_mounts("state-vol", "artifacts-vol", "cache-vol")


# --------------------------------------------------------------------------- #
# RuntimeMounts
# --------------------------------------------------------------------------- #


def test_runtime_mounts_ro_and_rw() -> None:
    args = _mounts().to_docker_args()

    # SDK and workspace are mounted read-only.
    assert "./qnn/qnn:/opt/qairt:ro" in args
    assert "./models:/workspace:ro" in args

    # The three caller-provided volumes are read-write (no :ro suffix).
    for rw in ("state-vol:/state", "artifacts-vol:/artifacts", "cache-vol:/cache"):
        assert rw in args
        assert f"{rw}:ro" not in args

    # Every mount is preceded by a -v flag.
    assert args.count("-v") == 5


def test_runtime_mounts_include_jobs_and_host_compatibility_aliases() -> None:
    mounts = default_mounts(
        "/host/state",
        "/host/artifacts",
        "/host/cache",
        sdk_root="/host/sdk",
        workspace="/host/project",
        jobs_volume="/host/jobs",
        models_root="/host/project/models",
        compatibility_mounts=(
            BindMount(source="/host/project", target="/host/project", read_only=True),
            BindMount(source="/host/artifacts", target="/host/artifacts"),
        ),
    )

    args = mounts.to_docker_args()
    assert "/host/jobs:/state/jobs" in args
    assert "/host/project/models:/models:ro" in args
    assert "/host/project:/host/project:ro" in args
    assert "/host/artifacts:/host/artifacts" in args


# --------------------------------------------------------------------------- #
# DockerRunner.run argv
# --------------------------------------------------------------------------- #


def test_run_argv_contains_platform_mounts_workdir_image_and_command(docker_on_path) -> None:
    executor = FakeExecutor()
    image = DockerImageConfig(image_ref="ghcr.io/me/worker:tag")
    runner = DockerRunner(command_executor=executor, image=image)

    runner.run(
        mounts=_mounts(),
        command=["python", "-c", "print(1)"],
        env={"FOO": "bar"},
        user="1000:1001",
    )

    argv = executor.last_argv
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--platform") + 1] == DEFAULT_PLATFORM == "linux/amd64"
    assert argv[argv.index("--workdir") + 1] == "/workspace"
    assert argv[argv.index("--user") + 1] == "1000:1001"
    assert "./qnn/qnn:/opt/qairt:ro" in argv
    assert "./models:/workspace:ro" in argv
    assert argv[argv.index("-e") + 1] == "FOO=bar"
    # The image ref sits immediately before the command.
    assert argv[-4] == "ghcr.io/me/worker:tag"
    assert argv[-3:] == ["python", "-c", "print(1)"]
    # Network defaults to on: no isolation flag.
    assert "--network" not in argv


def test_run_network_false_adds_network_none(docker_on_path) -> None:
    executor = FakeExecutor()
    runner = DockerRunner(command_executor=executor, image=DockerImageConfig(image_ref="img"))

    runner.run(mounts=_mounts(), command=["true"], network=False)

    argv = executor.last_argv
    assert argv[argv.index("--network") + 1] == "none"


def test_run_nonzero_is_structured(docker_on_path) -> None:
    executor = FakeExecutor(
        [
            FakeCompleted(stdout=_docker_version()),
            FakeCompleted(returncode=125, stderr="runtime failed"),
        ]
    )
    runner = DockerRunner(
        command_executor=executor,
        image=DockerImageConfig(image_ref="img"),
    )

    with pytest.raises(DockerUnavailableError) as caught:
        runner.run(mounts=_mounts(), command=["true"])

    error = caught.value.to_tool_error()
    assert error.stage == "docker-run"
    assert error.details["returncode"] == 125


def test_run_build_isolated_disables_network(docker_on_path) -> None:
    executor = FakeExecutor()
    runner = DockerRunner(command_executor=executor, image=DockerImageConfig(image_ref="img"))

    runner.run_build_isolated(mounts=_mounts(), command=["build-it"])

    argv = executor.last_argv
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[-1] == "build-it"


def test_add_host_gateway_uses_harness_alias() -> None:
    constraints = replace(
        DEFAULT_CONSTRAINTS,
        docker_host_alias="adb-host.example",
    )
    runner = DockerRunner(
        image=DockerImageConfig(image_ref="img"),
        constraints=constraints,
    )

    argv = runner.build_run_argv(
        mounts=_mounts(),
        command=["true"],
        add_host_gateway=True,
    )

    assert "adb-host.example:host-gateway" in argv
    assert "host.docker.internal:host-gateway" not in argv


def test_build_image_uses_pinned_platform_dockerfile_and_tag(docker_on_path, tmp_path) -> None:
    executor = FakeExecutor()
    image = DockerImageConfig(image_ref="qairt-worker:test")
    runner = DockerRunner(command_executor=executor, image=image)
    dockerfile = tmp_path / "worker.Dockerfile"

    runner.build_image(context=tmp_path, dockerfile=dockerfile)

    argv = executor.last_argv
    assert argv[:3] == ["docker", "build", "--platform"]
    assert argv[3] == "linux/amd64"
    assert argv[argv.index("--file") + 1] == str(dockerfile)
    assert argv[argv.index("--tag") + 1] == "qairt-worker:test"
    for expected in (
        "UBUNTU_VERSION=22.04",
        "PYTHON_VERSION=3.10",
        "QAIRT_DEPENDENCIES_FILE="
        "docker/requirements-qairt-2.49.0.260730.txt",
        "HARNESS_CONSTRAINTS_FILE=harness/constraints.json",
        "TORCH_VERSION=2.4.1",
    ):
        assert expected in argv


def test_sdk_smoke_argv_mounts_only_sdk_and_import_test_is_offline(tmp_path) -> None:
    sdk = tmp_path / "qairt" / "2.49.0.260730"
    sdk.mkdir(parents=True)
    (sdk / "sdk.yaml").write_text(
        "version: 2.49.0\nbuild_id: 260730134355\n",
        encoding="utf-8",
    )
    runner = DockerRunner(image=DockerImageConfig(image_ref="worker:test"))

    argv = runner.build_sdk_smoke_argv(sdk_root=sdk)

    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--platform") + 1] == "linux/amd64"
    assert argv[argv.index("--network") + 1] == "none"
    assert f"{sdk.resolve()}:/opt/qairt:ro" in argv
    assert "QAIRT_SDK_ROOT=/opt/qairt" in argv
    assert "QNN_SDK_ROOT=/opt/qairt" in argv
    assert (
        "QAIRT_AGENT_HARNESS_CONSTRAINTS="
        "/opt/qairt-agent/harness/constraints.json"
        in argv
    )
    assert (
        "PYTHONPATH=/opt/qairt-agent/qairt-agent-src.zip:"
        "/opt/qairt/lib/python:/opt/qairt/benchmarks/QNN"
    ) in argv
    assert all("HEXAGON_TOOLS_DIR=" not in item for item in argv)
    assert argv[-3:] == [
        "/opt/venv/bin/python",
        "-m",
        "qairt_agent.docker.smoke",
    ]


def test_sdk_smoke_executes_after_availability_and_image_checks(
    docker_on_path,
    tmp_path,
) -> None:
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "sdk.yaml").write_text("version: 2.49.0\n", encoding="utf-8")
    image_id = "sha256:" + "d" * 64
    executor = FakeExecutor(
        results=[
            FakeCompleted(stdout=_docker_version()),
            FakeCompleted(stdout=image_id),
            FakeCompleted(stdout='{"ok": true}\n'),
        ]
    )
    runner = DockerRunner(
        command_executor=executor,
        image=DockerImageConfig(image_ref="worker:test"),
    )

    result = runner.smoke_test_sdk(sdk_root=sdk)

    assert result.returncode == 0
    assert executor.calls[0] == [
        "docker",
        "version",
        "--format",
        "{{json .}}",
    ]
    assert executor.calls[1][:3] == ["docker", "image", "inspect"]
    assert executor.calls[2][-3:] == [
        "/opt/venv/bin/python",
        "-m",
        "qairt_agent.docker.smoke",
    ]


def test_sdk_smoke_fails_closed_on_import_failure(docker_on_path, tmp_path) -> None:
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "sdk.yaml").write_text("version: 2.49.0\n", encoding="utf-8")
    executor = FakeExecutor(
        results=[
            FakeCompleted(stdout=_docker_version()),
            FakeCompleted(stdout="sha256:" + "e" * 64),
            FakeCompleted(returncode=1, stderr="No module named 'yaml'"),
        ]
    )
    runner = DockerRunner(
        command_executor=executor,
        image=DockerImageConfig(image_ref="worker:test"),
    )

    with pytest.raises(DockerUnavailableError, match="Python API smoke test"):
        runner.smoke_test_sdk(sdk_root=sdk)


def test_worker_image_pins_sdk_dependencies_and_never_copies_sdk() -> None:
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "docker" / "worker.Dockerfile").read_text(encoding="utf-8")
    smoke = (root / "src" / "qairt_agent" / "docker" / "smoke.py").read_text(
        encoding="utf-8"
    )
    requirements = (
        root / DEFAULT_CONSTRAINTS.dependencies_file
    ).read_text(encoding="utf-8")
    requirement_lines = {
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert requirement_lines == {
        "absl-py==2.1.0",
        "aenum==3.1.15",
        "attrs==23.2.0",
        "dash==2.12.1",
        "decorator==4.4.2",
        "invoke==1.7.3",
        "joblib==1.4.0",
        "jsonschema==4.19.0",
        "lxml==5.2.1",
        "mako==1.2.0",
        "matplotlib==3.10.8",
        "mock==3.0.5",
        "numpy==1.26.4",
        "opencv-python==4.8.1.78",
        "optuna==3.3.0",
        "packaging==24.0",
        "pandas==2.3.3",
        "paramiko==3.5.1",
        "pathlib2==2.3.6",
        "pillow==10.2.0",
        "plotly==5.20.0",
        "psutil==6.1.1",
        "pydantic==2.8.2",
        "pytest==8.1.1",
        "pyyaml==6.0.3",
        "rich==13.9.4",
        "safetensors==0.4.3",
        "scikit-optimize==0.9.0",
        "scipy==1.15.3",
        "six==1.16.0",
        "tabulate==0.9.0",
        "typing-extensions==4.14.0",
        "xlsxwriter==1.2.2",
        "islpy==2025.2.5",
        "onnx==1.16.1",
        "onnx-ir==0.1.12",
        "onnxruntime==1.17.1",
        "onnxscript==0.5.6",
        "sentencepiece==0.2.0",
        "tiktoken==0.7.0",
        "tqdm==4.65.0",
        "transformers==4.57.1",
    }
    assert "COPY docker/.generated/qairt-agent-src.zip" in dockerfile
    assert "COPY pyproject.toml README.md" not in dockerfile
    assert "COPY src ./src" not in dockerfile
    assert (
        "PYTHONPATH=/opt/qairt-agent/qairt-agent-src.zip:"
        "/opt/qairt/lib/python:/opt/qairt/benchmarks/QNN"
    ) in dockerfile

    assert '"python${PYTHON_VERSION}" -m venv /opt/venv' in dockerfile
    assert '"${TORCH_INDEX_URL}"' in dockerfile
    assert '"torch==${TORCH_VERSION}"' in dockerfile
    assert "${QAIRT_DEPENDENCIES_FILE}" in dockerfile
    assert "${HARNESS_CONSTRAINTS_FILE}" in dockerfile
    assert "pip install --no-cache-dir --no-deps ." not in dockerfile
    assert "/opt/venv/bin/python -m pip check" in dockerfile
    assert "sys.version_info[:2] == expected" in dockerfile
    assert "sys.version_info[:2] == (3, 10)" not in dockerfile
    assert "_EXPECTED_VERSIONS" not in smoke
    assert "load_harness_constraints" in smoke
    assert "from qairt import Model, compile, convert" in smoke
    assert "import qairt.optimizer.onnx as onnx_optimizer" in smoke
    assert "import qairt.api.transforms._transform as transform_api" in smoke
    assert "import qairt.api.transforms.model_transformer_config" in smoke
    assert "import qairt.api.configs.common as common_config" in smoke
    assert "Qwen3_5BuilderHTP" in smoke
    assert "Qwen3OmniAudioEncoderBuilderHTP" in smoke
    assert "from qti.aisw.genai import genie as native_genie" in smoke
    assert "converter_common.ir_graph" in smoke
    assert "converter_common.modeltools" in smoke
    assert "dlc_utils.dlcontainer" in smoke
    assert "model_level_utils.py_net_run" in smoke
    assert "dummy_lib_genie" in smoke
    assert "PATH=/opt/venv/bin:" in dockerfile
    assert "HEXAGON_TOOLS_DIR" not in dockerfile
    assert "COPY qnn" not in dockerfile
    assert '"libpython${PYTHON_VERSION}"' in dockerfile
    assert '"python${PYTHON_VERSION}"' in dockerfile
    assert '"python${PYTHON_VERSION}-venv"' in dockerfile
    for system_package in {
        "clang",
        "flatbuffers-compiler",
        "libc++-dev",
        "libc++abi-dev",
        "libflatbuffers-dev",
        "libgl1",
        "libllvm14",
        "libncurses6",
        "lsb-release",
        "make",
        "python3-distutils",
        "rename",
    }:
        assert system_package in dockerfile


# --------------------------------------------------------------------------- #
# image_provenance / emulation
# --------------------------------------------------------------------------- #


def test_image_provenance_arm64_emulates() -> None:
    digest = "sha256:" + "a" * 64
    config = DockerImageConfig(image_ref="img", digest=digest)

    provenance = image_provenance(
        config, host_arch="arm64", sdk_build="260730134355", adapter_capability="explicit_factory"
    )

    assert provenance.emulation is True
    assert provenance.platform_abi == "ubuntu22.04-amd64"
    assert provenance.host_arch == "arm64"
    assert provenance.image_digest == digest
    assert provenance.sdk_build == "260730134355"
    assert provenance.adapter_capability == "explicit_factory"


@pytest.mark.parametrize("host_arch", ["x86_64", "amd64"])
def test_image_provenance_native_amd64_no_emulation(host_arch: str) -> None:
    config = DockerImageConfig(image_ref="img")

    provenance = image_provenance(
        config, host_arch=host_arch, sdk_build="b", adapter_capability="cap"
    )

    assert provenance.emulation is False
    assert provenance.platform_abi == "ubuntu22.04-amd64"
    assert provenance.host_arch == host_arch


# --------------------------------------------------------------------------- #
# build_context_excludes
# --------------------------------------------------------------------------- #


def test_build_context_excludes_keep_sdk_models_artifacts_out() -> None:
    excludes = build_context_excludes()

    for pattern in ("qnn/", "models/", "artifacts/", "*.onnx", "*.bin", "*.raw"):
        assert pattern in excludes


# --------------------------------------------------------------------------- #
# Availability / fail closed
# --------------------------------------------------------------------------- #


def test_require_available_raises_when_docker_not_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    runner = DockerRunner(command_executor=FakeExecutor())

    with pytest.raises(DockerUnavailableError, match=r"[Ii]nstall Docker"):
        runner.require_available()


def test_require_available_raises_when_probe_fails(docker_on_path) -> None:
    executor = FakeExecutor(results=[FakeCompleted(returncode=1, stderr="cannot connect to daemon")])
    runner = DockerRunner(command_executor=executor)

    with pytest.raises(DockerUnavailableError, match=r"[Ii]nstall Docker"):
        runner.require_available()


def test_require_available_rejects_docker_below_harness_minimum(
    docker_on_path,
) -> None:
    executor = FakeExecutor(
        results=[FakeCompleted(stdout=_docker_version("26.1.0", "23.0.6"))]
    )
    runner = DockerRunner(command_executor=executor)

    with pytest.raises(DockerUnavailableError, match="minimum 24.0.0"):
        runner.require_available()


def test_run_fails_closed_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    executor = FakeExecutor()
    runner = DockerRunner(command_executor=executor)

    with pytest.raises(DockerUnavailableError):
        runner.run(mounts=_mounts(), command=["true"])

    # No docker run was dispatched when unavailable.
    assert all(call[:2] != ["docker", "run"] for call in executor.calls)


def test_require_image_returns_immutable_local_image_id() -> None:
    image_id = "sha256:" + "f" * 64
    executor = FakeExecutor(results=[FakeCompleted(stdout=image_id + "\n")])
    runner = DockerRunner(
        command_executor=executor,
        image=DockerImageConfig(image_ref="worker:tag"),
    )

    assert runner.require_image() == image_id
    assert executor.last_argv == [
        "docker",
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        "worker:tag",
    ]


# --------------------------------------------------------------------------- #
# resolve_image_digest
# --------------------------------------------------------------------------- #


def test_resolve_image_digest_returns_digest() -> None:
    digest = "sha256:" + "ab" * 32
    executor = FakeExecutor(results=[FakeCompleted(stdout=f"ubuntu@{digest}\n")])

    resolved = resolve_image_digest("ubuntu:22.04", runner=executor)

    assert resolved == digest
    argv = executor.last_argv
    assert argv[:3] == ["docker", "image", "inspect"]
    assert "ubuntu:22.04" in argv


def test_resolve_image_digest_raises_on_empty_stdout() -> None:
    executor = FakeExecutor(results=[FakeCompleted(returncode=0, stdout="")])

    with pytest.raises(DockerUnavailableError):
        resolve_image_digest("ubuntu:22.04", runner=executor)


def test_resolve_image_digest_raises_on_failure() -> None:
    executor = FakeExecutor(results=[FakeCompleted(returncode=1, stderr="no such image")])

    with pytest.raises(DockerUnavailableError):
        resolve_image_digest("ubuntu:22.04", runner=executor)


# --------------------------------------------------------------------------- #
# DockerImageConfig validation
# --------------------------------------------------------------------------- #


def test_image_config_rejects_non_amd64_platform() -> None:
    with pytest.raises(ValidationError):
        DockerImageConfig(image_ref="img", platform="linux/arm64")


def test_image_config_defaults_are_pinned() -> None:
    config = DockerImageConfig(image_ref="img")

    assert config.platform == "linux/amd64"
    assert config.ubuntu_version == "22.04"
    assert config.python_version == "3.10"
    assert config.digest is None
