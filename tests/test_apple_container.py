from __future__ import annotations

import json

import pytest

from qairt_agent.apple_container import AppleContainerRunner
from qairt_agent.docker import WorkerImageConfig, default_mounts
from qairt_agent.errors import AppleContainerUnavailableError, ErrorCode


class FakeCompleted:
    def __init__(
        self,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeExecutor:
    def __init__(self, results: list[FakeCompleted] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.results = list(results or [])

    def __call__(self, argv: list[str]) -> FakeCompleted:
        self.calls.append(list(argv))
        if self.results:
            return self.results.pop(0)
        return FakeCompleted()

    @property
    def last_argv(self) -> list[str]:
        return self.calls[-1]


@pytest.fixture
def container_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qairt_agent.apple_container.runner.shutil.which",
        lambda name: "/usr/local/bin/container" if name == "container" else None,
    )


def _available_results() -> list[FakeCompleted]:
    return [
        FakeCompleted(
            stdout="container CLI version 1.0.0 "
            "(build: release, commit: ee848e3)\n"
        ),
        FakeCompleted(
            stdout=json.dumps(
                {
                    "status": "running",
                    "apiServerVersion": (
                        "container-apiserver version 1.0.0 "
                        "(commit: ee848e3)"
                    ),
                }
            )
        ),
    ]


def _inspect_document() -> str:
    return json.dumps(
        [
            {
                "name": "worker:test",
                "variants": [
                    {
                        "platform": {
                            "os": "linux",
                            "architecture": "amd64",
                        },
                        "config": {"digest": "sha256:" + "a" * 64},
                    }
                ],
            }
        ]
    )


def test_run_uses_linux_amd64_rosetta_mounts_env_workdir_and_user(
    container_on_path,
) -> None:
    executor = FakeExecutor([*_available_results(), FakeCompleted()])
    runner = AppleContainerRunner(
        command_executor=executor,
        image=WorkerImageConfig(image_ref="worker:test"),
        host_arch=lambda: "arm64",
    )
    mounts = default_mounts(
        "/host/state",
        "/host/artifacts",
        "/host/cache",
        sdk_root="/host/sdk",
        workspace="/host/project",
    )

    runner.run(
        mounts=mounts,
        command=["python", "-m", "worker"],
        env={"FOO": "bar"},
        user="501:20",
        workdir="/workspace",
    )

    argv = executor.last_argv
    assert argv[:3] == ["container", "run", "--rm"]
    assert argv[argv.index("--platform") + 1] == "linux/amd64"
    assert "--rosetta" in argv
    assert argv[argv.index("--user") + 1] == "501:20"
    assert argv[argv.index("--workdir") + 1] == "/workspace"
    assert "FOO=bar" in argv
    assert (
        "type=bind,source=/host/sdk,target=/opt/qairt,readonly"
        in argv
    )
    assert "type=bind,source=/host/state,target=/state" in argv
    assert argv[-4:] == ["worker:test", "python", "-m", "worker"]


def test_build_isolated_is_honest_best_effort_no_dns() -> None:
    runner = AppleContainerRunner(
        image=WorkerImageConfig(image_ref="worker:test"),
        host_arch=lambda: "arm64",
    )

    argv = runner.build_run_argv(
        mounts=default_mounts("state", "artifacts", "cache"),
        command=["true"],
        network=False,
    )

    assert "--no-dns" in argv
    assert "--network" not in argv


def test_run_nonzero_is_structured(container_on_path) -> None:
    executor = FakeExecutor(
        [
            *_available_results(),
            FakeCompleted(returncode=125, stderr="runtime failed"),
        ]
    )
    runner = AppleContainerRunner(
        command_executor=executor,
        image=WorkerImageConfig(image_ref="worker:test"),
    )

    with pytest.raises(AppleContainerUnavailableError) as caught:
        runner.run(
            mounts=default_mounts("state", "artifacts", "cache"),
            command=["true"],
        )

    error = caught.value.to_tool_error()
    assert error.stage == "apple-container-run"
    assert error.details["returncode"] == 125


def test_build_image_uses_harness_build_args(
    container_on_path,
    tmp_path,
) -> None:
    executor = FakeExecutor([*_available_results(), FakeCompleted()])
    runner = AppleContainerRunner(
        command_executor=executor,
        image=WorkerImageConfig(image_ref="worker:test"),
    )

    runner.build_image(
        context=tmp_path,
        dockerfile=tmp_path / "worker.Dockerfile",
    )

    argv = executor.last_argv
    assert argv[:3] == ["container", "build", "--platform"]
    assert argv[3] == "linux/amd64"
    assert argv[argv.index("--file") + 1] == str(
        tmp_path / "worker.Dockerfile"
    )
    assert argv[argv.index("--tag") + 1] == "worker:test"
    assert "UBUNTU_VERSION=22.04" in argv
    assert "PYTHON_VERSION=3.10" in argv
    assert (
        "QAIRT_DEPENDENCIES_FILE="
        "docker/requirements-qairt-2.48.0.260626.txt"
        in argv
    )
    assert "HARNESS_CONSTRAINTS_FILE=harness/constraints.json" in argv
    assert "TORCH_VERSION=2.4.1" in argv


def test_require_image_checks_amd64_variant_and_returns_stable_hash() -> None:
    executor = FakeExecutor(
        [FakeCompleted(stdout=_inspect_document())]
    )
    runner = AppleContainerRunner(
        command_executor=executor,
        image=WorkerImageConfig(image_ref="worker:test"),
    )

    first = runner.require_image()

    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64
    assert executor.last_argv == [
        "container",
        "image",
        "inspect",
        "worker:test",
    ]


def test_require_image_rejects_missing_amd64_variant() -> None:
    document = _inspect_document().replace("amd64", "arm64")
    runner = AppleContainerRunner(
        command_executor=FakeExecutor(
            [FakeCompleted(stdout=document)]
        ),
        image=WorkerImageConfig(image_ref="worker:test"),
    )

    with pytest.raises(
        AppleContainerUnavailableError,
        match="no linux/amd64 variant",
    ):
        runner.require_image()


def test_sdk_smoke_uses_rosetta_read_only_sdk_and_no_dns(
    container_on_path,
    tmp_path,
) -> None:
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "sdk.yaml").write_text("version: 2.48.0\n", encoding="utf-8")
    executor = FakeExecutor(
        [
            *_available_results(),
            FakeCompleted(stdout=_inspect_document()),
            FakeCompleted(stdout='{"ok":true}\n'),
        ]
    )
    runner = AppleContainerRunner(
        command_executor=executor,
        image=WorkerImageConfig(image_ref="worker:test"),
        host_arch=lambda: "arm64",
    )

    result = runner.smoke_test_sdk(sdk_root=sdk)

    assert result.returncode == 0
    argv = executor.last_argv
    assert "--rosetta" in argv
    assert "--no-dns" in argv
    assert (
        f"type=bind,source={sdk.resolve()},target=/opt/qairt,readonly"
        in argv
    )
    assert (
        "PYTHONPATH=/opt/qairt-agent/qairt-agent-src.zip:"
        "/opt/qairt/lib/python:/opt/qairt/benchmarks/QNN"
    ) in argv
    assert (
        "QAIRT_AGENT_HARNESS_CONSTRAINTS="
        "/opt/qairt-agent/harness/constraints.json"
        in argv
    )
    assert argv[-3:] == [
        "/opt/venv/bin/python",
        "-m",
        "qairt_agent.docker.smoke",
    ]


def test_wrong_container_cli_version_is_structured(
    container_on_path,
) -> None:
    runner = AppleContainerRunner(
        command_executor=FakeExecutor(
            [FakeCompleted(stdout="container CLI version 1.1.0\n")]
        )
    )

    with pytest.raises(AppleContainerUnavailableError) as caught:
        runner.require_available()

    payload = caught.value.to_tool_error()
    assert payload.code == ErrorCode.APPLE_CONTAINER_UNAVAILABLE
    assert payload.stage == "apple-container"
    assert payload.details["required_version"] == "1.0.0"
    assert payload.details["actual_version"] == "1.1.0"


def test_wrong_container_api_server_version_is_structured(
    container_on_path,
) -> None:
    runner = AppleContainerRunner(
        command_executor=FakeExecutor(
            [
                FakeCompleted(
                    stdout="container CLI version 1.0.0\n"
                ),
                FakeCompleted(
                    stdout=json.dumps(
                        {
                            "status": "running",
                            "apiServerVersion": (
                                "container-apiserver version 1.1.0"
                            ),
                        }
                    )
                ),
            ]
        )
    )

    with pytest.raises(AppleContainerUnavailableError) as caught:
        runner.require_available()

    payload = caught.value.to_tool_error()
    assert payload.details["required_version"] == "1.0.0"
    assert payload.details["api_server_version"] == "1.1.0"


def test_host_adb_alias_must_be_configured() -> None:
    missing = AppleContainerRunner(
        command_executor=FakeExecutor(
            [FakeCompleted(stdout="[]\n")]
        )
    )

    with pytest.raises(
        AppleContainerUnavailableError,
        match="not configured for ADB",
    ) as caught:
        missing.require_host_alias("host.container.internal")

    assert "sudo container system dns create" in str(
        caught.value.details["setup_command"]
    )

    present = AppleContainerRunner(
        command_executor=FakeExecutor(
            [
                FakeCompleted(
                    stdout='[{"domain":"host.container.internal"}]\n'
                )
            ]
        )
    )
    present.require_host_alias("host.container.internal")

    false_positive = AppleContainerRunner(
        command_executor=FakeExecutor(
            [
                FakeCompleted(
                    stdout=(
                        '[{"description":"host.container.internal"}]\n'
                    )
                )
            ]
        )
    )
    with pytest.raises(AppleContainerUnavailableError):
        false_positive.require_host_alias("host.container.internal")
