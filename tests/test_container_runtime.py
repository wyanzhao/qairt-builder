"""The container contract exists once and both backends render it identically.

The two runners were ~300 near-duplicate lines. A change made in one and not the
other was a silent divergence between backends -- and the backends are chosen by
host platform, so the divergence would only surface on someone else's machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qairt_agent.apple_container import AppleContainerRunner
from qairt_agent.container_runtime import (
    SDK_MOUNT_TARGET,
    SMOKE_COMMAND,
    flatten_env,
    image_build_args,
    resolve_harness_build_path,
    worker_smoke_env,
)
from qairt_agent.docker import DockerRunner
from qairt_agent.harness import DEFAULT_CONSTRAINTS


def _pairs(argv: list[str], flag: str) -> list[str]:
    return [argv[index + 1] for index, item in enumerate(argv) if item == flag]


def test_both_backends_pass_the_same_build_args() -> None:
    calls: dict[str, list[str]] = {}

    docker = DockerRunner(
        constraints=DEFAULT_CONSTRAINTS,
        command_executor=lambda argv: calls.setdefault("docker", argv)
        and None
        or _completed(),
    )
    apple = AppleContainerRunner(
        constraints=DEFAULT_CONSTRAINTS,
        command_executor=lambda argv: calls.setdefault("apple", argv)
        and None
        or _completed(),
    )
    docker.is_available = lambda: True  # type: ignore[method-assign]
    docker.require_available = lambda: None  # type: ignore[method-assign]
    apple.require_available = lambda: None  # type: ignore[method-assign]

    context = DEFAULT_CONSTRAINTS.source_path.parent.parent
    docker.build_image(context=context, dockerfile=context / "docker" / "worker.Dockerfile")
    apple.build_image(context=context, dockerfile=context / "docker" / "worker.Dockerfile")

    assert _pairs(calls["docker"], "--build-arg") == _pairs(calls["apple"], "--build-arg")
    assert _pairs(calls["docker"], "--build-arg") == image_build_args(
        DEFAULT_CONSTRAINTS, "harness/constraints.json"
    )[1::2]


def _completed():
    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    return _Result()


def test_both_backends_smoke_under_the_same_environment(tmp_path) -> None:
    sdk = tmp_path / "qairt"
    sdk.mkdir()
    (sdk / "sdk.yaml").write_text("version: 2.49.0", encoding="utf-8")

    docker_argv = DockerRunner(constraints=DEFAULT_CONSTRAINTS).build_sdk_smoke_argv(
        sdk_root=sdk
    )
    apple_argv = AppleContainerRunner(
        constraints=DEFAULT_CONSTRAINTS
    ).build_sdk_smoke_argv(sdk_root=sdk)

    # The environment must be identical: the smoke test proves the image can
    # import the pinned SDK, and a backend-specific environment would make that
    # proof backend-specific.
    assert _pairs(docker_argv, "-e") == _pairs(apple_argv, "--env")
    assert _pairs(docker_argv, "-e") == [
        f"{key}={value}" for key, value in worker_smoke_env().items()
    ]
    # Both run the same command against the same mount target.
    assert docker_argv[-len(SMOKE_COMMAND):] == list(SMOKE_COMMAND)
    assert apple_argv[-len(SMOKE_COMMAND):] == list(SMOKE_COMMAND)
    assert f":{SDK_MOUNT_TARGET}:ro" in " ".join(docker_argv)
    assert f"target={SDK_MOUNT_TARGET}" in " ".join(apple_argv)


def test_the_isolation_difference_is_kept_not_papered_over() -> None:
    sdk_docker = DockerRunner(constraints=DEFAULT_CONSTRAINTS).build_sdk_smoke_argv(
        sdk_root="/opt/qairt"
    )
    sdk_apple = AppleContainerRunner(
        constraints=DEFAULT_CONSTRAINTS
    ).build_sdk_smoke_argv(sdk_root="/opt/qairt")

    # Docker gets real IP-egress isolation; Apple `container` 1.0 does not, and
    # the argv must keep saying so.
    assert "--network" in sdk_docker and "none" in sdk_docker
    assert "--no-dns" in sdk_apple
    assert "--network" not in sdk_apple


def test_constraints_outside_the_build_context_fail_closed_per_backend(
    tmp_path,
) -> None:
    from qairt_agent.errors import (
        AppleContainerUnavailableError,
        DockerUnavailableError,
    )

    elsewhere = tmp_path / "harness" / "constraints.json"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text(
        DEFAULT_CONSTRAINTS.source_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    from qairt_agent.harness import load_harness_constraints

    constraints = load_harness_constraints(elsewhere)

    for error_type in (DockerUnavailableError, AppleContainerUnavailableError):
        with pytest.raises(error_type):
            resolve_harness_build_path(
                constraints,
                tmp_path / "somewhere-else",
                on_outside_context=lambda a, b, error_type=error_type: error_type(
                    "outside", stage="build"
                ),
            )


def test_flatten_env_preserves_order_and_pairs() -> None:
    assert flatten_env({"A": "1", "B": "2"}, flag="-e") == ["-e", "A=1", "-e", "B=2"]
    assert flatten_env({"A": "1"}, flag="--env") == ["--env", "A=1"]
