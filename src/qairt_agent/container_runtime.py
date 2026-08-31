"""The container contract both worker backends implement.

`docker/runner.py` and `apple_container/runner.py` were ~300 near-duplicate
lines: identical build-arg assembly, identical smoke environment, identical
harness-path resolution, differing only in the command word, the isolation flag
and the exception type. Every change to the container contract had to be made
twice, and a change made once was a silent divergence between the two backends.

This module holds the parts that must be identical. What genuinely differs --
Docker's `--network none` against Apple `container`'s weaker `--no-dns`, the
file-bind staging Apple `container` needs, the Rosetta flag, the availability
probe -- stays in each backend, because those are real differences and
pretending otherwise would be worse than the duplication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from qairt_agent.docker.image import WORKER_PYTHONPATH
from qairt_agent.harness import (
    DEFAULT_CONSTRAINTS,
    DEFAULT_CONSTRAINTS_LOGICAL_PATH,
    HarnessConstraints,
)

#: Where the SDK is mounted inside the worker, and what the smoke test runs.
SDK_MOUNT_TARGET = "/opt/qairt"
WORKER_WORKDIR = "/opt/qairt-agent"
SMOKE_COMMAND = ("/opt/venv/bin/python", "-m", "qairt_agent.docker.smoke")


def resolve_harness_build_path(
    constraints: HarnessConstraints,
    context: str | Path,
    *,
    on_outside_context: Callable[[Path, Path], Exception],
) -> str:
    """The constraints path as the image build sees it.

    A constraints file outside the build context cannot be copied into the
    image, so the build must fail rather than silently baking in the default.
    The one exception is the checked-in default, which has a known logical
    path inside the context.
    """

    context_path = Path(context).expanduser().resolve()
    constraints_path = constraints.source_path.expanduser().resolve()
    try:
        return constraints_path.relative_to(context_path).as_posix()
    except ValueError:
        if constraints_path == DEFAULT_CONSTRAINTS.source_path.resolve():
            return DEFAULT_CONSTRAINTS_LOGICAL_PATH
        raise on_outside_context(constraints_path, context_path)


def image_build_args(
    constraints: HarnessConstraints, harness_build_path: str
) -> list[str]:
    """The `--build-arg` pairs the worker Dockerfile expects.

    Every value comes from the reviewed harness; nothing here may be overridden
    ad hoc, which is why this is assembled in one place for both backends.
    """

    return [
        "--build-arg",
        f"UBUNTU_VERSION={constraints.ubuntu_version}",
        "--build-arg",
        f"PYTHON_VERSION={constraints.python_version}",
        "--build-arg",
        f"QAIRT_DEPENDENCIES_FILE={constraints.dependencies_file}",
        "--build-arg",
        f"HARNESS_CONSTRAINTS_FILE={harness_build_path}",
        "--build-arg",
        f"TORCH_VERSION={constraints.torch_version}",
        "--build-arg",
        f"TORCH_INDEX_URL={constraints.torch_index_url}",
    ]


def worker_smoke_env() -> dict[str, str]:
    """The environment the mounted-SDK smoke test runs under.

    Identical for both backends: the smoke test proves the *image* can import
    the pinned SDK, and an environment that differed between backends would
    make that proof backend-specific.
    """

    return {
        "QAIRT_SDK_ROOT": SDK_MOUNT_TARGET,
        "QNN_SDK_ROOT": SDK_MOUNT_TARGET,
        "QAIRT_AGENT_HARNESS_CONSTRAINTS": (
            f"{WORKER_WORKDIR}/harness/constraints.json"
        ),
        "PYTHONPATH": WORKER_PYTHONPATH,
        "LD_LIBRARY_PATH": f"{SDK_MOUNT_TARGET}/lib/x86_64-linux-clang",
    }


def flatten_env(env: dict[str, str], *, flag: str) -> list[str]:
    """Render an environment block as repeated ``flag KEY=VALUE`` pairs.

    Docker spells the flag ``-e`` and Apple `container` spells it ``--env``;
    the pairs themselves must not diverge.
    """

    argv: list[str] = []
    for key, value in env.items():
        argv += [flag, f"{key}={value}"]
    return argv


def require_sdk_root(
    sdk_root: str | Path, *, on_missing: Callable[[Path], Exception]
) -> Path:
    """Refuse to smoke-test against something that is not an SDK install."""

    resolved = Path(sdk_root).expanduser().resolve()
    if not (resolved / "sdk.yaml").is_file():
        raise on_missing(resolved)
    return resolved


__all__ = [
    "SDK_MOUNT_TARGET",
    "SMOKE_COMMAND",
    "WORKER_PYTHONPATH",
    "WORKER_WORKDIR",
    "flatten_env",
    "image_build_args",
    "require_sdk_root",
    "resolve_harness_build_path",
    "worker_smoke_env",
]
