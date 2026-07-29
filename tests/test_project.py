from __future__ import annotations

import pytest

from qairt_agent.errors import ProjectNotInitializedError
from qairt_agent.harness import HarnessConstraintsError
from qairt_agent.project import (
    CONFIG_FILENAME,
    ProjectConfig,
    _parse_memory_gb,
    discover_sdk_path,
    doctor,
    init,
    load,
    resolve_logical_uri,
    select_container_backend,
    select_worker_backend,
    to_logical_uri,
)


def test_init_creates_config_and_dirs(tmp_path) -> None:
    config = init(tmp_path)
    assert (tmp_path / CONFIG_FILENAME).exists()
    assert (tmp_path / ".qairt-agent" / "jobs").is_dir()
    assert (tmp_path / ".qairt-agent" / "state").is_dir()
    assert (tmp_path / "artifacts").is_dir()
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "harness" / "constraints.json").is_file()
    assert (tmp_path / "docker" / "worker.Dockerfile").is_file()
    assert (
        tmp_path / "docker" / "requirements-qairt-2.48.0.260626.txt"
    ).is_file()
    assert (tmp_path / "docker" / ".generated" / "qairt-agent-src.zip").is_file()
    assert (tmp_path / ".dockerignore").is_file()
    assert config.sdk_root == "./qnn/qnn"
    assert config.worker_backend == "auto"
    assert config.target_chipset == "SM8850"


def test_init_does_not_require_sdk(tmp_path) -> None:
    # init succeeds even with no SDK present; doctor reports it separately.
    config = init(tmp_path)
    assert not config.sdk_path.exists()


def test_load_round_trips_and_missing_raises(tmp_path) -> None:
    init(tmp_path)
    config = load(tmp_path)
    assert config.docker_platform == "linux/amd64"
    assert config.target_soc_model == 87
    assert config.docker_image.startswith("qairt-agent-worker:")

    with pytest.raises(ProjectNotInitializedError, match="not initialized"):
        load(tmp_path / "nope")


def test_configured_harness_missing_fails_closed(tmp_path) -> None:
    init(tmp_path)
    (tmp_path / "harness" / "constraints.json").unlink()

    with pytest.raises(HarnessConstraintsError, match="not found"):
        load(tmp_path)


def test_harness_path_cannot_escape_project_build_context(tmp_path) -> None:
    init(tmp_path)
    config_path = tmp_path / CONFIG_FILENAME
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace(
            'constraints = "harness/constraints.json"',
            'constraints = "../outside.json"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(HarnessConstraintsError, match="inside the project"):
        load(tmp_path)


def test_toml_minimal_parser_round_trip(tmp_path) -> None:
    config = init(tmp_path)
    text = (tmp_path / CONFIG_FILENAME).read_text(encoding="utf-8")
    reparsed = ProjectConfig.from_toml(tmp_path.resolve(), text)
    assert reparsed.sdk_root == config.sdk_root
    assert reparsed.target_soc_model == config.target_soc_model
    assert reparsed.docker_image == config.docker_image


def test_logical_uri_round_trip_and_escape_guard(tmp_path) -> None:
    init(tmp_path)
    model = tmp_path / "models" / "qwen3" / "model.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x")

    uri = to_logical_uri(model, tmp_path)
    assert uri == "models/qwen3/model.onnx"
    assert resolve_logical_uri(uri, tmp_path) == model.resolve()

    outside = tmp_path.parent / "elsewhere.onnx"
    with pytest.raises(ValueError, match="outside the project root"):
        to_logical_uri(outside, tmp_path)

    with pytest.raises(ValueError, match="escapes the project root"):
        resolve_logical_uri("../escape.onnx", tmp_path)


def test_doctor_reports_missing_sdk_honestly(tmp_path) -> None:
    init(tmp_path)
    report = doctor(tmp_path)
    names = {check["name"]: check for check in report["checks"]}
    assert names["sdk_present"]["ok"] is False
    assert names["target"]["ok"] is True
    assert report["ok"] is False  # critical sdk check fails


def test_doctor_passes_with_prepared_sdk(tmp_path, monkeypatch) -> None:
    config = init(tmp_path)
    sdk = config.sdk_path
    sdk.mkdir(parents=True)
    (sdk / "sdk.yaml").write_text(
        "version: 2.48.0.260626\nbuild_id: 260626120635\n", encoding="utf-8"
    )
    (sdk / "lib" / "python").mkdir(parents=True)
    config.dockerfile_path.parent.mkdir(parents=True, exist_ok=True)
    config.dockerfile_path.write_text("FROM ubuntu:22.04\n", encoding="utf-8")
    monkeypatch.setattr(
        "qairt_agent.project._probe_apple_container",
        lambda image_ref, **kwargs: (
            True,
            True,
            f"image {image_ref} found",
        ),
    )

    report = doctor(tmp_path)
    names = {check["name"]: check for check in report["checks"]}
    assert names["sdk_metadata"]["ok"] is True
    assert names["qairt_capability"]["ok"] is True
    assert names["target"]["ok"] is True
    assert report["ok"] is True


def test_discovers_versioned_sdk_under_qnn_layout(tmp_path) -> None:
    installation = tmp_path / "qnn" / "qnn"
    sdk = installation / "qairt" / "2.48.0.260626"
    sdk.mkdir(parents=True)
    (sdk / "sdk.yaml").write_text(
        "version: 2.48.0\nbuild_id: 260626120635\n", encoding="utf-8"
    )

    assert discover_sdk_path(installation) == sdk.resolve()


def test_auto_backend_routes_by_host_and_native_is_explicit() -> None:
    assert (
        select_worker_backend(
            "auto", native_abi=False, system_name="Darwin"
        )
        == "apple_container"
    )
    assert (
        select_worker_backend(
            "auto", native_abi=False, system_name="Linux"
        )
        == "docker"
    )
    assert (
        select_worker_backend(
            "auto", native_abi=True, system_name="Linux"
        )
        == "docker"
    )
    assert select_worker_backend("docker", native_abi=True) == "docker"
    assert select_worker_backend("native", native_abi=False) == "native"
    assert select_container_backend(system_name="Darwin") == "apple_container"
    assert select_container_backend(system_name="Linux") == "docker"


def test_init_preserves_existing_explicit_backend(tmp_path) -> None:
    config = init(tmp_path)
    text = config.to_toml().replace('backend = "auto"', 'backend = "native"')
    (tmp_path / CONFIG_FILENAME).write_text(text, encoding="utf-8")

    assert init(tmp_path).worker_backend == "native"


@pytest.mark.parametrize(
    ("value", "expected_gb"),
    [
        ("96G", 96.0),
        ("4096M", 4.0),
        ("1T", 1024.0),
        ("32g", 32.0),
    ],
)
def test_parse_memory_gb(value: str, expected_gb: float) -> None:
    assert _parse_memory_gb(value) == pytest.approx(expected_gb)


def test_doctor_includes_host_resources_check(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "qairt_agent.project._host_memory_gb", lambda: 128.0
    )
    config = init(tmp_path)
    sdk = tmp_path / "qnn" / "qnn" / "qairt" / "2.48.0.260626"
    sdk.mkdir(parents=True)
    (sdk / "sdk.yaml").write_text(
        "version: 2.48.0\nbuild_id: 260626120635\n",
        encoding="utf-8",
    )
    (sdk / "lib" / "python").mkdir(parents=True)

    result = doctor(tmp_path)

    resource_checks = [
        c for c in result["checks"] if c["name"] == "host_resources"
    ]
    assert len(resource_checks) == 1
    assert resource_checks[0]["ok"] is True
    assert "128 GB" in resource_checks[0]["message"]
