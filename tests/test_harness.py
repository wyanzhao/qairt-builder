from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from qairt_agent.agent import QairtAgentClient
from qairt_agent.cli import _default_client
from qairt_agent.contracts import TargetSpec
from qairt_agent.harness import (
    DEFAULT_CONSTRAINTS,
    HarnessConstraintsError,
    load_harness_constraints,
    parse_version,
)
from qairt_agent.pipeline import QairtAgent
from qairt_agent.project import init, load


def test_checked_in_harness_has_worker_and_runtime_pins() -> None:
    constraints = DEFAULT_CONSTRAINTS

    assert constraints.qairt_version == "2.48.0"
    assert constraints.qairt_build_id == "260626120635"
    assert constraints.ubuntu_version == "22.04"
    assert constraints.python_version_tuple == (3, 10)
    assert constraints.platform == "linux/amd64"
    assert constraints.apple_container_version == "1.0.0"
    assert constraints.docker_minimum_version == "24.0.0"
    assert constraints.dependencies_file.endswith(".txt")


def test_project_load_consumes_updated_harness_values(tmp_path) -> None:
    init(tmp_path)
    path = tmp_path / "harness" / "constraints.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["worker"]["image"] = "worker:next"
    document["worker"]["dockerfile"] = "images/next.Dockerfile"
    path.write_text(json.dumps(document), encoding="utf-8")

    config = load(tmp_path)

    assert config.docker_image == "worker:next"
    assert config.dockerfile == "images/next.Dockerfile"
    assert config.harness.worker_image == "worker:next"


def test_active_harness_drives_contract_stage_key_and_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("QAIRT_AGENT_HARNESS_CONSTRAINTS", raising=False)
    baseline_key = QairtAgent._stage_key("build", "a" * 64, {})

    document = json.loads(
        DEFAULT_CONSTRAINTS.source_path.read_text(encoding="utf-8")
    )
    document["qairt"]["version"] = "2.49.0"
    document["qairt"]["build_id"] = "next-build"
    document["worker"]["ubuntu_version"] = "24.04"
    document["worker"]["platform"] = "linux/amd64"
    path = tmp_path / "constraints.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setenv("QAIRT_AGENT_HARNESS_CONSTRAINTS", str(path))

    target = TargetSpec()
    assert target.qairt_version == "2.49.0"
    assert target.qairt_build_id == "next-build"
    with pytest.raises(ValidationError, match="must match harness"):
        TargetSpec(qairt_version="2.48.0")

    assert QairtAgent._stage_key("build", "a" * 64, {}) != baseline_key
    provenance = QairtAgentClient(
        jobs_root=tmp_path / "jobs",
        background=False,
    )._provenance_for(  # noqa: SLF001 - compatibility-contract assertion
        SimpleNamespace(resolved_preset_sha256="b" * 64)
    )
    assert provenance.sdk_build == "next-build"
    assert provenance.platform_abi == "ubuntu24.04-amd64"


def test_project_bound_client_uses_custom_harness_without_global_env(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("QAIRT_AGENT_HARNESS_CONSTRAINTS", raising=False)
    init(tmp_path)
    path = tmp_path / "harness" / "constraints.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["qairt"]["version"] = "2.49.0"
    document["qairt"]["build_id"] = "project-build"
    path.write_text(json.dumps(document), encoding="utf-8")
    nested = tmp_path / "src" / "nested"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    client = _default_client(None)
    workflow = client._normalize_spec(  # noqa: SLF001 - binding assertion
        {
            "family": "qwen3",
            "sources": {
                "text": {
                    "onnx_path": "/models/model.onnx",
                    "encodings_path": "/models/model.encodings",
                }
            },
            "output_root": "/artifacts/out",
        }
    )
    provenance = client._provenance_for(  # noqa: SLF001
        SimpleNamespace(resolved_preset_sha256="c" * 64)
    )

    assert workflow.target.qairt_version == "2.49.0"
    assert workflow.target.qairt_build_id == "project-build"
    assert provenance.sdk_build == "project-build"
    assert "QAIRT_AGENT_HARNESS_CONSTRAINTS" not in os.environ


def test_invalid_harness_fails_closed(tmp_path) -> None:
    path = tmp_path / "constraints.json"
    path.write_text('{"schema_version": 99}', encoding="utf-8")

    with pytest.raises(
        HarnessConstraintsError,
        match="schema_version",
    ):
        load_harness_constraints(path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("container CLI version 1.0.0 (release)", (1, 0, 0)),
        ("Docker version 26.1.4", (26, 1, 4)),
        ("unknown", None),
    ],
)
def test_parse_version(text, expected) -> None:
    assert parse_version(text) == expected


def test_checked_in_harness_has_worker_resource_pins() -> None:
    constraints = DEFAULT_CONSTRAINTS

    assert constraints.worker_memory == "96G"
    assert constraints.worker_cpus == 8


def test_harness_resource_defaults_for_legacy_constraints(tmp_path) -> None:
    path = tmp_path / "constraints.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "qairt": {"version": "2.48.0", "build_id": "260626120635"},
                "worker": {
                    "ubuntu_version": "22.04",
                    "python_version": "3.10",
                    "platform": "linux/amd64",
                    "image": "w:1",
                    "dockerfile": "d.Dockerfile",
                    "dependencies_file": "r.txt",
                    "torch_version": "2.4.1",
                    "torch_index_url": "https://example.com",
                },
                "runtime_cli": {
                    "apple_container": {
                        "version": "1.0.0",
                        "host_alias": "h.ci",
                    },
                    "docker": {
                        "minimum_version": "24.0.0",
                        "host_alias": "h.di",
                    },
                },
                "target": {
                    "chipset": "SM8850",
                    "dsp_arch": "v81",
                    "soc_model": 660,
                },
            }
        ),
        encoding="utf-8",
    )

    constraints = load_harness_constraints(path)

    assert constraints.worker_memory == "96G"
    assert constraints.worker_cpus == 8
