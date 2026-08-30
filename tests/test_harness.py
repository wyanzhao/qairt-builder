from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from qairt_agent.agent import QairtAgentClient
from qairt_agent.cli import _default_client
from qairt_agent.contracts import TargetSpec
from qairt_agent.harness import (
    DEFAULT_CONSTRAINTS,
    ENV_TARGET_ACCEPTANCE,
    HarnessConstraintsError,
    default_targets_dir,
    load_harness_constraints,
    load_target_registry,
    parse_version,
    require_verified_target,
    resolve_target,
    resolve_target_tuple,
)
from qairt_agent.pipeline import QairtAgent
from qairt_agent.project import init, load


def test_checked_in_harness_has_worker_and_runtime_pins() -> None:
    constraints = DEFAULT_CONSTRAINTS

    assert constraints.qairt_version == "2.49.0"
    assert constraints.qairt_build_id == "260730134355"
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
    document["qairt"]["version"] = "2.50.0"
    document["qairt"]["build_id"] = "next-build"
    document["worker"]["ubuntu_version"] = "24.04"
    document["worker"]["platform"] = "linux/amd64"
    path = tmp_path / "constraints.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    # A constraints file names a target by reference, so its registry travels
    # with it; a project that copies one without the other cannot resolve.
    registry = tmp_path / "targets"
    registry.mkdir()
    for entry in sorted(default_targets_dir().glob("*.json")):
        shutil.copyfile(entry, registry / entry.name)
    monkeypatch.setenv("QAIRT_AGENT_HARNESS_CONSTRAINTS", str(path))

    target = TargetSpec()
    assert target.qairt_version == "2.50.0"
    assert target.qairt_build_id == "next-build"
    with pytest.raises(ValidationError, match="must match harness"):
        TargetSpec(qairt_version="2.49.0")

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


def test_target_registry_separates_the_two_soc_numbering_schemes() -> None:
    registry = load_target_registry(DEFAULT_CONSTRAINTS)

    assert set(registry) == {"sm8750", "sm8850"}
    # soc_model is Qnn_SocModel_t; soc_id is the Android SoC ID. Keeping both
    # in the entry is what stops them being conflated again.
    assert (registry["sm8850"].soc_model, registry["sm8850"].soc_id) == (87, (660,))
    assert (registry["sm8750"].soc_model, registry["sm8750"].soc_id) == (69, (618, 639))
    assert registry["sm8850"].dsp_arch == "v81"
    assert registry["sm8750"].dsp_arch == "v79"


def test_unregistered_target_and_unregistered_tuple_fail_closed() -> None:
    with pytest.raises(HarnessConstraintsError, match="unregistered target"):
        resolve_target("sm9999")

    # The old pin used SM8850's Android SoC ID where the QNN SoC model belongs.
    with pytest.raises(HarnessConstraintsError, match="not a reviewed target"):
        resolve_target_tuple("SM8850", "v81", 660)

    assert resolve_target_tuple("SM8850", "v81", 87).name == "sm8850"


def test_unverified_target_is_refused_until_an_acceptance_run(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(ENV_TARGET_ACCEPTANCE, raising=False)
    entry = resolve_target("sm8850")
    unverified = replace(entry, verified=None)

    with pytest.raises(HarnessConstraintsError, match="no verified block"):
        require_verified_target(unverified)

    # A target cannot be verified without a run and a run is refused while it
    # is unverified, so the qualifying run names the target explicitly.
    monkeypatch.setenv(ENV_TARGET_ACCEPTANCE, "sm8850")
    assert require_verified_target(unverified) is unverified

    # Naming a different target does not qualify this one.
    monkeypatch.setenv(ENV_TARGET_ACCEPTANCE, "sm8750")
    with pytest.raises(HarnessConstraintsError, match="no verified block"):
        require_verified_target(unverified)


def test_both_seeded_targets_record_a_real_device_acceptance_run() -> None:
    for entry in load_target_registry(DEFAULT_CONSTRAINTS).values():
        assert entry.verified is not None, entry.name
        assert entry.verified["sdk_build"] == DEFAULT_CONSTRAINTS.qairt_build_id
        assert entry.verified["device"], entry.name
        assert entry.verified["how"], entry.name
