from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from qairt_agent.harness import HarnessConstraintsError
from qairt_agent.project import init, load
from qairt_agent.worker_scaffold import (
    AGENT_SOURCE_ARCHIVE,
    ensure_worker_build_context,
    worker_build_context_issues,
)


def test_fresh_init_materializes_self_contained_worker_context(tmp_path) -> None:
    config = init(tmp_path)
    constraints = config.harness

    assert config.dockerfile_path.is_file()
    assert (tmp_path / constraints.dependencies_file).is_file()
    assert (tmp_path / AGENT_SOURCE_ARCHIVE).is_file()
    assert stat.S_IMODE((tmp_path / AGENT_SOURCE_ARCHIVE).stat().st_mode) == 0o644
    assert (tmp_path / ".dockerignore").is_file()
    assert worker_build_context_issues(tmp_path, constraints) == ()

    with zipfile.ZipFile(tmp_path / AGENT_SOURCE_ARCHIVE) as archive:
        members = archive.namelist()
    assert "qairt_agent/cli.py" in members
    assert "qairt_agent/docker/smoke.py" in members
    assert not any(member.endswith((".pyc", ".pyo")) for member in members)

    dockerfile = config.dockerfile_path.read_text(encoding="utf-8")
    assert "COPY docker/.generated/qairt-agent-src.zip" in dockerfile
    assert "COPY pyproject.toml README.md" not in dockerfile
    assert "COPY src " not in dockerfile


def test_generated_archive_is_importable_without_checkout_on_pythonpath(
    tmp_path,
) -> None:
    config = init(tmp_path)
    archive = tmp_path / AGENT_SOURCE_ARCHIVE
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(archive)
    environment["QAIRT_AGENT_HARNESS_CONSTRAINTS"] = str(config.harness_path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import qairt_agent; print(qairt_agent.__file__)",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"{archive}/qairt_agent/__init__.py" in result.stdout.strip()


def test_init_keeps_managed_dockerignore_last_and_excludes_large_inputs(
    tmp_path,
) -> None:
    (tmp_path / ".dockerignore").write_text(
        "# user rule\n!qnn/**\n!models/**\n",
        encoding="utf-8",
    )

    init(tmp_path)

    lines = (tmp_path / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert lines[:3] == ["# user rule", "!qnn/**", "!models/**"]
    assert lines[-1] == "# qairt-agent managed exclusions: end"
    for required in (
        "qnn/",
        "models/",
        "artifacts/",
        ".qairt-agent/",
        "*.onnx",
        "*.pkl",
        "*.pickle",
    ):
        assert lines.index(required) > lines.index("!models/**")


def test_unmatched_managed_dockerignore_marker_fails_without_data_loss(
    tmp_path,
) -> None:
    original = (
        "# user rule\n"
        "# qairt-agent managed exclusions: begin\n"
        "!keep-this-user-rule\n"
    )
    path = tmp_path / ".dockerignore"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(HarnessConstraintsError, match="unmatched"):
        init(tmp_path)

    assert path.read_text(encoding="utf-8") == original


def test_init_source_archive_is_deterministic_and_refreshable(tmp_path) -> None:
    config = init(tmp_path)
    archive = tmp_path / AGENT_SOURCE_ARCHIVE
    first = archive.read_bytes()

    ensure_worker_build_context(tmp_path, config.harness)

    assert archive.read_bytes() == first


def test_updated_harness_dependency_lock_fails_closed_until_supplied(
    tmp_path,
) -> None:
    init(tmp_path)
    harness_path = tmp_path / "harness" / "constraints.json"
    document = json.loads(harness_path.read_text(encoding="utf-8"))
    document["worker"]["dependencies_file"] = "docker/requirements-next.txt"
    harness_path.write_text(json.dumps(document), encoding="utf-8")
    constraints = load(tmp_path).harness

    with pytest.raises(
        HarnessConstraintsError,
        match="is not bundled by this qairt-agent distribution",
    ):
        ensure_worker_build_context(tmp_path, constraints)

    custom_lock = tmp_path / constraints.dependencies_file
    custom_lock.write_text("numpy==1.26.4\n", encoding="utf-8")
    ensure_worker_build_context(tmp_path, constraints)
    assert custom_lock.read_text(encoding="utf-8") == "numpy==1.26.4\n"


def test_wheel_declares_every_non_python_init_asset() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert (
        '"docker/worker.Dockerfile" = '
        '"qairt_agent/_data/worker/docker/worker.Dockerfile"'
    ) in pyproject
    assert (
        '"docker/requirements-qairt-2.48.0.260626.txt" = '
        '"qairt_agent/_data/worker/docker/'
        'requirements-qairt-2.48.0.260626.txt"'
    ) in pyproject


def test_context_audit_detects_missing_archive_and_late_ignore_negation(
    tmp_path,
) -> None:
    config = init(tmp_path)
    (tmp_path / AGENT_SOURCE_ARCHIVE).unlink()
    with (tmp_path / ".dockerignore").open("a", encoding="utf-8") as stream:
        stream.write("!qnn/**\n")

    issues = worker_build_context_issues(tmp_path, config.harness)

    assert any("source archive missing" in issue for issue in issues)
    assert any("must be the final" in issue for issue in issues)
