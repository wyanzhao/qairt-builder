from __future__ import annotations

import sys
from dataclasses import replace

import pytest

from qairt_agent.docker import smoke
from qairt_agent.harness import DEFAULT_CONSTRAINTS


def test_dependency_smoke_reads_lock_and_harness_torch_pin(tmp_path) -> None:
    lock = tmp_path / "requirements.txt"
    lock.write_text(
        "numpy==2.1.0\ntransformers==5.0.0\n",
        encoding="utf-8",
    )
    constraints = replace(
        DEFAULT_CONSTRAINTS,
        torch_version="2.7.0",
    )

    expected = smoke._locked_versions(lock, constraints)

    assert expected == {
        "numpy": "2.1.0",
        "torch": "2.7.0",
        "transformers": "5.0.0",
    }


def test_dependency_smoke_rejects_unlocked_requirement(tmp_path) -> None:
    lock = tmp_path / "requirements.txt"
    lock.write_text("numpy>=2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="name==version"):
        smoke._locked_versions(lock, DEFAULT_CONSTRAINTS)


def test_smoke_rejects_sdk_metadata_before_importing_qairt(
    tmp_path,
    monkeypatch,
) -> None:
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "sdk.yaml").write_text(
        "version: 9.9.9\nbuild_id: wrong\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(smoke, "_SDK_ROOT", sdk)
    monkeypatch.setattr(
        smoke,
        "load_harness_constraints",
        lambda: replace(
            DEFAULT_CONSTRAINTS,
            python_version=(
                f"{sys.version_info.major}.{sys.version_info.minor}"
            ),
        ),
    )
    monkeypatch.setattr(
        smoke,
        "_check_sdk_dependencies",
        lambda root: (_ for _ in ()).throw(
            AssertionError("dependency checks must not run for wrong SDK")
        ),
    )

    with pytest.raises(RuntimeError, match="does not match harness"):
        smoke.main()
