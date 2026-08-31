"""The wheel must carry the data files the harness names.

Two undocumented steps used to break here. Renaming the dependency lock (step 2
of an SDK upgrade) or adding a target left the packaging config pointing at a
filename that no longer existed, and the wheel shipped silently missing it --
a failure that only surfaces on an installed copy, far from the change that
caused it.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _force_include() -> dict[str, str]:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return payload["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]


def _covered(relative: Path, mapping: dict[str, str]) -> bool:
    """Whether a repo-relative path is shipped, directly or via a directory."""

    candidates = {Path(key) for key in mapping}
    return any(
        relative == candidate or candidate in relative.parents
        for candidate in candidates
    )


def test_every_reviewed_target_is_packaged() -> None:
    mapping = _force_include()
    targets = sorted((REPO_ROOT / "harness" / "targets").glob("*.json"))

    assert targets, "the reviewed target registry is empty"
    for target in targets:
        relative = target.relative_to(REPO_ROOT)
        assert _covered(relative, mapping), (
            f"{relative} is not shipped in the wheel; add it to "
            "[tool.hatch.build.targets.wheel.force-include] in pyproject.toml"
        )


def test_the_dependency_lock_the_constraints_name_is_packaged() -> None:
    constraints = json.loads(
        (REPO_ROOT / "harness" / "constraints.json").read_text(encoding="utf-8")
    )
    lock = Path(constraints["worker"]["dependencies_file"])

    assert (REPO_ROOT / lock).is_file(), (
        f"harness/constraints.json names {lock}, which does not exist"
    )
    assert _covered(lock, _force_include()), (
        f"harness/constraints.json names {lock}, but pyproject.toml's "
        "[tool.hatch.build.targets.wheel.force-include] does not ship it; "
        "renaming the lock during an SDK upgrade must update pyproject.toml too"
    )


def test_the_harness_constraints_itself_is_packaged() -> None:
    assert _covered(Path("harness/constraints.json"), _force_include())
