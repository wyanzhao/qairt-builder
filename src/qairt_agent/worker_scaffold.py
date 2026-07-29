"""Materialize the self-contained worker image build context for a project.

The control-plane package may be used from an editable checkout or an installed
wheel.  A model project must not depend on the framework source repository
being its own project root, so ``qairt-agent init`` copies the pinned worker
assets and stages the exact running ``qairt_agent`` Python sources as a
deterministic zip archive.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from qairt_agent.docker.image import build_context_excludes
from qairt_agent.harness import HarnessConstraints, HarnessConstraintsError

AGENT_SOURCE_ARCHIVE = "docker/.generated/qairt-agent-src.zip"
_GENERATED_GITIGNORE = "docker/.generated/.gitignore"
_DOCKERIGNORE_BEGIN = "# qairt-agent managed exclusions: begin"
_DOCKERIGNORE_END = "# qairt-agent managed exclusions: end"


def _source_tree_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _package_data_root() -> Path:
    return Path(__file__).resolve().parent / "_data" / "worker"


def _resolve_inside(project_root: Path, logical_path: str, *, label: str) -> Path:
    resolved = (project_root / logical_path).resolve()
    if not resolved.is_relative_to(project_root):
        raise HarnessConstraintsError(
            f"{label} must resolve inside the project root: {logical_path!r}"
        )
    return resolved


def _bundled_asset(logical_path: str) -> Path | None:
    """Find one source-tree or wheel-bundled worker asset."""

    for candidate in (
        _source_tree_root() / logical_path,
        _package_data_root() / logical_path,
    ):
        if candidate.is_file():
            return candidate
    return None


def _copy_asset_if_missing(
    *,
    project_root: Path,
    logical_path: str,
    label: str,
) -> Path:
    destination = _resolve_inside(
        project_root,
        logical_path,
        label=label,
    )
    if destination.is_file():
        return destination
    source = _bundled_asset(logical_path)
    if source is None:
        raise HarnessConstraintsError(
            f"{label} is missing at {destination} and is not bundled by this "
            "qairt-agent distribution; add the reviewed file or install a "
            "distribution matching the selected harness"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def _archive_member_paths(package_root: Path) -> list[Path]:
    members: list[Path] = []
    for path in package_root.rglob("*"):
        relative = path.relative_to(package_root)
        if (
            not path.is_file()
            or path.is_symlink()
            or "__pycache__" in relative.parts
            or path.suffix in {".pyc", ".pyo"}
            or path.name == ".DS_Store"
        ):
            continue
        members.append(path)
    return sorted(
        members,
        key=lambda item: item.relative_to(package_root).as_posix(),
    )


def _write_agent_source_archive(destination: Path) -> None:
    """Write a deterministic, Python-ABI-neutral source archive atomically."""

    package_root = Path(__file__).resolve().parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".qairt-agent-src-",
        suffix=".zip",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for source in _archive_member_paths(package_root):
                relative = source.relative_to(package_root)
                info = zipfile.ZipInfo(
                    (Path("qairt_agent") / relative).as_posix(),
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, source.read_bytes())
        if (
            destination.is_file()
            and destination.read_bytes() == temporary.read_bytes()
        ):
            destination.chmod(0o644)
            return
        temporary.replace(destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)


def _merge_dockerignore(project_root: Path) -> Path:
    """Keep the managed exclusion block last so later negations cannot undo it."""

    path = _resolve_inside(
        project_root,
        ".dockerignore",
        label="worker .dockerignore",
    )
    original = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = original.splitlines()
    retained: list[str] = []
    in_managed_block = False
    for line in lines:
        if line.strip() == _DOCKERIGNORE_BEGIN:
            in_managed_block = True
            continue
        if line.strip() == _DOCKERIGNORE_END:
            in_managed_block = False
            continue
        if not in_managed_block:
            retained.append(line)
    if in_managed_block:
        raise HarnessConstraintsError(
            f"{path} contains an unmatched {_DOCKERIGNORE_BEGIN!r} marker; "
            "refusing to discard subsequent user rules"
        )
    while retained and not retained[-1].strip():
        retained.pop()
    managed = [
        _DOCKERIGNORE_BEGIN,
        *build_context_excludes(),
        _DOCKERIGNORE_END,
    ]
    content = "\n".join([*retained, *([""] if retained else []), *managed]) + "\n"
    if content != original:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def ensure_worker_build_context(
    project_root: str | Path,
    constraints: HarnessConstraints,
) -> tuple[Path, ...]:
    """Create or refresh every generated input required by ``image build``."""

    root = Path(project_root).expanduser().resolve()
    dockerfile = _copy_asset_if_missing(
        project_root=root,
        logical_path=constraints.dockerfile,
        label="worker.dockerfile",
    )
    dependencies = _copy_asset_if_missing(
        project_root=root,
        logical_path=constraints.dependencies_file,
        label="worker.dependencies_file",
    )
    archive = _resolve_inside(
        root,
        AGENT_SOURCE_ARCHIVE,
        label="qairt-agent worker source archive",
    )
    _write_agent_source_archive(archive)
    generated_gitignore = _resolve_inside(
        root,
        _GENERATED_GITIGNORE,
        label="generated worker .gitignore",
    )
    generated_gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")
    dockerignore = _merge_dockerignore(root)
    return dockerfile, dependencies, archive, dockerignore


def worker_build_context_issues(
    project_root: str | Path,
    constraints: HarnessConstraints,
) -> tuple[str, ...]:
    """Return missing or unsafe worker build-context inputs."""

    root = Path(project_root).expanduser().resolve()
    issues: list[str] = []
    for logical, label in (
        (constraints.dockerfile, "worker Dockerfile"),
        (constraints.dependencies_file, "worker dependency lock"),
        (AGENT_SOURCE_ARCHIVE, "qairt-agent worker source archive"),
    ):
        try:
            path = _resolve_inside(root, logical, label=label)
        except HarnessConstraintsError as exc:
            issues.append(str(exc))
            continue
        if not path.is_file():
            issues.append(f"{label} missing at {path}")

    try:
        dockerignore = _resolve_inside(
            root,
            ".dockerignore",
            label="worker .dockerignore",
        )
    except HarnessConstraintsError as exc:
        issues.append(str(exc))
        return tuple(issues)
    if not dockerignore.is_file():
        issues.append(f"build-context exclusions missing at {dockerignore}")
    else:
        lines = dockerignore.read_text(encoding="utf-8").splitlines()
        try:
            begin = len(lines) - 1 - lines[::-1].index(_DOCKERIGNORE_BEGIN)
            end = len(lines) - 1 - lines[::-1].index(_DOCKERIGNORE_END)
        except ValueError:
            issues.append(
                f"managed build-context exclusions missing from {dockerignore}"
            )
        else:
            managed = set(lines[begin + 1 : end])
            missing = [
                pattern
                for pattern in build_context_excludes()
                if pattern not in managed
            ]
            if end != len(lines) - 1:
                issues.append(
                    "managed build-context exclusions must be the final "
                    f".dockerignore block at {dockerignore}"
                )
            if missing:
                issues.append(
                    "managed build-context exclusions are incomplete: "
                    + ", ".join(missing)
                )
    return tuple(issues)


__all__ = [
    "AGENT_SOURCE_ARCHIVE",
    "ensure_worker_build_context",
    "worker_build_context_issues",
]
