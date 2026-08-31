"""Content-addressed artifacts and immutable manifest publication."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import UUID

from pydantic import BaseModel, ValidationError

from qairt_agent.contracts import (
    ArtifactKind,
    ArtifactRef,
    BuildSpec,
    RunManifest,
    StageRecord,
    utc_now,
)
from qairt_agent.errors import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactPublishError,
    ManifestConflictError,
    ManifestInvalidError,
)

_REVISION_THREAD_LOCKS_GUARD = threading.Lock()
_REVISION_THREAD_LOCKS: dict[Path, threading.Lock] = {}


def _revision_thread_lock(path: Path) -> threading.Lock:
    """Return one process-local lock for a revision reservation path."""

    with _REVISION_THREAD_LOCKS_GUARD:
        return _REVISION_THREAD_LOCKS.setdefault(path, threading.Lock())


@contextmanager
def _revision_publish_lock(run_directory: Path, revision: int) -> Iterator[None]:
    """Serialize one run/revision across threads and cooperating processes.

    The process-local lock is required because ``flock`` ownership semantics
    differ across supported Unix hosts.  The persistent advisory lock file
    extends the same critical section across independent worker processes.
    """

    lock_path = run_directory / f".manifest-r{revision:06d}.lock"
    with _revision_thread_lock(lock_path):
        run_directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def canonical_json_bytes(payload: BaseModel | Mapping[str, Any] | list[Any]) -> bytes:
    """Serialize a JSON payload deterministically for hashing and publication."""

    if isinstance(payload, BaseModel):
        value = payload.model_dump(mode="json")
    else:
        value = payload
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactPublishError(
            "payload is not canonically JSON serializable",
            details={"reason": str(exc)},
        ) from exc
    return (serialized + "\n").encode("utf-8")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> tuple[str, int]:
    """Return the SHA256 and byte size of a file."""

    resolved = Path(path)
    if not resolved.is_file():
        raise ArtifactNotFoundError(
            f"artifact does not exist or is not a file: {resolved}",
            details={"path": str(resolved)},
        )
    digest = hashlib.sha256()
    size_bytes = 0
    with resolved.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


#: Verifications this process has already done, keyed by what could invalidate
#: them: the exact path, its stat identity, and the digest that was expected.
#:
#: Every continuation stage re-hashes all cumulative artifacts at its
#: boundaries. On the smoke fixture that is free; on a real build it is
#: repeated full reads of multi-GB context binaries within one worker process.
#: Cross-process and cross-run behaviour is unchanged -- a cold start still
#: reads and hashes everything -- and a file whose stat moved is always
#: re-hashed in full, so this never accepts a changed artifact.
_VERIFIED: dict[tuple[str, int, int, str], None] = {}

#: Counters for the stage diagnostics, so the evidence trail shows what was
#: re-hashed versus stat-checked rather than implying everything was re-read.
_VERIFICATION_COUNTS = {"full": 0, "cached": 0}


def verification_statistics() -> dict[str, int]:
    """Full re-hashes and cache hits since the last reset, for stage records."""

    return dict(_VERIFICATION_COUNTS)


def reset_verification_cache() -> None:
    """Forget every cached verification (used by tests and worker startup)."""

    _VERIFIED.clear()
    _VERIFICATION_COUNTS.update({"full": 0, "cached": 0})


def verify_artifact(ref: ArtifactRef) -> None:
    """Raise when a referenced artifact is missing, resized, or rehashed."""

    try:
        status = os.stat(ref.path)
    except OSError:
        # Let the hashing path produce the usual missing/unreadable failure.
        status = None
    key = (
        (str(ref.path), status.st_mtime_ns, status.st_size, ref.sha256)
        if status is not None
        else None
    )
    if key is not None and key in _VERIFIED:
        _VERIFICATION_COUNTS["cached"] += 1
        return

    actual_sha256, actual_size = sha256_file(ref.path)
    _VERIFICATION_COUNTS["full"] += 1
    if actual_sha256 != ref.sha256 or actual_size != ref.size_bytes:
        raise ArtifactIntegrityError(
            f"artifact integrity check failed: {ref.path}",
            details={
                "path": str(ref.path),
                "expected_sha256": ref.sha256,
                "actual_sha256": actual_sha256,
                "expected_size_bytes": ref.size_bytes,
                "actual_size_bytes": actual_size,
            },
        )
    if key is not None:
        _VERIFIED[key] = None


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync after an atomic rename."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        directory_fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_publish_json(
    path: str | Path,
    payload: BaseModel | Mapping[str, Any] | list[Any],
    *,
    kind: ArtifactKind = ArtifactKind.OTHER,
    logical_name: str | None = None,
    overwrite: bool = False,
) -> ArtifactRef:
    """Publish canonical JSON using a flushed same-directory temporary file.

    Existing identical content is idempotent.  Existing different content is
    rejected unless ``overwrite=True``; manifest publication never enables it.
    """

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload)
    expected_sha256 = hashlib.sha256(data).hexdigest()

    if destination.exists():
        existing_sha256, existing_size = sha256_file(destination)
        if existing_sha256 == expected_sha256 and existing_size == len(data):
            return ArtifactRef(
                path=destination,
                sha256=existing_sha256,
                size_bytes=existing_size,
                kind=kind,
                media_type="application/json",
                logical_name=logical_name,
            )
        if not overwrite:
            raise ArtifactPublishError(
                f"refusing to overwrite published artifact: {destination}",
                details={"path": str(destination)},
            )

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except OSError as exc:
        raise ArtifactPublishError(
            f"failed to publish JSON artifact: {destination}",
            details={"path": str(destination), "reason": str(exc)},
        ) from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return ArtifactRef(
        path=destination,
        sha256=expected_sha256,
        size_bytes=len(data),
        kind=kind,
        media_type="application/json",
        logical_name=logical_name,
    )


class ManifestStore:
    """Filesystem store with one immutable JSON file per manifest revision."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _run_directory(self, run_id: UUID) -> Path:
        return self.root / str(run_id)

    @staticmethod
    def _revision_glob(revision: int) -> str:
        return f"manifest-r{revision:06d}-*.json"

    def publish(self, manifest: RunManifest) -> ArtifactRef:
        """Atomically publish a manifest without changing an existing revision."""

        data = canonical_json_bytes(manifest)
        digest = hashlib.sha256(data).hexdigest()
        run_directory = self._run_directory(manifest.run_id)
        destination = run_directory / f"manifest-r{manifest.revision:06d}-{digest}.json"

        with _revision_publish_lock(run_directory, manifest.revision):
            if manifest.parent_manifest is not None:
                parent = self.load(manifest.parent_manifest)
                if parent.run_id != manifest.run_id or parent.revision != manifest.revision - 1:
                    raise ManifestConflictError(
                        "manifest parent does not identify the immediately preceding revision",
                        details={
                            "run_id": str(manifest.run_id),
                            "revision": manifest.revision,
                            "parent_run_id": str(parent.run_id),
                            "parent_revision": parent.revision,
                        },
                    )

            existing_revisions = tuple(run_directory.glob(self._revision_glob(manifest.revision)))
            conflicting = [path for path in existing_revisions if path.resolve() != destination.resolve()]
            if conflicting:
                raise ManifestConflictError(
                    f"run {manifest.run_id} revision {manifest.revision} is already published",
                    details={
                        "run_id": str(manifest.run_id),
                        "revision": manifest.revision,
                        "existing": [str(path) for path in conflicting],
                    },
                )

            return atomic_publish_json(
                destination,
                manifest,
                kind=ArtifactKind.MANIFEST,
                logical_name=f"run-manifest-r{manifest.revision}",
            )

    def create(
        self,
        build_spec: BuildSpec,
        *,
        run_id: UUID | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[RunManifest, ArtifactRef]:
        """Create and publish revision zero."""

        manifest = RunManifest(
            run_id=run_id or UUID(bytes=os.urandom(16), version=4),
            build_spec=build_spec,
            metadata=dict(metadata or {}),
        )
        return manifest, self.publish(manifest)

    def _coerce_manifest_ref(
        self,
        ref_or_path: ArtifactRef | str | Path,
        expected_sha256: str | None,
    ) -> ArtifactRef:
        if isinstance(ref_or_path, ArtifactRef):
            if expected_sha256 is not None and expected_sha256.lower() != ref_or_path.sha256:
                raise ArtifactIntegrityError(
                    "provided manifest hash does not match ArtifactRef",
                    details={
                        "artifact_ref_sha256": ref_or_path.sha256,
                        "provided_sha256": expected_sha256.lower(),
                    },
                )
            return ref_or_path

        if expected_sha256 is None:
            raise ArtifactIntegrityError(
                "expected_sha256 is required when loading a manifest path",
                details={"path": str(ref_or_path)},
            )
        path = Path(ref_or_path).expanduser().resolve()
        if not path.is_file():
            raise ArtifactNotFoundError(
                f"manifest does not exist: {path}",
                details={"path": str(path)},
            )
        return ArtifactRef(
            path=path,
            sha256=expected_sha256,
            size_bytes=path.stat().st_size,
            kind=ArtifactKind.MANIFEST,
            media_type="application/json",
        )

    def load(
        self,
        ref_or_path: ArtifactRef | str | Path,
        expected_sha256: str | None = None,
    ) -> RunManifest:
        """Verify a manifest hash before parsing its contents."""

        ref = self._coerce_manifest_ref(ref_or_path, expected_sha256)
        if ref.kind != ArtifactKind.MANIFEST:
            raise ManifestInvalidError(
                "artifact is not declared as a manifest",
                details={"path": str(ref.path), "kind": ref.kind.value},
            )
        verify_artifact(ref)
        try:
            return RunManifest.model_validate_json(ref.path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise ManifestInvalidError(
                f"invalid run manifest: {ref.path}",
                details={"path": str(ref.path), "reason": str(exc)},
            ) from exc

    def verify(
        self,
        ref_or_path: ArtifactRef | str | Path,
        expected_sha256: str | None = None,
    ) -> ArtifactRef:
        """Verify and return a normalized manifest reference."""

        ref = self._coerce_manifest_ref(ref_or_path, expected_sha256)
        self.load(ref)
        return ref

    @staticmethod
    def _merge_artifacts(
        current: tuple[ArtifactRef, ...],
        additions: tuple[ArtifactRef, ...],
    ) -> tuple[ArtifactRef, ...]:
        merged = list(current)
        for addition in additions:
            replacement_index = None
            for index, existing in enumerate(merged):
                same_logical_name = (
                    addition.logical_name is not None
                    and addition.logical_name == existing.logical_name
                )
                same_path = addition.path == existing.path
                if same_logical_name or same_path:
                    replacement_index = index
                    break
            if replacement_index is None:
                merged.append(addition)
            else:
                merged[replacement_index] = addition
        return tuple(merged)

    def revise(
        self,
        previous: ArtifactRef | str | Path,
        *,
        expected_sha256: str | None = None,
        stage: StageRecord | None = None,
        artifacts: tuple[ArtifactRef, ...] = (),
        metadata_update: Mapping[str, Any] | None = None,
    ) -> tuple[RunManifest, ArtifactRef]:
        """Append state and atomically publish the next immutable revision."""

        previous_ref = self._coerce_manifest_ref(previous, expected_sha256)
        current = self.load(previous_ref)
        metadata = {**current.metadata, **dict(metadata_update or {})}
        stages = current.stages + ((stage,) if stage is not None else ())
        revised = RunManifest(
            run_id=current.run_id,
            revision=current.revision + 1,
            created_at=current.created_at,
            revision_created_at=utc_now(),
            parent_manifest=previous_ref,
            build_spec=current.build_spec,
            stages=stages,
            artifacts=self._merge_artifacts(current.artifacts, artifacts),
            metadata=metadata,
        )
        return revised, self.publish(revised)

    def fork_snapshot(
        self,
        previous: ArtifactRef | str | Path,
        *,
        expected_sha256: str | None = None,
        run_id: UUID | None = None,
        replacement_build_spec: BuildSpec | None = None,
        metadata_update: Mapping[str, Any] | None = None,
    ) -> tuple[RunManifest, ArtifactRef]:
        """Publish a content-verified revision-zero snapshot under a new run.

        A fork is not a manifest parent edge: the source run remains a strictly
        linear immutable chain. Provenance is recorded in metadata while the
        source snapshot's cumulative stages and artifacts are copied verbatim.
        A caller may supply a separately validated effective BuildSpec for the
        new run; the source and effective spec hashes then remain explicit in
        metadata.
        """

        previous_ref = self._coerce_manifest_ref(previous, expected_sha256)
        source = self.load(previous_ref)
        expected_directory = self._run_directory(source.run_id)
        if previous_ref.path.expanduser().resolve().parent != expected_directory:
            raise ManifestConflictError(
                "manifest snapshot is outside its declared run directory",
                details={
                    "manifest": str(previous_ref.path),
                    "run_id": str(source.run_id),
                    "expected_directory": str(expected_directory),
                },
            )
        # Verify every copied reference before minting an independent snapshot.
        verify_manifest_graph(previous_ref)

        new_run_id = run_id or UUID(bytes=os.urandom(16), version=4)
        if new_run_id == source.run_id:
            raise ManifestConflictError(
                "manifest fork requires a new run_id; same-run branching is forbidden",
                details={"run_id": str(source.run_id)},
            )
        effective_build_spec = replacement_build_spec or source.build_spec
        source_build_spec_sha256 = hashlib.sha256(
            canonical_json_bytes(source.build_spec)
        ).hexdigest()
        effective_build_spec_sha256 = hashlib.sha256(
            canonical_json_bytes(effective_build_spec)
        ).hexdigest()
        metadata = {
            **source.metadata,
            **dict(metadata_update or {}),
            "forked_from_manifest": previous_ref.model_dump(mode="json"),
            "forked_from_run_id": str(source.run_id),
            "forked_from_revision": source.revision,
            "forked_from_build_spec_sha256": source_build_spec_sha256,
            "effective_build_spec_sha256": effective_build_spec_sha256,
            "build_spec_rebased": effective_build_spec != source.build_spec,
        }
        snapshot = RunManifest(
            run_id=new_run_id,
            revision=0,
            build_spec=effective_build_spec,
            stages=source.stages,
            artifacts=source.artifacts,
            metadata=metadata,
        )
        return snapshot, self.publish(snapshot)


def verify_manifest_graph(ref: ArtifactRef) -> tuple[ArtifactRef, ...]:
    """Verify a manifest and every artifact reachable from its revision graph.

    The returned references exclude ``ref`` itself and are de-duplicated in
    deterministic discovery order. Parent manifests are included, together
    with every manifest-level artifact and every stage input/output.
    """

    discovered: list[ArtifactRef] = []
    seen: set[tuple[Path, str]] = set()

    def visit_artifact(artifact: ArtifactRef) -> None:
        key = (artifact.path.expanduser().resolve(), artifact.sha256)
        if key in seen:
            return
        seen.add(key)
        verify_artifact(artifact)
        discovered.append(artifact)
        if artifact.kind is not ArtifactKind.MANIFEST:
            return
        try:
            manifest = RunManifest.model_validate_json(artifact.path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise ManifestInvalidError(
                f"invalid run manifest: {artifact.path}",
                details={"path": str(artifact.path), "reason": str(exc)},
            ) from exc
        if manifest.parent_manifest is not None:
            visit_artifact(manifest.parent_manifest)
        for nested in manifest.artifacts:
            visit_artifact(nested)
        for stage in manifest.stages:
            for nested in (*stage.inputs, *stage.outputs):
                visit_artifact(nested)

    visit_artifact(ref)
    return tuple(item for item in discovered if item != ref)


__all__ = [
    "ManifestStore",
    "atomic_publish_json",
    "canonical_json_bytes",
    "reset_verification_cache",
    "sha256_file",
    "verification_statistics",
    "verify_artifact",
    "verify_manifest_graph",
]
