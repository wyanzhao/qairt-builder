from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

import qairt_agent.artifacts as artifacts_module
from qairt_agent.artifacts import ManifestStore, atomic_publish_json, verify_artifact
from qairt_agent.contracts import (
    ArtifactKind,
    ArtifactRef,
    BuildSpec,
    ModelFamily,
    ModelSourceSpec,
    RunManifest,
    StageRecord,
    StageStatus,
)
from qairt_agent.errors import (
    ArtifactIntegrityError,
    ArtifactPublishError,
    ManifestConflictError,
)


def make_spec() -> BuildSpec:
    return BuildSpec(
        name="qwen-smoke",
        family=ModelFamily.QWEN3_DENSE,
        sources={
            "text": ModelSourceSpec(
                onnx_path="/models/model.onnx",
                encodings_path="/models/model.encodings",
            )
        },
        output_root="/artifacts/qwen-smoke",
    )


def test_artifact_ref_hashes_file_and_detects_tampering(tmp_path) -> None:
    artifact_path = tmp_path / "model.bin"
    artifact_path.write_bytes(b"context-binary")
    ref = ArtifactRef.from_path(
        artifact_path,
        kind=ArtifactKind.CONTEXT_BINARY,
        logical_name="decoder-0",
    )

    verify_artifact(ref)
    assert ref.size_bytes == len(b"context-binary")
    assert len(ref.sha256) == 64

    artifact_path.write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="integrity"):
        verify_artifact(ref)


def test_verify_artifact_bypasses_metadata_keyed_hash_cache(tmp_path) -> None:
    artifact_path = tmp_path / "model.bin"
    artifact_path.write_bytes(b"original")
    ref = ArtifactRef.from_path(artifact_path)
    stat = artifact_path.stat()

    # Warm the performance cache, then replace the bytes while deliberately
    # preserving every cache-key field (resolved path, size, and mtime).
    artifacts_module.sha256_file(artifact_path)
    artifact_path.write_bytes(b"tampered")
    os.utime(
        artifact_path,
        ns=(stat.st_atime_ns, stat.st_mtime_ns),
    )

    with pytest.raises(ArtifactIntegrityError, match="integrity"):
        verify_artifact(ref)


def test_atomic_json_publish_is_canonical_and_idempotent(tmp_path) -> None:
    output = tmp_path / "payload.json"
    first = atomic_publish_json(output, {"z": 1, "a": [2, 3]})
    second = atomic_publish_json(output, {"a": [2, 3], "z": 1})

    assert first.sha256 == second.sha256
    assert output.read_text() == '{"a":[2,3],"z":1}\n'
    assert not tuple(tmp_path.glob("*.tmp"))

    with pytest.raises(ArtifactPublishError, match="overwrite"):
        atomic_publish_json(output, {"different": True})


def test_manifest_store_creates_loads_and_revises_immutable_manifests(tmp_path) -> None:
    store = ManifestStore(tmp_path / "manifests")
    initial, initial_ref = store.create(make_spec(), metadata={"owner": "test"})

    loaded = store.load(initial_ref.path, initial_ref.sha256)
    assert loaded == initial
    assert initial.revision == 0
    assert initial_ref.kind == ArtifactKind.MANIFEST

    stage = StageRecord(
        name="inspect",
        status=StageStatus.SUCCEEDED,
        completed_at=datetime.now(timezone.utc),
        metrics={"onnx_nodes": 42},
    )
    revised, revised_ref = store.revise(
        initial_ref,
        stage=stage,
        metadata_update={"worker": "ubuntu22"},
    )

    assert revised.revision == 1
    assert revised.parent_manifest == initial_ref
    assert revised.stages == (stage,)
    assert revised.metadata == {"owner": "test", "worker": "ubuntu22"}
    assert revised_ref.path != initial_ref.path
    assert store.load(initial_ref) == initial
    assert store.load(revised_ref) == revised


def test_manifest_store_forks_verified_snapshot_without_branching_source(
    tmp_path,
) -> None:
    store = ManifestStore(tmp_path / "manifests")
    initial, initial_ref = store.create(make_spec(), metadata={"owner": "test"})
    artifact_path = tmp_path / "context.bin"
    artifact_path.write_bytes(b"context")
    artifact = ArtifactRef.from_path(
        artifact_path,
        kind=ArtifactKind.CONTEXT_BINARY,
        logical_name="decoder",
    )
    stage = StageRecord(
        name="validate",
        status=StageStatus.SUCCEEDED,
        completed_at=datetime.now(timezone.utc),
        outputs=(artifact,),
    )
    source, source_ref = store.revise(
        initial_ref,
        stage=stage,
        artifacts=(artifact,),
    )

    snapshot, snapshot_ref = store.fork_snapshot(
        source_ref,
        metadata_update={"forked_for_job_id": "rerun-1"},
    )

    assert snapshot.run_id != source.run_id
    assert snapshot.revision == 0
    assert snapshot.parent_manifest is None
    assert snapshot.build_spec == source.build_spec
    assert snapshot.stages == source.stages
    assert snapshot.artifacts == source.artifacts
    assert snapshot.metadata["owner"] == "test"
    assert snapshot.metadata["forked_for_job_id"] == "rerun-1"
    assert snapshot.metadata["forked_from_run_id"] == str(source.run_id)
    assert snapshot.metadata["forked_from_revision"] == source.revision
    assert (
        snapshot.metadata["forked_from_manifest"]["sha256"]
        == source_ref.sha256
    )
    assert store.load(snapshot_ref) == snapshot
    assert len(tuple(initial_ref.path.parent.glob("manifest-r000001-*.json"))) == 1

    with pytest.raises(ManifestConflictError, match="same-run branching"):
        store.fork_snapshot(source_ref, run_id=source.run_id)


def test_manifest_store_fork_can_rebase_effective_build_spec_with_hashes(
    tmp_path,
) -> None:
    store = ManifestStore(tmp_path / "manifests")
    source, source_ref = store.create(make_spec())
    replacement = source.build_spec.model_copy(
        update={
            "benchmark": source.build_spec.benchmark.model_copy(
                update={"measured_runs": 99}
            )
        }
    )

    snapshot, snapshot_ref = store.fork_snapshot(
        source_ref,
        replacement_build_spec=replacement,
    )

    assert snapshot.build_spec == replacement
    assert snapshot.metadata["build_spec_rebased"] is True
    assert snapshot.metadata["forked_from_build_spec_sha256"] != (
        snapshot.metadata["effective_build_spec_sha256"]
    )
    assert store.load(source_ref).build_spec == source.build_spec
    assert store.load(snapshot_ref) == snapshot


def test_manifest_hash_is_verified_before_json_is_parsed(tmp_path) -> None:
    store = ManifestStore(tmp_path)
    _, ref = store.create(make_spec())
    payload = json.loads(ref.path.read_text())
    payload["revision"] = 999
    ref.path.write_text(json.dumps(payload))

    with pytest.raises(ArtifactIntegrityError, match="integrity"):
        store.load(ref)


def test_loading_a_path_requires_the_callers_manifest_hash(tmp_path) -> None:
    store = ManifestStore(tmp_path)
    _, ref = store.create(make_spec())

    with pytest.raises(ArtifactIntegrityError, match="expected_sha256"):
        store.load(ref.path)


def test_store_rejects_two_different_payloads_for_same_revision(tmp_path) -> None:
    store = ManifestStore(tmp_path)
    initial, _ = store.create(make_spec())
    conflicting = RunManifest(
        run_id=initial.run_id,
        build_spec=initial.build_spec,
        metadata={"different": True},
    )

    with pytest.raises(ManifestConflictError, match="already published"):
        store.publish(conflicting)


def test_store_rejects_a_parent_from_the_wrong_revision(tmp_path) -> None:
    store = ManifestStore(tmp_path)
    initial, initial_ref = store.create(make_spec())
    invalid = RunManifest(
        run_id=initial.run_id,
        revision=2,
        parent_manifest=initial_ref,
        build_spec=initial.build_spec,
    )

    with pytest.raises(ManifestConflictError, match="immediately preceding"):
        store.publish(invalid)


def test_store_serializes_concurrent_publication_of_one_revision(
    tmp_path,
    monkeypatch,
) -> None:
    store = ManifestStore(tmp_path)
    initial, initial_ref = store.create(make_spec())
    start = threading.Barrier(2)
    second_publish_reached = threading.Event()
    publish_count = 0
    publish_count_lock = threading.Lock()
    real_publish = artifacts_module.atomic_publish_json

    def synchronized_publish(path, payload, **kwargs):
        nonlocal publish_count
        if Path(path).name.startswith("manifest-r000001-"):
            with publish_count_lock:
                publish_count += 1
                ordinal = publish_count
            if ordinal == 1:
                # Without a revision lock, the second writer reaches publication
                # before either destination exists and both different manifests
                # are accepted. With the lock, it remains blocked until the first
                # writer has published.
                second_publish_reached.wait(timeout=0.5)
            else:
                second_publish_reached.set()
        return real_publish(path, payload, **kwargs)

    monkeypatch.setattr(artifacts_module, "atomic_publish_json", synchronized_publish)

    def revise(index: int) -> str:
        stage = StageRecord(
            name=f"concurrent-{index}",
            status=StageStatus.SUCCEEDED,
            completed_at=datetime.now(timezone.utc),
        )
        start.wait()
        try:
            store.revise(initial_ref, stage=stage)
        except ManifestConflictError:
            return "conflict"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(revise, (1, 2)))

    assert sorted(outcomes) == ["conflict", "published"]
    assert publish_count == 1
    run_directory = store.root / str(initial.run_id)
    revisions = tuple(run_directory.glob("manifest-r000001-*.json"))
    assert len(revisions) == 1
