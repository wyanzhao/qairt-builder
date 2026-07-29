from __future__ import annotations

import json
import os
import time
from datetime import timedelta

import pytest

from qairt_agent.agent import spec_sha256
from qairt_agent.artifacts import ManifestStore
from qairt_agent.contracts import (
    ArtifactKind,
    ArtifactRef,
    BuildSpec,
    JobState,
    StageRecord,
    StageProvenance,
    StageReceipt,
    StageStatus,
    ToolResult,
    WorkflowSpec,
    to_workflow_spec,
    utc_now,
)
from qairt_agent.errors import JobConflictError
from qairt_agent.families.presets import to_build_spec
from qairt_agent.jobs.engine_runner import EngineStageRunner
from qairt_agent.jobs.journal import JobJournal
from qairt_agent.jobs.worker import StageContext, StageOutcome, WorkflowWorker
from qairt_agent.pipeline import QairtAgent


PROVENANCE = StageProvenance(
    sdk_build="260626120635",
    adapter_capability="explicit_factory",
    platform_abi="ubuntu22.04-x86_64",
    resolved_preset_sha256="p" * 64,
)


def _workflow_spec() -> WorkflowSpec:
    return to_workflow_spec(
        BuildSpec.model_validate(
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
    )


def _journal(tmp_path, job_id: str = "job-recovery") -> tuple[JobJournal, dict]:
    workflow = _workflow_spec()
    payload = json.loads(workflow.model_dump_json())
    journal = JobJournal.create(
        tmp_path / "jobs",
        job_id,
        spec_original=payload,
        spec_resolved={"preset_id": "qwen3_dense"},
        spec_sha256=spec_sha256(workflow),
    )
    return journal, workflow.model_dump(mode="json")


def _manifest(ctx, name: str = "manifest.json") -> ArtifactRef:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.output_dir / name
    path.write_text('{"ok":true}', encoding="utf-8")
    return ArtifactRef.from_path(path, kind=ArtifactKind.MANIFEST)


def test_process_heartbeat_survives_blocking_stage_and_stops_at_terminal(tmp_path) -> None:
    journal, spec = _journal(tmp_path)
    observed: list[str] = []

    def runner(ctx):
        deadline = time.monotonic() + 5.0
        previous = None
        while time.monotonic() < deadline:
            heartbeat = journal.heartbeat()
            if heartbeat is not None:
                assert heartbeat["pid"] == os.getpid()
                os.kill(int(heartbeat["pid"]), 0)
                stamp = str(heartbeat["heartbeat_at"])
                if previous is not None and stamp != previous:
                    observed.extend([previous, stamp])
                    break
                previous = stamp
            time.sleep(0.01)
        ref = _manifest(ctx)
        return StageOutcome(manifest=ref)

    worker = WorkflowWorker(
        journal,
        spec=spec,
        resolved={},
        provenance=PROVENANCE,
        stage_runner=runner,
        stages=("build",),
        heartbeat_interval=0.02,
        heartbeat_stale_after=0.2,
    )
    status = worker.run()

    assert status.state == JobState.SUCCEEDED
    assert len(observed) == 2
    final_heartbeat = journal.heartbeat()
    assert final_heartbeat is not None
    assert status.heartbeat_at is not None
    assert status.heartbeat_at.isoformat() == final_heartbeat["heartbeat_at"]

    time.sleep(0.08)
    assert journal.heartbeat() == final_heartbeat
    assert journal.touch_heartbeat() is False
    assert journal.heartbeat() == final_heartbeat


def test_stale_inflight_job_is_orphaned_and_retries_new_attempt_directory(tmp_path) -> None:
    journal, spec = _journal(tmp_path)
    base = utc_now()
    attempts: list[tuple[int, str]] = []

    def runner(ctx):
        attempts.append((ctx.attempt, ctx.output_dir.name))
        ref = _manifest(ctx)
        return StageOutcome(manifest=ref)

    worker = WorkflowWorker(
        journal,
        spec=spec,
        resolved={},
        provenance=PROVENANCE,
        stage_runner=runner,
        stages=("build",),
        heartbeat_interval=0.02,
        heartbeat_stale_after=10.0,
        clock=lambda: base + timedelta(seconds=11),
    )
    stage_key = worker._stage_key("build", None)  # noqa: SLF001 - recovery fixture
    journal.set_state(JobState.STAGING, current_stage="build")
    journal.set_state(JobState.RUNNING, current_stage="build")
    journal.touch_heartbeat(pid=987654, now=base)
    journal.append_event(
        "stage_started",
        {"stage_name": "build", "stage_key": stage_key, "attempt": 1},
    )

    status = worker.run()

    assert status.state == JobState.SUCCEEDED
    assert attempts == [(2, "attempt-002")]
    receipts = journal.receipts()
    assert len(receipts) == 1
    assert receipts[0].attempt == 2
    assert receipts[0].status == StageStatus.SUCCEEDED
    orphan_events = [
        event
        for event in journal.events()
        if event["type"] == "state_changed"
        and event["payload"]["state"] == JobState.ORPHANED.value
    ]
    assert len(orphan_events) == 1
    assert orphan_events[0]["payload"]["previous_state"] == JobState.RUNNING.value


def test_fresh_inflight_job_is_not_double_executed(tmp_path) -> None:
    journal, spec = _journal(tmp_path)
    now = utc_now()
    called = False

    def runner(ctx):
        nonlocal called
        called = True
        return StageOutcome()

    journal.set_state(JobState.STAGING)
    journal.set_state(JobState.RUNNING, current_stage="build")
    journal.touch_heartbeat(now=now)
    worker = WorkflowWorker(
        journal,
        spec=spec,
        resolved={},
        provenance=PROVENANCE,
        stage_runner=runner,
        stages=("build",),
        heartbeat_interval=0.02,
        heartbeat_stale_after=10.0,
        clock=lambda: now + timedelta(seconds=9),
    )

    with pytest.raises(JobConflictError, match="fresh worker heartbeat"):
        worker.run()

    assert called is False
    assert journal.state().state == JobState.RUNNING


def test_success_receipt_reuses_manifest_after_crash_before_publish(tmp_path) -> None:
    journal, spec = _journal(tmp_path)
    base = utc_now()
    calls: list[int] = []

    def runner(ctx):
        calls.append(ctx.attempt)
        ref = _manifest(ctx)
        return StageOutcome(manifest=ref)

    first = WorkflowWorker(
        journal,
        spec=spec,
        resolved={},
        provenance=PROVENANCE,
        stage_runner=runner,
        stages=("build",),
        heartbeat_interval=0.02,
        heartbeat_stale_after=1.0,
    )
    key = first._stage_key("build", None)  # noqa: SLF001 - recovery fixture
    output_dir = journal.dir / "work" / "build" / key[:16] / "attempt-001"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text('{"completed":true}', encoding="utf-8")
    manifest = ArtifactRef.from_path(manifest_path, kind=ArtifactKind.MANIFEST)
    journal.append_event(
        "stage_started",
        {"stage_name": "build", "stage_key": key, "attempt": 1},
    )
    journal.record_receipt(
        StageReceipt(
            stage_key=key,
            stage_name="build",
            attempt=1,
            status=StageStatus.SUCCEEDED,
            started_at=base,
            completed_at=base,
            outputs=(manifest,),
            provenance=PROVENANCE,
        )
    )
    journal.set_state(JobState.RUNNING, current_stage="build")
    journal.touch_heartbeat(now=base)

    resumed = WorkflowWorker(
        journal,
        spec=spec,
        resolved={},
        provenance=PROVENANCE,
        stage_runner=runner,
        stages=("build",),
        heartbeat_interval=0.02,
        heartbeat_stale_after=1.0,
        clock=lambda: base + timedelta(seconds=2),
    )
    status = resumed.run()

    assert status.state == JobState.SUCCEEDED
    assert status.manifest == manifest
    assert calls == []
    assert len(journal.receipts()) == 1


def test_engine_runner_uses_fresh_attempt_in_durable_pipeline_tree(tmp_path) -> None:
    workflow = _workflow_spec()
    build_spec = to_build_spec(workflow).model_copy(
        update={"output_root": tmp_path / "published"}
    )
    store = ManifestStore(build_spec.output_root / "manifests")
    run, manifest = store.create(build_spec)
    diagnose_config = {
        "reference_trace": {"tap": [1.0, 2.0]},
        "actual_trace": {"tap": [1.0, 2.0]},
    }
    stage_key = QairtAgent._stage_key(  # noqa: SLF001 - production path assertion
        "diagnose_quality",
        manifest.sha256,
        diagnose_config,
    )
    stage_root = (
        build_spec.output_root
        / "runs"
        / str(run.run_id)
        / "stages"
        / "diagnose_quality"
        / stage_key[:16]
    )
    attempt_one = stage_root / "attempt-001"
    attempt_two = stage_root / "attempt-002"
    attempt_one.mkdir(parents=True)
    partial = attempt_one / "quality_diagnosis.json"
    partial.write_text('{"partial":true}', encoding="utf-8")

    runner = EngineStageRunner(
        engine=QairtAgent(),
        build_spec=build_spec.model_dump(mode="json"),
        pipeline="low_level",
        diagnose_config=diagnose_config,
        diagnose_kind="quality",
    )
    journal, spec = _journal(tmp_path, "job-production-attempt")
    custom_runner_dir = (
        tmp_path / "job-work" / "diagnose" / "key" / "attempt-002"
    )
    outcome = runner(
        StageContext(
            stage_name="diagnose",
            spec=spec,
            resolved={},
            journal=journal,
            current_manifest=manifest,
            output_dir=custom_runner_dir,
            attempt=2,
        )
    )

    assert outcome.manifest is not None
    assert (attempt_two / "quality_diagnosis.json").is_file()
    assert not custom_runner_dir.exists()
    assert partial.read_text(encoding="utf-8") == '{"partial":true}'
    final = store.load(outcome.manifest)
    assert final.stages[-1].attempt == 2


def test_engine_runner_receipt_carries_transitive_manifest_artifacts(tmp_path) -> None:
    build_spec = to_build_spec(_workflow_spec()).model_copy(
        update={"output_root": tmp_path / "published"}
    )
    store = ManifestStore(build_spec.output_root / "manifests")
    _, initial = store.create(build_spec)
    context_path = tmp_path / "published" / "context.bin"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_bytes(b"verified-context")
    context = ArtifactRef.from_path(
        context_path,
        kind=ArtifactKind.CONTEXT_BINARY,
        logical_name="decoder_context",
    )
    stage = StageRecord(
        name="build",
        status=StageStatus.SUCCEEDED,
        started_at=utc_now(),
        completed_at=utc_now(),
        outputs=(context,),
    )
    _, final_manifest = store.revise(initial, stage=stage, artifacts=(context,))

    class ManifestEngine:
        def build(self, spec, *, execution_context=None):
            return ToolResult.success({"stage": "build"}, manifest=final_manifest)

    runner = EngineStageRunner(
        engine=ManifestEngine(),
        build_spec=build_spec.model_dump(mode="json"),
        pipeline="low_level",
    )
    journal, spec = _journal(tmp_path, "job-manifest-graph")
    outcome = runner(
        StageContext(
            stage_name="build",
            spec=spec,
            resolved={},
            journal=journal,
            current_manifest=None,
            output_dir=tmp_path / "job-work" / "build" / "key" / "attempt-001",
            attempt=1,
        )
    )
    key = "a" * 64
    journal.record_receipt(
        StageReceipt(
            stage_key=key,
            stage_name="build",
            attempt=1,
            status=StageStatus.SUCCEEDED,
            started_at=utc_now(),
            completed_at=utc_now(),
            outputs=WorkflowWorker._receipt_outputs(outcome),  # noqa: SLF001
            provenance=PROVENANCE,
        )
    )

    assert context in outcome.artifacts
    assert journal.receipt_for_stage_key(key) is not None
    context_path.unlink()
    assert journal.receipt_for_stage_key(key) is None
