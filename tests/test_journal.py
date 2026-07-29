from __future__ import annotations

from datetime import timedelta

import pytest

from qairt_agent.contracts import ArtifactRef, StageStatus, utc_now
from qairt_agent.contracts import (
    JobState,
    StageProvenance,
    StageReceipt,
)
from qairt_agent.errors import JobConflictError, JobNotFoundError
from qairt_agent.jobs.journal import JobJournal
from qairt_agent.jobs.keys import compute_stage_key, hash_inputs


def _provenance() -> StageProvenance:
    return StageProvenance(
        sdk_build="260626120635",
        adapter_capability="explicit_factory",
        platform_abi="ubuntu22.04-x86_64",
        resolved_preset_sha256="p" * 64,
    )


def _make_journal(tmp_path, job_id="job-1", **kwargs) -> JobJournal:
    return JobJournal.create(
        tmp_path / "jobs",
        job_id,
        spec_original={"preset": "qwen3_dense"},
        spec_resolved={"preset_id": "qwen3_dense", "pipeline": "low_level"},
        spec_sha256="s" * 64,
        launcher={"kind": "local"},
        **kwargs,
    )


def _receipt_with_output(tmp_path, name: str, key: str, *, content: bytes = b"data") -> StageReceipt:
    out = tmp_path / f"{name}.bin"
    out.write_bytes(content)
    ref = ArtifactRef.from_path(out)
    return StageReceipt(
        stage_key=key,
        stage_name=name,
        status=StageStatus.SUCCEEDED,
        started_at=utc_now(),
        completed_at=utc_now(),
        outputs=(ref,),
        provenance=_provenance(),
    )


def test_create_open_exists_list(tmp_path) -> None:
    journal = _make_journal(tmp_path, "job-a")
    assert JobJournal.exists(tmp_path / "jobs", "job-a")
    assert journal.state().state == JobState.QUEUED
    assert journal.state().seq == 1  # job_created event

    reopened = JobJournal.open(tmp_path / "jobs", "job-a")
    assert reopened.state().job_id == "job-a"

    _make_journal(tmp_path, "job-b")
    assert JobJournal.list_jobs(tmp_path / "jobs") == ["job-a", "job-b"]


def test_create_rejects_duplicate(tmp_path) -> None:
    _make_journal(tmp_path, "job-a")
    with pytest.raises(JobConflictError, match="already exists"):
        _make_journal(tmp_path, "job-a")


def test_open_missing_raises(tmp_path) -> None:
    with pytest.raises(JobNotFoundError, match="not found"):
        JobJournal.open(tmp_path / "jobs", "nope")


def test_state_transitions_and_event_sequence(tmp_path) -> None:
    journal = _make_journal(tmp_path)
    journal.set_state(JobState.STAGING)
    journal.set_state(JobState.RUNNING, current_stage="build")
    journal.set_state(JobState.SUCCEEDED)

    state = journal.state()
    assert state.state == JobState.SUCCEEDED
    assert state.current_stage == "build"

    events = journal.events()
    types = [event["type"] for event in events]
    assert types[0] == "job_created"
    assert types.count("state_changed") == 3
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))


def test_cannot_leave_terminal_state(tmp_path) -> None:
    journal = _make_journal(tmp_path)
    journal.set_state(JobState.SUCCEEDED)
    with pytest.raises(JobConflictError, match="terminal"):
        journal.set_state(JobState.RUNNING)


def test_watch_after_seq_resumes(tmp_path) -> None:
    journal = _make_journal(tmp_path)
    journal.set_state(JobState.RUNNING)
    journal.append_event("message", {"text": "hello"})
    full = journal.events()
    cutoff = full[1]["seq"]
    tail = journal.events(after_seq=cutoff)
    assert [event["seq"] for event in tail] == [event["seq"] for event in full[2:]]
    assert all(event["seq"] > cutoff for event in tail)


def test_receipts_verify_against_real_artifacts(tmp_path) -> None:
    journal = _make_journal(tmp_path)
    key = compute_stage_key(
        stage_name="build",
        inputs_sha256=hash_inputs(["a" * 64]),
        resolved_preset_sha256="p" * 64,
        sdk_build="260626120635",
        adapter_capability="explicit_factory",
        platform_abi="ubuntu22.04-x86_64",
    )
    receipt = _receipt_with_output(tmp_path, "build", key)
    journal.record_receipt(receipt)

    assert [r.stage_name for r in journal.receipts()] == ["build"]
    assert [r.stage_key for r in journal.verified_receipts()] == [key]
    assert journal.receipt_for_stage_key(key) is not None
    assert journal.last_verified_receipt().stage_key == key
    # state now carries the receipt and current stage
    assert journal.state().current_stage == "build"
    assert len(journal.state().stages) == 1


def test_tampered_output_is_not_verified(tmp_path) -> None:
    journal = _make_journal(tmp_path)
    receipt = _receipt_with_output(tmp_path, "build", "k" * 64, content=b"original")
    journal.record_receipt(receipt)
    assert len(journal.verified_receipts()) == 1

    # Tamper with the output file -> verification fails -> not reusable.
    receipt.outputs[0].path.write_bytes(b"tampered")
    assert journal.verified_receipts() == []
    assert journal.receipt_for_stage_key("k" * 64) is None


def test_recording_same_receipt_is_idempotent(tmp_path) -> None:
    journal = _make_journal(tmp_path)
    receipt = _receipt_with_output(tmp_path, "build", "k" * 64)
    journal.record_receipt(receipt)
    journal.record_receipt(receipt)  # identical -> no conflict, no duplicate stage
    assert len(journal.state().stages) == 1


def test_heartbeat_cancel_logs_specs(tmp_path) -> None:
    journal = _make_journal(tmp_path)
    journal.touch_heartbeat(pid=1234)
    assert journal.heartbeat()["pid"] == 1234

    assert journal.cancel_requested() is False
    journal.request_cancel()
    assert journal.cancel_requested() is True
    journal.clear_cancel()
    assert journal.cancel_requested() is False

    journal.write_log("build", "line one")
    journal.write_log("build", "line two")
    assert journal.read_log("build") == "line one\nline two\n"
    assert journal.read_log("missing") == ""

    assert journal.spec_original()["preset"] == "qwen3_dense"
    assert journal.spec_resolved()["pipeline"] == "low_level"
    assert journal.launcher()["kind"] == "local"


def test_terminal_job_rejects_heartbeat_touch(tmp_path) -> None:
    journal = _make_journal(tmp_path)
    before = utc_now()
    assert journal.touch_heartbeat(pid=1234, now=before) is True
    journal.set_state(JobState.SUCCEEDED)
    payload = journal.heartbeat()

    assert journal.touch_heartbeat(
        pid=5678,
        now=before + timedelta(seconds=10),
    ) is False
    assert journal.heartbeat() == payload
    assert journal.state().heartbeat_at == before


def test_worker_lease_rejects_concurrent_owner(tmp_path) -> None:
    journal = _make_journal(tmp_path)

    with journal.worker_lease():
        with pytest.raises(JobConflictError, match="active worker"):
            with journal.worker_lease():
                raise AssertionError("a second worker must not acquire the lease")

    # The lease is reusable after the original owner exits normally.
    with journal.worker_lease():
        pass


def test_stale_inflight_transition_records_orphan_event(tmp_path) -> None:
    journal = _make_journal(tmp_path)
    heartbeat_at = utc_now()
    journal.set_state(JobState.RUNNING, current_stage="build")
    journal.touch_heartbeat(pid=1234, now=heartbeat_at)

    with journal.worker_lease():
        status = journal.mark_orphaned_if_stale(
            30.0,
            now=heartbeat_at + timedelta(seconds=31),
        )

    assert status.state == JobState.ORPHANED
    event = journal.events()[-1]
    assert event["type"] == "state_changed"
    assert event["payload"]["state"] == "orphaned"
    assert event["payload"]["previous_state"] == "running"


def test_stage_key_changes_with_provenance(tmp_path) -> None:
    base = dict(
        stage_name="build",
        inputs_sha256=hash_inputs(["a" * 64]),
        resolved_preset_sha256="p" * 64,
        sdk_build="260626120635",
        adapter_capability="explicit_factory",
        platform_abi="ubuntu22.04-x86_64",
    )
    key_a = compute_stage_key(**base)
    assert compute_stage_key(**base) == key_a
    # A different image digest or device fingerprint changes the key.
    assert compute_stage_key(**base, image_digest="sha256:x") != key_a
    assert compute_stage_key(**base, device_fingerprint="serial@host") != key_a
    # Different inputs change the key.
    assert compute_stage_key(**{**base, "inputs_sha256": hash_inputs(["b" * 64])}) != key_a
