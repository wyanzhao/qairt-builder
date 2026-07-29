"""Workflow worker: drives stages, records verified receipts, publishes once.

The worker is the only component that advances a job through the state machine
and the only manifest publisher.  Stage execution is delegated to a pluggable
``stage_runner`` so the journal/worker logic is testable without the QAIRT SDK;
the default runner wraps :class:`qairt_agent.pipeline.QairtAgent`.

Reuse rules:

* ``resume`` skips any stage whose verified receipt already exists in this
  journal (replay from the last verified receipt; incomplete attempts are never
  reused).
* ``rerun`` additionally consults a parent journal: a stage whose key is
  unchanged from a verified parent receipt is reused instead of re-run.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from qairt_agent.artifacts import ManifestStore, canonical_json_bytes, verify_artifact
from qairt_agent.contracts import (
    ArtifactKind,
    ArtifactRef,
    BuildSpec,
    RunManifest,
    StageStatus,
    utc_now,
)
from qairt_agent.contracts import (
    JobState,
    JobStatus,
    StageProvenance,
    StageReceipt,
)
from qairt_agent.errors import JobCancelledError, JobConflictError, ToolErrorData
from qairt_agent.jobs.heartbeat import HeartbeatWriter
from qairt_agent.jobs.journal import IN_FLIGHT_JOB_STATES, JobJournal
from qairt_agent.jobs.keys import compute_stage_key, content_identity, hash_inputs

DEFAULT_WORKFLOW_STAGES: tuple[str, ...] = ("build", "validate", "benchmark")
DEFAULT_HEARTBEAT_INTERVAL = 5.0
DEFAULT_HEARTBEAT_STALE_AFTER = 30.0


@dataclass(frozen=True)
class StageContext:
    """Everything a stage runner needs to execute one stage."""

    stage_name: str
    spec: dict[str, Any]
    resolved: dict[str, Any]
    journal: JobJournal
    current_manifest: ArtifactRef | None
    output_dir: Path
    attempt: int


@dataclass(frozen=True)
class StageOutcome:
    """The result of running one stage."""

    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    manifest: ArtifactRef | None = None
    verified: bool = False


StageRunner = Callable[[StageContext], StageOutcome]


class ManifestPublisher:
    """Serializes manifest commits so parallel slices never race a revision.

    Slices/worker stages submit receipts to the journal; the single publisher
    is the only thing that advances ``state.manifest``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def commit(self, journal: JobJournal, manifest: ArtifactRef | None, stage_name: str) -> None:
        if manifest is None:
            return
        with self._lock:
            current = journal.state()
            journal.set_state(
                current.state,
                manifest=manifest,
                current_stage=stage_name,
                event_payload={"manifest_sha256": manifest.sha256},
            )


class WorkflowWorker:
    """Run a job's stages, recording verified receipts and publishing once."""

    def __init__(
        self,
        journal: JobJournal,
        *,
        spec: dict[str, Any],
        resolved: dict[str, Any],
        provenance: StageProvenance,
        stage_runner: StageRunner,
        stages: tuple[str, ...] = DEFAULT_WORKFLOW_STAGES,
        reuse_from: JobJournal | None = None,
        publisher: ManifestPublisher | None = None,
        output_dir: Path | None = None,
        initial_manifest: ArtifactRef | None = None,
        effective_build_spec: BuildSpec | None = None,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        heartbeat_stale_after: float = DEFAULT_HEARTBEAT_STALE_AFTER,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if heartbeat_stale_after <= 0:
            raise ValueError("heartbeat_stale_after must be positive")
        self.journal = journal
        self.spec = spec
        self.resolved = resolved
        self.provenance = provenance
        self.stage_runner = stage_runner
        self.stages = stages
        self.reuse_from = reuse_from
        self.publisher = publisher or ManifestPublisher()
        self.output_dir = output_dir or (journal.dir / "work")
        self.initial_manifest = initial_manifest
        self.effective_build_spec = effective_build_spec
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_stale_after = heartbeat_stale_after
        self.clock = clock
        configured_project_root = (
            os.environ.get("QAIRT_AGENT_PROJECT_ROOT") or ""
        ).strip()
        if configured_project_root:
            self.project_root = Path(
                configured_project_root
            ).expanduser().resolve()
        elif (
            journal.root.name == "jobs"
            and journal.root.parent.name == ".qairt-agent"
        ):
            self.project_root = journal.root.parent.parent.resolve()
        else:
            self.project_root = journal.root.parent.resolve()

    # ------------------------------------------------------------------ #
    # Stage keys are stage-aware so an adjusted spec only invalidates the
    # stages its change actually affects: the build key depends on the
    # build-relevant spec projection, while later stages depend on the input
    # manifest plus their own config projection.
    _BUILD_KEYS = (
        "name",
        "preset",
        "sku",
        "sources",
        "output_root",
        "sequence",
        "split",
        "transforms",
        "quantization",
        "vectors",
        "compile",
        "quality",
        "target",
        "metadata",
    )
    _QUALITY_KEYS = ("quality", "vectors")
    _BENCHMARK_KEYS = ("benchmark",)
    _BUILD_SPEC_REUSE_FIELDS = (
        "name",
        "family",
        "sources",
        "output_root",
        "sequence",
        "split",
        "transforms",
        "quantization",
        "vectors",
        "compile",
        "quality",
        "target",
        "metadata",
    )

    def _project(self, keys: tuple[str, ...]) -> dict[str, Any]:
        return {key: self.spec[key] for key in keys if key in self.spec}

    def _stage_config(self, stage_name: str) -> dict[str, Any]:
        configs = self.spec.get("stage_configs")
        if not isinstance(configs, dict):
            return {}
        if stage_name == "validate":
            selected = configs.get("validation", configs.get("validate", {}))
        else:
            selected = configs.get(stage_name, {})
        return dict(selected) if isinstance(selected, dict) else {}

    def _input_basis(self, stage_name: str, manifest: ArtifactRef | None) -> str:
        if stage_name == "build":
            projection = {
                "spec": self._project(self._BUILD_KEYS),
                "stage_config": self._stage_config("build"),
            }
            return hashlib.sha256(
                canonical_json_bytes(
                    content_identity(
                        projection,
                        project_root=self.project_root,
                    )
                )
            ).hexdigest()
        if manifest is not None:
            # A manifest is immutable input, not merely a trusted string from a
            # receipt. Re-hash it before deriving a continuation-stage key.
            verify_artifact(manifest)
        manifest_identity = (
            self._manifest_stage_key_identity(manifest)
            if manifest is not None
            else ""
        )
        manifest_sha = hash_inputs([manifest_identity]) if manifest is not None else ""
        if stage_name == "benchmark":
            projection = self._project(self._BENCHMARK_KEYS)
        elif stage_name in {"validate", "diagnose"}:
            projection = self._project(self._QUALITY_KEYS)
        else:
            projection = self.spec
        combined = {
            "manifest": manifest_sha,
            "config": projection,
            "stage_config": self._stage_config(stage_name),
        }
        combined = content_identity(
            combined,
            project_root=self.project_root,
        )
        return hashlib.sha256(canonical_json_bytes(combined)).hexdigest()

    @staticmethod
    def _manifest_stage_key_identity(manifest: ArtifactRef) -> str:
        """Use a fork snapshot's verified source SHA as its stable key identity."""

        if not (manifest.logical_name or "").startswith("run-manifest-r"):
            return manifest.sha256
        parsed = RunManifest.model_validate_json(manifest.path.read_bytes())
        if parsed.revision != 0:
            return manifest.sha256
        raw_source = parsed.metadata.get("forked_from_manifest")
        if not isinstance(raw_source, dict):
            return manifest.sha256
        source = ArtifactRef.model_validate(raw_source)
        if source.kind is not ArtifactKind.MANIFEST:
            raise JobConflictError(
                "fork stage-key identity is not a manifest artifact",
                stage="manifest_fork",
                details={"manifest": str(manifest.path)},
            )
        verify_artifact(source)
        return source.sha256

    def _stage_key(self, stage_name: str, manifest: ArtifactRef | None) -> str:
        return compute_stage_key(
            stage_name=stage_name,
            inputs_sha256=self._input_basis(stage_name, manifest),
            resolved_preset_sha256=self.provenance.resolved_preset_sha256 or "",
            sdk_build=self.provenance.sdk_build,
            adapter_capability=self.provenance.adapter_capability,
            platform_abi=self.provenance.platform_abi,
            image_digest=self.provenance.image_digest,
            device_fingerprint=self.provenance.device_fingerprint,
        )

    def _next_attempt(self, stage_name: str, key: str) -> int:
        attempts = {
            r.attempt for r in self.journal.receipts() if r.stage_name == stage_name and r.stage_key == key
        }
        # A hard-killed stage has no receipt, but its immutable stage_started
        # event reserves the attempt number and output directory.
        for event in self.journal.events():
            if event.get("type") != "stage_started":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            if payload.get("stage_name") != stage_name or payload.get("stage_key") != key:
                continue
            attempt = payload.get("attempt")
            if isinstance(attempt, int) and attempt > 0:
                attempts.add(attempt)
        return (max(attempts) + 1) if attempts else 1

    def _try_reuse(self, stage_name: str, key: str) -> StageReceipt | None:
        # Resume: a verified receipt already in this journal wins.
        existing = self.journal.receipt_for_stage_key(key)
        if existing is not None:
            return existing
        # Rerun: reuse an unchanged stage from the parent job.
        if self.reuse_from is not None:
            parent = self.reuse_from.receipt_for_stage_key(key)
            if parent is not None:
                reused = StageReceipt(
                    stage_key=parent.stage_key,
                    stage_name=parent.stage_name,
                    attempt=self._next_attempt(stage_name, key),
                    status=StageStatus.SUCCEEDED,
                    started_at=utc_now(),
                    completed_at=utc_now(),
                    inputs=parent.inputs,
                    outputs=parent.outputs,
                    metrics={**parent.metrics, "reused_from_parent": True},
                    provenance=parent.provenance,
                )
                self.journal.record_receipt(reused)
                return reused
        return None

    def _build_spec_reuse_identity(self, build_spec: BuildSpec) -> str:
        """Hash only fields whose change requires rebuilding compiled artifacts."""

        payload = build_spec.model_dump(mode="python")
        projected = {
            key: payload[key]
            for key in self._BUILD_SPEC_REUSE_FIELDS
        }
        projected["stage_config"] = payload["stage_configs"]["build"]
        return hashlib.sha256(
            canonical_json_bytes(
                content_identity(
                    projected,
                    project_root=self.project_root,
                )
            )
        ).hexdigest()

    def _has_verified_ancestor_build_receipt(self) -> bool:
        """Prove the current build key exists in this job's verified ancestry."""

        expected_key = self._stage_key("build", None)
        candidates: list[JobJournal] = [self.journal]
        if self.reuse_from is not None:
            candidates.append(self.reuse_from)
        parent_job_id = self.journal.state().parent_job_id
        if (
            parent_job_id is not None
            and JobJournal.exists(self.journal.root, parent_job_id)
        ):
            candidates.append(JobJournal.open(self.journal.root, parent_job_id))

        visited: set[str] = set()
        while candidates:
            candidate = candidates.pop(0)
            if candidate.job_id in visited:
                continue
            visited.add(candidate.job_id)
            if any(
                receipt.stage_name == "build"
                and receipt.stage_key == expected_key
                for receipt in candidate.verified_receipts()
            ):
                return True
            ancestor_id = candidate.state().parent_job_id
            if (
                ancestor_id is not None
                and ancestor_id not in visited
                and JobJournal.exists(self.journal.root, ancestor_id)
            ):
                candidates.append(
                    JobJournal.open(self.journal.root, ancestor_id)
                )
        return False

    def _replacement_build_spec_for_fork(
        self,
        source: RunManifest,
    ) -> BuildSpec | None:
        """Return a safe effective spec, or fail closed on provenance drift."""

        effective = self.effective_build_spec
        if effective is None or effective == source.build_spec:
            return None
        source_identity = self._build_spec_reuse_identity(source.build_spec)
        effective_identity = self._build_spec_reuse_identity(effective)
        if source_identity != effective_identity:
            raise JobConflictError(
                "cannot rebase a manifest fork across build-relevant spec changes",
                stage="manifest_fork",
                details={
                    "source_build_identity": source_identity,
                    "effective_build_identity": effective_identity,
                    "source_family": source.build_spec.family.value,
                    "effective_family": effective.family.value,
                },
            )
        if not self._has_verified_ancestor_build_receipt():
            raise JobConflictError(
                "cannot rebase manifest BuildSpec without a verified matching build receipt",
                stage="manifest_fork",
                details={
                    "build_stage_key": self._stage_key("build", None),
                    "source_run_id": str(source.run_id),
                },
            )
        return effective

    def _verify_outputs(self, outcome: StageOutcome) -> None:
        if outcome.verified:
            return
        for ref in outcome.artifacts:
            verify_artifact(ref)
        if outcome.manifest is not None:
            verify_artifact(outcome.manifest)

    @staticmethod
    def _receipt_outputs(outcome: StageOutcome) -> tuple[ArtifactRef, ...]:
        """Keep the committed manifest as the last reusable receipt output."""

        outputs = list(outcome.artifacts)
        manifest = outcome.manifest
        if manifest is not None and not any(
            item.path == manifest.path and item.sha256 == manifest.sha256 for item in outputs
        ):
            outputs.append(manifest)
        return tuple(outputs)

    def _fork_reused_manifest(
        self,
        manifest: ArtifactRef,
        *,
        next_stage: str | None,
        reason: str | None = None,
        source_job_id: str | None = None,
    ) -> tuple[ArtifactRef, bool]:
        """Snapshot a production manifest before this rerun diverges or ends."""

        if not (manifest.logical_name or "").startswith("run-manifest-r"):
            # Custom/fake runners may use opaque JSON tagged as MANIFEST. They
            # have no ManifestStore chain to snapshot.
            return manifest, False
        fork_reason = reason or (
            f"before_stage:{next_stage}"
            if next_stage is not None
            else "all_stages_reused"
        )
        path = manifest.path.expanduser().resolve()
        if len(path.parents) < 2:
            raise JobConflictError(
                "reused manifest has no discoverable ManifestStore root",
                stage="manifest_fork",
                details={"manifest": str(path)},
            )
        store = ManifestStore(path.parent.parent)
        source = store.load(manifest)
        replacement_build_spec = self._replacement_build_spec_for_fork(source)
        expected_build_spec = replacement_build_spec or source.build_spec
        effective_build_spec_sha256 = hashlib.sha256(
            canonical_json_bytes(expected_build_spec)
        ).hexdigest()
        for event in reversed(self.journal.events()):
            if event.get("type") != "manifest_forked":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            if (
                payload.get("source_sha256") != manifest.sha256
                or payload.get("next_stage") != next_stage
                or payload.get("fork_reason") != fork_reason
            ):
                continue
            fork_path = Path(str(payload.get("fork_manifest", "")))
            expected_sha = str(payload.get("fork_sha256", ""))
            if not fork_path.is_file():
                raise JobConflictError(
                    "recorded rerun manifest fork is missing",
                    stage="manifest_fork",
                    details={"fork_manifest": str(fork_path)},
                )
            fork_ref = ArtifactRef.from_path(
                fork_path,
                kind=ArtifactKind.MANIFEST,
                logical_name="run-manifest-r0",
            )
            if fork_ref.sha256 != expected_sha:
                raise JobConflictError(
                    "recorded rerun manifest fork failed integrity verification",
                    stage="manifest_fork",
                    details={
                        "fork_manifest": str(fork_path),
                        "expected_sha256": expected_sha,
                        "actual_sha256": fork_ref.sha256,
                    },
                )
            forked = ManifestStore(fork_path.parent.parent).load(fork_ref)
            if forked.build_spec != expected_build_spec:
                raise JobConflictError(
                    "recorded rerun fork has a stale effective BuildSpec",
                    stage="manifest_fork",
                    details={
                        "fork_manifest": str(fork_path),
                        "expected_build_spec_sha256": (
                            effective_build_spec_sha256
                        ),
                        "actual_build_spec_sha256": hashlib.sha256(
                            canonical_json_bytes(forked.build_spec)
                        ).hexdigest(),
                    },
                )
            return fork_ref, True
        _, fork_ref = store.fork_snapshot(
            manifest,
            replacement_build_spec=replacement_build_spec,
            metadata_update={
                "forked_from_job_id": (
                    source_job_id
                    or (
                        self.reuse_from.job_id
                        if self.reuse_from is not None
                        else self.journal.state().parent_job_id
                    )
                ),
                "forked_for_job_id": self.journal.job_id,
                "fork_reason": fork_reason,
            },
        )
        self.journal.append_event(
            "manifest_forked",
            {
                "source_manifest": str(manifest.path),
                "source_sha256": manifest.sha256,
                "fork_manifest": str(fork_ref.path),
                "fork_sha256": fork_ref.sha256,
                "next_stage": next_stage,
                "fork_reason": fork_reason,
                "effective_build_spec_sha256": effective_build_spec_sha256,
                "build_spec_rebased": replacement_build_spec is not None,
            },
        )
        return fork_ref, True

    @staticmethod
    def _manifest_attempt_floor(
        manifest: ArtifactRef | None,
        stage_name: str,
    ) -> int:
        if manifest is None or not (
            manifest.logical_name or ""
        ).startswith("run-manifest-r"):
            return 1
        parsed = RunManifest.model_validate_json(manifest.path.read_bytes())
        return 1 + max(
            (
                stage.attempt
                for stage in parsed.stages
                if stage.name == stage_name
            ),
            default=0,
        )

    # ------------------------------------------------------------------ #
    def run(self) -> JobStatus:
        journal = self.journal
        with journal.worker_lease():
            current = journal.mark_orphaned_if_stale(
                self.heartbeat_stale_after,
                now=self.clock(),
            )
            if current.state.terminal:
                return current
            if current.state in IN_FLIGHT_JOB_STATES:
                raise JobConflictError(
                    f"job '{journal.job_id}' still has a fresh worker heartbeat",
                    stage="worker",
                    details={
                        "job_id": journal.job_id,
                        "state": current.state.value,
                        "heartbeat_at": (
                            current.heartbeat_at.isoformat()
                            if current.heartbeat_at is not None
                            else None
                        ),
                        "stale_after_seconds": self.heartbeat_stale_after,
                    },
                )

            # Capture the worker pid before starting the helper process.  The
            # heartbeat must identify the QAIRT worker, not its helper.
            touch = partial(journal.touch_heartbeat, pid=os.getpid())
            with HeartbeatWriter(
                touch,
                interval=self.heartbeat_interval,
                mode="process",
            ):
                return self._run_claimed()

    def _run_claimed(self) -> JobStatus:
        journal = self.journal
        try:
            journal.set_state(JobState.STAGING)
            journal.set_state(JobState.RUNNING)
            state_manifest = journal.state().manifest
            manifest: ArtifactRef | None = state_manifest or self.initial_manifest
            initial_manifest_pending_fork = (
                state_manifest is None and self.initial_manifest is not None
            )
            reused_manifest_pending_fork = False

            for stage_name in self.stages:
                if journal.cancel_requested():
                    return journal.set_state(JobState.CANCELLED)

                if initial_manifest_pending_fork and manifest is not None:
                    manifest, _ = self._fork_reused_manifest(
                        manifest,
                        next_stage=stage_name,
                        reason=f"initial_manifest_before_stage:{stage_name}",
                        source_job_id=journal.state().parent_job_id,
                    )
                    initial_manifest_pending_fork = False
                    self.publisher.commit(
                        journal,
                        manifest,
                        f"fork_initial_before_{stage_name}",
                    )

                key = self._stage_key(stage_name, manifest)
                reused = self._try_reuse(stage_name, key)
                if reused is not None:
                    journal.append_event(
                        "stage_reused",
                        {"stage_name": stage_name, "stage_key": key},
                    )
                    if reused.outputs:
                        manifest = reused.outputs[-1]
                        self.publisher.commit(journal, manifest, stage_name)
                        reused_manifest_pending_fork = (
                            bool(reused.metrics.get("reused_from_parent"))
                            and manifest.kind.value == "manifest"
                        )
                    continue

                if reused_manifest_pending_fork and manifest is not None:
                    manifest, _ = self._fork_reused_manifest(
                        manifest,
                        next_stage=stage_name,
                    )
                    reused_manifest_pending_fork = False
                    self.publisher.commit(
                        journal,
                        manifest,
                        f"fork_before_{stage_name}",
                    )
                    key = self._stage_key(stage_name, manifest)
                    resumed = journal.receipt_for_stage_key(key)
                    if resumed is not None:
                        journal.append_event(
                            "stage_reused",
                            {
                                "stage_name": stage_name,
                                "stage_key": key,
                                "reason": "resume_after_manifest_fork",
                            },
                        )
                        if resumed.outputs:
                            manifest = resumed.outputs[-1]
                            self.publisher.commit(journal, manifest, stage_name)
                        continue

                attempt = max(
                    self._next_attempt(stage_name, key),
                    self._manifest_attempt_floor(manifest, stage_name),
                )
                journal.append_event(
                    "stage_started",
                    {"stage_name": stage_name, "stage_key": key, "attempt": attempt},
                )
                context = StageContext(
                    stage_name=stage_name,
                    spec=self.spec,
                    resolved=self.resolved,
                    journal=journal,
                    current_manifest=manifest,
                    output_dir=self.output_dir / stage_name / key[:16] / f"attempt-{attempt:03d}",
                    attempt=attempt,
                )
                started = utc_now()
                try:
                    outcome = self.stage_runner(context)
                    self._verify_outputs(outcome)
                except JobCancelledError:
                    return journal.set_state(JobState.CANCELLED)
                except Exception as exc:  # noqa: BLE001 - record any stage failure
                    error = ToolErrorData.from_exception(exc, stage=stage_name)
                    journal.record_receipt(
                        StageReceipt(
                            stage_key=key,
                            stage_name=stage_name,
                            attempt=attempt,
                            status=StageStatus.FAILED,
                            started_at=started,
                            completed_at=utc_now(),
                            provenance=self.provenance,
                            error=error,
                        )
                    )
                    return journal.set_state(JobState.FAILED, error=error, current_stage=stage_name)

                journal.record_receipt(
                    StageReceipt(
                        stage_key=key,
                        stage_name=stage_name,
                        attempt=attempt,
                        status=StageStatus.SUCCEEDED,
                        started_at=started,
                        completed_at=utc_now(),
                        inputs=(manifest,) if manifest is not None else (),
                        outputs=self._receipt_outputs(outcome),
                        metrics=outcome.metrics,
                        provenance=self.provenance,
                    )
                )
                self.publisher.commit(journal, outcome.manifest, stage_name)
                if outcome.manifest is not None:
                    manifest = outcome.manifest
                elif outcome.artifacts:
                    manifest = outcome.artifacts[-1]

            if reused_manifest_pending_fork and manifest is not None:
                manifest, _ = self._fork_reused_manifest(
                    manifest,
                    next_stage=None,
                )
                self.publisher.commit(journal, manifest, "fork_after_reuse")

            journal.set_state(JobState.COLLECTING)
            journal.set_state(JobState.COMMITTING)
            return journal.set_state(JobState.SUCCEEDED, manifest=manifest)
        except Exception as exc:  # noqa: BLE001 - top-level guard
            error = ToolErrorData.from_exception(exc, stage="worker")
            current = journal.state()
            if current.state.terminal:
                return current
            return journal.set_state(JobState.FAILED, error=error)


__all__ = [
    "DEFAULT_WORKFLOW_STAGES",
    "ManifestPublisher",
    "StageContext",
    "StageOutcome",
    "StageRunner",
    "WorkflowWorker",
]
