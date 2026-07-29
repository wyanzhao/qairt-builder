"""Asynchronous, journal-backed Python API.

``QairtAgentClient`` is the thin entry point Claude Code/Codex and the CLI use.
``prepare`` creates a persistent journal and returns immediately; ``run_job``
executes that journal's workflow in the current process (used by a detached
worker subprocess and by ``--follow``); ``submit`` is the convenience that
prepares and runs in a background thread for in-process callers.  ``rerun``
reuses unchanged stages from a parent job; ``resume`` continues an interrupted
job from its last verified receipt.  The synchronous
:class:`qairt_agent.pipeline.QairtAgent` remains the stage engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from qairt_agent.artifacts import canonical_json_bytes
from qairt_agent.contracts import ArtifactRef, BuildSpec, utc_now
from qairt_agent.contracts import (
    JobState,
    JobStatus,
    StageProvenance,
    WorkflowSpec,
    to_workflow_spec,
)
from qairt_agent.errors import InvalidSpecError, JobConflictError
from qairt_agent.families.presets import (
    resolve_workflow,
    to_build_spec,
)
from qairt_agent.harness import (
    DEFAULT_CONSTRAINTS,
    load_harness_constraints,
    use_harness_constraints,
)
from qairt_agent.jobs.engine_runner import EngineStageRunner
from qairt_agent.jobs.journal import IN_FLIGHT_JOB_STATES, JobJournal
from qairt_agent.jobs.worker import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_HEARTBEAT_STALE_AFTER,
    DEFAULT_WORKFLOW_STAGES,
    WorkflowWorker,
)

# Backwards-compatible exported snapshot. Runtime provenance is resolved from
# the active project harness in ``_provenance_for``.
PINNED_SDK_BUILD = DEFAULT_CONSTRAINTS.qairt_build_id


def new_job_id(now: datetime | None = None) -> str:
    timestamp = (now or utc_now()).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:8]}"


def spec_sha256(spec: WorkflowSpec) -> str:
    return hashlib.sha256(canonical_json_bytes(spec)).hexdigest()


class JobHandle:
    """A live handle to a journal-backed job."""

    def __init__(
        self,
        journal: JobJournal,
        *,
        thread: threading.Thread | None = None,
        client: "QairtAgentClient | None" = None,
    ) -> None:
        self._journal = journal
        self._thread = thread
        self._client = client

    @property
    def job_id(self) -> str:
        return self._journal.job_id

    @property
    def status_path(self) -> str:
        return str(self._journal.dir / "state.json")

    @property
    def journal(self) -> JobJournal:
        return self._journal

    def status(self) -> JobStatus:
        return self._journal.state()

    def events(self, after_seq: int = 0) -> list[dict[str, Any]]:
        return self._journal.events(after_seq=after_seq)

    def cancel(self) -> None:
        self._journal.request_cancel()

    def resume(self) -> "JobHandle":
        if self._client is None:
            raise JobConflictError("this handle is not bound to a client; cannot resume", stage="resume")
        return self._client.resume(self.job_id)

    def wait(self, timeout: float | None = None) -> JobStatus:
        if self._thread is not None:
            self._thread.join(timeout)
        return self._journal.state()

    def submission(self) -> dict[str, Any]:
        """The single-line submission report: ``job_id/state/status_path``."""

        status = self.status()
        return {
            "job_id": status.job_id,
            "state": status.state.value,
            "status_path": self.status_path,
        }


class QairtAgentClient:
    """Thin asynchronous facade over the journal and the synchronous engine."""

    def __init__(
        self,
        *,
        jobs_root: str | Path = ".qairt-agent/jobs",
        engine_factory: Any | None = None,
        background: bool = True,
        provenance: StageProvenance | None = None,
        harness_constraints: str | Path | None = None,
        heartbeat_interval: float | None = None,
        heartbeat_stale_after: float | None = None,
    ) -> None:
        self.jobs_root = Path(jobs_root).expanduser().resolve()
        self._engine_factory = engine_factory
        self.background = background
        self._provenance = provenance
        self._harness_constraints = (
            Path(harness_constraints).expanduser().resolve()
            if harness_constraints is not None
            else None
        )
        self.heartbeat_interval = (
            float(os.environ.get("QAIRT_AGENT_HEARTBEAT_INTERVAL", DEFAULT_HEARTBEAT_INTERVAL))
            if heartbeat_interval is None
            else heartbeat_interval
        )
        self.heartbeat_stale_after = (
            float(
                os.environ.get(
                    "QAIRT_AGENT_HEARTBEAT_STALE_AFTER",
                    DEFAULT_HEARTBEAT_STALE_AFTER,
                )
            )
            if heartbeat_stale_after is None
            else heartbeat_stale_after
        )
        if self.heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if self.heartbeat_stale_after <= 0:
            raise ValueError("heartbeat_stale_after must be positive")

    @classmethod
    def from_project(
        cls,
        project_root: str | Path,
        **kwargs: Any,
    ) -> "QairtAgentClient":
        """Construct a client bound to one initialized project's harness."""

        from qairt_agent import project

        config = project.load(project_root)
        jobs_root = kwargs.pop("jobs_root", config.jobs_path)
        if "harness_constraints" in kwargs:
            raise TypeError(
                "from_project owns harness_constraints; configure it in "
                "qairt-agent.toml"
            )
        return cls(
            jobs_root=jobs_root,
            harness_constraints=config.harness_path,
            **kwargs,
        )

    def _harness_scope(self) -> Any:
        if self._harness_constraints is None:
            return nullcontext()
        return use_harness_constraints(self._harness_constraints)

    # ------------------------------------------------------------------ #
    # spec handling
    # ------------------------------------------------------------------ #

    def _normalize_spec(self, spec: Any) -> WorkflowSpec:
        with self._harness_scope():
            return self._normalize_spec_active(spec)

    def _normalize_spec_active(self, spec: Any) -> WorkflowSpec:
        if isinstance(spec, WorkflowSpec):
            return WorkflowSpec.model_validate(
                spec.model_dump(mode="python")
            )
        if isinstance(spec, BuildSpec):
            return to_workflow_spec(
                BuildSpec.model_validate(spec.model_dump(mode="python"))
            )
        if isinstance(spec, (str, Path)):
            path = Path(spec).expanduser()
            if not path.exists():
                raise InvalidSpecError(f"spec file not found: {path}", stage="spec")
            return self._normalize_spec_active(
                json.loads(path.read_text(encoding="utf-8"))
            )
        if isinstance(spec, dict):
            if "preset" in spec:
                return WorkflowSpec.model_validate(spec)
            return to_workflow_spec(BuildSpec.model_validate(spec))
        raise InvalidSpecError(f"cannot interpret spec of type {type(spec).__name__}", stage="spec")

    def _default_engine(self) -> Any:
        if self._engine_factory is not None:
            return self._engine_factory()
        from qairt_agent.pipeline import QairtAgent

        return QairtAgent()

    def _provenance_for(self, resolved: Any) -> StageProvenance:
        if self._provenance is not None:
            return self._provenance.model_copy(
                update={"resolved_preset_sha256": resolved.resolved_preset_sha256}
            )
        with self._harness_scope():
            constraints = load_harness_constraints()
        return StageProvenance(
            sdk_build=constraints.qairt_build_id,
            adapter_capability="explicit_factory",
            platform_abi=(
                f"ubuntu{constraints.ubuntu_version}-"
                f"{constraints.platform_arch}"
            ),
            resolved_preset_sha256=resolved.resolved_preset_sha256,
            host_arch=platform.machine(),
        )

    def _make_runner(self, workflow_spec: WorkflowSpec, resolved: Any, engine: Any) -> EngineStageRunner:
        build_spec = to_build_spec(workflow_spec)
        validate_config: dict[str, Any] = {}
        if workflow_spec.quality.sqnr_modes:
            validate_config.update(
                {
                    "sqnr_modes": [
                        mode.value
                        for mode in workflow_spec.quality.sqnr_modes
                    ],
                    "dump_intermediates_on_failure": (
                        workflow_spec.quality.dump_intermediates_on_failure
                    ),
                }
            )
        validate_config.update(workflow_spec.stage_configs.validation)
        benchmark_config = workflow_spec.benchmark.model_dump(mode="json")
        benchmark_config.update(workflow_spec.stage_configs.benchmark)
        return EngineStageRunner(
            engine=engine,
            build_spec=build_spec.model_dump(mode="json"),
            pipeline=resolved.pipeline.value,
            # Build owns workflow-level vector preparation.  Validation must
            # consume the exact per-AR manifest published in runtime_index,
            # rather than re-injecting the source manifest (which may still be
            # AR2073).  Direct EngineStageRunner users retain the legacy
            # ``vector_manifest`` parameter; native workflows can make an
            # intentional override via stage_configs.validation.
            vector_manifest=None,
            build_config=dict(workflow_spec.stage_configs.build),
            validate_config=validate_config,
            benchmark_config=benchmark_config,
            diagnose_config=dict(workflow_spec.stage_configs.diagnose.config),
            diagnose_kind=workflow_spec.stage_configs.diagnose.kind.value,
        )

    def _build_worker(
        self,
        journal: JobJournal,
        workflow_spec: WorkflowSpec,
        resolved: Any,
        *,
        stages: tuple[str, ...],
        reuse_from: JobJournal | None = None,
        initial_manifest: ArtifactRef | None = None,
        engine: Any | None = None,
    ) -> WorkflowWorker:
        engine = engine if engine is not None else self._default_engine()
        runner = self._make_runner(workflow_spec, resolved, engine)
        effective_build_spec = to_build_spec(workflow_spec)
        return WorkflowWorker(
            journal,
            spec=workflow_spec.model_dump(mode="json"),
            resolved=resolved.to_dict(),
            provenance=self._provenance_for(resolved),
            stage_runner=runner,
            stages=stages,
            reuse_from=reuse_from,
            initial_manifest=initial_manifest,
            effective_build_spec=effective_build_spec,
            heartbeat_interval=self.heartbeat_interval,
            heartbeat_stale_after=self.heartbeat_stale_after,
        )

    # ------------------------------------------------------------------ #
    # prepare / run
    # ------------------------------------------------------------------ #

    def prepare(
        self,
        spec: Any,
        *,
        stages: tuple[str, ...] = ("build",),
        parent_job_id: str | None = None,
        initial_manifest_job: str | None = None,
        reuse_from_job: str | None = None,
    ) -> JobHandle:
        """Create a persistent journal for a job without launching it."""

        workflow_spec = self._normalize_spec(spec)
        resolved = resolve_workflow(workflow_spec)

        initial_manifest_dump: dict[str, Any] | None = None
        if initial_manifest_job is not None:
            parent = JobJournal.open(self.jobs_root, initial_manifest_job)
            manifest = parent.state().manifest
            if manifest is not None:
                initial_manifest_dump = json.loads(manifest.model_dump_json())
            parent_job_id = parent_job_id or initial_manifest_job

        job_id = new_job_id()
        journal = JobJournal.create(
            self.jobs_root,
            job_id,
            spec_original=json.loads(workflow_spec.model_dump_json()),
            spec_resolved=resolved.to_dict(),
            spec_sha256=spec_sha256(workflow_spec),
            launcher={
                "kind": "prepared",
                "pid": os.getpid(),
                "host_arch": platform.machine(),
                "stages": list(stages),
                "parent_job_id": parent_job_id,
                "initial_manifest": initial_manifest_dump,
                "reuse_from_job": reuse_from_job,
            },
            parent_job_id=parent_job_id,
        )
        return JobHandle(journal, client=self)

    def _worker_for(self, journal: JobJournal, *, engine: Any | None = None) -> WorkflowWorker:
        with self._harness_scope():
            return self._worker_for_active(journal, engine=engine)

    def _worker_for_active(
        self,
        journal: JobJournal,
        *,
        engine: Any | None = None,
    ) -> WorkflowWorker:
        workflow_spec = WorkflowSpec.model_validate(journal.spec_original())
        resolved = resolve_workflow(workflow_spec)
        launcher = journal.launcher()
        stages = tuple(launcher.get("stages") or DEFAULT_WORKFLOW_STAGES)

        initial_manifest: ArtifactRef | None = None
        manifest_dump = launcher.get("initial_manifest")
        if manifest_dump:
            initial_manifest = ArtifactRef.model_validate(manifest_dump)

        reuse_from: JobJournal | None = None
        reuse_id = launcher.get("reuse_from_job")
        if reuse_id:
            reuse_from = JobJournal.open(self.jobs_root, reuse_id)

        return self._build_worker(
            journal,
            workflow_spec,
            resolved,
            stages=stages,
            reuse_from=reuse_from,
            initial_manifest=initial_manifest,
            engine=engine,
        )

    def run_job(self, job_id: str, *, engine: Any | None = None) -> JobStatus:
        """Run a prepared job to completion in the current process."""

        with self._harness_scope():
            journal = JobJournal.open(self.jobs_root, job_id)
            worker = self._worker_for(journal, engine=engine)
            return worker.run()

    def _launch(self, journal: JobJournal, worker: WorkflowWorker) -> JobHandle:
        handle = JobHandle(journal, client=self)
        if not self.background:
            with self._harness_scope():
                worker.run()
            return handle

        def run_worker() -> None:
            with self._harness_scope():
                worker.run()

        thread = threading.Thread(
            target=run_worker,
            name=f"qairt-job-{journal.job_id}",
            daemon=True,
        )
        handle._thread = thread  # noqa: SLF001 - bound at construction
        thread.start()
        return handle

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def submit(
        self,
        spec: Any,
        *,
        stages: tuple[str, ...] = ("build",),
        from_job: str | None = None,
    ) -> JobHandle:
        handle = self.prepare(spec, stages=stages, initial_manifest_job=from_job)
        return self._launch(handle.journal, self._worker_for(handle.journal))

    def workflow(self, spec: Any) -> JobHandle:
        return self.submit(spec, stages=DEFAULT_WORKFLOW_STAGES)

    def rerun(
        self,
        from_job: str,
        spec: Any | None = None,
        *,
        stages: tuple[str, ...] | None = None,
    ) -> JobHandle:
        parent = JobJournal.open(self.jobs_root, from_job)
        if spec is None:
            workflow_spec = self._normalize_spec(parent.spec_original())
        else:
            workflow_spec = self._normalize_spec(spec)
        parent_stages = tuple(parent.launcher().get("stages") or DEFAULT_WORKFLOW_STAGES)
        handle = self.prepare(
            workflow_spec,
            stages=stages or parent_stages,
            parent_job_id=from_job,
            reuse_from_job=from_job,
        )
        return self._launch(handle.journal, self._worker_for(handle.journal))

    def _assert_resumable(self, journal: JobJournal) -> bool:
        """Return True if the job has work to do; raise for failed/cancelled."""

        status = journal.state()
        if status.state == JobState.SUCCEEDED:
            return False
        if status.state in {JobState.FAILED, JobState.CANCELLED}:
            raise JobConflictError(
                f"cannot resume job '{journal.job_id}' in terminal state '{status.state.value}'; "
                "use rerun",
                stage="resume",
                details={"job_id": journal.job_id, "state": status.state.value},
            )
        if status.state in IN_FLIGHT_JOB_STATES and not journal.heartbeat_stale(
            self.heartbeat_stale_after
        ):
            raise JobConflictError(
                f"job '{journal.job_id}' still has a fresh worker heartbeat",
                stage="resume",
                details={
                    "job_id": journal.job_id,
                    "state": status.state.value,
                    "heartbeat_at": (
                        status.heartbeat_at.isoformat()
                        if status.heartbeat_at is not None
                        else None
                    ),
                    "stale_after_seconds": self.heartbeat_stale_after,
                },
            )
        return True

    def execute(self, job_id: str) -> JobStatus:
        """Run a job synchronously in the current process (worker subprocess path)."""

        with self._harness_scope():
            journal = JobJournal.open(self.jobs_root, job_id)
            if not self._assert_resumable(journal):
                return journal.state()
            journal.clear_cancel()
            return self._worker_for(journal).run()

    def prepare_resume(self, job_id: str) -> JobHandle:
        """Validate and clear cancellation without starting a worker.

        Detached CLI callers use this boundary before launching the selected
        native/Docker worker.  Keeping preparation separate prevents a
        ``background=False`` control-plane client from executing QAIRT work in
        the host process before the detached worker is spawned.
        """

        journal = JobJournal.open(self.jobs_root, job_id)
        if self._assert_resumable(journal):
            journal.clear_cancel()
        return JobHandle(journal, client=self)

    def resume(self, job_id: str) -> JobHandle:
        handle = self.prepare_resume(job_id)
        if handle.status().state == JobState.SUCCEEDED:
            return handle
        return self._launch(handle.journal, self._worker_for(handle.journal))

    def job(self, job_id: str) -> JobHandle:
        return JobHandle(JobJournal.open(self.jobs_root, job_id), client=self)

    def list_jobs(self) -> list[str]:
        return JobJournal.list_jobs(self.jobs_root)


__all__ = ["JobHandle", "QairtAgentClient", "new_job_id", "spec_sha256"]
