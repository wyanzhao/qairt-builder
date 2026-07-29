"""Persistent, file-based job journal and async orchestration.

The journal (``.qairt-agent/jobs/<job-id>/``) is the single source of truth for
a workflow job: immutable original/resolved specs, an append-only event log, an
atomic ``state.json``, immutable verified stage receipts, a heartbeat, and
logs.  No database is used.
"""

from qairt_agent.jobs.heartbeat import HeartbeatWriter
from qairt_agent.jobs.journal import JobJournal
from qairt_agent.jobs.keys import compute_stage_key, hash_inputs
from qairt_agent.jobs.worker import (
    DEFAULT_WORKFLOW_STAGES,
    ManifestPublisher,
    StageContext,
    StageOutcome,
    WorkflowWorker,
)

__all__ = [
    "DEFAULT_WORKFLOW_STAGES",
    "HeartbeatWriter",
    "JobJournal",
    "ManifestPublisher",
    "StageContext",
    "StageOutcome",
    "WorkflowWorker",
    "compute_stage_key",
    "hash_inputs",
]
