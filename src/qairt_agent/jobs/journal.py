"""Pure-file job journal.

Layout (no database)::

    <root>/<job_id>/
        spec.original.json     immutable original spec
        spec.resolved.json     immutable resolved workflow
        launcher.json          launcher provenance
        state.json             atomic JobStatus snapshot
        heartbeat.json         last heartbeat timestamp/pid
        cancel                 presence requests cancellation
        events/0000000001.json append-only, one event per sequence number
        receipts/<...>.json    immutable verified StageReceipt
        logs/<stage>.log       per-stage logs

``state.seq`` always equals the last appended event sequence number, so
``job watch --after-seq N`` can resume from the journal alone.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qairt_agent.artifacts import verify_artifact
from qairt_agent.contracts import utc_now
from qairt_agent.contracts import JobState, JobStatus, StageReceipt
from qairt_agent.errors import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    JobConflictError,
    JobNotFoundError,
)

_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
IN_FLIGHT_JOB_STATES = frozenset(
    {
        JobState.STAGING,
        JobState.RUNNING,
        JobState.COLLECTING,
        JobState.COMMITTING,
    }
)


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[key] = lock
        return lock


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


class JobJournal:
    """A persistent, file-backed journal for one workflow job."""

    def __init__(self, root: str | Path, job_id: str) -> None:
        self._root = Path(root).expanduser().resolve()
        self._job_id = job_id
        self._dir = self._root / job_id

    # ------------------------------------------------------------------ #
    # paths / lifecycle
    # ------------------------------------------------------------------ #

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def root(self) -> Path:
        return self._root

    @property
    def dir(self) -> Path:
        return self._dir

    @classmethod
    def create(
        cls,
        root: str | Path,
        job_id: str,
        *,
        spec_original: dict[str, Any],
        spec_resolved: dict[str, Any],
        spec_sha256: str,
        launcher: dict[str, Any] | None = None,
        parent_job_id: str | None = None,
    ) -> "JobJournal":
        journal = cls(root, job_id)
        if journal._dir.exists():
            raise JobConflictError(
                f"job '{job_id}' already exists",
                stage="journal",
                details={"job_id": job_id, "path": str(journal._dir)},
            )
        for sub in ("events", "receipts", "logs"):
            (journal._dir / sub).mkdir(parents=True, exist_ok=True)

        _atomic_write_json(journal._dir / "spec.original.json", spec_original)
        _atomic_write_json(journal._dir / "spec.resolved.json", spec_resolved)
        _atomic_write_json(journal._dir / "launcher.json", launcher or {})

        now = utc_now()
        status = JobStatus(
            job_id=job_id,
            state=JobState.QUEUED,
            seq=0,
            parent_job_id=parent_job_id,
            spec_sha256=spec_sha256,
            created_at=now,
            updated_at=now,
            launcher=launcher or {},
        )
        with journal._fs_lock():
            journal._commit_locked(
                status,
                "job_created",
                {
                    "job_id": job_id,
                    "parent_job_id": parent_job_id,
                    "spec_sha256": spec_sha256,
                },
                initial=True,
            )
        return journal

    @classmethod
    def open(cls, root: str | Path, job_id: str) -> "JobJournal":
        """Attach for mutation/worker execution and reconcile crash residue."""

        journal = cls(root, job_id)
        if not (journal._dir / "state.json").exists():
            raise JobNotFoundError(
                f"job '{job_id}' not found",
                stage="journal",
                details={"job_id": job_id, "path": str(journal._dir)},
            )
        with journal._fs_lock():
            journal._reconcile_locked()
        return journal

    @classmethod
    def open_readonly(
        cls,
        root: str | Path,
        job_id: str,
    ) -> "JobJournal":
        """Attach for status/event reads without locking or reconciliation."""

        journal = cls(root, job_id)
        if not (journal._dir / "state.json").exists():
            raise JobNotFoundError(
                f"job '{job_id}' not found",
                stage="journal",
                details={"job_id": job_id, "path": str(journal._dir)},
            )
        return journal

    @classmethod
    def exists(cls, root: str | Path, job_id: str) -> bool:
        return (Path(root).expanduser().resolve() / job_id / "state.json").exists()

    @classmethod
    def list_jobs(cls, root: str | Path) -> list[str]:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            return []
        return sorted(
            entry.name
            for entry in root_path.iterdir()
            if entry.is_dir() and (entry / "state.json").exists()
        )

    # ------------------------------------------------------------------ #
    # locking
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def _fs_lock(self) -> Iterator[None]:
        self._dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._dir / ".journal.lock"
        thread_lock = _thread_lock(lock_path)
        with thread_lock:
            with lock_path.open("a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextlib.contextmanager
    def worker_lease(self) -> Iterator[None]:
        """Hold the exclusive execution lease for this job.

        The file lock is released by the OS after a hard worker exit.  The
        companion in-process lock is needed because ``flock`` semantics are
        process-scoped on some supported hosts.
        """

        lock_path = self._dir / ".worker.lock"
        thread_lock = _thread_lock(lock_path)
        if not thread_lock.acquire(blocking=False):
            raise JobConflictError(
                f"job '{self._job_id}' already has an active worker",
                stage="worker",
                details={"job_id": self._job_id, "state": self.state().state.value},
            )
        handle = lock_path.open("a+")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise JobConflictError(
                    f"job '{self._job_id}' already has an active worker",
                    stage="worker",
                    details={"job_id": self._job_id, "state": self.state().state.value},
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            thread_lock.release()

    # ------------------------------------------------------------------ #
    # state + events
    # ------------------------------------------------------------------ #

    def _read_state(self) -> JobStatus:
        raw = json.loads((self._dir / "state.json").read_text(encoding="utf-8"))
        return JobStatus.model_validate(raw)

    def state(self) -> JobStatus:
        return self._read_state()

    def _event_paths_by_seq(self) -> dict[int, Path]:
        return {
            int(path.stem): path
            for path in (self._dir / "events").glob("*.json")
            if path.stem.isdigit()
        }

    def _reconcile_locked(self) -> JobStatus:
        """Repair an event-first interrupted commit and recover receipts."""

        current = self._read_state()
        event_paths = self._event_paths_by_seq()
        # State is the transaction commit point.  An event beyond state.seq
        # was durably written before a crash and is not yet authoritative.
        removed_tail = False
        for seq, path in event_paths.items():
            if seq > current.seq:
                path.unlink()
                removed_tail = True
        if removed_tail:
            directory_fd = os.open(self._dir / "events", os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        event_paths = self._event_paths_by_seq()
        missing_events = [
            seq for seq in range(1, current.seq + 1)
            if seq not in event_paths
        ]
        if missing_events:
            raise JobConflictError(
                f"job '{self._job_id}' journal is missing committed events",
                stage="journal",
                details={
                    "job_id": self._job_id,
                    "state_seq": current.seq,
                    "missing_sequences": missing_events,
                },
            )

        known = {
            (receipt.stage_key, receipt.attempt)
            for receipt in current.stages
        }
        recovered = [
            receipt
            for receipt in self.receipts()
            if (receipt.stage_key, receipt.attempt) not in known
        ]
        if recovered:
            target = current.model_copy(
                update={
                    "stages": current.stages + tuple(recovered),
                    "current_stage": recovered[-1].stage_name,
                }
            )
            current = self._commit_locked(
                target,
                "receipts_recovered",
                {
                    "receipts": [
                        {
                            "stage_name": item.stage_name,
                            "stage_key": item.stage_key,
                            "attempt": item.attempt,
                        }
                        for item in recovered
                    ]
                },
            )
        return current

    def _commit_locked(
        self,
        target: JobStatus,
        event_type: str,
        payload: dict[str, Any],
        *,
        initial: bool = False,
    ) -> JobStatus:
        current = None if initial else self._read_state()
        if current is not None:
            if current.state.terminal and target.state != current.state:
                raise JobConflictError(
                    f"cannot transition job '{self._job_id}' out of terminal "
                    f"state '{current.state.value}'",
                    stage="journal",
                    details={
                        "job_id": self._job_id,
                        "state": current.state.value,
                    },
                )
            if current.heartbeat_at is not None and (
                target.heartbeat_at is None
                or current.heartbeat_at > target.heartbeat_at
            ):
                target = target.model_copy(
                    update={"heartbeat_at": current.heartbeat_at}
                )
        seq = 1 if current is None else current.seq + 1
        now = utc_now()
        final = target.model_copy(update={"seq": seq, "updated_at": now})
        event = {
            "seq": seq,
            "type": event_type,
            "at": now.isoformat(),
            "payload": payload,
        }
        _atomic_write_json(
            self._dir / "events" / f"{seq:010d}.json",
            event,
        )
        _atomic_write_bytes(
            self._dir / "state.json",
            final.model_dump_json(indent=2).encode("utf-8"),
        )
        return final

    def append_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> JobStatus:
        with self._fs_lock():
            current = self._read_state()
            return self._commit_locked(
                current,
                event_type,
                payload or {},
            )

    def set_state(
        self,
        state: JobState,
        *,
        current_stage: str | None = None,
        error: Any | None = None,
        manifest: Any | None = None,
        heartbeat_at: datetime | None = None,
        event_payload: dict[str, Any] | None = None,
    ) -> JobStatus:
        with self._fs_lock():
            current = self._read_state()
            target = JobStatus(
                job_id=self._job_id,
                state=state,
                seq=current.seq,
                parent_job_id=current.parent_job_id,
                spec_sha256=current.spec_sha256,
                created_at=current.created_at,
                updated_at=current.updated_at,
                current_stage=(
                    current_stage
                    if current_stage is not None
                    else current.current_stage
                ),
                stages=current.stages,
                manifest=(
                    manifest if manifest is not None else current.manifest
                ),
                heartbeat_at=(
                    heartbeat_at
                    if heartbeat_at is not None
                    else current.heartbeat_at
                ),
                launcher=current.launcher,
                error=error,
            )
            payload = {
                "state": state.value,
                "current_stage": target.current_stage,
            }
            if event_payload:
                payload.update(event_payload)
            return self._commit_locked(
                target,
                "state_changed",
                payload,
            )

    def events(self, after_seq: int = 0) -> list[dict[str, Any]]:
        events_dir = self._dir / "events"
        if not events_dir.exists():
            return []
        result: list[tuple[int, dict[str, Any]]] = []
        for path in events_dir.glob("*.json"):
            if not path.stem.isdigit():
                continue
            seq = int(path.stem)
            if seq <= after_seq:
                continue
            result.append((seq, json.loads(path.read_text(encoding="utf-8"))))
        result.sort(key=lambda item: item[0])
        return [event for _, event in result]

    # ------------------------------------------------------------------ #
    # receipts
    # ------------------------------------------------------------------ #

    def _receipt_path(self, receipt: StageReceipt) -> Path:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in receipt.stage_name)
        return self._dir / "receipts" / f"{safe_name}-{receipt.stage_key[:16]}-{receipt.attempt:03d}.json"

    def record_receipt(self, receipt: StageReceipt) -> JobStatus:
        with self._fs_lock():
            path = self._receipt_path(receipt)
            payload = json.loads(receipt.model_dump_json())
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing != payload:
                    raise JobConflictError(
                        f"conflicting receipt for stage "
                        f"'{receipt.stage_name}' attempt {receipt.attempt}",
                        stage="journal",
                        details={"path": str(path)},
                    )
            else:
                _atomic_write_json(path, payload)

            current = self._read_state()
            already = any(
                item.stage_key == receipt.stage_key
                and item.attempt == receipt.attempt
                for item in current.stages
            )
            stages = (
                current.stages
                if already
                else current.stages + (receipt,)
            )
            target = current.model_copy(
                update={
                    "stages": stages,
                    "current_stage": receipt.stage_name,
                }
            )
            return self._commit_locked(
                target,
                "stage_receipt",
                {
                    "stage_name": receipt.stage_name,
                    "stage_key": receipt.stage_key,
                    "attempt": receipt.attempt,
                    "status": receipt.status.value,
                },
            )

    def receipts(self) -> list[StageReceipt]:
        receipts_dir = self._dir / "receipts"
        if not receipts_dir.exists():
            return []
        result: list[StageReceipt] = []
        for path in sorted(receipts_dir.glob("*.json")):
            result.append(StageReceipt.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        return result

    def verified_receipts(self) -> list[StageReceipt]:
        """Receipts that succeeded and whose output artifacts still verify."""

        verified: list[StageReceipt] = []
        for receipt in self.receipts():
            if not receipt.verified:
                continue
            try:
                for ref in receipt.outputs:
                    verify_artifact(ref)
            except (ArtifactIntegrityError, ArtifactNotFoundError):
                continue
            verified.append(receipt)
        return verified

    def receipt_for_stage_key(self, stage_key: str) -> StageReceipt | None:
        for receipt in self.verified_receipts():
            if receipt.stage_key == stage_key:
                return receipt
        return None

    def last_verified_receipt(self) -> StageReceipt | None:
        verified = self.verified_receipts()
        if not verified:
            return None
        return max(verified, key=lambda r: r.completed_at or r.started_at)

    # ------------------------------------------------------------------ #
    # heartbeat / cancel / logs / specs
    # ------------------------------------------------------------------ #

    def touch_heartbeat(
        self,
        pid: int | None = None,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Atomically refresh the liveness record while the job is nonterminal.

        Heartbeats do not append journal events: a 15--40 minute vendor call
        should not create hundreds of semantically empty event records.
        """

        at = now or utc_now()
        with self._fs_lock():
            current = self._read_state()
            if current.state.terminal:
                return False
            _atomic_write_json(
                self._dir / "heartbeat.json",
                {
                    "heartbeat_at": at.isoformat(),
                    "pid": pid if pid is not None else os.getpid(),
                },
            )
            refreshed = current.model_copy(update={"heartbeat_at": at, "updated_at": at})
            _atomic_write_bytes(
                self._dir / "state.json",
                refreshed.model_dump_json(indent=2).encode("utf-8"),
            )
        return True

    def heartbeat(self) -> dict[str, Any] | None:
        path = self._dir / "heartbeat.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def last_heartbeat_at(self) -> datetime | None:
        """Return the newest parseable heartbeat timestamp."""

        status = self.state()
        candidates: list[datetime] = []
        payload = self.heartbeat()
        if payload is not None:
            raw = payload.get("heartbeat_at")
            if isinstance(raw, str):
                with contextlib.suppress(ValueError):
                    candidates.append(datetime.fromisoformat(raw))
        if status.heartbeat_at is not None:
            candidates.append(status.heartbeat_at)
        if not candidates:
            return None
        normalized = [
            value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            for value in candidates
        ]
        return max(normalized)

    def heartbeat_stale(
        self,
        stale_after: float,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Whether an in-flight job has missed its liveness deadline."""

        if stale_after <= 0:
            raise ValueError("heartbeat stale_after must be positive")
        status = self.state()
        if status.state not in IN_FLIGHT_JOB_STATES:
            return False
        last = self.last_heartbeat_at() or status.updated_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        current = now or utc_now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return (current - last).total_seconds() >= stale_after

    def mark_orphaned_if_stale(
        self,
        stale_after: float,
        *,
        now: datetime | None = None,
    ) -> JobStatus:
        """Transition a stale in-flight job to ``ORPHANED``.

        Callers must hold :meth:`worker_lease` while invoking this method, so a
        live worker cannot race the recovery transition.
        """

        status = self.state()
        if status.state not in IN_FLIGHT_JOB_STATES:
            return status
        if not self.heartbeat_stale(stale_after, now=now):
            return status
        previous_state = status.state
        heartbeat_at = self.last_heartbeat_at()
        return self.set_state(
            JobState.ORPHANED,
            current_stage=status.current_stage,
            event_payload={
                "previous_state": previous_state.value,
                "last_heartbeat_at": (
                    heartbeat_at.isoformat() if heartbeat_at is not None else None
                ),
                "stale_after_seconds": stale_after,
            },
        )

    def request_cancel(self) -> JobStatus:
        (self._dir / "cancel").write_text(utc_now().isoformat(), encoding="utf-8")
        return self.append_event("cancel_requested", {"at": utc_now().isoformat()})

    def cancel_requested(self) -> bool:
        return (self._dir / "cancel").exists()

    def clear_cancel(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            (self._dir / "cancel").unlink()

    def write_log(self, stage_name: str, line: str) -> None:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in stage_name)
        log_path = self._dir / "logs" / f"{safe_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line if line.endswith("\n") else line + "\n")

    def read_log(self, stage_name: str) -> str:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in stage_name)
        log_path = self._dir / "logs" / f"{safe_name}.log"
        if not log_path.exists():
            return ""
        return log_path.read_text(encoding="utf-8")

    def spec_original(self) -> dict[str, Any]:
        return json.loads((self._dir / "spec.original.json").read_text(encoding="utf-8"))

    def spec_resolved(self) -> dict[str, Any]:
        return json.loads((self._dir / "spec.resolved.json").read_text(encoding="utf-8"))

    def launcher(self) -> dict[str, Any]:
        path = self._dir / "launcher.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["JobJournal"]
