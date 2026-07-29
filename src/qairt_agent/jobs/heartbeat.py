"""Independent heartbeat writer.

The heartbeat keeps updating the journal even while a long QAIRT Python call is
in progress, so an external observer can distinguish a busy worker from a dead
one.  Production workers use process mode because a QAIRT pybind call may hold
the GIL for many minutes; thread mode remains available for lightweight tests.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
from collections.abc import Callable
from typing import Literal


def _process_loop(
    touch: Callable[[], None],
    interval: float,
    stop: object,
    parent_pid: int,
) -> None:
    """Heartbeat child entry point (must remain module-level for ``spawn``)."""

    # ``multiprocessing.synchronize.Event`` is not a public typing surface.
    wait = getattr(stop, "wait")
    while os.getppid() == parent_pid:
        try:
            touch()
        except Exception:  # noqa: BLE001 - heartbeat must never raise
            pass
        if wait(interval):
            return


class HeartbeatWriter:
    """Periodically invoke ``touch`` until stopped.

    ``touch`` must be cheap and side-effecting (e.g. rewrite ``heartbeat.json``).
    Exceptions from ``touch`` are swallowed: a failing heartbeat must never
    crash the job it monitors.
    """

    def __init__(
        self,
        touch: Callable[[], None],
        interval: float = 5.0,
        *,
        mode: Literal["thread", "process"] = "thread",
        process_start_method: str = "spawn",
    ) -> None:
        if interval <= 0:
            raise ValueError("heartbeat interval must be positive")
        if mode not in {"thread", "process"}:
            raise ValueError("heartbeat mode must be 'thread' or 'process'")
        self._touch = touch
        self._interval = interval
        self._mode = mode
        self._process_start_method = process_start_method
        self._thread_stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process_stop: object | None = None
        self._process: multiprocessing.Process | None = None

    def start(self) -> "HeartbeatWriter":
        if self.running:
            return self
        if self._mode == "process":
            context = multiprocessing.get_context(self._process_start_method)
            self._process_stop = context.Event()
            self._process = context.Process(
                target=_process_loop,
                args=(
                    self._touch,
                    self._interval,
                    self._process_stop,
                    os.getpid(),
                ),
                name="qairt-heartbeat",
                daemon=True,
            )
            self._process.start()
        else:
            self._thread_stop.clear()
            try:
                self._touch()
            except Exception:  # noqa: BLE001 - heartbeat must never raise
                pass
            self._thread = threading.Thread(
                target=self._run,
                name="qairt-heartbeat",
                daemon=True,
            )
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._thread_stop.wait(self._interval):
            try:
                self._touch()
            except Exception:  # noqa: BLE001 - heartbeat must never raise
                pass

    def stop(self) -> None:
        if self._process is not None:
            assert self._process_stop is not None
            getattr(self._process_stop, "set")()
            self._process.join(timeout=self._interval + 1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process.close()
            self._process = None
            self._process_stop = None
        self._thread_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)
            self._thread = None

    @property
    def running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            or self._process is not None
            and self._process.is_alive()
        )

    def __enter__(self) -> "HeartbeatWriter":
        return self.start()

    def __exit__(self, *exc: object) -> bool:
        self.stop()
        return False


__all__ = ["HeartbeatWriter"]
