"""Small process-sidecar used by :mod:`qairt_agent.device.lease`."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _write_heartbeat(path: Path, owner_token: str) -> None:
    path.write_text(f"{owner_token}:{time.time_ns()}\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 4:
        return 2
    path = Path(arguments[0])
    parent_pid = int(arguments[1])
    owner_token = arguments[2]
    interval = float(arguments[3])
    if interval <= 0:
        return 2
    while os.getppid() == parent_pid:
        try:
            _write_heartbeat(path, owner_token)
        except OSError:
            # A lease heartbeat is a safety mechanism.  A transient filesystem
            # error must not permanently disable it while the owner is alive.
            time.sleep(min(interval, 1.0))
            continue
        time.sleep(interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
