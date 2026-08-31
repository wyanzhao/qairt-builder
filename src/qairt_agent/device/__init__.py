"""ADB device transfer and lifecycle: config, client, leases, and health.

ADB is used only for moving files to/from a device and for device lifecycle.
A device is never auto-selected (``AdbConfig.from_env`` fails closed), and
cleanup only ever deletes one exact attempt directory under
``/data/local/tmp/qairt-agent/`` — never a broad recursive delete.
"""

from qairt_agent.device.adb import (
    REMOTE_ROOT,
    AdbClient,
    AdbConfig,
    AttemptSession,
    canonicalize_adb_server,
    parse_remote_attempt_dir,
    remote_attempt_dir,
)
from qairt_agent.device.doctor import device_doctor, device_gc, require_healthy
from qairt_agent.device.lease import DeviceLease, list_stale_leases
from qairt_agent.device.soc import SOC_VERIFICATION_SCHEMA, verify_device_soc
from qairt_agent.device.runtime import (
    ENV_ADB_CANONICAL_SERVER,
    ENV_LEASES_DIR,
    DeviceRuntime,
    DeviceStageSession,
)

__all__ = [
    "REMOTE_ROOT",
    "AdbClient",
    "AdbConfig",
    "AttemptSession",
    "DeviceLease",
    "DeviceRuntime",
    "DeviceStageSession",
    "ENV_LEASES_DIR",
    "ENV_ADB_CANONICAL_SERVER",
    "canonicalize_adb_server",
    "SOC_VERIFICATION_SCHEMA",
    "device_doctor",
    "device_gc",
    "verify_device_soc",
    "list_stale_leases",
    "parse_remote_attempt_dir",
    "remote_attempt_dir",
    "require_healthy",
]
