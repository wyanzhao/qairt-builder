"""Verify that the attached handset really is the resolved target's SoC.

Every ``harness/targets/<name>.json`` records an Android ``soc_id`` list
precisely so this comparison is possible -- the SM8750 handset on hand reports
``soc_id 618`` -- yet nothing read it, so a run could publish a report under a
target identity the hardware contradicts.

The two numbering schemes stay strictly apart: ``soc_model`` is the
``Qnn_SocModel_t`` value the compiler consumes and is never compared here;
``soc_id`` is what the device reports and is never passed to the SDK.

The policy is fail closed on contradiction, warn on absence. An old or
locked-down Android that will not report its SoC is a gap in what we can see,
not evidence of the wrong chip; a device that reports a number outside the
registry list is evidence of the wrong chip.
"""

from __future__ import annotations

from typing import Any

from qairt_agent.errors import DeviceUnavailableError
from qairt_agent.harness import TargetEntry, acceptance_target_name

SOC_VERIFICATION_SCHEMA = "qairt-agent.device-soc-verification/1"


def verify_device_soc(
    client: Any,
    entry: TargetEntry,
    *,
    serial: str,
    acceptance: bool | None = None,
) -> dict[str, Any]:
    """Compare the handset's reported ``soc_id`` with the registry entry.

    ``acceptance`` defaults to whether ``QAIRT_AGENT_TARGET_ACCEPTANCE`` names
    this target. A qualifying run downgrades a contradiction to a recorded
    warning for exactly the same reason it may run an unverified target: the
    entry's ``soc_id`` list is one of the things that run exists to confirm.
    """

    qualifying = (
        acceptance_target_name() == entry.name
        if acceptance is None
        else bool(acceptance)
    )
    reading = client.read_soc_id()
    observed = reading.get("soc_id")
    expected = list(entry.soc_id)
    record: dict[str, Any] = {
        "schema": SOC_VERIFICATION_SCHEMA,
        "serial": serial,
        "target": entry.name,
        "expected_soc_id": expected,
        "observed_soc_id": observed,
        "sources": list(reading.get("sources", ())),
        "acceptance_run": qualifying,
    }

    if observed is None:
        record["status"] = "unreadable"
        record["warning"] = (
            f"{serial} did not report a usable soc_id; the target identity "
            f"{entry.name} could not be confirmed against the handset"
        )
        return record
    if not expected:
        record["status"] = "not_recorded"
        record["warning"] = (
            f"target {entry.name} records no soc_id list, so the handset's "
            f"reported soc_id {observed} could not be checked"
        )
        return record
    if observed in expected:
        record["status"] = "verified"
        return record

    record["status"] = "contradicted"
    message = (
        f"attached device {serial} reports soc_id {observed}, which is not in "
        f"the registered soc_id list {expected} for target {entry.name} "
        f"({entry.tuple_text})"
    )
    if qualifying:
        record["status"] = "contradicted_acceptance_override"
        record["warning"] = message + "; allowed because this run is qualifying the target"
        return record
    raise DeviceUnavailableError(
        message,
        stage="device",
        retryable=False,
        details=record,
    )
