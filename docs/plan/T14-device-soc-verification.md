# T14 — Attached-device SoC verification

Status: done (2026-08-30)
Depends on: —
Effort: S

## Goal

Before any device stage, verify that the handset behind
`QAIRT_AGENT_ADB_SERIAL` actually is the resolved target's SoC, using the
Android `soc_id` list the registry already carries. A report must never
publish under a target identity the hardware contradicts.

## Context

Review finding 8 in
[review-findings-2026-08-30.md](review-findings-2026-08-30.md). Each
`harness/targets/<name>.json` records the Android `soc_id` list precisely so
this check is possible (the SM8750 handset reports `soc_id 618`), but no code
reads the device's SoC, and the device doctor's target check is unconditional
(`device/doctor.py:193`). The `soc_model` vs `soc_id` distinction is already
kept straight everywhere — this task only adds the missing read-and-compare.

## Design

1. Add a device-side read to the ADB layer: `getprop ro.soc.id` /
   `ro.soc.model`, falling back to `/sys/devices/soc0/soc_id`. Pure adb
   read, no SDK involvement.
2. Compare against the resolved target's `soc_id` list at the first device
   touch of a stage (validation/benchmark device setup, and `device doctor`).
   Mismatch fails closed naming serial, reported soc_id, target name, and the
   registry list. An unreadable value (old Android, restricted prop) is a
   recorded warning with the raw outputs kept, not a failure — fail closed on
   contradiction, warn on absence.
3. `QAIRT_AGENT_TARGET_ACCEPTANCE=<name>` also downgrades this check to a
   recorded warning for the qualifying run, mirroring the unverified-target
   escape hatch, because a new target's registry entry may need its soc_id
   list confirmed by exactly this run.
4. Replace the doctor's tautological target check with this real one; record
   the observed soc_id in the device section of run manifests/receipts.

## Files

`src/qairt_agent/device/adb.py`, `src/qairt_agent/device/doctor.py`,
`src/qairt_agent/device/runtime.py` (or the device-setup site in
`pipeline.py`), `src/qairt_agent/harness.py` (expose the soc_id list on the
resolved target), tests (`test_device.py`, `test_harness.py`),
`.claude/skills/qairt-add-target/SKILL.md` (the acceptance run confirms the
soc_id list), `CLAUDE.md` device-work note.

## Acceptance criteria

- With a fake adb reporting a soc_id in the registry list, device stages
  proceed and the manifest records the observed value (test).
- With a fake adb reporting a soc_id not in the list, the stage fails closed
  before any SDK call, naming both sides (test).
- Unreadable soc_id → recorded warning, stage proceeds (test).
- Acceptance-mode behavior tested (warning, not failure).
- `device doctor` reports the real comparison, not `ok=True` unconditionally.

## Result

Landed 2026-08-30. `AdbClient.read_soc_id` reads `ro.soc.id`, `ro.soc.model`,
then `/sys/devices/soc0/soc_id` — a pure adb read — and keeps every raw source
output so an unreadable value stays inspectable. The new
`device/soc.verify_device_soc` compares it with the resolved registry entry:
contradiction raises `DeviceUnavailableError` (non-retryable) naming serial,
observed id, target, and list; unreadable and empty-list are recorded warnings;
`QAIRT_AGENT_TARGET_ACCEPTANCE` downgrades a contradiction for the qualifying
run. `DeviceRuntime.stage` runs it immediately after building the adb client —
before the lease, before any push, before `create_device` — and publishes the
record on `DeviceStageSession.soc_verification`; `pipeline._device_stage` passes
the entry resolved from the spec's target, and each device stage's metadata now
carries `device_soc` beside `device_identifier`. The doctor's unconditional
`target_resolved` check is joined by a real `device_soc` check that fails on a
contradiction and passes-with-message on absence.

Eight tests cover it: the property read and its recorded sources, verified /
contradicted / unreadable / acceptance-override, DeviceRuntime proving the check
runs before the device is constructed and no lease is left, and the doctor
comparison. Suite 577 passed / 2 skipped, compileall clean.

Not verified on hardware in this session (no handset attached); the code path is
covered by fakes end to end.
