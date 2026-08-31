---
name: qairt-add-target
description: Add or qualify a Qualcomm target (chipset, DSP arch, soc_model) in the reviewed target registry, including the acceptance run that makes it verified. Use when adding hardware support, when a spec is refused for an unregistered target, or when a target plans but device stages refuse it.
---

# Add a target

Targets live in a reviewed registry: one `harness/targets/<name>.json` per
target. `harness/constraints.json` only names which one is active.

## The numbering trap

`soc_model` is the **`Qnn_SocModel_t`** value the compiler consumes. It is a
different scheme from the Android `soc_id` a device reports. SM8850 is
`soc_model 87` and reports `soc_id 660`; SM8750 is `soc_model 69` and reports
`soc_id 618`/`639`. Conflating the two is what made an earlier pin wrong in this
repository. Record both: `soc_model` plus the `soc_id` list.

## Entry shape

`chipset`, `dsp_arch`, `soc_model`, the Android `soc_id` list, and a `verified`
block recording the real-device acceptance run that qualified it.

## Selecting a target from a spec

By `name`, or by supplying the complete `chipset`/`dsp_arch`/`soc_model` tuple,
which is accepted only on an exact match with a registered entry. A partial
tuple is never completed implicitly. There is no built-in default.

## The chicken-and-egg, and its one exception

A target with no `verified` block plans fine but build and device stages refuse
it — it has never been proven on hardware. Since a target cannot become
verified without a run, and a run is refused while unverified, the qualifying
run is the documented exception:

```bash
QAIRT_AGENT_TARGET_ACCEPTANCE=<name> qairt-agent workflow --spec <spec>
```

Set it for that run only, then record the outcome in the registry entry:
SDK build id, date, the device string, and how it was qualified.

The acceptance run is also what confirms the `soc_id` list. Every device stage
compares the handset's reported `soc_id` against the entry's list and fails
closed on a contradiction — but for the qualifying run that check is downgraded
to a recorded warning, exactly because the list is one of the things being
confirmed. Read the run's `device_soc` record (or `qairt-agent device doctor`),
and copy the observed id into the entry before marking it verified. Leaving the
list wrong means every later run either refuses the handset or, if the list is
empty, records `not_recorded` and proves nothing.

`python tools/make_smoke_fixture.py --target <name>` generates a runnable
fixture for the target, so qualification needs no proprietary model.

## SM8750 needs care

QAIRT's own compile default is `v79` / `soc_model 69` — exactly the SM8750
tuple — so a resolved-value check cannot distinguish an intended target from a
silent fallback. An empty `device_custom_configs` list (the SDK's "skipping
device config creation" path) therefore fails closed in its own right,
whichever target was named. Do not weaken that guard.

## The steps that are easy to miss

1. **Deployment cell.** If the target is meant to be launched from, add
   `configs/{preset}/{name}.json`. A test resolves every cell, requires the
   directory/file names to agree with the preset and target inside, and
   requires that target to be verified — so a cell for an unqualified target
   fails the suite while a registry entry alone does not.
2. **Packaging.** The wheel ships `harness/targets` as a directory, so a new
   entry is included automatically. It was a per-file list, and adding a target
   used to ship a wheel silently missing it; `tests/test_packaging.py` keeps
   that from coming back.
3. **The registry tests are structural.** They check that names agree with
   filenames and that no two entries share a tuple, and they require a
   `verified` block only for the *active* target and any target a `configs/`
   cell deploys. An entry committed ahead of its acceptance run passes.

## Verify

`qairt-agent plan` renders the resolved target under `effective_target`,
including whether it is verified. A test resolves every deployment config and
requires its directory/file names to agree with the preset and target inside.
