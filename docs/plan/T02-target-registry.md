# T02 — Target registry (SM8850, SM8750)

Status: done (2026-08-30) — registry landed with both targets verified on real
hardware. The SM8850 `soc_model` question is settled by a device run: 87 works,
660 was the Android SoC ID.
Depends on: T01
Effort: M

## Goal

Replace the single hard-pinned HTP target with a **reviewed named target
registry** seeded with SM8850 and SM8750. A spec selects a target by name;
unregistered or mismatched targets fail closed. No default target, no
fallback.

## Context

- Today `harness/constraints.json` pins `SM8850 / v81 / soc_model 660` and
  preflight/adapter enforce exactly that tuple
  (`qairt_adapter/preflight.py:24-30`, `adapter.py:442-455`); the compile path
  re-validates the resolved config to refuse the SDK's silent v79 fallback
  (`adapter.py:966-978`).
- Decision: hardware must be configurable (SM8850 and SM8750 initially), while
  keeping the fail-closed discipline. Stage keys already fold in a
  device/runtime fingerprint, so cross-target cache reuse is already
  invalidated.

## Verified SoC numbering (2026-08-29, read from QAIRT 2.49.0.260730)

The task asked for the provisional SM8750 tuple to be checked against the SDK.
Doing that first surfaced a discrepancy in the **existing** SM8850 pin, so this
section is the evidence base for seeding the registry.

Two different numbering schemes exist and are easy to confuse:

| Scheme | Source in the SDK | SM8850 | SM8750 |
| --- | --- | --- | --- |
| `Qnn_SocModel_t` enum | `include/QNN/QnnTypes.h` | **87** | **69** |
| Android SoC ID | `qti/aisw/tools/core/utilities/devices/android/android_device_constants.py` | 660 | 618 and 639 |
| DSP architecture | `qti/aisw/converters/common/backend_aware_configs/htp_v2.json` (`soc_model_to_arch`) | `v81` | `v79` |

`include/QNN/HTP/QnnHtpDevice.h:60` documents the device field as
`uint32_t socModel; // An enum value defined in Qnn Header that represent SoC
model`, i.e. `Qnn_SocModel_t`. The SDK's own Python default confirms the same
scheme: `qairt/api/compiler/config.py:615` prints "Defaulting to dsp_arch: v79
and soc_model: 69", and v79 + 69 is exactly SM8750. `HtpDeviceConfig`
(`qairt/api/common/backends/htp/config.py:177-191`) carries `soc_id` and
`soc_model` as **separate** fields.

Consequences:

- The provisional SM8750 tuple in the original task (`v79`, `soc_model 69`) is
  **correct**.
- The repository's current pin `SM8850 / v81 / soc_model 660` uses SM8850's
  **Android SoC ID** where the QNN SoC model belongs; the QNN value is **87**.
  This is a suspected pre-existing defect, not a T02 design choice.

This was **not** changed in place. Changing a compiled-artifact target value on
static evidence alone is exactly the kind of edit that must be proven on
hardware first: a wrong `soc_model` can produce a context binary that loads and
runs while being compiled for the wrong SoC. The registry work must therefore:

1. carry `soc_model` (QNN enum) and `soc_id` (Android, may be a list) as
   distinct fields, so the two schemes can never be conflated again;
2. seed `sm8850` with `soc_model 87` / `soc_id [660]` and `sm8750` with
   `soc_model 69` / `soc_id [618, 639]`, both **unverified**, so device stages
   refuse them until a real run confirms;
3. record in the `verified` block which value the accepted device run actually
   compiled and executed with — that run, not this document, settles it.

If the device run shows 660 is what SM8850 accepts, the finding is recorded and
the seed corrected; either way the ambiguity stops being invisible.

## Interim state after T01 (2026-08-29)

T01 needed a device, and the device available is SM8750, so the single pinned
target moved to **SM8750 / v79 / soc_model 69** — using the `Qnn_SocModel_t`
value the table above establishes. SM8850 is not buildable until this task
lands its registry. Three things T02 must absorb:

- **The `soc_model` correction is now live for SM8750 only.** SM8850's entry
  must be seeded as `v81 / soc_model 87`, not the old 660, and that value has
  never been proven on hardware.
- **`TargetSpec` now defaults from and validates against
  `harness/constraints.json`,** so a spec naming a different tuple fails at
  spec time. The registry has to replace that single-tuple check with a
  named-target lookup without losing the early failure.
- **`qairt-agent.toml` carries its own `[target]` block** that duplicates the
  harness tuple; today doctor compares them and fails closed on drift. With a
  registry the project config should select a target *name* instead of
  restating the tuple, removing the duplication rather than doubling it.

Also landed in T01 and worth keeping: `_validate_compiler_target` now fails
closed on an empty `device_custom_configs` list, because QAIRT's own compile
default (`v79`/`soc_model 69`) is exactly the SM8750 tuple, so the old
"resolved value must not be v79" heuristic can no longer distinguish an
intended target from a silent fallback. Any registry entry whose tuple happens
to match an SDK default inherits that same hazard.

## Design

1. **Registry** — `harness/targets/<name>.json`, one reviewed file per target:

   ```json
   {
     "name": "sm8850",
     "chipset": "SM8850",
     "dsp_arch": "v81",
     "soc_model": 87,
     "soc_id": [660],
     "verified": {"sdk_build": "260730134355", "date": "...", "how": "..."}
   }
   ```

   Seed `sm8850.json` and `sm8750.json` from the "Verified SoC numbering"
   table above — `soc_model` is the `Qnn_SocModel_t` value (87 / 69) and
   `soc_id` is the separate Android SoC ID list (`[660]` / `[618, 639]`).
   A registry entry without a `verified` block is loadable for `plan` but
   rejected by preflight for build/device stages, which is where both seeds
   start.
2. **Spec surface** — `target` selects a registry name (keep the resolved
   `chipset/dsp_arch/soc_model` fields as the normalized output in manifests
   and plan output, so downstream consumers are unchanged). Compatibility: an
   inline full tuple may remain accepted only if it exactly matches a
   registered target; anything else is a structured preflight error.
3. **Constraints file** — remove the `target` block from
   `harness/constraints.json` (it stays the source of truth for SDK/image/ABI
   only). Update the root `CLAUDE.md` sentence accordingly.
4. **Guard rework** — the "no V79 fallback" rule becomes "no target other than
   the exact registered tuple the spec named": v79 is legal when the spec
   names `sm8750`, and a resolved compile config that disagrees with the named
   target still fails closed (keep `_validate_compiler_target`, parameterized).
5. **Per-target acceptance** — a target's capability claims (build, validate,
   benchmark) require at least one real-device acceptance run on that target;
   record it in the registry `verified` block. Device selection stays via
   `QAIRT_AGENT_ADB_SERIAL`/`QAIRT_AGENT_ADB_SERVER` per run.
6. **Provenance** — the registry entry (name + tuple + file SHA) enters stage
   provenance beside the existing device fingerprint.

## Files

`harness/targets/*.json` (new), `contracts.py` (`TargetSpec`),
`qairt_adapter/preflight.py`, `qairt_adapter/adapter.py` (target validation
sites above), `project.py`/`harness.py` (registry loading),
`pipeline.py` plan output, `device/doctor.py`, examples, tests
(`test_contracts.py`, `test_harness.py`, `test_sdk_adapter.py`,
`test_presets.py`), root `CLAUDE.md`, `docs/architecture.md`.

## Out of scope

- Auto-detecting the target from the connected device (explicit selection
  only; a mismatch between claimed target and device metadata may WARN-fail
  per preflight design, never auto-correct).
- Any non-HTP backend.

## Acceptance criteria

- Both seed targets load; an unregistered name and a mismatched inline tuple
  each produce a structured preflight error (tests).
- SM8750 tuple verified against SDK metadata and marked `verified`, or the
  entry remains unverified and preflight for device stages rejects it — no
  silently trusted provisional numbers. (Static SDK verification is done; the
  `verified` block still requires a device run.)
- The `soc_model` / `soc_id` split is explicit in the schema, in plan output,
  and in at least one test, so the two numbering schemes cannot be conflated
  again.
- The SM8850 `soc_model` discrepancy (87 vs the current 660) is settled by a
  real-device run and the outcome recorded here, not assumed either way.
- `qairt-agent plan` renders the resolved tuple for both targets; full test
  suite + compileall + doctor pass.
- Real-device acceptance run recorded for each target that claims support
  (SM8850 required now; SM8750 when its device is available — until then its
  registry entry stays unverified and that state is visible in `doctor`).
- `CLAUDE.md`, `docs/architecture.md`, and examples updated together.


## Result (2026-08-30)

### The soc_model question is settled on hardware

The SM8850 handset (`PNM-AN20`) reports `ro.soc.model=SM8850` with
`soc_id=660`, and a context compiled with **`soc_model 87`** for `v81` loaded
and executed on it: build, validate and benchmark all succeeded, SQNR 41.63 dB
against the supplied golden, manifest
`453248bb-2ee8-43c8-bb58-3d2f5319b59e` revision 3. Both numbering schemes are
now confirmed from two directions — the SM8750 device reports `soc_id 618` and
runs with `soc_model 69`, the SM8850 device reports `soc_id 660` and runs with
`soc_model 87` — so the earlier `SM8850 / 660` pin was conflating them.

### What landed

- `harness/targets/sm8850.json` and `harness/targets/sm8750.json`, each with
  `chipset`, `dsp_arch`, `soc_model` (Qnn enum), `soc_id` (Android, a list),
  `notes`, and a `verified` block naming the run that qualified it. Both are
  verified.
- `harness/constraints.json` no longer holds a tuple; it names the active
  target (`"target": {"name": "sm8850"}`). The registry travels with the
  constraints file: `install_default_constraints` copies both, the worker image
  `COPY`s both, and the wheel force-includes both — a project that has one
  without the other cannot resolve a target, so they are never separated.
- `TargetSpec` resolves a `name`, or a complete tuple that matches a registered
  entry exactly. A partial tuple is refused rather than completed implicitly,
  and registry errors surface as ordinary pydantic validation errors so a bad
  target reads like any other bad field.
- `PINNED_TARGET_SOC`/`PINNED_DSP_ARCH`/`PINNED_SOC_MODEL` are gone. The
  adapter's `_validate_target` returns the resolved registry entry and
  `compile_context` now builds its `soc_details`, weight-sharing mode and
  result artifact from **its own target argument** instead of a module
  constant — which also removes a latent inconsistency, since it previously
  validated the caller's target and then compiled with the pinned one. Both
  GenAI lanes take an explicit `target`, so which SoC a container was built for
  is an argument of the call and appears in its build report.
- `_validate_compiler_target` compares against the named target rather than a
  pin, and still fails closed on an empty `device_custom_configs` list.

### The chicken-and-egg the design had to answer

A target is refused until it is verified, and it cannot be verified without a
run. Left there, no target could ever be added. The qualifying run is therefore
one explicit, named exception: `QAIRT_AGENT_TARGET_ACCEPTANCE=<name>` permits
build/device stages for that one target, preflight downgrades its
`target.unverified` issue to a warning stating the run is qualifying, and the
CLI forwards the variable into the worker. It cannot be switched on by
accident, and it never covers a second target.

This was exercised for real: SM8850 was seeded **unverified**, the suite went
red on precisely the guard (`target 'sm8850' ... has no verified block ... build
and device stages are refused`), the acceptance run was made under the
qualifying flag, and the suite went green once the `verified` block recorded it.

### Deviation from the design, and why

Design item 3 said to remove the `target` block from
`harness/constraints.json` entirely. It keeps a **name-only** selector instead.
The tuple — the thing that was actually hard-pinned — is gone, but something
must still say which reviewed target this checkout builds for by default, and
constraints.json is already the file an upgrade patch touches. Putting the
default in code would have been a built-in default, which the task forbids;
requiring every spec and test to name a target would have made the default
implicit in whoever wrote the spec. A named selector keeps the choice explicit,
reviewed, and in one place.

Tests: `test_target_registry_separates_the_two_soc_numbering_schemes`,
`test_unregistered_target_and_unregistered_tuple_fail_closed` (which asserts
the old 660 tuple is refused), `test_unverified_target_is_refused_until_an_
acceptance_run`, `test_both_seeded_targets_record_a_real_device_acceptance_run`,
`test_target_spec_selects_a_registered_target_by_name`,
`test_target_spec_accepts_only_a_tuple_that_matches_a_registered_target`, and
`test_standalone_vit_keeps_diagnostic_outputs_out_of_production`, which now
proves a spec naming `sm8750` compiles for sm8750 while the harness names
sm8850.

### Still open

- Per-target acceptance is recorded for both targets on QAIRT 2.49.0.260730.
  A future SDK invalidates both `verified` blocks; the upgrade procedure should
  re-run acceptance per target rather than carrying the old records forward.
- `qairt-agent.toml` still restates the resolved tuple beside the name so
  `doctor` can detect drift between project config and harness. That is a
  deliberate cross-check, not the old duplication, but it is worth revisiting
  if a project ever needs to select a target per run.
