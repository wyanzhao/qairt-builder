---
name: qairt-sdk-upgrade
description: Upgrade the pinned QAIRT SDK version in this repository, including the signature probe, worker image rebuild, and the acceptance gates that must pass before any capability claim is updated. Use when moving to a new QAIRT release or when doctor reports an SDK version mismatch.
---

# Upgrade the pinned QAIRT SDK

`harness/constraints.json` is the reviewed source of truth for the QAIRT
version/build, Ubuntu image, Python ABI, worker image, runtime CLI versions,
dependency lock path, Torch version, and the *name* of the active target.

**Do not relax a version mismatch into a warning.** Until a new SDK is proven,
preflight must fail closed. Do not override individual pins ad hoc; a project
may select a different reviewed constraints file with
`QAIRT_AGENT_HARNESS_CONSTRAINTS`.

## Ordered procedure

1. Change `harness/constraints.json` in one reviewable patch.
2. Add or rename the pinned dependency file referenced by
   `worker.dependencies_file`. **Do not silently reuse an old release lock** —
   re-derive it from the new SDK's own dependency check and verify item by item.
   Renaming it is a **two-file change**: `pyproject.toml` force-includes the
   lock by literal filename, so the new name must go there too or the wheel
   ships without it. `tests/test_packaging.py` reads
   `worker.dependencies_file` and fails naming `pyproject.toml` when the two
   disagree — run the suite before assuming the rename is complete.
3. Update Docker / Apple-container image inputs only through the harness values.
4. Run `tools/sdk_signature_probe.py` **inside the worker container**, where the
   SDK imports. It fails naming any bound surface the new build dropped. Update
   family capability tests for the new build.
5. Run the complete gate set: `.venv/bin/pytest -q`,
   `.venv/bin/python -m compileall -q src tests`, `qairt-agent doctor`, the
   worker SDK import smoke, and **at least one real-device golden/latency
   acceptance run**.
6. Update examples and capability claims only after those gates pass.

## Per-model knowledge is sourced, not copied

Whatever the GenAI Builder already knows about a family is read from the SDK at
build time, so an upper-layer change cannot leave a stale duplicate here:

- MHA2SHA start points come from the SDK's own family builder; the profile
  stores only where to find them plus a reviewed fingerprint. **A fingerprint
  mismatch fails closed naming the new values** — that is the upgrade telling
  you attention head splitting changed, not a nuisance to silence.
- The native-KV/HMX selection is QAIRT's `gen_kv_format_config`. One documented
  subtraction is applied on top: names whose role proves they are not caches
  are removed, and what was removed is reported.

If either fails after an upgrade, read the new SDK values and re-review them
deliberately. Do not update the fingerprint to make the error go away.

## After the image changes

Rerun `qairt-agent init` and `qairt-agent image build`. The worker executes
sources baked into the image; a stale worker still succeeds while running old
code.
