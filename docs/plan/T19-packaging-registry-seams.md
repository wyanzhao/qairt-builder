# T19 — Packaging and registry seams

Status: done (2026-08-30)
Depends on: —
Effort: S

## Goal

Remove the undocumented extra steps that make adding a target or upgrading
the SDK break in places the skills never mention: the literal filename
force-includes in `pyproject.toml` and the population-pinning assertions in
`tests/test_harness.py`.

## Context

Review findings 22–23 in
[review-findings-2026-08-30.md](review-findings-2026-08-30.md). The wheel
force-includes `harness/targets/sm8850.json`, `sm8750.json`, and
`docker/requirements-qairt-2.49.0.260730.txt` by name, so renaming the lock
(SDK upgrade step 2) or adding `sm8950.json` ships a wheel silently missing
the file. `test_harness.py:164` asserts the registry is exactly
`{sm8750, sm8850}` and `:206` requires every entry verified against the
pinned build — which contradicts the add-target workflow's documented
intermediate state (entry committed, acceptance run pending).

## Design

1. **Wheel includes.** Replace per-file force-includes with directory/glob
   inclusion for `harness/targets/*.json`, and include the dependency lock via
   the value `harness/constraints.json` names rather than a literal (if the
   build backend cannot evaluate that, add a packaging test that reads
   `worker.dependencies_file` from the constraints and asserts the built
   wheel contains it — the failure then names the real fix).
2. **Registry tests.** Rewrite the population assertion structurally: every
   file in `harness/targets/` loads, names agree, tuples are unique, and the
   *active* target (plus every target referenced by a `configs/` cell) is
   verified against the pinned build. An unverified, unreferenced entry — the
   add-target intermediate state — must pass the suite; the existing gate
   that refuses device stages for it already provides the safety.
3. **Skills.** Add the previously undocumented steps to
   `qairt-add-target` (configs cell if deployable, packaging note) and
   `qairt-sdk-upgrade` (lock rename ↔ packaging linkage), so the documented
   procedures are complete.

## Files

`pyproject.toml`, `tests/test_harness.py`, a small packaging test (wheel
contents), `.claude/skills/qairt-add-target/SKILL.md`,
`.claude/skills/qairt-sdk-upgrade/SKILL.md`.

## Acceptance criteria

- Adding a syntactically valid `harness/targets/sm8950.json` without a
  `verified` block passes the suite; device stages for it still refuse
  (existing gate test extended to the new entry fixture).
- A built wheel contains every `harness/targets/*.json` and the lock file the
  constraints name (test).
- Renaming the lock without updating packaging fails a test that names
  `pyproject.toml` (or is impossible because inclusion is derived).
- Both skills list the complete step set; no other behavior change.

## Result

Landed 2026-08-30.

**Wheel includes.** The two per-file target force-includes became one directory
mapping, so `harness/targets` ships whole and adding `sm8950.json` needs no
packaging edit. The dependency lock stays a literal — hatchling cannot read
`harness/constraints.json` — but the new `tests/test_packaging.py` reads
`worker.dependencies_file` and asserts packaging covers it, failing with a
message that names `pyproject.toml` and the rename that caused it. Deviation
from the design: the assertion is against the packaging config rather than a
built wheel, because neither `hatchling` nor `build` is installed in the
project venv and building one would need a network install in the test path.

**Registry tests.** The population pin (`set(registry) == {sm8750, sm8850}`) is
gone. In its place: a structural test (every file loads, names agree with
filenames, no two entries share a tuple) and a verification test scoped to what
actually needs it — the *active* target plus every target a `configs/` cell
deploys. A new test builds a temp registry containing an unverified `sm8950`
entry, asserts it loads, and asserts `require_verified_target` still refuses it:
the add-target intermediate state now passes the suite while the device gate
still holds.

**Skills.** `qairt-add-target` gained the three easy-to-miss steps (deployment
cell, packaging, what the registry tests actually assert);
`qairt-sdk-upgrade` step 2 now says the lock rename is a two-file change and
names the test that catches a half-done one.

Suite 610 passed / 2 skipped, compileall clean.
