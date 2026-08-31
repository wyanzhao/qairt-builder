# T17 — Regression detection and cross-run comparison

Status: done (2026-08-30)
Depends on: —
Effort: M

## Goal

Make "diagnosis starts after a reported regression" a capability instead of a
manual step: fix `_automatic_diagnosis`'s trigger logic, honor
`stage_configs.diagnose.kind`, and add a first-class cross-run comparison of
the headline metrics (`production_latency_us` against its published CV,
per-AR/per-slice SQNR) between any two verified reports.

## Context

Review findings 16–18 in
[review-findings-2026-08-30.md](review-findings-2026-08-30.md).
`_automatic_diagnosis` (`pipeline.py:7838`) selects the quality path whenever
any observation has `noise_energy > 0` — true of every healthy quantized run —
so the latency path is unreachable after any validate stage and
`kind: "latency"` is silently ignored (`pipeline.py:8184`). No automated
cross-run delta exists for the headline metrics; the program contract already
says a latency change must be read against `production_latency_cv_percent`
(8–17% dispersion), which is precisely a job for tooling, not eyeballs.

## Design

1. **Honor `kind`.** `stage_configs.diagnose.kind` (`quality` | `latency`)
   selects the path explicitly; automatic mode without `kind` runs both and
   reports what each found. Remove the nonzero-noise heuristic as a *trigger*
   (nonzero noise is the quantized steady state, not a regression signal); it
   remains usable as an attribution input once the quality path is selected.
2. **`qairt-agent compare`.** New command:
   `qairt-agent compare --from-job A --to-job B` (or `--from-manifest` /
   `--to-manifest`). It loads both hash-verified reports, refuses
   non-comparable pairs fail-closed (different preset, target, AR set, CL,
   sqnr_modes, or latency meter/lane — each named), and emits a JSON delta:
   - latency: per-AR `production_latency_us` delta in absolute and in units
     of the pooled CV, with the verdict left to the caller (report-only, no
     thresholds — consistent with the program's no-pass/fail rule);
   - quality: per-AR, per-slice-boundary SQNR/RMSE/cosine deltas, worst
     movers first;
   - provenance: both run_ids, manifest SHAs, spec-identity fields, and
     whether the two runs' evidence chains verify.
3. **Wire into diagnose.** `diagnose --from-job B --baseline A` uses the same
   comparison to decide which path (quality/latency) the evidence actually
   implicates before drilling down, replacing the unreachable-latency logic.
4. Out of scope here: automatic MHA2SHA lineage extraction (finding 18). It
   is recorded and may become its own task; `compare` and layer attribution
   still accept caller-supplied lineage as today.

## Files

`src/qairt_agent/pipeline.py` (`_automatic_diagnosis`, new compare logic),
`src/qairt_agent/cli.py`, `src/qairt_agent/contracts.py` (compare
report schema, diagnose config), tests (`test_pipeline.py`, `test_cli.py`),
`docs/native-workflow.md`, `.claude/skills/qairt-diagnose-quality/SKILL.md`,
`.claude/skills/qairt-diagnose-latency/SKILL.md`, `CLAUDE.md` diagnosis
section, `examples/README.md` if a canonical compare example is added.

## Acceptance criteria

- `kind: "latency"` runs the latency path on a manifest that also carries
  SQNR observations (test fails on pre-fix code).
- Automatic mode on a healthy quantized run no longer auto-selects quality on
  the nonzero-noise heuristic (test).
- `compare` on two compatible fixture runs emits per-AR deltas with CV
  context and verified provenance; on mismatched identity fields it fails
  closed naming the field (tests).
- `compare` output is report-only: no pass/fail verdicts anywhere.
- Skills and docs updated with the compare-first workflow.

## Result

Landed 2026-08-30.

**Path selection.** `_automatic_diagnosis` split into
`_automatic_quality_diagnosis` and `_automatic_latency_diagnosis`, each
returning `None` when its evidence is absent. `kind` runs one path and fails
closed naming the missing evidence; without a kind both run and the payload
carries `considered` (`found`/`no_evidence`/`skipped_by_kind`), publishing
`qairt-agent.automatic-diagnosis.v1` with both sub-reports when both have
evidence. `diagnose_latency` now passes `kind="latency"`, so a requested
latency diagnosis stops silently producing a quality report. The nonzero-noise
heuristic is gone as a trigger and remains only as attribution input.

**`qairt-agent compare`.** New `src/qairt_agent/compare.py` plus the CLI
command, accepting `--from-job`/`--to-job` or `--from-manifest`/`--to-manifest`.
Non-comparable pairs fail closed naming every differing field (preset, family,
target, AR set, context lengths, sqnr_modes, latency meter, lane). Latency
deltas are published in absolute terms and in units of the pooled CV, per AR,
with uncomparable ARs listed and explained; quality deltas are per tap
(`ar/scope/mode/slice/tensor`), worst movers first; provenance carries both run
ids, revisions, and the verified SHAs of every report read. No thresholds, no
verdicts — asserted by a test that greps the serialized output.

A path-addressed manifest is verified against the sha256 the publisher wrote
into its filename; a path carrying no recorded expectation is refused rather
than self-hashed, keeping `ManifestStore`'s discipline intact.

**Wired into diagnose.** `diagnose --baseline JOB` (and
`stage_configs.diagnose.config.baseline_manifest`) runs the comparison first and
publishes `implicated`, naming which path the measured change points at with
the rule stated inline: quality by any tap whose SQNR fell, latency by any AR
that moved at least one pooled CV. Both paths still run — this routes attention,
it does not judge the run. `--kind` is exposed on the same command.

Deviation from the design: the comparison lives in its own module rather than
inside `pipeline.py`, which is what T24 is trying to shrink.

Out of scope as planned: automatic MHA2SHA lineage extraction (finding 18).

Nine new tests. Suite 605 passed / 2 skipped, compileall clean. `CLAUDE.md`,
`docs/native-workflow.md`, and both diagnosis skills carry the compare-first
workflow.
