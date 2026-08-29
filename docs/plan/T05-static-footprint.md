# T05 — Static footprint reporting

Status: done (2026-08-29) — `static_footprint` published by every build lane and
copied into benchmark reports.
Depends on: —
Effort: S

## Goal

Record the static on-disk footprint of build outputs — the decided RAM proxy —
in build receipts and benchmark reports. Report-only, no thresholds.

## Context

Decision: RAM measurement is limited to static footprint (no on-device
RSS/PSS, no VTCM/DDR). Today nothing records artifact sizes
(`review-findings-2026-08-29.md` gap 1).

## Design

1. Capture `bytes` for every published build artifact at publish time
   (`artifacts.py` already streams the file for SHA256 — size is free).
2. Build receipt / manifest metrics gain a `static_footprint` block:

   ```json
   {
     "artifacts": [{"role": "context", "path": "...", "bytes": 0}],
     "contexts_total_bytes": 0,
     "genai_container_total_bytes": 0,
     "total_bytes": 0
   }
   ```

   - Low-level lane: per-context bytes (per slice x CL), plus converted DLCs.
   - GenAI lane: recursive size of the saved container directory plus per
     `models/split_N` file where the raw-slice index exposes them.
   - Diagnostic contexts are listed separately and never mixed into the
     production totals (mirrors the latency separation rule).
3. Benchmark reports embed the same block (copied from the verified build
   receipt, not re-measured) so a latency report alone answers "how big is
   what I just measured".
4. Where both weight-shared and hypothetical standalone sizes are known
   (non-weight-sharing builds exist for comparison), the report may include a
   `weight_sharing_delta_bytes` — optional, only when both measurements exist,
   never estimated.
5. `policy: "report_only"`; no pass/fail anywhere.

## Files

`artifacts.py`, `contracts.py` (report/metrics schema), `pipeline.py` (build +
benchmark publication sites), tests, examples/docs mentions, root `CLAUDE.md`
one-line scope note.

## Acceptance criteria

- Every build publishes `static_footprint` with per-artifact bytes; totals
  reconcile with `du` on the run directory for the covered roles (test with
  temp artifacts).
- Benchmark reports carry the block verbatim from the verified receipt.
- Diagnostic artifacts never counted in production totals (test).
- No estimated numbers: absent measurements are absent fields, not zeros.

## Result (2026-08-29)

Step 1 of the design was already satisfied: `ArtifactRef.size_bytes` is
populated at publish time by the same pass that hashes the file, so the
footprint reads published references and never stats a file a second time or
estimates one.

`pipeline._static_footprint(result, artifacts)` builds the block for all three
build entry points (low-level `build`, standalone ViT, and
`build_genai_container`); it lands in both the stage `data` payload and the
stage metrics, so it reaches the manifest stage record and the job
`StageReceipt` together. `QairtAgent._build_static_footprint` copies it into the
latency report from the hash-verified manifest, stamped with `source:
"build_receipt"` and the measuring stage.

Two decisions worth recording:

- **`total_bytes` covers deployable roles only.** Converted DLCs are listed with
  their own `converted_models_total_bytes` but are build intermediates, not
  something the device holds, so summing them into the headline number would
  overstate the RAM proxy. The block names what it summed in `total_includes`
  rather than leaving the reader to infer it.
- **`weight_sharing_delta_bytes` was not implemented.** Design item 4 allows it
  only when both a weight-shared and a standalone measurement exist; no run
  produces both, so there is no measurement to report and an estimated field
  would violate the task's own no-estimates rule. Revisit only if a
  non-weight-sharing comparison build becomes a real workflow.

One bug was found and fixed during self-review rather than by a passing test:
Omni packaging nests `audio_container_path` and `text_container_path` **inside**
`container_path`, so the first implementation counted those files two or three
times (5 entries for 3 files). Container roots now claim each file once, widest
root first, and raw slices already inside a counted container are skipped. The
assertion added to
`test_qwen35_omni_packages_audio_and_text_but_runtime_is_unsupported` fails
against the pre-fix code.

Tests: `test_build_publishes_a_static_footprint_measured_from_disk`,
`test_static_footprint_keeps_diagnostic_contexts_out_of_the_totals`,
`test_genai_static_footprint_measures_the_saved_container` (also proves the raw
slice inside the container is not double counted),
`test_benchmark_report_carries_the_build_footprint_verbatim`. Documentation:
root `CLAUDE.md` and `docs/native-workflow.md` (new "Static footprint" section).
