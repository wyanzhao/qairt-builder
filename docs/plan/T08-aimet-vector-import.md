# T08 — AIMET vector import and runbook

Status: blocked (2026-08-30) — **waiting on the AIMET pickles themselves.**
The maintainer confirmed they are not available yet. The T01 dependency is
cleared (the 2.49 worker image is built and its SDK import smoke passes, so
Torch archives can be dispatched), so nothing but the input files is missing.
Depends on: the delivered AIMET pickle files
Effort: S

## Goal

Turn the delivered AIMET-quantized golden pickles into the immutable per-AR
vector manifests the pipeline consumes, and document the procedure as a
repeatable runbook.

## Context

- Decision: golden vectors come from the AIMET-quantized model as trusted
  local pickle. Qwen3.5 needs **one manifest per AR** (AR1 and AR128) through
  `vectors.validation_manifests_by_ar`; Qwen3 (wide source) supplies its
  manifest(s) for retargeting.
- The sanctioned path already exists: `qairt-agent vectors import-pickle
  --trusted-local` with the restricted NumPy loader or the rlimit-subprocess
  Torch loader (root `CLAUDE.md`). This task is mostly execution + runbook +
  small ergonomics only if a real gap appears during execution.

## Runbook (to land as `docs/vector-import-runbook.md`)

1. Preferred payload per AR:
   `{"inputs": {name: tensor}, "goldens": {name: tensor}}`. Separate
   input-only / golden-only files use `--section inputs|goldens`.
2. Import each AR separately:

   ```bash
   qairt-agent vectors import-pickle qwen35_ar1_golden.pkl \
     --output-dir artifacts/imported-vectors/qwen3_5/ar1 \
     --trusted-local --format auto --section auto --isolate
   qairt-agent vectors import-pickle qwen35_ar128_golden.pkl \
     --output-dir artifacts/imported-vectors/qwen3_5/ar128 \
     --trusted-local --format auto --section auto --isolate
   ```

3. Torch (`torch.save`) archives dispatch to the pinned Ubuntu worker — build
   and smoke the image first (T01). Direct `pickle.dump(torch.Tensor)` is
   unsupported; re-export via `torch.save` or NumPy.
4. Record the resulting manifest paths in the T03 config cells
   (`validation_manifests_by_ar: {"1": ..., "128": ...}`).
5. Tensor names in the pickle must match the exported graph I/O names exactly
   (manifest-to-model binding is validated per AR at plan/validate time);
   goldens are optional — an inputs-only manifest triggers the audited ORT
   fallback capture, but supplied AIMET goldens are the decided production
   reference, so deliver goldens whenever available.
6. Manifests and raw tensors are immutable and content-addressed; re-imports
   go to a new directory, never edit in place.

## What unblocks this

One AIMET-quantized pickle per model/AR, ideally shaped
`{"inputs": {name: tensor}, "goldens": {name: tensor}}`. For Qwen3.5 that is
two files, AR1 and AR128. Separate input-only / golden-only files also work via
`--section inputs|goldens`.

Tensor names must match the exported graph I/O exactly, because the
manifest-to-model binding is validated per AR at plan/validate time. Goldens
are optional in the mechanism — an inputs-only manifest triggers the audited
ORT fallback capture — but the decided production reference is the AIMET
golden, so deliver goldens wherever they exist.

Nothing else is outstanding: the CLI path (`qairt-agent vectors import-pickle
--trusted-local`), the restricted NumPy loader, and the rlimit-subprocess Torch
loader all exist, and the worker that runs Torch archives is built and smoked.

Until the files arrive, `configs/qwen3_5/sm8850.json` points at the intended
manifest destinations rather than existing files, so that cell plans but its
validation fails closed on the missing manifest. That is the correct state, not
a defect.

## Acceptance criteria

- Runbook landed at `docs/vector-import-runbook.md` and linked from the root
  `CLAUDE.md` and `configs/README.md` (T03).
- One real import per model/AR completed on this machine with manifests
  verifying (`qairt-agent artifact verify`), paths recorded in the configs.
- Any CLI friction found during the real import either fixed (small) or filed
  as a new task here — not worked around undocumented.
