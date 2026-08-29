# T08 — AIMET vector import and runbook

Status: planned
Depends on: T01 (Torch archives require the rebuilt worker image; restricted
NumPy pickles import host-side and can proceed earlier)
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

## Acceptance criteria

- Runbook landed at `docs/vector-import-runbook.md` and linked from the root
  `CLAUDE.md` and `configs/README.md` (T03).
- One real import per model/AR completed on this machine with manifests
  verifying (`qairt-agent artifact verify`), paths recorded in the configs.
- Any CLI friction found during the real import either fixed (small) or filed
  as a new task here — not worked around undocumented.
