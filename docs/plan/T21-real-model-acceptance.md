# T21 — First real-model end-to-end acceptance

Status: blocked — awaiting a real model export (and T08's pickles for the
Qwen3.5/GenAI half)
Depends on: T08 (GenAI half), T20 recommended first, T12
Effort: L

## Goal

Put one real model through each lane on real hardware and record reopenable
evidence. This is the single highest-value action in the program: every
hardware acceptance to date used the generated smoke fixture (a 64×32 MatMul),
so all real-scale behaviors — memory retention, re-hash cost, GenAI graph
ordering, name-heuristic native-KV routing on real tensor names, split
boundaries on a real decoder stack — are currently unexercised
(review-findings-2026-08-30.md, finding 24, re-verified).

## Context

The framework is honest about this gap (T08 blocked, T11 GenAI half blocked),
but capability claims to other teams must not run ahead of it. The low-level
half does not need T08: it needs any real wide Qwen3 dense export
(ONNX + AIMET encodings + a golden pickle or capture-mode vectors).

## Plan

1. **Low-level half (unblocks first).** A real Qwen3 dense export
   (e.g. AR2073/CL4096) through `qwen3_dense` on the active verified target:
   `plan` → `workflow` (build+validate+benchmark) with default per-AR fanout,
   `sqnr_modes` full_reference+chain, device latency. Record: peak worker
   memory, per-stage wall time including hash-verification share, whether the
   AR1/AR128 conversion + split + MHA2SHA + native-KV chain holds on real
   tensor names (the documented name-subtraction list meets real names here
   for the first time), SQNR against the supplied goldens, and
   `production_latency_us` with CV.
2. **GenAI half (needs T08).** Qwen3.5 AR1+AR128 exports + imported per-AR
   manifests through `qwen3_5`: container build, raw-slice SQNR per AR,
   generation benchmark with explicit prompt. Confirms the positional
   AR→graph binding question (T13) on real graphs.
3. Update the target registry `verified` blocks with the real-model run (the
   fixture-based qualification remains recorded history), close the loop on
   `docs/plan/` statuses (T08, T11 GenAI half), and only then update
   capability claims in `README.md`/`CLAUDE.md` per the upgrade-gate rule.
4. Feed surprises back as findings: anything real scale breaks becomes a new
   task in this board, not an ad hoc patch.

## Acceptance criteria

- One low-level real-model run and (after T08) one GenAI real-model run, each
  producing hash-verified build/validate/benchmark reports reopenable from
  the manifests, on a registered verified target.
- Peak memory and stage wall-time numbers recorded in this file's Result
  section, compared against T20's synthetic predictions.
- Registry `verified` blocks reference the real-model run IDs.
- No capability claim updated anywhere before the corresponding half's
  evidence exists.
