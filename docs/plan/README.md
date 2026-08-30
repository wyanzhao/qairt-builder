# QAIRT Builder Long-Horizon Plan

This directory is the execution board for multi-session work on this
repository. Each task file is self-contained: an agent (Claude Code, Codex, or
a human) picks one task, executes it against its acceptance criteria, and
updates the status here and in the task file. The operating contract in the
root `CLAUDE.md` always applies; until a task lands, the current behavior
documented there stands.

## Program decisions (maintainer review, 2026-08-29)

These decisions are settled. Do not re-litigate them inside task work; open a
new decision entry here instead.

- **Primary model: Qwen3.5** (hybrid/linear attention). Inputs are two
  independent exports — AR1 and AR128 ONNX + AIMET encodings — routed through
  the GenAI Builder lane with `metadata.attached_models_by_ar`. Context length
  is 4096. Split policy follows GenAI Builder defaults.
- **Secondary models: Qwen3 dense / Qwen3 MoE** (no linear attention). Input is
  one wide export (for example AR2073/CL4096); the low-level lane performs
  AR/CL conversion to AR1/AR128.
- **Golden vectors** come from the AIMET-quantized model and are delivered as
  trusted local pickle; they must be imported into immutable per-AR vector
  manifests before use.
- **Reference policy**: AIMET goldens remain the production reference for
  tensor-level SQNR/RMSE/cosine. An ONNX Runtime **float-graph** reference is
  added as an explicit second reference for layer-by-layer comparison against
  on-target outputs. Layer-by-layer comparison is a **debug-only mode**: the
  capability must exist but is never enabled by default.
- **RAM**: static artifact footprint only (context/container byte sizes).
- **Latency**: keep the host wall-clock + optrace methodology; add GenAI-lane
  measurement defaults and explicit token-count support.
- **Hardware**: SM8850 and SM8750, selected through a reviewed target registry.
  The single hard-pinned target is removed; fail-closed discipline (no
  fallback, no unregistered target) is retained.
- **SDK**: upgrade the pin to QAIRT 2.49.0.260730 (build id 260730134355). The
  SDK may be provided under `qnn/qnn` via a symlink to the real install.
  *(Landed 2026-08-29 by T01.)*
- **`soc_model` correction (2026-08-29, maintainer-approved)**: `soc_model` is
  the `Qnn_SocModel_t` enum value, not the Android SoC ID. The previous
  `SM8850 / 660` pin conflated the two. SM8750 is `v79 / 69`, SM8850 is
  `v81 / 87`, and the SM8750 device on hand reports `soc_id 618`, confirming
  the two schemes are distinct.
- **Out of scope by decision**: direct Genie API integration, power/thermal
  measurement, token-level accuracy metrics (top-1 agreement, KL), end-to-end
  Omni audio runtime, end-to-end Qwen3-VL multimodal execution.
- All landed documentation and code are English-only.

## Status board

| ID | Task | Status | Depends on |
| --- | --- | --- | --- |
| [T01](T01-sdk-upgrade-2.49.md) | QAIRT SDK upgrade to 2.49.0.260730 | done (2026-08-30) | — |
| [T02](T02-target-registry.md) | Target registry (SM8850, SM8750) | done (2026-08-30) | T01 |
| [T03](T03-config-matrix.md) | Model x target config matrix | done (2026-08-30) — reduced to one validated cell | T02 |
| [T04](T04-ort-float-reference.md) | ORT float reference and layerwise debug | tier 1 done (2026-08-29); tier 2 planned | T01 for tier 2 and device acceptance |
| [T05](T05-static-footprint.md) | Static footprint reporting | done (2026-08-29) | — |
| [T06](T06-latency-refinements.md) | Latency refinements (GenAI defaults, tokens) | done (2026-08-29) | — |
| [T07](T07-review-bugfixes.md) | Review bug fixes | done (2026-08-29) | — |
| [T08](T08-aimet-vector-import.md) | AIMET vector import and runbook | blocked — awaiting the AIMET pickle files | the delivered pickles |
| [T09](T09-docs-agent-organization.md) | Docs and agent organization | done (2026-08-29) | — |

Status values: `planned`, `blocked(<task>)`, `in-progress(<date>)`,
`done(<date>)`.

## Execution protocol

1. Pick exactly one task. Read its file, the root `CLAUDE.md`, and
   [review-findings-2026-08-29.md](review-findings-2026-08-29.md) for the
   file:line evidence the task cites.
2. Implement to the task's acceptance criteria. When behavior changes, move the
   typed contract, `qairt-agent plan` output, canonical example, tests, and
   documentation together (root `CLAUDE.md` rule).
3. Run the gates: `.venv/bin/pytest -q`, `.venv/bin/python -m compileall -q src
   tests`, and `qairt-agent doctor` where applicable. Tasks touching the SDK or
   device additionally require the acceptance runs named in the task.
4. Update this status board and the task file's `Status:` line with the date
   and a one-line result. If the task exposed a new decision, record it under
   "Program decisions" with the date.
5. Never mark a capability claim done without the reopenable report evidence
   the contract requires.

## Dependency notes

T01 gates everything that touches the SDK surface: T02 verifies target tuples
against the 2.49 SDK, T04 tier 2 depends on the T01 capability probe, and T08
Torch-archive import needs the rebuilt worker image. T05, T06, and T07 are
SDK-neutral and may proceed in parallel at any time. T03 needs the registry
from T02.

**Amendment (2026-08-29).** The gating is softer than assumed for *static*
questions. With the SDK available at `qnn/qnn`, reading its source answers
several "needs T01" questions without changing the pin: the SM8750 and SM8850
SoC numbering (T02), the `executor.generate` metrics question (T06), the
`split_llm` distribution (T07), and a 23/23 signature probe (T01 item 5) were
all settled that way, and T04 tier 1 shipped without T01 at all. What still
genuinely requires T01 is anything that must **execute** the SDK: the worker
image and its import smoke, T08's Torch archives, T04 tier 2's diagnostic
context execution, and every device acceptance run — including the one that
must settle the SM8850 `soc_model` discrepancy T02 now documents.

## Program decisions (amendment 2026-08-30)

- **Per-model knowledge is sourced from the SDK, not copied.** Where the GenAI
  Builder already knows something about a family — MHA2SHA start points, the
  native-KV/HMX tensor selection — the low-level lane reads it from the SDK at
  build time. Hand-maintained copies go stale silently, and two of the three we
  had already produced real bugs. Where a value cannot be sourced (the
  `split_llm` layer distribution, because planning must work without the SDK),
  the reproduction stays but is labelled `advisory` rather than presented as
  observed. Drift is made loud, never followed silently: a reviewed fingerprint
  guards the start points, and the signature probe covers every newly bound
  surface.

## Progress log

- **2026-08-29** — session 2. Environment created (`.venv`, CPython 3.11.15);
  baseline 478 passed / 2 skipped. Landed T07 (5 fixes), T05 (static
  footprint), T06 (latency refinements), and T04 tier 1 (ORT float reference);
  suite now 504 passed / 2 skipped with compileall clean. `qnn/qnn` symlinked
  to the 2.49 SDK and `qairt-agent init` run, leaving `sdk_metadata` (the 2.48
  pin) as the only critical doctor failure — the fail-closed behavior T01 will
  clear. Static SDK verification recorded in T01 and T02. Nothing committed —
  the working tree carries the whole session.
