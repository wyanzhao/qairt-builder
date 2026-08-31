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
- **Latency**: *superseded 2026-08-30 — see the amendment below.* Originally:
  keep the host wall-clock + optrace methodology; add GenAI-lane measurement
  defaults and explicit token-count support.
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
| [T04](T04-ort-float-reference.md) | ORT float reference and layerwise debug | tier 1 + tier 2 low-level done (2026-08-30) | — |
| [T05](T05-static-footprint.md) | Static footprint reporting | done (2026-08-29) | — |
| [T06](T06-latency-refinements.md) | Latency refinements (GenAI defaults, tokens) | done (2026-08-29) | — |
| [T07](T07-review-bugfixes.md) | Review bug fixes | done (2026-08-29) | — |
| [T08](T08-aimet-vector-import.md) | AIMET vector import and runbook | blocked — awaiting the AIMET pickle files | the delivered pickles |
| [T09](T09-docs-agent-organization.md) | Docs and agent organization | done (2026-08-29) | — |
| [T10](T10-latency-measurement-correction.md) | Latency is host-harness time, not device time | done (2026-08-30) | — |
| [T11](T11-device-only-latency.md) | Make device the only latency metric | low-level half done (2026-08-30); GenAI half blocked | T10; GenAI half needs T08 |
| [T12](T12-doc-truth-reconciliation.md) | Documentation truth reconciliation | done (2026-08-30) | — |
| [T13](T13-identity-and-routing-guards.md) | Identity and routing guards | done (2026-08-30) | — |
| [T14](T14-device-soc-verification.md) | Attached-device SoC verification | done (2026-08-30) | — |
| [T15](T15-cli-job-robustness.md) | CLI and job lifecycle robustness | done (2026-08-30) | — |
| [T16](T16-measurement-report-honesty.md) | Measurement report honesty fixes | done (2026-08-30) | — |
| [T17](T17-regression-detection.md) | Regression detection and cross-run comparison | done (2026-08-30) | — |
| [T18](T18-onboarding-gaps.md) | Onboarding: vector-import skill and spec reference | done (2026-08-30) | — |
| [T19](T19-packaging-registry-seams.md) | Packaging and registry seams | done (2026-08-30) | — |
| [T20](T20-build-scale-preparation.md) | Build-stage scaling (memory, hash cost) | done (2026-08-30) | — |
| [T21](T21-real-model-acceptance.md) | First real-model end-to-end acceptance | blocked — awaiting real model export | T08 (GenAI half); T20, T12 recommended |
| [T22](T22-family-registry-unification.md) | Family registry unification | done (2026-08-30) | — |
| [T23](T23-typed-boundaries.md) | Typed boundaries (adapter Protocol, report contracts) | done (2026-08-30) | — |
| [T24](T24-pipeline-decomposition.md) | Pipeline decomposition and runner dedup | done (2026-08-30) | T23; T22 recommended |

Status values: `planned`, `blocked(<task>)`, `in-progress(<date>)`,
`done(<date>)`.

### Review wave 2026-08-30 (T12–T24)

Evidence base: [review-findings-2026-08-30.md](review-findings-2026-08-30.md)
(an eight-dimension full-repo review at commit `c9f551c`; the highest-impact
claims were independently re-verified). Suggested execution order, three
phases — within a phase, tasks are independent and may run in any order:

- **Phase A — correctness and truth (do first):** T12 (highest leverage: the
  contract currently *understates* landed capabilities, which misleads every
  agent reading it), then T13, T14, T15, T16, T17.
- **Phase B — enablement:** T18, T19, T20, then T21 when a real model export
  (and T08's pickles for the GenAI half) arrive. T21 is the program's single
  most valuable pending action: no real model has been through either lane
  yet, and capability claims to other teams must not run ahead of it.
- **Phase C — structure (long-horizon, schedule around feature work):** T22,
  T23, then T24 (T24 depends on T23's typed seams).

Phase C changes no behavior; do not batch it with Phase A/B patches. Findings
recorded without a task (accepted or low severity) are listed at the end of
the findings file — check that list before filing anything new from the same
review.

**All three phases landed on 2026-08-30** except T21, which is the one task
still waiting on inputs rather than on work. It remains the program's single
most valuable pending action: no real model has been through either lane, and
capability claims to other teams must not run ahead of it.

## Execution protocol

1. Pick exactly one task. Read its file, the root `CLAUDE.md`, and
   [review-findings-2026-08-29.md](review-findings-2026-08-29.md) for the
   file:line evidence the task cites.
2. Implement to the task's acceptance criteria. When behavior changes, move the
   typed contract, `qairt-agent plan` output, canonical example, tests, and
   documentation together (root `CLAUDE.md` rule).
3. Run the gates: `.venv/bin/pytest -q`, `.venv/bin/python -m compileall -q src
   tests`, and `qairt-agent doctor` where applicable. A device acceptance run
   needs no proprietary model: `python tools/make_smoke_fixture.py` generates a
   deterministic one, so the run is reproducible by whoever reads the result. Tasks touching the SDK or
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
- **Latency means device time (supersedes the 2026-08-29 latency decision).**
  The host wall-clock number was never device time: QAIRT relaunches
  `qnn-net-run` per call, so per-call context load, HVX/HMX power-on and deinit
  sit inside the sample. On SM8750 a 49 KB graph measured ~4900 ms against 79 us
  of accelerator compute. T10 corrected the reported scope and published
  device-side execute time and per-op cycles from QAIRT's own profiling log;
  T11 makes device the only metric and demotes wall to harness diagnostics.
  The two lanes have **different meters** and must never be conflated: the
  low-level lane reports accelerator/QNN execute time, while the GenAI lane can
  only report Genie's dialog-level device timings and token rates, because
  `generate()` reaches Genie rather than `CompiledModel.__call__` and
  `qairt.Profiler` cannot observe it. Per-op attribution on the GenAI lane
  requires an explicit single-`ar` raw `CompiledModel` profiling run.
- **optrace is not the source of per-op cycles.** `option="optrace"` needs a
  schematic binary this program's compile does not emit and fails without one;
  per-op cycles come from `level="detailed"` alone.
- **Production latency is `accelerator_compute_us`** (maintainer decision,
  2026-08-31): QAIRT's "Accelerator (execute excluding wait) time", the cost of
  the model on the hardware with both this program's host orchestration and the
  device's queueing and memory wait outside it. It is published as
  `production_latency_us` with `production_latency_source`. Its absolute value
  is small — on SM8750 roughly 4% of accelerator execute time — which makes it
  the most dispersed metric in the block, measured at 8–17% CV against ~2% for
  accelerator execute, so `production_latency_cv_percent` is published with it
  and a change is read against that dispersion. This closes T10's open
  question.
- **Deployment latency for Qwen3.5 is a different meter and is not this
  number.** Genie measures `token-generation-rate` and `time-to-first-token` on
  the device; those are the deployment-relevant numbers, they arrive through the
  GenAI lane, and the two are never combined or converted into one another.

## Progress log

- **2026-08-30** — session 3. Landed the whole 2026-08-30 review wave except
  what needs real model inputs: **T12–T20 and T22–T24**. T08 and T21 remain
  blocked on the AIMET pickles and a real export, and T11's GenAI half with
  them. Suite 552 → **698 passed / 2 skipped**, with two new gates: a scoped
  `mypy` type check (15 files, clean) and `tests/test_pipeline_decomposition.py`
  pinning the new module shape.

  Three things worth recording beyond the task files. The GenAI lane does expose
  a readable resolved compile target after all (`HTPMixin` builds the same
  `CompileConfig` the low-level lane validates), so it now runs the identical
  guard instead of echoing its input into the receipt. A *fifth* family registry
  turned up during T22 — `ModelFamily._missing_` carried its own alias table —
  and was folded into the canonical records with the other four. And
  `pipeline.py` went from 8,285 lines to 943 across twelve stage modules with no
  report, JSON, or CLI change, verified by the suite at every landing.

  Nothing committed — the working tree carries the session.

- **2026-08-29** — session 2. Environment created (`.venv`, CPython 3.11.15);
  baseline 478 passed / 2 skipped. Landed T07 (5 fixes), T05 (static
  footprint), T06 (latency refinements), and T04 tier 1 (ORT float reference);
  suite now 504 passed / 2 skipped with compileall clean. `qnn/qnn` symlinked
  to the 2.49 SDK and `qairt-agent init` run, leaving `sdk_metadata` (the 2.48
  pin) as the only critical doctor failure — the fail-closed behavior T01 will
  clear. Static SDK verification recorded in T01 and T02. Nothing committed —
  the working tree carries the whole session.
