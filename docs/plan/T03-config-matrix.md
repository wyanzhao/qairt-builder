# T03 — Model x target config matrix

Status: done (2026-08-30) — scope reduced by the maintainer to one validated
cell rather than a full matrix.
Depends on: T02
Effort: S

## Goal

Store deployable specs as a matrix of `configs/{model}/{target}.json`, one
file per model x hardware cell, validated by `qairt-agent plan` in tests.

## Context

- `examples/` stays as canonical minimal teaching templates
  (`examples/README.md` governs their status). `configs/` is the concrete
  deployment matrix used by real runs.
- Decisions: Qwen3.5 uses independent AR1/AR128 exports, CL4096, GenAI Builder
  split defaults; Qwen3 dense/MoE use one wide export (AR2073/CL4096) with
  low-level AR/CL conversion; targets are `sm8850` and `sm8750` from the T02
  registry.

## Design

1. Layout:

   ```text
   configs/
     qwen3_5/sm8850.json
     qwen3_5/sm8750.json
     qwen3_dense/sm8850.json
     qwen3_dense/sm8750.json
     qwen3_moe/sm8850.json        # when a MoE export exists
   ```

2. Each config is a full `WorkflowSpec` input: preset, sources (container-root
   style paths `/models/...`), `sequence` (Qwen3.5: `ars [1,128]`,
   `context_lengths [4096]`, `weight_sharing`, `native_kv`; split section
   omitted where GenAI Builder defaults apply), `vectors` with per-AR
   `validation_manifests_by_ar` pointing at T08 import outputs,
   `metadata.attached_models_by_ar` for Qwen3.5, the T02 target name, an
   explicit benchmark `prompt`, and a per-cell `output_root`
   (`/artifacts/{model}-{target}`) — `name`/`output_root` are build identity,
   one cell one root, never shared across cells.
3. A test iterates every file under `configs/` through spec parsing and
   `plan` resolution (no SDK needed) and asserts the resolved
   pipeline/AR/native-KV/target and output layout — configs cannot rot
   silently.
4. `configs/README.md` documents the matrix, path conventions, and how a cell
   maps to a run (`qairt-agent workflow --spec configs/qwen3_5/sm8850.json`).

## Out of scope

- `vit` / `qwen3_vl` cells (presets remain available; add cells only when those
  models enter the program).
- Committing any model, encoding, or vector payloads.

## Acceptance criteria

- All seed cells parse and `plan` cleanly in the test suite.
- Path conventions documented; no host-specific absolute paths inside configs.
- Root `CLAUDE.md` points to `configs/` beside `examples/`.


## Result (2026-08-30)

### Scope reduced, deliberately

The maintainer asked for the matrix to be dropped: one cell that a test
validates is enough. Building the full model x target grid now would have
committed five more configs whose sources and vectors do not exist yet, so
every extra cell would have been an unverifiable guess about paths that T08 has
not produced.

What landed instead:

- `configs/qwen3_5/sm8850.json` — the primary model on the target this program
  is currently accepted on, as a complete `WorkflowSpec`: per-AR attached
  models, per-AR validation manifests, `apply_encodings`, CL4096, an explicit
  benchmark prompt, its own `output_root`, and `"target": {"name": "sm8850"}`.
- `configs/README.md` — the layout rule (`configs/{preset}/{target}.json`), how
  a cell differs from an `examples/` template, and the path/identity
  conventions.
- `test_deployment_configs_resolve_to_their_named_target` — iterates every cell
  under `configs/`, resolves it without the SDK, and asserts that the directory
  name matches the preset, the file name matches the target, the resolved tuple
  matches the registry entry, that entry is **verified**, and the output layout
  and `/artifacts/` root are present. It is written over a glob, so adding a
  cell later extends the coverage without touching the test.

Verified live: `qairt-agent plan --spec configs/qwen3_5/sm8850.json` resolves
the GenAI Builder pipeline, ARs `[1, 128]`, 4 decoder slices, native KV, the
`sm8850` target (SM8850/v81/soc_model 87, verified), and the GenAI benchmark
defaults 3+10 per generate call.

### Adding the remaining cells later

The test is a glob, and the registry now guarantees a cell cannot name an
unverified target, so a new cell is one file plus the real vector paths. The
cells deliberately not written yet are `qwen3_5/sm8750`, `qwen3_dense/*` and
`qwen3_moe/*`; they need their exports and their T08 vector manifests to exist
first.
