# T18 — Onboarding: vector-import skill and spec reference

Status: done (2026-08-30)
Depends on: —
Effort: M

## Goal

Close the two documentation gaps a model company's engineer (or an agent)
hits first: no skill for the mandatory golden-pickle import, and no single
spec-authoring reference. Also give the AR/CL/native-KV decision for
non-linear-attention models a written decision guide, so the "what should I
choose?" conversation happens at the agent layer with real guidance instead
of folklore.

## Context

Review findings 19–21 in
[review-findings-2026-08-30.md](review-findings-2026-08-30.md). The pickle
import runbook text sits inside blocked task T08; the import *capability* is
landed and tested, so the skill does not need T08's pickles to be written.
Spec field knowledge is scattered across four documents. Interactive wizards
are out — the CLI stays JSON-in/JSON-out by design — so the decision support
belongs in a skill and a reference document.

## Design

1. **`.claude/skills/qairt-import-vectors/SKILL.md`.** The operational
   sequence for `qairt-agent vectors import-pickle`: trusted-local
   requirement, `--format auto` / `--section` semantics, the Torch-archive
   worker dispatch (Docker vs Apple `container` staging differences), the
   per-AR manifest layout Qwen3.5 needs
   (`validation_manifests_by_ar`), and the traps (direct
   `pickle.dump(torch.Tensor)` rejected; artifacts never under the models
   directory). Content sourced from T08's draft runbook section and the
   landed code, not invented.
2. **`docs/spec-reference.md`.** One consolidated reference for
   `WorkflowSpec`: every top-level field, per-preset requirements
   (`attached_models_by_ar` for Qwen3.5, component scoping for VL,
   single-manifest vs `validation_manifests_by_ar`), `stage_configs.*`
   including `quality.sqnr_modes`, `slice_vector_manifests`,
   `float_reference`, benchmark `prompt`/`prompt_path`, and diagnose config.
   Generated-from-contracts where practical (a doc test asserting every
   documented field exists on the model and every model field is documented
   keeps it honest).
3. **AR/CL/native-KV decision guide** (section of `docs/spec-reference.md` or
   a short `.claude/skills/qairt-author-spec/SKILL.md`): for a wide export
   (e.g. AR2073/CL4096) — why the program pins weight sharing to AR{1,128},
   what CL values are legal (divisible by 256 under native KV; one CL per
   workflow for automatic vector binding), when native KV must stay off, and
   what `qairt-agent plan` output to check before building
   (`pipeline`, AR policy, `effective_benchmark`, `effective_target`,
   `effective_compile`). The skill formulation is preferred: it is the
   agent-native answer to "prompt the user for AR/CL/KV choices" — the agent
   asks, then writes the spec.
4. Cross-link from `README.md`, `docs/first-run.md`, and `examples/README.md`
   so the scattered partial descriptions point at the reference instead of
   growing further.

## Files

`.claude/skills/qairt-import-vectors/SKILL.md`,
`.claude/skills/qairt-author-spec/SKILL.md` (if the skill form is chosen),
`docs/spec-reference.md`, `README.md`, `docs/first-run.md`,
`examples/README.md`, `CLAUDE.md` skills list, a doc-consistency test in
`tests/` (documented-fields ↔ contract-fields).

## Acceptance criteria

- The import skill runs against the smoke path today: following it with a
  locally generated pickle (NumPy tree) produces an immutable manifest
  without T08's real pickles.
- `docs/spec-reference.md` covers every `WorkflowSpec` field and every
  `stage_configs` key; the consistency test fails when a field is added to
  contracts without documentation (and vice versa).
- The decision guide answers the wide-export questions (AR set, CL choice,
  native-KV) with the real constraints, citing where each is enforced.
- `CLAUDE.md` skills list updated; suite clean.

## Result

Landed 2026-08-30.

**`.claude/skills/qairt-import-vectors/SKILL.md`.** The operational sequence for
`qairt-agent vectors import-pickle`: why the trusted-local step is explicit
(unpickling executes code), what each flag does, the per-AR layout Qwen3.5 needs,
the Docker vs Apple-`container` dispatch differences including the honest note
that `--no-dns` is not IP-egress isolation, and the traps (direct
`pickle.dump(torch.Tensor)`, tensor names bound per AR at plan/validate time,
goldens optional in the mechanism but decided in policy, immutable re-imports,
artifacts never under the models directory). Content came from T08's draft
runbook and the landed code.

The skill's rehearsal path was **executed**, not assumed: a NumPy-tree pickle
imports to an `execution_ready` `vector_manifest.json` with no proprietary
input, and `tests/test_vectors_pickle.py` now drives that exact command through
the CLI so the skill cannot teach something that stops running. Writing the
example also caught a wrong filename in the draft (`manifest.json` vs
`vector_manifest.json`).

**`docs/spec-reference.md`.** Every `WorkflowSpec` field across fourteen contract
models, every documented `stage_configs` key, the metadata keys the pipeline
reads, and the AR/CL/native-KV decision guide for a wide export — each decision
citing where it is enforced. `tests/test_spec_reference.py` asserts both
directions: a contract field with no documentation fails, and a documented
snake_case field that exists on no model fails.

The decision guide first went into the reference only. **Revised on maintainer
feedback the same day:** the reference explains the constraints, but nothing was
telling an agent to *ask*, so a default like `weight_sharing=true` could be
inherited silently by a user who never saw the choice. `qairt-author-spec` now
exists as the task originally preferred, and elicitation is its point: preset,
AR set, **weight sharing**, context length, native KV, decoder slices, vectors,
target — each with what the answer commits the user to and where it is enforced.
`tests/test_author_spec_skill.py` asserts the skill's hard claims against the
contracts, including that it keeps saying weight sharing has no device
acceptance evidence.

**Cross-links.** `README.md`, `docs/first-run.md`, `examples/README.md`, and
`CLAUDE.md` now point at the reference instead of each carrying a partial
description; `CLAUDE.md`'s skills list names `qairt-import-vectors`.

Suite 632 passed / 2 skipped (16 new), compileall clean.
