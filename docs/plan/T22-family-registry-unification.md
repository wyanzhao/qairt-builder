# T22 — Family registry unification

Status: done (2026-08-30)
Depends on: —
Effort: M

## Goal

Collapse the four hand-synced family registries into one source of truth so
adding a model family is a single-seam change with a documented procedure,
and alias spellings can no longer diverge.

## Context

Review finding 30 in
[review-findings-2026-08-30.md](review-findings-2026-08-30.md). Family
identity currently lives in `contracts.ModelFamily` (+ `_FAMILY_TO_PRESET`
and per-family spec validators), `families/profiles.FamilyId` (+
`FamilyProfile`), `families/presets.FamilyPreset` (+ `PRESET_REGISTRY`,
`_PRESET_TO_FAMILY`), and `vector_retarget._FAMILY_ALIASES` — with aliases
already diverging (`qwen3_4b` vs `qwen3-4b` vs `qwen_3`, VIT missing from
one). A new family (a Llama-style dense, a new ViT variant) touches 5–6 files
as a scavenger hunt (finding also cross-referenced by the model-routing
review).

## Design

1. One canonical family record per family — id, canonical spelling, alias
   set, preset binding, lane, retarget/KV/AR policy hooks — defined in one
   module under `families/`. The existing enums/registries become derived
   views generated from it (or thin wrappers that assert equivalence at
   import time), so external call sites keep their current types and no
   behavioral change leaks out.
2. Alias resolution goes through one function; the current divergent
   spellings are all registered once, and a test asserts every alias resolves
   identically through every legacy entry point.
3. Write the "add a family" procedure as a short section in
   `families/` module docs or `docs/spec-reference.md` (T18): the one record
   to add, the per-family knowledge that must come from the SDK vs the spec,
   and the tests that must exist (routing enforcement, preset resolution,
   retarget policy).
4. Keep `pipeline.py`'s inline family branches out of scope — that is T24's
   decomposition concern. This task only unifies identity/registries.

## Files

`src/qairt_agent/families/` (new canonical module + presets/profiles
refactor), `src/qairt_agent/contracts.py`, `src/qairt_agent/vector_retarget.py`,
tests (`test_families.py`, `test_presets.py`, `test_vector_retarget.py`,
`test_contracts.py`), docs per item 3.

## Acceptance criteria

- Every family/alias is declared exactly once; a grep for any alias string
  finds one defining site plus derived views.
- A test adds a synthetic family via the single record and asserts it is
  visible through contracts, profiles, presets, and vector-retarget without
  further edits.
- All existing routing/mis-routing tests pass unchanged (no behavior
  change).
- The add-a-family procedure is written and cites the synthetic-family test
  as its template.

## Result

Landed 2026-08-30. No behavior change beyond the deliberate alias widening below.

**One record per family.** `src/qairt_agent/family_registry.py` declares each
family once: key, build lane, decoder profile id (`None` for standalone ViT),
preset ids, canonical spelling, alias set, and the retarget/KV/AR policy. It
imports nothing from the package — `families/profiles` must stay importable
without the SDK and `contracts` imports the registry, so a dependency either way
would be a cycle.

**Five derived views, not four.** The review named four registries; a fifth
turned up during the work — `ModelFamily._missing_` carried its own
hand-written alias table. All five now derive: `contracts._FAMILY_TO_PRESET`,
`ModelFamily._missing_`, `presets._PRESET_TO_FAMILY`, the `FamilyProfile.aliases`
tuples, and `vector_retarget._FAMILY_ALIASES`. Every previously accepted
spelling still resolves to exactly what it did.

**The divergence is closed by union.** `qwen3_4b`/`qwen3-4b` was known only to
vector retargeting and `qwen_3` only to the profile table; both now resolve
through every entry point, as does `vision-transformer` through `ModelFamily`.
This is a deliberate widening — the alternative was leaving a spelling that
works in one place and fails in another.

Omni Thinker stays its own record while sharing the Qwen3.5 build lane: the
retarget axis is finer than `ModelFamily`, and collapsing them would have lost
that distinction.

Seven tests in `tests/test_family_registry.py`, including one that asserts every
declared spelling resolves identically through all three entry points, one that
pins the two previously divergent spellings, and a grep gate that fails if an
alias literal reappears in any derived view. The add-a-family procedure is
written in `docs/spec-reference.md` and cites the synthetic-family test as its
template.

Out of scope as planned: `pipeline.py`'s inline family branches (T24).

Suite 639 passed / 2 skipped (7 new), compileall clean.
