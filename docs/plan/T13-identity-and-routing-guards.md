# T13 — Identity and routing guards

Status: done (2026-08-30)
Depends on: —
Effort: M

## Goal

Close the three places where the framework trusts an unverified identity
claim: the declared preset vs the supplied HF config, the GenAI lane's
compile target, and the GenAI raw-slice AR→graph order. All three follow the
existing pattern: verify what was actually resolved, fail closed on
contradiction, never guess.

## Context

Review findings 5–7 in
[review-findings-2026-08-30.md](review-findings-2026-08-30.md). The preset is
the routing authority by decision — this task does not change that — but
authority without a cross-check means a mis-declared export silently bypasses
every family gate (`families/profiles.py:257`, `pipeline.py:1346`). The
low-level lane verifies the SDK-resolved compile target
(`adapter.py:1114-1147`) but the GenAI lane only writes
`set_targets([target_spec])` and echoes the input (`adapter.py:2352`,
`:2546-2550`). GenAI raw-slice AR binding is positional
(`adapter.py:1960-1964`) with a name-only ABI check that cannot detect order
inversion.

## Design

1. **Preset↔config cross-check.** At build-path family resolution, when
   `sources.*.config_path` is present, read `architectures` (and
   `model_type`) from the HF config and evaluate it against the resolved
   family profile's known architecture names (the detection table already
   exists in `families/`). A contradiction — config names an architecture the
   profile explicitly maps to a *different* family — fails closed with an
   error naming both sides. An architecture the table does not know is a
   recorded warning in the build receipt, not a failure (new families must
   not be blocked by an incomplete table). No filename heuristics; the
   preset remains authoritative when the config is silent.
2. **GenAI target verification.** After `builder.set_targets`, read back
   whatever resolved-target surface the 2.49 builder exposes (inspect the SDK
   source; candidates: the builder's target list property, the compile config
   it assembles). If a readable resolved value exists, compare
   chipset/dsp_arch/soc_model against the registry entry and fail closed on
   mismatch — the analogue of `_validate_compiler_target`. If the 2.49 SDK
   exposes no readable resolved value, record that fact in the build receipt
   as `genai_target_verification: "input_only"` and add the probe to
   `tools/sdk_signature_probe.py` so a future SDK that grows the surface gets
   wired in. Either way the report stops presenting an input echo as a
   resolved value.
3. **AR→graph binding verification.** Strengthen the ABI check at
   `adapter.py:1980-1996` to compare shapes (the AR-bearing dimension differs
   between AR1 and AR128 by construction), so a positional inversion is
   detected. Where shapes alone cannot discriminate (an AR-ambiguous graph),
   fail closed naming the graphs rather than guessing. Record the reviewed
   SDK ordering assumption the same way MHA2SHA start points are recorded: a
   fingerprint of the ordering-relevant SDK surface, mismatch fails closed
   naming the new values.

## Files

`src/qairt_agent/families/profiles.py`, `src/qairt_agent/pipeline.py`
(family resolution site), `src/qairt_agent/qairt_adapter/adapter.py` (GenAI
build + raw-slice binding), `tools/sdk_signature_probe.py`,
`src/qairt_agent/contracts.py` (receipt fields), tests
(`test_families.py`, `test_sdk_adapter.py`, `test_pipeline.py`), `CLAUDE.md`
input-contract note, `docs/native-workflow.md`.

## Acceptance criteria

- A spec declaring `qwen3_dense` over a config whose `architectures` maps to
  Qwen3.5 fails at spec/plan time with an error naming the preset, the config
  value, and the file (test).
- A config with an unknown architecture produces a recorded warning and does
  not fail (test).
- GenAI build receipts either carry an SDK-resolved target that was verified
  against the registry, or explicitly carry
  `genai_target_verification: "input_only"` — never an input echo labeled as
  resolved (test with fake builder both ways).
- A deliberately inverted graph order in the fake SDK is detected and fails
  closed (test).
- Suite and compileall clean; `qairt-agent plan` output unchanged except any
  new verification fields.

## Result

Landed 2026-08-30. All three unverified identity claims are now checked.

**Preset vs config.** `families/profiles.cross_check_declared_family` compares
the declared family with the config's `architectures`/`model_type` (outer and
`text_config`), reusing the extraction the detector already used. A contradicting
architecture fails closed at spec/plan time naming preset, architecture, implied
family, and config file; unknown architectures, a disagreeing nested
`model_type`, and a silent config are recorded under
`effective_config.family_cross_check` instead. Wired into
`_generate_family_config`, so plan and every build path get it.

**GenAI target.** The 2.49 builders do expose a readable resolved value:
`HTPMixin.set_targets` assembles the same `CompileConfig` the low-level lane
validates, reachable as `builder._compilation_config`. `_verify_genai_target`
therefore applies `_validate_compiler_target` verbatim -- empty
`device_custom_configs` included -- for the text, vision, and audio builders, and
the receipt carries `target.verification` with `resolved_verified` or
`input_only`. `tools/sdk_signature_probe.py` gained `HTPMixin.set_targets` /
`set_compilation_options` and the `GraphInfo`/`TensorInfo` fields, so a future
SDK dropping the surface fails the upgrade gate rather than silently downgrading
every receipt to `input_only`. The probe entries were verified against the SDK
source (`htp_mixin.py:490,1091`; `graph_info_models.py:42,60`); the probe itself
still has to run in the worker, where the SDK imports.

**AR to graph.** `_ar_graph_order_error` proves the binding by shape: a
dimension taking each requested AR exactly once across the graphs binds them, an
inverted order is refused naming the graphs, and no discriminating dimension is
refused rather than guessed. The old test fixture carried no `dimensions` and
correctly now fails closed; it was given realistic shapes and two new cases.

Deviation from the design: no separate "ordering fingerprint" was added. The
shape proof verifies the binding directly on each build, which is stronger than
fingerprinting an assumption, and the probe covers the surface it reads.

Suite 569 passed / 2 skipped, compileall clean.
