# T07 — Review bug fixes

Status: done (2026-08-29) — all five items fixed, each with a regression test
(baseline before the session: 478 passed / 2 skipped).
Depends on: — (item 4 verification pairs naturally with T01)
Effort: S

## Goal

Fix the latent defects found by the 2026-08-29 review. Each fix ships with a
regression test.

## Items

1. **GenAI + `steps` benchmark config raises `UnboundLocalError`**
   (`pipeline.py:5723-5728`): `steps` and `initial_native_state` are bound
   only in the mutually exclusive `routes` branch. Fix: reject the invalid
   combination up front with a structured `InvalidSpecError` naming the
   conflicting keys. Test: benchmark config with `container_path` + `steps`.
2. **`SplitPlan` decoder ranges disagree with the SDK splitter**
   (`families/split_plan.py:67-112`): with `split_lm_head=True` the SDK folds
   the last decoder layer into the lm_head split and distributes N-1 layers;
   the local plan distributes N. Fix: reproduce the SDK's distribution
   (front-loaded remainder over N-1 when `split_lm_head`), and verify against
   the 2.49 splitter behavior; alternatively, if exact reproduction is deemed
   brittle, mark the ranges `advisory: true` in `SliceBoundary` and say so in
   SKU capture. Preferred: reproduce + test with the known 28-layer/4-slice
   case (expect 7,7,7,6 + fold).
3. **Orphaned standalone quantizer stage** (`adapter.py:415-440`): reachable
   only from the deprecated expert MCP surface, untested, `encodings_path`
   always `None` (needs `dump_encoding_json=True`). Decision for this program:
   the production input is always AIMET `apply_encodings`, so **mark the stage
   deprecated** in `mcp_server.py` legacy docs and the adapter docstring, set
   `dump_encoding_json=True` so the surviving debug path is at least correct,
   and add the missing minimal test. Removal is a future decision, not this
   task.
4. **`native_kv.py` magic numbers and loose matching**:
   - `ar % 32 == 0` output gate (`native_kv.py:45`): name it
     (`_HMX_OUTPUT_AR_MULTIPLE = 32`), document the SDK rationale (verify the
     semantic against the 2.49 SDK during T01), and keep behavior unchanged
     unless verification contradicts it.
   - `_is_kv_name` substring matching (`native_kv.py:16-20`): tighten to the
     tensor-role vocabulary already used by the runtime index (`past_key*`,
     `past_value*`, explicit `key_cache`/`value_cache` style names from the
     profile roles) so incidental names like `key_padding_mask` cannot be
     swept in; the audit still fails closed on any expectation mismatch.
5. **`if "device_identifier" in locals():`** (`pipeline.py:4979`): replace
   with an explicitly initialized variable/flag.

## Acceptance criteria

- One regression test per item, failing before and passing after.
- No behavior change beyond the fixes (full suite green).
- `review-findings-2026-08-29.md` items marked resolved with commit refs.

## Result (2026-08-29)

SDK evidence came from the 2.49.0.260730 install on this machine, so items 2
and 4 are verified rather than assumed.

1. **Done.** `QairtAgent._reject_genai_chain_keys` rejects a GenAI run that
   also carries `routes`/`contexts`/`steps`/`initial_native_state`, naming the
   conflicting keys in `InvalidSpecError.details`. The check runs after runtime
   binding, so an automatically bound GenAI lane is covered too — not only an
   explicit `container_path`. The reported crash was the *optrace* path; the
   plain path was worse, succeeding while silently ignoring `steps`.
   Test: `test_genai_benchmark_rejects_low_level_chain_keys`.
2. **Done, exact reproduction.** QAIRT 2.49
   `qairt/optimizer/onnx/passes/splitters/llm_splitter.py` does
   `lm_head = residual_adds.pop()` before distributing, then front-loads the
   remainder (`layers_per_split`/`extra_layers`, `i <= extra_layers`).
   `build_split_plan` now distributes `N-1` layers when `split_lm_head` and
   records the folded layer on the lm_head `SliceSpec`; `SliceSpec.layer_count`
   derives from the range instead of the slice kind; the slice-count guard
   fails closed against the distributable count. SKU capture now records the
   lm_head boundary too, so the folded layer is visible in the evidence rather
   than unaccounted for. The 28-layer/4-slice case yields 7,7,7,6 with layer 27
   folded, as predicted. Tests:
   `test_lm_head_split_folds_the_final_decoder_layer`,
   `test_without_lm_head_split_every_layer_stays_in_a_decoder_slice`,
   `test_reject_decoder_slices_beyond_the_distributed_layers`, and updated
   `test_balanced_decoder_ranges_and_edge_slices` /
   `test_capture_sku_binds_sha_and_boundaries`.
3. **Done.** `QairtSdkAdapter.quantize` defaults `dump_encoding_json=True`
   (SDK `quantizer_module.py` sets `QuantizerOutputConfig.encoding_json` only
   then) and carries a deprecation note; the MCP legacy tool description says
   AIMET `apply_encodings` is the production path. Removal remains a future
   decision. Tests: `test_standalone_quantizer_dumps_encodings_by_default`,
   `test_standalone_quantizer_honours_an_explicit_dump_override`.
4. **Done, with one behavior fix.** `_HMX_OUTPUT_AR_MULTIPLE = 32` is named and
   sourced to `qairt/gen_ai_api/builders/gen_ai_utils.py::gen_kv_format_config`,
   and now includes the SDK's `ar > 0` guard — which is not cosmetic:
   `adapter.py` passes `ar=int(slice_artifact.ar or 0)`, so an AR-less slice hit
   `0 % 32 == 0` and wrongly marked output tensors. For name matching the SDK
   uses a bare `"key" in name or "value" in name`; a strict allow-list was
   rejected because it would silently *drop* real cache tensors under
   unanticipated naming, which is worse than over-matching. The rule keeps the
   SDK's inclusion test and subtracts a documented non-cache role vocabulary
   (`mask`, `padding`, `position`, `index`, `length`, `scale`, `offset`, plus
   the existing linear-attention `recurrent_state`/`conv_state`), applied
   identically by the generator and the audit so they cannot disagree. Tests:
   `test_unknown_ar_keeps_outputs_out_of_the_hmx_layout`,
   `test_non_cache_role_names_are_never_marked`.
5. **Done.** `device_identifier`/`remote_attempt_dir` are initialized to `None`
   and the metrics gate is `device_identifier is not None`. Note this item had
   no failing-before test available — the `locals()` form produced correct
   output — so the test is a characterization test that locks the contract.
   It earns its place: while implementing, the intermediate state (variable
   initialized, `locals()` check retained) made a non-device validation report
   `device_identifier: null` with `remote_cleanup: "confirmed"`, which the test
   catches. Test:
   `test_validate_records_device_evidence_only_for_a_device_stage`.
