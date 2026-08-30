# Code review findings — 2026-08-29

Condensed evidence base for the plan tasks. A full-repo review (pipeline
stages, GenAI lane, validation/benchmark, device runtime) established the
following. File:line references are as of commit `8b42269`.

## Confirmed capabilities (real, not stubbed)

- Low-level lane stage order is implemented end to end: inspection →
  vector prep → AR/CL conversion (`GraphContext.change_seq_and_context_length`,
  `src/qairt_agent/qairt_adapter/adapter.py:214`) → split + MHA2SHA in one
  `transform` call (`adapter.py:265-313`) → `qairt.convert`
  (`adapter.py:369-382`) → context compile with AR{1,128} weight sharing
  (`adapter.py:1134-1166`, `set_mode("weight_sharing", ...)`).
- GenAI lane binds `Qwen3_5BuilderHTP.from_pretrained` directly
  (`adapter.py:1984-2013`), attaches per-AR models
  (`attach_model_for_arn`, `adapter.py:2069-2081`), and delegates
  transform/convert/quantize/compile to `builder.build()`
  (`adapter.py:2128-2131`). Runtime execution loads the saved container and
  uses its public executor (`adapter.py:1473-1509`); latency measures
  `executor.generate(prompt)` (`src/qairt_agent/pipeline.py:5437-5440`).
- SQNR: three modes (`full_reference`, `teacher_forced`, `chain`), per-AR
  fan-out with fail-closed coverage, RMSE/max-abs/cosine/normalized-RMSE,
  report-only policy (`src/qairt_agent/diagnostics/sqnr.py`,
  `pipeline.py:4458-5191`).
- Latency: host `perf_counter_ns` wall time, warmup 10 / 50 samples,
  p10/p50/p90/p95, MAD, robust CV, optional A/A calibration (on by default),
  optrace per-op cycles normalized with max-thread (never summed)
  (`src/qairt_agent/diagnostics/latency.py`, `pipeline.py:5193-6266`).
- Device execution is entirely through the QAIRT Python API
  (`CompiledModel.__call__` with `qairt.Device(ANDROID, RemoteDeviceIdentifier)`,
  `adapter.py:1443-1547`). No `qnn-net-run`, `genie-t2t-run`, or any vendor CLI
  anywhere, by contract. ADB is transfer/lifecycle only, exact-path staging and
  cleanup (`src/qairt_agent/device/adb.py`).

## Confirmed gaps (drive the plan tasks)

1. **No memory measurement of any kind** — no static footprint, no on-device
   RSS/PSS, no VTCM/DDR. `BenchmarkSpec` has only
   `warmup_runs/measured_runs/optrace` (`src/qairt_agent/contracts.py:453-458`).
   → T05 (static footprint; decided scope). **Resolved (T05, 2026-08-29)**
   within that decided scope: every build publishes `static_footprint`.
   On-device RSS/PSS and VTCM/DDR remain out of scope by decision.
2. **ORT is fallback-only**: an ONNX Runtime reference is captured only when a
   manifest lacks goldens (`pipeline.py:3733-3771`); it cannot coexist with
   AIMET goldens as a second reference. → T04.
   **Resolved for slice boundaries (T04 tier 1, 2026-08-29)**: the debug-only
   `stage_configs.validation.float_reference` runs the float graph alongside
   the supplied goldens and reports both. Layer-level drilldown still needs
   gap 3.
3. **Diagnostic contexts are never executed**: they are built and
   hash-verified, and `op_level_dump_available` is reported, but no diagnose
   path runs them to collect per-op tensors (`pipeline.py:3652-3730`,
   `pipeline.py:6812-6820`). "Bisect" is single-pass attribution over supplied
   lineage, not iterative narrowing. → T04 tier 2.
4. **GenAI benchmark granularity**: one `generate()` call is one sample; no
   prefill/decode split, TTFT, or tokens/sec (SDK public-API limitation, the
   report states it: `pipeline.py:5849-5862`). Default measurement volume is
   180 full generations per benchmark (10+50 plus A/A double run). → T06.
   **Partly resolved (T06, 2026-08-29)**: the GenAI lane now resolves 3+10
   (36 generations with A/A). The prefill/decode split stays unavailable, and
   the 2.49 source confirms the executor reports no generated-token count, so
   `p50_ms_per_token` remains caller-supplied and is labelled as such.
5. **Hard-pinned target**: `SM8850/v81/660` in `harness/constraints.json`,
   enforced by preflight and adapter (`qairt_adapter/preflight.py:24-30`,
   `adapter.py:442-455`, `adapter.py:966-978`). → T02.
   **New (2026-08-29): the pinned `soc_model` value itself looks wrong.**
   QAIRT 2.49 `include/QNN/QnnTypes.h` defines `QNN_SOC_MODEL_SM8850 = 87`
   (and `SM8750 = 69`), and `include/QNN/HTP/QnnHtpDevice.h:60` documents the
   device field as a `Qnn_SocModel_t` enum value. 660 is SM8850's **Android
   SoC ID** (`android_device_constants.py`), a different scheme. Not changed
   in place at the time: a wrong `soc_model` produces a context binary compiled
   for the wrong SoC that can still load. **Settled for SM8750 (T01,
   2026-08-30)**: the maintainer approved the reading, the pin moved to
   SM8750 / v79 / soc_model 69, and a real build/validate/benchmark run on the
   handset succeeded with it — the device itself reports `soc_id 618`, the
   other scheme. SM8850's `soc_model 87` is still unproven on hardware and
   returns with T02's registry.
6. **Linear-attention knowledge is low-level-lane only**: MHA2SHA start points
   (`src/qairt_agent/families/profiles.py:138-152`), recurrent/conv state
   contracts (`adapter.py:553-724`), and native-KV state exclusion
   (`src/qairt_agent/qairt_adapter/native_kv.py:16-20`) are never exercised by
   the GenAI lane, which passes six generic options and trusts
   `Qwen3_5BuilderHTP` (`adapter.py:2059-2068`). Accepted: the SDK owns the
   GenAI lane; SQNR is the after-the-fact guard.

## Latent bugs (→ T07)

All items below except the last were fixed by T07 (2026-08-29); the last is a
T01 verification item.

- **Resolved (T07)** `pipeline.py:5723-5728`: GenAI benchmark config that also
  sets `steps` raises `UnboundLocalError` (`steps`/`initial_native_state` bound
  only in the mutually exclusive `routes` branch) instead of a structured
  `InvalidSpecError`. Worse in practice than reported: without optrace the run
  *succeeded* while silently ignoring `steps`. Now rejected up front by
  `QairtAgent._reject_genai_chain_keys`.
- **Resolved (T07)** `src/qairt_agent/families/split_plan.py:67-112`: local
  decoder layer ranges distribute N layers, but with `split_lm_head=True` the
  SDK splitter folds the last decoder layer into the lm_head split and
  distributes N-1 (`llm_splitter` pops one residual add). Captured
  `SliceBoundary` metadata can disagree with the real graph. Confirmed against
  QAIRT 2.49 `llm_splitter.py` and reproduced exactly.
- **Resolved (T07)** Standalone quantizer stage (`adapter.py:415-440`) is
  orphaned: reachable only from the deprecated expert MCP stage, zero tests,
  and its `encodings_path` is always `None` because `dump_encoding_json` is
  never set. Now defaulted to `True`, documented as deprecated, and tested.
- **Resolved (T07)** `native_kv.py:45`: undocumented `ar % 32 == 0` gate for
  output tensors; `_is_kv_name` (`native_kv.py:16-20`) is substring matching
  (`key`/`value`) that can over-match names like `key_padding_mask`. The gate
  was also missing the SDK's `ar > 0` guard, which matters because
  `adapter.py:3176` passes `ar=0` for a slice with no AR.
- **Resolved (T07)** `pipeline.py:4979`: `if "device_identifier" in locals():`
  used to detect whether a device stage ran. Fragile as well as unclear:
  initializing the variable for readability silently added
  `device_identifier: null` and `remote_cleanup: "confirmed"` to non-device
  validation metrics.
- `adapter.py:265`: binds the private module `qairt.api.transforms._transform`;
  brittle across SDK versions (verify at T01). **Still open, and now watched.**
  Confirmed on 2.49.0.260730 by an import-based probe inside the worker: the
  module and its `transform(...)` signature are unchanged, but there is **no
  public `qairt.api.transforms.transform` re-export** to rebind to, so the
  private binding has to stay. `tools/sdk_signature_probe.py` covers it (29/29
  present) so the next upgrade fails loudly rather than silently.

## Environment facts (as reviewed)

- Repo pins QAIRT 2.48.0.260626; no SDK present under `qnn/qnn` in this
  checkout. The development machine has QAIRT 2.49.0.260730
  (build id 260730134355, qnn_backend_api_version 2.18.0) at
  `/Users/yzwang/Workspace/hexagon_env/sdks/qairt/2.49.0.260730`.
  Since 2026-08-29 `qnn/qnn` is a symlink to that install (`qnn/` is
  gitignored). Discovery resolves the link; the 2.48 pin still fails preflight
  against it, which is the intended fail-closed behavior until T01 lands.
- No `.venv` existed at review time; the test suite (27 modules, all
  SDK-faked) was not executed during the review. Created 2026-08-29 with
  CPython 3.11.15 (`pyproject` requires `>=3.10,<3.13`; the host default is
  3.14) via `uv venv --python python3.11 .venv` and
  `uv pip install -e ".[dev,mcp]"`. Baseline before any T07 change:
  478 passed, 2 skipped (the two skips need Torch, which lives in the worker
  image, not on the host).
- All tests run against injected fake SDK modules; no real-SDK or real-device
  coverage exists in-repo. Real-device acceptance is an out-of-band release
  gate by design.
