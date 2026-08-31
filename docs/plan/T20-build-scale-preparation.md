# T20 — Build-stage scaling: memory release and hash-verification cost

Status: done (2026-08-30)
Depends on: —
Effort: M

## Goal

Prepare the monolithic build stage for real-model scale before T21 runs one:
stop retaining every live SDK object for the whole build, and bound the
repeated full-artifact re-hashing cost, without weakening the
content-addressed evidence discipline.

## Context

Review findings 25–26 in
[review-findings-2026-08-30.md](review-findings-2026-08-30.md), both
re-verified. Live `graph_context`/`sdk_model` objects are attached to
artifacts (`adapter.py:248/:347/:406`), accumulated across the whole
CL×AR×slice loop (`adapter.py:3290-3296`) and held by the pipeline through
vector prep, route publishing, and footprint (`pipeline.py:3679-3750`). On
the smoke fixture this is invisible; on a real multi-GB Qwen3 build it is an
OOM risk stacked on top of from-zero crash restarts. Separately, every
continuation stage re-hashes all cumulative artifacts at stage boundaries.

## Design

1. **Scoped retention.** After a slice's converted/compiled outputs are
   published (hashed, written, recorded), drop the live SDK references for
   that slice (`_LIVE_SDK_FIELDS` already names them —
   `pipeline.py:92-99` — the release just never happens). Downstream
   consumers that genuinely need a live object (native-KV export stamping,
   diagnostic context build) must take it before publish or re-load from the
   published artifact; make each such consumer explicit in the adapter
   contract. The BuildResult returned to the pipeline carries artifact
   references, not live objects, unless a same-stage consumer is registered.
2. **Within-build checkpoints (bounded scope).** Full mid-build resumability
   is not attempted here. The scope is: per-(CL, AR, slice) publish points
   already exist as artifacts — record them incrementally in the stage
   attempt state so a crashed build stage can *report* what completed, and so
   T21's real run can quantify what a restart would cost. A follow-up task
   may build true resume on that evidence; do not build it speculatively.
3. **Hash-verification cost.** Keep verify-before-use, but add a per-run
   in-memory verification cache keyed by (path, size, mtime, expected SHA):
   within one worker process, an artifact verified once is not re-read unless
   its stat changed. Cross-run and cross-process behavior is unchanged (cold
   verification stays full). Record cache hits in stage diagnostics so the
   evidence trail shows what was re-verified versus stat-checked.
4. Measure both effects with a synthetic large-artifact fixture (hundreds of
   MB of generated tensors — no proprietary model needed) and record
   before/after numbers in this file's Result section.

## Files

`src/qairt_agent/qairt_adapter/adapter.py`, `src/qairt_agent/pipeline.py`,
`src/qairt_agent/artifacts.py` (verification cache),
`src/qairt_agent/jobs/` (attempt-state recording), tests
(`test_sdk_adapter.py`, `test_pipeline.py`, `test_artifacts.py`,
`test_job_recovery.py`).

## Acceptance criteria

- After each slice publish, no live SDK object for that slice remains
  reachable from the accumulated build state (test with weakref/fake
  adapter).
- Consumers needing live objects are explicit; the diagnostic-context and
  native-KV paths still pass their existing tests.
- The verification cache never accepts a stat-changed file without a full
  re-hash (test), and cold-start behavior is byte-identical to today.
- Synthetic large-artifact run shows reduced peak retention and reduced
  repeat-hash wall time; numbers recorded in Result.
- Evidence discipline unchanged: no artifact is ever used after a failed
  verification, and reports still record verification provenance.

## Result

Landed 2026-08-30.

**Scoped retention.** `LIVE_SDK_FIELDS` and `without_live_sdk_objects` moved into
`qairt_adapter/types.py` — one definition, in the module that owns those objects;
`pipeline.py` now imports it instead of keeping a second copy.
`QairtSdkAdapter._release_published` swaps published artifacts in the build
accumulators for copies with their live fields cleared: per slice after its
contexts and diagnostic contexts are written, and per context length after its
variants are done. No consumer was broken because the pipeline never read these
fields — it strips them before serializing — and every in-build consumer
(convert, compile, the Qwen3.5 derivation check, native-KV stamping) has already
run at the release point. The released copy compares equal to the original
(`compare=False`), so nothing identity-based downstream changes.

**Within-build checkpoints (bounded).** `build(..., on_publish=...)` fires once
per (context length, slice) with the ARs, converted models and context paths;
the pipeline appends them to `runs/{run_id}/stages/build/progress.jsonl`.
Deliberately append-only, outside the manifest, and never read for reuse — reuse
still goes through verified receipts. It is the crash breadcrumb a future resume
task would be built on. Resume itself was not attempted, as scoped.

**Hash-verification cost.** `verify_artifact` keeps a per-process cache keyed by
(path, mtime_ns, size, expected sha). A stat change always forces a full
re-hash, cold-start behaviour is unchanged, and `verification_statistics()`
exposes full-versus-cached counts for stage diagnostics.

**Measured** (synthetic, no proprietary model; script in the session scratchpad):

| | before | after |
| --- | --- | --- |
| 4 x 256 MB verified at 4 stage boundaries | 1.54 s, 16 full reads | 0.38 s, 4 full reads + 12 cached (-75%) |
| 8 slices x 2 ARs of live objects retained | 512.0 MB peak | 64.0 MB peak (-87%) |

The retention figure is the shape of the effect, not a real build's absolute
number: it uses 16 MB stand-ins for live SDK graphs. What it shows is that peak
retention becomes one slice's worth instead of the whole CL x AR x slice loop.

Suite 616 passed / 2 skipped (6 new), compileall clean.
