# T15 — CLI and job lifecycle robustness

Status: done (2026-08-30)
Depends on: —
Effort: M

## Goal

Every failure a caller can trigger returns the structured JSON error contract,
`job watch` terminates when a worker dies uncleanly, and the MCP submit path
stops bypassing the detached pinned worker. One broken debug stage tool is
fixed with it.

## Context

Review findings 9–12 in
[review-findings-2026-08-30.md](review-findings-2026-08-30.md). Invalid specs
escape as pydantic tracebacks (`cli.py:1394`, `agent.py:221`);
`job watch --follow` spins forever because ORPHANED is not terminal and the
watch path never checks heartbeat staleness (`cli.py:684`,
`contracts.py:1355`); MCP `submit_job` runs jobs in an in-process daemon
thread (`mcp_server.py:79`, `agent.py:414`) contrary to the detached-worker
contract, and defaults to `stages=('build',)` while calling itself a workflow
submit; the standalone `compile_context` stage tool reconstructs native-KV
expectations without `model_path` and can never pass the adapter's audit
(`pipeline.py:4841`, `qairt_adapter/types.py:189`).

## Design

1. **Spec errors.** Wrap spec load/validation (`json.loads`,
   `model_validate`) in the CLI entry path and raise `InvalidSpecError`
   carrying the pydantic error list (field path, message) so `cli.main`'s
   existing handler emits the JSON error contract on stdout with a nonzero
   exit. Same for `--stage-config` files.
2. **Watch termination.** In `_follow`, check heartbeat staleness on each
   poll using the same threshold `mark_orphaned_if_stale` uses; when stale,
   call the existing orphan-marking path (or report
   `state: "ORPHANED"` read-only if marking requires worker identity) and
   exit with a structured event instead of spinning. Add `--timeout` as an
   optional hard cap.
3. **MCP dispatch.** Route MCP `submit_job` through the same detached worker
   launch the CLI uses. In-process execution remains only behind the explicit
   inline flag with the existing compatible-environment guard (no silent
   native fallback — same rule as the CLI). Align the default stages with the
   CLI workflow default (build+validate+benchmark) and document the change in
   `docs/mcp-tools.md`.
4. **compile_context fix.** Carry `model_path` through the JSON round-trip of
   `NativeKvGraphExpectation` (serialize + reconstruct), and add the missing
   test that drives the standalone stage tool against the real expectation
   type rather than a mock that skips the audit.

## Files

`src/qairt_agent/cli.py`, `src/qairt_agent/agent.py`,
`src/qairt_agent/mcp_server.py`, `src/qairt_agent/contracts.py`,
`src/qairt_agent/jobs/journal.py` (staleness reuse), `src/qairt_agent/pipeline.py`
(compile_context), `docs/mcp-tools.md`, tests (`test_cli.py`,
`test_mcp_server.py`, `test_job_recovery.py`, `test_pipeline.py`).

## Acceptance criteria

- Malformed JSON, unknown field, and missing required field each produce the
  JSON error contract on stdout, no traceback (tests).
- A journal with RUNNING state and a stale heartbeat causes `--follow` to
  exit with an orphan event within one poll interval (test with fake clock).
- MCP submit launches the detached worker (asserted via injected launcher in
  tests); inline remains explicit-only; default stages match the CLI.
- The standalone `compile_context` stage tool succeeds against a fake adapter
  that enforces the real `model_path` requirement (test fails on pre-fix
  code).
- Suite and compileall clean.

## Result

Landed 2026-08-30, all four items.

**Spec errors.** `QairtAgentClient._normalize_spec_active` now converts
`json.JSONDecodeError`, `OSError`, and pydantic `ValidationError` into
`InvalidSpecError`, keeping every complaint's field path, message, and type in
`details.errors` plus the source file. Malformed JSON, a missing required field,
and an unknown key each emit the JSON error contract on stdout and exit 1.

**Watch termination.** `_follow` checks heartbeat staleness each poll with the
same threshold recovery uses, emits a structured `orphaned` event plus a final
status carrying `state: "orphaned"`, and returns. Marking the job ORPHANED needs
the worker lease, so the watch reports read-only and points at `job resume`.
`job watch --follow --timeout SECONDS` adds a hard cap that emits
`watch_timeout`.

**MCP dispatch.** `submit_job` prepares the journal and then launches the same
detached worker the CLI spawns (injectable `launcher`), reports
`execution: "detached"` with `worker_pid`, and defaults to the CLI workflow
stages instead of `("build",)`. The in-process path survives only behind an
explicit `inline=true`. The default client is now `background=False`, so it
prepares journals and never runs QAIRT work in the MCP process.
`docs/mcp-tools.md` updated.

**compile_context.** `NativeKvGraphExpectation` gained `to_dict`/`from_dict` and
the stage tool uses them, so `model_path` survives the JSON round trip. Without
it `normalize_sdk_kv_config`'s `by_stem` map was always empty, QAIRT's ONNX-stem
graph names were never remapped to this program's names, and the audit could not
pass. The new test drives the standalone stage tool with a real expectation and
a real slice ONNX; on pre-fix code `model_path` is `None`.

Suite 588 passed / 2 skipped (11 new), compileall clean.
