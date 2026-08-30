---
name: qairt-first-run
description: Get a QAIRT build, validate and benchmark running end to end on a real device from a fresh clone, using a generated smoke fixture that needs no proprietary model. Use when someone is setting this repository up for the first time, when "nothing runs", or when a first device run needs to be proven before real model work.
---

# First run

Full prose walkthrough: `docs/first-run.md`. This is the operational sequence.

## Rules that matter more than the steps

- **Only QAIRT Python APIs.** Never construct or invoke QAIRT/QNN CLI commands,
  vendor executables, or the C++ API. `qairt-agent` is this project's own CLI
  and is fine.
- **Never run the privileged DNS command yourself.** If Apple `container` needs
  `sudo container system dns create host.container.internal --localhost 203.0.113.113`,
  give it to the user and wait. This is stated in `docs/worker-runtimes.md`.
- **Rerun `init` + `image build` after changing agent source.** The worker runs
  sources baked into the image, not the working tree. A stale worker still
  *succeeds*, silently producing results from old code — this has already
  caused a wrong conclusion once in this repository's history.

## Sequence

1. `qairt-agent init --root .` — needs the SDK at `qnn/qnn` (symlink is fine).
2. `qairt-agent image build --root .` — accepted only after a mounted-SDK
   import smoke test.
3. `qairt-agent doctor --root .` — fails closed on a version mismatch. Do not
   relax it; the pin is in `harness/constraints.json`.
4. `python tools/make_smoke_fixture.py --output-dir models/smoke` — generates a
   tiny ONNX, AIMET-style encodings, vectors whose golden is the float output,
   and two specs. Deterministic from a fixed seed.
5. `qairt-agent plan --spec models/smoke/spec.json` — check `effective_target`.
   An unverified target plans but device stages refuse it.
6. Export `QAIRT_AGENT_ADB_SERIAL` and `QAIRT_AGENT_ADB_SERVER`, then
   `qairt-agent workflow --spec models/smoke/spec.json`.
7. `qairt-agent job watch <JOB_ID> --follow`.

## Reading the result

- SQNR: `stages/validate/*/attempt-001/sqnr_report.json`. Report-only; there is
  no threshold or pass/fail anywhere in this program by decision.
- Latency: read `latency_metric` first. It names `device_execution`. The
  wall-clock number under `harness_diagnostics` is marked `not_latency` and
  must never be reported as the model's speed.
- Reference numbers on the smoke fixture: SQNR ~37.8 dB, accelerator compute
  ~69 µs, wall p50 ~2600 ms. The gap between the last two is expected.

## If a device stage fails

- Stale lease after an unplugged handset: `qairt-agent device gc`.
- `df` / free-space or ADB errors: `qairt-agent device doctor`.
- Never broaden remote cleanup beyond the exact
  `/data/local/tmp/qairt-agent/<job>/<stage>/<attempt>/` path.
