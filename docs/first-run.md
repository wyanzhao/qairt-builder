# First run

A walkthrough from a fresh clone to a real report on a real device, using a
fixture this repository can generate. Nothing here needs a proprietary model.

The other documents are reference material: [`README.md`](../README.md) covers
installation and the spec schema, [`native-workflow.md`](native-workflow.md)
covers every stage in detail, [`architecture.md`](architecture.md) covers why
the pieces are shaped the way they are, and [`worker-runtimes.md`](worker-runtimes.md)
covers container and ADB setup. This page is the ordered path through them.

## What you need first

- **The QAIRT SDK**, unpacked or symlinked at `qnn/qnn`. It is not in this
  repository and `init` never moves it. The pinned version is in
  `harness/constraints.json`; a mismatch fails closed rather than warning.
- **A container runtime.** macOS uses Apple `container`, Linux uses Docker.
  Both run the same Ubuntu 22.04 / Python 3.10 / `linux/amd64` worker.
- **An Android device** with a chipset in `harness/targets/`, reachable over
  ADB, for anything that touches hardware. Steps 1–4 do not need one.

## 1. Install and initialize

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install -e '.[mcp,dev]'
```

```bash
qairt-agent init --root .
```

`init` writes `qairt-agent.toml`, copies the harness-selected Dockerfile and
dependency lock, and stages the exact agent sources for the worker image.

## 2. Build the worker image

```bash
qairt-agent image build --root .
```

The image is accepted only after a mounted-SDK import smoke test. **Rerun
`init` and `image build` after changing agent source**, or the worker will keep
running the sources baked into the previous image — a mistake that is easy to
make and hard to see, because the stale worker still succeeds.

On macOS, one privileged step is needed once so the container can reach the
host's ADB server. It is yours to run; the agent never runs it:

```bash
sudo container system dns create host.container.internal --localhost 203.0.113.113
```

## 3. Check the environment

```bash
qairt-agent doctor --root .
```

`doctor` verifies the SDK version and build id, the Python ABI, the worker
image, and the active target. It fails closed on a version mismatch — that is
deliberate, not a bug to work around.

## 4. Generate the smoke fixture

Model payloads are never committed, so every spec in `examples/` and `configs/`
points at a model you must supply. To get a runnable path with no proprietary
input, generate one:

```bash
python tools/make_smoke_fixture.py --output-dir models/smoke
```

This writes a tiny ONNX (one MatMul, one bias add, one Relu), AIMET-style
encodings computed from the real tensor ranges, a vector manifest whose golden
is the float graph's own output, and two specs. Everything derives from a fixed
seed, so the files are byte-identical on any machine and produce the same
content-addressed hashes.

Confirm the plan resolves before running anything:

```bash
qairt-agent plan --spec models/smoke/spec.json
```

Read `effective_target` in the output. A target with no `verified` block plans
fine but **build and device stages will refuse it**: it has never been proven on
hardware. Pass `--target` to the generator to select a different registered one.

## 5. Run the workflow on a device

```bash
export QAIRT_AGENT_ADB_SERIAL=<serial from `adb devices`>
export QAIRT_AGENT_ADB_SERVER=localhost:5037
qairt-agent workflow --spec models/smoke/spec.json
```

`workflow` runs build, validate and benchmark as one detached job and prints a
job id. Follow it with:

```bash
qairt-agent job watch <JOB_ID> --follow
```

### Running the stages separately

`workflow` is build + validate + benchmark in one job. To run them apart —
re-validating without rebuilding, or re-benchmarking after a device change —
note that **`validate`, `benchmark` and `diagnose` require `--from-job`**, not
`--spec`. They reuse the build job's manifest and spec; there is no way to
validate a build that does not exist.

```bash
qairt-agent build --spec models/smoke/spec.json
```

That prints a `job_id`. Feed it to the later stages:

```bash
qairt-agent validate --from-job <BUILD_JOB_ID>
```

```bash
qairt-agent benchmark --from-job <BUILD_JOB_ID>
```

Both mint their own job ids and can be followed with `qairt-agent job watch`.
Passing `--spec` to them instead is refused with
`'validate' requires --from-job <build job id>`.

## 6. Read the reports

Everything lands under the spec's `output_root` (`artifacts/smoke` by default).

**Quality** — `runs/<run>/stages/validate/<key>/attempt-001/sqnr_report.json`.
Each observation carries SQNR, RMSE and cosine against the supplied golden.
These are report-only: there is no threshold and no pass/fail verdict anywhere
in this program, by decision.

**Latency** — `runs/<run>/stages/benchmark/<key>/attempt-001/latency_report.json`.
Read `latency_metric` first. It names the block that is the latency, and that
is `device_execution`. Inside it, **`production_latency_us` is the number this
program reports as the model's latency**: QAIRT's accelerator compute time
excluding wait, averaged over ten profiled executes. Read
`production_latency_cv_percent` with it — the metric's absolute value is small,
which makes it the most dispersed one in the block. `accelerator_execute_us`,
`qnn_execute_us`, per-operator cycles, `spread` and `samples` are all published
beside it.

**Do not read the wall-clock number as latency.** It lives under
`harness_diagnostics`, marked `not_latency: true`, because QAIRT relaunches
`qnn-net-run` for every remote call: a wall sample measures process launch,
context load, HVX/HMX power-on, deinit and ADB transport far more than it
measures the model. On the reference device the smoke fixture measures roughly
2600 ms of wall time around **69 µs** of accelerator compute. The wall number
is still useful — it detects harness and transport degradation — but it is not
the model's speed.

**Footprint** — the `static_footprint` block, present in both build and
benchmark reports. It is the only RAM metric this program publishes.

## 7. Drill into a divergence

When SQNR is lower than expected, the second generated spec turns on the
layer-level float reference:

```bash
qairt-agent workflow --spec models/smoke/spec-layer-debug.json
```

This is **debug-only** and never runs unless the config is present. It builds
diagnostic contexts, executes them, and compares every tapped tensor against
the float graph under ONNX Runtime, ordered by the float graph's topology:
`runs/<run>/stages/validate/<key>/attempt-001/float_reference_report.json`.

On the smoke fixture it produces something worth understanding before you trust
this mode on a real model:

| tensor | SQNR |
| --- | --- |
| `h0` (MatMul output) | 40.27 dB |
| `h1` (bias add output) | **2.51 dB** |
| `output` (Relu output) | 37.76 dB |

`h1` looks catastrophic and is not. It feeds a Relu, which discards the
negative half of the range — exactly where quantization error is largest
relative to signal — so the error never reaches the output. This is why the
report is labelled `first_observed_divergence_not_root_cause`: the first bad
row is where to start looking, not the answer. `device_tensor_source`
distinguishes a tensor tapped from a diagnostic context from a production
context boundary.

## Multi-slice: chain modes and per-slice device time

Some behaviour only exists when one slice feeds another. Generate a two-slice
fixture whose shapes compose:

```bash
python tools/make_smoke_fixture.py --output-dir models/smoke --chain
```

Build each slice, fill the built context paths into
`models/smoke/chain/chain-stage-config.json`, and run a chain workflow. On the
reference device this produces:

- **Per-slice device time.** `device_execution.by_slice` carries one block per
  slice — 77.8 µs and 73.1 µs of accelerator compute — and `totals` is a sum of
  per-slice means, labelled as such because the slices run sequentially. The
  wall p50 for the same work was 9994.5 ms.
- **Local versus propagated error.** With
  `quality.sqnr_modes = ["full_reference", "teacher_forced", "chain"]`,
  `teacher_forced` feeds each slice its own golden boundary while `chain` feeds
  it the previous slice's device output. Slice 1 measured 44.45 dB
  teacher-forced and 40.45 dB chained: about 4 dB of its observed error is
  inherited from slice 0, not its own. That separation is the reason both modes
  exist.

Layer drilldown does **not** work over a chain assembled this way. Diagnostic
contexts belong to the build that produced them, so two independently built
slices leave one of them without one; the stage says so and names the slices.

## When a device stage will not start

A run that dies with the handset unplugged, or a killed worker, can leave a
lease held and a staged directory behind on the device. Nothing reclaims it
automatically, because a lease is deliberately owner-checked rather than
timed out from the outside.

Look first, without changing anything:

```bash
qairt-agent device gc --dry-run
```

It reports `stale_leases`, what it would clean, and what it would skip:

```json
{"ok": true, "dry_run": true, "stale_leases": 0, "cleaned": [], "skipped": []}
```

Drop `--dry-run` to release them. It rechecks the owner token under a
per-device lock and removes only the exact
`/data/local/tmp/qairt-agent/<job>/<stage>/<attempt>/` sandbox — never a parent
directory. For ADB reachability and free-space problems, use
`qairt-agent device doctor` instead.

## Where to go next

- Deploying a real model: `configs/README.md` for the per-cell layout, and
  `examples/README.md` for which examples are production templates and which
  are capability-gated or legacy.
- Importing AIMET goldens from a pickle: `qairt-agent vectors import-pickle`,
  which is explicit because pickle can execute code.
- Adding a target: `harness/targets/`, and the acceptance procedure in the root
  `CLAUDE.md`.
- Upgrading the SDK: the ordered procedure in the root `CLAUDE.md`, which
  requires `tools/sdk_signature_probe.py` to pass inside the worker.
