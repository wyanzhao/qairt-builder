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

## 6. Read the reports

Everything lands under the spec's `output_root` (`artifacts/smoke` by default).

**Quality** — `runs/<run>/stages/validate/<key>/attempt-001/sqnr_report.json`.
Each observation carries SQNR, RMSE and cosine against the supplied golden.
These are report-only: there is no threshold and no pass/fail verdict anywhere
in this program, by decision.

**Latency** — `runs/<run>/stages/benchmark/<key>/attempt-001/latency_report.json`.
Read `latency_metric` first. It names the block that is the latency, and that
is `device_execution` — accelerator compute and execute time, QNN execute time,
and per-operator cycles, averaged over ten profiled executes with `spread` and
`samples` beside the mean.

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
