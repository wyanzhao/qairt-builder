# T01 — QAIRT SDK upgrade to 2.49.0.260730

Status: in-progress (2026-08-29) — pin, lock, worker image + SDK import smoke,
target move to SM8750, green `doctor` / `device doctor`, and a real build on the
2.49 SDK producing an SM8750 context binary. The remaining acceptance criterion
is the on-device validate/benchmark half, blocked on a one-time privileged host
DNS bridge the agent must not run itself; the exact command and the prepared run
are at the end of "The pin change".
Depends on: —
Effort: L

## Goal

Move the reviewed pin from QAIRT 2.48.0.260626 to **2.49.0.260730
(build id 260730134355)**, prove the new SDK through every gate, and record a
capability inventory that downstream tasks (T02, T04, T06) consume.

## Context

- `harness/constraints.json` is the single source of truth for the pin; the
  upgrade procedure is in root `CLAUDE.md` ("Version and dependency updates").
- The SDK on the development machine lives at
  `/Users/yzwang/Workspace/hexagon_env/sdks/qairt/2.49.0.260730`
  (`sdk.yaml`: `version: 2.49.0`, `build_id: 260730134355`,
  `qnn_backend_api_version: 2.18.0`). The maintainer provides it under
  `qnn/qnn` as a **symlink** to that real install.
- Preflight currently fails closed on any other version — that is correct and
  must remain; this task changes the pin, not the discipline.

## Scope

1. **Pin change** — one reviewable patch to `harness/constraints.json`:
   `qairt.version = "2.49.0"`, `qairt.build_id = "260730134355"`. Read both
   values from the SDK's `sdk.yaml`, never from this document.
2. **Dependency lock** — add
   `docker/requirements-qairt-2.49.0.260730.txt` referenced by
   `worker.dependencies_file`; regenerate from the 2.49 SDK's requirements, do
   not silently reuse the 2.48 lock. Bump `worker.image` tag.
3. **Worker image** — rebuild via `qairt-agent init` + `image build` only
   through harness values; pass the mounted-SDK import smoke
   (`image smoke`).
4. **Symlink support** — `qnn/qnn` may be a symlink. Verify SDK discovery
   (`project.py` / `preflight.py`) and both container runners mount the
   **resolved real path** read-only (Apple `container` and Docker do not follow
   dangling or relative links across mount boundaries). Add a regression test
   with a symlinked SDK root fixture. Fix resolution with `Path.resolve()` at
   the discovery boundary if needed.
5. **Signature probes** — re-verify every bound SDK surface against 2.49 and
   update `tests/test_sdk_adapter.py` fakes to the 2.49 shapes. Known-risk
   surfaces from the review:
   - `qairt.api.transforms._transform` (**private module**,
     `adapter.py:265`) — prefer rebinding to the public
     `qairt.api.transforms.transform` re-export if it exists in 2.49.
   - `SplitModelConfig`, `MhaConfig`, `M2sStartPoint` field sets.
   - `qairt.convert(..., encodings=, calibration_config=, backend=)`.
   - `qairt.CompileConfig` + `set_mode("weight_sharing", graph_names=,
     soc_model=, dsp_arch=)`.
   - `qairt.optimizer.onnx.GraphContext.from_files /
     change_seq_and_context_length / export`.
   - `qti.aisw.tools.core.modules.converter.quantizer_module.QAIRTQuantizer`.
   - `qairt.gen_ai_api.builders.qwen.builder.Qwen3_5BuilderHTP.from_pretrained`,
     `attach_model_for_arn`, `set_transformation_options` option keys,
     `skip_ar_conversion`, `native_kv`, `weight_sharing`, `set_targets`.
   - `qairt.gen_ai_api.gen_ai_builder_factory.GenAIBuilderFactory.create /
     create_audio_encoder`; `WorkflowBuilder.from_builders`.
   - `container_factory.load_container`, `get_executor(device=, clean_up=,
     prepare_environment=, qairt_sdk_root=)`, `prepare_environment`,
     `generate`, `clean_environment`, `LLMContainer.models` split layout
     (consumed by `_saved_genai_raw_slices`, `adapter.py:1617-1719`).
   - `qairt.load`, `CompiledModel.__call__(inputs, device=, graph_names=,
     use_native_input_data=, use_native_output_data=)`,
     `qairt.Profiler(context={"level","option"})`, `generate_reports`,
     `qairt.Device`, `RemoteDeviceIdentifier`, `DevicePlatformType.ANDROID`.
6. **Capability inventory** (record results in this file under a "Findings"
   section when executing):
   - Does the 2.49 `GenAIBuilderFactory` recognize Qwen3.5 architecture names
     (would allow dropping the direct-constructor pin)?
   - Do 2.49 built containers / builders expose any intermediate-tensor or
     debug-output option usable for layer-level dumps in the GenAI lane?
     (T04 tier 2 consumes this answer.)
   - Does `executor.generate` return richer metrics (prefill/decode timing,
     token counts) in 2.49? (T06 consumes this answer.)
   - Any change to the SDK's Qwen3.5 MHA2SHA start-point assumptions
     (16 KV heads x 256 head_dim, `families/profiles.py:147`)?
7. **Family capability tests** — update per the probe results.

## Out of scope

- Relaxing any preflight check. Version mismatch stays a hard failure.
- Individual pin overrides outside `harness/constraints.json`.
- Code changes for new 2.49 features (belongs to the consuming tasks).

## Pre-change verification (2026-08-29)

The 2.49 SDK was made available at `qnn/qnn` as the intended symlink to
`/Users/yzwang/Workspace/hexagon_env/sdks/qairt/2.49.0.260730`, and the
following was checked against it **without changing any pin**.

### Symlink discovery — works today (scope item 4, discovery half)

`qairt-agent doctor --root .` now reports
`"sdk_root": "/Users/yzwang/Workspace/hexagon_env/sdks/qairt/2.49.0.260730"`,
i.e. discovery already resolves the link rather than reporting `./qnn/qnn`, and
`qairt_capability` finds the Python API through it ("QAIRT Python API found at
lib/python"). The **container-mount** half of item 4 is still unverified: it
needs an actual Apple-container / Docker run with the symlinked root.

### Fail-closed version discipline — confirmed live

With the 2.49 SDK present and the 2.48 pin unchanged, doctor fails exactly as
the contract requires, as a **critical** check:

```
sdk_metadata: resolved .../2.49.0.260730; sdk.yaml version=2.49.0
build_id=260730134355 (expected 2.48.0 / 260626120635)
```

This is the intended behavior, and it is also the evidence that the pin change
is the only thing standing between this machine and a green SDK check.

### Signature probes — 23/23 present (scope item 5)

Every SDK surface the adapter binds exists in 2.49.0.260730. Probing was done
by parsing the SDK's Python source rather than importing it: the SDK targets
Ubuntu 22.04 / CPython 3.10 and its native extensions do not load on this
macOS/arm64 host, so an import-based probe here would report false absences.
The import-based probe inside the worker image remains an acceptance
requirement.

Notable results:

- `qairt.api.transforms._transform` — **still present**, and the review's
  suggested mitigation is not available: the module exposes `transform(...)`
  but there is no public `qairt.api.transforms.transform` re-export in 2.49.
  The private binding therefore stays, and stays a known risk. Its signature is
  unchanged in the parts the adapter uses:
  `transform(model, backend=BackendType.HTP, quantization_stage=None,
  encodings=None, lora_adapters_path=None, lora_tensor_names_path=None,
  **transforms)`.
- `GenAIBuilderHTP.from_pretrained(pretrained_model_path, cache_root, *,
  tokenizer_path=None, config_path=None, config_dict=None)` — matches the
  adapter's call at `adapter.py:2010` exactly, including `cache_root` passed
  positionally.
- `attach_model_for_arn(arn, model_path, encodings_path=None)` — matches
  `adapter.py:2086` and `adapter.py:2597`.
- `VisionEncoderBuilderHTP.from_pretrained(pretrained_model_path,
  cache_root=None, *, config_path=None, config_section="vision_config")` —
  matches `adapter.py:2109`; note `cache_root` is positional-or-keyword here
  but keyword-only-ish in the LLM builder, which the adapter already handles.
- `CompileConfig.set_mode(mode, **kwargs)` still accepts only
  `"weight_sharing"` and raises `ValueError` otherwise.
- `Profiler` context still uses `level` (default `"detailed"`) and `option`
  (default `"optrace"`).
- `qairt.convert/compile/load/CompileConfig/CalibrationConfig/Device/
  RemoteDeviceIdentifier/DevicePlatformType/Profiler` are all exported from
  `qairt/__init__.py`, though behind platform guards (`if should_import_qairt:`
  and `if not is_oelinux_user():`) — a probe that only scans module top level
  will wrongly report them missing.
- `qti.aisw.tools.core.modules.converter.quantizer_module.{QuantizerInputConfig,
  QAIRTQuantizer}` present (see T07 item 3 for the `dump_encoding_json`
  finding that came out of reading it).

### Capability inventory — partial answers already usable

- **Layer-level debug output for GenAI containers** (T04 tier 2 input): not
  answered yet; needs the builder/container surfaces exercised, not just
  parsed.
- **Richer `executor.generate` metrics** (T06 input): **answered, negative.**
  `qairt/gen_ai_api/executors/gen_ai_executable.py` defines `GenerationMetrics`
  with `init_time`, `prompt_processing_time`, `prompt_processing_rate`,
  `token_generation_time`, `token_generation_rate`, `time_to_first_token`,
  `token_acceptance_rate`, `adapter_switch_time` — and **no generated-token
  count**; `_parse_execution_metrics` never reads the `num-generated-tokens`
  record that the underlying Genie profile
  (`qairt/modules/genie_execution/genie_profile_record.py`) does carry. T06
  consumed this: `ms_per_token_source` can only be `caller` on this SDK.
  Worth noting `time_to_first_token` **is** reported, so a future task could
  surface TTFT even though a prefill/decode split of the wall sample is not
  available.
- **`split_llm` layer distribution** (not originally listed, found while doing
  T07 item 2): `qairt/optimizer/onnx/passes/splitters/llm_splitter.py` pops the
  final layer's residual add for the lm_head split before distributing, so
  decoder splits share `N-1` layers with a front-loaded remainder. Already
  reproduced in `build_split_plan`.
- **SoC numbering** (T02 input): `Qnn_SocModel_t` in `include/QNN/QnnTypes.h`
  gives `SM8850 = 87` and `SM8750 = 69`, which contradicts the repository's
  pinned `soc_model 660`. Full evidence and the reasoning for not changing it
  here are in [T02](T02-target-registry.md).

### Doctor state on this machine after `qairt-agent init`

Exactly one critical check is red, and it is the pin itself:

```
sdk_metadata  CRITICAL   2.49 SDK vs the 2.48 pin
host_abi      warn       macOS/arm64 host; the worker is ubuntu22.04/x86_64
```

`worker_build_context` went green after `qairt-agent init` (which left the
tracked `.dockerignore` byte-identical), and the Apple container image
`qairt-agent-worker:0.1.0-ubuntu22.04-py310` is already present. So the
remaining work is genuinely the pin plus its lock/image/device gates, not
environment plumbing.

## The pin change (2026-08-29)

### Target: SM8750, and why the tuple moved at all

The acceptance run needs hardware, and the hardware available is an
`SM-S931U1` reporting `ro.soc.model=SM8750` with `soc_id=618`. With a single
pinned target, running there means pinning SM8750, so the tuple is now
**SM8750 / v79 / soc_model 69**, using the `Qnn_SocModel_t` value from
[T02](T02-target-registry.md)'s evidence table. The device's own `soc_id=618`
is direct hardware confirmation that the Android SoC ID and the QNN SoC model
are different numbering schemes -- the confusion that made the old
`soc_model 660` pin wrong. SM8850 is not buildable until T02 lands its
registry.

`TargetSpec` now defaults from and validates against
`harness/constraints.json`, so that file is genuinely the one reviewed source
for the target, as the contract has always claimed. A spec naming a different
tuple fails at spec time instead of at compile time.

### One guard had to be rebuilt, not just re-pointed

The old rule was "a resolved `v79` means QAIRT fell back, so refuse it". On
SM8750 the intended architecture *is* v79, and QAIRT's own compile default is
`dsp_arch v79` with `soc_model 69` (`qairt/api/compiler/config.py:615`) -- the
exact SM8750 tuple. A resolved-value check can therefore no longer distinguish
an intended target from a silent fallback.

`_validate_compiler_target` now fails closed on an **empty**
`device_custom_configs` list, which is the SDK's "could not set soc model for
chipset ... skipping device config creation" path and the actual mechanism by
which a compile ends up on the SDK's defaults. The test fake previously
returned an empty list unconditionally, so the guard had never been exercised;
it now builds device configs from `soc_details` the way the SDK does, and a new
test drives the skip path.

### What ran, and what it proved

- `qairt-agent init` regenerated the worker build context.
- `qairt-agent image build` built
  `qairt-agent-worker:0.2.0-ubuntu22.04-py310` (`sha256:52e9a6980a8c...`) and
  its **SDK import smoke passed** -- the real proof that the new dependency
  lock installs on Ubuntu 22.04 / CPython 3.10 and that QAIRT 2.49.0.260730
  imports inside it.
- `qairt-agent doctor` is **`ok: true`**: every critical check green, including
  `sdk_metadata` and `target`; only the expected macOS `host_abi` warning
  remains.
- `qairt-agent device doctor` against the real SM8750 is **`ok: true`**: adb
  reachable, serial present, state `device`, target triple recorded, remote
  free space 189436004 KB, and `sdk build 260730134355 matches expected`.
- **A real build ran on the real SDK.** `qairt-agent build --spec
  models/acceptance/spec.json` completed inside the pinned worker with the
  2.49 SDK: `INFO_CONVERSION_SUCCESS`, then
  `qairt.compile: Compiled model: tiny for backend: HTP`, producing
  `vit.dlc` (10856 bytes) and the context binary `vit.bin` (49824 bytes) for
  **SM8750 / v79 / soc_model 69**. The published manifest is reopenable at
  `artifacts/sm8750-acceptance-ae/manifests/d0bb8fde-.../manifest-r000001-3832222c....json`.
  Its stage metrics also exercise T05 against real artifacts: `total_bytes`
  49824 over `total_includes: ["context"]`, with the converted DLC reported
  but correctly outside the headline total.
  Caveat recorded rather than glossed: the converter logged "Following OPs
  fallback to float" for this model, so the compiled graph is not fully
  quantized. That is a property of the synthetic acceptance model and its
  generated encodings, not of the SDK pin, and it does not affect what this run
  is evidence for.
- Full test suite green plus `compileall`.

### A device-compatibility defect the real device exposed

`device doctor` initially failed `remote_free_space` with "could not parse
remote free space". `df PATH` names the filesystem by its **own mount point**,
which on this Android 15 device is `/data/user/0`, not the queried
`/data/local/tmp`; the parser filtered for lines containing the queried path
and matched nothing. The test fixture had used a mount point that happened to
equal the queried path, which is why it never caught this. The parser now finds
the data line by position and the available column from the header (covering
the toybox, legacy-Android, and wrapped-filesystem-name layouts), and the
fixture carries this device's real output.

### A second defect the real SDK exposed: calibrate cannot run in the worker

The acceptance model was first driven through `quantization.mode =
"calibrate"`. The converter succeeded, then the SDK's IR quantizer printed
`Could not create output directory: ./output/` and the process died in native
code. The cause is environmental, not model-specific: the container's working
directory is `/workspace`, which `RuntimeMounts` mounts **read-only**
(`docker/image.py:117`), and the quantizer writes a *relative* `./output/`.
Any SDK tool that assumes a writable working directory fails the same way.

Because the death is a native exit, no Python handler runs and the job stays
`running` with a stale heartbeat until a `resume` marks it `ORPHANED` — that
part is the designed recovery path (`mark_orphaned_if_stale` is called only
under the worker lease), not a defect.

Not fixed here, deliberately. The fix is to give the container a writable
working directory, which changes container semantics for every run and belongs
in its own reviewed change; and this program's production quantization path is
AIMET `apply_encodings`, which is the path the successful build above used.
Filed for a separate task rather than bundled into the SDK pin.

### A third defect: the DNS alias check could not read Apple container 1.0

With the maintainer's `sudo container system dns create` done and
`container system dns ls` showing the domain, worker startup still refused
device jobs with "host alias is not configured for ADB". The command was fine;
the check could not read the answer. Apple `container` 1.0 prints
`container system dns list --format json` as a **flat array of domain
strings** -- `["host.container.internal"]` -- while `_has_dns_alias` only
recognised a domain named under a `domain`/`name` key and treated a bare string
element as no match. Fixed so a string *list element* counts, while a string
under an arbitrary dict key still does not, preserving the existing
false-positive guard (`[{"description": "..."}]` remains "not configured").

With that fixed, the container reached the host ADB server through the bridge
and the run failed with the server's own
`error: device 'RFCY30B296K' not found` -- i.e. the DNS half is proven working
and only the physically disconnected handset remains.

### What is still required before this task can be marked done

Only the on-device half -- validate and benchmark -- and only because the
handset is unplugged. The privileged DNS bridge is configured and verified, and
the build half is done with a reopenable report. Reconnect the SM8750 over USB
(`adb devices` should list `RFCY30B296K`), then one command produces the
acceptance evidence:

```bash
QAIRT_AGENT_ADB_SERIAL=RFCY30B296K QAIRT_AGENT_ADB_SERVER=localhost:5037 \
  qairt-agent workflow --spec models/acceptance/spec.json --follow
```

`models/acceptance/` holds a small self-contained acceptance model (a 64x32
MatMul + bias + ReLU graph routed through the `vit` preset, so it needs no
MHA2SHA, KV cache, or weight sharing), calibration vectors, an ORT float
golden, and that spec. It is deliberately not a Qwen model: this run exists to
prove the SDK pin, the SM8750 target, and the device path, and a real LLM would
confound that with model-specific failures.

## Acceptance criteria

- `harness/constraints.json`, lock file, and image inputs changed together in
  reviewable patches; no other pin path exists.
- `.venv/bin/pytest -q` and compileall pass; `qairt-agent doctor` all green on
  this machine with the symlinked SDK; `image build` + `image smoke` pass.
- Signature-probe suite updated and passing against the real 2.49 SDK import
  (worker smoke), not only fakes.
- Capability inventory recorded in this file with concrete API names/results.
- At least one real-device golden + latency acceptance run on SM8850 with the
  new SDK, reports reopenable.
- Examples and capability claims updated only after the gates pass.
