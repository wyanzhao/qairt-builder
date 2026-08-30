# Deployment configs

One file per deployable cell, laid out as `configs/{preset}/{target}.json`:
the directory names the model preset, the file names the target from the
reviewed registry in `harness/targets/`. A cell is launched directly:

```bash
qairt-agent plan --spec configs/qwen3_5/sm8850.json
qairt-agent workflow --spec configs/qwen3_5/sm8850.json
```

These differ from `examples/` in intent. `examples/` are minimal teaching
templates governed by `examples/README.md`; a config here is what a real run is
started from, so it carries the complete policy for that model on that target —
per-AR sources, vectors, quality modes, the benchmark prompt, and its own
`output_root`.

## Conventions

- **Paths are container-root style** (`/models/...`, `/artifacts/...`), never
  host-specific. The worker mounts the project's models directory; a config
  that hardcodes a developer's home directory is not portable to the worker.
- **`name` and `output_root` are build identity.** One cell gets one root
  (`/artifacts/{model}-{target}`); sharing a root across cells would make two
  different builds contend for the same manifest chain.
- **The target is named, never spelled out.** `"target": {"name": "sm8850"}`
  resolves through the registry, which is also what stops a cell from naming a
  chipset that has never been verified on hardware.
- **No payloads are committed here.** Models, encodings and vectors live
  outside the repository; a config only points at them.

`test_deployment_configs_resolve_to_their_named_target` parses every cell,
resolves it without the SDK, and checks that the directory/file names agree
with the preset and target inside — so a config cannot rot silently, and a
misfiled cell fails the suite rather than surfacing at build time.

## Current cells

| Cell | Preset | Target | Inputs it expects |
| --- | --- | --- | --- |
| `qwen3_5/sm8850.json` | `qwen3_5` (GenAI Builder) | `sm8850` | Independent AR1/AR128 ONNX + AIMET encodings, per-AR validation manifests |

The vector manifest paths point at the output of the AIMET pickle import
([T08](../docs/plan/T08-aimet-vector-import.md)); until that import has been
run on real vectors, those paths are the intended destination rather than
existing files, and validation of this cell will fail closed on the missing
manifest.
