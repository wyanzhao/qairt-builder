---
name: qairt-import-vectors
description: Import golden/input vectors delivered as a trusted local pickle into the immutable per-AR vector manifests production validation consumes. Use when someone hands over AIMET goldens, when a spec needs validation_manifests_by_ar, or when a pickle import is rejected.
---

# Import vectors from a pickle

Production validation never reads a pickle. It reads an immutable,
content-addressed manifest plus raw tensors. This skill is the conversion, and
the reasons it is deliberately explicit.

## Why the extra step exists

Unpickling executes code. `--trusted-local` is a statement about the file's
provenance, not a formality: only pass it for a pickle you received through a
channel you trust. The loaders are restricted anyway — a NumPy tree goes through
a restricted loader in-process; a Torch archive is loaded in an rlimit
subprocess with `torch.load(weights_only=True, map_location="cpu")` and
normalized to NumPy before the parent accepts anything.

## The command

```bash
qairt-agent vectors import-pickle golden.pkl \
  --output-dir artifacts/imported-vectors/qwen3_5/ar1 \
  --trusted-local --format auto --section auto --isolate
```

| Flag | What it does |
| --- | --- |
| `--output-dir` | Where the manifest and raw tensors land. **Never under the models directory** — the worker mounts that read-only and the import dies with EROFS. |
| `--trusted-local` | Required. Asserts the file's provenance. |
| `--format` | `auto` recognizes a modern `torch.save` zip archive and falls back to NumPy. Force with `numpy` or `torch` when auto guesses wrong. |
| `--section` | `auto` expects `{"inputs": {...}, "goldens": {...}}`. For a file that is only one of them, pass `inputs` or `goldens` — the **whole** pickle is assigned to that section. |
| `--isolate` | Run the load in its own subprocess. Use it. |
| `--case-id` / `--bundle-id` | Name the case inside the manifest. |

## Per-AR manifests

Qwen3.5 and Omni Thinker need **one manifest per AR**, and a low-level multi-AR
run binds each AR to its own entry and fails closed when one is missing. Import
each AR into its own directory:

```bash
qairt-agent vectors import-pickle qwen35_ar1_golden.pkl \
  --output-dir artifacts/imported-vectors/qwen3_5/ar1 \
  --trusted-local --format auto --section auto --isolate
qairt-agent vectors import-pickle qwen35_ar128_golden.pkl \
  --output-dir artifacts/imported-vectors/qwen3_5/ar128 \
  --trusted-local --format auto --section auto --isolate
```

Then point the spec at them:

```json
"vectors": {
  "mode": "provided",
  "validation_manifests_by_ar": {
    "1": "artifacts/imported-vectors/qwen3_5/ar1/vector_manifest.json",
    "128": "artifacts/imported-vectors/qwen3_5/ar128/vector_manifest.json"
  }
}
```

## Torch archives run in the worker

With the normal `apple_container` or `docker` backend the host CLI keeps NumPy
imports local but dispatches a Torch archive to the pinned Ubuntu worker, where
Torch is installed. This needs an initialized project and a built worker image.

- **Docker** mounts the exact archive read-only and runs with `--network none`.
- **Apple `container`** cannot bind a regular file, so the CLI creates a private
  temporary directory holding one content-verified `archive.pt`, mounts that
  directory read-only, verifies the original, the staged copy and the returned
  manifest against the original path and SHA, then removes the staging
  directory. Apple `container` 1.0 offers only `--no-dns`, which is **not** hard
  IP-egress isolation — know that before importing something you do not trust.

The output directory is mounted read-write and the container runs as the host
uid/gid.

## Traps

- **Direct `pickle.dump(torch.Tensor)` is not supported.** Re-export through
  `torch.save` or convert to NumPy first.
- **Tensor names must match the exported graph's I/O exactly.** The
  manifest-to-model binding is validated per AR at plan and validate time, so a
  renamed tensor fails there, not here.
- **Goldens are optional in the mechanism, not in the policy.** An inputs-only
  manifest triggers the audited ONNX Runtime fallback capture, which is
  recorded as a fallback. The decided production reference is the AIMET golden;
  deliver goldens wherever they exist.
- **Re-imports go to a new directory.** Manifests and raw tensors are immutable
  and content-addressed; never edit one in place.
- **Artifacts never live under the models directory.** Worth repeating: it is
  mounted read-only.

## Try it without the real pickles

The capability does not need proprietary inputs. Generate a NumPy tree, import
it, and read the manifest back:

```bash
python - <<'PY'
import pickle, numpy as np
payload = {
    "inputs": {"x": np.ones((1, 64), dtype=np.float32)},
    "goldens": {"y": np.zeros((1, 32), dtype=np.float32)},
}
open("/tmp/demo-vectors.pkl", "wb").write(pickle.dumps(payload))
PY
qairt-agent vectors import-pickle /tmp/demo-vectors.pkl \
  --output-dir artifacts/imported-vectors/demo \
  --trusted-local --format auto --section auto --isolate
```

The result is the same immutable manifest shape a real import produces, so the
per-AR wiring above can be rehearsed before the real files arrive. The command
prints `manifest_path` (the `vector_manifest.json` a spec points at),
`bundle_path`, and `execution_ready` — `execution_ready: true` means the inputs
can actually be fed to a graph, which is what validation needs.

## Related

- [`docs/spec-reference.md`](../../../docs/spec-reference.md) — the `vectors`
  block and everything else a spec carries.
- `qairt-first-run` — the end-to-end path a newcomer runs first.
