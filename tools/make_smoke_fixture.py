"""Generate the smoke fixture: a tiny ONNX, AIMET-style encodings, and vectors.

Model payloads are never committed, so a clone has nothing it can actually
run: every example and config points at a model the user must supply. This
script closes that gap from the other side — it *generates* a complete, tiny,
deterministic input set, so a first run needs no proprietary model and an
acceptance result recorded in `docs/plan/` can be reproduced by someone else.

The graph is deliberately trivial (one MatMul, one bias add, one Relu) but it
is a real quantized build: it exercises convert, apply_encodings, context
generation, device execution, SQNR against a supplied golden, device latency,
and — because it has internal activations — the layer-level float reference.

Everything is derived from a fixed seed, so two machines produce byte-identical
files and therefore the same content-addressed hashes.

    python tools/make_smoke_fixture.py --output-dir models/smoke

Then:

    qairt-agent plan --spec models/smoke/spec.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

# Changing any of these changes every hash the fixture produces.
SEED = 20260830
IN_FEATURES = 64
OUT_FEATURES = 32
OPSET = 17
IR_VERSION = 9
ACTIVATION_BITWIDTH = 16
PARAM_BITWIDTH = 8


def build_model(path: Path, weight: np.ndarray, bias: np.ndarray) -> Path:
    """Write the float graph. `h0` and `h1` stay internal on purpose.

    They are what the layer-level float reference drills into: a graph whose
    only visible tensor is its output cannot demonstrate that mode.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "w"], ["h0"], name="fc"),
            helper.make_node("Add", ["h0", "b"], ["h1"], name="bias_add"),
            helper.make_node("Relu", ["h1"], ["output"], name="act"),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, IN_FEATURES])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, OUT_FEATURES])],
        initializer=[
            helper.make_tensor(
                "w", TensorProto.FLOAT, list(weight.shape), weight.ravel().tolist()
            ),
            helper.make_tensor(
                "b", TensorProto.FLOAT, list(bias.shape), bias.ravel().tolist()
            ),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)]
    )
    model.ir_version = IR_VERSION
    onnx.save_model(model, path)
    return path


def _asymmetric(values: np.ndarray, bitwidth: int) -> dict[str, object]:
    """An asymmetric activation encoding covering the observed range."""

    low = float(min(values.min(), 0.0))
    high = float(max(values.max(), 0.0))
    if high == low:
        high = low + 1.0
    levels = (1 << bitwidth) - 1
    scale = (high - low) / levels
    offset = int(round(low / scale))
    return {
        "bitwidth": bitwidth,
        "dtype": "int",
        "is_symmetric": "False",
        "max": high,
        "min": low,
        "offset": offset,
        "scale": scale,
    }


def _symmetric(values: np.ndarray, bitwidth: int) -> dict[str, object]:
    """A symmetric parameter encoding, as AIMET emits for weights."""

    limit = float(np.abs(values).max()) or 1.0
    levels = 1 << (bitwidth - 1)
    scale = limit / levels
    return {
        "bitwidth": bitwidth,
        "dtype": "int",
        "is_symmetric": "True",
        "max": float(limit - scale),
        "min": -limit,
        "offset": -levels,
        "scale": scale,
    }


def build_encodings(
    path: Path,
    *,
    activations: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
) -> Path:
    """Write AIMET-style encodings matching the tensors the graph produces.

    These are computed from the real ranges rather than invented, so
    apply_encodings produces a build whose SQNR is a meaningful number instead
    of noise.
    """

    document = {
        "version": "0.6.1",
        "activation_encodings": {
            name: [_asymmetric(values, ACTIVATION_BITWIDTH)]
            for name, values in sorted(activations.items())
        },
        "param_encodings": {
            name: [_symmetric(values, PARAM_BITWIDTH)]
            for name, values in sorted(parameters.items())
        },
        "quantizer_args": {
            "activation_bitwidth": ACTIVATION_BITWIDTH,
            "param_bitwidth": PARAM_BITWIDTH,
            "dtype": "int",
            "is_symmetric": False,
            "per_channel_quantization": False,
            "quant_scheme": "post_training_tf_enhanced",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def build_chain_fixture(
    output_dir: Path,
    *,
    target: str,
    artifacts_root: Path,
    rng: np.random.Generator,
) -> dict[str, str]:
    """Two slices whose shapes compose, plus the routes that chain them.

    A single-slice fixture cannot exercise the paths that only exist when one
    slice feeds another: chain-scope device capture, the multi-slice diagnostic
    execution branch, and the `chain`/`teacher_forced` SQNR modes. Those were
    covered only by fake adapters until this existed.
    """

    middle = 48
    weight0 = rng.normal(0.0, 0.125, (IN_FEATURES, middle)).astype(np.float32)
    bias0 = rng.normal(0.0, 0.05, (middle,)).astype(np.float32)
    weight1 = rng.normal(0.0, 0.125, (middle, OUT_FEATURES)).astype(np.float32)
    bias1 = rng.normal(0.0, 0.05, (OUT_FEATURES,)).astype(np.float32)
    sample = rng.normal(0.0, 1.0, (1, IN_FEATURES)).astype(np.float32)

    # Float reference for both slices, so each slice has its own golden and
    # teacher_forced can feed a slice its own boundary rather than a device
    # output.
    h0_a = sample @ weight0
    h1_a = h0_a + bias0
    hidden = np.maximum(h1_a, 0.0).astype(np.float32)
    h0_b = hidden @ weight1
    h1_b = h0_b + bias1
    output = np.maximum(h1_b, 0.0).astype(np.float32)

    from qairt_agent.vectors import VectorPreparer

    slices: list[dict[str, object]] = [
        {
            "name": "slice0",
            "weight": weight0,
            "bias": bias0,
            "input_name": "input",
            "output_name": "hidden",
            "inputs": {"input": sample},
            "goldens": {"hidden": hidden},
            "activations": {"input": sample, "h0": h0_a, "h1": h1_a, "hidden": hidden},
        },
        {
            "name": "slice1",
            "weight": weight1,
            "bias": bias1,
            "input_name": "hidden",
            "output_name": "output",
            "inputs": {"hidden": hidden},
            "goldens": {"output": output},
            "activations": {"hidden": hidden, "h0": h0_b, "h1": h1_b, "output": output},
        },
    ]

    produced: dict[str, str] = {}
    slice_manifests: dict[str, str] = {}
    for item in slices:
        name = str(item["name"])
        slice_dir = output_dir / name
        weight = item["weight"]
        bias = item["bias"]
        model_path = slice_dir / f"{name}.onnx"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        graph = helper.make_graph(
            [
                helper.make_node("MatMul", [item["input_name"], "w"], ["h0"], name="fc"),
                helper.make_node("Add", ["h0", "b"], ["h1"], name="bias_add"),
                helper.make_node("Relu", ["h1"], [item["output_name"]], name="act"),
            ],
            name,
            [
                helper.make_tensor_value_info(
                    str(item["input_name"]), TensorProto.FLOAT, [1, weight.shape[0]]
                )
            ],
            [
                helper.make_tensor_value_info(
                    str(item["output_name"]), TensorProto.FLOAT, [1, weight.shape[1]]
                )
            ],
            initializer=[
                helper.make_tensor(
                    "w", TensorProto.FLOAT, list(weight.shape), weight.ravel().tolist()
                ),
                helper.make_tensor(
                    "b", TensorProto.FLOAT, list(bias.shape), bias.ravel().tolist()
                ),
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_operatorsetid("", OPSET)]
        )
        model.ir_version = IR_VERSION
        onnx.save_model(model, model_path)

        encodings_path = build_encodings(
            slice_dir / f"{name}.encodings",
            activations=dict(item["activations"]),
            parameters={"w": weight, "b": bias},
        )
        manifest_path = VectorPreparer(slice_dir / "vectors").prepare_case(
            f"{name}-ar1",
            dict(item["inputs"]),
            goldens=dict(item["goldens"]),
            metadata={"purpose": "chain smoke slice", "slice": name},
        )
        slice_manifests[name] = str(manifest_path)

        spec = {
            "name": f"smoke-{name}",
            "preset": "vit",
            "sources": {
                "text": {"onnx": str(model_path), "encodings": str(encodings_path)}
            },
            "quantization": {"mode": "apply_encodings"},
            "sequence": {"ars": [1], "weight_sharing": False, "native_kv": False},
            "split": {"decoder_slice_count": 1, "split_lm_head": False},
            "transforms": {"mha2sha": False},
            "vectors": {"mode": "provided", "validation_manifest": str(manifest_path)},
            "benchmark": {"warmup_runs": 2, "measured_runs": 5, "optrace": False},
            "target": {"backend": "HTP", "name": target},
            "output_root": str(artifacts_root / name),
        }
        spec_path = slice_dir / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        produced[f"{name}_spec"] = str(spec_path)
        produced[f"{name}_model"] = str(model_path)
        produced[f"{name}_context"] = str(
            artifacts_root / name / "runs" / "<run_id>" / "build" / "contexts" / f"{name}.bin"
        )

    chain_manifest = VectorPreparer(output_dir / "vectors").prepare_case(
        "chain-ar1",
        {"input": sample},
        goldens={"output": output},
        metadata={"purpose": "chain smoke end-to-end"},
    )

    # The stage config both validate and benchmark take. Context paths are
    # filled in after the two builds, because each build mints its own run id.
    chain_config = {
        "ar": 1,
        "context_length": 4096,
        "vector_manifest": str(chain_manifest),
        "routes": [
            {
                "slice_id": "slice0",
                "input_names": ["input"],
                "output_names": ["hidden"],
                "graph_names": {"1": "slice0"},
            },
            {
                "slice_id": "slice1",
                "input_names": ["hidden"],
                "output_names": ["output"],
                "graph_names": {"1": "slice1"},
                "from_previous": {"hidden": "hidden"},
            },
        ],
        "contexts": {"slice0": "<fill in>", "slice1": "<fill in>"},
        "slice_vector_manifests": slice_manifests,
        "warmup_runs": 2,
        "measured_runs": 5,
        "aa_calibration": False,
    }
    chain_config_path = output_dir / "chain-stage-config.json"
    chain_config_path.write_text(json.dumps(chain_config, indent=2) + "\n")
    produced["chain_vector_manifest"] = str(chain_manifest)
    produced["chain_stage_config"] = str(chain_config_path)
    return produced


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="models/smoke",
        help="where to write the fixture (default: models/smoke)",
    )
    parser.add_argument(
        "--target",
        default="sm8750",
        help="target name from harness/targets/ to write into the spec",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/smoke",
        help=(
            "where the run writes artifacts (default: artifacts/smoke). It must "
            "not live under the models directory: the worker mounts that "
            "read-only, so a build there fails with EROFS."
        ),
    )
    parser.add_argument(
        "--chain",
        action="store_true",
        help=(
            "also emit a two-slice fixture whose shapes compose, for chain-scope "
            "device capture and multi-slice diagnostic execution"
        ),
    )
    arguments = parser.parse_args()

    output_dir = Path(arguments.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    weight = rng.normal(0.0, 0.125, (IN_FEATURES, OUT_FEATURES)).astype(np.float32)
    bias = rng.normal(0.0, 0.05, (OUT_FEATURES,)).astype(np.float32)
    sample = rng.normal(0.0, 1.0, (1, IN_FEATURES)).astype(np.float32)

    model_path = build_model(output_dir / "tiny.onnx", weight, bias)

    # The golden is the float graph's own output. It is what validation
    # compares the device against, so it is computed here rather than assumed.
    h0 = sample @ weight
    h1 = h0 + bias
    output = np.maximum(h1, 0.0).astype(np.float32)

    encodings_path = build_encodings(
        output_dir / "tiny.encodings",
        activations={"input": sample, "h0": h0, "h1": h1, "output": output},
        parameters={"w": weight, "b": bias},
    )

    from qairt_agent.vectors import VectorPreparer

    manifest_path = VectorPreparer(output_dir / "vectors").prepare_case(
        "smoke-ar1",
        {"input": sample},
        goldens={"output": output},
        metadata={"purpose": "smoke fixture", "generator": "tools/make_smoke_fixture.py"},
    )

    spec = {
        "name": "smoke",
        "preset": "vit",
        "sources": {
            "text": {
                "onnx": str(model_path),
                "encodings": str(encodings_path),
            }
        },
        "quantization": {"mode": "apply_encodings"},
        "sequence": {"ars": [1], "weight_sharing": False, "native_kv": False},
        "split": {"decoder_slice_count": 1, "split_lm_head": False},
        "transforms": {"mha2sha": False},
        "vectors": {"mode": "provided", "validation_manifest": str(manifest_path)},
        "benchmark": {"warmup_runs": 2, "measured_runs": 5, "optrace": False},
        "target": {"backend": "HTP", "name": arguments.target},
        "output_root": str(Path(arguments.artifacts_root).expanduser().resolve()),
    }
    spec_path = output_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    # The same fixture, wired for the layer-level float reference. `h0` and
    # `h1` are internal to the graph, so this is a real drilldown rather than a
    # boundary comparison wearing a layer label.
    debug_spec = json.loads(json.dumps(spec))
    debug_spec["name"] = "smoke-layer-debug"
    debug_spec["output_root"] = str(
        Path(str(spec["output_root"]) + "-layer-debug")
    )
    debug_spec["quality"] = {"dump_intermediates_on_failure": True}
    debug_spec["compile"] = {"enable_intermediate_outputs": True}
    debug_spec["stage_configs"] = {
        "validation": {
            "ar": 1,
            "float_reference": {
                "granularity": "layer",
                "ar": 1,
                "model_path": str(model_path),
                "providers": ["CPUExecutionProvider"],
            },
        }
    }
    debug_spec_path = output_dir / "spec-layer-debug.json"
    debug_spec_path.write_text(json.dumps(debug_spec, indent=2) + "\n")

    chain_outputs: dict[str, str] = {}
    if arguments.chain:
        chain_outputs = build_chain_fixture(
            output_dir / "chain",
            target=arguments.target,
            artifacts_root=Path(arguments.artifacts_root).expanduser().resolve()
            / "chain",
            rng=rng,
        )

    print(
        json.dumps(
            {
                "ok": True,
                **chain_outputs,
                "model": str(model_path),
                "encodings": str(encodings_path),
                "vector_manifest": str(manifest_path),
                "spec": str(spec_path),
                "layer_debug_spec": str(debug_spec_path),
                "next": f"qairt-agent plan --spec {spec_path}",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
