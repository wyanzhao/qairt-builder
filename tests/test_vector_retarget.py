from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from qairt_agent.vector_retarget import (
    VectorRetargetError,
    inspect_onnx_abi,
    retarget_vector_manifest,
    validate_provided_ar_manifest,
)
from qairt_agent.vectors import VectorPreparer


def _identity_model(
    path: Path,
    inputs: Sequence[tuple[str, int, Sequence[int | str | None]]],
) -> Path:
    graph_inputs = [
        helper.make_tensor_value_info(name, dtype, list(shape))
        for name, dtype, shape in inputs
    ]
    graph_outputs = [
        helper.make_tensor_value_info(f"{name}_out", dtype, list(shape))
        for name, dtype, shape in inputs
    ]
    nodes = [
        helper.make_node("Identity", [name], [f"{name}_out"])
        for name, _, _ in inputs
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "vector-retarget-test", graph_inputs, graph_outputs),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    # ONNX Runtime 1.17 supports IR <= 9.
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save_model(model, path)
    return path


def test_retarget_crops_pads_casts_rebuilds_positions_and_captures_ort(
    tmp_path: Path,
) -> None:
    source_preparer = VectorPreparer(tmp_path / "source")
    source = source_preparer.prepare_case(
        "ar6-cl2",
        {
            "input_ids": np.arange(6, dtype=np.int32).reshape(1, 6),
            "position_ids": np.array([[10, 13, 99, 2, 3, 4]], dtype=np.int64),
            "past_key_0_in": np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2),
            "unused_source": np.ones((1,), dtype=np.float32),
        },
        goldens={"stale": np.ones((1,), dtype=np.float32)},
        metadata={"source_ar": 6, "source_cl": 2},
    )
    target = _identity_model(
        tmp_path / "ar4-cl4.onnx",
        (
            ("input_ids", TensorProto.INT64, (1, 4)),
            ("position_ids", TensorProto.INT64, (1, 4)),
            ("past_key_0_in", TensorProto.FLOAT, (1, 1, 4, 2)),
        ),
    )

    result = retarget_vector_manifest(
        source,
        target,
        family="qwen3-4b",
        ar=4,
        cl=4,
        output_dir=tmp_path / "retargeted",
    )
    manifest = VectorPreparer.load_manifest(result)
    inputs = VectorPreparer.load_tensors(result)
    goldens = VectorPreparer.load_tensors(result, section="goldens")

    assert manifest.metadata["reference_source"] == "onnxruntime"
    assert manifest.metadata["reference_priority"] == "onnxruntime_fallback"
    assert manifest.metadata["retarget"]["dropped_source_inputs"] == ["unused_source"]
    assert manifest.metadata["source_vector_manifest"]["supplied_goldens_ignored"] == ["stale"]
    assert set(inputs) == {"input_ids", "position_ids", "past_key_0_in"}
    assert set(goldens) == {
        "input_ids_out",
        "position_ids_out",
        "past_key_0_in_out",
    }

    assert inputs["input_ids"].dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(
        inputs["input_ids"],
        np.array([[0, 1, 2, 3]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        inputs["position_ids"],
        np.array([[10, 11, 12, 13]], dtype=np.int64),
    )
    expected_cache = np.zeros((1, 1, 4, 2), dtype=np.float32)
    expected_cache[:, :, :2, :] = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    np.testing.assert_array_equal(inputs["past_key_0_in"], expected_cache)
    np.testing.assert_array_equal(goldens["input_ids_out"], inputs["input_ids"])
    np.testing.assert_array_equal(goldens["position_ids_out"], inputs["position_ids"])
    np.testing.assert_array_equal(goldens["past_key_0_in_out"], expected_cache)

    transforms = manifest.metadata["retarget"]["input_transforms"]
    assert transforms["input_ids"]["dtype_transform"] == "safe_cast"
    assert transforms["input_ids"]["axis_transforms"][1]["operation"] == "prefix_crop"
    assert transforms["position_ids"]["value_transform"] == "contiguous_position_ids"
    assert transforms["position_ids"]["position_start"] == 10
    assert transforms["past_key_0_in"]["axis_transforms"][2]["operation"] == "zero_pad"


def test_retarget_resolves_named_dynamic_sequence_dimension(tmp_path: Path) -> None:
    preparer = VectorPreparer(tmp_path / "source")
    source = preparer.prepare_case(
        "dynamic-source",
        {"input_ids": np.arange(5, dtype=np.int64).reshape(1, 5)},
    )
    target = _identity_model(
        tmp_path / "dynamic.onnx",
        (("input_ids", TensorProto.INT64, ("batch", "sequence")),),
    )

    result = retarget_vector_manifest(
        source,
        target,
        family="qwen3",
        ar=3,
        cl=8,
        output_dir=tmp_path / "retargeted",
    )
    inputs = VectorPreparer.load_tensors(result)
    assert inputs["input_ids"].shape == (1, 3)
    manifest = VectorPreparer.load_manifest(result)
    resolution = manifest.metadata["retarget"]["input_transforms"]["input_ids"][
        "shape_resolution"
    ]
    assert [item["resolution"] for item in resolution] == [
        "source_batch",
        "symbolic_ar",
    ]


def test_retarget_qwen_rope_cos_sin_uses_penultimate_ar_axis(
    tmp_path: Path,
) -> None:
    source_cos = np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2)
    source_sin = source_cos + 100
    source = VectorPreparer(tmp_path / "source").prepare_case(
        "qwen-rope-ar4",
        {
            "position_ids_cos": source_cos,
            "position_ids_sin": source_sin,
        },
    )
    target = _identity_model(
        tmp_path / "qwen-rope-ar1.onnx",
        (
            ("position_ids_cos", TensorProto.FLOAT, (1, 1, 1, 2)),
            ("position_ids_sin", TensorProto.FLOAT, (1, 1, 1, 2)),
        ),
    )

    result = retarget_vector_manifest(
        source,
        target,
        family="qwen3-4b",
        ar=1,
        cl=4096,
        output_dir=tmp_path / "retargeted",
    )
    inputs = VectorPreparer.load_tensors(result)
    np.testing.assert_array_equal(
        inputs["position_ids_cos"],
        source_cos[:, :, :1, :],
    )
    np.testing.assert_array_equal(
        inputs["position_ids_sin"],
        source_sin[:, :, :1, :],
    )
    transforms = VectorPreparer.load_manifest(result).metadata["retarget"][
        "input_transforms"
    ]
    assert transforms["position_ids_cos"]["semantic_role"] == "rotary_position"
    assert (
        transforms["position_ids_cos"]["value_transform"]
        == "prefix_crop_rotary_table"
    )
    assert transforms["position_ids_cos"]["axis_transforms"][2] == {
        "axis": 2,
        "source_size": 4,
        "target_size": 1,
        "operation": "prefix_crop",
    }


def test_retarget_rejects_zero_padded_rope_extension(tmp_path: Path) -> None:
    source = VectorPreparer(tmp_path / "source").prepare_case(
        "qwen-rope-ar1",
        {
            "position_ids_cos": np.ones((1, 1, 1, 2), dtype=np.float32),
        },
    )
    target = _identity_model(
        tmp_path / "qwen-rope-ar4.onnx",
        (("position_ids_cos", TensorProto.FLOAT, (1, 1, 4, 2)),),
    )

    with pytest.raises(VectorRetargetError, match="cannot be extended"):
        retarget_vector_manifest(
            source,
            target,
            family="qwen3",
            ar=4,
            cl=4096,
            output_dir=tmp_path / "retargeted",
        )


def test_retarget_fails_closed_for_unknown_dynamic_dimension(tmp_path: Path) -> None:
    source = VectorPreparer(tmp_path / "source").prepare_case(
        "unknown-dynamic",
        {"feature": np.ones((1, 5), dtype=np.float32)},
    )
    target = _identity_model(
        tmp_path / "unknown-dynamic.onnx",
        (("feature", TensorProto.FLOAT, ("batch", "mystery")),),
    )

    with pytest.raises(VectorRetargetError, match="unresolved dynamic dimension"):
        retarget_vector_manifest(
            source,
            target,
            family="qwen3",
            ar=3,
            cl=8,
            output_dir=tmp_path / "retargeted",
        )


def test_retarget_synthesizes_known_inputs_but_not_unknown_inputs(tmp_path: Path) -> None:
    preparer = VectorPreparer(tmp_path / "source")
    source = preparer.prepare_case(
        "missing-known",
        {"input_ids": np.array([[5]], dtype=np.int64)},
    )
    target = _identity_model(
        tmp_path / "known.onnx",
        (
            ("input_ids", TensorProto.INT64, (1, 1)),
            ("position_ids", TensorProto.INT64, (1, 1)),
            ("past_value_0_in", TensorProto.FLOAT, (1, 1, 4, 2)),
        ),
    )
    result = retarget_vector_manifest(
        source,
        target,
        family="qwen3",
        ar=1,
        cl=4,
        output_dir=tmp_path / "known-output",
    )
    inputs = VectorPreparer.load_tensors(result)
    np.testing.assert_array_equal(inputs["position_ids"], np.array([[0]], dtype=np.int64))
    np.testing.assert_array_equal(
        inputs["past_value_0_in"],
        np.zeros((1, 1, 4, 2), dtype=np.float32),
    )

    unknown_target = _identity_model(
        tmp_path / "unknown.onnx",
        (
            ("input_ids", TensorProto.INT64, (1, 1)),
            ("adapter_scale", TensorProto.FLOAT, (1,)),
        ),
    )
    with pytest.raises(VectorRetargetError, match="no safe family-aware synthesis rule"):
        retarget_vector_manifest(
            source,
            unknown_target,
            family="qwen3",
            ar=1,
            cl=4,
            output_dir=tmp_path / "unknown-output",
        )


def test_retarget_rejects_unsafe_dtype_and_wrong_static_ar(tmp_path: Path) -> None:
    unsafe = VectorPreparer(tmp_path / "unsafe-source").prepare_case(
        "unsafe",
        {"input_ids": np.array([[1.5]], dtype=np.float64)},
    )
    target = _identity_model(
        tmp_path / "unsafe.onnx",
        (("input_ids", TensorProto.FLOAT, (1, 1)),),
    )
    with pytest.raises(VectorRetargetError, match="cannot be safely cast"):
        retarget_vector_manifest(
            unsafe,
            target,
            family="qwen3",
            ar=1,
            cl=4,
            output_dir=tmp_path / "unsafe-output",
        )

    ids = VectorPreparer(tmp_path / "ids-source").prepare_case(
        "wrong-ar",
        {"input_ids": np.arange(4, dtype=np.int64).reshape(1, 4)},
    )
    with pytest.raises(VectorRetargetError, match="proves AR=1"):
        retarget_vector_manifest(
            ids,
            target,
            family="qwen3",
            ar=2,
            cl=4,
            output_dir=tmp_path / "wrong-ar-output",
        )


def test_qwen35_requires_provided_per_ar_manifest_and_prefers_golden(
    tmp_path: Path,
) -> None:
    target = _identity_model(
        tmp_path / "qwen35-ar1.onnx",
        (("input_ids", TensorProto.INT64, (1, 1)),),
    )
    provided = VectorPreparer(tmp_path / "provided").prepare_case(
        "qwen35-ar1",
        {"input_ids": np.array([[7]], dtype=np.int64)},
        goldens={"input_ids_out": np.array([[7]], dtype=np.int64)},
        metadata={"family": "qwen3_5", "ar": 1, "cl": 4096},
    )

    with pytest.raises(VectorRetargetError, match="independent per-AR"):
        retarget_vector_manifest(
            provided,
            target,
            family="qwen3.5",
            ar=1,
            cl=4096,
            output_dir=tmp_path / "retargeted",
        )

    binding = validate_provided_ar_manifest(
        provided,
        target,
        family="qwen3.5",
        ar=1,
        cl=4096,
    )
    assert binding.reference_source == "provided_golden"
    assert binding.needs_onnxruntime_capture is False
    assert binding.golden_names == ("input_ids_out",)
    assert binding.to_dict()["input_abi"][0]["shape"] == [1, 1]


def test_provided_binding_selects_ort_fallback_and_validates_exact_abi(
    tmp_path: Path,
) -> None:
    target = _identity_model(
        tmp_path / "qwen35-ar1.onnx",
        (("input_ids", TensorProto.INT64, (1, 1)),),
    )
    no_golden = VectorPreparer(tmp_path / "provided").prepare_case(
        "no-golden",
        {"input_ids": np.array([[7]], dtype=np.int64)},
    )
    binding = validate_provided_ar_manifest(
        no_golden,
        target,
        family="qwen3.5-omni-thinker",
        ar=1,
        cl=4096,
    )
    assert binding.reference_source == "onnxruntime"
    assert binding.needs_onnxruntime_capture is True

    wrong_shape = VectorPreparer(tmp_path / "wrong").prepare_case(
        "wrong",
        {"input_ids": np.array([[7, 8]], dtype=np.int64)},
    )
    with pytest.raises(VectorRetargetError, match="shape mismatch"):
        validate_provided_ar_manifest(
            wrong_shape,
            target,
            family="qwen3.5",
            ar=1,
            cl=4096,
        )


def test_inspect_onnx_abi_excludes_initializer_inputs(tmp_path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    weight_input = helper.make_tensor_value_info("weight", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [1], [2.0])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "weight"], ["y"])],
        "initializer-input",
        [x, weight_input],
        [y],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "initializer-input.onnx"
    onnx.save_model(model, path)

    inputs, outputs = inspect_onnx_abi(path)
    assert [item.name for item in inputs] == ["x"]
    assert [item.name for item in outputs] == ["y"]
