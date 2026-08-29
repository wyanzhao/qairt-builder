from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qairt_agent.vectors import TensorSource, VectorPreparer


def test_prepare_load_and_write_input_list(tmp_path: Path) -> None:
    preparer = VectorPreparer(tmp_path / "vectors")
    input_ids = np.arange(8, dtype=np.int32).reshape(1, 8)
    logits = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(1, 3, 4)

    manifest_path = preparer.prepare_case(
        "qwen/ar8",
        {"input_ids": input_ids},
        goldens={"logits": logits},
        roles={"input_ids": "tokens", "logits": "logits"},
        metadata={"ar": 8, "cl": 128},
    )
    digest = preparer.manifest_sha256(manifest_path)
    manifest = preparer.load_manifest(manifest_path, expected_sha256=digest)

    assert manifest.case_id == "qwen/ar8"
    assert manifest.metadata == {"ar": 8, "cl": 128}
    np.testing.assert_array_equal(
        preparer.load_tensors(manifest_path, section="inputs")["input_ids"],
        input_ids,
    )
    np.testing.assert_array_equal(
        preparer.load_tensors(manifest_path, section="goldens")["logits"],
        logits,
    )

    input_list = preparer.write_input_list([manifest_path], tmp_path / "input_list.txt")
    line = input_list.read_text(encoding="utf-8").strip()
    assert line.startswith("input_ids:=")
    assert Path(line.split(":=", 1)[1]).is_file()


def test_import_raw_source_and_detect_tampering(tmp_path: Path) -> None:
    raw_path = tmp_path / "mask.raw"
    mask = np.arange(6, dtype=np.uint16).reshape(2, 3)
    mask.tofile(raw_path)

    preparer = VectorPreparer(tmp_path / "vectors")
    manifest_path = preparer.import_case(
        "raw-case",
        {"attention_mask": TensorSource(raw_path, dtype=np.uint16, shape=(2, 3))},
    )
    loaded = preparer.load_tensors(manifest_path)
    np.testing.assert_array_equal(loaded["attention_mask"], mask)

    manifest = preparer.load_manifest(manifest_path)
    record = manifest.inputs["attention_mask"]
    stored_path = manifest_path.parent / record.path
    stored_path.write_bytes(stored_path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        preparer.load_tensors(manifest_path)


def test_capture_reference_publishes_new_manifest(tmp_path: Path) -> None:
    preparer = VectorPreparer(tmp_path / "vectors")
    manifest_path = preparer.prepare_case(
        "capture",
        {"x": np.array([1.0, 2.0], dtype=np.float32)},
    )

    def reference(inputs: dict[str, np.ndarray], names: tuple[str, ...]) -> dict[str, np.ndarray]:
        assert names == ("twice",)
        return {"twice": inputs["x"] * 2, "unused": inputs["x"] * 3}

    captured_path = preparer.capture_reference(
        manifest_path,
        reference,
        output_names=("twice",),
    )
    captured = preparer.load_manifest(captured_path)
    assert captured.source_manifest_sha256 == preparer.manifest_sha256(manifest_path)
    assert set(captured.goldens) == {"twice"}
    np.testing.assert_array_equal(
        preparer.load_tensors(captured_path, section="goldens")["twice"],
        np.array([2.0, 4.0], dtype=np.float32),
    )


def test_raw_source_requires_shape_and_dtype(tmp_path: Path) -> None:
    raw_path = tmp_path / "x.raw"
    raw_path.write_bytes(b"\x00\x01")
    preparer = VectorPreparer(tmp_path / "vectors")

    with pytest.raises(ValueError, match="requires explicit dtype and shape"):
        preparer.prepare_case("invalid", {"x": raw_path})


def test_manifest_loader_accepts_artifact_like_object(tmp_path: Path) -> None:
    preparer = VectorPreparer(tmp_path / "vectors")
    manifest_path = preparer.prepare_case("artifact-ref", {"x": np.ones(1, dtype=np.float32)})
    artifact = SimpleNamespace(
        path=manifest_path,
        sha256=preparer.manifest_sha256(manifest_path),
    )

    assert preparer.load_manifest(artifact).case_id == "artifact-ref"


def _external_data_model(directory: Path) -> Path:
    """A float graph whose initializers live in a side-car data file."""

    import onnx
    from onnx import TensorProto, helper

    directory.mkdir(parents=True, exist_ok=True)
    weights = np.full((64,), 2.0, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "bias"], ["h0"], name="layer0"),
            helper.make_node("Identity", ["h0"], ["y"], name="head"),
        ],
        "external-data",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [64])],
        initializer=[
            # raw=True: only raw_data tensors are eligible for external storage.
            helper.make_tensor(
                "bias", TensorProto.FLOAT, [64], weights.tobytes(), raw=True
            )
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    model.ir_version = 9
    path = directory / "model.onnx"
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="external-data.bin",
        size_threshold=0,
    )
    return path


def test_float_capture_resolves_external_data_and_restores_the_directory(
    tmp_path: Path,
) -> None:
    model = _external_data_model(tmp_path / "model")
    before = sorted(item.name for item in model.parent.iterdir())
    assert "external-data.bin" in before

    values, provenance = VectorPreparer.capture_onnx_float_activations(
        model,
        {"x": np.ones((64,), dtype=np.float32)},
        ["h0"],
    )

    np.testing.assert_allclose(values["h0"], np.full((64,), 3.0, dtype=np.float32))
    assert provenance["reference_source"] == "onnxruntime_float"
    assert provenance["promoted_tensors"] == ["h0"]
    assert [Path(item).name for item in provenance["reference_model_external_data"]] == [
        "external-data.bin"
    ]
    # The instrumented copy written beside the model is always cleaned up.
    assert sorted(item.name for item in model.parent.iterdir()) == before


def test_producible_tensor_names_include_internal_activations(
    tmp_path: Path,
) -> None:
    model = _external_data_model(tmp_path / "model")

    names = VectorPreparer.onnx_producible_tensor_names(model)

    assert {"x", "bias", "h0", "y"} <= names
    assert "not_a_tensor" not in names
