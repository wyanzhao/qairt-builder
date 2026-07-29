from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

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


def test_capture_reference_never_overwrites_supplied_golden(tmp_path: Path) -> None:
    preparer = VectorPreparer(tmp_path / "vectors")
    supplied = np.array([7.0, 8.0], dtype=np.float32)
    manifest_path = preparer.prepare_case(
        "capture-preserves-golden",
        {"x": np.array([1.0, 2.0], dtype=np.float32)},
        goldens={"y": supplied},
    )

    captured_path = preparer.capture_reference(
        manifest_path,
        lambda inputs: {
            "y": np.array([-1.0, -1.0], dtype=np.float32),
            "z": inputs["x"] * 3,
        },
    )
    goldens = preparer.load_tensors(captured_path, section="goldens")

    np.testing.assert_array_equal(goldens["y"], supplied)
    np.testing.assert_array_equal(
        goldens["z"],
        np.array([3.0, 6.0], dtype=np.float32),
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


@pytest.mark.parametrize("record_path", ["../outside.raw", "/tmp/outside.raw"])
def test_tensor_paths_cannot_escape_vector_case(
    tmp_path: Path,
    record_path: str,
) -> None:
    case = tmp_path / "case"
    case.mkdir()
    outside = tmp_path / "outside.raw"
    outside.write_bytes(np.array([1], dtype=np.int32).tobytes())
    manifest = {
        "schema": "qairt-agent.vector-manifest",
        "case_id": "escape",
        "inputs": {
            "x": {
                "name": "x",
                "path": record_path,
                "dtype": "int32",
                "shape": [1],
                "sha256": "0" * 64,
                "nbytes": 4,
                "storage": "raw",
                "byte_order": "little",
            }
        },
        "goldens": {},
        "metadata": {},
    }
    manifest_path = case / "vector_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="relative|escapes"):
        VectorPreparer.load_tensors(manifest_path)


def test_loaded_raw_tensors_are_writable(tmp_path: Path) -> None:
    preparer = VectorPreparer(tmp_path / "vectors")
    manifest_path = preparer.prepare_case(
        "writable",
        {"x": np.array([1], dtype=np.int32)},
    )

    loaded = preparer.load_tensors(manifest_path)["x"]
    loaded[0] = 2

    assert loaded.tolist() == [2]
