from __future__ import annotations

import hashlib
import io
import os
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qairt_agent.contracts import TensorRepresentation, VectorBundle, VectorTensor
from qairt_agent.diagnostics.sqnr import compute_sqnr
from qairt_agent.errors import PickleRejectedError
from qairt_agent.vectors_pickle import (
    RestrictedUnpickler,
    _looks_like_torch_archive,
    _torch_load_archive,
    import_pickle_artifacts,
    import_pickle_vectors,
    safe_load_pickle,
)
from qairt_agent.vectors import VectorPreparer


class Widget:
    """A module-level user class so it can be pickled by reference."""

    def __init__(self) -> None:
        self.value = 1


def _sample_payload() -> dict[str, Any]:
    return {
        "logits": np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(2, 3),
        "hidden": np.arange(4, dtype=np.float16).reshape(2, 2),
        "token_ids": np.array([1, 2, 3, 4], dtype=np.int64),
        "scale": np.float32(0.5),
    }


def _assert_trees_equal(expected: Any, actual: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(expected) == set(actual)
        for key in expected:
            _assert_trees_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple))
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            _assert_trees_equal(left, right)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert expected.dtype == actual.dtype
        assert expected.shape == actual.shape
        np.testing.assert_array_equal(expected, actual)
    elif isinstance(expected, np.generic):
        assert isinstance(actual, np.generic)
        assert expected.dtype == actual.dtype
        assert expected == actual
    else:
        assert type(expected) is type(actual)
        assert expected == actual


def test_safe_load_round_trips_nested_structure() -> None:
    payload = {
        "logits": np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(2, 3),
        "layers": [
            np.arange(4, dtype=np.float16).reshape(2, 2),
            {"ids": np.array([7, 8, 9], dtype=np.int64)},
        ],
        "scale": np.float32(0.25),
    }
    loaded = safe_load_pickle(pickle.dumps(payload))
    _assert_trees_equal(payload, loaded)


def test_safe_load_accepts_path_source(tmp_path: Path) -> None:
    payload = {"x": np.ones((2, 2), dtype=np.float32)}
    source = tmp_path / "vectors.pkl"
    source.write_bytes(pickle.dumps(payload))
    loaded = safe_load_pickle(source)
    _assert_trees_equal(payload, loaded)


def test_import_vectors_writes_raw_and_builds_bundle(tmp_path: Path) -> None:
    payload = {
        "logits": np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(2, 3),
        "layers": [
            {"hidden": np.arange(4, dtype=np.float16).reshape(2, 2)},
            {"hidden": np.ones((3,), dtype=np.int64)},
        ],
    }
    data = pickle.dumps(payload)
    out_dir = tmp_path / "out"

    bundle = import_pickle_vectors(data, output_dir=out_dir, trusted_local=True)

    assert isinstance(bundle, VectorBundle)
    assert bundle.source_sha256 == hashlib.sha256(data).hexdigest()
    assert bundle.bundle_id  # defaulted uuid4 hex
    names = {tensor.name for tensor in bundle.tensors}
    assert names == {"logits", "layers.0.hidden", "layers.1.hidden"}

    for tensor in bundle.tensors:
        assert isinstance(tensor, VectorTensor)
        assert tensor.representation == TensorRepresentation.LOGICAL_FP
        assert tensor.layout == "C"
        assert tensor.byte_order == "little"
        assert tensor.role == "golden"
        assert tensor.sha256 and tensor.nbytes is not None

    by_name = {tensor.name: tensor for tensor in bundle.tensors}
    assert by_name["logits"].shape == (2, 3)
    assert by_name["logits"].dtype == "float32"
    assert by_name["layers.0.hidden"].dtype == "float16"
    assert by_name["layers.1.hidden"].dtype == "int64"

    raw_files = list(out_dir.glob("*.raw"))
    assert len(raw_files) == 3
    written = {path.read_bytes() for path in raw_files}
    expected_logits = payload["logits"].astype("<f4").tobytes(order="C")
    assert expected_logits in written
    # The recorded hash matches the bytes actually written for that tensor.
    logits_file = next(
        path for path in raw_files if path.read_bytes() == expected_logits
    )
    assert logits_file.name.endswith(f"-{by_name['logits'].sha256[:12]}.raw")
    assert hashlib.sha256(expected_logits).hexdigest() == by_name["logits"].sha256
    assert len(expected_logits) == by_name["logits"].nbytes
    assert by_name["logits"].path == logits_file.resolve()


def test_import_artifacts_writes_consumable_sectioned_manifest(
    tmp_path: Path,
) -> None:
    payload = {
        "inputs": {
            "input_ids": np.array([[1, 2]], dtype=np.int64),
            "attention_mask": np.ones((1, 2), dtype=np.int32),
        },
        "goldens": {
            "logits": np.arange(6, dtype=np.float32).reshape(1, 2, 3),
        },
    }

    imported = import_pickle_artifacts(
        pickle.dumps(payload),
        output_dir=tmp_path / "imported",
        case_id="ar2",
        trusted_local=True,
    )

    assert imported.execution_ready is True
    assert imported.bundle_path.is_file()
    assert imported.manifest_path.is_file()
    np.testing.assert_array_equal(
        VectorPreparer.load_tensors(imported.manifest_path)["input_ids"],
        payload["inputs"]["input_ids"],
    )
    np.testing.assert_array_equal(
        VectorPreparer.load_tensors(
            imported.manifest_path,
            section="goldens",
        )["logits"],
        payload["goldens"]["logits"],
    )
    manifest = VectorPreparer.load_manifest(imported.manifest_path)
    assert manifest.metadata["source"] == "trusted_local_pickle"
    assert manifest.metadata["execution_ready"] is True
    assert all(
        not Path(record.path).is_absolute()
        for record in (*manifest.inputs.values(), *manifest.goldens.values())
    )
    assert all(
        tensor.path is not None and not tensor.path.is_absolute()
        for tensor in imported.bundle.tensors
    )


def test_import_artifacts_records_verified_external_source_provenance(
    tmp_path: Path,
) -> None:
    data = pickle.dumps({"goldens": {"logits": np.ones(2, dtype=np.float32)}})
    digest = hashlib.sha256(data).hexdigest()
    source_key = "/host/models/qwen/golden.pt"

    imported = import_pickle_artifacts(
        data,
        output_dir=tmp_path / "verified-source",
        trusted_local=True,
        source_key=source_key,
        expected_source_sha256=digest,
    )

    manifest = VectorPreparer.load_manifest(imported.manifest_path)
    assert imported.bundle.source_key == source_key
    assert imported.bundle.source_sha256 == digest
    assert imported.bundle.metadata["source_path"] == source_key
    assert manifest.metadata["source_path"] == source_key
    assert manifest.metadata["source_sha256"] == digest


def test_import_artifacts_rejects_external_source_hash_mismatch(
    tmp_path: Path,
) -> None:
    data = pickle.dumps({"goldens": {"logits": np.ones(2, dtype=np.float32)}})

    with pytest.raises(
        PickleRejectedError,
        match="does not match its declared provenance",
    ):
        import_pickle_artifacts(
            data,
            output_dir=tmp_path / "mismatched-source",
            trusted_local=True,
            source_key="/host/models/qwen/golden.pt",
            expected_source_sha256="0" * 64,
        )

    assert not (tmp_path / "mismatched-source").exists()


def test_import_artifacts_marks_unsectioned_pickle_golden_only(
    tmp_path: Path,
) -> None:
    imported = import_pickle_artifacts(
        pickle.dumps({"logits": np.ones((1, 4), dtype=np.float32)}),
        output_dir=tmp_path / "imported",
        trusted_local=True,
    )

    assert imported.execution_ready is False
    manifest = VectorPreparer.load_manifest(imported.manifest_path)
    assert manifest.inputs == {}
    assert set(manifest.goldens) == {"logits"}
    np.testing.assert_array_equal(
        VectorPreparer.load_tensors(
            imported.manifest_path,
            section="goldens",
        )["logits"],
        np.ones((1, 4), dtype=np.float32),
    )


def test_imported_golden_raw_is_consumable_by_sqnr(
    tmp_path: Path,
) -> None:
    reference = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    imported = import_pickle_artifacts(
        pickle.dumps({"logits": reference}),
        output_dir=tmp_path / "reference",
        trusted_local=True,
        section="goldens",
    )
    actual_manifest = VectorPreparer(tmp_path / "actual").prepare_case(
        "device-output",
        {"placeholder": np.ones((1,), dtype=np.int32)},
        goldens={"logits": reference + np.float32(0.01)},
    )

    golden = VectorPreparer.load_tensors(
        imported.manifest_path,
        section="goldens",
    )["logits"]
    actual = VectorPreparer.load_tensors(
        actual_manifest,
        section="goldens",
    )["logits"]
    measured = compute_sqnr(golden, actual)
    assert measured is not None
    assert measured > 40.0


def test_import_artifacts_can_assign_unsectioned_pickle_to_inputs(
    tmp_path: Path,
) -> None:
    payload = {
        "input_ids": np.array([[1, 2]], dtype=np.int64),
        "attention_mask": np.ones((1, 2), dtype=np.int32),
    }
    imported = import_pickle_artifacts(
        pickle.dumps(payload),
        output_dir=tmp_path / "imported",
        trusted_local=True,
        source_format="numpy-pickle",
        section="inputs",
    )

    assert imported.execution_ready is True
    assert imported.source_format == "numpy-pickle"
    assert imported.section == "inputs"
    manifest = VectorPreparer.load_manifest(imported.manifest_path)
    assert set(manifest.inputs) == {"input_ids", "attention_mask"}
    assert manifest.goldens == {}
    assert manifest.metadata["source_format"] == "numpy-pickle"
    assert manifest.metadata["section"] == "inputs"
    assert manifest.metadata["reference_priority"] == "onnxruntime_fallback_required"


def test_import_artifacts_rejects_flattened_name_collision(
    tmp_path: Path,
) -> None:
    payload = {
        "goldens": {
            "layer.logits": np.ones((1,), dtype=np.float32),
            "layer": {"logits": np.zeros((1,), dtype=np.float32)},
        }
    }
    with pytest.raises(PickleRejectedError, match="name collision"):
        import_pickle_artifacts(
            pickle.dumps(payload),
            output_dir=tmp_path / "imported",
            trusted_local=True,
        )


@pytest.mark.parametrize(
    "payload,section",
    [
        ({"inputs": {}, "goldens": {"y": np.ones(1, dtype=np.float32)}}, "auto"),
        ({"metadata": "not-a-tensor"}, "inputs"),
    ],
)
def test_import_artifacts_rejects_empty_selected_section(
    tmp_path: Path,
    payload: object,
    section: str,
) -> None:
    with pytest.raises(PickleRejectedError, match="contains no tensor leaves"):
        import_pickle_artifacts(
            pickle.dumps(payload),
            output_dir=tmp_path / section,
            trusted_local=True,
            section=section,
        )


def test_auto_detects_only_torch_save_shaped_zip() -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("archive/data.pkl", b"payload")
        stream.writestr("archive/version", b"3")
    assert _looks_like_torch_archive(archive.getvalue()) is True

    unrelated = io.BytesIO()
    with zipfile.ZipFile(unrelated, "w") as stream:
        stream.writestr("data.pkl", b"payload")
    assert _looks_like_torch_archive(unrelated.getvalue()) is False


def test_explicit_torch_format_rejects_plain_pickle_before_loading() -> None:
    with pytest.raises(PickleRejectedError, match="torch.save zip archives only"):
        safe_load_pickle(
            pickle.dumps({"logits": np.ones(1, dtype=np.float32)}),
            source_format="torch",
        )


def test_torch_loader_is_weights_only_cpu_and_normalizes_tensor() -> None:
    calls: dict[str, Any] = {}

    class FakeTensor:
        layout = "torch.strided"
        dtype = "torch.float32"
        is_quantized = False

        def detach(self) -> "FakeTensor":
            calls["detach"] = True
            return self

        def cpu(self) -> "FakeTensor":
            calls["cpu"] = True
            return self

        def numpy(self) -> np.ndarray:
            return np.array([1.0, 2.0], dtype=np.float32)

    class FakeTorch:
        Tensor = FakeTensor
        strided = "torch.strided"

        @staticmethod
        def load(
            source: io.BytesIO,
            *,
            weights_only: bool,
            map_location: str,
        ) -> object:
            calls["payload"] = source.read()
            calls["weights_only"] = weights_only
            calls["map_location"] = map_location
            return {"goldens": {"logits": FakeTensor()}}

    loaded = _torch_load_archive(b"torch-archive", torch_module=FakeTorch)
    np.testing.assert_array_equal(
        loaded["goldens"]["logits"],
        np.array([1.0, 2.0], dtype=np.float32),
    )
    assert calls == {
        "payload": b"torch-archive",
        "weights_only": True,
        "map_location": "cpu",
        "detach": True,
        "cpu": True,
    }


def test_torch_loader_fails_closed_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def reject_torch(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise ImportError("torch intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)
    with pytest.raises(PickleRejectedError, match="requires the pinned torch"):
        _torch_load_archive(b"payload")


def test_import_vectors_requires_trusted_local(tmp_path: Path) -> None:
    data = pickle.dumps({"x": np.ones(2, dtype=np.float32)})
    with pytest.raises(PickleRejectedError):
        import_pickle_vectors(data, output_dir=tmp_path / "out", trusted_local=False)


def test_rejects_disallowed_global_os_system() -> None:
    class Evil:
        def __reduce__(self):  # noqa: D401 - malicious by construction
            return (os.system, ("echo pwned",))

    data = pickle.dumps(Evil())
    with pytest.raises(PickleRejectedError):
        safe_load_pickle(data)


def test_rejects_custom_user_class() -> None:
    data = pickle.dumps(Widget())
    with pytest.raises(PickleRejectedError):
        safe_load_pickle(data)


def test_rejects_torch_tensor() -> None:
    torch = pytest.importorskip("torch")
    data = pickle.dumps(torch.zeros(2))
    with pytest.raises(PickleRejectedError):
        safe_load_pickle(data)


def test_import_torch_save_archive_to_raw_manifest_when_available(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    archive = io.BytesIO()
    torch.save(
        {
            "inputs": {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            },
            "goldens": {
                "logits": torch.tensor([[0.25, -0.5]], dtype=torch.float32),
            },
        },
        archive,
    )

    imported = import_pickle_artifacts(
        archive.getvalue(),
        output_dir=tmp_path / "torch-import",
        trusted_local=True,
        source_format="torch",
    )
    assert imported.source_format == "torch"
    assert imported.execution_ready is True
    np.testing.assert_array_equal(
        VectorPreparer.load_tensors(imported.manifest_path)["input_ids"],
        np.array([[1, 2]], dtype=np.int64),
    )
    np.testing.assert_allclose(
        VectorPreparer.load_tensors(
            imported.manifest_path,
            section="goldens",
        )["logits"],
        np.array([[0.25, -0.5]], dtype=np.float32),
    )


def test_rejects_object_dtype_array() -> None:
    data = pickle.dumps(np.array([object(), object()], dtype=object))
    with pytest.raises(PickleRejectedError):
        safe_load_pickle(data)


def test_rejects_oversized_input() -> None:
    data = pickle.dumps({"x": np.ones(4, dtype=np.float32)})
    with pytest.raises(PickleRejectedError):
        safe_load_pickle(data, max_input_bytes=4)


def test_restricted_unpickler_is_the_gate() -> None:
    unpickler = RestrictedUnpickler(io.BytesIO(b""))
    with pytest.raises(PickleRejectedError):
        unpickler.find_class("os", "system")
    # An allowlisted NumPy global still resolves.
    assert unpickler.find_class("numpy", "ndarray") is np.ndarray


def test_isolate_matches_in_process() -> None:
    pytest.importorskip("resource")
    payload = _sample_payload()
    data = pickle.dumps(payload)
    in_process = safe_load_pickle(data, isolate=False)
    isolated = safe_load_pickle(data, isolate=True, timeout=60.0)
    _assert_trees_equal(payload, in_process)
    _assert_trees_equal(in_process, isolated)


def test_isolate_rejects_malicious_input() -> None:
    pytest.importorskip("resource")

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    with pytest.raises(PickleRejectedError):
        safe_load_pickle(pickle.dumps(Evil()), isolate=True, timeout=60.0)
