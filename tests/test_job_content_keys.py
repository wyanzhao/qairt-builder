from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from qairt_agent.agent import QairtAgentClient
from qairt_agent.contracts import (
    ArtifactKind,
    ArtifactRef,
    JobState,
    StageProvenance,
    ToolResult,
)
from qairt_agent.errors import ArtifactIntegrityError
from qairt_agent.jobs.journal import JobJournal
from qairt_agent.jobs.worker import WorkflowWorker
from qairt_agent.vectors import VectorPreparer

PROVENANCE = StageProvenance(
    sdk_build="260626120635",
    adapter_capability="explicit_factory",
    platform_abi="ubuntu22.04-x86_64",
    resolved_preset_sha256="a" * 64,
)


def _write_model(
    path: Path,
    weight: float,
    *,
    external_data: bool = False,
) -> Path | None:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    initializer = numpy_helper.from_array(
        np.full((1, 2), weight, dtype=np.float32),
        name="weight",
    )
    graph = helper.make_graph(
        [helper.make_node("Add", ["input", "weight"], ["output"])],
        "content-key-test",
        [input_info],
        [output_info],
        [initializer],
    )
    model = helper.make_model(graph)
    if not external_data:
        onnx.save_model(model, path)
        return None

    external_path = path.with_suffix(".data")
    if external_path.exists():
        external_path.unlink()
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_path.name,
        size_threshold=0,
        convert_attribute=False,
    )
    return external_path


def _make_spec(tmp_path: Path, *, external_data: bool = False) -> tuple[dict[str, Any], Path]:
    model = tmp_path / "model.onnx"
    external_path = _write_model(model, 1.0, external_data=external_data)
    encodings = tmp_path / "model.encodings"
    encodings.write_text('{"version": 1}\n', encoding="utf-8")

    config_dir = tmp_path / "model-config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text('{"hidden_size": 8}\n', encoding="utf-8")
    # Large, non-config blobs beside config.json are intentionally not part of
    # the config-directory identity.
    (config_dir / "weights.bin").write_bytes(b"not-a-config")

    attached_model = tmp_path / "ar128.onnx"
    _write_model(attached_model, 128.0)
    attached_encodings = tmp_path / "ar128.encodings"
    attached_encodings.write_text('{"version": 128}\n', encoding="utf-8")

    vector_manifest = VectorPreparer(tmp_path / "vectors").prepare_case(
        "ar1",
        {"input_ids": np.array([[1]], dtype=np.int32)},
        goldens={"logits": np.array([[0.5]], dtype=np.float32)},
    )
    spec = {
        "preset": "qwen3_dense",
        "output_root": str(tmp_path / "artifacts"),
        "sources": {
            "text": {
                "onnx_path": str(model),
                "encodings_path": str(encodings),
                "config_path": str(config_dir),
            }
        },
        "sequence": {"ars": [1, 128]},
        "split": {"decoder_slice_count": 1},
        "transforms": {"mha2sha": True},
        "quantization": {"mode": "apply_encodings"},
        "compile": {},
        "target": {"chipset": "SM8850"},
        "vectors": {
            "mode": "provided",
            "validation_manifest": str(vector_manifest),
        },
        "metadata": {
            "attached_models_by_ar": {
                "128": {
                    "model_path": str(attached_model),
                    "encodings_path": str(attached_encodings),
                }
            }
        },
        "stage_configs": {
            "build": {},
            "validation": {},
            "benchmark": {},
            "diagnose": {"kind": "quality", "config": {}},
        },
    }
    return spec, external_path or model


def _worker(tmp_path: Path, spec: dict[str, Any], name: str = "job") -> WorkflowWorker:
    journal = JobJournal.create(
        tmp_path / "jobs",
        name,
        spec_original=spec,
        spec_resolved={},
        spec_sha256="b" * 64,
    )
    return WorkflowWorker(
        journal,
        spec=spec,
        resolved={},
        provenance=PROVENANCE,
        stage_runner=lambda _ctx: None,  # type: ignore[arg-type,return-value]
        stages=("build",),
    )


def _mutate_vector_and_manifest(manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = payload["inputs"]["input_ids"]
    raw_path = manifest_path.parent / record["path"]
    replacement = np.array([[9]], dtype=np.int32).tobytes()
    raw_path.write_bytes(replacement)
    record["sha256"] = hashlib.sha256(replacement).hexdigest()
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda spec: _write_model(Path(spec["sources"]["text"]["onnx_path"]), 2.0),
        lambda spec: Path(spec["sources"]["text"]["encodings_path"]).write_text(
            '{"version": 2}\n', encoding="utf-8"
        ),
        lambda spec: (
            Path(spec["sources"]["text"]["config_path"]) / "config.json"
        ).write_text('{"hidden_size": 16}\n', encoding="utf-8"),
        lambda spec: _write_model(
            Path(spec["metadata"]["attached_models_by_ar"]["128"]["model_path"]),
            256.0,
        ),
        lambda spec: Path(
            spec["metadata"]["attached_models_by_ar"]["128"]["encodings_path"]
        ).write_text('{"version": 256}\n', encoding="utf-8"),
        lambda spec: _mutate_vector_and_manifest(
            Path(spec["vectors"]["validation_manifest"])
        ),
    ],
    ids=[
        "onnx",
        "aimet-encodings",
        "config-directory",
        "attached-ar-model",
        "attached-ar-encodings",
        "vector-raw-and-manifest",
    ],
)
def test_build_key_invalidates_when_input_changes_at_same_path(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    spec, _ = _make_spec(tmp_path)
    worker = _worker(tmp_path, spec)
    before = worker._stage_key("build", None)  # noqa: SLF001

    mutate(spec)

    assert worker._stage_key("build", None) != before  # noqa: SLF001


def test_build_key_includes_onnx_external_data_bytes(tmp_path: Path) -> None:
    spec, external_path = _make_spec(tmp_path, external_data=True)
    worker = _worker(tmp_path, spec)
    before = worker._stage_key("build", None)  # noqa: SLF001

    original = external_path.read_bytes()
    external_path.write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])

    assert worker._stage_key("build", None) != before  # noqa: SLF001


def test_rerun_does_not_reuse_build_after_same_path_model_mutation(
    tmp_path: Path,
) -> None:
    class BuildEngine:
        def __init__(self) -> None:
            self.calls = 0

        def build(self, _spec: object) -> ToolResult[dict[str, int]]:
            self.calls += 1
            manifest_path = tmp_path / f"build-{self.calls}.manifest.json"
            manifest_path.write_text(
                json.dumps({"build": self.calls}) + "\n",
                encoding="utf-8",
            )
            return ToolResult.success(
                {"build": self.calls},
                manifest=ArtifactRef.from_path(
                    manifest_path,
                    kind=ArtifactKind.MANIFEST,
                ),
            )

    spec, _ = _make_spec(tmp_path)
    engine = BuildEngine()
    client = QairtAgentClient(
        jobs_root=tmp_path / "client-jobs",
        engine_factory=lambda: engine,
        background=False,
        provenance=PROVENANCE,
    )
    first = client.submit(spec, stages=("build",))
    assert first.wait().state == JobState.SUCCEEDED

    _write_model(Path(spec["sources"]["text"]["onnx_path"]), 3.0)
    second = client.rerun(first.job_id)

    assert second.wait().state == JobState.SUCCEEDED
    assert engine.calls == 2
    assert not second.journal.receipts()[0].metrics.get("reused_from_parent", False)


def test_vector_manifest_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    spec, _ = _make_spec(tmp_path)
    worker = _worker(tmp_path, spec)
    worker._stage_key("build", None)  # noqa: SLF001

    manifest_path = Path(spec["vectors"]["validation_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = manifest_path.parent / payload["inputs"]["input_ids"]["path"]
    raw_path.write_bytes(b"\x09\x00\x00\x00")

    with pytest.raises(ArtifactIntegrityError, match="does not match its manifest"):
        worker._stage_key("build", None)  # noqa: SLF001


def test_quality_dump_policy_invalidates_build_key(tmp_path: Path) -> None:
    spec, _ = _make_spec(tmp_path)
    spec["quality"] = {
        "sqnr_modes": [],
        "dump_intermediates_on_failure": False,
    }
    before = _worker(tmp_path, spec, "quality-before")._stage_key(  # noqa: SLF001
        "build",
        None,
    )

    spec["quality"] = {
        "sqnr_modes": ["teacher_forced", "chain"],
        "dump_intermediates_on_failure": True,
    }
    after = _worker(tmp_path, spec, "quality-after")._stage_key(  # noqa: SLF001
        "build",
        None,
    )

    assert after != before


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("name", "renamed-build"),
        ("output_root", "relocated-artifacts"),
    ],
)
def test_build_identity_fields_invalidate_build_key(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    spec, _ = _make_spec(tmp_path)
    spec["name"] = "original-build"
    worker = _worker(tmp_path, spec, f"{field}-before")
    before = worker._stage_key("build", None)  # noqa: SLF001

    spec[field] = (
        str(tmp_path / replacement)
        if field == "output_root"
        else replacement
    )

    assert worker._stage_key("build", None) != before  # noqa: SLF001


@pytest.mark.parametrize(
    ("stage_name", "stage_configs"),
    [
        (
            "validate",
            {
                "validation": {"actual_trace": "{input}"},
                "benchmark": {},
                "diagnose": {"kind": "quality", "config": {}},
            },
        ),
        (
            "benchmark",
            {
                "validation": {},
                "benchmark": {"context_path": "{input}"},
                "diagnose": {"kind": "quality", "config": {}},
            },
        ),
        (
            "diagnose",
            {
                "validation": {},
                "benchmark": {},
                "diagnose": {
                    "kind": "quality",
                    "config": {"reference_trace": "{input}"},
                },
            },
        ),
    ],
)
def test_continuation_key_hashes_its_stage_config_inputs(
    tmp_path: Path,
    stage_name: str,
    stage_configs: dict[str, Any],
) -> None:
    spec, _ = _make_spec(tmp_path)
    stage_input = tmp_path / f"{stage_name}.bin"
    stage_input.write_bytes(b"before")
    spec["stage_configs"] = json.loads(
        json.dumps(stage_configs).replace("{input}", str(stage_input))
    )
    worker = _worker(tmp_path, spec)
    manifest_path = tmp_path / "current.manifest.json"
    manifest_path.write_text('{"revision": 1}\n', encoding="utf-8")
    manifest = ArtifactRef.from_path(manifest_path, kind=ArtifactKind.MANIFEST)
    before = worker._stage_key(stage_name, manifest)  # noqa: SLF001

    stage_input.write_bytes(b"after")

    assert worker._stage_key(stage_name, manifest) != before  # noqa: SLF001


def test_build_key_does_not_walk_output_or_sdk_directories(tmp_path: Path) -> None:
    spec, _ = _make_spec(tmp_path)
    sdk_root = tmp_path / "qnn-sdk"
    sdk_root.mkdir()
    (sdk_root / "sdk.yaml").write_text("version: 2.48\n", encoding="utf-8")
    sdk_payload = sdk_root / "huge-library.bin"
    sdk_payload.write_bytes(b"sdk-before")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    generated = output_dir / "context.bin"
    generated.write_bytes(b"output-before")
    spec["stage_configs"]["build"] = {
        "sdk_root": str(sdk_root),
        "output_dir": str(output_dir),
    }
    worker = _worker(tmp_path, spec)
    before = worker._stage_key("build", None)  # noqa: SLF001

    sdk_payload.write_bytes(b"sdk-after")
    generated.write_bytes(b"output-after")

    assert worker._stage_key("build", None) == before  # noqa: SLF001


def test_relative_input_paths_are_resolved_from_project_not_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec, _ = _make_spec(tmp_path)
    path_fields = (
        (spec["sources"]["text"], "onnx_path"),
        (spec["sources"]["text"], "encodings_path"),
        (spec["sources"]["text"], "config_path"),
        (spec["vectors"], "validation_manifest"),
        (spec["metadata"]["attached_models_by_ar"]["128"], "model_path"),
        (
            spec["metadata"]["attached_models_by_ar"]["128"],
            "encodings_path",
        ),
    )
    for owner, key in path_fields:
        owner[key] = str(Path(owner[key]).relative_to(tmp_path))

    first_cwd = tmp_path / "cwd-a"
    second_cwd = tmp_path / "cwd-b"
    first_cwd.mkdir()
    second_cwd.mkdir()
    (first_cwd / "ordinary-label").write_text("one", encoding="utf-8")
    (second_cwd / "ordinary-label").write_text("two", encoding="utf-8")
    spec["metadata"]["label"] = "ordinary-label"

    monkeypatch.chdir(first_cwd)
    first = _worker(tmp_path, spec, "relative-a")._stage_key(  # noqa: SLF001
        "build",
        None,
    )
    monkeypatch.chdir(second_cwd)
    second = _worker(tmp_path, spec, "relative-b")._stage_key(  # noqa: SLF001
        "build",
        None,
    )

    assert second == first
