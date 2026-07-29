from __future__ import annotations

import pytest
from pydantic import ValidationError

from qairt_agent.contracts import (
    ArtifactKind,
    ArtifactRef,
    BuildSpec,
    ComponentKind,
    ComponentSpec,
    DiagnoseKind,
    FamilyPreset,
    JobState,
    JobStatus,
    ModelFamily,
    ModelSourceSpec,
    ModelSourcesSpec,
    OutputLayoutSpec,
    PipelineKind,
    SkuOverlay,
    StageProvenance,
    StageReceipt,
    StageStatus,
    TensorRepresentation,
    VectorBundle,
    VectorMode,
    VectorSpec,
    VectorTensor,
    WorkflowSpec,
    preset_id_for_family,
    to_workflow_spec,
    utc_now,
)
from qairt_agent.errors import ErrorCode, ToolErrorData


def make_build_spec(**updates) -> BuildSpec:
    values = dict(
        name="qwen-smoke",
        family=ModelFamily.QWEN3_DENSE,
        sources=ModelSourcesSpec(
            text=ModelSourceSpec(
                onnx_path="/models/model.onnx",
                encodings_path="/models/model.encodings",
            )
        ),
        output_root="/artifacts/qwen-smoke",
        vectors=VectorSpec(mode=VectorMode.PROVIDED, validation_manifest="/vectors/golden.json"),
    )
    values.update(updates)
    return BuildSpec(**values)


def make_workflow_spec(**updates) -> WorkflowSpec:
    values = dict(
        name="qwen-smoke",
        preset="qwen3_dense",
        sources=ModelSourcesSpec(
            text=ModelSourceSpec(
                onnx_path="/models/model.onnx",
                encodings_path="/models/model.encodings",
            )
        ),
        output_root="/artifacts/qwen-smoke",
        vectors=VectorSpec(mode=VectorMode.PROVIDED, validation_manifest="/vectors/golden.json"),
    )
    values.update(updates)
    return WorkflowSpec(**values)


def test_preset_id_for_family_maps_all_families() -> None:
    assert preset_id_for_family(ModelFamily.QWEN3_DENSE) == "qwen3_dense"
    assert preset_id_for_family(ModelFamily.QWEN3_MOE) == "qwen3_moe"
    assert preset_id_for_family(ModelFamily.QWEN3_VL) == "qwen3_vl"
    assert preset_id_for_family(ModelFamily.QWEN3_5) == "qwen3_5"
    assert preset_id_for_family(ModelFamily.QWEN3_5_OMNI) == "qwen3_5_omni"
    assert preset_id_for_family(ModelFamily.VIT) == "vit"
    assert preset_id_for_family("qwen3.5") == "qwen3_5"


def test_to_workflow_spec_is_lossless_for_reused_subspecs() -> None:
    build_spec = make_build_spec(
        stage_configs={
            "validation": {"actual_manifest": "/vectors/actual.json"},
            "benchmark": {
                "context_path": "/contexts/model.bin",
                "graph_name": "decoder_ar1",
            },
            "diagnose": {
                "kind": "latency",
                "config": {
                    "baseline_ops": [{"op": "MatMul", "cycles": 10}],
                    "candidate_ops": [{"op": "MatMul", "cycles": 12}],
                },
            },
        }
    )
    workflow_spec = to_workflow_spec(build_spec)

    assert workflow_spec.preset == "qwen3_dense"
    assert workflow_spec.sku is None
    assert workflow_spec.sources == build_spec.sources
    assert workflow_spec.sequence == build_spec.sequence
    assert workflow_spec.split == build_spec.split
    assert workflow_spec.target == build_spec.target
    assert workflow_spec.output_root == build_spec.output_root
    assert workflow_spec.stage_configs == build_spec.stage_configs
    assert workflow_spec.stage_configs.diagnose.kind == DiagnoseKind.LATENCY


def test_workflow_spec_rejects_sku_preset_mismatch() -> None:
    sku = SkuOverlay(sku_id="sku-1", preset_id="qwen3_moe")
    with pytest.raises(ValidationError, match="preset_id must match"):
        make_workflow_spec(preset="qwen3_dense", sku=sku)


def test_workflow_stage_configs_accept_validate_stage_alias() -> None:
    spec = make_workflow_spec(
        stage_configs={"validate": {"actual_manifest": "/vectors/actual.json"}}
    )

    assert spec.stage_configs.validation == {
        "actual_manifest": "/vectors/actual.json"
    }


def test_family_preset_capability_gate_requires_runtime_unsupported() -> None:
    with pytest.raises(ValidationError, match="runtime_supported"):
        FamilyPreset(
            preset_id="bad",
            pipeline=PipelineKind.LOW_LEVEL,
            capability_gate="UNSUPPORTED_SDK_CAPABILITY",
            runtime_supported=True,
        )


def test_capability_gate_pipeline_requires_gate() -> None:
    with pytest.raises(ValidationError, match="capability_gate"):
        FamilyPreset(
            preset_id="bad",
            pipeline=PipelineKind.GENAI_CAPABILITY_GATE,
            runtime_supported=False,
        )


def test_component_spec_rejects_blank_name() -> None:
    with pytest.raises(ValidationError, match="blank"):
        ComponentSpec(kind=ComponentKind.DECODER, name="   ")


@pytest.mark.parametrize(
    "relative",
    ["/tmp/out", "../outside", "runs/{unknown}", "~/outside"],
)
def test_output_layout_rejects_paths_outside_output_root(relative: str) -> None:
    with pytest.raises(ValidationError):
        OutputLayoutSpec(directories={"contexts": relative})


def _provenance() -> StageProvenance:
    return StageProvenance(
        sdk_build="260626120635",
        adapter_capability="explicit_factory",
        platform_abi="ubuntu22.04-x86_64",
    )


def test_stage_receipt_invariants_and_verified() -> None:
    error = ToolErrorData(code=ErrorCode.STAGE_FAILED, message="boom", stage="convert")
    failed = StageReceipt(
        stage_key="k",
        stage_name="convert",
        status=StageStatus.FAILED,
        completed_at=utc_now(),
        provenance=_provenance(),
        error=error,
    )
    assert failed.verified is False

    succeeded = StageReceipt(
        stage_key="k",
        stage_name="convert",
        status=StageStatus.SUCCEEDED,
        started_at=utc_now(),
        completed_at=utc_now(),
        provenance=_provenance(),
    )
    assert succeeded.verified is True

    with pytest.raises(ValidationError, match="require an error"):
        StageReceipt(
            stage_key="k",
            stage_name="convert",
            status=StageStatus.FAILED,
            completed_at=utc_now(),
            provenance=_provenance(),
        )


def test_job_state_terminal_and_status_invariants() -> None:
    assert JobState.SUCCEEDED.terminal
    assert JobState.FAILED.terminal
    assert JobState.CANCELLED.terminal
    assert not JobState.RUNNING.terminal

    error = ToolErrorData(code=ErrorCode.INTERNAL_ERROR, message="boom")
    with pytest.raises(ValidationError, match="require an error"):
        JobStatus(
            job_id="j",
            state=JobState.FAILED,
            spec_sha256="a" * 64,
            created_at=utc_now(),
            updated_at=utc_now(),
        )

    status = JobStatus(
        job_id="j",
        state=JobState.FAILED,
        spec_sha256="a" * 64,
        created_at=utc_now(),
        updated_at=utc_now(),
        error=error,
    )
    assert status.error == error


def test_vector_bundle_partitions_by_representation() -> None:
    bundle = VectorBundle(
        tensors=(
            VectorTensor(name="logits", dtype="float32", shape=(1, 8)),
            VectorTensor(
                name="kv",
                dtype="uint8",
                shape=(1, 16),
                representation=TensorRepresentation.HMX_NATIVE,
            ),
        )
    )
    logical = bundle.by_representation(TensorRepresentation.LOGICAL_FP)
    native = bundle.by_representation(TensorRepresentation.HMX_NATIVE)
    assert [t.name for t in logical] == ["logits"]
    assert [t.name for t in native] == ["kv"]


def test_vector_tensor_rejects_bad_layout() -> None:
    with pytest.raises(ValidationError, match="layout"):
        VectorTensor(name="x", dtype="float32", layout="Z")


def test_artifact_ref_still_satisfies_receipt_io() -> None:
    ref = ArtifactRef(path="/a/b.bin", sha256="a" * 64, size_bytes=1, kind=ArtifactKind.DLC)
    receipt = StageReceipt(
        stage_key="k",
        stage_name="convert",
        status=StageStatus.SUCCEEDED,
        started_at=utc_now(),
        completed_at=utc_now(),
        outputs=(ref,),
        provenance=_provenance(),
    )
    assert receipt.outputs[0].sha256 == "a" * 64
