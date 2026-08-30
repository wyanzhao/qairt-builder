from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from qairt_agent.contracts import (
    BenchmarkSpec,
    BuildSpec,
    CompileSpec,
    EmbeddingMode,
    GraphVariantSpec,
    ModelFamily,
    ModelSourceSpec,
    ModelSourcesSpec,
    QuantizationMode,
    QuantizationSpec,
    SequenceSpec,
    SplitSpec,
    SqnrMode,
    StageRecord,
    StageStatus,
    TargetSpec,
    ToolResult,
    TransformSpec,
    VectorMode,
    VectorSpec,
)
from qairt_agent.errors import ErrorCode, ToolErrorData


def make_spec(**updates) -> BuildSpec:
    values = {
        "name": "qwen-smoke",
        "family": ModelFamily.QWEN3_DENSE,
        "sources": ModelSourcesSpec(
            text=ModelSourceSpec(
                onnx_path="/models/model.onnx",
                encodings_path="/models/model.encodings",
            ),
        ),
        "output_root": "/artifacts/qwen-smoke",
        "vectors": VectorSpec(
            mode=VectorMode.PROVIDED,
            validation_manifest="/vectors/golden/manifest.json",
        ),
    }
    values.update(updates)
    return BuildSpec(**values)


def test_build_spec_defaults_to_weight_shared_ar1_ar128() -> None:
    spec = make_spec()

    assert spec.ar_values == (1, 128)
    assert spec.sequence.weight_sharing is True
    assert spec.sequence.native_kv is True
    assert spec.split.total_splits == 3
    assert {mode.value for mode in type(spec.split.embedding_mode)} == {"lut", "compiled", "external"}
    assert spec.model_path.as_posix() == "/models/model.onnx"
    assert spec.benchmark == BenchmarkSpec(warmup_runs=10, measured_runs=50)


def test_vectors_can_bind_exact_golden_manifest_per_ar() -> None:
    spec = make_spec(
        vectors={
            "mode": "provided",
            "validation_manifests_by_ar": {
                "1": "/vectors/ar1.json",
                "128": "/vectors/ar128.json",
            },
        }
    )

    assert spec.vectors.validation_manifest is None
    assert spec.vectors.validation_manifests_by_ar == {
        1: Path("/vectors/ar1.json"),
        128: Path("/vectors/ar128.json"),
    }


def test_vectors_reject_manifest_for_unrequested_ar() -> None:
    with pytest.raises(ValidationError, match="not requested"):
        make_spec(
            vectors={
                "mode": "provided",
                "validation_manifests_by_ar": {
                    "2": "/vectors/ar2.json",
                },
            }
        )


def test_build_spec_accepts_flat_source_paths_and_canonical_qwen35_sequence() -> None:
    spec = BuildSpec(
        family="qwen3.5",
        model_path="/models/decode.onnx",
        encodings_path="/models/decode.encodings",
        output_root="/artifacts/qwen35",
        sequence={"ars": [1, 128], "qwen35_experimental_auto_ar": True},
    )

    assert spec.family == ModelFamily.QWEN3_5
    assert spec.sources.text.onnx_path.as_posix() == "/models/decode.onnx"
    serialized = spec.model_dump(mode="json")
    assert "sources" in serialized
    assert "source" not in serialized
    assert "sequence" in serialized
    assert "variants" not in serialized


def test_qwen3_4b_family_alias_routes_to_dense_low_level_family() -> None:
    spec = make_spec(family="qwen3-4b")

    assert spec.family is ModelFamily.QWEN3_DENSE


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"sequence": {"ars": [1, 1]}},
            "unique",
        ),
        (
            {
                "variants": [GraphVariantSpec(ar=1)],
                "compile": {"weight_sharing": True},
            },
            "exact AR set",
        ),
        (
            {
                "sources": ModelSourcesSpec(text=ModelSourceSpec(onnx_path="/models/model.onnx")),
                "quantization": QuantizationSpec(mode=QuantizationMode.APPLY_ENCODINGS),
            },
            "encodings",
        ),
        (
            {
                "sources": ModelSourcesSpec(
                    text=ModelSourceSpec(
                        onnx_path="/models/model.onnx",
                        encodings_path="/models/model.encodings",
                    ),
                ),
                "quality": {"sqnr_modes": [SqnrMode.CHAIN]},
                "vectors": {
                    "mode": "provided",
                    "calibration_manifest": "/vectors/calibration.json",
                },
            },
            "validation_manifest",
        ),
        (
            {
                "transforms": TransformSpec(mha2sha=False),
                "sequence": SequenceSpec(native_kv=True),
            },
            "MHA2SHA",
        ),
    ],
)
def test_build_spec_rejects_inconsistent_pipeline(updates, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_spec(**updates)


def test_graph_variant_rejects_ar_larger_than_context() -> None:
    with pytest.raises(ValidationError, match="less than or equal"):
        GraphVariantSpec(ar=128, context_length=64)


def test_qwen35_multi_ar_requires_explicit_experimental_opt_in() -> None:
    with pytest.raises(ValidationError, match="experimental"):
        make_spec(family=ModelFamily.QWEN3_5)


def test_qwen35_multi_ar_requires_weight_sharing() -> None:
    with pytest.raises(ValidationError, match="weight_sharing=true"):
        make_spec(
            family=ModelFamily.QWEN3_5,
            sequence={
                "ars": [1, 128],
                "weight_sharing": False,
                "qwen35_experimental_auto_ar": True,
            },
        )


def test_sequence_enforces_weight_sharing_and_native_kv_alignment() -> None:
    with pytest.raises(ValidationError, match="exact AR set"):
        SequenceSpec(ars=(1, 64), weight_sharing=True)
    with pytest.raises(ValidationError, match="divisible by 256"):
        SequenceSpec(
            ars=(1,),
            context_lengths=(1000,),
            weight_sharing=False,
            native_kv=True,
        )


def test_external_embedding_remains_a_semantic_split() -> None:
    split = SplitSpec(
        decoder_slice_count=2,
        embedding_mode=EmbeddingMode.EXTERNAL,
        split_lm_head=True,
    )

    assert split.split_embedding is True
    assert split.total_splits == 4


def test_nested_source_aliases_vectors_and_full_reference_are_canonical() -> None:
    spec = BuildSpec(
        family="qwen3",
        sources={
            "text": {
                "onnx": "/models/model.onnx",
                "encodings": "/models/model.encodings",
            }
        },
        output_root="/artifacts/qwen3",
        vectors={
            "mode": "provided",
            "validation_manifest": "/vectors/validation.json",
            "calibration_manifest": "/vectors/calibration.json",
        },
        quality={"sqnr_modes": ["full_reference"]},
    )
    serialized = spec.model_dump(mode="json")

    assert spec.quality.sqnr_modes == (SqnrMode.FULL_REFERENCE,)
    assert serialized["sources"]["text"]["onnx_path"] == "/models/model.onnx"
    assert "onnx" not in serialized["sources"]["text"]
    assert serialized["vectors"]["mode"] == "provided"
    assert serialized["target"]["name"] == "sm8850"
    assert serialized["target"]["soc_model"] == 87


@pytest.mark.parametrize(
    "flag_alias",
    [
        "experimental_auto_ar",
        "experimental_qwen35_auto_ar",
        "qwen35_experimental_ar_conversion",
        "allow_experimental_qwen35",
    ],
)
def test_qwen35_experimental_flag_aliases_serialize_canonically(flag_alias: str) -> None:
    spec = BuildSpec(
        family="qwen3.5",
        sources={
            "text": {
                "onnx": "/models/model.onnx",
                "encodings": "/models/model.encodings",
            }
        },
        output_root="/artifacts/qwen35",
        sequence={flag_alias: True},
    )
    serialized_sequence = spec.model_dump(mode="json")["sequence"]

    assert serialized_sequence["qwen35_experimental_auto_ar"] is True
    assert flag_alias not in serialized_sequence or flag_alias == "qwen35_experimental_auto_ar"


def test_legacy_vector_paths_normalize_to_canonical_vectors() -> None:
    validation_spec = BuildSpec(
        family="qwen3",
        source={
            "onnx": "/models/model.onnx",
            "encodings": "/models/model.encodings",
            "golden_vectors_path": "/vectors/validation.json",
        },
        output_root="/artifacts/validation",
    )
    calibration_spec = BuildSpec(
        family="qwen3",
        sources={"text": {"onnx": "/models/model.onnx"}},
        output_root="/artifacts/calibration",
        quantization={
            "mode": "calibrate",
            "calibration_vectors_path": "/vectors/calibration.json",
        },
    )

    assert validation_spec.vectors.validation_manifest.as_posix() == "/vectors/validation.json"
    assert calibration_spec.vectors.calibration_manifest.as_posix() == "/vectors/calibration.json"
    assert "calibration_vectors_path" not in calibration_spec.model_dump(mode="json")["quantization"]


def test_qwen3_vl_requires_integrated_vision_projector() -> None:
    common = {
        "family": "qwen3_vl",
        "sources": {
            "text": {
                "onnx": "/models/text.onnx",
                "encodings": "/models/text.encodings",
            },
            "vision": {
                "onnx": "/models/vision.onnx",
                "encodings": "/models/vision.encodings",
            },
        },
        "output_root": "/artifacts/qwen3-vl",
    }
    with pytest.raises(ValidationError, match="inside_vision_onnx"):
        BuildSpec(**common)

    spec = BuildSpec(
        **{
            **common,
            "sources": {
                **common["sources"],
                "vision_projector_location": "inside_vision_onnx",
            },
        }
    )
    assert spec.sources.vision_projector_location == "inside_vision_onnx"


def test_compile_output_selection_is_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        CompileSpec(enable_intermediate_outputs=True, output_tensors=("hidden",))


def test_stage_record_and_tool_result_are_structured_and_frozen() -> None:
    error = ToolErrorData(
        code=ErrorCode.STAGE_FAILED,
        message="converter failed",
        stage="convert",
        details={"op": "MatMul"},
    )
    completed = datetime.now(timezone.utc)
    failed_stage = StageRecord(
        name="convert",
        status=StageStatus.FAILED,
        completed_at=completed,
        error=error,
    )
    result = ToolResult[dict[str, str]].failure(error)

    assert failed_stage.error == error
    assert result.ok is False
    assert result.error == error
    assert result.model_dump(mode="json")["error"]["code"] == "stage_failed"
    with pytest.raises(ValidationError):
        failed_stage.status = StageStatus.SUCCEEDED


def test_tool_result_rejects_ambiguous_states() -> None:
    error = ToolErrorData(code=ErrorCode.INTERNAL_ERROR, message="boom")
    with pytest.raises(ValidationError):
        ToolResult(ok=True, error=error)
    with pytest.raises(ValidationError):
        ToolResult(ok=False)


def test_target_spec_selects_a_registered_target_by_name() -> None:
    default = TargetSpec()
    assert (default.name, default.chipset, default.dsp_arch, default.soc_model) == (
        "sm8850",
        "SM8850",
        "v81",
        87,
    )

    named = TargetSpec(name="sm8750")
    assert (named.chipset, named.dsp_arch, named.soc_model) == ("SM8750", "v79", 69)


def test_target_spec_accepts_only_a_tuple_that_matches_a_registered_target() -> None:
    assert TargetSpec(chipset="SM8850", dsp_arch="v81", soc_model=87).name == "sm8850"

    # 660 is SM8850's Android SoC ID, not its Qnn_SocModel_t value.
    with pytest.raises(ValidationError, match="not a reviewed target"):
        TargetSpec(chipset="SM8850", dsp_arch="v81", soc_model=660)

    with pytest.raises(ValidationError, match="partial tuple is never completed"):
        TargetSpec(chipset="SM8850")

    with pytest.raises(ValidationError, match="unregistered target"):
        TargetSpec(name="sm9999")

    with pytest.raises(ValidationError, match="does not match the supplied tuple"):
        TargetSpec(name="sm8750", chipset="SM8850", dsp_arch="v81", soc_model=87)
