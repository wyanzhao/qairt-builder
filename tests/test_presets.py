from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from qairt_agent.contracts import (
    ModelFamily,
    ModelSourceSpec,
    ModelSourcesSpec,
    QuantizationMode,
    QuantizationSpec,
    QualitySpec,
    SequenceSpec,
    SplitSpec,
    TransformSpec,
    VectorMode,
    VectorSpec,
)
from qairt_agent.contracts import BuildSpec, SkuOverlay, WorkflowSpec, to_workflow_spec
from qairt_agent.errors import (
    InvalidSpecError,
    PresetNotFoundError,
    UnsupportedSdkCapabilityError,
)
from qairt_agent.families.presets import (
    PRESET_REGISTRY,
    family_for_preset,
    get_preset,
    preset_sha256,
    resolve_workflow,
    to_build_spec,
)
from qairt_agent.families.sku import capture_sku, merge_sku
from qairt_agent.families.split_plan import build_split_plan


def make_workflow(preset: str = "qwen3_dense", **updates) -> WorkflowSpec:
    values = dict(
        name="model",
        preset=preset,
        sources=ModelSourcesSpec(
            text=ModelSourceSpec(
                onnx_path="/models/model.onnx",
                encodings_path="/models/model.encodings",
            )
        ),
        output_root="/artifacts/model",
        vectors=VectorSpec(mode=VectorMode.PROVIDED, validation_manifest="/vectors/golden.json"),
    )
    values.update(updates)
    return WorkflowSpec(**values)


def test_registry_contains_the_planned_presets() -> None:
    assert set(PRESET_REGISTRY) == {
        "qwen3_dense",
        "qwen3_moe",
        "qwen3_vl",
        "vit",
        "qwen3_5",
        "qwen3_5_omni_thinker",
        "qwen3_5_omni",
    }


def test_get_preset_unknown_raises() -> None:
    with pytest.raises(PresetNotFoundError, match="unknown preset"):
        get_preset("qwen3_omni")  # deliberately NOT an alias for qwen3_5_omni


def test_qwen35_omni_is_distinct_buildable_and_runtime_gated() -> None:
    omni = get_preset("qwen3_5_omni")
    assert omni.capability_gate is None
    assert omni.pipeline.value == "genai_builder"
    assert omni.runtime_supported is False

    attached = {
        "1": {"model_path": "/m/ar1.onnx", "encodings_path": "/m/ar1.enc"},
        "128": {"model_path": "/m/ar128.onnx", "encodings_path": "/m/ar128.enc"},
    }
    spec = make_workflow(
        preset="qwen3_5_omni",
        sources=ModelSourcesSpec(
            text=ModelSourceSpec(
                onnx_path="/m/text.onnx",
                encodings_path="/m/text.enc",
            ),
            audio=ModelSourceSpec(
                onnx_path="/m/audio.onnx",
                encodings_path="/m/audio.enc",
            ),
        ),
        metadata={"attached_models_by_ar": attached},
    )
    resolved = resolve_workflow(spec)
    assert resolved.pipeline.value == "genai_builder"
    assert resolved.runtime_supported is False
    assert to_build_spec(spec).family == ModelFamily.QWEN3_5_OMNI

    with pytest.raises(InvalidSpecError, match="sources.audio"):
        resolve_workflow(
            make_workflow(
                preset="qwen3_5_omni",
                metadata={"attached_models_by_ar": attached},
            )
        )


def test_qwen35_omni_thinker_uses_text_genai_lane_without_audio() -> None:
    attached = {
        "1": {"model_path": "/m/ar1.onnx", "encodings_path": "/m/ar1.enc"},
        "128": {"model_path": "/m/ar128.onnx", "encodings_path": "/m/ar128.enc"},
    }
    spec = make_workflow(
        "qwen3_5_omni_thinker",
        metadata={"attached_models_by_ar": attached},
    )

    resolved = resolve_workflow(spec)
    assert resolved.pipeline.value == "genai_builder"
    assert resolved.runtime_supported is True
    assert family_for_preset(spec.preset) is ModelFamily.QWEN3_5
    assert to_build_spec(spec).family is ModelFamily.QWEN3_5
    assert spec.sources.audio is None


def test_resolve_qwen3_dense_is_low_level() -> None:
    # Resolution is spec-authoritative: the preset supplies the pipeline
    # binding and policy gates, the spec supplies the effective ARs/slices.
    resolved = resolve_workflow(make_workflow("qwen3_dense", split=SplitSpec(decoder_slice_count=4)))
    assert resolved.pipeline.value == "low_level"
    assert resolved.ars == (1, 128)
    assert resolved.decoder_slices == 4
    assert resolved.weight_sharing is True
    assert resolved.native_kv is True
    assert resolved.runtime_supported is True
    assert len(resolved.resolved_preset_sha256) == 64
    # A bare spec keeps its own SplitSpec default (1), not the preset's 4.
    assert resolve_workflow(make_workflow("qwen3_dense")).decoder_slices == 1


def test_resolve_qwen35_requires_per_ar_onnx_and_genai_guards() -> None:
    # Missing attached models -> rejected.
    bare = make_workflow("qwen3_5")
    with pytest.raises(InvalidSpecError, match="attached_models_by_ar"):
        resolve_workflow(bare)

    attached = {
        "1": {"model_path": "/m/ar1.onnx", "encodings_path": "/m/ar1.enc"},
        "128": {"model_path": "/m/ar128.onnx", "encodings_path": "/m/ar128.enc"},
    }
    resolved = resolve_workflow(make_workflow("qwen3_5", metadata={"attached_models_by_ar": attached}))
    assert resolved.pipeline.value == "genai_builder"
    with pytest.raises(InvalidSpecError, match="weight sharing"):
        resolve_workflow(
            make_workflow(
                "qwen3_5",
                metadata={"attached_models_by_ar": attached},
                sequence=SequenceSpec(
                    ars=(1, 128),
                    weight_sharing=False,
                    native_kv=True,
                ),
            )
        )


def test_resolve_qwen35_rejects_calibration() -> None:
    spec = make_workflow(
        "qwen3_5",
        quantization=QuantizationSpec(mode=QuantizationMode.CALIBRATE),
        vectors=VectorSpec(
            mode=VectorMode.PROVIDED,
            validation_manifest="/v/g.json",
            calibration_manifest="/v/c.json",
        ),
        metadata={
            "attached_models_by_ar": {
                "1": {"model_path": "/m/ar1.onnx"},
                "128": {"model_path": "/m/ar128.onnx"},
            }
        },
    )
    with pytest.raises(InvalidSpecError, match="apply_encodings"):
        resolve_workflow(spec)


def test_resolve_vit_rejects_kv_and_weight_sharing() -> None:
    with pytest.raises(InvalidSpecError, match="native_kv"):
        resolve_workflow(make_workflow("vit"))

    spec = make_workflow(
        "vit",
        sequence=SequenceSpec(ars=(1,), weight_sharing=False, native_kv=False),
        split=SplitSpec(decoder_slice_count=1, split_lm_head=False),
        transforms=TransformSpec(mha2sha=False),
    )
    resolved = resolve_workflow(spec)
    assert resolved.pipeline.value == "low_level"
    assert resolved.native_kv is False
    assert resolved.weight_sharing is False


def test_preset_sha256_is_stable() -> None:
    preset = get_preset("qwen3_dense")
    assert preset_sha256(preset) == preset_sha256(preset)
    assert preset_sha256(preset) != preset_sha256(get_preset("qwen3_moe"))


def test_presets_publish_serializable_output_layouts_beneath_output_root() -> None:
    attached = {
        "1": {"model_path": "/m/ar1.onnx", "encodings_path": "/m/ar1.enc"},
        "128": {"model_path": "/m/ar128.onnx", "encodings_path": "/m/ar128.enc"},
    }
    resolved = resolve_workflow(
        make_workflow(
            "qwen3_5_omni_thinker",
            output_root="/artifacts/thinker",
            metadata={"attached_models_by_ar": attached},
        )
    )
    serialized = resolved.preset.model_dump(mode="json")
    layout = resolved.to_dict()["output_layout"]

    assert serialized["output_layout"]["directories"]["container"].endswith(
        "runs/{run_id}/genai/container"
    )
    assert layout["manifest_revisions"] == (
        "/artifacts/thinker/manifests/{run_id}"
    )
    assert layout["container"] == (
        "/artifacts/thinker/runs/{run_id}/genai/container"
    )
    assert set(layout) == set(resolved.preset.output_layout.directories)

    low_level = get_preset("qwen3_dense").output_layout.render(
        "/artifacts/qwen3", run_id="run-123"
    )
    assert low_level["transformed_slices"] == (
        "/artifacts/qwen3/runs/run-123/build/transformed"
    )
    assert low_level["contexts"] == (
        "/artifacts/qwen3/runs/run-123/build/contexts"
    )


@pytest.mark.parametrize(
    "filename",
    [
        "qwen3_dense.json",
        "qwen3_vl.json",
        "qwen3_5.json",
        "qwen3_5_omni.json",
        "qwen3_5_omni_thinker.json",
        "vit.json",
    ],
)
def test_canonical_model_examples_resolve_and_serialize(filename: str) -> None:
    payload = json.loads(
        (Path(__file__).parents[1] / "examples" / filename).read_text(
            encoding="utf-8"
        )
    )
    workflow = (
        WorkflowSpec.model_validate(payload)
        if "preset" in payload
        else to_workflow_spec(BuildSpec.model_validate(payload))
    )

    resolved = resolve_workflow(workflow)
    assert resolved.to_dict()["output_layout"]
    assert workflow.model_dump(mode="json")["output_root"].startswith("/artifacts/")


def test_to_build_spec_maps_families_and_standalone_vit() -> None:
    build = to_build_spec(make_workflow("qwen3_vl", sources=ModelSourcesSpec(
        text=ModelSourceSpec(onnx_path="/m/text.onnx", encodings_path="/m/text.enc"),
        vision=ModelSourceSpec(onnx_path="/m/vision.onnx", encodings_path="/m/vision.enc"),
        vision_projector_location="inside_vision_onnx",
    )))
    assert build.family == ModelFamily.QWEN3_VL

    assert family_for_preset("vit") is ModelFamily.VIT
    vit = to_build_spec(
        make_workflow(
            "vit",
            sequence=SequenceSpec(ars=(1,), weight_sharing=False, native_kv=False),
            split=SplitSpec(decoder_slice_count=1, split_lm_head=False),
            transforms=TransformSpec(mha2sha=False),
        )
    )
    assert vit.family is ModelFamily.VIT


def test_to_build_spec_materializes_diagnostic_compile_for_sqnr_failure_dump() -> None:
    build = to_build_spec(
        make_workflow(
            quality=QualitySpec(
                sqnr_modes=("teacher_forced", "chain"),
                dump_intermediates_on_failure=True,
            )
        )
    )

    assert build.quality.sqnr_modes == ("teacher_forced", "chain")
    assert build.compile.enable_intermediate_outputs is True


def test_merge_sku_only_revokes_runtime_support() -> None:
    preset = get_preset("qwen3_vl")
    sku = SkuOverlay(
        sku_id="sku",
        preset_id="qwen3_vl",
        ars=(1,),
        decoder_slices=2,
        runtime_supported=True,  # cannot grant what the preset lacks elsewhere
    )
    effective = merge_sku(preset, sku)
    assert effective["ars"] == (1,)
    assert effective["decoder_slices"] == 2
    # preset.runtime_supported is True for VL; overlay True keeps True
    assert effective["runtime_supported"] is True

    revoking = SkuOverlay(sku_id="sku2", preset_id="qwen3_vl", runtime_supported=False)
    assert merge_sku(preset, revoking)["runtime_supported"] is False


def test_merge_sku_rejects_preset_mismatch() -> None:
    preset = get_preset("qwen3_dense")
    sku = SkuOverlay(sku_id="sku", preset_id="qwen3_moe")
    with pytest.raises(ValueError, match="must match"):
        merge_sku(preset, sku)


def test_capture_sku_binds_sha_and_boundaries(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake-onnx-bytes")
    plan = build_split_plan(num_decoder_layers=8, decoder_slices=2)

    sku = capture_sku(
        preset_id="qwen3_dense",
        sku_id="local-qwen3-4b",
        model_path=model,
        architecture="Qwen3ForCausalLM",
        split_plan=plan,
        ars=(1, 128),
        decoder_slices=2,
    )

    assert sku.preset_id == "qwen3_dense"
    assert sku.model_sha256 is not None and len(sku.model_sha256) == 64
    assert sku.architecture == "Qwen3ForCausalLM"
    assert sku.tensor_abi["byte_order"] == "little"
    assert sku.tensor_abi["layout"] == "C"
    assert len(sku.slice_boundaries) == 2
    assert sku.captured_at is not None


def test_capture_sku_unknown_preset_raises() -> None:
    with pytest.raises(PresetNotFoundError):
        capture_sku(preset_id="nope", sku_id="s")


def test_workflow_spec_rejects_blank_preset() -> None:
    with pytest.raises(ValidationError, match="blank"):
        make_workflow(preset="   ")
