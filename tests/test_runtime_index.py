from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qairt_agent.runtime.index import (
    RUNTIME_INDEX_SCHEMA,
    load_runtime_index,
    make_runtime_index,
    select_runtime_binding,
)


def _write(path: Path, payload: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def test_low_level_index_selects_exact_ar_chain_binding(tmp_path: Path) -> None:
    ar1 = _write(tmp_path / "variants" / "ar1.onnx")
    ar128 = _write(tmp_path / "variants" / "ar128.onnx")
    decoder = _write(tmp_path / "contexts" / "decoder.bin")
    head = _write(tmp_path / "contexts" / "lm_head.bin")
    vectors = _write(tmp_path / "vectors" / "ar128.json", "{}")
    route_path = tmp_path / "config" / "routes.json"
    route_payload = {
        "schema": "qairt-agent.slice-routes",
        "context_length": 4096,
        "contexts": {
            "decoder": str(decoder),
            "lm_head": str(head),
        },
        "routes": [
            {
                "slice_id": "decoder",
                "input_names": ["input_ids"],
                "output_names": ["hidden"],
                "graph_names": {"1": "decoder_ar1", "128": "decoder_ar128"},
            },
            {
                "slice_id": "lm_head",
                "input_names": ["hidden"],
                "output_names": ["logits"],
                "graph_names": {"1": "head_ar1", "128": "head_ar128"},
                "from_previous": {"hidden": "hidden"},
            },
        ],
    }
    _write(route_path, json.dumps(route_payload))
    route_ref = SimpleNamespace(
        path=route_path,
        sha256=hashlib.sha256(route_path.read_bytes()).hexdigest(),
    )
    result = SimpleNamespace(
        variants=(
            SimpleNamespace(
                context_length=4096,
                ar=1,
                model_path=ar1,
                encodings_path=None,
                source_kind="derived",
            ),
            SimpleNamespace(
                context_length=4096,
                ar=128,
                model_path=ar128,
                encodings_path=None,
                source_kind="derived",
            ),
        ),
        transformed_slices=(),
        contexts=(
            SimpleNamespace(
                context_length=4096,
                slice_name="decoder",
                context_binary_path=decoder,
                ar_values=(1, 128),
                graph_names=("decoder_ar1", "decoder_ar128"),
                weight_sharing=True,
                native_kv_config_path=None,
            ),
            SimpleNamespace(
                context_length=4096,
                slice_name="lm_head",
                context_binary_path=head,
                ar_values=(1, 128),
                graph_names=("head_ar1", "head_ar128"),
                weight_sharing=True,
                native_kv_config_path=None,
            ),
        ),
        diagnostic_contexts=(),
    )

    index = make_runtime_index(
        result=result,
        lane="low_level",
        family="qwen3",
        default_ar=1,
        default_context_length=4096,
        route_artifacts=(route_ref,),
        validation_manifests_by_ar={128: vectors},
    )
    binding = select_runtime_binding(index, ar=128)

    assert index["schema"] == RUNTIME_INDEX_SCHEMA
    assert binding["scope"] == "chain"
    assert binding["reference_model_path"] == str(ar128.resolve())
    assert binding["vector_manifest"] == str(vectors.resolve())
    assert binding["contexts"]["decoder"] == str(decoder)
    assert binding["routes"][1]["graph_names"]["128"] == "head_ar128"


def test_genai_index_stays_explicit_about_runtime_support(tmp_path: Path) -> None:
    container = tmp_path / "container"
    container.mkdir()
    result = SimpleNamespace()
    index = make_runtime_index(
        result=result,
        lane="genai_builder",
        family="qwen3_5_omni",
        default_ar=1,
        default_context_length=4096,
        runtime_supported=False,
        container_path=container,
    )
    path = _write(tmp_path / "runtime_index.json", json.dumps(index))

    loaded = load_runtime_index(path)
    binding = select_runtime_binding(loaded)
    assert binding["lane"] == "genai_builder"
    assert binding["runtime_supported"] is False
    assert binding["container_path"] == str(container.resolve())


def test_qwen3_vl_index_never_selects_text_route_as_multimodal_e2e(
    tmp_path: Path,
) -> None:
    text_context = _write(tmp_path / "contexts" / "decoder.bin")
    vision_context = _write(tmp_path / "contexts" / "vision_projector.bin")
    vision_onnx = _write(tmp_path / "models" / "vision_projector.onnx")
    vectors = _write(tmp_path / "vectors" / "ar1.json", "{}")
    route_path = tmp_path / "config" / "routes.json"
    route_payload = {
        "schema": "qairt-agent.slice-routes",
        "context_length": 4096,
        "component": "text",
        "coverage": "text_only",
        "excluded_components": ["vision_projector"],
        "contexts": {"decoder": str(text_context)},
        "routes": [
            {
                "slice_id": "decoder",
                "input_names": ["input_ids"],
                "output_names": ["logits"],
                "graph_names": {"1": "decoder_ar1"},
            }
        ],
    }
    _write(route_path, json.dumps(route_payload))
    route_ref = SimpleNamespace(
        path=route_path,
        sha256=hashlib.sha256(route_path.read_bytes()).hexdigest(),
    )
    result = SimpleNamespace(
        variants=(),
        transformed_slices=(
            SimpleNamespace(
                slice_name="vision_projector",
                split_index=0,
                model_path=vision_onnx,
                encodings_path=None,
                ar=None,
                context_length=None,
            ),
        ),
        contexts=(
            SimpleNamespace(
                context_length=4096,
                slice_name="decoder",
                context_binary_path=text_context,
                ar_values=(1,),
                graph_names=("decoder_ar1",),
                weight_sharing=False,
                native_kv_config_path=None,
            ),
            SimpleNamespace(
                context_length=None,
                slice_name="vision_projector",
                context_binary_path=vision_context,
                ar_values=(1,),
                graph_names=("vision",),
                weight_sharing=False,
                native_kv_config_path=None,
            ),
        ),
        diagnostic_contexts=(),
    )
    index = make_runtime_index(
        result=result,
        lane="low_level",
        family="qwen3_vl",
        default_ar=1,
        default_context_length=4096,
        route_artifacts=(route_ref,),
        validation_manifests_by_ar={1: vectors},
    )

    assert index["execution_contract"] == {
        "kind": "multimodal_components",
        "automatic_end_to_end_supported": False,
        "required_components": ["vision", "text"],
        "boundary_binding": "not_executable",
        "reason": (
            "the build contains separate vision/projector and text contexts, "
            "but no audited tensor bridge or ImageT2T executor binding"
        ),
    }
    assert (
        index["contexts"]["4096"]["vision_projector"]["component"]
        == "vision"
    )
    assert (
        index["contexts"]["4096"]["vision_projector"][
            "context_length_scope"
        ]
        == "independent"
    )
    with pytest.raises(ValueError, match="automatic end-to-end"):
        select_runtime_binding(index)
    unsafe_legacy = dict(index)
    unsafe_legacy.pop("execution_contract")
    with pytest.raises(ValueError, match="fail-closed"):
        select_runtime_binding(unsafe_legacy, component="text")
    unsafe_path = _write(
        tmp_path / "unsafe_runtime_index.json",
        json.dumps(unsafe_legacy),
    )
    with pytest.raises(ValueError, match="fail-closed"):
        load_runtime_index(unsafe_path)

    text = select_runtime_binding(index, component="text")
    assert text["component"] == "text"
    assert text["coverage"] == "text_only"
    assert text["excluded_components"] == ["vision_projector"]
    assert text["context_path"] == str(text_context.resolve())
    assert text["context_path"] != str(vision_context.resolve())

    vision = select_runtime_binding(index, component="vision")
    assert vision["component"] == "vision"
    assert vision["coverage"] == "vision_only"
    assert vision["context_path"] == str(vision_context.resolve())
    assert vision["graph_name"] == "vision"
    assert vision["graph_ar"] == 1
    assert vision["reference_model_path"] == str(vision_onnx.resolve())
    assert vision["vector_manifest"] is None
