"""`docs/spec-reference.md` must stay in step with the contracts.

Spec knowledge was scattered across four documents, each describing a part and
none of them checked. A consolidated reference only helps if it cannot quietly
fall behind the models, so this asserts both directions: every contract field is
documented, and every documented field exists.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from qairt_agent import contracts
from qairt_agent.family_registry import FamilyRecord

REFERENCE = Path(__file__).parents[1] / "docs" / "spec-reference.md"

#: The models the reference covers, in the order it presents them.
DOCUMENTED_MODELS = (
    "WorkflowSpec",
    "ModelSourcesSpec",
    "ModelSourceSpec",
    "SequenceSpec",
    "SplitSpec",
    "TransformSpec",
    "QuantizationSpec",
    "VectorSpec",
    "CompileSpec",
    "TargetSpec",
    "QualitySpec",
    "BenchmarkSpec",
    "WorkflowStageConfigs",
    "DiagnoseStageConfig",
)


def _documented_names() -> set[str]:
    """Field names the reference mentions in a table cell or as `code`."""

    text = REFERENCE.read_text(encoding="utf-8")
    return set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text))


@pytest.mark.parametrize("model_name", DOCUMENTED_MODELS)
def test_every_contract_field_is_documented(model_name: str) -> None:
    model = getattr(contracts, model_name)
    documented = _documented_names()

    missing = sorted(
        name for name in model.model_fields if name not in documented
    )

    assert not missing, (
        f"{model_name} fields missing from docs/spec-reference.md: {missing}"
    )


def test_the_reference_documents_no_field_that_does_not_exist() -> None:
    """A documented name must resolve somewhere real.

    Only names that *look* like spec fields are checked -- the reference also
    quotes values, CLI flags and paths -- so the rule is: any backticked
    snake_case identifier must be a field on one of the documented models, an
    enum value, or an explicitly allowed prose term.
    """

    known: set[str] = set()
    for model_name in DOCUMENTED_MODELS:
        known.update(getattr(contracts, model_name).model_fields)
    # The reference also documents the add-a-family procedure, whose fields
    # live on the canonical family record rather than on a spec model.
    known.update(field.name for field in dataclasses.fields(FamilyRecord))

    # Stage-config keys, metadata keys, report fields and CLI/env names the
    # reference legitimately mentions but which are not contract fields.
    allowed = {
        "ar",
        "aa_calibration",
        "aimet_config_path",
        "apply_encodings",
        "architectures",
        "attached_models_by_ar",
        "calibrate",
        "chain",
        "component",
        "config",
        "context_length",
        "context_lengths",
        "effective_benchmark",
        "effective_compile",
        "effective_target",
        "external",
        "float_reference",
        "full_reference",
        "granularity",
        "gen_kv_format_config",
        "inside_vision_onnx",
        "layer",
        "lut",
        "model_config",
        "model_config_path",
        "ms_per_token_source",
        "op_level_dump_available",
        "p50_ms_per_token",
        "pipeline",
        "prompt",
        "prompt_path",
        "provided",
        "capture",
        "qairt_agent_adb_serial",
        "qwen35_runtime_validation",
        "qwen3_5",
        "qwen3_5_omni",
        "qwen3_5_omni_thinker",
        "qwen3_dense",
        "qwen3_moe",
        "qwen3_vl",
        "slice_boundary",
        "slice_vector_manifests",
        "soc_id",
        "teacher_forced",
        "tensor_map",
        "token_count",
        "validate",
        "vit",
        "adapt_moe",
        "m2s_head_split_map",
        "baseline_manifest",
        "kind",
        "quality",
        "latency",
        # Prose terms: an SDK function and a test name the procedure cites.
        "split_llm",
        "test_a_synthetic_family_becomes_visible_everywhere_from_one_record",
    }

    identifier = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
    text = REFERENCE.read_text(encoding="utf-8")
    unknown = sorted(
        name
        for name in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text)
        if identifier.match(name)
        and name not in known
        and name not in allowed
    )

    assert not unknown, (
        "docs/spec-reference.md names fields that are not on any documented "
        f"contract model: {unknown}"
    )
