"""SKU overlays and ``preset capture``.

A SKU overlay binds a reference preset to one concrete model: the model SHA,
architecture, tensor ABI, and exact decoder-slice boundaries.  ``capture`` is
the explicit operation that produces a reproducible overlay from a real
config/ONNX; resolution merges an overlay back over the preset defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qairt_agent.artifacts import sha256_file
from qairt_agent.contracts import utc_now
from qairt_agent.contracts import (
    FamilyPreset,
    SliceBoundary,
    SkuOverlay,
)
from qairt_agent.families.presets import get_preset
from qairt_agent.families.split_plan import SplitPlan


def merge_sku(preset: FamilyPreset, sku: SkuOverlay | None) -> dict[str, object]:
    """Return effective preset defaults with a SKU overlay applied.

    The overlay only overrides fields it pins (non-``None``); everything else
    falls back to the preset default.  Used at authoring time to preview the
    effective policy of a captured SKU.
    """

    if sku is not None and sku.preset_id != preset.preset_id:
        raise ValueError("sku.preset_id must match the preset being merged")

    effective: dict[str, object] = {
        "preset_id": preset.preset_id,
        "ars": tuple(preset.default_ars),
        "decoder_slices": preset.default_decoder_slices,
        "weight_sharing": preset.default_weight_sharing,
        "native_kv": preset.default_native_kv,
        "lm_head_independent": preset.lm_head_independent,
        "runtime_supported": preset.runtime_supported,
        "slice_boundaries": (),
    }
    if sku is None:
        return effective

    if sku.ars is not None:
        effective["ars"] = tuple(sku.ars)
    if sku.decoder_slices is not None:
        effective["decoder_slices"] = sku.decoder_slices
    if sku.weight_sharing is not None:
        effective["weight_sharing"] = sku.weight_sharing
    if sku.native_kv is not None:
        effective["native_kv"] = sku.native_kv
    if sku.lm_head_independent is not None:
        effective["lm_head_independent"] = sku.lm_head_independent
    if sku.runtime_supported is not None:
        # An overlay may only revoke runtime support, never grant it.
        effective["runtime_supported"] = preset.runtime_supported and sku.runtime_supported
    if sku.slice_boundaries:
        effective["slice_boundaries"] = tuple(sku.slice_boundaries)
    return effective


def _boundaries_from_split_plan(split_plan: SplitPlan) -> tuple[SliceBoundary, ...]:
    # Every slice that owns decoder layers is captured, including the lm_head
    # split when ``split_llm`` folds the final layer into it.
    return tuple(
        SliceBoundary(
            slice_id=slice_spec.name,
            layer_start=slice_spec.layer_start,
            layer_end=slice_spec.layer_end,
            advisory=True,
        )
        for slice_spec in split_plan.slices
        if slice_spec.layer_start is not None and slice_spec.layer_end is not None
    )


def capture_sku(
    *,
    preset_id: str,
    sku_id: str,
    model_path: str | Path | None = None,
    model_sha256: str | None = None,
    architecture: str | None = None,
    tensor_abi: dict[str, object] | None = None,
    split_plan: SplitPlan | None = None,
    ars: tuple[int, ...] | None = None,
    decoder_slices: int | None = None,
    embedding_mode: str | None = None,
    lm_head_independent: bool | None = None,
    weight_sharing: bool | None = None,
    native_kv: bool | None = None,
    runtime_supported: bool | None = None,
) -> SkuOverlay:
    """Bind a reference preset overlay to a concrete model.

    ``model_sha256`` is computed from ``model_path`` when not supplied.  The
    tensor ABI defaults to the framework's canonical little-endian, C-contiguous
    raw contract.  Slice boundaries are taken from ``split_plan`` when given.
    """

    preset = get_preset(preset_id)  # validates the preset exists

    resolved_sha = model_sha256
    if resolved_sha is None and model_path is not None:
        resolved_sha, _ = sha256_file(model_path)

    abi: dict[str, Any] = {"byte_order": "little", "layout": "C", "storage": "raw"}
    if tensor_abi:
        abi.update(tensor_abi)

    boundaries = _boundaries_from_split_plan(split_plan) if split_plan is not None else ()

    return SkuOverlay(
        sku_id=sku_id,
        preset_id=preset.preset_id,
        model_sha256=resolved_sha,
        architecture=architecture,
        tensor_abi=abi,
        ars=tuple(ars) if ars is not None else None,
        decoder_slices=decoder_slices,
        embedding_mode=embedding_mode,
        lm_head_independent=lm_head_independent,
        weight_sharing=weight_sharing,
        native_kv=native_kv,
        slice_boundaries=boundaries,
        runtime_supported=runtime_supported,
        captured_at=utc_now(),
    )


__all__ = ["capture_sku", "merge_sku"]
