"""Family presets and workflow resolution.

Presets replace the flat ``family`` enum.  Each preset binds a production
pipeline (low-level Python API, GenAI Builder, or a capability-gated future
builder) to a component graph and a default policy.  Resolution merges an
optional SKU overlay, validates the workflow against the preset, and fails
closed for presets the pinned SDK cannot dispatch.

This module is QAIRT-import-free: planning and resolution work without the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass

from qairt_agent.artifacts import canonical_json_bytes
from qairt_agent.contracts import (
    BenchmarkSpec,
    BuildSpec,
    ModelFamily,
    OutputLayoutSpec,
    QuantizationMode,
)
from qairt_agent.contracts import (
    ComponentKind,
    ComponentSpec,
    FamilyPreset,
    PipelineKind,
    SkuOverlay,
    WorkflowSpec,
    preset_id_for_family,
)
from qairt_agent.errors import (
    InvalidSpecError,
    PresetNotFoundError,
    UnsupportedSdkCapabilityError,
)

# --------------------------------------------------------------------------- #
# Preset registry
# --------------------------------------------------------------------------- #


def _component(kind: ComponentKind, name: str, **flags: bool) -> ComponentSpec:
    return ComponentSpec(kind=kind, name=name, **flags)


_DECODER = _component(
    ComponentKind.DECODER,
    "decoder",
    slicable=True,
    weight_sharing_eligible=True,
    native_kv_eligible=True,
    ar_aware=True,
)
_EMBEDDING = _component(ComponentKind.EMBEDDING, "embedding", independent=True)
_LM_HEAD = _component(ComponentKind.LM_HEAD, "lm_head", independent=True)

_COMMON_OUTPUT_DIRECTORIES = {
    "manifest_revisions": "manifests/{run_id}",
    "run": "runs/{run_id}",
    "effective_config": "runs/{run_id}/config",
    "source_records": "manifests/{run_id}",
    "vectors": "runs/{run_id}/vectors",
    "reports": "runs/{run_id}/diagnostics",
    "stage_attempts": "runs/{run_id}/stages",
}

LOW_LEVEL_OUTPUT_LAYOUT = OutputLayoutSpec(
    directories={
        **_COMMON_OUTPUT_DIRECTORIES,
        "ar_variants": "runs/{run_id}/build/variants",
        "transformed_slices": "runs/{run_id}/build/transformed",
        "converted_models": "runs/{run_id}/build/converted",
        "contexts": "runs/{run_id}/build/contexts",
        "diagnostic_contexts": "runs/{run_id}/build/diagnostic_contexts",
    }
)

GENAI_OUTPUT_LAYOUT = OutputLayoutSpec(
    directories={
        **_COMMON_OUTPUT_DIRECTORIES,
        "container": "runs/{run_id}/genai/container",
        "builder_cache": "runs/{run_id}/genai/cache",
    }
)


QWEN3_DENSE_PRESET = FamilyPreset(
    preset_id="qwen3_dense",
    pipeline=PipelineKind.LOW_LEVEL,
    components=(_EMBEDDING, _DECODER, _LM_HEAD),
    default_ars=(1, 128),
    default_decoder_slices=4,
    default_weight_sharing=True,
    default_native_kv=True,
    output_layout=LOW_LEVEL_OUTPUT_LAYOUT,
    notes=("Qwen3 Dense uses the SDK generic HTP builder; device golden validation required.",),
)

QWEN3_MOE_PRESET = FamilyPreset(
    preset_id="qwen3_moe",
    pipeline=PipelineKind.LOW_LEVEL,
    components=(_EMBEDDING, _DECODER, _LM_HEAD),
    default_ars=(1, 128),
    default_decoder_slices=4,
    default_weight_sharing=True,
    default_native_kv=True,
    output_layout=LOW_LEVEL_OUTPUT_LAYOUT,
    notes=("MoE adaptation is applied during the low-level transform stage.",),
)

QWEN3_VL_PRESET = FamilyPreset(
    preset_id="qwen3_vl",
    pipeline=PipelineKind.LOW_LEVEL,
    components=(
        _component(ComponentKind.VIT, "vit", independent=True),
        _component(ComponentKind.PROJECTOR, "projector"),
        _EMBEDDING,
        _DECODER,
        _LM_HEAD,
    ),
    default_ars=(1, 128),
    default_decoder_slices=4,
    default_weight_sharing=True,
    default_native_kv=True,
    output_layout=LOW_LEVEL_OUTPUT_LAYOUT,
    notes=(
        "Reusable ViT/projector components feed the text decoder chain.",
        "QAIRT 2.49 cannot execute an IMAGE_ENCODER workflow; runtime_supported reflects the SDK.",
    ),
)

VIT_PRESET = FamilyPreset(
    preset_id="vit",
    pipeline=PipelineKind.LOW_LEVEL,
    components=(_component(ComponentKind.VIT, "vit"),),
    default_ars=(1,),
    default_decoder_slices=1,
    default_weight_sharing=False,
    default_native_kv=False,
    embedding_independent=False,
    lm_head_independent=False,
    output_layout=LOW_LEVEL_OUTPUT_LAYOUT,
    notes=("Single-component context; no AR, KV cache, weight sharing, embedding, or lm_head.",),
)

QWEN3_5_PRESET = FamilyPreset(
    preset_id="qwen3_5",
    pipeline=PipelineKind.GENAI_BUILDER,
    components=(_EMBEDDING, _DECODER, _LM_HEAD),
    default_ars=(1, 128),
    default_decoder_slices=4,
    default_weight_sharing=True,
    default_native_kv=True,
    requires_per_ar_onnx=True,
    output_layout=GENAI_OUTPUT_LAYOUT,
    notes=(
        "GenAI Builder lane; every AR needs an independent ONNX+encodings pair.",
        "Must dispatch the pinned Qwen3_5BuilderHTP builder.",
    ),
)

QWEN3_5_OMNI_THINKER_PRESET = FamilyPreset(
    preset_id="qwen3_5_omni_thinker",
    pipeline=PipelineKind.GENAI_BUILDER,
    components=(_EMBEDDING, _DECODER, _LM_HEAD),
    default_ars=(1, 128),
    default_decoder_slices=4,
    default_weight_sharing=True,
    default_native_kv=True,
    requires_per_ar_onnx=True,
    output_layout=GENAI_OUTPUT_LAYOUT,
    notes=(
        "Standalone Qwen3.5-Omni Thinker text model; dispatches the pinned "
        "Qwen3_5BuilderHTP text lane and does not require an audio source.",
        "Every AR needs an independent ONNX+encodings pair.",
    ),
)

QWEN3_5_OMNI_PRESET = FamilyPreset(
    preset_id="qwen3_5_omni",
    pipeline=PipelineKind.GENAI_BUILDER,
    components=(
        _component(ComponentKind.AUDIO, "audio", independent=True),
        _component(ComponentKind.TEXT, "text", independent=True),
    ),
    default_ars=(1, 128),
    default_decoder_slices=4,
    default_weight_sharing=True,
    default_native_kv=True,
    requires_per_ar_onnx=True,
    runtime_supported=False,
    output_layout=GENAI_OUTPUT_LAYOUT,
    notes=(
        "Packages Qwen3OmniAudioEncoderBuilderHTP and a pinned Qwen3_5BuilderHTP "
        "as AUDIO_ENCODER -> TEXT_GENERATOR components.",
        "QAIRT 2.49 does not provide validated end-to-end audio workflow execution; "
        "runtime_supported remains false.",
    ),
)


PRESET_REGISTRY: dict[str, FamilyPreset] = {
    preset.preset_id: preset
    for preset in (
        QWEN3_DENSE_PRESET,
        QWEN3_MOE_PRESET,
        QWEN3_VL_PRESET,
        VIT_PRESET,
        QWEN3_5_PRESET,
        QWEN3_5_OMNI_THINKER_PRESET,
        QWEN3_5_OMNI_PRESET,
    )
}


def get_preset(preset_id: str) -> FamilyPreset:
    """Return the preset for ``preset_id`` or raise :class:`PresetNotFoundError`."""

    preset = PRESET_REGISTRY.get(preset_id)
    if preset is None:
        known = ", ".join(sorted(PRESET_REGISTRY))
        raise PresetNotFoundError(
            f"unknown preset '{preset_id}'; known presets: {known}",
            stage="preset",
            details={"preset_id": preset_id, "known": sorted(PRESET_REGISTRY)},
        )
    return preset


def preset_sha256(preset: FamilyPreset) -> str:
    """Stable content hash of a preset, used in stage-key provenance."""

    import hashlib

    return hashlib.sha256(canonical_json_bytes(preset)).hexdigest()


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResolvedWorkflow:
    """A workflow spec merged with its preset and SKU overlay, fully validated."""

    preset: FamilyPreset
    sku: SkuOverlay | None
    pipeline: PipelineKind
    ars: tuple[int, ...]
    decoder_slices: int
    weight_sharing: bool
    native_kv: bool
    embedding_mode: str
    lm_head_independent: bool
    runtime_supported: bool
    resolved_preset_sha256: str
    output_root: str

    def to_dict(self) -> dict[str, object]:
        return {
            "preset_id": self.preset.preset_id,
            "pipeline": self.pipeline.value,
            "ars": list(self.ars),
            "decoder_slices": self.decoder_slices,
            "weight_sharing": self.weight_sharing,
            "native_kv": self.native_kv,
            "embedding_mode": self.embedding_mode,
            "lm_head_independent": self.lm_head_independent,
            "runtime_supported": self.runtime_supported,
            "resolved_preset_sha256": self.resolved_preset_sha256,
            "sku_id": self.sku.sku_id if self.sku else None,
            "output_layout": self.preset.output_layout.render(self.output_root),
        }


def resolve_workflow(spec: WorkflowSpec) -> ResolvedWorkflow:
    """Resolve a workflow spec against its preset, failing closed where required.

    The spec is authoritative for ARs/slices/policy; the preset supplies the
    pipeline binding, component graph, provenance hash, and validation gates.
    """

    preset = get_preset(spec.preset)

    if preset.capability_gate is not None:
        raise UnsupportedSdkCapabilityError(
            f"preset '{preset.preset_id}' is not dispatchable on QAIRT 2.49 "
            f"({preset.capability_gate}); it must not alias to another family",
            stage="preset",
            details={
                "preset_id": preset.preset_id,
                "capability_gate": preset.capability_gate,
                "pipeline": preset.pipeline.value,
            },
        )

    ars = tuple(spec.sequence.ars)
    decoder_slices = spec.split.decoder_slice_count
    weight_sharing = spec.sequence.weight_sharing
    native_kv = spec.sequence.native_kv
    embedding_mode = spec.split.embedding_mode.value
    lm_head_independent = spec.split.split_lm_head

    # SKU overlay can only further constrain runtime support, never grant it.
    runtime_supported = preset.runtime_supported
    if spec.sku is not None and spec.sku.runtime_supported is False:
        runtime_supported = False

    if preset.pipeline == PipelineKind.GENAI_BUILDER:
        if spec.quantization.mode != QuantizationMode.APPLY_ENCODINGS:
            raise InvalidSpecError(
                "genai_builder presets require quantization.mode='apply_encodings'",
                stage="preset",
                details={"preset_id": preset.preset_id},
            )
        if not spec.transforms.mha2sha:
            raise InvalidSpecError(
                "genai_builder presets require the MHA2SHA transform",
                stage="preset",
                details={"preset_id": preset.preset_id},
            )
        if len(ars) > 1 and not weight_sharing:
            raise InvalidSpecError(
                "genai_builder presets require weight sharing when multiple ARs "
                "are requested",
                stage="preset",
                details={
                    "preset_id": preset.preset_id,
                    "ars": list(ars),
                },
            )
        if preset.requires_per_ar_onnx:
            attached = spec.metadata.get("attached_models_by_ar")
            provided = set()
            if isinstance(attached, dict):
                provided = {str(key) for key in attached}
            missing = [str(ar) for ar in ars if str(ar) not in provided]
            if missing:
                raise InvalidSpecError(
                    f"{preset.preset_id} requires an independent ONNX+encodings pair for every AR "
                    "via metadata.attached_models_by_ar",
                    stage="preset",
                    details={"preset_id": preset.preset_id, "missing_ars": missing},
                )
            missing_encodings = [
                str(ar)
                for ar in ars
                if not isinstance(attached, dict)
                or not isinstance(attached.get(str(ar), attached.get(ar)), dict)
                or attached.get(str(ar), attached.get(ar)).get("encodings_path") is None
            ]
            if missing_encodings:
                raise InvalidSpecError(
                    f"{preset.preset_id} requires explicit AIMET encodings for every attached AR",
                    stage="preset",
                    details={
                        "preset_id": preset.preset_id,
                        "missing_encodings_ars": missing_encodings,
                    },
                )

    if preset.preset_id == "qwen3_5_omni" and spec.sources.audio is None:
        raise InvalidSpecError(
            "qwen3_5_omni requires sources.audio with an ONNX model and AIMET encodings",
            stage="preset",
            details={"preset_id": preset.preset_id},
        )

    if preset.preset_id == "vit":
        if native_kv:
            raise InvalidSpecError(
                "vit preset has no KV cache; sequence.native_kv must be false",
                stage="preset",
            )
        if weight_sharing:
            raise InvalidSpecError(
                "vit preset has no weight sharing; sequence.weight_sharing must be false",
                stage="preset",
            )
        if ars != (1,):
            raise InvalidSpecError(
                "vit preset requires sequence.ars=[1]",
                stage="preset",
            )
        if decoder_slices != 1 or spec.split.split_lm_head:
            raise InvalidSpecError(
                "vit preset is a single component; use one decoder slice and split_lm_head=false",
                stage="preset",
            )
        if spec.transforms.mha2sha:
            raise InvalidSpecError(
                "vit preset does not use the LLM MHA2SHA transform",
                stage="preset",
            )

    return ResolvedWorkflow(
        preset=preset,
        sku=spec.sku,
        pipeline=preset.pipeline,
        ars=ars,
        decoder_slices=decoder_slices,
        weight_sharing=weight_sharing,
        native_kv=native_kv,
        embedding_mode=embedding_mode,
        lm_head_independent=lm_head_independent,
        runtime_supported=runtime_supported,
        resolved_preset_sha256=preset_sha256(preset),
        output_root=str(spec.output_root),
    )


# --------------------------------------------------------------------------- #
# family <-> preset bridge
# --------------------------------------------------------------------------- #

_PRESET_TO_FAMILY: dict[str, ModelFamily] = {
    "qwen3_dense": ModelFamily.QWEN3_DENSE,
    "qwen3_moe": ModelFamily.QWEN3_MOE,
    "qwen3_vl": ModelFamily.QWEN3_VL,
    "qwen3_5": ModelFamily.QWEN3_5,
    "qwen3_5_omni_thinker": ModelFamily.QWEN3_5,
    "qwen3_5_omni": ModelFamily.QWEN3_5_OMNI,
    "vit": ModelFamily.VIT,
}


def family_for_preset(preset_id: str) -> ModelFamily | None:
    """Return the family a preset maps to, or ``None`` if it has no build lane."""

    return _PRESET_TO_FAMILY.get(preset_id)


# One GenAI ``generate()`` call is a whole prompt-to-text workload, not a single
# graph invocation, so the low-level 10 warmup + 50 measured policy (doubled
# again by A/A calibration) would run 180 full generations per benchmark.
GENAI_BENCHMARK_DEFAULTS: dict[str, int] = {"warmup_runs": 3, "measured_runs": 10}


def is_genai_builder_family(family: ModelFamily | str) -> bool:
    """Whether a family's preset binds the GenAI Builder lane."""

    return get_preset(preset_id_for_family(family)).pipeline is PipelineKind.GENAI_BUILDER


def apply_lane_benchmark_defaults(
    family: ModelFamily | str,
    benchmark: BenchmarkSpec,
) -> BenchmarkSpec:
    """Fill unset benchmark fields with the resolved lane's defaults.

    Resolution must happen where the caller's input is still distinguishable
    from the schema default — ``model_fields_set`` reads as complete after a
    manifest round-trip — so both spec entry points call this and no later stage
    re-resolves.  A field the caller set always wins.
    """

    if not is_genai_builder_family(family):
        return benchmark
    unset = {
        field: value
        for field, value in GENAI_BENCHMARK_DEFAULTS.items()
        if field not in benchmark.model_fields_set
    }
    return benchmark.model_copy(update=unset) if unset else benchmark


def effective_benchmark_policy(spec: BuildSpec) -> dict[str, object]:
    """Render the benchmark sampling policy a run will actually execute."""

    genai = is_genai_builder_family(spec.family)
    return {
        **spec.benchmark.model_dump(mode="json"),
        "lane": "genai_builder" if genai else "low_level",
        "sample_unit": "generate_call" if genai else "graph_invocation",
        "aa_calibration_doubles_runs": True,
    }


def to_build_spec(spec: WorkflowSpec) -> BuildSpec:
    """Convert a workflow spec into the build spec the stage engine consumes.

    Every dispatchable preset has an explicit build-lane family mapping.
    """

    family = family_for_preset(spec.preset)
    if family is None:
        raise InvalidSpecError(
            f"preset '{spec.preset}' has no build lane",
            stage="preset",
            details={"preset_id": spec.preset},
        )
    compile_policy = spec.compile
    if (
        spec.quality.sqnr_modes
        and spec.quality.dump_intermediates_on_failure
        and not compile_policy.enable_intermediate_outputs
        and not compile_policy.output_tensors
    ):
        compile_policy = compile_policy.model_copy(
            update={"enable_intermediate_outputs": True}
        )
    return BuildSpec(
        name=spec.name,
        family=family,
        sources=spec.sources,
        output_root=spec.output_root,
        sequence=spec.sequence,
        split=spec.split,
        transforms=spec.transforms,
        quantization=spec.quantization,
        vectors=spec.vectors,
        compile=compile_policy,
        target=spec.target,
        quality=spec.quality,
        benchmark=apply_lane_benchmark_defaults(family, spec.benchmark),
        stage_configs=spec.stage_configs,
        metadata=dict(spec.metadata),
    )


__all__ = [
    "PRESET_REGISTRY",
    "GENAI_BENCHMARK_DEFAULTS",
    "GENAI_OUTPUT_LAYOUT",
    "LOW_LEVEL_OUTPUT_LAYOUT",
    "QWEN3_5_OMNI_PRESET",
    "QWEN3_5_OMNI_THINKER_PRESET",
    "QWEN3_5_PRESET",
    "QWEN3_DENSE_PRESET",
    "QWEN3_MOE_PRESET",
    "QWEN3_VL_PRESET",
    "ResolvedWorkflow",
    "VIT_PRESET",
    "apply_lane_benchmark_defaults",
    "effective_benchmark_policy",
    "family_for_preset",
    "get_preset",
    "is_genai_builder_family",
    "preset_id_for_family",
    "preset_sha256",
    "resolve_workflow",
    "to_build_spec",
]
