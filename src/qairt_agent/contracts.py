"""JSON-serializable contracts for the QAIRT agent.

These models deliberately contain no QAIRT imports.  A control-plane process
can therefore validate requests and inspect manifests without the SDK being
installed.
"""

from __future__ import annotations

import hashlib
import mimetypes
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PositiveInt,
    field_validator,
    model_validator,
)

from qairt_agent.errors import ErrorCode, ToolError, ToolErrorData
from qairt_agent.harness import (
    HarnessConstraintsError,
    load_harness_constraints,
    resolve_target,
    resolve_target_tuple,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class FrozenContract(BaseModel):
    """Base class for immutable, strict public contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
        validate_default=True,
    )


class ModelFamily(str, Enum):
    QWEN3_DENSE = "qwen3"
    QWEN3_MOE = "qwen3_moe"
    QWEN3_VL = "qwen3_vl"
    QWEN3_5 = "qwen3_5"
    QWEN3_5_OMNI = "qwen3_5_omni"
    VIT = "vit"

    @classmethod
    def _missing_(cls, value: object) -> "ModelFamily | None":
        if not isinstance(value, str):
            return None
        normalized = value.lower().replace("-", "_").replace(".", "_")
        aliases = {
            "qwen3_dense": cls.QWEN3_DENSE,
            "qwen3_4b": cls.QWEN3_DENSE,
            "qwen3moe": cls.QWEN3_MOE,
            "qwen3_vl": cls.QWEN3_VL,
            "qwen35": cls.QWEN3_5,
            "qwen3_5": cls.QWEN3_5,
            "qwen35_omni_thinker": cls.QWEN3_5,
            "qwen3_5_omni_thinker": cls.QWEN3_5,
            "qwen35_omni": cls.QWEN3_5_OMNI,
            "qwen3_5_omni": cls.QWEN3_5_OMNI,
            "vit": cls.VIT,
        }
        return aliases.get(normalized)


class ArtifactKind(str, Enum):
    ONNX = "onnx"
    AIMET_ENCODINGS = "aimet_encodings"
    TEST_VECTORS = "test_vectors"
    GOLDEN_VECTORS = "golden_vectors"
    DLC = "dlc"
    CONTEXT_BINARY = "context_binary"
    MANIFEST = "manifest"
    REPORT = "report"
    CONFIG = "config"
    LOG = "log"
    OTHER = "other"


class ArtifactRef(FrozenContract):
    """Content-addressed reference to a published file."""

    path: Path
    sha256: str
    size_bytes: int = Field(ge=0)
    kind: ArtifactKind = ArtifactKind.OTHER
    media_type: str = "application/octet-stream"
    logical_name: str | None = None

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if not _SHA256_RE.fullmatch(normalized):
            raise ValueError("sha256 must contain exactly 64 lowercase hexadecimal characters")
        return normalized

    @field_validator("logical_name")
    @classmethod
    def validate_logical_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("logical_name cannot be blank")
        return stripped

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        kind: ArtifactKind = ArtifactKind.OTHER,
        media_type: str | None = None,
        logical_name: str | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> "ArtifactRef":
        """Hash a file and return its immutable reference."""

        resolved = Path(path).expanduser().resolve()
        digest = hashlib.sha256()
        size_bytes = 0
        with resolved.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
                size_bytes += len(chunk)
        inferred_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        return cls(
            path=resolved,
            sha256=digest.hexdigest(),
            size_bytes=size_bytes,
            kind=kind,
            media_type=media_type or inferred_type,
            logical_name=logical_name,
        )


class ModelSourceSpec(FrozenContract):
    """Source model and optional sidecar artifacts."""

    onnx_path: Path
    encodings_path: Path | None = None
    tokenizer_path: Path | None = None
    config_path: Path | None = None
    aimet_config_path: Path | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_nested_path_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        aliases = {
            "onnx": "onnx_path",
            "encodings": "encodings_path",
            "tokenizer": "tokenizer_path",
            "config": "config_path",
            "aimet_config": "aimet_config_path",
        }
        for old_name, canonical_name in aliases.items():
            if old_name in data and canonical_name not in data:
                data[canonical_name] = data.pop(old_name)
        return data


class ModelSourcesSpec(FrozenContract):
    """Primary source plus optional vision/audio multimodal components."""

    text: ModelSourceSpec
    vision: ModelSourceSpec | None = None
    audio: ModelSourceSpec | None = None
    vision_projector_location: Literal["inside_vision_onnx"] | None = None


class GraphVariantSpec(FrozenContract):
    """Compatibility contract for callers migrating from paired AR/CL variants.

    Canonical build specifications use :class:`SequenceSpec`, which represents
    the full AR × context-length product derived from one base graph.
    """

    ar: PositiveInt
    context_length: PositiveInt = 4096
    model_path: Path | None = None
    encodings_path: Path | None = None
    name: str | None = None

    @model_validator(mode="after")
    def validate_ar_fits_context(self) -> "GraphVariantSpec":
        if self.ar > self.context_length:
            raise ValueError("ar must be less than or equal to context_length")
        return self


class EmbeddingMode(str, Enum):
    LUT = "lut"
    COMPILED = "compiled"
    EXTERNAL = "external"

    @classmethod
    def _missing_(cls, value: object) -> "EmbeddingMode | None":
        if value in {"compiled_split", "context_binary"}:
            return cls.COMPILED
        return None


class SplitSpec(FrozenContract):
    """Semantic model-splitting policy."""

    decoder_slice_count: PositiveInt = 1
    embedding_mode: EmbeddingMode = EmbeddingMode.LUT
    split_lm_head: bool = True

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_split_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "decoder_splits" in data and "decoder_slice_count" not in data:
            data["decoder_slice_count"] = data.pop("decoder_splits")
        if "split_embedding" in data:
            split_embedding = bool(data.pop("split_embedding"))
            if not split_embedding:
                data["embedding_mode"] = EmbeddingMode.EXTERNAL
        if data.get("embedding_mode") == "auto":
            data["embedding_mode"] = EmbeddingMode.LUT
        return data

    @property
    def split_embedding(self) -> bool:
        # Even an external embedding is extracted as a semantic split; the
        # mode controls packaging/execution, not graph-boundary discovery.
        return True

    @property
    def total_splits(self) -> int:
        return self.decoder_slice_count + 1 + int(self.split_lm_head)

    @property
    def decoder_splits(self) -> int:
        """Backward-compatible read alias."""

        return self.decoder_slice_count


class SequenceSpec(FrozenContract):
    """Graph-shape, weight-sharing, and native-KV policy."""

    ars: tuple[PositiveInt, ...] = (1, 128)
    context_lengths: tuple[PositiveInt, ...] = (4096,)
    weight_sharing: bool = True
    native_kv: bool = True
    qwen35_experimental_auto_ar: bool = False

    @model_validator(mode="before")
    @classmethod
    def accept_sequence_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        aliases = {
            "auto_regression_numbers": "ars",
            "auto_regression_number": "ars",
            "context_length": "context_lengths",
            "experimental_auto_ar": "qwen35_experimental_auto_ar",
            "experimental_qwen35_auto_ar": "qwen35_experimental_auto_ar",
            "qwen35_experimental": "qwen35_experimental_auto_ar",
            "qwen35_experimental_ar_conversion": "qwen35_experimental_auto_ar",
            "allow_experimental_qwen35": "qwen35_experimental_auto_ar",
            "allow_experimental_qwen35_auto_ar": "qwen35_experimental_auto_ar",
        }
        for old_name, canonical_name in aliases.items():
            if old_name in data and canonical_name not in data:
                data[canonical_name] = data.pop(old_name)
        for axis in ("ars", "context_lengths"):
            if axis in data and isinstance(data[axis], int):
                data[axis] = (data[axis],)
        return data

    @field_validator("ars", "context_lengths")
    @classmethod
    def values_must_be_nonempty_and_unique(
        cls, value: tuple[PositiveInt, ...]
    ) -> tuple[PositiveInt, ...]:
        if not value:
            raise ValueError("sequence axes cannot be empty")
        if len(value) != len(set(value)):
            raise ValueError("sequence axes must contain unique values")
        return value

    @model_validator(mode="after")
    def validate_ar_context_product(self) -> "SequenceSpec":
        if max(self.ars) > min(self.context_lengths):
            raise ValueError("every AR must fit every configured context length")
        if self.weight_sharing and set(self.ars) != {1, 128}:
            raise ValueError("BuildSpec weight sharing requires the exact AR set {1, 128}")
        if self.native_kv:
            non_aligned = tuple(length for length in self.context_lengths if length % 256)
            if non_aligned:
                raise ValueError(
                    f"native_kv requires context lengths divisible by 256; got {non_aligned}"
                )
        return self


class TransformSpec(FrozenContract):
    """AR conversion and graph transformation policy."""

    mha2sha: bool = True
    mha2sha_validate: bool = False
    permute_kv_cache_io: bool = False
    family_options: dict[str, JsonValue] = Field(default_factory=dict)


class QuantizationMode(str, Enum):
    APPLY_ENCODINGS = "apply_encodings"
    CALIBRATE = "calibrate"

    @classmethod
    def _missing_(cls, value: object) -> "QuantizationMode | None":
        if value == "precomputed_encodings":
            return cls.APPLY_ENCODINGS
        return None


class QuantizationSpec(FrozenContract):
    """Converter/quantizer policy."""

    mode: QuantizationMode = QuantizationMode.APPLY_ENCODINGS
    act_precision: Literal[8, 16] = 16
    bias_precision: Literal[8, 32] = 32
    weights_precision: Literal[4, 8, 16] = 8
    act_calibration_method: Literal["min-max", "sqnr", "entropy", "mse", "percentile"] = "min-max"
    param_calibration_method: Literal["min-max", "sqnr", "entropy", "mse", "percentile"] = "min-max"


class CompileSpec(FrozenContract):
    """Context-binary generation policy."""

    enable_intermediate_outputs: bool = False
    output_tensors: tuple[str, ...] = ()
    compiler_options: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_output_selection(self) -> "CompileSpec":
        if self.enable_intermediate_outputs and self.output_tensors:
            raise ValueError("enable_intermediate_outputs and output_tensors are mutually exclusive")
        return self


class TargetSpec(FrozenContract):
    """Pinned QAIRT worker and target-device contract."""

    backend: Literal["HTP"] = "HTP"
    name: str = Field(
        default_factory=lambda: load_harness_constraints().target_name
    )
    chipset: str = ""
    dsp_arch: str = ""
    soc_model: int = 0
    platform: Literal["android", "x86_64_linux"] = "android"
    device_id: str | None = None
    qairt_version: str = Field(
        default_factory=lambda: load_harness_constraints().qairt_version
    )
    qairt_build_id: str = Field(
        default_factory=lambda: load_harness_constraints().qairt_build_id
    )

    @model_validator(mode="after")
    def validate_harness_version(self) -> "TargetSpec":
        constraints = load_harness_constraints()
        if self.qairt_version != constraints.qairt_version:
            raise ValueError(
                f"qairt_version must match harness {constraints.qairt_version}"
            )
        if self.qairt_build_id != constraints.qairt_build_id:
            raise ValueError(
                f"qairt_build_id must match harness {constraints.qairt_build_id}"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def resolve_registered_target(cls, value: Any) -> Any:
        """Resolve a target by name, or accept a tuple that matches one.

        A target is legal because it is registered under ``harness/targets/``,
        not because it matches a constant. A caller may name one, or supply the
        full ``chipset``/``dsp_arch``/``soc_model`` tuple as long as it matches
        a registered entry exactly -- anything else fails here, at spec time,
        rather than at compile time.
        """

        if not isinstance(value, dict):
            return value
        data = dict(value)
        supplied = {
            key: data.get(key)
            for key in ("chipset", "dsp_arch", "soc_model")
            if data.get(key) not in (None, "", 0)
        }
        # Registry errors surface as ordinary validation errors so a bad target
        # reads like any other bad field rather than escaping the model.
        try:
            if len(supplied) == 3:
                entry = resolve_target_tuple(
                    str(supplied["chipset"]),
                    str(supplied["dsp_arch"]),
                    int(supplied["soc_model"]),
                )
            elif supplied:
                raise ValueError(
                    "target must name a registered target or supply the "
                    "complete chipset/dsp_arch/soc_model tuple; a partial "
                    "tuple is never completed implicitly "
                    f"(got {sorted(supplied)})"
                )
            else:
                entry = resolve_target(data.get("name"))
        except HarnessConstraintsError as error:
            raise ValueError(str(error)) from error
        if len(supplied) == 3:
            if data.get("name") and str(data["name"]).lower() != entry.name:
                raise ValueError(
                    f"target name {data['name']!r} does not match the supplied "
                    f"tuple, which is the registered target {entry.name!r}"
                )
        data["name"] = entry.name
        data["chipset"] = entry.chipset
        data["dsp_arch"] = entry.dsp_arch
        data["soc_model"] = entry.soc_model
        return data


class SqnrMode(str, Enum):
    FULL_REFERENCE = "full_reference"
    CHAIN = "chain"
    TEACHER_FORCED = "teacher_forced"


class VectorMode(str, Enum):
    PROVIDED = "provided"
    CAPTURE = "capture"


class VectorSpec(FrozenContract):
    """Validation/calibration vector-manifest inputs or capture policy."""

    mode: VectorMode = VectorMode.CAPTURE
    validation_manifest: Path | None = None
    validation_manifests_by_ar: dict[int, Path] = Field(default_factory=dict)
    calibration_manifest: Path | None = None

    @model_validator(mode="after")
    def validate_vector_mode(self) -> "VectorSpec":
        manifests_present = bool(
            self.validation_manifest
            or self.validation_manifests_by_ar
            or self.calibration_manifest
        )
        if self.mode == VectorMode.PROVIDED and not manifests_present:
            raise ValueError("provided vector mode requires at least one vector manifest")
        if self.mode == VectorMode.CAPTURE and manifests_present:
            raise ValueError("capture vector mode cannot include pre-existing manifests")
        invalid_ars = [
            ar
            for ar in self.validation_manifests_by_ar
            if isinstance(ar, bool) or ar <= 0
        ]
        if invalid_ars:
            raise ValueError(
                "validation_manifests_by_ar keys must be positive integer ARs"
            )
        return self


class QualitySpec(FrozenContract):
    """Report-only numerical validation policy."""

    sqnr_modes: tuple[SqnrMode, ...] = ()
    dump_intermediates_on_failure: bool = True

    @field_validator("sqnr_modes")
    @classmethod
    def unique_sqnr_modes(cls, value: tuple[SqnrMode, ...]) -> tuple[SqnrMode, ...]:
        if len(value) != len(set(value)):
            raise ValueError("sqnr_modes must be unique")
        return value


class BenchmarkSpec(FrozenContract):
    """Production latency measurement policy."""

    warmup_runs: int = Field(default=10, ge=0)
    measured_runs: PositiveInt = 50
    optrace: bool = False


class DiagnoseKind(str, Enum):
    """Diagnostic lane selected by the asynchronous workflow."""

    QUALITY = "quality"
    LATENCY = "latency"


class DiagnoseStageConfig(FrozenContract):
    """Inputs for the selected diagnostic continuation operation."""

    kind: DiagnoseKind = DiagnoseKind.QUALITY
    config: dict[str, JsonValue] = Field(default_factory=dict)


class WorkflowStageConfigs(FrozenContract):
    """Explicit continuation inputs for each asynchronous workflow stage.

    QAIRT continuation operations consume different artifact sets.  Keeping
    those mappings separate prevents validation, benchmarking, and diagnosis
    from accidentally receiving one shared catch-all configuration.
    """

    build: dict[str, JsonValue] = Field(default_factory=dict)
    validation: dict[str, JsonValue] = Field(default_factory=dict)
    benchmark: dict[str, JsonValue] = Field(default_factory=dict)
    diagnose: DiagnoseStageConfig = Field(default_factory=DiagnoseStageConfig)

    @model_validator(mode="before")
    @classmethod
    def accept_validate_stage_alias(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "validate" not in value:
            return value
        if "validation" in value:
            raise ValueError(
                "stage_configs cannot include both 'validate' and 'validation'"
            )
        data = dict(value)
        data["validation"] = data.pop("validate")
        return data


class BuildSpec(FrozenContract):
    """Complete input to a stateless build operation."""

    name: str = "model"
    family: ModelFamily
    sources: ModelSourcesSpec
    output_root: Path
    sequence: SequenceSpec = Field(default_factory=SequenceSpec)
    split: SplitSpec = Field(default_factory=SplitSpec)
    transforms: TransformSpec = Field(default_factory=TransformSpec)
    quantization: QuantizationSpec = Field(default_factory=QuantizationSpec)
    vectors: VectorSpec = Field(default_factory=VectorSpec)
    compile: CompileSpec = Field(default_factory=CompileSpec)
    target: TargetSpec = Field(default_factory=TargetSpec)
    quality: QualitySpec = Field(default_factory=QualitySpec)
    benchmark: BenchmarkSpec = Field(default_factory=BenchmarkSpec)
    stage_configs: WorkflowStageConfigs = Field(default_factory=WorkflowStageConfigs)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_fields(cls, value: Any) -> Any:
        """Normalize legacy inputs into the canonical representation."""

        if not isinstance(value, dict):
            return value
        data = dict(value)
        legacy_validation_manifest = data.pop("golden_vectors_path", None)

        if "sources" not in data:
            if "source" in data:
                data["sources"] = {"text": data.pop("source")}
            else:
                source: dict[str, Any] = {}
                aliases = {
                    "model_path": "onnx_path",
                    "onnx_path": "onnx_path",
                    "encodings_path": "encodings_path",
                    "tokenizer_path": "tokenizer_path",
                    "config_path": "config_path",
                    "aimet_config_path": "aimet_config_path",
                }
                for key, target in aliases.items():
                    if key in data:
                        source[target] = data.pop(key)
                if source:
                    data["sources"] = {"text": source}

        sources_value = data.get("sources")
        if isinstance(sources_value, ModelSourcesSpec):
            sources_data = sources_value.model_dump(mode="python", exclude_none=True)
        else:
            sources_data = dict(sources_value or {})
        text_value = sources_data.get("text")
        if isinstance(text_value, ModelSourceSpec):
            text_data = text_value.model_dump(mode="python", exclude_none=True)
        else:
            text_data = dict(text_value or {})
        nested_golden = text_data.pop("golden_vectors_path", None)
        legacy_validation_manifest = legacy_validation_manifest or nested_golden
        if text_data:
            sources_data["text"] = text_data
        if sources_data:
            data["sources"] = sources_data

        sequence_value = data.get("sequence")
        sequence = (
            sequence_value.model_dump(mode="python")
            if isinstance(sequence_value, SequenceSpec)
            else dict(sequence_value or {})
        )
        variants = data.pop("variants", None)
        if variants is not None:
            parsed_variants = [
                variant if isinstance(variant, GraphVariantSpec) else GraphVariantSpec.model_validate(variant)
                for variant in variants
            ]
            if any(
                variant.model_path is not None or variant.encodings_path is not None
                for variant in parsed_variants
            ):
                raise ValueError(
                    "per-variant model/encoding overrides are not part of BuildSpec; "
                    "use one base source and sequence axes"
                )
            sequence.setdefault("ars", tuple(dict.fromkeys(variant.ar for variant in parsed_variants)))
            sequence.setdefault(
                "context_lengths",
                tuple(dict.fromkeys(variant.context_length for variant in parsed_variants)),
            )

        compile_value = data.get("compile")
        if isinstance(compile_value, CompileSpec):
            compile_data = compile_value.model_dump(mode="python")
        else:
            compile_data = dict(compile_value or {})
        if "weight_sharing" in compile_data:
            sequence.setdefault("weight_sharing", compile_data.pop("weight_sharing"))
            data["compile"] = compile_data

        transforms_value = data.get("transforms")
        if isinstance(transforms_value, TransformSpec):
            transform_data = transforms_value.model_dump(mode="python")
        else:
            transform_data = dict(transforms_value or {})
        if "native_kv" in transform_data:
            sequence.setdefault("native_kv", transform_data.pop("native_kv"))
            data["transforms"] = transform_data

        vectors_value = data.get("vectors")
        vectors_data = (
            vectors_value.model_dump(mode="python", exclude_none=True)
            if isinstance(vectors_value, VectorSpec)
            else dict(vectors_value or {})
        )
        quantization_value = data.get("quantization")
        quantization_data = (
            quantization_value.model_dump(mode="python", exclude_none=True)
            if isinstance(quantization_value, QuantizationSpec)
            else dict(quantization_value or {})
        )
        legacy_calibration_manifest = quantization_data.pop("calibration_vectors_path", None)
        if quantization_value is not None:
            data["quantization"] = quantization_data
        if legacy_validation_manifest is not None:
            vectors_data.setdefault("validation_manifest", legacy_validation_manifest)
        if legacy_calibration_manifest is not None:
            vectors_data.setdefault("calibration_manifest", legacy_calibration_manifest)
        if (
            vectors_data.get("validation_manifest") is not None
            or vectors_data.get("validation_manifests_by_ar")
            or vectors_data.get("calibration_manifest") is not None
        ):
            vectors_data.setdefault("mode", VectorMode.PROVIDED)
        if vectors_data:
            data["vectors"] = vectors_data

        diagnostics = data.pop("diagnostics", None)
        if diagnostics is not None:
            diagnostics_data = (
                diagnostics.model_dump(mode="python")
                if isinstance(diagnostics, BaseModel)
                else dict(diagnostics)
            )
            quality_data = dict(data.get("quality") or {})
            benchmark_data = dict(data.get("benchmark") or {})
            for key in ("sqnr_modes", "dump_intermediates_on_failure"):
                if key in diagnostics_data:
                    quality_data.setdefault(key, diagnostics_data[key])
            for key in ("warmup_runs", "measured_runs", "optrace"):
                if key in diagnostics_data:
                    benchmark_data.setdefault(key, diagnostics_data[key])
            data["quality"] = quality_data
            data["benchmark"] = benchmark_data

        if sequence:
            data["sequence"] = sequence
        return data

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("name cannot be blank")
        return stripped

    @model_validator(mode="after")
    def validate_pipeline_contract(self) -> "BuildSpec":
        if self.quantization.mode == QuantizationMode.APPLY_ENCODINGS:
            if self.sources.text.encodings_path is None:
                raise ValueError("apply_encodings requires sources.text.encodings_path")
            if self.sources.vision is not None and self.sources.vision.encodings_path is None:
                raise ValueError(
                    "apply_encodings requires sources.vision.encodings_path when vision is present"
                )
            if self.sources.audio is not None and self.sources.audio.encodings_path is None:
                raise ValueError(
                    "apply_encodings requires sources.audio.encodings_path when audio is present"
                )
        elif self.quantization.mode == QuantizationMode.CALIBRATE:
            if (
                self.vectors.mode == VectorMode.PROVIDED
                and self.vectors.calibration_manifest is None
            ):
                raise ValueError("calibrate mode with provided vectors requires calibration_manifest")

        if (
            self.quality.sqnr_modes
            and self.vectors.mode == VectorMode.PROVIDED
            and self.vectors.validation_manifest is None
            and not self.vectors.validation_manifests_by_ar
        ):
            raise ValueError(
                "SQNR diagnostics with provided vectors require validation_manifest "
                "or validation_manifests_by_ar"
            )
        unexpected_vector_ars = set(
            self.vectors.validation_manifests_by_ar
        ).difference(self.sequence.ars)
        if unexpected_vector_ars:
            raise ValueError(
                "validation_manifests_by_ar contains ARs not requested by sequence.ars: "
                f"{sorted(unexpected_vector_ars)}"
            )

        if self.sequence.native_kv and not self.transforms.mha2sha:
            raise ValueError("native_kv requires the MHA2SHA transform")
        if self.transforms.mha2sha_validate and not self.transforms.mha2sha:
            raise ValueError("mha2sha_validate requires the MHA2SHA transform")

        if self.family == ModelFamily.QWEN3_VL:
            if self.sources.vision is None:
                raise ValueError("Qwen3-VL requires sources.vision")
            if self.sources.vision_projector_location != "inside_vision_onnx":
                raise ValueError(
                    "Qwen3-VL requires vision_projector_location='inside_vision_onnx'"
                )
        elif self.sources.vision_projector_location is not None:
            raise ValueError("vision_projector_location is only valid for Qwen3-VL")

        if self.family == ModelFamily.QWEN3_5_OMNI:
            if self.sources.audio is None:
                raise ValueError("Qwen3.5-Omni requires sources.audio")
            if self.sources.vision is not None:
                raise ValueError("Qwen3.5-Omni does not accept sources.vision")
        elif self.sources.audio is not None:
            raise ValueError("sources.audio is only valid for Qwen3.5-Omni")

        attached_models = self.metadata.get("attached_models_by_ar")
        attached_ars = (
            {str(key) for key in attached_models}
            if isinstance(attached_models, dict)
            else set()
        )
        if (
            self.family in {ModelFamily.QWEN3_5, ModelFamily.QWEN3_5_OMNI}
            and len(self.sequence.ars) > 1
            and not self.sequence.qwen35_experimental_auto_ar
            and not {str(ar) for ar in self.sequence.ars}.issubset(attached_ars)
        ):
            raise ValueError(
                "Qwen3.5 multi-AR requires attached_models_by_ar for every AR or "
                "sequence.qwen35_experimental_auto_ar=true"
            )
        if (
            self.family in {ModelFamily.QWEN3_5, ModelFamily.QWEN3_5_OMNI}
            and len(self.sequence.ars) > 1
            and not self.sequence.weight_sharing
        ):
            raise ValueError(
                "Qwen3.5 multi-AR requires sequence.weight_sharing=true"
            )

        if self.family == ModelFamily.VIT:
            if tuple(self.sequence.ars) != (1,):
                raise ValueError("standalone ViT requires the exact AR set (1,)")
            if self.sequence.weight_sharing or self.sequence.native_kv:
                raise ValueError("standalone ViT forbids weight sharing and native KV")
            if self.split.decoder_slice_count != 1 or self.split.split_lm_head:
                raise ValueError(
                    "standalone ViT is a single component; decoder_slice_count must be 1 "
                    "and split_lm_head must be false"
                )
            if self.transforms.mha2sha:
                raise ValueError("standalone ViT does not run the LLM MHA2SHA transform")
            if self.sources.vision is not None or self.sources.audio is not None:
                raise ValueError("standalone ViT uses sources.text as its only ONNX source")
        return self

    @property
    def model_path(self) -> Path:
        return self.sources.text.onnx_path

    @property
    def encodings_path(self) -> Path | None:
        return self.sources.text.encodings_path

    @property
    def golden_vectors_path(self) -> Path | None:
        return self.vectors.validation_manifest

    @property
    def ar_values(self) -> tuple[int, ...]:
        return tuple(self.sequence.ars)

    @property
    def source(self) -> ModelSourceSpec:
        """Backward-compatible alias for the canonical text source."""

        return self.sources.text


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageRecord(FrozenContract):
    """One immutable attempt at a pipeline stage."""

    name: str
    attempt: PositiveInt = 1
    status: StageStatus
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    inputs: tuple[ArtifactRef, ...] = ()
    outputs: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    error: ToolErrorData | None = None

    @model_validator(mode="before")
    @classmethod
    def default_start_to_supplied_completion(cls, value: Any) -> Any:
        # Avoid manufacturing a start timestamp a few microseconds after a
        # caller-supplied completion timestamp.
        if isinstance(value, dict) and value.get("completed_at") is not None and "started_at" not in value:
            return {**value, "started_at": value["completed_at"]}
        return value

    @field_validator("started_at", "completed_at")
    @classmethod
    def timestamps_must_be_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_state(self) -> "StageRecord":
        if self.status == StageStatus.FAILED and self.error is None:
            raise ValueError("failed stage records require an error")
        if self.status != StageStatus.FAILED and self.error is not None:
            raise ValueError("only failed stage records may contain an error")
        if self.status in {StageStatus.PENDING, StageStatus.RUNNING} and self.completed_at is not None:
            raise ValueError("pending or running stage records cannot have completed_at")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be earlier than started_at")
        return self


class RunManifest(FrozenContract):
    """Immutable snapshot of one run at a specific revision."""

    run_id: UUID = Field(default_factory=uuid4)
    revision: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    revision_created_at: datetime = Field(default_factory=utc_now)
    parent_manifest: ArtifactRef | None = None
    build_spec: BuildSpec
    stages: tuple[StageRecord, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("created_at", "revision_created_at")
    @classmethod
    def manifest_timestamps_must_be_aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("manifest timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_revision_chain(self) -> "RunManifest":
        if self.revision == 0 and self.parent_manifest is not None:
            raise ValueError("revision 0 cannot have a parent manifest")
        if self.revision > 0 and self.parent_manifest is None:
            raise ValueError("manifest revisions greater than 0 require a parent manifest")
        if self.parent_manifest is not None and self.parent_manifest.kind != ArtifactKind.MANIFEST:
            raise ValueError("parent_manifest must reference a manifest artifact")
        attempts = [(stage.name, stage.attempt) for stage in self.stages]
        if len(attempts) != len(set(attempts)):
            raise ValueError("stage (name, attempt) pairs must be unique")
        return self


ResultT = TypeVar("ResultT")


class ToolResult(FrozenContract, Generic[ResultT]):
    """Discriminated success/failure envelope for Python and MCP tools."""

    ok: bool
    data: ResultT | None = None
    error: ToolErrorData | None = None
    manifest: ArtifactRef | None = None
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_result(self) -> "ToolResult[ResultT]":
        if self.ok and self.error is not None:
            raise ValueError("successful ToolResult cannot contain an error")
        if not self.ok and self.error is None:
            raise ValueError("failed ToolResult requires an error")
        if not self.ok and self.data is not None:
            raise ValueError("failed ToolResult cannot contain data")
        return self

    @classmethod
    def success(
        cls,
        data: ResultT | None = None,
        *,
        manifest: ArtifactRef | None = None,
        warnings: tuple[str, ...] = (),
    ) -> "ToolResult[ResultT]":
        return cls(ok=True, data=data, manifest=manifest, warnings=warnings)

    @classmethod
    def failure(
        cls,
        error: ToolErrorData | ToolError | BaseException,
        *,
        manifest: ArtifactRef | None = None,
    ) -> "ToolResult[ResultT]":
        if isinstance(error, ToolErrorData):
            payload = error
        elif isinstance(error, ToolError):
            payload = error.data
        else:
            payload = ToolErrorData.from_exception(error, code=ErrorCode.INTERNAL_ERROR)
        return cls(ok=False, error=payload, manifest=manifest)


# --------------------------------------------------------------------------- #
# Preset / SKU contracts
# --------------------------------------------------------------------------- #


class PipelineKind(str, Enum):
    """Which production pipeline a preset binds to.

    ``LOW_LEVEL`` uses the QAIRT low-level Python APIs (AR convert, split,
    MHA2SHA, converter, quantizer, context compiler).  ``GENAI_BUILDER`` lets
    a public SDK family builder own transform/convert/quantize/compile and
    saves a container.  Qwen3.5 binds
    ``Qwen3_5BuilderHTP.from_pretrained`` directly; supported other families
    may use ``GenAIBuilderFactory``.  ``GENAI_CAPABILITY_GATE`` is a future
    preset that the current SDK cannot dispatch; it fails closed rather than
    aliasing to another family.
    """

    LOW_LEVEL = "low_level"
    GENAI_BUILDER = "genai_builder"
    GENAI_CAPABILITY_GATE = "genai_capability_gate"


class ComponentKind(str, Enum):
    EMBEDDING = "embedding"
    DECODER = "decoder"
    LM_HEAD = "lm_head"
    VIT = "vit"
    PROJECTOR = "projector"
    AUDIO = "audio"
    TEXT = "text"


class ComponentSpec(FrozenContract):
    """One node in a preset's component graph."""

    kind: ComponentKind
    name: str
    slicable: bool = False
    independent: bool = False
    weight_sharing_eligible: bool = False
    native_kv_eligible: bool = False
    ar_aware: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("component name cannot be blank")
        return stripped


class OutputLayoutSpec(FrozenContract):
    """Relative, serializable directory contract beneath ``output_root``.

    The layout describes durable artifact locations without creating them.
    ``{run_id}`` is the only supported placeholder; callers render it when a
    run is allocated.  Keeping every entry relative makes it impossible for a
    family preset to redirect artifacts outside the user-selected output root.
    """

    directories: dict[str, str]

    @model_validator(mode="after")
    def validate_directories(self) -> "OutputLayoutSpec":
        if not self.directories:
            raise ValueError("output layout must define at least one directory")
        for role, relative in self.directories.items():
            normalized_role = role.strip()
            if not normalized_role or normalized_role != role:
                raise ValueError("output layout roles must be non-blank and trimmed")
            if not re.fullmatch(r"[a-z][a-z0-9_]*", role):
                raise ValueError(
                    "output layout roles must use lowercase snake_case"
                )
            candidate = Path(relative)
            if (
                not relative
                or candidate.is_absolute()
                or ".." in candidate.parts
                or relative.startswith("~")
            ):
                raise ValueError(
                    f"output layout directory '{role}' must be a safe relative path"
                )
            placeholders = set(re.findall(r"\{([^{}]+)\}", relative))
            if placeholders - {"run_id"}:
                raise ValueError(
                    "output layout directories support only the {run_id} placeholder"
                )
        return self

    def render(
        self,
        output_root: str | Path,
        *,
        run_id: str = "{run_id}",
    ) -> dict[str, str]:
        """Render every directory beneath an absolute ``output_root``."""

        root = Path(output_root).expanduser().resolve()
        return {
            role: str(root / relative.replace("{run_id}", str(run_id)))
            for role, relative in self.directories.items()
        }


def _default_output_layout() -> OutputLayoutSpec:
    """Compatibility layout for third-party presets with no specialized lane."""

    return OutputLayoutSpec(
        directories={
            "manifest_revisions": "manifests/{run_id}",
            "run": "runs/{run_id}",
        }
    )


class FamilyPreset(FrozenContract):
    """A preset: a pipeline binding plus a component graph.

    Presets replace the flat ``family`` enum.  They declare the production
    pipeline, the component graph, and the default policy (ARs, decoder slices,
    embedding/lm_head independence, weight sharing, native KV).  A preset that
    the pinned SDK cannot dispatch carries a ``capability_gate`` and
    ``runtime_supported=False`` and must fail closed.
    """

    preset_id: str
    pipeline: PipelineKind
    components: tuple[ComponentSpec, ...] = ()
    default_ars: tuple[PositiveInt, ...] = (1, 128)
    default_decoder_slices: PositiveInt = 4
    default_weight_sharing: bool = True
    default_native_kv: bool = True
    embedding_independent: bool = True
    lm_head_independent: bool = True
    requires_per_ar_onnx: bool = False
    runtime_supported: bool = True
    capability_gate: str | None = None
    output_layout: OutputLayoutSpec = Field(default_factory=_default_output_layout)
    notes: tuple[str, ...] = ()

    @field_validator("preset_id")
    @classmethod
    def validate_preset_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("preset_id cannot be blank")
        return stripped

    @model_validator(mode="after")
    def validate_capability_gate(self) -> "FamilyPreset":
        if self.capability_gate is not None and self.runtime_supported:
            raise ValueError("a preset with a capability_gate must set runtime_supported=false")
        if self.pipeline == PipelineKind.GENAI_CAPABILITY_GATE and self.capability_gate is None:
            raise ValueError("genai_capability_gate presets require an explicit capability_gate")
        return self

    def component(self, kind: ComponentKind) -> ComponentSpec | None:
        for component in self.components:
            if component.kind == kind:
                return component
        return None


class SliceBoundary(FrozenContract):
    """An exact, captured decoder-slice layer boundary (half-open)."""

    slice_id: str
    layer_start: int = Field(ge=0)
    layer_end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "SliceBoundary":
        if self.layer_end < self.layer_start:
            raise ValueError("layer_end cannot be earlier than layer_start")
        return self


class SkuOverlay(FrozenContract):
    """A reproducible binding of a preset to one concrete model.

    Produced by ``preset capture``: it ties a reference overlay to the model
    SHA, architecture, tensor ABI, and exact slice boundaries.  Every field is
    optional so an overlay can override only what it pins; resolution merges it
    over the preset defaults.
    """

    sku_id: str
    preset_id: str
    model_sha256: str | None = None
    architecture: str | None = None
    tensor_abi: dict[str, JsonValue] = Field(default_factory=dict)
    ars: tuple[PositiveInt, ...] | None = None
    decoder_slices: PositiveInt | None = None
    embedding_mode: str | None = None
    lm_head_independent: bool | None = None
    weight_sharing: bool | None = None
    native_kv: bool | None = None
    slice_boundaries: tuple[SliceBoundary, ...] = ()
    runtime_supported: bool | None = None
    captured_at: datetime | None = None

    @field_validator("sku_id", "preset_id")
    @classmethod
    def validate_ids(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("sku_id and preset_id cannot be blank")
        return stripped


# --------------------------------------------------------------------------- #
# Workflow spec
# --------------------------------------------------------------------------- #


class WorkflowSpec(FrozenContract):
    """Complete input to an asynchronous workflow job.

    The preset selects the pipeline and component graph; the optional SKU
    overlay pins model-specific overrides.  The remaining sub-specs are reused
    unchanged, so the conversion from :class:`BuildSpec` is lossless.
    """

    name: str = "model"
    preset: str
    sku: SkuOverlay | None = None
    sources: ModelSourcesSpec
    output_root: Path
    sequence: SequenceSpec = Field(default_factory=SequenceSpec)
    split: SplitSpec = Field(default_factory=SplitSpec)
    transforms: TransformSpec = Field(default_factory=TransformSpec)
    quantization: QuantizationSpec = Field(default_factory=QuantizationSpec)
    vectors: VectorSpec = Field(default_factory=VectorSpec)
    compile: CompileSpec = Field(default_factory=CompileSpec)
    target: TargetSpec = Field(default_factory=TargetSpec)
    quality: QualitySpec = Field(default_factory=QualitySpec)
    benchmark: BenchmarkSpec = Field(default_factory=BenchmarkSpec)
    stage_configs: WorkflowStageConfigs = Field(default_factory=WorkflowStageConfigs)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("name", "preset")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("name and preset cannot be blank")
        return stripped

    @model_validator(mode="after")
    def validate_sku_matches_preset(self) -> "WorkflowSpec":
        if self.sku is not None and self.sku.preset_id != self.preset:
            raise ValueError("sku.preset_id must match the workflow preset")
        return self

    @property
    def model_path(self) -> Path:
        return self.sources.text.onnx_path


# --------------------------------------------------------------------------- #
# Vector bundle
# --------------------------------------------------------------------------- #


class TensorRepresentation(str, Enum):
    """How a vector's bytes are interpreted.

    SQNR compares only the decoded ``LOGICAL_FP`` values.  ``GRAPH_NATIVE`` and
    ``HMX_NATIVE`` describe native-KV physical byte/layout, which is verified
    for integrity separately and never fed through SQNR.
    """

    LOGICAL_FP = "logical_fp"
    GRAPH_NATIVE = "graph_native"
    HMX_NATIVE = "hmx_native"


class VectorTensor(FrozenContract):
    """One tensor in a bundle with full provenance and representation."""

    name: str
    path: Path | None = None
    representation: TensorRepresentation = TensorRepresentation.LOGICAL_FP
    dtype: str
    shape: tuple[int, ...] = ()
    layout: str = "C"
    byte_order: str = "little"
    sha256: str | None = None
    nbytes: int | None = Field(default=None, ge=0)
    valid_region: tuple[int, ...] | None = None
    component: str | None = None
    slice_id: str | None = None
    layer: int | None = None
    op: str | None = None
    ar: int | None = None
    context_length: int | None = None
    phase: str | None = None
    step: int | None = None
    graph_name: str | None = None
    role: str | None = None

    @field_validator("name", "dtype")
    @classmethod
    def validate_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("tensor name and dtype cannot be blank")
        return stripped

    @field_validator("layout")
    @classmethod
    def validate_layout(cls, value: str) -> str:
        if value not in {"C", "F"}:
            raise ValueError("layout must be 'C' (row-major) or 'F' (column-major)")
        return value


class VectorBundle(FrozenContract):
    """A content-addressed bundle of validation/calibration vectors.

    Records component/slice/layer/op, AR/CL/phase/step, dtype/shape/layout,
    valid region, source key/hash, graph binding, and representation for every
    tensor.
    """

    bundle_id: str = Field(default_factory=lambda: uuid4().hex)
    tensors: tuple[VectorTensor, ...] = ()
    source_key: str | None = None
    source_sha256: str | None = None
    graph_binding: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    def by_representation(self, representation: TensorRepresentation) -> tuple[VectorTensor, ...]:
        return tuple(tensor for tensor in self.tensors if tensor.representation == representation)


# --------------------------------------------------------------------------- #
# Job-journal contracts
# --------------------------------------------------------------------------- #


class JobState(str, Enum):
    """Lifecycle states for a persistent journal-backed job."""

    QUEUED = "queued"
    STAGING = "staging"
    RUNNING = "running"
    COLLECTING = "collecting"
    COMMITTING = "committing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ORPHANED = "orphaned"

    @property
    def terminal(self) -> bool:
        return self in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


class StageProvenance(FrozenContract):
    """Everything that participates in a stage key besides the inputs.

    A stage's outputs may be reused only when this provenance matches exactly;
    device stages additionally carry a ``device_fingerprint``.
    """

    sdk_build: str
    adapter_capability: str
    platform_abi: str
    resolved_preset_sha256: str | None = None
    image_digest: str | None = None
    host_arch: str | None = None
    emulation: bool = False
    device_fingerprint: str | None = None


class StageExecutionContext(FrozenContract):
    """Attempt identity supplied by the persistent workflow worker.

    This is deliberately separate from stage configuration so recovery paths
    and attempt numbers never participate in content/cache keys. ``output_dir``
    remains available to custom stage runners; the production QAIRT pipeline
    keeps artifacts in the durable ``BuildSpec.output_root`` run tree.
    """

    output_dir: Path
    attempt: PositiveInt


class StageReceipt(FrozenContract):
    """One immutable, verified receipt for a completed stage attempt.

    A worker verifies every input/output SHA before a receipt is published; the
    parent job only commits a manifest revision against verified receipts.
    """

    stage_key: str
    stage_name: str
    attempt: PositiveInt = 1
    status: StageStatus
    started_at: datetime
    completed_at: datetime | None = None
    inputs: tuple[ArtifactRef, ...] = ()
    outputs: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    provenance: StageProvenance
    error: ToolErrorData | None = None

    @model_validator(mode="before")
    @classmethod
    def default_start_to_supplied_completion(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("completed_at") is not None and "started_at" not in value:
            return {**value, "started_at": value["completed_at"]}
        return value

    @model_validator(mode="after")
    def validate_state(self) -> "StageReceipt":
        if self.status == StageStatus.FAILED and self.error is None:
            raise ValueError("failed stage receipts require an error")
        if self.status != StageStatus.FAILED and self.error is not None:
            raise ValueError("only failed stage receipts may contain an error")
        if self.status in {StageStatus.PENDING, StageStatus.RUNNING} and self.completed_at is not None:
            raise ValueError("pending or running stage receipts cannot have completed_at")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be earlier than started_at")
        return self

    @property
    def verified(self) -> bool:
        """A receipt is reusable only when it succeeded."""

        return self.status == StageStatus.SUCCEEDED


class JobStatus(FrozenContract):
    """An atomic snapshot of a job's journal state."""

    job_id: str
    state: JobState
    seq: int = Field(default=0, ge=0)
    parent_job_id: str | None = None
    spec_sha256: str
    created_at: datetime
    updated_at: datetime
    current_stage: str | None = None
    stages: tuple[StageReceipt, ...] = ()
    manifest: ArtifactRef | None = None
    heartbeat_at: datetime | None = None
    launcher: dict[str, JsonValue] = Field(default_factory=dict)
    error: ToolErrorData | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "JobStatus":
        if self.state == JobState.FAILED and self.error is None:
            raise ValueError("failed jobs require an error")
        if self.state != JobState.FAILED and self.error is not None:
            raise ValueError("only failed jobs may contain an error")
        return self


# --------------------------------------------------------------------------- #
# Spec conversion
# --------------------------------------------------------------------------- #

_FAMILY_TO_PRESET: dict[ModelFamily, str] = {
    ModelFamily.QWEN3_DENSE: "qwen3_dense",
    ModelFamily.QWEN3_MOE: "qwen3_moe",
    ModelFamily.QWEN3_VL: "qwen3_vl",
    ModelFamily.QWEN3_5: "qwen3_5",
    ModelFamily.QWEN3_5_OMNI: "qwen3_5_omni",
    ModelFamily.VIT: "vit",
}


def preset_id_for_family(family: ModelFamily | str) -> str:
    """Map a family (enum or string) to its preset id."""

    if isinstance(family, ModelFamily):
        return _FAMILY_TO_PRESET[family]
    return _FAMILY_TO_PRESET[ModelFamily(family)]


def to_workflow_spec(spec: BuildSpec) -> WorkflowSpec:
    """Convert a build spec into an equivalent workflow spec.

    The conversion is lossless for the reused sub-specs; the flat ``family``
    becomes a ``preset`` reference.  No SKU overlay is synthesized — capture is
    a separate, explicit operation.
    """

    return WorkflowSpec(
        name=spec.name,
        preset=preset_id_for_family(spec.family),
        sku=None,
        sources=spec.sources,
        output_root=spec.output_root,
        sequence=spec.sequence,
        split=spec.split,
        transforms=spec.transforms,
        quantization=spec.quantization,
        vectors=spec.vectors,
        compile=spec.compile,
        target=spec.target,
        quality=spec.quality,
        benchmark=spec.benchmark,
        stage_configs=spec.stage_configs,
        metadata=dict(spec.metadata),
    )


__all__ = [
    "ArtifactKind",
    "ArtifactRef",
    "BenchmarkSpec",
    "BuildSpec",
    "CompileSpec",
    "ComponentKind",
    "ComponentSpec",
    "DiagnoseKind",
    "DiagnoseStageConfig",
    "EmbeddingMode",
    "FamilyPreset",
    "GraphVariantSpec",
    "JobState",
    "JobStatus",
    "ModelFamily",
    "ModelSourceSpec",
    "ModelSourcesSpec",
    "OutputLayoutSpec",
    "PipelineKind",
    "QualitySpec",
    "QuantizationMode",
    "QuantizationSpec",
    "RunManifest",
    "SequenceSpec",
    "SkuOverlay",
    "SliceBoundary",
    "SplitSpec",
    "SqnrMode",
    "StageProvenance",
    "StageReceipt",
    "StageRecord",
    "StageStatus",
    "TargetSpec",
    "TensorRepresentation",
    "ToolErrorData",
    "ToolResult",
    "TransformSpec",
    "VectorBundle",
    "VectorMode",
    "VectorSpec",
    "VectorTensor",
    "WorkflowSpec",
    "WorkflowStageConfigs",
    "preset_id_for_family",
    "to_workflow_spec",
]
