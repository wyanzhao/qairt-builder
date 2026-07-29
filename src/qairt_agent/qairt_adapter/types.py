"""Serializable result types for the QAIRT adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    message: str
    severity: IssueSeverity = IssueSeverity.ERROR


@dataclass(frozen=True)
class PreflightReport:
    issues: tuple[PreflightIssue, ...]
    sdk_root: Path | None
    sdk_version: str | None
    sdk_build_id: str | None
    target_soc: str | None
    dsp_arch: str | None
    soc_model: int | None

    @property
    def ok(self) -> bool:
        return not any(issue.severity is IssueSeverity.ERROR for issue in self.issues)

    @property
    def errors(self) -> tuple[PreflightIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is IssueSeverity.ERROR)

    @property
    def warnings(self) -> tuple[PreflightIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is IssueSeverity.WARNING)


@dataclass(frozen=True)
class ModelVariantArtifact:
    model_path: Path
    encodings_path: Path | None
    ar: int
    context_length: int
    source_kind: str
    family: str | None
    external_data_paths: tuple[Path, ...] = ()
    graph_context: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class TransformedSliceArtifact:
    slice_name: str
    split_index: int
    model_path: Path
    encodings_path: Path | None
    ar: int | None
    context_length: int | None
    external_data_paths: tuple[Path, ...] = ()
    graph_context: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ConvertedModelArtifact:
    model_path: Path | None
    source_model_path: Path
    quantization_mode: str
    slice_name: str | None
    ar: int | None
    context_length: int | None
    sdk_model: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class QuantizedModelArtifact:
    dlc_path: Path
    encodings_path: Path | None
    sdk_output: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CompiledContextArtifact:
    context_binary_path: Path
    slice_name: str | None
    graph_names: tuple[str, ...]
    ar_values: tuple[int, ...]
    target_soc: str
    dsp_arch: str
    soc_model: int
    weight_sharing: bool
    native_kv_config_path: Path | None
    context_length: int | None = None
    sdk_compiled_model: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class BuildResult:
    """Unified production build result.

    ``contexts`` contains exactly the production contexts grouped per
    ``(context_length, slice)``.  Standalone diagnostic contexts are kept in a
    separate collection so they cannot be mistaken for deployable output.
    """

    variants: tuple[ModelVariantArtifact, ...]
    transformed_slices: tuple[TransformedSliceArtifact, ...]
    converted_models: tuple[ConvertedModelArtifact, ...]
    contexts: tuple[CompiledContextArtifact, ...]
    diagnostic_contexts: tuple[CompiledContextArtifact, ...] = ()
    config_artifact_paths: tuple[Path, ...] = ()
    auxiliary_artifact_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class GenAIAttachedModel:
    """An independently exported model attached to one GenAI Builder AR."""

    model_path: Path
    encodings_path: Path | None = None


@dataclass(frozen=True)
class GenAIRawSliceArtifact:
    """One saved compiled split exposed by an LLMContainer."""

    slice_id: str
    context_binary_path: Path
    graph_names_by_ar: dict[int, str]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]


@dataclass(frozen=True)
class GenAIContainerBuildResult:
    """Result of a QAIRT GenAI Builder container/component packaging lane."""

    container_path: Path
    metadata_path: Path
    family: str
    builder_class: str
    container_class: str
    factory_support: str
    compatibility_mode: str
    compatibility_notes: tuple[str, ...]
    ar_values: tuple[int, ...]
    context_lengths: tuple[int, ...]
    num_splits: int
    split_embedding: bool
    split_lm_head: bool
    target_soc: str
    dsp_arch: str
    soc_model: int
    weight_sharing: bool
    native_kv: bool
    runtime_supported: bool
    attached_ar_values: tuple[int, ...] = ()
    vision_builder_class: str | None = None
    audio_builder_class: str | None = None
    audio_container_path: Path | None = None
    text_container_path: Path | None = None
    raw_slices: tuple[GenAIRawSliceArtifact, ...] = ()
    raw_tensor_runtime_supported: bool = False
    raw_tensor_runtime_notes: tuple[str, ...] = ()
    sdk_container: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class NativeKvGraphExpectation:
    graph_name: str
    ar: int
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]


@dataclass(frozen=True)
class NativeKvAuditReport:
    issues: tuple[str, ...]
    graph_names: tuple[str, ...]
    tensor_count: int

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class ProfileResult:
    execution_result: Any = field(repr=False)
    reports: tuple[Any, ...] = field(default=(), repr=False)
    graph_name: str = ""
    level: str = "detailed"
    option: str = "optrace"


@dataclass(frozen=True)
class Qwen35ValidationEvidence:
    """Strict gates for experimental single-source Qwen3.5 AR derivation."""

    ar_rewrite_passed: bool
    state_io_passed: bool
    mha2sha_passed: bool
    initializer_compatibility_passed: bool
    standalone_vs_joint_passed: bool
    notes: tuple[str, ...] = ()
    evidence_id: str = ""
    structural_digest: str = ""
    approved_context_keys: tuple[str, ...] = ()
    diagnostic_context_paths: tuple[Path, ...] = ()
    runtime_report_paths: tuple[Path, ...] = ()

    @property
    def ok(self) -> bool:
        return all(
            (
                self.ar_rewrite_passed,
                self.state_io_passed,
                self.mha2sha_passed,
                self.initializer_compatibility_passed,
                self.standalone_vs_joint_passed,
            )
        )

    @property
    def failed_gates(self) -> tuple[str, ...]:
        checks = {
            "ar_rewrite": self.ar_rewrite_passed,
            "state_io": self.state_io_passed,
            "mha2sha": self.mha2sha_passed,
            "initializer_compatibility": self.initializer_compatibility_passed,
            "standalone_vs_joint": self.standalone_vs_joint_passed,
        }
        return tuple(name for name, passed in checks.items() if not passed)


@dataclass(frozen=True)
class Qwen35RuntimeValidationRequest:
    """Inputs handed to an explicit device/vector validation callback."""

    slice_name: str
    ar_values: tuple[int, ...]
    graph_names: tuple[str, ...]
    standalone_contexts: tuple[CompiledContextArtifact, ...]
    joint_context: CompiledContextArtifact
    validation_payload: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class Qwen35RuntimeValidationResult:
    """Device-backed golden and standalone-vs-joint validation result."""

    standalone_vs_golden_passed: bool
    joint_vs_golden_passed: bool
    standalone_vs_joint_passed: bool
    executed_graph_names: tuple[str, ...] = ()
    golden_vector_ids: tuple[str, ...] = ()
    report_paths: tuple[Path, ...] = ()
    details: str = ""

    @property
    def ok(self) -> bool:
        return all(
            (
                self.standalone_vs_golden_passed,
                self.joint_vs_golden_passed,
                self.standalone_vs_joint_passed,
            )
        )


@dataclass(frozen=True)
class Qwen35DerivationValidation:
    evidence: Qwen35ValidationEvidence
    diagnostic_contexts: tuple[CompiledContextArtifact, ...]
    structural_report_path: Path
    runtime_results: tuple[Qwen35RuntimeValidationResult, ...]
