"""Serializable result types for the QAIRT adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, Protocol, runtime_checkable


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


#: Fields that hold a live QAIRT SDK object rather than a published path.
#:
#: They are excluded from serialization, and released once the artifact they
#: belong to has been published: a real multi-GB build accumulates one per
#: (context length, AR, slice), and holding them all for the whole build is an
#: OOM risk on top of from-zero crash restarts.
LIVE_SDK_FIELDS = frozenset(
    {
        "execution_result",
        "graph_context",
        "reports",
        "sdk_container",
        "sdk_compiled_model",
        "sdk_model",
        "sdk_output",
    }
)


def without_live_sdk_objects(artifact: Any) -> Any:
    """A copy of ``artifact`` with its live SDK references dropped.

    The live fields are declared ``compare=False``, so the released copy still
    compares equal to the original: releasing changes what is *reachable*, not
    what the artifact means.
    """

    if not is_dataclass(artifact) or isinstance(artifact, type):
        return artifact
    cleared = {
        field.name: None
        for field in fields(artifact)
        if field.name in LIVE_SDK_FIELDS
    }
    return replace(artifact, **cleared) if cleared else artifact


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
    """One graph the native-KV data-format config must cover.

    ``model_path`` is the transformed slice ONNX. QAIRT's own
    ``gen_kv_format_config`` reads the graph from disk, so the path is what
    lets this program call the SDK's selection rule instead of reimplementing
    it; ``input_names``/``output_names`` remain for auditing.
    """

    graph_name: str
    ar: int
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    model_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_name": self.graph_name,
            "ar": self.ar,
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "model_path": None if self.model_path is None else str(self.model_path),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NativeKvGraphExpectation":
        """Rebuild from JSON, ``model_path`` included.

        Dropping ``model_path`` here is not a cosmetic loss: QAIRT names each
        graph by its ONNX stem, and the path is the only thing that maps that
        stem back to this program's graph name. Without it the audit compares
        SDK stems against our names and can never pass.
        """

        raw_path = value.get("model_path")
        return cls(
            graph_name=str(value["graph_name"]),
            ar=int(value["ar"]),
            input_names=tuple(str(name) for name in value["input_names"]),
            output_names=tuple(str(name) for name in value["output_names"]),
            model_path=None if raw_path is None else Path(str(raw_path)),
        )


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


# --------------------------------------------------------------------------- #
# The pipeline <-> adapter boundary
# --------------------------------------------------------------------------- #


@runtime_checkable
class QairtAdapterProtocol(Protocol):
    """The adapter surface the pipeline actually consumes.

    The pipeline took its adapter through ``Callable[[], Any]``, so ~30 methods
    crossed that boundary with no type at all: drift between the real adapter,
    the 500-line test fake, and the call sites was caught only by eye. Declaring
    the surface once lets a type checker compare all three against the same
    thing.

    Signatures are deliberately permissive (``*args``/``**kwargs`` where the
    real method takes many keyword-only options): the value here is that the
    *method exists with the right name and arity class*, which is exactly the
    drift that used to go unnoticed. Tightening individual signatures is
    incremental work that can follow.

    Methods the pipeline probes with ``hasattr``/``getattr`` before calling --
    ``load_compiled``, ``profile``, ``capture_device_execution``,
    ``initialize_execution``, ``release_execution``, ``create_calibration_config``
    -- are declared in :class:`QairtAdapterOptionalProtocol` instead, because an
    adapter that lacks them still produces a valid (degraded, and labelled) run.
    """

    def preflight(self, *args: Any, **kwargs: Any) -> Any: ...

    def build(self, *args: Any, **kwargs: Any) -> BuildResult: ...

    def build_standalone_vit(self, *args: Any, **kwargs: Any) -> BuildResult: ...

    def build_genai_container(
        self, *args: Any, **kwargs: Any
    ) -> GenAIContainerBuildResult: ...

    def build_qwen35_omni_components(
        self, *args: Any, **kwargs: Any
    ) -> GenAIContainerBuildResult: ...

    def ar_convert(self, *args: Any, **kwargs: Any) -> ModelVariantArtifact: ...

    def transform(
        self, *args: Any, **kwargs: Any
    ) -> "tuple[TransformedSliceArtifact, ...]": ...

    def convert(self, *args: Any, **kwargs: Any) -> ConvertedModelArtifact: ...

    def quantize(self, *args: Any, **kwargs: Any) -> QuantizedModelArtifact: ...

    def compile_context(
        self, *args: Any, **kwargs: Any
    ) -> CompiledContextArtifact: ...

    def run_graph(self, *args: Any, **kwargs: Any) -> Any: ...

    def create_genai_executor(self, *args: Any, **kwargs: Any) -> Any: ...

    def clean_genai_executor(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class QairtAdapterOptionalProtocol(Protocol):
    """Adapter methods the pipeline probes for before using.

    Each of these has a documented degraded path: a missing
    ``capture_device_execution`` publishes ``device_execution.available =
    false`` with a reason rather than failing the benchmark, and a missing
    ``profile`` refuses an optrace request by name. They are typed here so a
    fake that *claims* to provide one is checked against the real signature.
    """

    def load_compiled(self, *args: Any, **kwargs: Any) -> Any: ...

    def profile(self, *args: Any, **kwargs: Any) -> ProfileResult: ...

    def capture_device_execution(
        self, *args: Any, **kwargs: Any
    ) -> "Mapping[str, Any]": ...

    def initialize_execution(self, *args: Any, **kwargs: Any) -> Any: ...

    def release_execution(self, *args: Any, **kwargs: Any) -> Any: ...

    def create_calibration_config(self, *args: Any, **kwargs: Any) -> Any: ...


#: What ``QairtAgent(adapter_factory=...)`` must return.
QairtAdapterFactory = Callable[[], QairtAdapterProtocol]
