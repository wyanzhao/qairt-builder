from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from qairt_agent.artifacts import ManifestStore, verify_artifact
from qairt_agent.contracts import (
    ArtifactRef,
    BenchmarkSpec,
    BuildSpec,
    QuantizationMode,
    RunManifest,
    StageExecutionContext,
    StageStatus,
)
from qairt_agent.errors import ErrorCode
from qairt_agent.pipeline import QairtAgent, _sdk_generated_token_count
from qairt_agent.qairt_adapter import (
    BuildResult,
    CompiledContextArtifact,
    ConvertedModelArtifact,
    GenAIContainerBuildResult,
    GenAIRawSliceArtifact,
    ModelVariantArtifact,
    PreflightReport,
    TransformedSliceArtifact,
    Qwen35ValidationEvidence,
)
from qairt_agent.diagnostics.device_metrics import parse_device_execution
from qairt_agent.qairt_adapter.errors import (
    QairtConfigurationError,
    QairtPreflightError,
)
from qairt_agent.vectors import VectorPreparer


def _write_onnx(
    path: Path,
    *,
    input_name: str = "x",
    output_name: str = "y",
    shape: tuple[int, ...] = (1,),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        [helper.make_node("Identity", [input_name], [output_name], name="identity")],
        "pipeline-test",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, list(shape))],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, list(shape))],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    model.ir_version = 9
    onnx.save_model(model, path)
    return path


def _write(path: Path, payload: bytes = b"artifact") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _make_spec(
    tmp_path: Path,
    *,
    quantization_mode: str = "apply_encodings",
    vectors: Mapping[str, Any] | None = None,
    transforms: Mapping[str, Any] | None = None,
    sequence: Mapping[str, Any] | None = None,
    quality: Mapping[str, Any] | None = None,
) -> BuildSpec:
    model = _write_onnx(tmp_path / "source" / "model.onnx")
    encodings = _write(
        tmp_path / "source" / "model.encodings",
        b'{"activation_encodings": {}, "param_encodings": {}}',
    )
    return BuildSpec(
        name="pipeline-test",
        family="qwen3",
        sources={
            "text": {
                "onnx_path": model,
                "encodings_path": encodings,
            }
        },
        output_root=tmp_path / "artifacts",
        quantization={"mode": quantization_mode},
        vectors=dict(vectors or {"mode": "capture"}),
        transforms=dict(transforms or {}),
        sequence=dict(sequence or {}),
        quality=dict(quality or {}),
        benchmark={"warmup_runs": 1, "measured_runs": 2},
        metadata={
            "model_config": {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 4096,
                "vocab_size": 32,
            }
        },
    )


def _load_run(ref: ArtifactRef) -> RunManifest:
    return ManifestStore(ref.path.parent.parent).load(ref)


def _vector_case(
    tmp_path: Path,
    *,
    inputs: Mapping[str, Any] | None = None,
    goldens: Mapping[str, Any] | None = None,
    case_id: str = "case",
) -> Path:
    return VectorPreparer(tmp_path / "vectors").prepare_case(
        case_id,
        dict(inputs or {"x": np.array([2.0], dtype=np.float32)}),
        goldens=dict(goldens or {"y": np.array([4.0], dtype=np.float32)}),
    )


def _make_vl_spec(tmp_path: Path) -> BuildSpec:
    text = _write_onnx(
        tmp_path / "vl" / "text.onnx",
        shape=(3,),
    )
    text_encodings = _write(tmp_path / "vl" / "text.encodings", b"{}")
    vision = _write_onnx(
        tmp_path / "vl" / "vision_projector.onnx",
        input_name="pixel_values",
        output_name="visual_embeddings",
        shape=(1, 16),
    )
    vision_encodings = _write(
        tmp_path / "vl" / "vision_projector.encodings",
        b"{}",
    )
    vectors = _vector_case(
        tmp_path / "vl",
        inputs={
            "x": np.array([2.0, 3.0, 4.0], dtype=np.float32),
            "pixel_values": np.ones((1, 16), dtype=np.float32),
        },
        goldens={
            "y": np.array([2.0, 3.0, 4.0], dtype=np.float32),
        },
        case_id="qwen3-vl-source",
    )
    return BuildSpec(
        name="qwen3-vl-runtime-scope",
        family="qwen3_vl",
        sources={
            "text": {
                "onnx_path": text,
                "encodings_path": text_encodings,
            },
            "vision": {
                "onnx_path": vision,
                "encodings_path": vision_encodings,
            },
            "vision_projector_location": "inside_vision_onnx",
        },
        output_root=tmp_path / "artifacts",
        sequence={
            "ars": [1],
            "context_lengths": [4096],
            "weight_sharing": False,
            "native_kv": True,
        },
        vectors={
            "mode": "provided",
            "validation_manifest": vectors,
        },
        benchmark={"warmup_runs": 0, "measured_runs": 1},
        metadata={
            "model_config": {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
                "text_config": {
                    "hidden_size": 16,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "max_position_embeddings": 4096,
                    "vocab_size": 32,
                },
            }
        },
    )


# Shape and values captured from QAIRT 2.49 on SM8750 running the tiny
# acceptance graph; see tests/test_device_metrics.py for the full report.
SAMPLE_PROFILING_REPORT: dict[str, Any] = {
    "metadata": {"appName": "qnn-net-run", "appVersion": "v2.49.0.260730134355"},
    "messages": [
        {
            "method": "BACKEND_CREATE_FROM_BINARY",
            "profilingEvents": [
                {
                    "identifier": "QNN (load binary) time",
                    "unit": "MICROSEC",
                    "type": "INIT",
                    "value": 9009,
                }
            ],
        },
        {
            "method": "BACKEND_EXECUTE",
            "profilingEvents": [
                {
                    "identifier": "Accelerator (execute excluding wait) time",
                    "unit": "MICROSEC",
                    "type": "",
                    "value": 77,
                },
                {
                    "identifier": "Accelerator (execute) time",
                    "unit": "MICROSEC",
                    "type": "",
                    "value": 1734,
                },
                {
                    "identifier": "QNN (execute) time",
                    "unit": "MICROSEC",
                    "type": "EXECUTE",
                    "value": 3001,
                },
                {
                    "identifier": "Accelerator (execute) time (cycles)",
                    "unit": "CYCLES",
                    "type": "",
                    "value": 29239,
                    "sub-events": [
                        {
                            "identifier": "fc:OpId_17",
                            "unit": "CYCLES",
                            "type": "NODE",
                            "value": 6137,
                        },
                        {
                            "identifier": "act:OpId_23",
                            "unit": "CYCLES",
                            "type": "NODE",
                            "value": 8567,
                        },
                    ],
                },
            ],
        },
        {
            "method": "BACKEND_DEINIT",
            "profilingEvents": [
                {
                    "identifier": "QNN (deinit) time",
                    "unit": "MICROSEC",
                    "type": "DEINIT",
                    "value": 19359,
                }
            ],
        },
    ],
}


@dataclass
class AdapterLog:
    calls: list[tuple[int, str, dict[str, Any]]] = field(default_factory=list)
    instances: int = 0

    def record(self, adapter_id: int, name: str, **details: Any) -> None:
        self.calls.append((adapter_id, name, details))

    def names(self) -> list[str]:
        return [name for _, name, _ in self.calls]


@dataclass(frozen=True)
class ExecutionResultLike:
    """Matches QAIRT 2.49's public ExecutionResult.data surface."""

    data: Mapping[str, Any]


class _PoisonLiveObject:
    def __str__(self) -> str:
        raise AssertionError("live SDK objects must not be serialized")


class FakeAdapter:
    def __init__(
        self,
        log: AdapterLog,
        *,
        result_style: str = "mapping",
        profile_cycles: float = 100.0,
    ) -> None:
        log.instances += 1
        self.adapter_id = log.instances
        self.log = log
        self.result_style = result_style
        self.profile_cycles = float(profile_cycles)
        self.ready = False
        self.initialized_contexts: set[str] = set()

    def preflight(self, spec: BuildSpec) -> PreflightReport:
        self.ready = True
        self.log.record(self.adapter_id, "preflight", family=spec.family.value)
        return PreflightReport(
            issues=(),
            sdk_root=Path("/opt/qairt/2.49.0.260730"),
            sdk_version="2.49.0",
            sdk_build_id="260730134355",
            target_soc="SM8750",
            dsp_arch="v79",
            soc_model=69,
        )

    def create_calibration_config(self, **kwargs: Any) -> dict[str, Any]:
        if not self.ready:
            raise QairtPreflightError("preflight must precede calibration config")
        self.log.record(self.adapter_id, "create_calibration_config", **kwargs)
        return {"kind": "fake-calibration", **kwargs}

    def build(
        self,
        spec: BuildSpec,
        runtime: Mapping[str, Any],
        output_dir: Path,
        **kwargs: Any,
    ) -> BuildResult:
        if not self.ready:
            raise QairtPreflightError("preflight must precede build")
        self.log.record(
            self.adapter_id,
            "build",
            runtime=runtime,
            kwargs=kwargs,
        )
        variant_model = _write_onnx(output_dir / "variants" / "ar1.onnx")
        variant_encodings = _write(
            output_dir / "variants" / "ar1.encodings",
            b'{"variant": 1}',
        )
        variant_data = _write(output_dir / "variants" / "ar1.data", b"external-weights")
        slice_model = _write_onnx(output_dir / "slices" / "decoder_00.onnx")
        slice_encodings = _write(
            output_dir / "slices" / "decoder_00.encodings",
            b'{"slice": 0}',
        )
        slice_data = _write(output_dir / "slices" / "decoder_00.data", b"slice-weights")
        dlc = _write(output_dir / "converted" / "decoder_00.dlc")
        context = _write(output_dir / "contexts" / "decoder_00.bin")
        native_kv = _write(
            output_dir / "contexts" / "decoder_00.native_kv.json",
            b'{"graphs": []}',
        )
        generated_config = _write(
            output_dir / "config" / "adapter_config.json",
            b'{"adapter": "fake"}',
        )
        auxiliary = _write(output_dir / "logs" / "build.log", b"fake build")
        live = _PoisonLiveObject()
        return BuildResult(
            variants=(
                ModelVariantArtifact(
                    model_path=variant_model,
                    encodings_path=variant_encodings,
                    ar=1,
                    context_length=4096,
                    source_kind="derived",
                    family="qwen3",
                    external_data_paths=(variant_data,),
                    graph_context=live,
                ),
            ),
            transformed_slices=(
                TransformedSliceArtifact(
                    slice_name="decoder_00",
                    split_index=1,
                    model_path=slice_model,
                    encodings_path=slice_encodings,
                    ar=1,
                    context_length=4096,
                    external_data_paths=(slice_data,),
                    graph_context=live,
                ),
            ),
            converted_models=(
                ConvertedModelArtifact(
                    model_path=dlc,
                    source_model_path=slice_model,
                    quantization_mode=spec.quantization.mode.value,
                    slice_name="decoder_00",
                    ar=1,
                    context_length=4096,
                    sdk_model=live,
                ),
            ),
            contexts=(
                CompiledContextArtifact(
                    context_binary_path=context,
                    slice_name="decoder_00",
                    graph_names=("decoder_ar1", "decoder_ar128"),
                    ar_values=(1, 128),
                    target_soc="SM8750",
                    dsp_arch="v79",
                    soc_model=69,
                    weight_sharing=True,
                    native_kv_config_path=native_kv,
                    context_length=4096,
                    sdk_compiled_model=live,
                ),
            ),
            config_artifact_paths=(generated_config,),
            auxiliary_artifact_paths=(auxiliary,),
        )

    def build_standalone_vit(
        self,
        spec: BuildSpec,
        runtime: Mapping[str, Any],
        output_dir: Path,
    ) -> BuildResult:
        if not self.ready:
            raise QairtPreflightError("preflight must precede standalone ViT build")
        self.log.record(
            self.adapter_id,
            "build_standalone_vit",
            runtime=runtime,
        )
        dlc = _write(output_dir / "converted" / "vit.dlc")
        context = _write(output_dir / "contexts" / "vit.bin")
        policy = _write(
            output_dir / "config" / "standalone_vit_build.json",
            b'{"lane":"low_level_python_api"}',
        )
        return BuildResult(
            variants=(),
            transformed_slices=(),
            converted_models=(
                ConvertedModelArtifact(
                    model_path=dlc,
                    source_model_path=spec.sources.text.onnx_path,
                    quantization_mode=spec.quantization.mode.value,
                    slice_name=None,
                    ar=None,
                    context_length=None,
                ),
            ),
            contexts=(
                CompiledContextArtifact(
                    context_binary_path=context,
                    slice_name="vit",
                    graph_names=("vit",),
                    ar_values=(1,),
                    target_soc="SM8750",
                    dsp_arch="v79",
                    soc_model=69,
                    weight_sharing=False,
                    native_kv_config_path=None,
                ),
            ),
            config_artifact_paths=(policy,),
        )

    def load_compiled(self, path: str | Path) -> str:
        self.log.record(self.adapter_id, "load_compiled", path=str(path))
        return f"loaded:{path}"

    def initialize_execution(
        self,
        context: Any,
        *,
        device: Any = None,
        **options: Any,
    ) -> Any:
        self.log.record(
            self.adapter_id,
            "initialize_execution",
            context=str(context),
            device=device,
        )
        self.initialized_contexts.add(str(context))
        return context

    def release_execution(self, context: Any) -> None:
        self.log.record(self.adapter_id, "release_execution", context=str(context))
        self.initialized_contexts.discard(str(context))

    def capture_device_execution(
        self,
        context: Any,
        inputs: Mapping[str, np.ndarray],
        *,
        graph_name: str,
        device: Any = None,
        native_io: bool = False,
        level: str = "detailed",
        working_dir: Any = None,
        **options: Any,
    ) -> dict[str, Any]:
        if str(context) in self.initialized_contexts:
            # Mirrors the real adapter: an initialized model carries an
            # execution context built with profiling disabled, so QAIRT would
            # report nothing.
            raise QairtConfigurationError(
                "capture_device_execution must run before initialize_execution"
            )
        self.log.record(
            self.adapter_id,
            "capture_device_execution",
            context=str(context),
            graph_name=graph_name,
            level=level,
        )
        return parse_device_execution(SAMPLE_PROFILING_REPORT)

    def create_genai_executor(
        self,
        path: str | Path,
        *,
        device: Any,
    ) -> Any:
        self.log.record(
            self.adapter_id,
            "create_genai_executor",
            path=str(path),
            device=device,
        )
        log = self.log
        adapter_id = self.adapter_id

        class Executor:
            def generate(self, prompt: Any) -> Any:
                log.record(
                    adapter_id,
                    "generate",
                    prompt=prompt,
                )
                return SimpleNamespace(
                    generated_text="ok",
                    metrics={
                        "prompt_processing_rate": 100.0,
                        "token_generation_rate": 50.0,
                    },
                )

        return Executor()

    def clean_genai_executor(self, executor: Any) -> None:
        self.log.record(
            self.adapter_id,
            "clean_genai_executor",
            executor=executor,
        )

    def build_genai_container(
        self,
        model_path: str | Path,
        *,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> GenAIContainerBuildResult:
        if not self.ready:
            raise QairtPreflightError("preflight must precede GenAI build")
        self.log.record(
            self.adapter_id,
            "build_genai_container",
            model_path=str(model_path),
            output_dir=str(output_dir),
            kwargs=kwargs,
        )
        destination = Path(output_dir)
        metadata = _write(
            destination / "qairt_agent_genai_build.json",
            b'{"lane": "genai"}',
        )
        _write(destination / "textGenerator" / "decoder.bin", b"genai-context")
        return GenAIContainerBuildResult(
            container_path=destination,
            metadata_path=metadata,
            family="qwen3",
            builder_class="GenAIBuilderHTP",
            container_class="LLMContainer",
            factory_support="generic_fallback",
            compatibility_mode="generic_fallback_requires_device_validation",
            compatibility_notes=("device golden validation required",),
            ar_values=(1, 128),
            context_lengths=(4096,),
            num_splits=4,
            split_embedding=True,
            split_lm_head=True,
            target_soc="SM8750",
            dsp_arch="v79",
            soc_model=69,
            weight_sharing=True,
            native_kv=True,
            runtime_supported=True,
            raw_slices=(
                GenAIRawSliceArtifact(
                    slice_id="split_000",
                    context_binary_path=destination
                    / "textGenerator"
                    / "decoder.bin",
                    graph_names_by_ar={
                        1: "decoder_ar1",
                        128: "decoder_ar128",
                    },
                    input_names=("x",),
                    output_names=("y",),
                ),
            ),
            raw_tensor_runtime_supported=True,
            raw_tensor_runtime_notes=("fake raw route",),
            sdk_container=_PoisonLiveObject(),
        )

    def build_qwen35_omni_components(
        self,
        model_path: str | Path,
        *,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> GenAIContainerBuildResult:
        if not self.ready:
            raise QairtPreflightError("preflight must precede Omni build")
        self.log.record(
            self.adapter_id,
            "build_qwen35_omni_components",
            model_path=str(model_path),
            output_dir=str(output_dir),
            kwargs=kwargs,
        )
        destination = Path(output_dir)
        audio = destination / "audioEncoder"
        text = destination / "textGenerator"
        _write(audio / "audio.bin")
        _write(text / "decoder.bin")
        metadata = _write(
            destination / "qairt_agent_genai_build.json",
            b'{"lane":"qwen3_5_omni_component_packaging"}',
        )
        return GenAIContainerBuildResult(
            container_path=destination,
            metadata_path=metadata,
            family="qwen3.5-omni",
            builder_class="Qwen3_5BuilderHTP",
            audio_builder_class="Qwen3OmniAudioEncoderBuilderHTP",
            container_class="WorkflowContainer",
            factory_support="explicit",
            compatibility_mode="explicit_components_runtime_unsupported",
            compatibility_notes=("runtime unsupported",),
            ar_values=(1, 128),
            context_lengths=(4096,),
            num_splits=4,
            split_embedding=True,
            split_lm_head=True,
            target_soc="SM8750",
            dsp_arch="v79",
            soc_model=69,
            weight_sharing=True,
            native_kv=True,
            runtime_supported=False,
            attached_ar_values=(1, 128),
            audio_container_path=audio,
            text_container_path=text,
            sdk_container=_PoisonLiveObject(),
        )

    def compile_context(
        self,
        models: Any,
        *,
        output_path: str | Path,
        graph_names: Any,
        ar_values: Any,
        target_soc: str,
        dsp_arch: str,
        soc_model: int,
        slice_name: str | None = None,
        weight_sharing: bool = True,
        native_kv_config: Any = None,
        native_kv_expectations: Any = (),
        qwen35_validation_evidence: Any = None,
        context_length: int | None = None,
        **kwargs: Any,
    ) -> CompiledContextArtifact:
        if not self.ready:
            raise QairtPreflightError("preflight must precede compile_context")
        self.log.record(
            self.adapter_id,
            "compile_context",
            models=tuple(models),
            graph_names=tuple(graph_names),
            ar_values=tuple(ar_values),
            native_kv_config=native_kv_config,
            native_kv_expectations=tuple(native_kv_expectations),
            qwen35_validation_evidence=qwen35_validation_evidence,
            kwargs=kwargs,
        )
        destination = _write(Path(output_path))
        return CompiledContextArtifact(
            context_binary_path=destination,
            slice_name=slice_name,
            graph_names=tuple(graph_names),
            ar_values=tuple(ar_values),
            target_soc=target_soc,
            dsp_arch=dsp_arch,
            soc_model=soc_model,
            weight_sharing=weight_sharing,
            native_kv_config_path=None,
            context_length=context_length,
        )

    def run_graph(
        self,
        context: Any,
        inputs: Mapping[str, np.ndarray],
        *,
        graph_name: str,
        native_io: bool = False,
        **options: Any,
    ) -> Any:
        if not self.ready:
            raise QairtPreflightError("preflight must precede run_graph")
        copied = {name: np.asarray(value).copy() for name, value in inputs.items()}
        self.log.record(
            self.adapter_id,
            "run_graph",
            context=str(context),
            graph_name=graph_name,
            inputs=copied,
            native_io=native_io,
            options=options,
        )
        if graph_name.startswith("embedding"):
            outputs = {"h": copied["x"] + 1.0}
        elif graph_name.startswith("decoder"):
            source = copied["hidden"] if "hidden" in copied else copied["x"]
            outputs = {"y": source * 3.0 if "hidden" in copied else source.copy()}
        elif graph_name.startswith("state"):
            next_state = copied["kv_in"] + copied["token"]
            outputs = {"logits": next_state.copy(), "kv_out": next_state}
        else:
            outputs = {"y": copied["x"] * 2.0}
        if self.result_style == "data":
            return ExecutionResultLike(data={graph_name: [outputs]})
        return outputs

    def profile(
        self,
        context: Any,
        inputs: Mapping[str, np.ndarray],
        *,
        graph_name: str,
        device: Any = None,
        native_io: bool = False,
        level: str = "detailed",
        option: str = "optrace",
        **execution_options: Any,
    ) -> Any:
        self.log.record(
            self.adapter_id,
            "profile",
            context=str(context),
            graph_name=graph_name,
            device=device,
            level=level,
            option=option,
        )
        execution_result = self.run_graph(
            context,
            inputs,
            graph_name=graph_name,
            native_io=native_io,
            device=device,
            **execution_options,
        )
        return SimpleNamespace(
            execution_result=execution_result,
            reports=(
                {
                    "ops": [
                        {
                            "op_id": "MatMul_0",
                            "cycles": self.profile_cycles,
                            "critical_path": True,
                            "lineage": {
                                "layer": 0,
                                "op_type": "MatMul",
                                "output_tensor": "y",
                            },
                        },
                        {
                            "op_id": "Add_1",
                            "thread_cycles": [20.0, 30.0],
                            "lineage": {
                                "layer": 0,
                                "op_type": "Add",
                                "output_tensor": "y",
                            },
                        },
                    ]
                },
            ),
            graph_name=graph_name,
            level=level,
            option=option,
        )


class FakeAdapterFactory:
    def __init__(
        self,
        *,
        result_style: str = "mapping",
        profile_cycles: float = 100.0,
    ) -> None:
        self.log = AdapterLog()
        self.result_style = result_style
        self.profile_cycles = float(profile_cycles)
        self.adapters: list[FakeAdapter] = []

    def __call__(self) -> FakeAdapter:
        adapter = FakeAdapter(
            self.log,
            result_style=self.result_style,
            profile_cycles=self.profile_cycles,
        )
        self.adapters.append(adapter)
        return adapter


class MultiArFakeAdapter(FakeAdapter):
    """Extend the default fake build with a real AR128 ONNX/DLC variant."""

    def build(
        self,
        spec: BuildSpec,
        runtime: Mapping[str, Any],
        output_dir: Path,
        **kwargs: Any,
    ) -> BuildResult:
        result = super().build(
            spec,
            runtime,
            output_dir,
            **kwargs,
        )
        if 128 not in spec.sequence.ars:
            return result
        variant_model = _write_onnx(
            output_dir / "variants" / "ar128.onnx"
        )
        variant_encodings = _write(
            output_dir / "variants" / "ar128.encodings",
            b'{"variant": 128}',
        )
        slice_model = _write_onnx(
            output_dir / "slices" / "ar128" / "decoder_00.onnx"
        )
        slice_encodings = _write(
            output_dir / "slices" / "ar128" / "decoder_00.encodings",
            b'{"slice": 0, "ar": 128}',
        )
        dlc = _write(
            output_dir / "converted" / "ar128" / "decoder_00.dlc"
        )
        return replace(
            result,
            variants=result.variants
            + (
                ModelVariantArtifact(
                    model_path=variant_model,
                    encodings_path=variant_encodings,
                    ar=128,
                    context_length=4096,
                    source_kind="derived",
                    family="qwen3",
                ),
            ),
            transformed_slices=result.transformed_slices
            + (
                TransformedSliceArtifact(
                    slice_name="decoder_00",
                    split_index=1,
                    model_path=slice_model,
                    encodings_path=slice_encodings,
                    ar=128,
                    context_length=4096,
                ),
            ),
            converted_models=result.converted_models
            + (
                ConvertedModelArtifact(
                    model_path=dlc,
                    source_model_path=slice_model,
                    quantization_mode=spec.quantization.mode.value,
                    slice_name="decoder_00",
                    ar=128,
                    context_length=4096,
                ),
            ),
        )


class MultiArFakeAdapterFactory(FakeAdapterFactory):
    def __call__(self) -> FakeAdapter:
        return MultiArFakeAdapter(
            self.log,
            result_style=self.result_style,
            profile_cycles=self.profile_cycles,
        )


class FakeVlAdapter(FakeAdapter):
    def build(
        self,
        spec: BuildSpec,
        runtime: Mapping[str, Any],
        output_dir: Path,
        **kwargs: Any,
    ) -> BuildResult:
        text = super().build(spec, runtime, output_dir, **kwargs)
        assert spec.sources.vision is not None
        vision_dlc = _write(output_dir / "converted" / "vision_projector.dlc")
        vision_context = _write(
            output_dir / "contexts" / "vision_projector.bin"
        )
        return BuildResult(
            variants=text.variants,
            transformed_slices=text.transformed_slices
            + (
                TransformedSliceArtifact(
                    slice_name="vision_projector",
                    split_index=0,
                    model_path=spec.sources.vision.onnx_path,
                    encodings_path=spec.sources.vision.encodings_path,
                    ar=None,
                    context_length=None,
                ),
            ),
            converted_models=text.converted_models
            + (
                ConvertedModelArtifact(
                    model_path=vision_dlc,
                    source_model_path=spec.sources.vision.onnx_path,
                    quantization_mode=spec.quantization.mode.value,
                    slice_name="vision_projector",
                    ar=None,
                    context_length=None,
                ),
            ),
            contexts=text.contexts
            + (
                CompiledContextArtifact(
                    context_binary_path=vision_context,
                    slice_name="vision_projector",
                    graph_names=("vision_projector",),
                    ar_values=(1,),
                    target_soc="SM8750",
                    dsp_arch="v79",
                    soc_model=69,
                    weight_sharing=False,
                    native_kv_config_path=None,
                    context_length=None,
                ),
            ),
            diagnostic_contexts=text.diagnostic_contexts,
            config_artifact_paths=text.config_artifact_paths,
            auxiliary_artifact_paths=text.auxiliary_artifact_paths,
        )

    def run_graph(
        self,
        context: Any,
        inputs: Mapping[str, np.ndarray],
        *,
        graph_name: str,
        native_io: bool = False,
        **options: Any,
    ) -> Any:
        if graph_name != "vision_projector":
            return super().run_graph(
                context,
                inputs,
                graph_name=graph_name,
                native_io=native_io,
                **options,
            )
        copied = {
            name: np.asarray(value).copy()
            for name, value in inputs.items()
        }
        self.log.record(
            self.adapter_id,
            "run_graph",
            context=str(context),
            graph_name=graph_name,
            inputs=copied,
            native_io=native_io,
            options=options,
        )
        outputs = {"visual_embeddings": copied["pixel_values"]}
        if self.result_style == "data":
            return ExecutionResultLike(data={graph_name: [outputs]})
        return outputs


class FakeVlAdapterFactory(FakeAdapterFactory):
    def __call__(self) -> FakeVlAdapter:
        return FakeVlAdapter(self.log, result_style=self.result_style)


class FakeQualityModesAdapter(FakeAdapter):
    """Two-slice identity reference with a device error in the first slice."""

    def build(
        self,
        spec: BuildSpec,
        runtime: Mapping[str, Any],
        output_dir: Path,
        **kwargs: Any,
    ) -> BuildResult:
        if not self.ready:
            raise QairtPreflightError("preflight must precede build")
        self.log.record(
            self.adapter_id,
            "build",
            runtime=runtime,
            kwargs=kwargs,
            diagnostic_outputs=spec.compile.enable_intermediate_outputs,
        )
        variant = _write_onnx(output_dir / "variants" / "ar1.onnx")
        variant_encodings = _write(
            output_dir / "variants" / "ar1.encodings",
            b"{}",
        )
        slice0 = _write_onnx(
            output_dir / "slices" / "slice0.onnx",
            input_name="x",
            output_name="hidden",
        )
        slice1 = _write_onnx(
            output_dir / "slices" / "slice1.onnx",
            input_name="hidden",
            output_name="y",
        )
        slice0_encodings = _write(
            output_dir / "slices" / "slice0.encodings",
            b"{}",
        )
        slice1_encodings = _write(
            output_dir / "slices" / "slice1.encodings",
            b"{}",
        )
        dlc0 = _write(output_dir / "converted" / "slice0.dlc")
        dlc1 = _write(output_dir / "converted" / "slice1.dlc")
        context0 = _write(output_dir / "contexts" / "slice0.bin")
        context1 = _write(output_dir / "contexts" / "slice1.bin")
        diagnostics: tuple[CompiledContextArtifact, ...] = ()
        if spec.compile.enable_intermediate_outputs:
            diagnostic0 = _write(
                output_dir / "diagnostic_contexts" / "slice0.bin"
            )
            diagnostic1 = _write(
                output_dir / "diagnostic_contexts" / "slice1.bin"
            )
            diagnostics = (
                CompiledContextArtifact(
                    context_binary_path=diagnostic0,
                    slice_name="slice0",
                    graph_names=("slice0_ar1",),
                    ar_values=(1,),
                    target_soc="SM8750",
                    dsp_arch="v79",
                    soc_model=69,
                    weight_sharing=False,
                    native_kv_config_path=None,
                    context_length=4096,
                ),
                CompiledContextArtifact(
                    context_binary_path=diagnostic1,
                    slice_name="slice1",
                    graph_names=("slice1_ar1",),
                    ar_values=(1,),
                    target_soc="SM8750",
                    dsp_arch="v79",
                    soc_model=69,
                    weight_sharing=False,
                    native_kv_config_path=None,
                    context_length=4096,
                ),
            )
        return BuildResult(
            variants=(
                ModelVariantArtifact(
                    model_path=variant,
                    encodings_path=variant_encodings,
                    ar=1,
                    context_length=4096,
                    source_kind="derived",
                    family="qwen3",
                ),
            ),
            transformed_slices=(
                TransformedSliceArtifact(
                    slice_name="slice0",
                    split_index=0,
                    model_path=slice0,
                    encodings_path=slice0_encodings,
                    ar=1,
                    context_length=4096,
                ),
                TransformedSliceArtifact(
                    slice_name="slice1",
                    split_index=1,
                    model_path=slice1,
                    encodings_path=slice1_encodings,
                    ar=1,
                    context_length=4096,
                ),
            ),
            converted_models=(
                ConvertedModelArtifact(
                    model_path=dlc0,
                    source_model_path=slice0,
                    quantization_mode=spec.quantization.mode.value,
                    slice_name="slice0",
                    ar=1,
                    context_length=4096,
                ),
                ConvertedModelArtifact(
                    model_path=dlc1,
                    source_model_path=slice1,
                    quantization_mode=spec.quantization.mode.value,
                    slice_name="slice1",
                    ar=1,
                    context_length=4096,
                ),
            ),
            contexts=(
                CompiledContextArtifact(
                    context_binary_path=context0,
                    slice_name="slice0",
                    graph_names=("slice0_ar1",),
                    ar_values=(1,),
                    target_soc="SM8750",
                    dsp_arch="v79",
                    soc_model=69,
                    weight_sharing=False,
                    native_kv_config_path=None,
                    context_length=4096,
                ),
                CompiledContextArtifact(
                    context_binary_path=context1,
                    slice_name="slice1",
                    graph_names=("slice1_ar1",),
                    ar_values=(1,),
                    target_soc="SM8750",
                    dsp_arch="v79",
                    soc_model=69,
                    weight_sharing=False,
                    native_kv_config_path=None,
                    context_length=4096,
                ),
            ),
            diagnostic_contexts=diagnostics,
        )

    def run_graph(
        self,
        context: Any,
        inputs: Mapping[str, np.ndarray],
        *,
        graph_name: str,
        native_io: bool = False,
        **options: Any,
    ) -> Any:
        copied = {
            name: np.asarray(value).copy()
            for name, value in inputs.items()
        }
        self.log.record(
            self.adapter_id,
            "run_graph",
            context=str(context),
            graph_name=graph_name,
            inputs=copied,
            native_io=native_io,
            options=options,
        )
        if graph_name == "slice0_ar1":
            return {"hidden": copied["x"] + 1.0}
        if graph_name == "slice1_ar1":
            return {"y": copied["hidden"].copy()}
        raise AssertionError(f"unexpected graph {graph_name}")


class FakeQualityModesAdapterFactory(FakeAdapterFactory):
    def __call__(self) -> FakeQualityModesAdapter:
        return FakeQualityModesAdapter(
            self.log,
            result_style=self.result_style,
        )


class FakeDeviceRuntime:
    def __init__(self) -> None:
        self.device = object()
        self.calls: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def stage(self, _adapter: Any, **kwargs: Any):
        self.calls.append(dict(kwargs))
        yield SimpleNamespace(
            device=self.device,
            identifier="TEST@localhost:5037",
            adb=SimpleNamespace(
                attempt_dir=(
                    "/data/local/tmp/qairt-agent/test/stage/attempt-001/"
                )
            ),
        )


def _fake_agent(factory: FakeAdapterFactory) -> QairtAgent:
    return QairtAgent(
        adapter_factory=factory,
        device_runtime=FakeDeviceRuntime(),
    )


def test_offline_plan_and_generate_config_publish_deterministic_manifests(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)

    plan = agent.plan(spec, offline=True)
    generated = agent.generate_config(spec)

    assert plan.ok and plan.manifest is not None
    assert generated.ok and generated.manifest is not None
    assert factory.log.calls == []
    assert plan.data is not None
    assert plan.data["offline"] is True
    assert plan.data["preflight"] is None
    assert {item["slice"] for item in plan.data["contexts"]} == {
        "decoder_00",
        "lm_head",
    }
    assert plan.data["effective_config"]["family"] == "qwen3"
    assert plan.data["effective_config"]["sequence"]["native_kv"] is True

    for result, stage_name in ((plan, "plan"), (generated, "generate_config")):
        assert result.manifest is not None
        manifest = _load_run(result.manifest)
        assert manifest.revision == 1
        assert manifest.metadata["state_model"] == "stateless"
        assert manifest.stages[-1].name == stage_name
        assert len(str(manifest.stages[-1].metrics["stage_key"])) == 64
        for artifact in manifest.artifacts:
            verify_artifact(artifact)


def test_continuation_requires_exact_manifest_sha_before_adapter_use(
    tmp_path: Path,
) -> None:
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None
    calls_before = list(factory.log.calls)
    manifests_before = set(plan.manifest.path.parent.glob("manifest-*.json"))

    result = agent.prepare_vectors(
        plan.manifest.path,
        "0" * 64,
        config={
            "cases": [
                {
                    "case_id": "bad-sha",
                    "inputs": {"x": np.array([1.0], dtype=np.float32)},
                }
            ]
        },
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.ARTIFACT_HASH_MISMATCH
    assert result.manifest is None
    assert factory.log.calls == calls_before
    assert set(plan.manifest.path.parent.glob("manifest-*.json")) == manifests_before


def test_build_collects_and_verifies_all_materialized_artifacts(
    tmp_path: Path,
) -> None:
    factory = FakeAdapterFactory()
    result = _fake_agent(factory).build(_make_spec(tmp_path))

    assert result.ok and result.manifest is not None
    assert factory.log.names()[:2] == ["preflight", "build"]
    manifest = _load_run(result.manifest)
    paths = {artifact.path.name for artifact in manifest.artifacts}
    assert {
        "model.onnx",
        "model.encodings",
        "effective_config.json",
        "ar1.onnx",
        "ar1.encodings",
        "ar1.data",
        "decoder_00.onnx",
        "decoder_00.encodings",
        "decoder_00.data",
        "decoder_00.dlc",
        "decoder_00.bin",
        "decoder_00.native_kv.json",
        "adapter_config.json",
        "build.log",
        "slice_routes_cl4096.json",
    } <= paths
    for artifact in manifest.artifacts:
        verify_artifact(artifact)
    assert result.data is not None
    json.dumps(result.data)
    assert manifest.stages[-1].metrics["production_context_count"] == 1


def test_low_level_build_retargets_source_vectors_and_auto_runs_followups(
    tmp_path: Path,
) -> None:
    source_vectors = _vector_case(
        tmp_path,
        inputs={"x": np.array([2.0, 3.0, 4.0], dtype=np.float32)},
        goldens={"y": np.array([999.0], dtype=np.float32)},
        case_id="source-ar3",
    )
    spec = _make_spec(
        tmp_path,
        vectors={
            "mode": "provided",
            "validation_manifest": source_vectors,
        },
        sequence={
            "ars": [1],
            "context_lengths": [4096],
            "weight_sharing": False,
            "native_kv": True,
        },
    )
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)

    built = agent.build(spec)

    assert built.ok, built.error
    assert built.manifest is not None and built.data is not None
    binding = built.data["validation_vector_bindings"][0]
    assert binding["binding"] == "derived_from_source_manifest"
    assert binding["reference_source"] == "onnxruntime"
    runtime_index_path = Path(built.data["runtime_index"]["path"])
    runtime_index = json.loads(runtime_index_path.read_text(encoding="utf-8"))
    derived_manifest = Path(
        runtime_index["vectors"]["validation_manifests_by_ar"]["1"]
    )
    assert derived_manifest != source_vectors
    np.testing.assert_array_equal(
        VectorPreparer.load_tensors(derived_manifest)["x"],
        np.array([2.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        VectorPreparer.load_tensors(
            derived_manifest,
            section="goldens",
        )["y"],
        np.array([2.0], dtype=np.float32),
    )

    validated = agent.validate(
        built.manifest.path,
        built.manifest.sha256,
    )
    assert validated.ok, validated.error
    assert validated.data is not None
    assert validated.data["coverage"]["complete"] is True
    assert validated.data["coverage"]["executed_ars"] == [1]
    assert validated.data["first_chain_error"] is None
    assert validated.data["reference_source"] == "provided"

    benchmarked = agent.benchmark(
        validated.manifest.path,
        validated.manifest.sha256,
        config={
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )
    assert benchmarked.ok, benchmarked.error
    assert benchmarked.data is not None
    assert benchmarked.data["scope"] == "graph"
    assert benchmarked.data["runtime_binding"]["ar"] == 1


def test_low_level_auto_followups_cover_every_requested_ar_and_keep_reports(
    tmp_path: Path,
) -> None:
    ar1_vectors = _vector_case(
        tmp_path / "ar1",
        inputs={"x": np.array([1.0], dtype=np.float32)},
        goldens={"y": np.array([1.0], dtype=np.float32)},
        case_id="low-level-ar1",
    )
    ar128_vectors = _vector_case(
        tmp_path / "ar128",
        inputs={"x": np.array([128.0], dtype=np.float32)},
        goldens={"y": np.array([128.0], dtype=np.float32)},
        case_id="low-level-ar128",
    )
    spec = _make_spec(
        tmp_path,
        vectors={
            "mode": "provided",
            "validation_manifests_by_ar": {
                1: ar1_vectors,
                128: ar128_vectors,
            },
        },
        sequence={
            "ars": [1, 128],
            "context_lengths": [4096],
            "weight_sharing": True,
            "native_kv": False,
        },
    )
    factory = MultiArFakeAdapterFactory()
    agent = _fake_agent(factory)
    built = agent.build(spec)
    assert built.ok, built.error
    assert built.manifest is not None

    validated = agent.validate(
        built.manifest.path,
        built.manifest.sha256,
    )
    assert validated.ok, validated.error
    assert validated.manifest is not None
    assert validated.data is not None
    assert validated.data["coverage"] == {
        "mode": "all_requested_ars",
        "requested_ars": [1, 128],
        "executed_ars": [1, 128],
        "missing_ars": [],
        "complete": True,
        "context_lengths": [4096],
    }
    assert set(validated.data["results_by_ar"]) == {"1", "128"}
    assert validated.data["reference_source"] == "provided"
    assert validated.data["first_chain_error"] is None
    assert validated.data["first_chain_error_ar"] is None

    benchmarked = agent.benchmark(
        validated.manifest.path,
        validated.manifest.sha256,
        config={
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
            "optrace": True,
        },
    )
    assert benchmarked.ok, benchmarked.error
    assert benchmarked.manifest is not None
    assert benchmarked.data is not None
    assert benchmarked.data["coverage"]["complete"] is True
    assert benchmarked.data["coverage"]["executed_ars"] == [1, 128]
    assert set(benchmarked.data["results_by_ar"]) == {"1", "128"}

    final = _load_run(benchmarked.manifest)
    named = {
        artifact.logical_name: artifact
        for artifact in final.artifacts
        if artifact.logical_name is not None
    }
    for logical_name in (
        "sqnr_report_ar1",
        "sqnr_report_ar128",
        "sqnr_report",
        "latency_report_ar1",
        "latency_report_ar128",
        "latency_report",
        "optrace_evidence_ar1",
        "optrace_evidence_ar128",
        "optrace_evidence",
    ):
        assert logical_name in named
        verify_artifact(named[logical_name])

    ar1_report = json.loads(
        named["sqnr_report_ar1"].path.read_text(encoding="utf-8")
    )
    ar128_report = json.loads(
        named["sqnr_report_ar128"].path.read_text(encoding="utf-8")
    )
    assert ar1_report["runtime_binding"]["ar"] == 1
    assert ar128_report["runtime_binding"]["ar"] == 128

    graph_inputs = [
        (
            details["graph_name"],
            float(details["inputs"]["x"][0]),
        )
        for _, name, details in factory.log.calls
        if name == "run_graph"
    ]
    assert ("decoder_ar1", 1.0) in graph_inputs
    assert ("decoder_ar128", 128.0) in graph_inputs

    diagnosed = agent.diagnose_quality(
        benchmarked.manifest.path,
        benchmarked.manifest.sha256,
    )
    assert diagnosed.ok, diagnosed.error
    assert diagnosed.data is not None
    assert diagnosed.data["diagnosis_kind"] == "latency"
    assert {
        item["lineage"]["ar"]
        for item in diagnosed.data["attributions"]
    } == {1, 128}


def test_multi_ar_quality_diagnosis_preserves_failing_ar(
    tmp_path: Path,
) -> None:
    ar1_vectors = _vector_case(
        tmp_path / "ar1",
        inputs={"x": np.array([1.0], dtype=np.float32)},
        goldens={"y": np.array([1.0], dtype=np.float32)},
        case_id="quality-ar1",
    )
    ar128_vectors = _vector_case(
        tmp_path / "ar128",
        inputs={"x": np.array([128.0], dtype=np.float32)},
        goldens={"y": np.array([129.0], dtype=np.float32)},
        case_id="quality-ar128",
    )
    spec = _make_spec(
        tmp_path,
        vectors={
            "mode": "provided",
            "validation_manifests_by_ar": {
                1: ar1_vectors,
                128: ar128_vectors,
            },
        },
        sequence={
            "ars": [1, 128],
            "context_lengths": [4096],
            "weight_sharing": True,
            "native_kv": False,
        },
    )
    agent = _fake_agent(MultiArFakeAdapterFactory())
    built = agent.build(spec)
    assert built.ok and built.manifest is not None

    validated = agent.validate(
        built.manifest.path,
        built.manifest.sha256,
    )
    assert validated.ok, validated.error
    assert validated.manifest is not None
    assert validated.data is not None
    assert validated.data["first_chain_error_ar"] == 128

    diagnosed = agent.diagnose_quality(
        validated.manifest.path,
        validated.manifest.sha256,
    )
    assert diagnosed.ok, diagnosed.error
    assert diagnosed.data is not None
    assert diagnosed.data["diagnosis_kind"] == "quality"
    assert diagnosed.data["first_observed"]["ar"] == 128
    assert diagnosed.data["first_observed"]["report_scope"].startswith(
        "ar128/"
    )
    assert {
        item["ar"] for item in diagnosed.data["attributions"]
    } == {128}


def test_low_level_multi_ar_build_fails_closed_when_one_vector_is_missing(
    tmp_path: Path,
) -> None:
    ar1_vectors = _vector_case(
        tmp_path / "ar1",
        inputs={"x": np.array([1.0], dtype=np.float32)},
        goldens={"y": np.array([1.0], dtype=np.float32)},
        case_id="only-ar1",
    )
    spec = _make_spec(
        tmp_path,
        vectors={
            "mode": "provided",
            "validation_manifests_by_ar": {1: ar1_vectors},
        },
        sequence={
            "ars": [1, 128],
            "context_lengths": [4096],
            "weight_sharing": True,
            "native_kv": False,
        },
    )

    built = _fake_agent(MultiArFakeAdapterFactory()).build(spec)

    assert not built.ok
    assert built.error is not None
    assert built.error.code is ErrorCode.INVALID_SPEC
    assert built.error.stage == "build"
    assert "no validation vector manifest is available for AR128" in (
        built.error.message
    )


def test_explicit_ar_override_keeps_single_ar_followups(
    tmp_path: Path,
) -> None:
    ar1_vectors = _vector_case(
        tmp_path / "ar1",
        inputs={"x": np.array([1.0], dtype=np.float32)},
        goldens={"y": np.array([1.0], dtype=np.float32)},
        case_id="override-ar1",
    )
    ar128_vectors = _vector_case(
        tmp_path / "ar128",
        inputs={"x": np.array([128.0], dtype=np.float32)},
        goldens={"y": np.array([128.0], dtype=np.float32)},
        case_id="override-ar128",
    )
    spec = _make_spec(
        tmp_path,
        vectors={
            "mode": "provided",
            "validation_manifests_by_ar": {
                1: ar1_vectors,
                128: ar128_vectors,
            },
        },
        sequence={
            "ars": [1, 128],
            "context_lengths": [4096],
            "weight_sharing": True,
            "native_kv": False,
        },
    )
    factory = MultiArFakeAdapterFactory()
    agent = _fake_agent(factory)
    built = agent.build(spec)
    assert built.ok and built.manifest is not None
    factory.log.calls.clear()

    validated = agent.validate(
        built.manifest.path,
        built.manifest.sha256,
        config={"ar": 128},
    )
    assert validated.ok and validated.manifest is not None
    assert validated.data is not None
    assert validated.data["coverage"]["mode"] == "single_ar_override"
    assert validated.data["coverage"]["executed_ars"] == [128]

    benchmarked = agent.benchmark(
        validated.manifest.path,
        validated.manifest.sha256,
        config={
            "ar": 128,
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )
    assert benchmarked.ok and benchmarked.data is not None
    assert benchmarked.data["coverage"]["mode"] == "single_ar_override"
    assert {
        details["graph_name"]
        for _, name, details in factory.log.calls
        if name == "run_graph"
    } == {"decoder_ar128"}


def test_validate_executes_all_requested_sqnr_modes_with_reference_slice_chain(
    tmp_path: Path,
) -> None:
    vectors = _vector_case(
        tmp_path,
        inputs={"x": np.array([2.0], dtype=np.float32)},
        goldens={"y": np.array([2.0], dtype=np.float32)},
        case_id="quality-modes",
    )
    spec = _make_spec(
        tmp_path,
        vectors={
            "mode": "provided",
            "validation_manifest": vectors,
        },
        sequence={
            "ars": [1],
            "context_lengths": [4096],
            "weight_sharing": False,
            "native_kv": False,
        },
        quality={
            "sqnr_modes": [
                "full_reference",
                "teacher_forced",
                "chain",
            ],
            "dump_intermediates_on_failure": True,
        },
    )
    factory = FakeQualityModesAdapterFactory()
    agent = _fake_agent(factory)

    built = agent.build(spec)
    assert built.ok, built.error
    assert built.manifest is not None
    built_manifest = _load_run(built.manifest)
    assert built_manifest.build_spec.compile.enable_intermediate_outputs is True
    assert (
        built_manifest.stages[-1].metrics["diagnostic_context_count"]
        == 2
    )

    validated = agent.validate(
        built.manifest.path,
        built.manifest.sha256,
    )

    assert validated.ok, validated.error
    assert validated.data is not None
    payload = validated.data
    assert payload["requested_modes"] == [
        "full_reference",
        "teacher_forced",
        "chain",
    ]
    assert payload["executed_modes"] == [
        "full_reference",
        "teacher_forced",
        "chain",
    ]
    assert payload["mode_reports"]["full_reference"][
        "first_chain_error"
    ] == ["model", "y"]
    assert payload["mode_reports"]["teacher_forced"][
        "first_teacher_error"
    ] == ["slice0", "hidden"]
    assert payload["mode_reports"]["chain"]["first_chain_error"] == [
        "slice0",
        "hidden",
    ]
    observations = {
        (item["slice_id"], item["tensor_name"]): item
        for item in payload["observations"]
    }
    assert observations[("slice1", "y")]["teacher_forced"]["status"] == "exact"
    assert observations[("slice1", "y")]["attribution"] == "propagated_only"
    assert {
        item["reference_source"]
        for item in payload["slice_reference_evidence"]
    } == {"onnxruntime_slice_chain"}
    diagnostic = payload["diagnostic_evidence"]
    assert diagnostic["triggered"] is True
    assert diagnostic["status"] == "ready"
    assert diagnostic["op_level_dump_available"] is True
    assert {item["slice"] for item in diagnostic["contexts"]} == {
        "slice0",
        "slice1",
    }
    slice1_inputs = [
        call["inputs"]["hidden"]
        for _, name, call in factory.log.calls
        if name == "run_graph" and call["graph_name"] == "slice1_ar1"
    ]
    assert len(slice1_inputs) == 2
    assert sorted(float(value[0]) for value in slice1_inputs) == [2.0, 3.0]


def test_teacher_forced_auto_capture_fails_closed_with_missing_slice_models(
    tmp_path: Path,
) -> None:
    vectors = _vector_case(
        tmp_path,
        inputs={"x": np.array([2.0], dtype=np.float32)},
        goldens={"y": np.array([2.0], dtype=np.float32)},
    )
    spec = _make_spec(
        tmp_path,
        vectors={
            "mode": "provided",
            "validation_manifest": vectors,
        },
        sequence={
            "ars": [1],
            "context_lengths": [4096],
            "weight_sharing": False,
            "native_kv": False,
        },
    )
    agent = _fake_agent(FakeAdapterFactory())
    built = agent.build(spec)
    assert built.ok and built.manifest is not None

    failed = agent.validate(
        built.manifest.path,
        built.manifest.sha256,
        config={
            "sqnr_modes": ["teacher_forced"],
            "ar": 1,
            "context_length": 4096,
            "vector_manifest": str(vectors),
            "routes": [
                {
                    "slice_id": "missing_slice",
                    "input_names": ["x"],
                    "output_names": ["y"],
                    "graph_names": {"1": "missing_ar1"},
                }
            ],
            "contexts": {"missing_slice": str(tmp_path / "missing.bin")},
        },
    )

    assert not failed.ok
    assert failed.error is not None
    assert failed.error.code is ErrorCode.INVALID_SPEC
    assert failed.error.details["missing_slice_models"] == ["missing_slice"]


def test_qwen3_vl_auto_runtime_fails_closed_instead_of_running_text_only(
    tmp_path: Path,
) -> None:
    validate_factory = FakeVlAdapterFactory()
    validate_agent = _fake_agent(validate_factory)
    built = validate_agent.build(_make_vl_spec(tmp_path / "validate"))
    assert built.ok and built.manifest is not None and built.data is not None

    runtime_index = json.loads(
        Path(built.data["runtime_index"]["path"]).read_text(encoding="utf-8")
    )
    assert (
        runtime_index["execution_contract"][
            "automatic_end_to_end_supported"
        ]
        is False
    )
    route_path = Path(built.data["slice_routes"][0]["artifact"]["path"])
    route = json.loads(route_path.read_text(encoding="utf-8"))
    assert route["component"] == "text"
    assert route["coverage"] == "text_only"
    assert route["excluded_components"] == ["vision_projector"]
    assert set(route["contexts"]) == {"decoder_00"}

    calls_before = list(validate_factory.log.calls)
    failed_validation = validate_agent.validate(
        built.manifest.path,
        built.manifest.sha256,
    )
    assert not failed_validation.ok
    assert failed_validation.error is not None
    assert failed_validation.error.code is ErrorCode.INVALID_SPEC
    assert "automatic end-to-end execution is unavailable" in (
        failed_validation.error.message
    )
    assert validate_factory.log.calls == calls_before

    benchmark_factory = FakeVlAdapterFactory()
    benchmark_agent = _fake_agent(benchmark_factory)
    benchmark_build = benchmark_agent.build(
        _make_vl_spec(tmp_path / "benchmark")
    )
    assert benchmark_build.ok and benchmark_build.manifest is not None
    calls_before = list(benchmark_factory.log.calls)
    failed_benchmark = benchmark_agent.benchmark(
        benchmark_build.manifest.path,
        benchmark_build.manifest.sha256,
        config={
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )
    assert not failed_benchmark.ok
    assert failed_benchmark.error is not None
    assert failed_benchmark.error.code is ErrorCode.INVALID_SPEC
    assert "automatic end-to-end execution is unavailable" in (
        failed_benchmark.error.message
    )
    assert benchmark_factory.log.calls == calls_before


def test_qwen3_vl_text_component_is_explicitly_labelled_text_only(
    tmp_path: Path,
) -> None:
    factory = FakeVlAdapterFactory()
    agent = _fake_agent(factory)
    built = agent.build(_make_vl_spec(tmp_path))
    assert built.ok and built.manifest is not None

    validated = agent.validate(
        built.manifest.path,
        built.manifest.sha256,
        config={"component": "text"},
    )
    assert validated.ok, validated.error
    assert validated.data is not None and validated.manifest is not None
    assert validated.data["runtime_binding"]["component"] == "text"
    assert validated.data["runtime_binding"]["coverage"] == "text_only"
    assert validated.data["runtime_binding"]["excluded_components"] == [
        "vision_projector"
    ]

    benchmarked = agent.benchmark(
        validated.manifest.path,
        validated.manifest.sha256,
        config={
            "component": "text",
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )
    assert benchmarked.ok, benchmarked.error
    assert benchmarked.data is not None
    assert benchmarked.data["runtime_binding"]["component"] == "text"
    assert benchmarked.data["runtime_binding"]["coverage"] == "text_only"
    run_graph_contexts = [
        details["context"]
        for _, name, details in factory.log.calls
        if name == "run_graph"
    ]
    assert run_graph_contexts
    assert all("vision_projector.bin" not in value for value in run_graph_contexts)


def test_qwen3_vl_vision_component_requires_its_own_vectors_and_reference(
    tmp_path: Path,
) -> None:
    factory = FakeVlAdapterFactory()
    agent = _fake_agent(factory)
    spec = _make_vl_spec(tmp_path)
    assert spec.sources.vision is not None
    built = agent.build(spec)
    assert built.ok and built.manifest is not None
    vision_vectors = VectorPreparer(
        tmp_path / "vision-only-vectors"
    ).prepare_case(
        "vision-only",
        {"pixel_values": np.ones((1, 16), dtype=np.float32)},
    )

    validated = agent.validate(
        built.manifest.path,
        built.manifest.sha256,
        vector_manifest=str(vision_vectors),
        config={"component": "vision"},
    )

    assert validated.ok, validated.error
    assert validated.data is not None
    assert validated.data["reference_source"] == "onnxruntime"
    binding = validated.data["runtime_binding"]
    assert binding["component"] == "vision"
    assert binding["coverage"] == "vision_only"
    assert binding["graph_name"] == "vision_projector"
    assert binding["graph_ar"] == 1
    assert binding["reference_model_path"] == str(
        spec.sources.vision.onnx_path.resolve()
    )
    vision_calls = [
        details
        for _, name, details in factory.log.calls
        if name == "run_graph"
        and details["graph_name"] == "vision_projector"
    ]
    assert len(vision_calls) == 1
    assert set(vision_calls[0]["inputs"]) == {"pixel_values"}

    benchmarked = agent.benchmark(
        validated.manifest.path,
        validated.manifest.sha256,
        config={
            "component": "vision",
            "vector_manifest": str(vision_vectors),
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )
    assert benchmarked.ok, benchmarked.error
    assert benchmarked.data is not None
    assert benchmarked.data["runtime_binding"]["component"] == "vision"
    assert benchmarked.data["runtime_binding"]["coverage"] == "vision_only"


def test_validate_uses_onnxruntime_only_when_golden_is_missing(
    tmp_path: Path,
) -> None:
    vectors = VectorPreparer(tmp_path / "vectors").prepare_case(
        "inputs-only",
        {"x": np.array([2.0], dtype=np.float32)},
    )
    spec = _make_spec(
        tmp_path,
        vectors={
            "mode": "provided",
            "validation_manifest": vectors,
        },
        sequence={
            "ars": [1],
            "context_lengths": [4096],
            "weight_sharing": False,
            "native_kv": True,
        },
    )
    agent = _fake_agent(FakeAdapterFactory())
    built = agent.build(spec)
    assert built.ok and built.manifest is not None

    validated = agent.validate(
        built.manifest.path,
        built.manifest.sha256,
    )

    assert validated.ok, validated.error
    assert validated.data is not None
    assert validated.data["reference_source"] == "onnxruntime"
    assert validated.data["observations"]
    run = _load_run(validated.manifest)
    assert any(
        artifact.logical_name == "onnxruntime_reference_manifest"
        for artifact in run.artifacts
    )


def test_worker_attempt_does_not_move_build_artifacts_out_of_output_root(
    tmp_path: Path,
) -> None:
    factory = FakeAdapterFactory()
    spec = _make_spec(tmp_path)
    worker_dir = tmp_path / "jobs" / "work" / "build" / "attempt-002"
    result = _fake_agent(factory).build(
        spec,
        execution_context=StageExecutionContext(
            output_dir=worker_dir,
            attempt=2,
        ),
    )

    assert result.ok and result.manifest is not None
    manifest = _load_run(result.manifest)
    run_root = spec.output_root / "runs" / str(manifest.run_id)
    context = next(
        artifact
        for artifact in manifest.artifacts
        if artifact.path.name == "decoder_00.bin"
    )
    assert context.path.is_relative_to(run_root)
    assert manifest.stages[-1].attempt == 2
    assert not worker_dir.exists()


def test_genai_builder_lane_packages_container_without_low_level_build(
    tmp_path: Path,
) -> None:
    factory = FakeAdapterFactory()
    spec = _make_spec(tmp_path)
    worker_dir = tmp_path / "jobs" / "work" / "build" / "attempt-002"
    result = _fake_agent(factory).build_genai_container(
        spec,
        execution_context=StageExecutionContext(
            output_dir=worker_dir,
            attempt=2,
        ),
    )

    assert result.ok and result.manifest is not None and result.data is not None
    assert factory.log.names() == ["preflight", "build_genai_container"]
    assert result.data["lane"] == "genai_builder_production_packaging"
    assert "sdk_container" not in result.data["genai_container"]
    assert result.data["genai_container"]["runtime_supported"] is True

    manifest = _load_run(result.manifest)
    paths = {artifact.path.name for artifact in manifest.artifacts}
    assert {"qairt_agent_genai_build.json", "decoder.bin"} <= paths
    assert manifest.stages[-1].name == "build_genai_container"
    assert manifest.stages[-1].attempt == 2
    container_context = next(
        artifact
        for artifact in manifest.artifacts
        if artifact.path.name == "decoder.bin"
    )
    assert container_context.path.is_relative_to(
        spec.output_root / "runs" / str(manifest.run_id)
    )
    assert not worker_dir.exists()
    assert (
        manifest.stages[-1].metrics["compatibility_mode"]
        == "generic_fallback_requires_device_validation"
    )
    for artifact in manifest.artifacts:
        verify_artifact(artifact)


def test_genai_builder_saved_container_auto_benchmarks_with_public_executor(
    tmp_path: Path,
) -> None:
    base = _write_onnx(tmp_path / "qwen35" / "ar1.onnx")
    ar128 = _write_onnx(tmp_path / "qwen35" / "ar128.onnx")
    base_enc = _write(tmp_path / "qwen35" / "ar1.encodings", b"{}")
    ar128_enc = _write(tmp_path / "qwen35" / "ar128.encodings", b"{}")
    ar1_vectors = _vector_case(
        tmp_path / "ar1",
        inputs={"x": np.array([2.0], dtype=np.float32)},
        goldens={"y": np.array([2.0], dtype=np.float32)},
        case_id="qwen35-ar1",
    )
    ar128_vectors = _vector_case(
        tmp_path / "ar128",
        inputs={"x": np.array([2.0], dtype=np.float32)},
        goldens={"y": np.array([2.0], dtype=np.float32)},
        case_id="qwen35-ar128",
    )
    spec = BuildSpec(
        name="qwen35-runtime",
        family="qwen3_5",
        sources={
            "text": {
                "onnx_path": base,
                "encodings_path": base_enc,
            }
        },
        output_root=tmp_path / "artifacts",
        sequence={
            "ars": [1, 128],
            "context_lengths": [4096],
            "weight_sharing": True,
            "native_kv": True,
        },
        vectors={
            "mode": "provided",
            "validation_manifests_by_ar": {
                1: ar1_vectors,
                128: ar128_vectors,
            },
        },
        metadata={
            "model_config": {
                "architectures": ["Qwen3_5ForCausalLM"],
                "model_type": "qwen3_5",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 4096,
                "vocab_size": 32,
            },
            "attached_models_by_ar": {
                "1": {
                    "model_path": str(base),
                    "encodings_path": str(base_enc),
                },
                "128": {
                    "model_path": str(ar128),
                    "encodings_path": str(ar128_enc),
                },
            },
        },
    )
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)
    built = agent.build_genai_container(spec)
    assert built.ok and built.manifest is not None

    validated = agent.validate(
        built.manifest.path,
        built.manifest.sha256,
    )
    assert validated.ok, validated.error
    assert validated.data is not None
    assert validated.data["first_chain_error"] is None
    assert validated.data["coverage"]["executed_ars"] == [1, 128]
    assert set(validated.data["results_by_ar"]) == {"1", "128"}
    assert (
        validated.data["results_by_ar"]["1"]["report"]["runtime_binding"]["ar"]
        == 1
    )
    assert (
        validated.data["results_by_ar"]["128"]["report"]["runtime_binding"]["ar"]
        == 128
    )

    manifest_store = ManifestStore(validated.manifest.path.parent.parent)
    _, blocked_ref = manifest_store.fork_snapshot(validated.manifest)
    _, explicit_ar_ref = manifest_store.fork_snapshot(validated.manifest)
    blocked_optrace = agent.benchmark(
        blocked_ref.path,
        blocked_ref.sha256,
        config={
            "prompt": [{"role": "user", "content": "hello"}],
            "optrace": True,
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )
    assert not blocked_optrace.ok
    assert blocked_optrace.error is not None
    assert blocked_optrace.error.code is ErrorCode.INVALID_SPEC
    assert "multi-AR GenAI optrace is unavailable" in (
        blocked_optrace.error.message
    )

    benchmarked = agent.benchmark(
        validated.manifest.path,
        validated.manifest.sha256,
        config={
            "prompt": [{"role": "user", "content": "hello"}],
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )

    assert benchmarked.ok, benchmarked.error
    assert benchmarked.data is not None
    assert benchmarked.data["scope"] == "genai_generation"
    assert benchmarked.data["coverage"]["mode"] == (
        "executor_managed_generation"
    )
    assert benchmarked.data["coverage"]["requested_ars"] == [1, 128]
    assert benchmarked.data["coverage"]["graph_ar_coverage_proven"] is False
    assert benchmarked.data["generated_text"]["character_count"] == 2
    assert benchmarked.data["generation_metrics"]["token_generation_rate"] == 50.0
    assert factory.log.names()[-3:] == [
        "create_genai_executor",
        "generate",
        "clean_genai_executor",
    ]

    ar128_profile = agent.benchmark(
        explicit_ar_ref.path,
        explicit_ar_ref.sha256,
        config={
            "ar": 128,
            "prompt": [{"role": "user", "content": "hello"}],
            "optrace": True,
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )
    assert ar128_profile.ok, ar128_profile.error
    assert ar128_profile.data is not None
    assert ar128_profile.data["runtime_binding"]["ar"] == 128
    assert ar128_profile.data["optrace_evidence"]["logical_name"] == (
        "optrace_evidence"
    )
    assert any(
        name == "profile" and details["graph_name"] == "decoder_ar128"
        for _, name, details in factory.log.calls
    )


def _float_reference_onnx(path: Path) -> Path:
    """A two-layer float graph whose internal activations are not outputs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "one"], ["h0"], name="layer0"),
            helper.make_node("Mul", ["h0", "two"], ["h1"], name="layer1"),
            helper.make_node("Identity", ["h1"], ["y"], name="head"),
        ],
        "float-reference",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        initializer=[
            helper.make_tensor("one", TensorProto.FLOAT, [2], [1.0, 1.0]),
            helper.make_tensor("two", TensorProto.FLOAT, [2], [2.0, 2.0]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    model.ir_version = 9
    onnx.save_model(model, path)
    return path


def _float_reference_validate(
    tmp_path: Path,
    *,
    device_chain_outputs: Mapping[str, Mapping[str, Any]],
    float_reference: Mapping[str, Any] | None,
) -> Any:
    vector = _vector_case(
        tmp_path,
        inputs={"x": np.array([3.0, 4.0], dtype=np.float32)},
        goldens={"y": np.array([8.0, 10.0], dtype=np.float32)},
        case_id="float-reference",
    )
    agent = _fake_agent(FakeAdapterFactory())
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None
    config: dict[str, Any] = {
        # The production golden comparison always covers exactly the device
        # tensors under test, so a float-reference assertion is never masked by
        # an unrelated SQNR shape error.
        "references": {
            name: {tensor: np.asarray(value) for tensor, value in tensors.items()}
            for name, tensors in device_chain_outputs.items()
        },
        "device_chain_outputs": {
            name: dict(tensors) for name, tensors in device_chain_outputs.items()
        },
        "vector_manifest": vector,
    }
    if float_reference is not None:
        config["float_reference"] = dict(float_reference)
    return agent.validate(plan.manifest.path, plan.manifest.sha256, config=config)


def test_validation_without_the_debug_config_publishes_no_float_reference(
    tmp_path: Path,
) -> None:
    validated = _float_reference_validate(
        tmp_path,
        device_chain_outputs={
            "decoder": {"h1": np.array([8.0, 10.0], dtype=np.float32)}
        },
        float_reference=None,
    )

    assert validated.ok, validated.error
    assert validated.data is not None and validated.manifest is not None
    assert "float_reference" not in validated.data
    manifest = _load_run(validated.manifest)
    assert manifest.stages[-1].metrics["float_reference_debug"] is False
    assert not [
        ref
        for ref in manifest.artifacts
        if (ref.logical_name or "").startswith("float_reference")
    ]


def test_float_reference_compares_device_boundaries_with_the_float_graph(
    tmp_path: Path,
) -> None:
    model = _float_reference_onnx(tmp_path / "float" / "two_layer.onnx")

    validated = _float_reference_validate(
        tmp_path,
        device_chain_outputs={
            # h1 on the float graph is (3+1)*2, (4+1)*2 = 8, 10.
            "decoder": {"h1": np.array([8.0, 10.25], dtype=np.float32)}
        },
        float_reference={"ar": 1, "model_path": model},
    )

    assert validated.ok, validated.error
    assert validated.data is not None and validated.manifest is not None
    report = validated.data["float_reference"]
    assert report["mode"] == "debug_only"
    assert report["policy"] == "report_only"
    assert report["reference_source"] == "onnxruntime_float"
    assert report["granularity"] == "slice_boundary"
    assert report["ar"] == 1
    assert report["unmapped_tensors"] == []
    # h1 is an internal activation, so it had to be promoted to an output.
    assert report["promoted_tensors"] == ["h1"]
    assert report["reference_model_path"] == str(model)
    assert report["onnxruntime_providers"] == ["CPUExecutionProvider"]

    observation = report["observations"][0]
    assert (observation["slice"], observation["tensor"]) == ("decoder", "h1")
    assert observation["float_tensor"] == "h1"
    assert observation["quality"]["status"] == "measured"
    assert observation["quality"]["max_abs_error"] == pytest.approx(0.25)

    # The production golden comparison is untouched and still first-class.
    assert validated.data["reference_source"] == "provided"
    manifest = _load_run(validated.manifest)
    assert manifest.stages[-1].metrics["float_reference_debug"] is True
    published = json.loads(
        next(
            ref.path
            for ref in manifest.artifacts
            if ref.logical_name == "float_reference_report"
        ).read_text(encoding="utf-8")
    )
    assert published == report

    # The model on disk is never rewritten by the promotion.
    assert [item.name for item in onnx.load(str(model)).graph.output] == ["y"]


def test_float_reference_lists_unmapped_boundaries_instead_of_guessing(
    tmp_path: Path,
) -> None:
    model = _float_reference_onnx(tmp_path / "float" / "two_layer.onnx")

    validated = _float_reference_validate(
        tmp_path,
        device_chain_outputs={
            "decoder": {
                "h1": np.array([8.0, 10.0], dtype=np.float32),
                "decoder_00_out_0": np.array([1.0, 1.0], dtype=np.float32),
            }
        },
        float_reference={"ar": 1, "model_path": model},
    )

    assert validated.ok, validated.error
    assert validated.data is not None
    report = validated.data["float_reference"]
    assert [item["tensor"] for item in report["observations"]] == ["h1"]
    assert report["unmapped_tensors"] == [
        {
            "slice": "decoder",
            "tensor": "decoder_00_out_0",
            "attempted_float_tensor": "decoder_00_out_0",
            "reason": (
                "no exact name match in the float graph and no tensor_map entry"
            ),
        }
    ]


def test_float_reference_uses_an_explicit_tensor_map(tmp_path: Path) -> None:
    model = _float_reference_onnx(tmp_path / "float" / "two_layer.onnx")

    validated = _float_reference_validate(
        tmp_path,
        device_chain_outputs={
            "decoder": {"decoder_00_out_0": np.array([4.0, 5.0], dtype=np.float32)}
        },
        float_reference={
            "ar": 1,
            "model_path": model,
            "tensor_map": {"decoder": {"decoder_00_out_0": "h0"}},
        },
    )

    assert validated.ok, validated.error
    assert validated.data is not None
    report = validated.data["float_reference"]
    observation = report["observations"][0]
    assert observation["float_tensor"] == "h0"
    # h0 is exactly 3+1, 4+1 on the float graph.
    assert observation["quality"]["status"] == "exact"
    assert report["unmapped_tensors"] == []


def test_float_reference_fails_closed_on_debug_misuse(tmp_path: Path) -> None:
    model = _float_reference_onnx(tmp_path / "float" / "two_layer.onnx")
    outputs = {"decoder": {"h1": np.array([8.0, 10.0], dtype=np.float32)}}

    missing_ar = _float_reference_validate(
        tmp_path / "a",
        device_chain_outputs=outputs,
        float_reference={"model_path": model},
    )
    assert not missing_ar.ok and missing_ar.error is not None
    assert missing_ar.error.code is ErrorCode.INVALID_SPEC
    assert "single-AR debug mode" in missing_ar.error.message

    layerwise = _float_reference_validate(
        tmp_path / "b",
        device_chain_outputs=outputs,
        float_reference={"ar": 1, "model_path": model, "granularity": "layer"},
    )
    assert not layerwise.ok and layerwise.error is not None
    assert layerwise.error.code is ErrorCode.INVALID_SPEC
    assert "slice_boundary" in layerwise.error.message

    nothing_mappable = _float_reference_validate(
        tmp_path / "c",
        device_chain_outputs={
            "decoder": {"unknown_tensor": np.array([1.0, 1.0], dtype=np.float32)}
        },
        float_reference={"ar": 1, "model_path": model},
    )
    assert not nothing_mappable.ok and nothing_mappable.error is not None
    assert nothing_mappable.error.code is ErrorCode.INVALID_SPEC
    assert "never guessed" in nothing_mappable.error.message


def _genai_spec(tmp_path: Path, **overrides: Any) -> BuildSpec:
    model = _write_onnx(tmp_path / "qwen35" / "ar1.onnx")
    encodings = _write(tmp_path / "qwen35" / "ar1.encodings", b"{}")
    return BuildSpec(
        name="genai-benchmark-defaults",
        family="qwen3_5",
        sources={"text": {"onnx_path": model, "encodings_path": encodings}},
        output_root=tmp_path / "artifacts",
        sequence={
            "ars": [1],
            "context_lengths": [4096],
            "weight_sharing": False,
            "native_kv": False,
        },
        metadata={
            "model_config": {
                "architectures": ["Qwen3_5ForCausalLM"],
                "model_type": "qwen3_5",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 4096,
                "vocab_size": 32,
            },
            "attached_models_by_ar": {
                "1": {"model_path": str(model), "encodings_path": str(encodings)}
            },
        },
        **overrides,
    )


def test_genai_lane_resolves_cheaper_benchmark_defaults(tmp_path: Path) -> None:
    agent = _fake_agent(FakeAdapterFactory())

    planned = agent.plan(_genai_spec(tmp_path), offline=True)

    assert planned.ok, planned.error
    assert planned.data is not None and planned.manifest is not None
    benchmark = planned.data["effective_config"]["benchmark"]
    assert benchmark["warmup_runs"] == 3
    assert benchmark["measured_runs"] == 10
    assert benchmark["lane"] == "genai_builder"
    assert benchmark["sample_unit"] == "generate_call"
    assert benchmark["aa_calibration_doubles_runs"] is True

    # The resolved policy is materialized into the recorded BuildSpec, so every
    # later stage reads the numbers plan showed.
    recorded = _load_run(planned.manifest).build_spec.benchmark
    assert (recorded.warmup_runs, recorded.measured_runs) == (3, 10)

    built = agent.build_genai_container(_genai_spec(tmp_path))
    assert built.ok, built.error
    assert built.manifest is not None
    benchmarked = agent.benchmark(
        built.manifest.path,
        built.manifest.sha256,
        config={
            "prompt": [{"role": "user", "content": "hello"}],
            "aa_calibration": False,
        },
    )
    assert benchmarked.ok, benchmarked.error
    assert benchmarked.data is not None
    measurement = benchmarked.data["measurement"]
    assert measurement["warmup_count"] == 3
    assert measurement["repeat_count"] == 10
    assert benchmarked.data["measurement_scope"]["sample_unit"] == "generate_call"


def test_explicit_benchmark_values_beat_the_genai_lane_defaults(
    tmp_path: Path,
) -> None:
    agent = _fake_agent(FakeAdapterFactory())

    planned = agent.plan(
        _genai_spec(tmp_path, benchmark={"warmup_runs": 1}),
        offline=True,
    )

    assert planned.ok, planned.error
    assert planned.data is not None
    benchmark = planned.data["effective_config"]["benchmark"]
    # The caller set warmup only; the unset field still takes the lane default.
    assert benchmark["warmup_runs"] == 1
    assert benchmark["measured_runs"] == 10


def test_low_level_lane_keeps_the_original_benchmark_defaults(
    tmp_path: Path,
) -> None:
    agent = _fake_agent(FakeAdapterFactory())
    spec = _make_spec(tmp_path).model_copy(update={"benchmark": BenchmarkSpec()})

    planned = agent.plan(spec, offline=True)

    assert planned.ok, planned.error
    assert planned.data is not None
    benchmark = planned.data["effective_config"]["benchmark"]
    assert benchmark["warmup_runs"] == 10
    assert benchmark["measured_runs"] == 50
    assert benchmark["lane"] == "low_level"
    assert benchmark["sample_unit"] == "graph_invocation"


def test_latency_report_labels_the_token_metric_source_and_wall_scope(
    tmp_path: Path,
) -> None:
    vector = _vector_case(tmp_path)
    agent = _fake_agent(FakeAdapterFactory())
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None

    benchmarked = agent.benchmark(
        plan.manifest.path,
        plan.manifest.sha256,
        config={
            "context_path": tmp_path / "main.bin",
            "graph_name": "main_ar1",
            "vector_manifest": vector,
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
            "token_count": 4,
        },
    )

    assert benchmarked.ok, benchmarked.error
    assert benchmarked.data is not None
    assert benchmarked.data["token_count"] == 4
    assert benchmarked.data["ms_per_token_source"] == "caller"
    scope = benchmarked.data["measurement_scope"]
    assert scope["clock"] == "host_perf_counter_ns"
    assert scope["device_side_sync_barrier"] is False
    assert scope["sample_unit"] == "graph_invocation"


def _benchmark_once(tmp_path: Path, factory: FakeAdapterFactory) -> Any:
    vector = _vector_case(tmp_path)
    agent = _fake_agent(factory)
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None
    return agent.benchmark(
        plan.manifest.path,
        plan.manifest.sha256,
        config={
            "context_path": tmp_path / "main.bin",
            "graph_name": "main_ar1",
            "vector_manifest": vector,
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )


def test_latency_report_publishes_device_side_execution_beside_wall_time(
    tmp_path: Path,
) -> None:
    # The wall sample is host time around one SDK call; on the low-level lane
    # QAIRT relaunches qnn-net-run per call, so it is thousands of times the
    # device's own execute time.  The report must carry both.
    benchmarked = _benchmark_once(tmp_path, FakeAdapterFactory())

    assert benchmarked.ok, benchmarked.error
    assert benchmarked.data is not None
    device = benchmarked.data["device_execution"]

    assert device["accelerator_compute_us"] == 77.0
    assert device["qnn_execute_us"] == 3001.0
    assert device["source"] == "qairt_profiler_detailed_log"
    assert device["claim_scope"] == "one_profiled_execute_not_the_timed_samples"
    assert {item["identifier"] for item in device["per_op_cycles"]} == {
        "fc:OpId_17",
        "act:OpId_23",
    }
    # Per-process cost is reported, never folded into execute time.
    assert device["per_process_overhead_us"]["QNN (load binary) time"] == 9009.0


def test_latency_report_no_longer_claims_per_call_setup_is_excluded(
    tmp_path: Path,
) -> None:
    # The old report published setup_excluded=true, which was false: per-call
    # context load, HVX/HMX power-on and deinit happen inside the timed call.
    benchmarked = _benchmark_once(tmp_path, FakeAdapterFactory())

    assert benchmarked.data is not None
    assert "setup_excluded" not in benchmarked.data
    assert benchmarked.data["harness_setup_excluded"] is True
    assert benchmarked.data["sdk_per_call_setup_included"] is True
    assert benchmarked.data["metric_name"] == "host_orchestrated_call_latency"

    scope = benchmarked.data["measurement_scope"]
    assert "adb_staging" in scope["excluded_from_timer"]
    assert "qnn_net_run_process_launch" in scope["included_in_sample"]
    assert "hvx_hmx_power_on_and_acquire" in scope["included_in_sample"]


def test_benchmark_captures_device_metrics_before_initializing_and_releases_after(
    tmp_path: Path,
) -> None:
    # Ordering is load-bearing twice over: profiling must happen before
    # initialize() (an initialized model carries a profiling-disabled context),
    # and the persistent context must be released even though the timed loop
    # sits between them.
    factory = FakeAdapterFactory()
    benchmarked = _benchmark_once(tmp_path, factory)
    assert benchmarked.ok, benchmarked.error

    names = factory.log.names()
    capture = names.index("capture_device_execution")
    initialize = names.index("initialize_execution")
    release = names.index("release_execution")
    first_run = names.index("run_graph")

    assert capture < initialize < first_run < release
    # Nothing is left holding backend artifacts on the device.
    assert all(not adapter.initialized_contexts for adapter in factory.adapters)


def test_benchmark_survives_an_adapter_that_cannot_profile(tmp_path: Path) -> None:
    # Device evidence is report-only enrichment: an adapter without it still
    # produces a valid benchmark, and the report says why rather than claiming
    # a device number it never measured.
    class NoProfilingFactory(FakeAdapterFactory):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            adapter = super().__call__(*args, **kwargs)
            adapter.capture_device_execution = None  # type: ignore[assignment]
            return adapter

    benchmarked = _benchmark_once(tmp_path, NoProfilingFactory())

    assert benchmarked.ok, benchmarked.error
    assert benchmarked.data is not None
    assert "device_execution" not in benchmarked.data
    # The honest scope statement does not depend on the capture succeeding.
    assert benchmarked.data["sdk_per_call_setup_included"] is True


def test_sdk_generated_token_count_never_derives_from_a_rate() -> None:
    # QAIRT 2.49 reports a rate and a duration but no count; deriving one would
    # manufacture a number the SDK never reported.
    assert (
        _sdk_generated_token_count(
            {"token_generation_rate": 50.0, "token_generation_time": 200000}
        )
        is None
    )
    assert _sdk_generated_token_count({"num_generated_tokens": 12}) == 12
    assert _sdk_generated_token_count({"num-generated-tokens": {"value": 7}}) == 7
    assert _sdk_generated_token_count({"num_generated_tokens": 0}) is None
    assert _sdk_generated_token_count({"num_generated_tokens": 3.5}) is None
    assert _sdk_generated_token_count(None) is None


def _footprint_bytes_on_disk(footprint: Mapping[str, Any]) -> int:
    return sum(Path(item["path"]).stat().st_size for item in footprint["artifacts"])


def test_build_publishes_a_static_footprint_measured_from_disk(
    tmp_path: Path,
) -> None:
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)

    built = agent.build(_make_spec(tmp_path))

    assert built.ok, built.error
    assert built.data is not None and built.manifest is not None
    footprint = built.data["static_footprint"]
    assert footprint["policy"] == "report_only"
    assert footprint["unit"] == "bytes"

    for item in footprint["artifacts"]:
        assert item["bytes"] == Path(item["path"]).stat().st_size
    assert _footprint_bytes_on_disk(footprint) == sum(
        item["bytes"] for item in footprint["artifacts"]
    )

    contexts = [
        item for item in footprint["artifacts"] if item["role"] == "context"
    ]
    assert contexts and all(item["slice_name"] == "decoder_00" for item in contexts)
    assert footprint["contexts_total_bytes"] == sum(
        item["bytes"] for item in contexts
    )
    # Converted DLCs are reported but are build intermediates, so the headline
    # total covers only what the device has to hold.
    assert footprint["converted_models_total_bytes"] > 0
    assert footprint["total_includes"] == ["context"]
    assert footprint["total_bytes"] == footprint["contexts_total_bytes"]

    # Absent measurements are absent fields, never zeros.
    assert "genai_container_total_bytes" not in footprint
    assert "diagnostic" not in footprint

    stage_metrics = _load_run(built.manifest).stages[-1].metrics
    assert stage_metrics["static_footprint"] == footprint


def test_static_footprint_keeps_diagnostic_contexts_out_of_the_totals(
    tmp_path: Path,
) -> None:
    vectors = _vector_case(
        tmp_path,
        inputs={"x": np.array([2.0], dtype=np.float32)},
        goldens={"y": np.array([2.0], dtype=np.float32)},
        case_id="footprint-diagnostics",
    )
    spec = _make_spec(
        tmp_path,
        vectors={"mode": "provided", "validation_manifest": vectors},
        sequence={
            "ars": [1],
            "context_lengths": [4096],
            "weight_sharing": False,
            "native_kv": False,
        },
        quality={
            "sqnr_modes": ["full_reference"],
            "dump_intermediates_on_failure": True,
        },
    )
    agent = _fake_agent(FakeQualityModesAdapterFactory())

    built = agent.build(spec)

    assert built.ok, built.error
    assert built.data is not None
    footprint = built.data["static_footprint"]
    diagnostic = footprint["diagnostic"]
    assert diagnostic["counted_in_totals"] is False
    assert len(diagnostic["artifacts"]) == 2
    assert diagnostic["total_bytes"] == sum(
        Path(item["path"]).stat().st_size for item in diagnostic["artifacts"]
    )
    assert diagnostic["total_bytes"] > 0

    production_paths = {item["path"] for item in footprint["artifacts"]}
    assert not production_paths & {
        item["path"] for item in diagnostic["artifacts"]
    }
    assert footprint["total_bytes"] == footprint["contexts_total_bytes"]


def test_genai_static_footprint_measures_the_saved_container(
    tmp_path: Path,
) -> None:
    model = _write_onnx(tmp_path / "qwen3" / "model.onnx")
    encodings = _write(tmp_path / "qwen3" / "model.encodings", b"{}")
    spec = BuildSpec(
        name="genai-footprint",
        family="qwen3_moe",
        sources={"text": {"onnx_path": model, "encodings_path": encodings}},
        output_root=tmp_path / "artifacts",
        sequence={"ars": [1, 128], "context_lengths": [4096]},
        metadata={
            "model_config": {
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3_moe",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 4096,
                "vocab_size": 32,
            }
        },
    )
    agent = _fake_agent(FakeAdapterFactory())

    built = agent.build_genai_container(spec)

    assert built.ok, built.error
    assert built.data is not None
    footprint = built.data["static_footprint"]
    container_root = Path(built.data["genai_container"]["container_path"])
    expected = sum(
        path.stat().st_size for path in container_root.rglob("*") if path.is_file()
    )
    assert footprint["genai_container_total_bytes"] == expected
    assert footprint["total_includes"] == ["genai_container"]
    assert footprint["total_bytes"] == expected
    # A GenAI build has no low-level contexts at all: the field is absent, not 0.
    assert "contexts_total_bytes" not in footprint
    # The raw slice lives inside the container and must not be counted twice.
    assert sum(item["bytes"] for item in footprint["artifacts"]) == expected


def test_benchmark_report_carries_the_build_footprint_verbatim(
    tmp_path: Path,
) -> None:
    vectors = _vector_case(tmp_path)
    spec = _make_spec(
        tmp_path,
        vectors={"mode": "provided", "validation_manifest": vectors},
        sequence={"ars": [1], "context_lengths": [4096], "weight_sharing": False},
    )
    agent = _fake_agent(FakeAdapterFactory())

    built = agent.build(spec)
    assert built.ok and built.manifest is not None and built.data is not None
    validated = agent.validate(built.manifest.path, built.manifest.sha256)
    assert validated.ok, validated.error
    assert validated.manifest is not None

    benchmarked = agent.benchmark(
        validated.manifest.path,
        validated.manifest.sha256,
        config={"warmup_runs": 0, "measured_runs": 1, "aa_calibration": False},
    )

    assert benchmarked.ok, benchmarked.error
    assert benchmarked.data is not None and benchmarked.manifest is not None
    reported = benchmarked.data["static_footprint"]
    assert reported["source"] == "build_receipt"
    assert reported["measured_by_stage"] == "build"
    build_footprint = built.data["static_footprint"]
    assert {
        key: value
        for key, value in reported.items()
        if key not in {"source", "measured_by_stage", "measured_by_attempt"}
    } == build_footprint

    published = json.loads(
        next(
            ref.path
            for ref in _load_run(benchmarked.manifest).artifacts
            if ref.logical_name == "latency_report"
        ).read_text(encoding="utf-8")
    )
    assert published["static_footprint"] == reported


def test_genai_benchmark_rejects_low_level_chain_keys(
    tmp_path: Path,
) -> None:
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None

    conflicted = agent.benchmark(
        plan.manifest.path,
        plan.manifest.sha256,
        config={
            "container_path": tmp_path / "container",
            "prompt": [{"role": "user", "content": "hello"}],
            "steps": [{"slice": "decoder", "inputs": {}}],
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )

    assert not conflicted.ok
    assert conflicted.error is not None
    assert conflicted.error.code is ErrorCode.INVALID_SPEC
    assert "low-level slice-chain configuration" in conflicted.error.message
    assert conflicted.error.details["conflicting_keys"] == ["steps"]


def test_standalone_vit_uses_only_low_level_convert_compile_lane(
    tmp_path: Path,
) -> None:
    model = _write_onnx(tmp_path / "vit" / "vit.onnx")
    encodings = _write(tmp_path / "vit" / "vit.encodings", b"{}")
    spec = BuildSpec(
        family="vit",
        sources={
            "text": {
                "onnx_path": model,
                "encodings_path": encodings,
            }
        },
        output_root=tmp_path / "artifacts",
        sequence={"ars": [1], "weight_sharing": False, "native_kv": False},
        split={"decoder_slice_count": 1, "split_lm_head": False},
        transforms={"mha2sha": False},
    )
    factory = FakeAdapterFactory()

    result = QairtAgent(adapter_factory=factory).build(spec)

    assert result.ok and result.data is not None and result.manifest is not None
    assert factory.log.names() == ["preflight", "build_standalone_vit"]
    assert result.data["lane"] == "standalone_vit_low_level"
    manifest = _load_run(result.manifest)
    assert {"vit.dlc", "vit.bin", "standalone_vit_build.json"} <= {
        artifact.path.name for artifact in manifest.artifacts
    }


def test_standalone_vit_runtime_index_auto_validates_and_benchmarks_one_graph(
    tmp_path: Path,
) -> None:
    model = _write_onnx(tmp_path / "vit" / "vit.onnx")
    encodings = _write(tmp_path / "vit" / "vit.encodings", b"{}")
    vectors = _vector_case(
        tmp_path / "vit",
        inputs={"x": np.array([2.0], dtype=np.float32)},
        goldens={"y": np.array([4.0], dtype=np.float32)},
        case_id="vit",
    )
    spec = BuildSpec(
        family="vit",
        sources={
            "text": {
                "onnx_path": model,
                "encodings_path": encodings,
            }
        },
        output_root=tmp_path / "artifacts",
        sequence={
            "ars": [1],
            "weight_sharing": False,
            "native_kv": False,
        },
        split={"decoder_slice_count": 1, "split_lm_head": False},
        transforms={"mha2sha": False},
        vectors={
            "mode": "provided",
            "validation_manifest": vectors,
        },
        benchmark={"warmup_runs": 0, "measured_runs": 1},
    )
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)

    built = agent.build(spec)
    assert built.ok and built.manifest is not None
    validated = agent.validate(
        built.manifest.path,
        built.manifest.sha256,
    )
    assert validated.ok, validated.error
    assert validated.manifest is not None
    assert validated.data is not None
    assert validated.data["runtime_binding"]["scope"] == "graph"
    assert validated.data["runtime_binding"]["graph_name"] == "vit"

    benchmarked = agent.benchmark(
        validated.manifest.path,
        validated.manifest.sha256,
        config={
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )
    assert benchmarked.ok, benchmarked.error
    assert benchmarked.data is not None
    assert benchmarked.data["scope"] == "graph"
    assert benchmarked.data["runtime_binding"]["graph_name"] == "vit"


def test_qwen35_omni_packages_audio_and_text_but_runtime_is_unsupported(
    tmp_path: Path,
) -> None:
    text = _write_onnx(tmp_path / "omni" / "text.onnx")
    audio = _write_onnx(tmp_path / "omni" / "audio.onnx")
    text_enc = _write(tmp_path / "omni" / "text.encodings", b"{}")
    audio_enc = _write(tmp_path / "omni" / "audio.encodings", b"{}")
    ar1 = _write_onnx(tmp_path / "omni" / "text-ar1.onnx")
    ar128 = _write_onnx(tmp_path / "omni" / "text-ar128.onnx")
    ar1_enc = _write(tmp_path / "omni" / "text-ar1.encodings", b"{}")
    ar128_enc = _write(tmp_path / "omni" / "text-ar128.encodings", b"{}")
    spec = BuildSpec(
        family="qwen3_5_omni",
        sources={
            "text": {"onnx_path": text, "encodings_path": text_enc},
            "audio": {"onnx_path": audio, "encodings_path": audio_enc},
        },
        output_root=tmp_path / "artifacts",
        metadata={
            "model_config": {
                "architectures": ["Qwen3OmniForConditionalGeneration"],
                "model_type": "qwen3_omni",
                "audio_start_token_id": 1,
                "audio_end_token_id": 2,
                "text_config": {
                    "hidden_size": 16,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "max_position_embeddings": 4096,
                    "vocab_size": 32,
                },
            },
            "attached_models_by_ar": {
                "1": {
                    "model_path": str(ar1),
                    "encodings_path": str(ar1_enc),
                },
                "128": {
                    "model_path": str(ar128),
                    "encodings_path": str(ar128_enc),
                },
            },
        },
    )
    factory = FakeAdapterFactory()

    result = QairtAgent(adapter_factory=factory).build_genai_container(spec)

    assert result.ok and result.data is not None and result.manifest is not None
    assert factory.log.names() == [
        "preflight",
        "build_qwen35_omni_components",
    ]
    assert result.data["lane"] == "qwen3_5_omni_component_packaging"
    packaged = result.data["genai_container"]
    assert packaged["audio_builder_class"] == "Qwen3OmniAudioEncoderBuilderHTP"
    assert packaged["builder_class"] == "Qwen3_5BuilderHTP"
    assert packaged["runtime_supported"] is False
    manifest = _load_run(result.manifest)
    assert {"audio.bin", "decoder.bin", "qairt_agent_genai_build.json"} <= {
        artifact.path.name for artifact in manifest.artifacts
    }
    assert {
        "text-ar1.onnx",
        "text-ar1.encodings",
        "text-ar128.onnx",
        "text-ar128.encodings",
    } <= {artifact.path.name for artifact in manifest.artifacts}

    # Omni nests the audio/text component directories inside the container
    # directory; each file must be counted exactly once.
    footprint = result.data["static_footprint"]
    container_root = Path(packaged["container_path"])
    counted = [item["path"] for item in footprint["artifacts"]]
    assert len(counted) == len(set(counted))
    assert footprint["genai_container_total_bytes"] == sum(
        path.stat().st_size for path in container_root.rglob("*") if path.is_file()
    )


def test_calibration_build_preflights_before_creating_calibration_config(
    tmp_path: Path,
) -> None:
    calibration = _vector_case(
        tmp_path,
        inputs={"x": np.array([1.0], dtype=np.float32)},
        goldens={},
        case_id="calibration",
    )
    spec = _make_spec(
        tmp_path,
        quantization_mode=QuantizationMode.CALIBRATE.value,
        vectors={"mode": "provided", "calibration_manifest": calibration},
    )
    factory = FakeAdapterFactory()

    result = _fake_agent(factory).build(spec)

    assert result.ok, result.error
    assert factory.log.names()[:3] == [
        "preflight",
        "create_calibration_config",
        "build",
    ]
    build_call = next(details for _, name, details in factory.log.calls if name == "build")
    assert build_call["runtime"]["quantization"]["calibration_config"]["kind"] == (
        "fake-calibration"
    )


def test_build_runtime_preserves_mha2sha_and_kv_permute_flags(
    tmp_path: Path,
) -> None:
    spec = _make_spec(
        tmp_path,
        transforms={"mha2sha": False, "permute_kv_cache_io": True},
        sequence={"native_kv": False},
    )
    factory = FakeAdapterFactory()

    result = _fake_agent(factory).build(spec)

    assert result.ok, result.error
    build_call = next(details for _, name, details in factory.log.calls if name == "build")
    assert build_call["runtime"]["transforms"]["mha2sha"] is False
    assert build_call["runtime"]["transforms"]["permute_kv_cache_io"] is True
    assert result.data is not None
    assert result.data["effective_config"]["transforms"]["mha2sha"] is False
    assert (
        result.data["effective_config"]["transforms"]["permute_kv_cache_io"]
        is True
    )


def test_prepare_vectors_run_graph_data_result_and_run_chain_are_stateless(
    tmp_path: Path,
) -> None:
    factory = FakeAdapterFactory(result_style="data")
    agent = _fake_agent(factory)
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None

    prepared = agent.prepare_vectors(
        plan.manifest.path,
        plan.manifest.sha256,
        config={
            "cases": [
                {
                    "case_id": "graph-case",
                    "inputs": {"x": np.array([2.0], dtype=np.float32)},
                    "goldens": {"y": np.array([4.0], dtype=np.float32)},
                }
            ]
        },
    )
    assert prepared.ok and prepared.manifest is not None and prepared.data is not None
    vector_ref = prepared.data["vector_manifests"][0]
    vector_path = Path(vector_ref["path"])
    np.testing.assert_array_equal(
        VectorPreparer.load_tensors(vector_path, section="goldens")["y"],
        np.array([4.0], dtype=np.float32),
    )

    graph = agent.run_graph(
        prepared.manifest.path,
        prepared.manifest.sha256,
        config={
            "context_path": tmp_path / "main.bin",
            "graph_name": "main_ar1",
            "vector_manifest": vector_path,
            "vector_manifest_sha256": vector_ref["sha256"],
        },
    )
    assert graph.ok and graph.manifest is not None and graph.data is not None
    graph_output = Path(graph.data["output_manifest"]["path"])
    np.testing.assert_array_equal(
        VectorPreparer.load_tensors(graph_output, section="goldens")["y"],
        np.array([4.0], dtype=np.float32),
    )

    routes = [
        {
            "slice_id": "embedding",
            "input_names": ["x"],
            "output_names": ["h"],
            "graph_names": {"1": "embedding_ar1"},
        },
        {
            "slice_id": "decoder",
            "input_names": ["hidden"],
            "output_names": ["y"],
            "graph_names": {"1": "decoder_ar1"},
            "from_previous": {"hidden": "h"},
        },
    ]
    chain = agent.run_chain(
        graph.manifest.path,
        graph.manifest.sha256,
        config={
            "routes": routes,
            "contexts": {
                "embedding": tmp_path / "embedding.bin",
                "decoder": tmp_path / "decoder.bin",
            },
            "vector_manifest": vector_path,
            "ar": 1,
        },
    )

    assert chain.ok and chain.manifest is not None and chain.data is not None
    decoder_output = Path(chain.data["output_manifests"]["decoder"]["path"])
    np.testing.assert_array_equal(
        VectorPreparer.load_tensors(decoder_output, section="inputs")["y"],
        np.array([9.0], dtype=np.float32),
    )
    graph_calls = [
        (adapter_id, details["graph_name"])
        for adapter_id, name, details in factory.log.calls
        if name == "run_graph"
    ]
    assert [name for _, name in graph_calls] == [
        "main_ar1",
        "embedding_ar1",
        "decoder_ar1",
    ]
    assert len({adapter_id for adapter_id, _ in graph_calls}) == 2
    assert all(
        details["options"].get("device") is not None
        for _, name, details in factory.log.calls
        if name == "run_graph"
    )
    stages = _load_run(chain.manifest).stages
    assert [stage.name for stage in stages] == [
        "plan",
        "prepare_vectors",
        "run_graph",
        "run_chain",
    ]
    assert all(stage.status is StageStatus.SUCCEEDED for stage in stages)


def test_run_graph_fails_closed_without_explicit_device_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("QAIRT_AGENT_ADB_SERIAL", raising=False)
    monkeypatch.delenv("QAIRT_AGENT_ADB_SERVER", raising=False)
    vector = _vector_case(tmp_path)
    factory = FakeAdapterFactory()
    agent = QairtAgent(adapter_factory=factory)
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None

    result = agent.run_graph(
        plan.manifest.path,
        plan.manifest.sha256,
        config={
            "context_path": tmp_path / "main.bin",
            "graph_name": "main_ar1",
            "vector_manifest": vector,
        },
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.DEVICE_UNAVAILABLE
    assert "run_graph" not in factory.log.names()


def test_validate_executes_with_injected_device_and_reports_sqnr(
    tmp_path: Path,
) -> None:
    vector = _vector_case(tmp_path)
    factory = FakeAdapterFactory()
    runtime = FakeDeviceRuntime()
    agent = QairtAgent(
        adapter_factory=factory,
        device_runtime=runtime,
    )
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None

    result = agent.validate(
        plan.manifest.path,
        plan.manifest.sha256,
        config={
            "context_path": tmp_path / "main.bin",
            "graph_name": "main_ar1",
            "vector_manifest": vector,
        },
    )

    assert result.ok, result.error
    assert result.data is not None
    assert result.data["first_chain_error"] is None
    call = next(
        details
        for _, name, details in factory.log.calls
        if name == "run_graph"
    )
    assert call["options"]["device"] is runtime.device


def test_qwen35_runtime_evidence_uses_same_staged_device(
    tmp_path: Path,
) -> None:
    vector = _vector_case(tmp_path)
    factory = FakeAdapterFactory()
    runtime = FakeDeviceRuntime()
    agent = QairtAgent(
        adapter_factory=factory,
        device_runtime=runtime,
    )
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None
    manifest = _load_run(plan.manifest)
    adapter = factory()
    adapter.preflight(manifest.build_spec)
    validator, _refs = agent._qwen35_validator(  # noqa: SLF001
        adapter,
        {"cases": {"main_ar1": vector}},
        tmp_path / "qwen35-report",
        manifest=manifest,
        input_manifest_sha256=plan.manifest.sha256,
    )

    result = validator(
        SimpleNamespace(
            slice_name="decoder_00",
            ar_values=(1,),
            graph_names=("main_ar1",),
            standalone_contexts=(tmp_path / "standalone.bin",),
            joint_context=tmp_path / "joint.bin",
        )
    )

    assert result.standalone_vs_golden_passed is True
    assert result.joint_vs_golden_passed is True
    calls = [
        details
        for _, name, details in factory.log.calls
        if name == "run_graph"
    ]
    assert len(calls) == 2
    assert all(call["options"]["device"] is runtime.device for call in calls)
    assert len(runtime.calls) == 1


def test_validate_accepts_run_graph_output_manifest_without_section_override(
    tmp_path: Path,
) -> None:
    vector = _vector_case(tmp_path)
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None
    graph = agent.run_graph(
        plan.manifest.path,
        plan.manifest.sha256,
        config={
            "context_path": tmp_path / "main.bin",
            "graph_name": "main_ar1",
            "vector_manifest": vector,
        },
    )
    assert graph.ok and graph.manifest is not None and graph.data is not None

    validated = agent.validate(
        graph.manifest.path,
        graph.manifest.sha256,
        config={
            "vector_manifest": vector,
            "actual_manifest": graph.data["output_manifest"]["path"],
        },
    )

    assert validated.ok, validated.error
    assert validated.data is not None
    assert validated.data["first_chain_error"] is None
    assert validated.data["observations"][0]["device_chain"]["status"] == "exact"


def test_validate_records_device_evidence_only_for_a_device_stage(
    tmp_path: Path,
) -> None:
    vector = _vector_case(tmp_path)
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None

    offline = agent.validate(
        plan.manifest.path,
        plan.manifest.sha256,
        config={
            "references": {
                "decoder": {"tap": np.array([1.0, 2.0], dtype=np.float32)}
            },
            "device_chain_outputs": {
                "decoder": {"tap": np.array([1.0, 2.0], dtype=np.float32)}
            },
        },
    )
    assert offline.ok and offline.manifest is not None
    offline_metrics = _load_run(offline.manifest).stages[-1].metrics
    assert "device_identifier" not in offline_metrics
    assert "remote_cleanup" not in offline_metrics

    on_device = agent.validate(
        offline.manifest.path,
        offline.manifest.sha256,
        config={
            "context_path": tmp_path / "main.bin",
            "graph_name": "main_ar1",
            "vector_manifest": vector,
        },
    )
    assert on_device.ok, on_device.error
    assert on_device.manifest is not None
    device_metrics = _load_run(on_device.manifest).stages[-1].metrics
    assert device_metrics["device_identifier"] == "TEST@localhost:5037"
    assert device_metrics["remote_cleanup"] == "confirmed"


def test_validate_benchmark_and_diagnoses_publish_report_only_artifacts(
    tmp_path: Path,
) -> None:
    vector = _vector_case(tmp_path)
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None

    validated = agent.validate(
        plan.manifest.path,
        plan.manifest.sha256,
        config={
            "references": {
                "decoder": {"tap": np.array([1.0, 2.0], dtype=np.float32)}
            },
            "teacher_forced_outputs": {
                "decoder": {"tap": np.array([1.0, 2.0], dtype=np.float32)}
            },
            "device_chain_outputs": {
                "decoder": {"tap": np.array([1.0, 2.25], dtype=np.float32)}
            },
        },
    )
    assert validated.ok and validated.manifest is not None and validated.data is not None
    assert validated.data["first_teacher_error"] is None
    assert validated.data["first_chain_error"] == ["decoder", "tap"]

    benchmark = agent.benchmark(
        validated.manifest.path,
        validated.manifest.sha256,
        config={
            "context_path": tmp_path / "main.bin",
            "graph_name": "main_ar1",
            "vector_manifest": vector,
            "warmup_runs": 1,
            "measured_runs": 2,
            "aa_calibration": False,
            "token_count": 2,
        },
    )
    assert benchmark.ok and benchmark.manifest is not None and benchmark.data is not None
    assert benchmark.data["scope"] == "graph"
    assert benchmark.data["measurement"]["summary"]["count"] == 2
    assert benchmark.data["policy"] == "report_only"
    benchmark_graph_calls = [
        details
        for _, name, details in factory.log.calls
        if name == "run_graph" and details["graph_name"] == "main_ar1"
    ]
    assert len(benchmark_graph_calls) == 3

    quality = agent.diagnose_quality(
        benchmark.manifest.path,
        benchmark.manifest.sha256,
        config={
            "reference_trace": {
                "layer0/MatMul": np.array([1.0], dtype=np.float32),
                "layer1/Add": np.array([2.0], dtype=np.float32),
            },
            "actual_trace": {
                "layer0/MatMul": np.array([1.0], dtype=np.float32),
                "layer1/Add": np.array([2.2], dtype=np.float32),
            },
            "order": ["layer0/MatMul", "layer1/Add"],
            "lineage": {"layer1/Add": {"op_type": "Add", "layer": 1}},
        },
    )
    assert quality.ok and quality.manifest is not None and quality.data is not None
    assert quality.data["first_observed_error"] == "layer1/Add"
    assert quality.data["claim_scope"] == "first_observed_divergence_not_root_cause"

    latency = agent.diagnose_latency(
        quality.manifest.path,
        quality.manifest.sha256,
        config={
            "baseline_ops": {
                "MatMul_0": {
                    "cycles": 100,
                    "critical_path": True,
                    "lineage": {"layer": 0},
                }
            },
            "candidate_ops": {
                "MatMul_0": {
                    "cycles": 140,
                    "critical_path": True,
                    "lineage": {"op_type": "MatMul"},
                }
            },
        },
    )
    assert latency.ok and latency.manifest is not None and latency.data is not None
    assert latency.data["attributions"][0]["op_id"] == "MatMul_0"
    assert latency.data["attributions"][0]["delta_cycles"] == 40.0
    assert latency.data["claim_scope"] == "op_work_not_additive_wall_latency"
    for artifact in _load_run(latency.manifest).artifacts:
        verify_artifact(artifact)


def test_chain_benchmark_passes_initial_native_state_through_prefill_decode(
    tmp_path: Path,
) -> None:
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None
    route = {
        "slice_id": "decoder",
        "input_names": ["token", "kv_in"],
        "output_names": ["logits", "kv_out"],
        "graph_names": {"1": "state_ar1", "128": "state_ar128"},
        "state_inputs": {"kv_in": "decoder.kv"},
        "state_outputs": {"kv_out": "decoder.kv"},
    }

    result = agent.benchmark(
        plan.manifest.path,
        plan.manifest.sha256,
        config={
            "routes": [route],
            "contexts": {"decoder": tmp_path / "decoder.bin"},
            "steps": [
                {"inputs": {"token": np.array([2.0])}, "ar": 128},
                {"inputs": {"token": np.array([3.0])}, "ar": 1},
            ],
            "initial_native_state": {"decoder.kv": np.array([0.0])},
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )

    assert result.ok, result.error
    assert result.data is not None and result.data["scope"] == "chain_sequence"
    calls = [
        details
        for _, name, details in factory.log.calls
        if name == "run_graph"
    ]
    assert [item["graph_name"] for item in calls] == ["state_ar128", "state_ar1"]
    np.testing.assert_array_equal(calls[0]["inputs"]["kv_in"], np.array([0.0]))
    np.testing.assert_array_equal(calls[1]["inputs"]["kv_in"], np.array([2.0]))


def test_failed_stage_revision_keeps_stage_key_and_can_be_retried(
    tmp_path: Path,
) -> None:
    vector = _vector_case(tmp_path)
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None

    failed = agent.run_graph(
        plan.manifest.path,
        plan.manifest.sha256,
        config={},
    )
    assert not failed.ok and failed.manifest is not None
    assert failed.error is not None and failed.error.code is ErrorCode.INVALID_SPEC
    failed_manifest = _load_run(failed.manifest)
    assert failed_manifest.revision == 2
    assert failed_manifest.stages[-1].status is StageStatus.FAILED

    retried = agent.run_graph(
        failed.manifest.path,
        failed.manifest.sha256,
        config={
            "context_path": tmp_path / "main.bin",
            "graph_name": "main_ar1",
            "vector_manifest": vector,
        },
    )
    assert retried.ok and retried.manifest is not None
    final_manifest = _load_run(retried.manifest)
    attempts = [
        (stage.attempt, stage.status)
        for stage in final_manifest.stages
        if stage.name == "run_graph"
    ]
    assert attempts == [
        (1, StageStatus.FAILED),
        (2, StageStatus.SUCCEEDED),
    ]
    assert len(str(failed_manifest.stages[-1].metrics["stage_key"])) == 64


def test_stage_key_hashes_tensor_content_not_only_shape_and_dtype() -> None:
    first = QairtAgent._stage_key(
        "validate",
        "a" * 64,
        {"actual": {"tap": np.array([1.0], dtype=np.float32)}},
    )
    second = QairtAgent._stage_key(
        "validate",
        "a" * 64,
        {"actual": {"tap": np.array([2.0], dtype=np.float32)}},
    )

    assert first != second


def test_stale_manifest_fails_before_adapter_execution(
    tmp_path: Path,
) -> None:
    vector = _vector_case(tmp_path)
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None
    advanced = agent.prepare_vectors(
        plan.manifest.path,
        plan.manifest.sha256,
        config={"manifest_path": vector},
    )
    assert advanced.ok and advanced.manifest is not None
    factory.log.calls.clear()
    instance_count = factory.log.instances
    revision_paths = set(plan.manifest.path.parent.glob("manifest-*.json"))

    stale = agent.run_graph(
        plan.manifest.path,
        plan.manifest.sha256,
        config={
            "context_path": tmp_path / "main.bin",
            "graph_name": "main_ar1",
            "vector_manifest": vector,
        },
    )

    assert not stale.ok
    assert stale.error is not None and stale.error.code is ErrorCode.MANIFEST_CONFLICT
    assert factory.log.calls == []
    assert factory.log.instances == instance_count
    assert set(plan.manifest.path.parent.glob("manifest-*.json")) == revision_paths


def test_continuation_revalidates_manifest_artifacts_before_adapter_use(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)
    plan = agent.plan(spec, offline=True)
    assert plan.manifest is not None
    spec.sources.text.onnx_path.write_bytes(b"mutated-after-manifest")
    instance_count = factory.log.instances

    result = agent.prepare_vectors(
        plan.manifest.path,
        plan.manifest.sha256,
        config={
            "cases": [
                {
                    "case_id": "must-not-run",
                    "inputs": {"x": np.array([1.0], dtype=np.float32)},
                }
            ]
        },
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.ARTIFACT_HASH_MISMATCH
    assert result.manifest == plan.manifest
    assert factory.log.instances == instance_count
    assert factory.log.calls == []


def test_compile_context_forwards_strict_inputs_and_same_parent_retry_reuses(
    tmp_path: Path,
) -> None:
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None
    models = (
        _write(tmp_path / "models" / "ar1.dlc", b"ar1"),
        _write(tmp_path / "models" / "ar128.dlc", b"ar128"),
    )
    output_path = tmp_path / "contexts" / "shared.bin"
    evidence = Qwen35ValidationEvidence(
        ar_rewrite_passed=True,
        state_io_passed=True,
        mha2sha_passed=True,
        initializer_compatibility_passed=True,
        standalone_vs_joint_passed=True,
        evidence_id="scoped-evidence",
    )
    config = {
        "models": models,
        "output_path": output_path,
        "graph_names": ["decoder_ar1", "decoder_ar128"],
        "ar_values": [1, 128],
        "source_kinds": ["derived", "derived"],
        "slice_name": "decoder_00",
        "context_length": 4096,
        "expect_native_kv": True,
        "native_kv_config": {
            "graphs": [
                {
                    "graph_name": "decoder_ar1",
                    "tensors": [
                        {
                            "tensor_name": "past_key_0_in",
                            "dataFormat": "QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT",
                        }
                    ],
                },
                {
                    "graph_name": "decoder_ar128",
                    "tensors": [
                        {
                            "tensor_name": "past_key_0_in",
                            "dataFormat": "QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT",
                        },
                        {
                            "tensor_name": "present_key_0_out",
                            "dataFormat": "QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT",
                        },
                    ],
                },
            ]
        },
        "native_kv_expectations": [
            {
                "graph_name": "decoder_ar1",
                "ar": 1,
                "input_names": ["past_key_0_in", "hidden"],
                "output_names": ["present_key_0_out"],
            },
            {
                "graph_name": "decoder_ar128",
                "ar": 128,
                "input_names": ["past_key_0_in", "hidden"],
                "output_names": ["present_key_0_out"],
            },
        ],
        "qwen35_validation_evidence": evidence,
    }

    first = agent.compile_context(
        plan.manifest.path,
        plan.manifest.sha256,
        config=config,
    )
    assert first.ok and first.manifest is not None
    compile_call = next(
        details
        for _, name, details in factory.log.calls
        if name == "compile_context"
    )
    expectations = compile_call["native_kv_expectations"]
    assert [item.graph_name for item in expectations] == [
        "decoder_ar1",
        "decoder_ar128",
    ]
    assert compile_call["qwen35_validation_evidence"] is evidence
    instance_count = factory.log.instances
    call_count = len(factory.log.calls)

    retried = agent.compile_context(
        plan.manifest.path,
        plan.manifest.sha256,
        config=config,
    )

    assert retried.ok and retried.manifest == first.manifest
    assert retried.data is not None and retried.data["reused"] is True
    assert factory.log.instances == instance_count
    assert len(factory.log.calls) == call_count


def test_compile_context_rejects_json_reconstructed_qwen35_evidence(
    tmp_path: Path,
) -> None:
    factory = FakeAdapterFactory()
    agent = _fake_agent(factory)
    plan = agent.plan(_make_spec(tmp_path), offline=True)
    assert plan.manifest is not None
    models = (
        _write(tmp_path / "models" / "ar1.dlc", b"ar1"),
        _write(tmp_path / "models" / "ar128.dlc", b"ar128"),
    )

    result = agent.compile_context(
        plan.manifest.path,
        plan.manifest.sha256,
        config={
            "models": models,
            "graph_names": ["ar1", "ar128"],
            "ar_values": [1, 128],
            "qwen35_validation_evidence": {
                "ar_rewrite_passed": True,
                "state_io_passed": True,
                "mha2sha_passed": True,
                "initializer_compatibility_passed": True,
                "standalone_vs_joint_passed": True,
            },
        },
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.INVALID_SPEC
    assert "cannot be reconstructed from JSON" in result.error.message
    assert "compile_context" not in factory.log.names()


def test_initial_manifest_publish_failure_is_structured(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path / "valid-spec")
    blocked_root = _write(tmp_path / "not-a-directory", b"file")
    invalid_output_spec = spec.model_copy(update={"output_root": blocked_root})

    result = QairtAgent(adapter_factory=FakeAdapterFactory()).generate_config(
        invalid_output_spec
    )

    assert not result.ok
    assert result.error is not None
    assert result.manifest is None


def test_benchmark_optrace_publishes_reusable_evidence_and_auto_hotspots(
    tmp_path: Path,
) -> None:
    vectors = _vector_case(
        tmp_path,
        inputs={"x": np.array([2.0], dtype=np.float32)},
        goldens={"y": np.array([4.0], dtype=np.float32)},
        case_id="optrace",
    )
    spec = _make_spec(
        tmp_path,
        vectors={
            "mode": "provided",
            "validation_manifest": vectors,
        },
        sequence={
            "ars": [1],
            "context_lengths": [4096],
            "weight_sharing": False,
            "native_kv": False,
        },
    )
    factory = FakeAdapterFactory(profile_cycles=125.0)
    agent = _fake_agent(factory)
    built = agent.build(spec)
    assert built.ok and built.manifest is not None

    benchmarked = agent.benchmark(
        built.manifest.path,
        built.manifest.sha256,
        config={
            "optrace": True,
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )

    assert benchmarked.ok, benchmarked.error
    assert benchmarked.manifest is not None
    assert benchmarked.data is not None
    assert benchmarked.data["optrace_evidence"]["logical_name"] == (
        "optrace_evidence"
    )
    manifest = _load_run(benchmarked.manifest)
    evidence_ref = next(
        artifact
        for artifact in manifest.artifacts
        if artifact.logical_name == "optrace_evidence"
    )
    evidence = json.loads(evidence_ref.path.read_text(encoding="utf-8"))
    assert evidence["schema"] == "qairt-agent.optrace-evidence.v1"
    assert evidence["profile_scope"] == "production_runtime_optrace"
    assert evidence["ops"][0]["cycles"] == 125.0
    assert evidence["ops"][1]["cycles"] == 30.0
    assert evidence["ops"][1]["cycle_basis"] == "max_thread"
    assert evidence["profiles"][0]["report_artifacts"]
    for artifact in manifest.artifacts:
        verify_artifact(artifact)

    diagnosed = agent.diagnose_quality(
        benchmarked.manifest.path,
        benchmarked.manifest.sha256,
    )

    assert diagnosed.ok, diagnosed.error
    assert diagnosed.data is not None
    assert diagnosed.data["diagnosis_kind"] == "latency"
    assert diagnosed.data["comparison_mode"] == "candidate_hotspot_only"
    assert diagnosed.data["regression_attribution_supported"] is False
    assert diagnosed.data["first_problem"]["candidate_cycles"] == 125.0
    assert diagnosed.data["layer_attributions"][0]["layer"] == 0
    assert diagnosed.data["tensor_attributions"][0]["tensor"] == "y"


def test_auto_latency_uses_compatible_fork_profile_as_delta_baseline(
    tmp_path: Path,
) -> None:
    vectors = _vector_case(
        tmp_path,
        inputs={"x": np.array([2.0], dtype=np.float32)},
        goldens={"y": np.array([4.0], dtype=np.float32)},
        case_id="optrace-baseline",
    )
    spec = _make_spec(
        tmp_path,
        vectors={
            "mode": "provided",
            "validation_manifest": vectors,
        },
        sequence={
            "ars": [1],
            "context_lengths": [4096],
            "weight_sharing": False,
            "native_kv": False,
        },
    )
    factory = FakeAdapterFactory(profile_cycles=100.0)
    agent = _fake_agent(factory)
    built = agent.build(spec)
    assert built.ok and built.manifest is not None
    benchmark_config = {
        "optrace": True,
        "warmup_runs": 0,
        "measured_runs": 1,
        "aa_calibration": False,
    }
    baseline = agent.benchmark(
        built.manifest.path,
        built.manifest.sha256,
        config=benchmark_config,
    )
    assert baseline.ok and baseline.manifest is not None

    store = ManifestStore(baseline.manifest.path.parent.parent)
    _, fork_ref = store.fork_snapshot(baseline.manifest)
    factory.profile_cycles = 160.0
    candidate = agent.benchmark(
        fork_ref.path,
        fork_ref.sha256,
        config=benchmark_config,
    )
    assert candidate.ok, candidate.error
    assert candidate.manifest is not None

    diagnosed = agent.diagnose_quality(
        candidate.manifest.path,
        candidate.manifest.sha256,
    )

    assert diagnosed.ok, diagnosed.error
    assert diagnosed.data is not None
    assert diagnosed.data["comparison_mode"] == "parent_profile_delta"
    assert diagnosed.data["regression_attribution_supported"] is True
    matmul = next(
        item
        for item in diagnosed.data["attributions"]
        if item["lineage"]["source_op_id"] == "MatMul_0"
    )
    assert matmul["baseline_cycles"] == 100.0
    assert matmul["candidate_cycles"] == 160.0
    assert matmul["delta_cycles"] == 60.0
    assert diagnosed.data["sources"]["baseline"] is not None


def test_auto_quality_uses_sqnr_observation_and_explicit_lineage(
    tmp_path: Path,
) -> None:
    spec = _make_spec(
        tmp_path,
        sequence={
            "ars": [1],
            "context_lengths": [4096],
            "weight_sharing": False,
            "native_kv": False,
        },
    )
    agent = _fake_agent(FakeAdapterFactory())
    built = agent.build(spec)
    assert built.ok and built.manifest is not None
    validated = agent.validate(
        built.manifest.path,
        built.manifest.sha256,
        config={
            "references": {
                "decoder": {
                    "hidden": np.array([1.0], dtype=np.float32),
                }
            },
            "device_chain_outputs": {
                "decoder": {
                    "hidden": np.array([1.25], dtype=np.float32),
                }
            },
            "lineage": {
                "decoder": {
                    "hidden": {
                        "layer": 7,
                        "op_type": "MatMul",
                    }
                }
            },
        },
    )
    assert validated.ok, validated.error
    assert validated.manifest is not None

    diagnosed = agent.diagnose_quality(
        validated.manifest.path,
        validated.manifest.sha256,
    )

    assert diagnosed.ok, diagnosed.error
    assert diagnosed.data is not None
    assert diagnosed.data["diagnosis_kind"] == "quality"
    assert diagnosed.data["attribution_scope"] == "slice_tensor_layer_op"
    assert diagnosed.data["first_observed"]["slice_id"] == "decoder"
    assert diagnosed.data["first_observed"]["tensor_name"] == "hidden"
    assert diagnosed.data["layer_attributions"][0]["layer"] == 7
    assert diagnosed.data["op_attributions"][0]["op"] == "MatMul"
    assert diagnosed.data["op_attribution_supported"] is True


def test_genai_optrace_profiles_raw_tensor_runtime_without_relabeling_wall_time(
    tmp_path: Path,
) -> None:
    vectors = _vector_case(
        tmp_path,
        inputs={"x": np.array([2.0], dtype=np.float32)},
        goldens={"y": np.array([4.0], dtype=np.float32)},
        case_id="genai-optrace",
    )
    spec = _make_spec(
        tmp_path,
        vectors={
            "mode": "provided",
            "validation_manifests_by_ar": {1: vectors},
        },
        sequence={
            "ars": [1],
            "context_lengths": [4096],
            "weight_sharing": False,
            "native_kv": True,
        },
    )
    spec = spec.model_copy(
        update={
            "metadata": {
                **spec.metadata,
                "attached_models_by_ar": {
                    "1": {
                        "model_path": str(spec.sources.text.onnx_path),
                        "encodings_path": str(
                            spec.sources.text.encodings_path
                        ),
                    }
                },
            }
        }
    )
    factory = FakeAdapterFactory(profile_cycles=90.0)
    agent = _fake_agent(factory)
    built = agent.build_genai_container(spec)
    assert built.ok and built.manifest is not None

    benchmarked = agent.benchmark(
        built.manifest.path,
        built.manifest.sha256,
        config={
            "prompt": [{"role": "user", "content": "hello"}],
            "optrace": True,
            "warmup_runs": 0,
            "measured_runs": 1,
            "aa_calibration": False,
        },
    )

    assert benchmarked.ok, benchmarked.error
    assert benchmarked.manifest is not None
    assert benchmarked.data is not None
    assert benchmarked.data["scope"] == "genai_generation"
    manifest = _load_run(benchmarked.manifest)
    evidence_ref = next(
        artifact
        for artifact in manifest.artifacts
        if artifact.logical_name == "optrace_evidence"
    )
    evidence = json.loads(evidence_ref.path.read_text(encoding="utf-8"))
    assert evidence["profile_scope"] == (
        "raw_compiled_slices_not_generation_wall_latency"
    )
    assert evidence["profiles"][0]["slice_id"] == "split_000"
    assert "create_genai_executor" in factory.log.names()
    assert "generate" in factory.log.names()
    assert "profile" in factory.log.names()


def test_auto_diagnose_fails_closed_without_sqnr_drop_or_optrace(
    tmp_path: Path,
) -> None:
    agent = _fake_agent(FakeAdapterFactory())
    built = agent.build(
        _make_spec(
            tmp_path,
            sequence={
                "ars": [1],
                "context_lengths": [4096],
                "weight_sharing": False,
                "native_kv": False,
            },
        )
    )
    assert built.ok and built.manifest is not None

    diagnosed = agent.diagnose_quality(
        built.manifest.path,
        built.manifest.sha256,
    )

    assert not diagnosed.ok
    assert diagnosed.error is not None
    assert diagnosed.error.code is ErrorCode.INVALID_SPEC
    assert "no provable quality divergence" in diagnosed.error.message
