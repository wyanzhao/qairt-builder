from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qairt_agent.families import FamilyId, build_split_plan
from qairt_agent.harness import DEFAULT_CONSTRAINTS, resolve_target

#: Whichever reviewed target this harness names; the fixtures track it
#: instead of hardcoding one chipset.
_ACTIVE_TARGET = resolve_target(constraints=DEFAULT_CONSTRAINTS)
from qairt_agent.qairt_adapter import (
    ExperimentalFeatureError,
    NativeKvConfigError,
    BuildResult,
    CompiledContextArtifact,
    ConvertedModelArtifact,
    ModelVariantArtifact,
    GenAIAttachedModel,
    GenAIContainerBuildResult,
    NATIVE_KV_DATA_FORMAT,
    NativeKvGraphExpectation,
    PreflightChecker,
    PreflightReport,
    QairtConfigurationError,
    QairtSdkAdapter,
    Qwen35ValidationEvidence,
    Qwen35RuntimeValidationResult,
    TransformedSliceArtifact,
    audit_native_kv_config,
    expected_tensors,
)


class PreflightTests(unittest.TestCase):
    def _checker(self) -> PreflightChecker:
        return PreflightChecker(
            environ={},
            system=lambda: "Linux",
            machine=lambda: "x86_64",
            python_version=(3, 10),
            os_release_reader=lambda: {"ID": "ubuntu", "VERSION_ID": "22.04"},
        )

    def _sdk(self, root: Path) -> Path:
        sdk = root / "2.49.0.260730"
        (sdk / "lib" / "python").mkdir(parents=True)
        (sdk / "sdk.yaml").write_text(
            "product: QAIRT\nversion: 2.49.0\nbuild_id: 260730134355\n",
            encoding="utf-8",
        )
        htp = (
            sdk
            / "lib"
            / "python"
            / "qti"
            / "aisw"
            / "converters"
            / "common"
            / "backend_aware_configs"
        )
        htp.mkdir(parents=True)
        (htp / "htp_v2.json").write_text(
            json.dumps(
                {
                    "soc_model_to_arch": {
                        _ACTIVE_TARGET.chipset: _ACTIVE_TARGET.dsp_arch
                    }
                }
            ),
            encoding="utf-8",
        )
        return sdk

    def test_pinned_sdk_and_target_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sdk = self._sdk(Path(directory))
            report = self._checker().check(
                {
                    "sdk_root": sdk,
                    "target": {
                        "chipset": _ACTIVE_TARGET.chipset,
                        "dsp_arch": _ACTIVE_TARGET.dsp_arch,
                        "soc_model": _ACTIVE_TARGET.soc_model,
                    },
                }
            )
        self.assertTrue(report.ok, report.issues)
        self.assertEqual(report.soc_model, _ACTIVE_TARGET.soc_model)

    def test_mismatched_arch_or_implicit_soc_model_is_rejected(self) -> None:
        # The right chipset with the wrong architecture, and no soc_model at
        # all, are both refused: nothing about the target may be implicit.
        with tempfile.TemporaryDirectory() as directory:
            sdk = self._sdk(Path(directory))
            report = self._checker().check(
                {
                    "sdk_root": sdk,
                    "target": {
                        "chipset": _ACTIVE_TARGET.chipset,
                        "dsp_arch": "v66",
                    },
                }
            )
        codes = {issue.code for issue in report.errors}
        self.assertIn("target.dsp_arch", codes)
        self.assertIn("target.soc_model", codes)

    def test_checker_uses_injected_harness_versions(self) -> None:
        constraints = replace(
            DEFAULT_CONSTRAINTS,
            qairt_version="2.49.0",
            qairt_build_id="next-build",
            ubuntu_version="24.04",
            python_version="3.11",
        )
        checker = PreflightChecker(
            environ={},
            system=lambda: "Linux",
            machine=lambda: "x86_64",
            python_version=(3, 11),
            os_release_reader=lambda: {
                "ID": "ubuntu",
                "VERSION_ID": "24.04",
            },
            constraints=constraints,
        )
        with tempfile.TemporaryDirectory() as directory:
            sdk = self._sdk(Path(directory))
            (sdk / "sdk.yaml").write_text(
                "product: QAIRT\nversion: 2.49.0\nbuild_id: next-build\n",
                encoding="utf-8",
            )
            report = checker.check(
                {
                    "sdk_root": sdk,
                    "target": {
                        "chipset": _ACTIVE_TARGET.chipset,
                        "dsp_arch": _ACTIVE_TARGET.dsp_arch,
                        "soc_model": _ACTIVE_TARGET.soc_model,
                    },
                }
            )

        self.assertTrue(report.ok, report.issues)


class FakeGenAiUtils:
    """Stands in for QAIRT's gen_ai_utils: same selection rule as the SDK's.

    Mirrors gen_kv_format_config exactly, including the ``ar > 0`` guard that
    the previous local reimplementation was missing, so these tests exercise
    the adapter's use of the SDK rather than a second copy of the rule.
    """

    def __init__(self, io_by_stem: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]):
        self.io_by_stem = io_by_stem
        self.seen_sequence_lengths: list[Any] = []

    def gen_kv_format_config(self, split_files) -> dict:
        graphs = []
        for exported in split_files:
            stem = Path(exported.onnx_path).stem
            inputs, outputs = self.io_by_stem[stem]
            ar = exported.info.sequence_length if exported.info is not None else None
            self.seen_sequence_lengths.append(ar)
            multiple_of_32 = ar is not None and ar > 0 and ar % 32 == 0
            candidates = inputs + outputs if multiple_of_32 else inputs
            tensors = [
                {
                    "tensor_name": name,
                    "dataFormat": NATIVE_KV_DATA_FORMAT,
                }
                for name in candidates
                if "key" in name or "value" in name
            ]
            if tensors:
                graphs.append({"graph_name": stem, "tensors": tensors})
        return {"graphs": graphs}


class FakeExportedFileInfo:
    def __init__(self, sequence_length=None, context_length=None) -> None:
        self.sequence_length = sequence_length
        self.context_length = context_length


class FakeExportedFiles:
    """Mirrors the SDK: _info is unset until the caller stamps it."""

    def __init__(self, onnx_path, data_path) -> None:
        self.onnx_path = Path(onnx_path)
        self.data_path = Path(data_path)
        self._info = None

    @property
    def info(self):
        return self._info


class NativeKvTests(unittest.TestCase):
    def _adapter_with_sdk(self, io_by_stem):
        utils = FakeGenAiUtils(io_by_stem)
        modules = {
            "qairt.gen_ai_api.builders.gen_ai_utils": utils,
            "qairt.optimizer.onnx.graph": SimpleNamespace(
                ExportedFiles=FakeExportedFiles,
                ExportedFileInfo=FakeExportedFileInfo,
            ),
        }
        adapter = QairtSdkAdapter(
            module_loader=lambda name: modules[name],
            require_successful_preflight=False,
        )
        return adapter, utils

    def _expectation(self, directory, stem, graph_name, ar, inputs, outputs):
        model = Path(directory) / f"{stem}.onnx"
        model.write_bytes(b"onnx")
        return NativeKvGraphExpectation(
            graph_name=graph_name,
            ar=ar,
            input_names=inputs,
            output_names=outputs,
            model_path=model,
        )

    def test_ar1_inputs_only_ar128_inputs_and_outputs(self) -> None:
        io_by_stem = {
            "slice_ar1": (("past_key_0_in", "recurrent_state_0_in", "hidden"), ("past_key_0_out",)),
            "slice_ar128": (("past_value_0_in", "conv_state_0_in"), ("past_value_0_out",)),
        }
        adapter, _ = self._adapter_with_sdk(io_by_stem)
        with tempfile.TemporaryDirectory() as directory:
            expectations = (
                self._expectation(directory, "slice_ar1", "decoder_ar1", 1,
                                  *io_by_stem["slice_ar1"]),
                self._expectation(directory, "slice_ar128", "decoder_ar128", 128,
                                  *io_by_stem["slice_ar128"]),
            )
            config = adapter._native_kv_config(expectations)

        tensors = {
            graph["graph_name"]: [tensor["tensor_name"] for tensor in graph["tensors"]]
            for graph in config["graphs"]
        }
        # Graph names are ours, tensor selection is QAIRT's.
        self.assertEqual(tensors["decoder_ar1"], ["past_key_0_in"])
        self.assertEqual(
            tensors["decoder_ar128"], ["past_value_0_in", "past_value_0_out"]
        )
        report = audit_native_kv_config(config, expected=expected_tensors(config))
        self.assertTrue(report.ok, report.issues)

    def test_the_ar_is_stamped_so_the_sdk_can_see_it(self) -> None:
        # GraphContext.export leaves ExportedFiles._info unset and the SDK reads
        # the AR from there. Unstamped, every graph would silently demote to
        # inputs-only and AR128 would lose its output marking.
        io_by_stem = {"slice_ar128": (("past_value_0_in",), ("past_value_0_out",))}
        adapter, utils = self._adapter_with_sdk(io_by_stem)
        with tempfile.TemporaryDirectory() as directory:
            expectations = (
                self._expectation(directory, "slice_ar128", "decoder_ar128", 128,
                                  *io_by_stem["slice_ar128"]),
            )
            config = adapter._native_kv_config(expectations)

        self.assertEqual(utils.seen_sequence_lengths, [128])
        self.assertEqual(
            [t["tensor_name"] for t in config["graphs"][0]["tensors"]],
            ["past_value_0_in", "past_value_0_out"],
        )

    def test_unknown_ar_keeps_outputs_out_of_the_hmx_layout(self) -> None:
        # A slice artifact without an AR reaches the expectation as ar=0; it is
        # passed to the SDK as None so its own ``ar > 0`` guard applies.
        io_by_stem = {"slice": (("past_key_0_in",), ("past_key_0_out",))}
        adapter, utils = self._adapter_with_sdk(io_by_stem)
        with tempfile.TemporaryDirectory() as directory:
            expectations = (
                self._expectation(directory, "slice", "decoder_unknown_ar", 0,
                                  *io_by_stem["slice"]),
            )
            config = adapter._native_kv_config(expectations)

        self.assertEqual(utils.seen_sequence_lengths, [None])
        self.assertEqual(
            [t["tensor_name"] for t in config["graphs"][0]["tensors"]],
            ["past_key_0_in"],
        )

    def test_non_cache_role_names_are_never_marked(self) -> None:
        # QAIRT's bare substring rule sweeps these in; the documented role
        # filter takes them back out and records that it did.
        io_by_stem = {
            "slice": (
                ("past_key_0_in", "key_padding_mask", "value_position_index"),
                ("past_key_0_out",),
            )
        }
        adapter, _ = self._adapter_with_sdk(io_by_stem)
        with tempfile.TemporaryDirectory() as directory:
            expectations = (
                self._expectation(directory, "slice", "decoder_ar128", 128,
                                  *io_by_stem["slice"]),
            )
            config = adapter._native_kv_config(expectations)

        self.assertEqual(
            [t["tensor_name"] for t in config["graphs"][0]["tensors"]],
            ["past_key_0_in", "past_key_0_out"],
        )
        self.assertEqual(
            [item["tensor_name"] for item in adapter._native_kv_role_filtered],
            ["key_padding_mask", "value_position_index"],
        )

        swept_in = {
            "graphs": [
                {
                    "graph_name": "decoder_ar128",
                    "tensors": [
                        {"tensor_name": name, "dataFormat": NATIVE_KV_DATA_FORMAT}
                        for name in ("past_key_0_in", "key_padding_mask", "past_key_0_out")
                    ],
                }
            ]
        }
        report = audit_native_kv_config(swept_in, expected=expected_tensors(config))
        self.assertFalse(report.ok)
        self.assertIn("key_padding_mask", " ".join(report.issues))

    def test_missing_slice_onnx_fails_closed(self) -> None:
        adapter, _ = self._adapter_with_sdk({})
        with self.assertRaises(NativeKvConfigError):
            adapter._native_kv_config(
                (
                    NativeKvGraphExpectation(
                        graph_name="decoder_ar1",
                        ar=1,
                        input_names=("past_key_0_in",),
                        output_names=(),
                    ),
                )
            )

    def test_wrong_layout_is_rejected(self) -> None:
        config = {
            "graphs": [
                {
                    "graph_name": "decoder",
                    "tensors": [{"tensor_name": "past_key_0_in", "dataFormat": "wrong"}],
                }
            ]
        }
        report = audit_native_kv_config(config, expected_graph_names=("decoder",))
        self.assertFalse(report.ok)
        self.assertIn(NATIVE_KV_DATA_FORMAT, " ".join(report.issues))


def _device_configs(kwargs: dict[str, Any]) -> list[Any]:
    """Mirror the SDK: build one device config per soc_model in soc_details."""

    details = str(kwargs.get("soc_details") or "")
    fields = dict(
        part.split(":", 1) for part in details.split(";") if ":" in part
    )
    if not fields.get("soc_model") or not fields.get("dsp_arch"):
        return []
    return [
        SimpleNamespace(dsp_arch=fields["dsp_arch"], soc_model=int(model))
        for model in fields["soc_model"].split("|")
    ]



#: The values QAIRT 2.49.0.260730's Qwen3_5BuilderHTP actually carries. The
#: adapter reads them from the SDK, so the fakes must supply them rather than
#: the adapter carrying a copy.
_QWEN3_5_SDK_START_POINTS = [
    (r"/model_layers_(\d+)_linear_attn_norm_Mul_3/Mul_output_0", 1, None),
    (r"/model_layers_(\d+)_self_attn_Mul_8/Mul_output_0", 2, {4096: 256}),
    (r"recurrent_state_(\d+)_out", 1, None),
    (r"conv_state_(\d+)_out", 1, None),
]


def _fake_qwen35_builder_module(points=None) -> SimpleNamespace:
    """A stand-in for the SDK module that owns Qwen3.5's MHA2SHA start points."""

    selected = _QWEN3_5_SDK_START_POINTS if points is None else points

    class Qwen3_5BuilderHTP:
        _QWEN3_5_START_POINTS = [
            SimpleNamespace(name_pattern=pattern, split_axis=axis, split_map=split_map)
            for pattern, axis, split_map in selected
        ]

    return SimpleNamespace(Qwen3_5BuilderHTP=Qwen3_5BuilderHTP)


class FakeCompileConfig:
    created: list["FakeCompileConfig"] = []
    # Set to simulate the SDK's "could not set soc model for chipset ...
    # skipping device config creation" path, which leaves the compiler on its
    # own default target.
    skip_device_configs = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.device_custom_configs = (
            [] if type(self).skip_device_configs else _device_configs(kwargs)
        )
        self.mode = None
        self.mode_kwargs = None
        type(self).created.append(self)

    def set_mode(self, mode: str, **kwargs: Any) -> "FakeCompileConfig":
        self.mode = mode
        self.mode_kwargs = kwargs
        return self


class FakeModel:
    def __init__(self, name: str = "model") -> None:
        self.name = name
        self.graphs_info = [SimpleNamespace(name=name)]
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.saved: list[str] = []

    def save(self, path: str) -> str:
        self.saved.append(path)
        return path

    def initialize(self, device: Any = None, **kwargs: Any) -> None:
        # QAIRT sets this attribute when it caches an execution context; the
        # adapter keys its profiling guard off exactly this.
        self._inference_handle = ("handle", device)
        self.calls.append(("initialize", {"device": device, **kwargs}))

    def destroy(self) -> None:
        self.__dict__.pop("_inference_handle", None)
        self.calls.append(("destroy", {}))

    def __call__(self, inputs: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((inputs, kwargs))
        return {"outputs": inputs, "kwargs": kwargs}


# The shape QAIRT's detailed profiling log parses into.
FAKE_DEVICE_REPORT: dict[str, Any] = {
    "metadata": {"appName": "qnn-net-run"},
    "messages": [
        {
            "method": "BACKEND_EXECUTE",
            "profilingEvents": [
                {
                    "identifier": "Accelerator (execute excluding wait) time",
                    "unit": "MICROSEC",
                    "value": 77,
                },
                {
                    "identifier": "Accelerator (execute) time",
                    "unit": "MICROSEC",
                    "value": 1734,
                },
            ],
        }
    ],
}


class FakeProfilingReport(dict):
    """Dict-like, as the optrace tests expect, plus QAIRT's ``.data``."""

    def __init__(self, payload: dict[str, Any], data: dict[str, Any]) -> None:
        super().__init__(payload)
        self.data = data


class FakeProfiler:
    def __init__(self, context: dict[str, Any]) -> None:
        self.context = context

    def __enter__(self) -> "FakeProfiler":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def generate_reports(self):
        payload = {"kind": self.context.get("option", "detailed"), **self.context}
        return [FakeProfilingReport(payload, FAKE_DEVICE_REPORT)]


class FakeLLMContainer:
    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        self.calls = calls

    def save(self, destination: str, *, exist_ok: bool = False) -> None:
        self.calls.append(("save", (destination, exist_ok)))
        Path(destination).mkdir(parents=True, exist_ok=exist_ok)
        (Path(destination) / "metadata.json").write_text(
            '{"container": "fake"}\n',
            encoding="utf-8",
        )


class RecordingGenAIBuilder:
    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        self.calls = calls
        self._encodings_path: str | None = None
        self._skip_ar_conversion: bool | None = None
        self._native_kv: bool | None = None
        self._weight_sharing: bool | None = None
        self.container = FakeLLMContainer(calls)

    @property
    def encodings_path(self) -> str | None:
        return self._encodings_path

    @encodings_path.setter
    def encodings_path(self, value: str) -> None:
        self._encodings_path = value
        self.calls.append(("encodings_path", value))

    @property
    def skip_ar_conversion(self) -> bool | None:
        return self._skip_ar_conversion

    @skip_ar_conversion.setter
    def skip_ar_conversion(self, value: bool) -> None:
        self._skip_ar_conversion = value
        self.calls.append(("skip_ar_conversion", value))

    @property
    def native_kv(self) -> bool | None:
        return self._native_kv

    @native_kv.setter
    def native_kv(self, value: bool) -> None:
        self._native_kv = value
        self.calls.append(("native_kv", value))

    @property
    def weight_sharing(self) -> bool | None:
        return self._weight_sharing

    @weight_sharing.setter
    def weight_sharing(self, value: bool) -> None:
        self._weight_sharing = value
        self.calls.append(("weight_sharing", value))

    def set_targets(self, targets: list[str]) -> None:
        self.calls.append(("set_targets", targets))

    def set_transformation_options(self, *, options: dict[str, Any]) -> None:
        self.calls.append(("set_transformation_options", options))

    def attach_model_for_arn(
        self,
        ar: int,
        model_path: str,
        encodings_path: str | None,
    ) -> None:
        self.calls.append(
            ("attach_model_for_arn", (ar, model_path, encodings_path))
        )

    def build(self) -> FakeLLMContainer:
        self.calls.append(("build", None))
        return self.container


class GenAIBuilderHTP(RecordingGenAIBuilder):
    pass


class Qwen3MoeBuilderHTP(RecordingGenAIBuilder):
    pass


class Qwen3_5BuilderHTP(RecordingGenAIBuilder):
    pass


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeCompileConfig.created.clear()
        self.compiles: list[tuple[Any, Any]] = []
        self.loads: list[str] = []
        self.fake_qairt = SimpleNamespace(
            CompileConfig=FakeCompileConfig,
            compile=self._compile,
            load=self._load,
            convert=lambda path, **kwargs: FakeModel(Path(path).stem),
            Profiler=FakeProfiler,
        )

    def _compile(self, models: Any, *, config: Any) -> FakeModel:
        self.compiles.append((models, config))
        return FakeModel("compiled")

    def _load(self, path: str) -> FakeModel:
        self.loads.append(path)
        return FakeModel(Path(path).stem)

    def _adapter(self, extra_modules: dict[str, Any] | None = None) -> QairtSdkAdapter:
        modules = {"qairt": self.fake_qairt, **(extra_modules or {})}

        def loader(name: str) -> Any:
            if name not in modules:
                raise ModuleNotFoundError(name)
            return modules[name]

        return QairtSdkAdapter(
            module_loader=loader,
            require_successful_preflight=False,
        )

    def test_compile_weight_sharing_passes_explicit_sm8750_v79(self) -> None:
        adapter = self._adapter()
        with tempfile.TemporaryDirectory() as directory:
            result = adapter.compile_context(
                [FakeModel("ar1"), FakeModel("ar128")],
                output_path=Path(directory) / "decoder.bin",
                graph_names=("decoder_ar1", "decoder_ar128"),
                ar_values=(1, 128),
                source_kinds=("derived", "derived"),
                target_soc="SM8750",
                dsp_arch="v79",
                soc_model=69,
                family=FamilyId.QWEN3_DENSE,
                slice_name="decoder_00",
            )
        config = FakeCompileConfig.created[-1]
        self.assertEqual(config.mode, "weight_sharing")
        self.assertEqual(config.mode_kwargs["soc_model"], 69)
        self.assertEqual(config.mode_kwargs["dsp_arch"], "v79")
        self.assertEqual(
            config.mode_kwargs["graph_names"],
            ["decoder_ar1", "decoder_ar128"],
        )
        self.assertEqual(result.ar_values, (1, 128))

    def test_saved_genai_container_executor_is_prepared_and_cleaned(self) -> None:
        calls: list[tuple[str, Any]] = []

        class Executor:
            def prepare_environment(self) -> "Executor":
                calls.append(("prepare_environment", None))
                return self

            def clean_environment(self) -> "Executor":
                calls.append(("clean_environment", None))
                return self

        executor = Executor()

        class Container:
            def get_executor(self, *, device: Any, **kwargs: Any) -> Executor:
                calls.append(("get_executor", (device, kwargs)))
                return executor

        def load_container(path: str) -> Container:
            calls.append(("load_container", path))
            return Container()

        adapter = self._adapter(
            {
                "qairt.gen_ai_api.containers.container_factory": SimpleNamespace(
                    load_container=load_container
                )
            }
        )
        loaded = adapter.create_genai_executor(
            "/tmp/container",
            device="android-device",
        )
        adapter.clean_genai_executor(loaded)

        self.assertIs(loaded, executor)
        self.assertEqual(
            calls,
            [
                ("load_container", "/tmp/container"),
                (
                    "get_executor",
                    (
                        "android-device",
                        {
                            "clean_up": False,
                            "prepare_environment": False,
                        },
                    ),
                ),
                ("prepare_environment", None),
                ("clean_environment", None),
            ],
        )

    def test_saved_genai_raw_slices_bind_public_graph_metadata(self) -> None:
        def graph(name: str) -> Any:
            return SimpleNamespace(
                name=name,
                inputs=[SimpleNamespace(name="x")],
                outputs=[SimpleNamespace(name="y")],
            )

        container = SimpleNamespace(
            models=[
                SimpleNamespace(
                    graphs_info=[
                        graph("decoder_ar1"),
                        graph("decoder_ar128"),
                    ]
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            model_path = destination / "models" / "split_0" / "model.bin"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"context")
            slices, supported, notes = QairtSdkAdapter._saved_genai_raw_slices(
                container,
                destination,
                (1, 128),
            )

        self.assertTrue(supported)
        self.assertTrue(notes)
        self.assertEqual(len(slices), 1)
        self.assertEqual(
            slices[0].graph_names_by_ar,
            {1: "decoder_ar1", 128: "decoder_ar128"},
        )
        self.assertEqual(slices[0].input_names, ("x",))
        self.assertEqual(slices[0].output_names, ("y",))

    def test_standalone_vit_uses_convert_then_single_graph_compile(self) -> None:
        adapter = self._adapter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "vit.onnx"
            encodings = root / "vit.encodings"
            model.write_bytes(b"onnx")
            encodings.write_bytes(b"{}")
            spec = {
                "sources": {
                    "text": {
                        "onnx_path": model,
                        "encodings_path": encodings,
                    }
                },
                "target": {
                    "chipset": "SM8750",
                    "dsp_arch": "v79",
                    "soc_model": 69,
                },
            }
            result = adapter.build_standalone_vit(
                spec,
                {
                    "sources": spec["sources"],
                    "sequence": {
                        "ars": [1],
                        "weight_sharing": False,
                        "native_kv": False,
                    },
                    "transforms": {"mha2sha": False},
                    "quantization": {"mode": "apply_encodings"},
                    "compile": {"compiler_options": {}},
                },
                root / "build",
            )

        self.assertEqual(len(result.converted_models), 1)
        self.assertEqual(len(result.contexts), 1)
        self.assertEqual(result.contexts[0].slice_name, "vit")
        self.assertEqual(result.contexts[0].graph_names, ("vit",))
        self.assertFalse(result.contexts[0].weight_sharing)
        self.assertEqual(len(self.compiles), 1)

    def test_standalone_vit_keeps_diagnostic_outputs_out_of_production(
        self,
    ) -> None:
        adapter = self._adapter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "vit.onnx"
            encodings = root / "vit.encodings"
            model.write_bytes(b"onnx")
            encodings.write_bytes(b"{}")
            spec = {
                "sources": {
                    "text": {
                        "onnx_path": model,
                        "encodings_path": encodings,
                    }
                },
                "target": {
                    "chipset": "SM8750",
                    "dsp_arch": "v79",
                    "soc_model": 69,
                },
            }
            result = adapter.build_standalone_vit(
                spec,
                {
                    "sources": spec["sources"],
                    "sequence": {
                        "ars": [1],
                        "weight_sharing": False,
                        "native_kv": False,
                    },
                    "transforms": {"mha2sha": False},
                    "quantization": {"mode": "apply_encodings"},
                    "compile": {
                        "compiler_options": {"profiling_level": "detailed"},
                        "output_tensors": ["encoder.layer.0/output"],
                    },
                },
                root / "build",
            )

        self.assertEqual(len(result.contexts), 1)
        self.assertEqual(len(result.diagnostic_contexts), 1)
        # The spec names sm8750 while the harness names sm8850: the compile
        # follows the spec, not a global constant.
        self.assertEqual(
            FakeCompileConfig.created[0].kwargs,
            {
                "backend": "HTP",
                "soc_details": "chipset:SM8750;dsp_arch:v79;soc_model:69",
                "data_format_config": None,
                "profiling_level": "detailed",
            },
        )
        self.assertEqual(
            FakeCompileConfig.created[1].kwargs,
            {
                "backend": "HTP",
                "soc_details": "chipset:SM8750;dsp_arch:v79;soc_model:69",
                "data_format_config": None,
                "profiling_level": "detailed",
                "set_output_tensors": ["encoder.layer.0/output"],
            },
        )
        self.assertIn(
            "diagnostic_contexts",
            result.diagnostic_contexts[0].context_binary_path.parts,
        )

    def test_qwen35_derived_ar_fails_closed_without_evidence(self) -> None:
        imports: list[str] = []

        def loader(name: str) -> Any:
            imports.append(name)
            return self.fake_qairt

        adapter = QairtSdkAdapter(
            module_loader=loader,
            require_successful_preflight=False,
        )
        with self.assertRaises(ExperimentalFeatureError):
            adapter.compile_context(
                [FakeModel("ar1"), FakeModel("ar128")],
                output_path="/tmp/qwen35.bin",
                graph_names=("ar1", "ar128"),
                ar_values=(1, 128),
                source_kinds=("derived", "derived"),
                target_soc="SM8750",
                dsp_arch="v79",
                soc_model=69,
                family=FamilyId.QWEN3_5,
            )
        self.assertEqual(imports, [])

    def test_qwen35_caller_constructed_boole_cannot_authorize_compile(self) -> None:
        evidence = Qwen35ValidationEvidence(True, True, True, True, True)
        adapter = self._adapter()
        with self.assertRaises(ExperimentalFeatureError):
            adapter.compile_context(
                [FakeModel("ar1"), FakeModel("ar128")],
                output_path="/tmp/qwen35.bin",
                graph_names=("ar1", "ar128"),
                ar_values=(1, 128),
                source_kinds=("derived", "derived"),
                target_soc="SM8750",
                dsp_arch="v79",
                soc_model=69,
                family=FamilyId.QWEN3_5,
                slice_name="decoder_00",
                context_length=4096,
                qwen35_validation_evidence=evidence,
            )
        self.assertEqual(len(self.compiles), 0)

    def test_qwen35_validator_mints_scoped_evidence(self) -> None:
        state_inputs = (
            SimpleNamespace(name="recurrent_state_0_in", shape=(1, 8), dtype="FLOAT"),
            SimpleNamespace(name="conv_state_0_in", shape=(1, 8), dtype="FLOAT"),
            SimpleNamespace(name="input_ids", shape=(1, 1), dtype="INT64"),
        )
        state_outputs = (
            SimpleNamespace(name="recurrent_state_0_out", shape=(1, 8), dtype="FLOAT"),
            SimpleNamespace(name="conv_state_0_out", shape=(1, 8), dtype="FLOAT"),
        )
        initializer = SimpleNamespace(
            name="weight",
            shape=(8, 8),
            dtype="FLOAT",
            num_elements=64,
            content_sha256="a" * 64,
        )
        node = SimpleNamespace(
            op_type="MatMul",
            inputs=("hidden", "weight"),
        )
        group_slice = SimpleNamespace(op_type="GroupSlice", inputs=("hidden",))
        info = SimpleNamespace(
            inputs=state_inputs,
            outputs=state_outputs,
            initializers=(initializer,),
            nodes=(node, group_slice),
        )

        class Inspector:
            def inspect(self, _path: Path):
                return info

        modules = {"qairt": self.fake_qairt}

        def loader(name: str) -> Any:
            return modules[name]

        adapter = QairtSdkAdapter(
            module_loader=loader,
            require_successful_preflight=False,
            onnx_inspector=Inspector(),
        )

        def graph_context(ar: int):
            return SimpleNamespace(
                graph_ir=SimpleNamespace(
                    meta={"seq_length": ar, "context_length": 4096}
                )
            )

        variants = tuple(
            ModelVariantArtifact(
                model_path=Path(f"variant_ar{ar}.onnx"),
                encodings_path=None,
                ar=ar,
                context_length=4096,
                source_kind="derived",
                family="qwen3.5",
                graph_context=graph_context(ar),
            )
            for ar in (1, 128)
        )

        class TraceContext:
            def get_tracing_info(self, merged: bool = False):
                return [{"passes": ["M2sInsertGroupSliceManually"]}]

        transformed = tuple(
            TransformedSliceArtifact(
                slice_name="decoder_00",
                split_index=1,
                model_path=Path(f"decoder_ar{ar}.onnx"),
                encodings_path=None,
                ar=ar,
                context_length=4096,
                graph_context=TraceContext(),
            )
            for ar in (1, 128)
        )
        converted = tuple(
            ConvertedModelArtifact(
                model_path=Path(f"decoder_ar{ar}.dlc"),
                source_model_path=Path(f"decoder_ar{ar}.onnx"),
                quantization_mode="apply_encodings",
                slice_name="decoder_00",
                ar=ar,
                context_length=4096,
                sdk_model=FakeModel(f"ar{ar}"),
            )
            for ar in (1, 128)
        )

        with tempfile.TemporaryDirectory() as directory:
            runtime_report = Path(directory) / "qwen35_runtime_report.json"
            runtime_report.write_text('{"passed": true}\n', encoding="utf-8")
            validation = adapter.validate_qwen35_derivation(
                variants,
                transformed,
                converted,
                output_dir=Path(directory) / "diagnostics",
                slice_name="decoder_00",
                graph_names=("ar1", "ar128"),
                ar_values=(1, 128),
                context_length=4096,
                target_soc="SM8750",
                dsp_arch="v79",
                soc_model=69,
                runtime_validator=lambda _request: Qwen35RuntimeValidationResult(
                    True,
                    True,
                    True,
                    executed_graph_names=("ar1", "ar128"),
                    golden_vector_ids=("golden-case-0",),
                    report_paths=(runtime_report,),
                    details="golden and joint comparison passed",
                ),
            )
            adapter.compile_context(
                converted,
                output_path=Path(directory) / "production.bin",
                graph_names=("ar1", "ar128"),
                ar_values=(1, 128),
                source_kinds=("derived", "derived"),
                target_soc="SM8750",
                dsp_arch="v79",
                soc_model=69,
                family=FamilyId.QWEN3_5,
                slice_name="decoder_00",
                context_length=4096,
                qwen35_validation_evidence=validation.evidence,
            )
        self.assertEqual(len(validation.diagnostic_contexts), 3)
        self.assertTrue(validation.evidence.evidence_id)

    def test_compile_rejects_mixed_slices_and_wrong_arch_before_sdk(self) -> None:
        adapter = self._adapter()
        with self.assertRaises(QairtConfigurationError):
            adapter.compile_context(
                [FakeModel("ar1"), FakeModel("ar128")],
                output_path="/tmp/unsafe.bin",
                graph_names=("ar1", "ar128"),
                ar_values=(1, 128),
                source_kinds=("derived", "derived"),
                target_soc="SM8750",
                dsp_arch="v81",
                soc_model=69,
            )
        self.assertEqual(self.compiles, [])

    def test_compile_refuses_when_the_sdk_skips_device_config_creation(
        self,
    ) -> None:
        # QAIRT prints "could not set soc model for chipset ... skipping device
        # config creation" and leaves an empty device-config list, which means
        # the compile runs on its own defaults -- dsp_arch v79 with soc_model
        # 69, the exact SM8750 tuple. A value check cannot catch that, so the
        # empty list has to fail on its own.
        adapter = self._adapter()
        FakeCompileConfig.skip_device_configs = True
        try:
            with self.assertRaises(QairtConfigurationError) as caught:
                adapter.compile_context(
                    [FakeModel("ar1"), FakeModel("ar128")],
                    output_path="/tmp/unsafe.bin",
                    graph_names=("decoder_ar1", "decoder_ar128"),
                    ar_values=(1, 128),
                    source_kinds=("derived", "derived"),
                    target_soc="SM8750",
                    dsp_arch="v79",
                    soc_model=69,
                )
        finally:
            FakeCompileConfig.skip_device_configs = False
        self.assertIn("no device configuration", str(caught.exception))
        self.assertEqual(self.compiles, [])

    def test_run_and_profile_select_exactly_one_graph(self) -> None:
        adapter = self._adapter()
        model = FakeModel("compiled")
        result = adapter.run_graph(
            model,
            {"input": [1]},
            graph_name="decoder_ar1",
            native_io=True,
            num_inferences=3,
        )
        self.assertEqual(result["kwargs"]["graph_names"], ["decoder_ar1"])
        self.assertTrue(result["kwargs"]["use_native_input_data"])

        profiled = adapter.profile(
            model,
            {"input": [1]},
            graph_name="decoder_ar1",
        )
        self.assertEqual(profiled.graph_name, "decoder_ar1")
        self.assertEqual(profiled.reports[0]["kind"], "optrace")

    def test_capture_device_execution_reads_qairt_own_device_numbers(self) -> None:
        # The wall-clock benchmark sample is host time around one SDK call.
        # This is what the device reported for the same work.
        adapter = self._adapter()
        model = FakeModel("decoder_ar1")

        block = adapter.capture_device_execution(
            model,
            {"input": [1]},
            graph_name="decoder_ar1",
        )

        self.assertEqual(block["accelerator_compute_us"], 77.0)
        self.assertEqual(block["accelerator_execute_us"], 1734.0)
        self.assertEqual(block["source"], "qairt_profiler_detailed_log")
        # No optrace option: that path additionally needs a schematic binary
        # our compile does not emit.
        self.assertIsNone(block["profiler_option"])

    def test_capture_device_execution_restores_the_working_directory(self) -> None:
        # QAIRT writes its profiling log to a relative output/ directory, so the
        # capture runs somewhere writable -- the worker mounts the project root
        # read-only, which otherwise fails the execute outright.  The process
        # cwd must come back, including when the execute raises.
        adapter = self._adapter()
        before = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "device_profiling"
            adapter.capture_device_execution(
                FakeModel("decoder_ar1"),
                {"input": [1]},
                graph_name="decoder_ar1",
                working_dir=target,
            )
            self.assertEqual(os.getcwd(), before)
            self.assertTrue(target.is_dir())

            class ExplodingModel(FakeModel):
                def __call__(self, inputs: Any, **kwargs: Any) -> dict[str, Any]:
                    raise RuntimeError("Failed to execute model")

            exploding = ExplodingModel("decoder_ar1")
            with self.assertRaises(RuntimeError):
                adapter.capture_device_execution(
                    exploding,
                    {"input": [1]},
                    graph_name="decoder_ar1",
                    working_dir=target,
                )
            self.assertEqual(os.getcwd(), before)

    def test_capture_device_execution_refuses_an_initialized_model(self) -> None:
        # initialize() caches an execution context created with profiling
        # disabled.  Profiling afterwards silently yields no data at all, and
        # QAIRT then fails deep inside report parsing.  Fail closed here, where
        # the message can say why.
        adapter = self._adapter()
        model = FakeModel("decoder_ar1")
        adapter.initialize_execution(model, device=None)

        with self.assertRaises(QairtConfigurationError) as caught:
            adapter.capture_device_execution(
                model,
                {"input": [1]},
                graph_name="decoder_ar1",
            )
        self.assertIn("before initialize_execution", str(caught.exception))

    def test_initialize_execution_establishes_and_releases_the_context(self) -> None:
        # Without initialize(), QAIRT rebuilds the backend and inferencer on
        # every call; measured on SM8750 that was 3990 ms versus 2492 ms.
        adapter = self._adapter()
        model = FakeModel("decoder_ar1")

        owner = adapter.initialize_execution(model, device="android")
        self.assertIs(owner, model)
        self.assertTrue(hasattr(model, "_inference_handle"))

        adapter.release_execution(owner)
        self.assertFalse(hasattr(model, "_inference_handle"))

    def test_initialize_execution_fails_closed_without_the_sdk_method(self) -> None:
        adapter = self._adapter()

        with self.assertRaises(QairtConfigurationError):
            adapter.initialize_execution(SimpleNamespace(), device=None)

    def test_create_device_uses_qairt_248_remote_android_identifier(self) -> None:
        class RemoteDeviceIdentifier:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        class Device:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        self.fake_qairt.RemoteDeviceIdentifier = RemoteDeviceIdentifier
        self.fake_qairt.Device = Device
        self.fake_qairt.DevicePlatformType = SimpleNamespace(ANDROID="android")

        device = self._adapter().create_device(
            serial="ABC123",
            server="host.docker.internal:5037",
        )

        self.assertEqual(device.kwargs["type"], "android")
        self.assertEqual(
            device.kwargs["identifier"].kwargs,
            {
                "serial_id": "ABC123",
                "hostname": "host.docker.internal",
                "port": 5037,
            },
        )

    def test_ar_and_transform_use_lazy_public_python_apis(self) -> None:
        calls: dict[str, Any] = {}

        class ExportableContext:
            def __init__(self, label: str) -> None:
                self.label = label

            def export(self, directory: Path, prefix: str):
                return SimpleNamespace(
                    onnx_path=Path(directory) / f"{prefix}.onnx",
                    data_path=Path(directory) / f"{prefix}.data",
                    encodings_path=Path(directory) / f"{prefix}.encodings",
                )

        class GraphContext:
            @classmethod
            def from_files(cls, model: str, encodings: str | None):
                calls["from_files"] = (model, encodings)
                return ExportableContext("base")

        def change(context: Any, ar: int, cl: int):
            calls["change"] = (context, ar, cl)

        class M2sStartPoint:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        optimizer = SimpleNamespace(
            GraphContext=GraphContext,
            change_seq_and_context_length=change,
            M2sStartPoint=M2sStartPoint,
        )

        class SplitModelConfig:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class MhaConfig:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        config_module = SimpleNamespace(
            SplitModelConfig=SplitModelConfig,
            MhaConfig=MhaConfig,
            QuantizationStage=SimpleNamespace(POST_QUANT="post"),
        )
        common_module = SimpleNamespace(BackendType=SimpleNamespace(HTP="htp"))

        def transform(model: str, **kwargs: Any):
            calls["transform"] = (model, kwargs)
            count = kwargs["split_model"].num_splits
            return [ExportableContext(str(index)) for index in range(count)]

        adapter = self._adapter(
            {
                "qairt.optimizer.onnx": optimizer,
                "qairt.api.transforms._transform": SimpleNamespace(transform=transform),
                "qairt.api.transforms.model_transformer_config": config_module,
                "qairt.api.configs.common": common_module,
                "qairt.gen_ai_api.builders.qwen.builder": _fake_qwen35_builder_module(),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            variant = adapter.ar_convert(
                "base.onnx",
                ar=1,
                context_length=4096,
                output_dir=Path(directory) / "variant",
                encodings_path="base.encodings",
                family=FamilyId.QWEN3_5,
                allow_experimental_qwen35=True,
            )
            slices = adapter.transform(
                variant,
                split_plan=build_split_plan(4, decoder_slices=2),
                family=FamilyId.QWEN3_5,
                output_dir=Path(directory) / "split",
                native_kv=True,
            )
        self.assertEqual(calls["change"][1:], (1, 4096))
        self.assertEqual(len(slices), 4)
        self.assertEqual(
            variant.external_data_paths,
            (variant.model_path.with_suffix(".data"),),
        )
        self.assertTrue(all(item.external_data_paths for item in slices))
        mha = calls["transform"][1]["mha_config"]
        self.assertTrue(mha.permute_kv_cache_io)
        # The start points are the SDK's own objects, passed straight through
        # rather than rebuilt from a copy in this repository.
        self.assertEqual(
            [item.name_pattern for item in mha.m2s_additional_start_points],
            [item[0] for item in _QWEN3_5_SDK_START_POINTS],
        )

    def _transform_with_sdk_points(self, module: Any) -> Any:
        """Run one transform against a fake SDK that owns the start points."""

        def transform(model: str, **kwargs: Any):
            return [SimpleNamespace(
                export=lambda directory, prefix: SimpleNamespace(
                    onnx_path=Path(directory) / f"{prefix}.onnx",
                    data_path=Path(directory) / f"{prefix}.data",
                    encodings_path=None,
                )
            )] * kwargs["split_model"].num_splits

        adapter = self._adapter(
            {
                "qairt.optimizer.onnx": SimpleNamespace(
                    M2sStartPoint=lambda **kwargs: SimpleNamespace(**kwargs)
                ),
                "qairt.api.transforms._transform": SimpleNamespace(transform=transform),
                "qairt.api.transforms.model_transformer_config": SimpleNamespace(
                    SplitModelConfig=lambda **kwargs: SimpleNamespace(**kwargs),
                    MhaConfig=lambda **kwargs: SimpleNamespace(**kwargs),
                    QuantizationStage=SimpleNamespace(POST_QUANT="post"),
                ),
                "qairt.api.configs.common": SimpleNamespace(
                    BackendType=SimpleNamespace(HTP="htp")
                ),
                "qairt.gen_ai_api.builders.qwen.builder": module,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "base.onnx"
            model.write_bytes(b"onnx")
            return adapter.transform(
                model,
                split_plan=build_split_plan(4, decoder_slices=2),
                family=FamilyId.QWEN3_5,
                output_dir=Path(directory) / "split",
                native_kv=True,
            )

    def test_changed_sdk_start_points_fail_closed_with_the_difference(self) -> None:
        # A silent upstream change would reslice attention heads. The profile
        # holds no copy of the values, only a fingerprint, so drift surfaces
        # here instead of in the compiled graph.
        drifted = list(_QWEN3_5_SDK_START_POINTS)
        drifted[1] = (drifted[1][0], 1, {4096: 128})

        with self.assertRaises(QairtConfigurationError) as caught:
            self._transform_with_sdk_points(_fake_qwen35_builder_module(drifted))

        message = str(caught.exception)
        self.assertIn("MHA2SHA start points for this family changed", message)
        self.assertIn("reviewed", message)
        # The new values are named, so a reviewer sees what moved.
        self.assertIn("4096: 128", message)

    def test_missing_sdk_start_points_are_never_guessed(self) -> None:
        with self.assertRaises(QairtConfigurationError) as caught:
            self._transform_with_sdk_points(SimpleNamespace())

        self.assertIn("no longer exposes", str(caught.exception))

    def test_adapter_source_contains_no_subprocess_path(self) -> None:
        import qairt_agent.qairt_adapter.adapter as adapter_module

        source = Path(adapter_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("subprocess.", source)

    def test_native_workflow_config_uses_qairt_python_types(self) -> None:
        class Role:
            IMAGE_ENCODER = "image"
            TEXT_GENERATOR = "text"

        class Node:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class Graph:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        module = SimpleNamespace(
            WorkflowNodeRole=Role,
            WorkflowNode=Node,
            WorkflowGraph=Graph,
        )
        adapter = self._adapter({"qairt.gen_ai_api.configs.workflow": module})
        graph = adapter.create_qwen3_vl_workflow_config(
            vision_path="vision.onnx",
            text_path="text.onnx",
        )
        self.assertEqual(graph.connections, (("imageEncoder", "textGenerator"),))
        self.assertEqual(graph.nodes[0].role, "image")

    def _quantizer_module(self, captured: list[Any]) -> Any:
        class QuantizerInputConfig:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)
                captured.append(kwargs)

        class QAIRTQuantizer:
            @staticmethod
            def quantize(config: Any) -> Any:
                # Mirrors the SDK: encoding_json is populated only when the
                # caller asked for the dump.
                encoding_json = (
                    Path(config.output_dlc).with_name(
                        Path(config.output_dlc).stem + "_encoding.json"
                    )
                    if getattr(config, "dump_encoding_json", False)
                    else None
                )
                return SimpleNamespace(
                    dlc_output=config.output_dlc,
                    encoding_json=encoding_json,
                )

        return SimpleNamespace(
            QuantizerInputConfig=QuantizerInputConfig,
            QAIRTQuantizer=QAIRTQuantizer,
        )

    def test_standalone_quantizer_dumps_encodings_by_default(self) -> None:
        captured: list[Any] = []
        adapter = self._adapter(
            {
                "qti.aisw.tools.core.modules.converter.quantizer_module": (
                    self._quantizer_module(captured)
                )
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = adapter.quantize(
                root / "in.dlc",
                output_dlc=root / "out.dlc",
                input_list=root / "inputs.txt",
            )

        self.assertTrue(captured[0]["dump_encoding_json"])
        self.assertIsNotNone(artifact.encodings_path)
        assert artifact.encodings_path is not None
        self.assertEqual(artifact.encodings_path.name, "out_encoding.json")

    def test_standalone_quantizer_honours_an_explicit_dump_override(self) -> None:
        captured: list[Any] = []
        adapter = self._adapter(
            {
                "qti.aisw.tools.core.modules.converter.quantizer_module": (
                    self._quantizer_module(captured)
                )
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = adapter.quantize(
                root / "in.dlc",
                output_dlc=root / "out.dlc",
                input_list=root / "inputs.txt",
                dump_encoding_json=False,
            )

        self.assertFalse(captured[0]["dump_encoding_json"])
        self.assertIsNone(artifact.encodings_path)


class GenAIBuilderPackagingTests(unittest.TestCase):
    def _adapter(
        self,
        builder_type: type[RecordingGenAIBuilder],
    ) -> tuple[
        QairtSdkAdapter,
        RecordingGenAIBuilder,
        list[tuple[str, Any]],
        list[str],
        Any,
    ]:
        calls: list[tuple[str, Any]] = []
        imports: list[str] = []
        builder = builder_type(calls)
        backend_htp = object()

        class Factory:
            @classmethod
            def create(
                cls,
                model_path: str,
                backend: Any,
                **kwargs: Any,
            ) -> RecordingGenAIBuilder:
                calls.append(
                    (
                        "factory.create",
                        {
                            "model_path": model_path,
                            "backend": backend,
                            "kwargs": kwargs,
                        },
                    )
                )
                return builder

        modules = {
            "qairt.gen_ai_api.gen_ai_builder_factory": SimpleNamespace(
                GenAIBuilderFactory=Factory
            ),
            "qairt.api.configs.common": SimpleNamespace(
                BackendType=SimpleNamespace(HTP=backend_htp)
            ),
        }

        class PinnedQwen3_5BuilderHTP:
            @classmethod
            def from_pretrained(
                cls,
                model_path: str,
                cache_root: str | None,
                **kwargs: Any,
            ) -> RecordingGenAIBuilder:
                calls.append(
                    (
                        "qwen35.from_pretrained",
                        {
                            "model_path": model_path,
                            "cache_root": cache_root,
                            "kwargs": kwargs,
                        },
                    )
                )
                return builder

        modules["qairt.gen_ai_api.builders.qwen.builder"] = SimpleNamespace(
            Qwen3_5BuilderHTP=PinnedQwen3_5BuilderHTP
        )

        def loader(name: str) -> Any:
            imports.append(name)
            if name not in modules:
                raise ModuleNotFoundError(name)
            return modules[name]

        return (
            QairtSdkAdapter(
                module_loader=loader,
                require_successful_preflight=False,
            ),
            builder,
            calls,
            imports,
            backend_htp,
        )

    @staticmethod
    def _file(root: Path, name: str) -> Path:
        path = root / name
        path.write_bytes(b"test")
        return path

    def test_qwen3_dense_uses_lazy_factory_and_records_generic_capability(
        self,
    ) -> None:
        adapter, builder, calls, imports, backend_htp = self._adapter(
            GenAIBuilderHTP
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self._file(root, "qwen3.onnx")
            encodings = self._file(root, "qwen3.encodings")
            tokenizer = self._file(root, "tokenizer.json")
            config = self._file(root, "config.json")
            destination = root / "container"
            split_plan = build_split_plan(8, decoder_slices=2)

            result = adapter.build_genai_container(
                model,
                output_dir=destination,
                family=FamilyId.QWEN3_DENSE,
                split_plan=split_plan,
                encodings_path=encodings,
                tokenizer_path=tokenizer,
                config_path=config,
                cache_root=root / "cache",
                ar_values=(1, 128),
                context_lengths=(2048, 4096),
                native_kv=True,
                weight_sharing=True,
            )
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

        self.assertIsInstance(result, GenAIContainerBuildResult)
        self.assertEqual(
            imports,
            [
                "qairt.gen_ai_api.gen_ai_builder_factory",
                "qairt.api.configs.common",
            ],
        )
        factory_call = calls[0][1]
        self.assertEqual(factory_call["model_path"], str(model))
        self.assertIs(factory_call["backend"], backend_htp)
        self.assertEqual(factory_call["kwargs"]["tokenizer_path"], str(tokenizer))
        self.assertEqual(factory_call["kwargs"]["config_path"], str(config))
        self.assertEqual(builder.encodings_path, str(encodings))
        self.assertIn(
            (
                "set_targets",
                ["chipset:SM8850;dsp_arch:v81;soc_model:87"],
            ),
            calls,
        )
        transform_options = next(
            value
            for name, value in calls
            if name == "set_transformation_options"
        )
        self.assertEqual(transform_options["arn"], [1, 128])
        self.assertEqual(transform_options["context_length"], [2048, 4096])
        self.assertEqual(transform_options["split.num_splits"], 4)
        self.assertTrue(transform_options["split.split_embedding"])
        self.assertTrue(transform_options["split.split_lm_head"])
        self.assertTrue(transform_options["mha2sha.permute_kv_cache_io"])
        self.assertTrue(builder.native_kv)
        self.assertTrue(builder.weight_sharing)
        self.assertEqual(result.factory_support, "generic_fallback")
        self.assertEqual(
            result.compatibility_mode,
            "generic_fallback_requires_device_validation",
        )
        self.assertEqual(
            metadata["capability"]["compatibility_mode"],
            "generic_fallback_requires_device_validation",
        )
        self.assertIn("build", [name for name, _value in calls])
        self.assertIn("save", [name for name, _value in calls])

    def test_qwen35_requires_attached_model_and_encodings_for_every_ar(
        self,
    ) -> None:
        adapter, _builder, _calls, imports, _backend = self._adapter(
            Qwen3_5BuilderHTP
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self._file(root, "qwen35.onnx")
            encodings = self._file(root, "qwen35.encodings")
            with self.assertRaises(ExperimentalFeatureError):
                adapter.build_genai_container(
                    model,
                    output_dir=root / "container",
                    family=FamilyId.QWEN3_5,
                    split_plan=build_split_plan(8),
                    encodings_path=encodings,
                    attached_models_by_ar={
                        1: {
                            "model_path": model,
                            "encodings_path": encodings,
                        }
                    },
                )
        self.assertEqual(imports, [])

    def test_qwen35_attaches_prefill_and_decode_before_weight_sharing(
        self,
    ) -> None:
        adapter, builder, calls, imports, _backend = self._adapter(
            Qwen3_5BuilderHTP
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self._file(root, "qwen35_base.onnx")
            base_encodings = self._file(root, "qwen35_base.encodings")
            decode = self._file(root, "qwen35_decode_ar1.onnx")
            decode_encodings = self._file(root, "qwen35_decode_ar1.encodings")
            prefill = self._file(root, "qwen35_prefill_ar128.onnx")
            prefill_encodings = self._file(
                root,
                "qwen35_prefill_ar128.encodings",
            )
            tokenizer = self._file(root, "tokenizer.json")
            config_path = root / "config.json"
            original_config = {
                "architectures": [
                    "Qwen3_5OmniThinkerForConditionalGeneration"
                ],
                "model_type": "qwen3_5_omni_thinker",
            }
            config_path.write_text(
                json.dumps(original_config),
                encoding="utf-8",
            )

            result = adapter.build_genai_container(
                base,
                output_dir=root / "container",
                family=FamilyId.QWEN3_5,
                split_plan=build_split_plan(8, decoder_slices=2),
                encodings_path=base_encodings,
                tokenizer_path=tokenizer,
                config_path=config_path,
                cache_root=root / "cache",
                attached_models_by_ar={
                    "1": GenAIAttachedModel(decode, decode_encodings),
                    "128": {
                        "model_path": prefill,
                        "encodings_path": prefill_encodings,
                    },
                },
                ar_values=(1, 128),
                context_lengths=(4096,),
                native_kv=True,
                weight_sharing=True,
            )
            config_after = json.loads(config_path.read_text(encoding="utf-8"))
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

        event_names = [name for name, _value in calls]
        self.assertNotIn(
            "qairt.gen_ai_api.gen_ai_builder_factory",
            imports,
        )
        self.assertIn(
            "qairt.gen_ai_api.builders.qwen.builder",
            imports,
        )
        constructor = next(
            value
            for name, value in calls
            if name == "qwen35.from_pretrained"
        )
        self.assertEqual(constructor["model_path"], str(base))
        self.assertEqual(constructor["cache_root"], str(root / "cache"))
        self.assertEqual(
            constructor["kwargs"]["tokenizer_path"],
            str(tokenizer),
        )
        self.assertEqual(
            constructor["kwargs"]["config_path"],
            str(config_path),
        )
        self.assertIsNone(constructor["kwargs"]["config_dict"])
        self.assertEqual(config_after, original_config)
        self.assertEqual(builder.skip_ar_conversion, False)
        self.assertLess(
            event_names.index("skip_ar_conversion"),
            event_names.index("set_transformation_options"),
        )
        attach_indices = [
            index
            for index, name in enumerate(event_names)
            if name == "attach_model_for_arn"
        ]
        self.assertEqual(len(attach_indices), 2)
        self.assertLess(max(attach_indices), event_names.index("weight_sharing"))
        attached = [
            value for name, value in calls if name == "attach_model_for_arn"
        ]
        self.assertEqual(
            attached,
            [
                (1, str(decode), str(decode_encodings)),
                (128, str(prefill), str(prefill_encodings)),
            ],
        )
        self.assertEqual(result.attached_ar_values, (1, 128))
        self.assertEqual(result.compatibility_mode, "explicit_family_builder")
        self.assertEqual(
            metadata["capability"]["compatibility_mode"],
            "explicit_family_builder",
        )
        self.assertIsNone(metadata["python_api"]["factory"])
        self.assertEqual(
            metadata["python_api"]["builder_constructor"],
            "qairt.gen_ai_api.builders.qwen.builder."
            "Qwen3_5BuilderHTP.from_pretrained",
        )
        self.assertTrue(
            any("single-source" in note for note in result.compatibility_notes)
        )

    def test_qwen3_moe_uses_explicit_factory_capability(self) -> None:
        adapter, _builder, _calls, _imports, _backend = self._adapter(
            Qwen3MoeBuilderHTP
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self._file(root, "qwen3_moe.onnx")
            encodings = self._file(root, "qwen3_moe.encodings")
            result = adapter.build_genai_container(
                model,
                output_dir=root / "container",
                family=FamilyId.QWEN3_MOE,
                split_plan=build_split_plan(4),
                encodings_path=encodings,
                ar_values=(1,),
                context_lengths=(4096,),
                native_kv=False,
                weight_sharing=False,
            )

        self.assertEqual(result.factory_support, "explicit")
        self.assertEqual(result.compatibility_mode, "explicit_factory")
        self.assertTrue(result.runtime_supported)

    def test_explicit_family_rejects_unexpected_factory_dispatch(self) -> None:
        adapter, _builder, calls, _imports, _backend = self._adapter(
            GenAIBuilderHTP
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self._file(root, "qwen3_moe.onnx")
            encodings = self._file(root, "qwen3_moe.encodings")
            with self.assertRaisesRegex(
                QairtConfigurationError,
                "expected Qwen3MoeBuilderHTP",
            ):
                adapter.build_genai_container(
                    model,
                    output_dir=root / "container",
                    family=FamilyId.QWEN3_MOE,
                    split_plan=build_split_plan(4),
                    encodings_path=encodings,
                    ar_values=(1,),
                    context_lengths=(4096,),
                    native_kv=False,
                    weight_sharing=False,
                )

        self.assertEqual(
            [name for name, _value in calls],
            ["factory.create"],
        )

    def test_qwen35_omni_packages_pinned_audio_and_text_builders(self) -> None:
        calls: list[tuple[str, Any]] = []

        class Qwen3OmniAudioEncoderBuilderHTP(RecordingGenAIBuilder):
            def __init__(self, events: list[tuple[str, Any]]) -> None:
                super().__init__(events)
                self.config = SimpleNamespace(
                    audio_start_token_id=1,
                    audio_end_token_id=2,
                )

        class PinnedQwen3_5BuilderHTP(RecordingGenAIBuilder):
            @classmethod
            def from_pretrained(cls, model_path, cache_root, **kwargs):
                calls.append(
                    (
                        "text.from_pretrained",
                        (model_path, cache_root, kwargs),
                    )
                )
                return cls(calls)

        # The adapter validates the exact public class name.
        PinnedQwen3_5BuilderHTP.__name__ = "Qwen3_5BuilderHTP"
        audio_builder = Qwen3OmniAudioEncoderBuilderHTP(calls)

        class Factory:
            @classmethod
            def create_audio_encoder(cls, model_path, **kwargs):
                calls.append(("factory.create_audio_encoder", (model_path, kwargs)))
                return audio_builder

        class Role:
            AUDIO_ENCODER = "audio"
            TEXT_GENERATOR = "text"

        class Node:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class Graph:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class WorkflowContainer:
            def save(self, destination, *, exist_ok=False):
                root = Path(destination)
                for component in ("audioEncoder", "textGenerator"):
                    (root / component).mkdir(parents=True, exist_ok=True)
                    (root / component / "model.bin").write_bytes(b"context")
                (root / "metadata.json").write_text("{}", encoding="utf-8")

        class WorkflowBuilder:
            @classmethod
            def from_builders(cls, builders, graph):
                calls.append(("workflow.from_builders", (builders, graph)))
                return cls()

            def build(self):
                calls.append(("workflow.build", None))
                return WorkflowContainer()

        modules = {
            "qairt.gen_ai_api.gen_ai_builder_factory": SimpleNamespace(
                GenAIBuilderFactory=Factory
            ),
            "qairt.gen_ai_api.builders.qwen.builder": SimpleNamespace(
                Qwen3_5BuilderHTP=PinnedQwen3_5BuilderHTP
            ),
            "qairt.gen_ai_api.configs.workflow": SimpleNamespace(
                WorkflowNodeRole=Role,
                WorkflowNode=Node,
                WorkflowGraph=Graph,
            ),
            "qairt.gen_ai_api.builders.workflow_builder": SimpleNamespace(
                WorkflowBuilder=WorkflowBuilder
            ),
        }
        adapter = QairtSdkAdapter(
            module_loader=lambda name: modules[name],
            require_successful_preflight=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                name: self._file(root, name)
                for name in (
                    "text.onnx",
                    "audio.onnx",
                    "text.encodings",
                    "audio.encodings",
                    "text-config.json",
                    "audio-config.json",
                    "tokenizer.json",
                    "ar1.onnx",
                    "ar1.encodings",
                    "ar128.onnx",
                    "ar128.encodings",
                )
            }
            result = adapter.build_qwen35_omni_components(
                paths["text.onnx"],
                audio_model_path=paths["audio.onnx"],
                output_dir=root / "container",
                split_plan=build_split_plan(8, decoder_slices=2),
                text_encodings_path=paths["text.encodings"],
                audio_encodings_path=paths["audio.encodings"],
                text_config_path=paths["text-config.json"],
                audio_config_path=paths["audio-config.json"],
                tokenizer_path=paths["tokenizer.json"],
                attached_models_by_ar={
                    1: {
                        "model_path": paths["ar1.onnx"],
                        "encodings_path": paths["ar1.encodings"],
                    },
                    128: {
                        "model_path": paths["ar128.onnx"],
                        "encodings_path": paths["ar128.encodings"],
                    },
                },
            )
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(
            result.audio_builder_class,
            "Qwen3OmniAudioEncoderBuilderHTP",
        )
        self.assertEqual(result.builder_class, "Qwen3_5BuilderHTP")
        self.assertEqual(result.container_class, "WorkflowContainer")
        self.assertFalse(result.runtime_supported)
        self.assertEqual(
            metadata["workflow"]["connections"],
            [["audioEncoder", "textGenerator"]],
        )
        self.assertIsNone(metadata["python_api"]["text_factory"])
        self.assertEqual(
            metadata["python_api"]["text_builder_constructor"],
            "qairt.gen_ai_api.builders.qwen.builder."
            "Qwen3_5BuilderHTP.from_pretrained",
        )
        self.assertIn("workflow.from_builders", [name for name, _ in calls])
        self.assertNotIn("factory.create", [name for name, _ in calls])
        audio_call = next(
            value for name, value in calls if name == "factory.create_audio_encoder"
        )
        text_call = next(
            value for name, value in calls if name == "text.from_pretrained"
        )
        self.assertEqual(
            audio_call[1]["config_path"],
            str(paths["audio-config.json"]),
        )
        self.assertEqual(
            text_call[2]["config_path"],
            str(paths["text-config.json"]),
        )

    def test_qwen35_omni_rejects_multiple_ars_without_weight_sharing(
        self,
    ) -> None:
        adapter = QairtSdkAdapter(
            module_loader=lambda name: None,
            require_successful_preflight=False,
        )

        with self.assertRaisesRegex(
            QairtConfigurationError,
            "multiple ARs require weight_sharing=True",
        ):
            adapter.build_qwen35_omni_components(
                "text.onnx",
                audio_model_path="audio.onnx",
                output_dir="container",
                split_plan=build_split_plan(8, decoder_slices=2),
                text_encodings_path="text.encodings",
                audio_encodings_path="audio.encodings",
                text_config_path="text-config.json",
                audio_config_path="audio-config.json",
                ar_values=(1, 128),
                weight_sharing=False,
            )

    def test_qwen3_vl_builds_workflow_but_records_runtime_unsupported(
        self,
    ) -> None:
        calls: list[tuple[str, Any]] = []
        imports: list[str] = []
        text_builder = GenAIBuilderHTP(calls)
        backend_htp = object()
        vision_builders: list[RecordingGenAIBuilder] = []

        class Factory:
            @classmethod
            def create(
                cls,
                model_path: str,
                backend: Any,
                **kwargs: Any,
            ) -> RecordingGenAIBuilder:
                calls.append(
                    (
                        "factory.create",
                        (model_path, backend, kwargs),
                    )
                )
                return text_builder

        class VisionEncoderBuilderHTP(RecordingGenAIBuilder):
            @classmethod
            def from_pretrained(
                cls,
                model_path: str,
                *,
                cache_root: str | None,
                config_path: str | None,
            ) -> "VisionEncoderBuilderHTP":
                calls.append(
                    (
                        "vision.from_pretrained",
                        (model_path, cache_root, config_path),
                    )
                )
                builder = cls(calls)
                vision_builders.append(builder)
                return builder

        class Role:
            IMAGE_ENCODER = "image"
            TEXT_GENERATOR = "text"

        class Node:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class Graph:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        class WorkflowContainer(FakeLLMContainer):
            def get_executor(self) -> Any:
                raise AssertionError("unsupported runtime must not be invoked")

        class WorkflowBuilder:
            def __init__(self, builders: dict[str, Any], graph: Any) -> None:
                self.builders = builders
                self.graph = graph

            @classmethod
            def from_builders(
                cls,
                builders: dict[str, Any],
                workflow_graph: Any,
            ) -> "WorkflowBuilder":
                calls.append(
                    (
                        "workflow.from_builders",
                        (tuple(builders), workflow_graph),
                    )
                )
                return cls(builders, workflow_graph)

            def build(self) -> WorkflowContainer:
                calls.append(("workflow.build", None))
                for builder in self.builders.values():
                    builder.build()
                return WorkflowContainer(calls)

        modules = {
            "qairt.gen_ai_api.gen_ai_builder_factory": SimpleNamespace(
                GenAIBuilderFactory=Factory
            ),
            "qairt.api.configs.common": SimpleNamespace(
                BackendType=SimpleNamespace(HTP=backend_htp)
            ),
            "qairt.gen_ai_api.builders.vision_encoder_builder_htp": (
                SimpleNamespace(
                    VisionEncoderBuilderHTP=VisionEncoderBuilderHTP
                )
            ),
            "qairt.gen_ai_api.builders.workflow_builder": SimpleNamespace(
                WorkflowBuilder=WorkflowBuilder
            ),
            "qairt.gen_ai_api.configs.workflow": SimpleNamespace(
                WorkflowNodeRole=Role,
                WorkflowNode=Node,
                WorkflowGraph=Graph,
            ),
        }

        def loader(name: str) -> Any:
            imports.append(name)
            return modules[name]

        adapter = QairtSdkAdapter(
            module_loader=loader,
            require_successful_preflight=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_model = self._file(root, "qwen3_vl_text.onnx")
            text_encodings = self._file(
                root,
                "qwen3_vl_text.encodings",
            )
            text_config = self._file(root, "text_config.json")
            tokenizer = self._file(root, "tokenizer.json")
            vision_model = self._file(
                root,
                "qwen3_vl_vision_projector.onnx",
            )
            vision_encodings = self._file(
                root,
                "qwen3_vl_vision_projector.encodings",
            )
            vision_config = self._file(root, "vision_config.json")
            result = adapter.build_genai_container(
                text_model,
                output_dir=root / "workflow_container",
                family=FamilyId.QWEN3_VL,
                split_plan=build_split_plan(8, decoder_slices=2),
                encodings_path=text_encodings,
                vision_model_path=vision_model,
                vision_encodings_path=vision_encodings,
                vision_config_path=vision_config,
                tokenizer_path=tokenizer,
                config_path=text_config,
            )
            metadata = json.loads(
                result.metadata_path.read_text(encoding="utf-8")
            )

        self.assertEqual(
            imports,
            [
                "qairt.gen_ai_api.gen_ai_builder_factory",
                "qairt.api.configs.common",
                "qairt.gen_ai_api.builders.vision_encoder_builder_htp",
                "qairt.gen_ai_api.builders.workflow_builder",
                "qairt.gen_ai_api.configs.workflow",
            ],
        )
        self.assertEqual(len(vision_builders), 1)
        vision_builder = vision_builders[0]
        self.assertEqual(
            vision_builder.encodings_path,
            str(vision_encodings),
        )
        target_events = [
            value for name, value in calls if name == "set_targets"
        ]
        self.assertEqual(
            target_events,
            [
                ["chipset:SM8850;dsp_arch:v81;soc_model:87"],
                ["chipset:SM8850;dsp_arch:v81;soc_model:87"],
            ],
        )
        workflow_call = next(
            value for name, value in calls if name == "workflow.from_builders"
        )
        self.assertEqual(
            workflow_call[0],
            ("imageEncoder", "textGenerator"),
        )
        self.assertEqual(
            workflow_call[1].connections,
            (("imageEncoder", "textGenerator"),),
        )
        self.assertFalse(result.runtime_supported)
        self.assertEqual(
            result.vision_builder_class,
            "VisionEncoderBuilderHTP",
        )
        self.assertEqual(metadata["capability"]["runtime_supported"], False)
        self.assertEqual(
            metadata["workflow"]["projector_location"],
            "inside_vision_onnx",
        )
        self.assertNotIn("get_executor", [name for name, _value in calls])


class ComposedBuildTests(unittest.TestCase):
    class Inspector:
        def inspect(self, path: Path | str):
            hidden = 16
            return SimpleNamespace(
                inputs=(
                    SimpleNamespace(name="input", shape=(1, 1), dtype="FLOAT"),
                ),
                outputs=(
                    SimpleNamespace(name="hidden", shape=(1, 4, hidden), dtype="FLOAT"),
                ),
            )

    class RecordingAdapter(QairtSdkAdapter):
        def __init__(self) -> None:
            super().__init__(
                require_successful_preflight=False,
                onnx_inspector=ComposedBuildTests.Inspector(),
            )
            self.converted_slice_names: list[str] = []

        def preflight(self, spec: Any) -> PreflightReport:
            return PreflightReport(
                issues=(),
                sdk_root=Path("/sdk"),
                sdk_version="2.49.0",
                sdk_build_id="260730134355",
                target_soc="SM8750",
                dsp_arch="v79",
                soc_model=69,
            )

        def ar_convert(self, model_path: str | Path, **kwargs: Any) -> ModelVariantArtifact:
            ar = kwargs["ar"]
            cl = kwargs["context_length"]
            return ModelVariantArtifact(
                model_path=Path(f"text_ar{ar}_cl{cl}.onnx"),
                encodings_path=None,
                ar=ar,
                context_length=cl,
                source_kind="derived",
                family=str(_enum_value_for_test(kwargs["family"].family)),
            )

        def transform(
            self,
            variant_or_path: ModelVariantArtifact,
            *,
            split_plan: Any,
            **kwargs: Any,
        ) -> tuple[TransformedSliceArtifact, ...]:
            return tuple(
                TransformedSliceArtifact(
                    slice_name=item.name,
                    split_index=item.index,
                    model_path=Path(f"{item.name}_ar{variant_or_path.ar}.onnx"),
                    encodings_path=None,
                    ar=variant_or_path.ar,
                    context_length=variant_or_path.context_length,
                )
                for item in split_plan.slices
            )

        def convert(
            self,
            slice_or_path: TransformedSliceArtifact,
            **kwargs: Any,
        ) -> ConvertedModelArtifact:
            self.converted_slice_names.append(slice_or_path.slice_name)
            return ConvertedModelArtifact(
                model_path=Path(kwargs["output_path"]),
                source_model_path=slice_or_path.model_path,
                quantization_mode="float",
                slice_name=slice_or_path.slice_name,
                ar=slice_or_path.ar,
                context_length=slice_or_path.context_length,
                sdk_model=FakeModel(slice_or_path.model_path.stem),
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
            **kwargs: Any,
        ) -> CompiledContextArtifact:
            return CompiledContextArtifact(
                context_binary_path=Path(output_path),
                slice_name=slice_name,
                graph_names=tuple(graph_names),
                ar_values=tuple(ar_values),
                target_soc=target_soc,
                dsp_arch=dsp_arch,
                soc_model=soc_model,
                weight_sharing=weight_sharing,
                native_kv_config_path=None,
            )

        def create_qwen3_vl_workflow_config(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "nodes": ["imageEncoder", "textGenerator"],
                "connections": [["imageEncoder", "textGenerator"]],
            }

    def _spec(self, embedding_mode: str, *, vision: bool = False) -> dict[str, Any]:
        sources: dict[str, Any] = {
            "text": {"onnx_path": "text.onnx"},
        }
        if vision:
            sources.update(
                {
                    "vision": {"onnx_path": "vision_projector.onnx"},
                    "vision_projector_location": "inside_vision_onnx",
                }
            )
        return {
            "sources": sources,
            "sequence": {
                "ars": (1, 128),
                "context_lengths": (4096,),
                "weight_sharing": True,
                "native_kv": False,
            },
            "split": {
                "embedding_mode": embedding_mode,
                "split_lm_head": True,
            },
            "quantization": {"mode": "float"},
            "target": {
                "chipset": "SM8750",
                "dsp_arch": "v79",
                "soc_model": 69,
            },
        }

    def test_lut_embedding_is_extracted_but_never_converted_or_compiled(self) -> None:
        adapter = self.RecordingAdapter()
        effective = {
            "profile": get_family_profile_for_test(FamilyId.QWEN3_DENSE),
            "split_plan": build_split_plan(2),
            "hidden_size": 16,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = adapter.build(
                self._spec("lut"),
                effective,
                directory,
            )
        self.assertNotIn("embedding", adapter.converted_slice_names)
        self.assertNotIn("embedding", {item.slice_name for item in result.contexts})
        self.assertTrue(
            any("embedding_lut" in path.name for path in result.config_artifact_paths)
        )

    def test_qwen3_vl_adds_vision_projector_context_and_workflow_artifact(self) -> None:
        adapter = self.RecordingAdapter()
        effective = {
            "profile": get_family_profile_for_test(FamilyId.QWEN3_VL),
            "split_plan": build_split_plan(2),
            "hidden_size": 16,
        }
        with tempfile.TemporaryDirectory() as directory:
            result: BuildResult = adapter.build(
                self._spec("external", vision=True),
                effective,
                directory,
            )
        self.assertIn("vision_projector", {item.slice_name for item in result.contexts})
        self.assertTrue(
            any("image_t2t_workflow" in path.name for path in result.config_artifact_paths)
        )

    def test_diagnostic_outputs_never_modify_production_contexts(self) -> None:
        class DiagnosticAdapter(self.RecordingAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.compile_options: list[dict[str, Any]] = []

            def compile_context(self, *args: Any, **kwargs: Any) -> CompiledContextArtifact:
                self.compile_options.append(dict(kwargs.get("compile_config_options", {})))
                return super().compile_context(*args, **kwargs)

        adapter = DiagnosticAdapter()
        effective = {
            "profile": get_family_profile_for_test(FamilyId.QWEN3_DENSE),
            "split_plan": build_split_plan(2),
            "hidden_size": 16,
        }
        spec = self._spec("lut")
        spec["compile"] = {"enable_intermediate_outputs": True}
        with tempfile.TemporaryDirectory() as directory:
            result = adapter.build(spec, effective, directory)

        self.assertEqual(len(result.contexts), 2)
        self.assertEqual(len(result.diagnostic_contexts), 2)
        self.assertEqual(adapter.compile_options[0], {})
        self.assertEqual(
            adapter.compile_options[1],
            {"enable_intermediate_outputs": True},
        )

    def test_qwen35_composed_build_fails_before_derivation_without_runner(self) -> None:
        adapter = self.RecordingAdapter()
        effective = {
            "profile": get_family_profile_for_test(FamilyId.QWEN3_5),
            "split_plan": build_split_plan(2),
            "hidden_size": 16,
        }
        spec = self._spec("compiled")
        spec["sequence"]["qwen35_experimental_auto_ar"] = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ExperimentalFeatureError):
                adapter.build(spec, effective, directory)
        self.assertEqual(adapter.converted_slice_names, [])


def _enum_value_for_test(value: Any) -> Any:
    return getattr(value, "value", value)


def get_family_profile_for_test(family: FamilyId):
    from qairt_agent.families import get_family_profile

    return get_family_profile(family)


if __name__ == "__main__":
    unittest.main()
