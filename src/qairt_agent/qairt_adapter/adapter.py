"""QAIRT 2.49 Python-only SDK adapter.

This module intentionally contains no subprocess integration.  Every SDK
operation is invoked through a lazily imported Python API so Claude Code/Codex
can reason about stage inputs and retain structured provenance.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from qairt_agent.families import (
    FamilyId,
    FamilyProfile,
    OnnxInspector,
    SplitPlan,
    build_split_plan,
    get_family_profile,
    validate_weight_sharing_sources,
)

from .errors import (
    ExperimentalFeatureError,
    NativeKvConfigError,
    QairtConfigurationError,
    QairtPreflightError,
    QairtSdkImportError,
)
from .native_kv import (
    audit_native_kv_config,
    build_native_kv_config,
    require_native_kv_audit,
)
from qairt_agent.harness import (
    HarnessConstraintsError,
    TargetEntry,
    require_verified_target,
    resolve_target,
    resolve_target_tuple,
)

from .preflight import (
    ACTIVE_TARGET,
    PreflightChecker,
    require_preflight,
)
from .types import (
    BuildResult,
    CompiledContextArtifact,
    ConvertedModelArtifact,
    GenAIAttachedModel,
    GenAIContainerBuildResult,
    GenAIRawSliceArtifact,
    ModelVariantArtifact,
    NativeKvGraphExpectation,
    PreflightReport,
    ProfileResult,
    QuantizedModelArtifact,
    Qwen35ValidationEvidence,
    Qwen35DerivationValidation,
    Qwen35RuntimeValidationRequest,
    Qwen35RuntimeValidationResult,
    TransformedSliceArtifact,
)


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _nested(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        current = _read(current, key)
        if current is None:
            return None
    return current


def _first(value: Any, paths: Sequence[tuple[str, ...]], default: Any = None) -> Any:
    for path in paths:
        candidate = _nested(value, *path)
        if candidate is not None:
            return candidate
    return default


def _path_or_none(value: Any) -> Path | None:
    return Path(value) if value not in (None, "") else None


def _exported_data_paths(exported: Any) -> tuple[Path, ...]:
    value = getattr(exported, "data_path", None)
    if value is None:
        return ()
    path = Path(value)
    if path.is_dir():
        return tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    return (path,)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _artifact_value(value: Any, name: str, default: Any = None) -> Any:
    return getattr(value, name, default)


def _sdk_path(value: Any) -> Path:
    if isinstance(value, (str, Path)):
        return Path(value)
    for field in ("model_path", "dlc_path", "context_binary_path"):
        path = _artifact_value(value, field)
        if path is not None:
            return Path(path)
    raise TypeError(f"cannot resolve SDK asset path from {type(value).__name__}")


def _profile(value: FamilyProfile | FamilyId | str | None) -> FamilyProfile | None:
    if value is None:
        return None
    if isinstance(value, FamilyProfile):
        return value
    return get_family_profile(value)


def _qwen35_evidence(value: Any) -> Qwen35ValidationEvidence | None:
    if value is None or isinstance(value, Qwen35ValidationEvidence):
        return value
    raise TypeError(
        "qwen35_validation_evidence must be minted by "
        "QairtSdkAdapter.validate_qwen35_derivation"
    )


class QairtSdkAdapter:
    """Strict, injectable boundary around the supported QAIRT Python APIs."""

    def __init__(
        self,
        *,
        module_loader: Callable[[str], Any] | None = None,
        preflight_checker: PreflightChecker | None = None,
        require_successful_preflight: bool = True,
        onnx_inspector: OnnxInspector | None = None,
    ) -> None:
        self._module_loader = module_loader or importlib.import_module
        self._preflight_checker = preflight_checker or PreflightChecker()
        self._require_successful_preflight = require_successful_preflight
        self._preflight_report: PreflightReport | None = None
        self._onnx_inspector = onnx_inspector or OnnxInspector()
        self._validated_qwen35_evidence_ids: set[str] = set()

    def _load_module(self, name: str) -> Any:
        try:
            return self._module_loader(name)
        except (ImportError, ModuleNotFoundError) as error:
            raise QairtSdkImportError(
                f"cannot import required QAIRT 2.49 Python module {name!r}; "
                "run preflight and expose <sdk_root>/lib/python to Python"
            ) from error

    def _ensure_ready(self) -> None:
        if not self._require_successful_preflight:
            return
        if self._preflight_report is None:
            raise QairtPreflightError(
                "preflight(spec) must succeed before invoking a QAIRT SDK stage"
            )
        require_preflight(self._preflight_report)

    def preflight(self, spec: Any) -> PreflightReport:
        """Validate and remember the pinned host/SDK/target contract."""

        report = self._preflight_checker.check(spec)
        self._preflight_report = report
        return report

    def ar_convert(
        self,
        model_path: str | Path,
        *,
        ar: int,
        context_length: int,
        output_dir: str | Path,
        encodings_path: str | Path | None = None,
        family: FamilyProfile | FamilyId | str | None = None,
        source_kind: str = "derived",
        allow_experimental_qwen35: bool = False,
        prefix: str | None = None,
    ) -> ModelVariantArtifact:
        """Rewrite a graph's AR/CL with ``GraphContext`` and export artifacts."""

        if ar <= 0 or context_length <= 0:
            raise QairtConfigurationError("AR and context_length must be positive")
        resolved_profile = _profile(family)
        if (
            resolved_profile is not None
            and resolved_profile.family is FamilyId.QWEN3_5
            and not allow_experimental_qwen35
        ):
            raise ExperimentalFeatureError(
                "Qwen3.5 single-source automatic AR conversion is experimental; "
                "set allow_experimental_qwen35=True and retain strict validation evidence"
            )
        validate_weight_sharing_sources(
            resolved_profile or FamilyId.QWEN3_DENSE,
            [source_kind],
        )
        self._ensure_ready()

        optimizer = self._load_module("qairt.optimizer.onnx")
        context = optimizer.GraphContext.from_files(
            str(model_path),
            str(encodings_path) if encodings_path is not None else None,
        )
        optimizer.change_seq_and_context_length(context, ar, context_length)

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        exported = context.export(destination, prefix=prefix or f"model_ar{ar}_cl{context_length}")
        return ModelVariantArtifact(
            model_path=Path(exported.onnx_path),
            encodings_path=_path_or_none(getattr(exported, "encodings_path", None)),
            ar=ar,
            context_length=context_length,
            source_kind=source_kind,
            family=resolved_profile.family.value if resolved_profile else None,
            external_data_paths=_exported_data_paths(exported),
            graph_context=context,
        )

    def transform(
        self,
        variant_or_path: ModelVariantArtifact | str | Path,
        *,
        split_plan: SplitPlan,
        family: FamilyProfile | FamilyId | str,
        output_dir: str | Path,
        encodings_path: str | Path | None = None,
        mha2sha: bool = True,
        native_kv: bool = False,
        validate: bool = False,
        input_raw_list_path: str | Path | None = None,
        input_raw_base_dir: str | Path | None = None,
        m2s_head_split_map: Mapping[int, int] | None = None,
        adapt_moe: Mapping[str, Any] | None = None,
    ) -> tuple[TransformedSliceArtifact, ...]:
        """Split a model and apply MHA2SHA through QAIRT's Python transform API."""

        resolved_profile = _profile(family)
        assert resolved_profile is not None
        model_path = _sdk_path(variant_or_path)
        selected_encodings = (
            Path(encodings_path)
            if encodings_path is not None
            else _path_or_none(_artifact_value(variant_or_path, "encodings_path"))
        )
        ar = _artifact_value(variant_or_path, "ar")
        context_length = _artifact_value(variant_or_path, "context_length")

        self._ensure_ready()
        transform_module = self._load_module("qairt.api.transforms._transform")
        config_module = self._load_module("qairt.api.transforms.model_transformer_config")
        common_module = self._load_module("qairt.api.configs.common")
        optimizer = self._load_module("qairt.optimizer.onnx")

        start_points = [
            optimizer.M2sStartPoint(
                name_pattern=item.output_name_regex,
                split_axis=item.axis,
                split_map=dict(item.split_map) if item.split_map is not None else None,
            )
            for item in resolved_profile.mha_start_points
        ]
        validation_kwargs: dict[str, str] = {}
        if input_raw_list_path is not None:
            validation_kwargs["input_raw_list_path"] = str(input_raw_list_path)
        if input_raw_base_dir is not None:
            validation_kwargs["input_raw_base_dir"] = str(input_raw_base_dir)

        split_config = config_module.SplitModelConfig(**split_plan.to_qairt_kwargs())
        transform_kwargs: dict[str, Any] = {
            "split_model": split_config,
        }
        if validate and not mha2sha:
            raise QairtConfigurationError("MHA2SHA validation requires mha2sha=True")
        if mha2sha:
            transform_kwargs["mha_config"] = config_module.MhaConfig(
                permute_kv_cache_io=native_kv,
                m2s_head_split_map=(
                    dict(m2s_head_split_map)
                    if m2s_head_split_map is not None
                    else None
                ),
                m2s_additional_start_points=start_points or None,
                enable_validation=validate,
                validation_kwargs=validation_kwargs or None,
            )
        if resolved_profile.is_moe:
            transform_kwargs["adapt_moe"] = dict(adapt_moe or {})
        elif adapt_moe is not None:
            raise QairtConfigurationError("adapt_moe is only valid for a MoE family profile")

        contexts = transform_module.transform(
            str(model_path),
            backend=common_module.BackendType.HTP,
            quantization_stage=config_module.QuantizationStage.POST_QUANT,
            encodings=str(selected_encodings) if selected_encodings is not None else None,
            **transform_kwargs,
        )
        if len(contexts) != split_plan.num_splits:
            raise QairtConfigurationError(
                f"QAIRT returned {len(contexts)} splits, expected {split_plan.num_splits}"
            )

        destination = Path(output_dir)
        artifacts: list[TransformedSliceArtifact] = []
        for slice_spec, context in zip(split_plan.slices, contexts):
            slice_dir = destination / slice_spec.name
            prefix_parts = [slice_spec.name]
            if ar is not None:
                prefix_parts.append(f"ar{ar}")
            if context_length is not None:
                prefix_parts.append(f"cl{context_length}")
            exported = context.export(slice_dir, prefix="_".join(prefix_parts))
            artifacts.append(
                TransformedSliceArtifact(
                    slice_name=slice_spec.name,
                    split_index=slice_spec.index,
                    model_path=Path(exported.onnx_path),
                    encodings_path=_path_or_none(getattr(exported, "encodings_path", None)),
                    ar=ar,
                    context_length=context_length,
                    external_data_paths=_exported_data_paths(exported),
                    graph_context=context,
                )
            )
        return tuple(artifacts)

    def convert(
        self,
        slice_or_path: TransformedSliceArtifact | ModelVariantArtifact | str | Path,
        *,
        encodings_path: str | Path | None = None,
        calibration_config: Any = None,
        backend: str = "HTP",
        output_path: str | Path | None = None,
        **options: Any,
    ) -> ConvertedModelArtifact:
        """Convert ONNX and apply AIMET encodings or SDK calibration."""

        source_path = _sdk_path(slice_or_path)
        selected_encodings = (
            Path(encodings_path)
            if encodings_path is not None
            else _path_or_none(_artifact_value(slice_or_path, "encodings_path"))
        )
        if selected_encodings is not None and calibration_config is not None:
            raise QairtConfigurationError(
                "convert accepts AIMET encodings or calibration_config, not both"
            )
        if backend.upper() != "HTP":
            raise QairtConfigurationError("production conversion is pinned to backend='HTP'")

        self._ensure_ready()
        qairt = self._load_module("qairt")
        model = qairt.convert(
            str(source_path),
            encodings=str(selected_encodings) if selected_encodings is not None else None,
            calibration_config=calibration_config,
            backend="HTP",
            **options,
        )
        saved_path: Path | None = None
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            returned_path = model.save(str(destination))
            saved_path = Path(returned_path or destination)

        if selected_encodings is not None:
            quantization_mode = "apply_encodings"
        elif calibration_config is not None:
            quantization_mode = "calibrate"
        else:
            quantization_mode = "float"
        return ConvertedModelArtifact(
            model_path=saved_path,
            source_model_path=source_path,
            quantization_mode=quantization_mode,
            slice_name=_artifact_value(slice_or_path, "slice_name"),
            ar=_artifact_value(slice_or_path, "ar"),
            context_length=_artifact_value(slice_or_path, "context_length"),
            sdk_model=model,
        )

    def create_calibration_config(
        self,
        *,
        dataset: str | Path | Sequence[Any],
        **options: Any,
    ) -> Any:
        """Create QAIRT's public Python ``CalibrationConfig`` after preflight."""

        self._ensure_ready()
        qairt = self._load_module("qairt")
        normalized_dataset: Any = (
            str(dataset) if isinstance(dataset, Path) else dataset
        )
        return qairt.CalibrationConfig(dataset=normalized_dataset, **options)

    def quantize(
        self,
        input_dlc: str | Path,
        *,
        output_dlc: str | Path,
        input_list: str | Path | None = None,
        **options: Any,
    ) -> QuantizedModelArtifact:
        """Run the standalone QAIRT DLC quantizer Python module.

        DEPRECATED. Production input for this program is always AIMET
        encodings applied through ``apply_encodings``; the standalone
        quantizer is reachable only from the deprecated expert MCP surface and
        is retained for debugging a calibration-based comparison.

        ``dump_encoding_json`` defaults to ``True`` here because the SDK leaves
        ``QuantizerOutputConfig.encoding_json`` unset otherwise (QAIRT 2.49
        ``qti/aisw/tools/core/modules/converter/quantizer_module.py``), which
        would make ``QuantizedModelArtifact.encodings_path`` permanently
        ``None``. Pass it explicitly to override.
        """

        self._ensure_ready()
        quantizer_module = self._load_module(
            "qti.aisw.tools.core.modules.converter.quantizer_module"
        )
        options.setdefault("dump_encoding_json", True)
        config = quantizer_module.QuantizerInputConfig(
            input_dlc=str(input_dlc),
            output_dlc=str(output_dlc),
            input_list=str(input_list) if input_list is not None else None,
            **options,
        )
        output = quantizer_module.QAIRTQuantizer().quantize(config)
        return QuantizedModelArtifact(
            dlc_path=Path(output.dlc_output),
            encodings_path=_path_or_none(getattr(output, "encoding_json", None)),
            sdk_output=output,
        )

    @staticmethod
    def _resolve_named_target(target: "TargetEntry | str | None") -> TargetEntry:
        """Resolve a caller's target, defaulting to the one the harness names.

        The GenAI lanes take a target rather than reading a module constant, so
        which SoC a container is built for is an argument of the call and shows
        up in its recorded build report.
        """

        if isinstance(target, TargetEntry):
            entry = target
        else:
            try:
                entry = resolve_target(target)
            except HarnessConstraintsError as error:
                raise QairtConfigurationError(str(error)) from error
        try:
            return require_verified_target(entry)
        except HarnessConstraintsError as error:
            raise QairtConfigurationError(str(error)) from error

    @staticmethod
    def _validate_target(
        target_soc: str, dsp_arch: str, soc_model: int
    ) -> TargetEntry:
        """Require an exact, reviewed, hardware-verified registry entry.

        There is no pinned tuple to compare against any more: a target is legal
        because it is registered under ``harness/targets/`` and has a
        ``verified`` block recording a real-device acceptance run, not because
        it matches a constant.
        """

        try:
            entry = resolve_target_tuple(
                str(target_soc), str(dsp_arch), int(soc_model)
            )
        except HarnessConstraintsError as error:
            raise QairtConfigurationError(str(error)) from error
        try:
            require_verified_target(entry)
        except HarnessConstraintsError as error:
            raise QairtConfigurationError(str(error)) from error
        return entry

    @staticmethod
    def _context_key(
        slice_name: str | None,
        context_length: int | None,
        ar_values: Sequence[int],
        graph_names: Sequence[str],
    ) -> str:
        payload = {
            "slice_name": slice_name,
            "context_length": context_length,
            "ar_values": list(ar_values),
            "graph_names": list(graph_names),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _validate_qwen35(
        self,
        family: FamilyProfile | None,
        source_kinds: Sequence[str],
        evidence: Qwen35ValidationEvidence | None,
        *,
        context_key: str,
        slice_name: str | None,
    ) -> None:
        if family is None or family.family is not FamilyId.QWEN3_5:
            return
        if not any(str(kind).lower() == "derived" for kind in source_kinds):
            return
        # State/MHA gates apply to recurrent decoder slices.  Embedding and
        # lm-head graphs are still checked by the normal weight-sharing SDK
        # compilation contract, but do not contain recurrent state or MHA.
        if slice_name is not None and not slice_name.startswith("decoder"):
            return
        if evidence is None:
            raise ExperimentalFeatureError(
                "Qwen3.5 derived AR weight sharing requires validation evidence for "
                "AR rewrite, recurrent/conv state I/O, MHA2SHA, initializer "
                "compatibility, and standalone-vs-joint execution"
            )
        if not evidence.ok:
            raise ExperimentalFeatureError(
                "Qwen3.5 derived AR validation failed: " + ", ".join(evidence.failed_gates)
            )
        if (
            not evidence.evidence_id
            or evidence.evidence_id not in self._validated_qwen35_evidence_ids
        ):
            raise ExperimentalFeatureError(
                "Qwen3.5 validation evidence was not minted by this adapter instance"
            )
        if context_key not in evidence.approved_context_keys:
            raise ExperimentalFeatureError(
                "Qwen3.5 validation evidence does not cover this slice/CL/AR graph set"
            )

    @staticmethod
    def _state_contract(model_info: Any) -> dict[str, tuple[str, ...]]:
        def selected(values: Sequence[Any], state_name: str) -> tuple[str, ...]:
            return tuple(
                sorted(
                    str(value.name)
                    for value in values
                    if state_name in str(value.name).lower()
                )
            )

        return {
            "recurrent_inputs": selected(model_info.inputs, "recurrent_state"),
            "recurrent_outputs": selected(model_info.outputs, "recurrent_state"),
            "conv_inputs": selected(model_info.inputs, "conv_state"),
            "conv_outputs": selected(model_info.outputs, "conv_state"),
        }

    @staticmethod
    def _weight_signatures(model_info: Any) -> tuple[tuple[Any, ...], ...]:
        initializers = {item.name: item for item in model_info.initializers}
        weight_names: set[str] = set()
        for node in model_info.nodes:
            if node.op_type in {"MatMul", "Gemm", "Conv"}:
                weight_names.update(name for name in node.inputs[1:] if name in initializers)
            elif node.op_type == "Gather" and node.inputs and node.inputs[0] in initializers:
                weight_names.add(node.inputs[0])
        return tuple(
            sorted(
                (
                    name,
                    initializers[name].shape,
                    initializers[name].dtype,
                    initializers[name].content_sha256,
                )
                for name in weight_names
            )
        )

    def _validate_qwen35_structure(
        self,
        variants: Sequence[ModelVariantArtifact],
        transformed_slices: Sequence[TransformedSliceArtifact],
    ) -> tuple[dict[str, bool], tuple[str, ...], str]:
        if len(variants) < 2:
            raise ExperimentalFeatureError(
                "Qwen3.5 validation requires at least two derived AR variants"
            )
        issues: list[str] = []
        variant_infos: dict[int, Any] = {}
        ar_rewrite_passed = True
        state_io_passed = True

        for variant in variants:
            info = self._onnx_inspector.inspect(variant.model_path)
            variant_infos[variant.ar] = info
            graph_ir = getattr(variant.graph_context, "graph_ir", None)
            metadata = getattr(graph_ir, "meta", {}) if graph_ir is not None else {}
            observed_ar = metadata.get("seq_length") if isinstance(metadata, Mapping) else None
            observed_cl = (
                metadata.get("context_length") if isinstance(metadata, Mapping) else None
            )
            if observed_ar is not None or observed_cl is not None:
                if observed_ar != variant.ar or observed_cl != variant.context_length:
                    ar_rewrite_passed = False
                    issues.append(
                        f"AR/CL metadata mismatch for {variant.model_path}: "
                        f"expected {variant.ar}/{variant.context_length}, "
                        f"got {observed_ar}/{observed_cl}"
                    )
            else:
                sequence_inputs = tuple(
                    tensor
                    for tensor in info.inputs
                    if any(
                        token in tensor.name.lower()
                        for token in ("input_ids", "attention_mask", "position_ids")
                    )
                )
                if not sequence_inputs or not any(
                    variant.ar in tensor.shape for tensor in sequence_inputs
                ):
                    ar_rewrite_passed = False
                    issues.append(
                        f"cannot prove AR={variant.ar} from graph metadata or sequence input shapes"
                    )

            contract = self._state_contract(info)
            if any(not names for names in contract.values()):
                state_io_passed = False
                issues.append(
                    f"{variant.model_path} is missing recurrent_state or conv_state graph I/O"
                )

        state_contracts = [
            self._state_contract(variant_infos[variant.ar])
            for variant in sorted(variants, key=lambda item: item.ar)
        ]
        if state_contracts and any(contract != state_contracts[0] for contract in state_contracts[1:]):
            state_io_passed = False
            issues.append("recurrent_state/conv_state names differ across derived AR variants")

        transformed_by_ar: dict[int, list[TransformedSliceArtifact]] = {}
        for artifact in transformed_slices:
            if artifact.ar is not None:
                transformed_by_ar.setdefault(artifact.ar, []).append(artifact)
        expected_ars = {variant.ar for variant in variants}
        if set(transformed_by_ar) != expected_ars:
            state_io_passed = False
            issues.append("transformed slice set does not cover every derived AR")

        mha2sha_passed = True
        initializer_compatibility_passed = True
        signatures_by_ar: dict[int, tuple[tuple[Any, ...], ...]] = {}
        transformed_state_contracts: dict[int, dict[str, set[str]]] = {}
        for ar, artifacts in transformed_by_ar.items():
            combined = {
                "recurrent_inputs": set(),
                "recurrent_outputs": set(),
                "conv_inputs": set(),
                "conv_outputs": set(),
            }
            all_signatures: list[tuple[Any, ...]] = []
            for artifact in artifacts:
                info = self._onnx_inspector.inspect(artifact.model_path)
                contract = self._state_contract(info)
                for key, names in contract.items():
                    combined[key].update(names)
                all_signatures.extend(self._weight_signatures(info))

                trace_method = getattr(artifact.graph_context, "get_tracing_info", None)
                trace_payload: Any = ()
                if callable(trace_method):
                    trace_payload = trace_method(merged=False)
                rendered_trace = json.dumps(trace_payload, default=str).lower()
                has_mha_trace = any(
                    token in rendered_trace for token in ("m2s", "mha2sha", "groupslice")
                )
                has_group_slice_op = any(
                    "groupslice" in node.op_type.lower() for node in info.nodes
                )
                if artifact.slice_name.startswith("decoder") and not (
                    has_mha_trace or has_group_slice_op
                ):
                    mha2sha_passed = False
                    issues.append(
                        f"no MHA2SHA trace or GroupSlice op found for {artifact.model_path}"
                    )
            transformed_state_contracts[ar] = combined
            signatures_by_ar[ar] = tuple(sorted(all_signatures))

        for ar, contract in transformed_state_contracts.items():
            if any(not names for names in contract.values()):
                state_io_passed = False
                issues.append(
                    f"transformed decoder slice AR={ar} lost recurrent_state or conv_state I/O"
                )
        transformed_contract_values = list(transformed_state_contracts.values())
        if transformed_contract_values and any(
            value != transformed_contract_values[0]
            for value in transformed_contract_values[1:]
        ):
            state_io_passed = False
            issues.append("transformed state I/O contract differs across AR variants")

        signature_values = list(signatures_by_ar.values())
        if not signature_values or any(not value for value in signature_values):
            initializer_compatibility_passed = False
            issues.append("no weight initializer signatures were available for comparison")
        elif any(
            signature != signature_values[0]
            for signature in signature_values[1:]
        ):
            initializer_compatibility_passed = False
            issues.append("weight initializer names/shapes/content differ across AR variants")
        elif any(item[3] is None for item in signature_values[0]):
            initializer_compatibility_passed = False
            issues.append("one or more weight initializers could not be content-hashed")

        gates = {
            "ar_rewrite_passed": ar_rewrite_passed,
            "state_io_passed": state_io_passed,
            "mha2sha_passed": mha2sha_passed,
            "initializer_compatibility_passed": initializer_compatibility_passed,
        }
        digest_payload = {
            "gates": gates,
            "issues": issues,
            "variants": [
                {
                    "path": str(item.model_path),
                    "ar": item.ar,
                    "context_length": item.context_length,
                }
                for item in variants
            ],
            "slices": [
                {
                    "path": str(item.model_path),
                    "slice_name": item.slice_name,
                    "ar": item.ar,
                }
                for item in transformed_slices
            ],
            "weight_signatures": signatures_by_ar,
        }
        digest = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return gates, tuple(issues), digest

    def _mint_qwen35_evidence(
        self,
        *,
        structural_gates: Mapping[str, bool],
        standalone_vs_joint_passed: bool,
        structural_digest: str,
        approved_context_keys: Sequence[str],
        diagnostic_contexts: Sequence[CompiledContextArtifact],
        runtime_report_paths: Sequence[Path],
        notes: Sequence[str] = (),
    ) -> Qwen35ValidationEvidence:
        evidence_id = uuid.uuid4().hex
        evidence = Qwen35ValidationEvidence(
            ar_rewrite_passed=bool(structural_gates["ar_rewrite_passed"]),
            state_io_passed=bool(structural_gates["state_io_passed"]),
            mha2sha_passed=bool(structural_gates["mha2sha_passed"]),
            initializer_compatibility_passed=bool(
                structural_gates["initializer_compatibility_passed"]
            ),
            standalone_vs_joint_passed=standalone_vs_joint_passed,
            notes=tuple(notes),
            evidence_id=evidence_id,
            structural_digest=structural_digest,
            approved_context_keys=tuple(approved_context_keys),
            diagnostic_context_paths=tuple(
                item.context_binary_path for item in diagnostic_contexts
            ),
            runtime_report_paths=tuple(runtime_report_paths),
        )
        self._validated_qwen35_evidence_ids.add(evidence_id)
        return evidence

    def validate_qwen35_derivation(
        self,
        variants: Sequence[ModelVariantArtifact],
        transformed_slices: Sequence[TransformedSliceArtifact],
        converted_models: Sequence[ConvertedModelArtifact],
        *,
        output_dir: str | Path,
        slice_name: str,
        graph_names: Sequence[str],
        ar_values: Sequence[int],
        context_length: int,
        target_soc: str,
        dsp_arch: str,
        soc_model: int,
        runtime_validator: Callable[
            [Qwen35RuntimeValidationRequest], Qwen35RuntimeValidationResult
        ]
        | None,
        validation_payload: Any = None,
        native_kv_config: str | Path | Mapping[str, Any] | None = None,
        native_kv_expectations: Sequence[NativeKvGraphExpectation] = (),
        expect_native_kv: bool = False,
    ) -> Qwen35DerivationValidation:
        """Mint non-forgeable Qwen3.5 evidence after structure and device gates.

        A diagnostic standalone context is compiled for each AR and a separate
        diagnostic joint context is compiled for weight-sharing comparison.
        The caller-provided runtime validator must execute those contexts with
        golden vectors and return the typed result.  A plain boolean/mapping is
        deliberately rejected.
        """

        if not slice_name.startswith("decoder"):
            raise ExperimentalFeatureError(
                "Qwen3.5 derivation validation is defined for decoder slices"
            )
        if runtime_validator is None:
            raise ExperimentalFeatureError(
                "Qwen3.5 build is fail-closed without a device/vector runtime_validator"
            )
        selected_models = tuple(converted_models)
        selected_names = tuple(str(name) for name in graph_names)
        selected_ars = tuple(int(value) for value in ar_values)
        if not (
            len(selected_models) == len(selected_names) == len(selected_ars)
            and len(selected_models) >= 2
        ):
            raise ExperimentalFeatureError(
                "Qwen3.5 validation models, graph_names, and ar_values must align"
            )

        structural_gates, structural_issues, structural_digest = (
            self._validate_qwen35_structure(variants, transformed_slices)
        )
        if not all(structural_gates.values()):
            raise ExperimentalFeatureError(
                "Qwen3.5 structural validation failed: " + "; ".join(structural_issues)
            )

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        structural_report_path = destination / "qwen35_structural_validation.json"
        structural_report_path.write_text(
            json.dumps(
                {
                    "family": FamilyId.QWEN3_5.value,
                    "slice_name": slice_name,
                    "context_length": context_length,
                    "ar_values": selected_ars,
                    "gates": structural_gates,
                    "issues": structural_issues,
                    "structural_digest": structural_digest,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        diagnostics: list[CompiledContextArtifact] = []
        for model, graph_name, ar in zip(selected_models, selected_names, selected_ars):
            standalone_expectations = tuple(
                item
                for item in native_kv_expectations
                if item.graph_name == graph_name
            )
            standalone_native_config: Mapping[str, Any] | None = None
            if expect_native_kv:
                standalone_native_config = build_native_kv_config(
                    standalone_expectations
                )
            diagnostics.append(
                self.compile_context(
                    [model],
                    output_path=destination / f"{slice_name}_ar{ar}_standalone.bin",
                    graph_names=[graph_name],
                    ar_values=[ar],
                    source_kinds=["derived"],
                    target_soc=target_soc,
                    dsp_arch=dsp_arch,
                    soc_model=soc_model,
                    family=FamilyId.QWEN3_5,
                    slice_name=slice_name,
                    weight_sharing=False,
                    native_kv_config=standalone_native_config,
                    native_kv_expectations=standalone_expectations,
                    expect_native_kv=expect_native_kv,
                    context_length=context_length,
                )
            )

        context_key = self._context_key(
            slice_name,
            context_length,
            selected_ars,
            selected_names,
        )
        provisional = self._mint_qwen35_evidence(
            structural_gates=structural_gates,
            standalone_vs_joint_passed=True,
            structural_digest=structural_digest,
            approved_context_keys=[context_key],
            diagnostic_contexts=diagnostics,
            runtime_report_paths=(),
            notes=("provisional diagnostic-joint compile authorization",),
        )
        try:
            joint = self.compile_context(
                selected_models,
                output_path=destination / f"{slice_name}_joint.bin",
                graph_names=selected_names,
                ar_values=selected_ars,
                source_kinds=["derived"] * len(selected_models),
                target_soc=target_soc,
                dsp_arch=dsp_arch,
                soc_model=soc_model,
                family=FamilyId.QWEN3_5,
                slice_name=slice_name,
                weight_sharing=True,
                native_kv_config=native_kv_config,
                native_kv_expectations=native_kv_expectations,
                expect_native_kv=expect_native_kv,
                context_length=context_length,
                qwen35_validation_evidence=provisional,
            )
        finally:
            self._validated_qwen35_evidence_ids.discard(provisional.evidence_id)
        diagnostics.append(joint)

        request = Qwen35RuntimeValidationRequest(
            slice_name=slice_name,
            ar_values=selected_ars,
            graph_names=selected_names,
            standalone_contexts=tuple(diagnostics[:-1]),
            joint_context=joint,
            validation_payload=validation_payload,
        )
        runtime_result = runtime_validator(request)
        if not isinstance(runtime_result, Qwen35RuntimeValidationResult):
            raise ExperimentalFeatureError(
                "runtime_validator must return Qwen35RuntimeValidationResult, not a boolean/mapping"
            )
        if not runtime_result.ok:
            raise ExperimentalFeatureError(
                "Qwen3.5 runtime validation failed: " + runtime_result.details
            )
        missing_executions = set(selected_names) - set(runtime_result.executed_graph_names)
        if missing_executions:
            raise ExperimentalFeatureError(
                "Qwen3.5 runtime validator did not attest execution of graphs: "
                f"{sorted(missing_executions)}"
            )
        if not runtime_result.golden_vector_ids:
            raise ExperimentalFeatureError(
                "Qwen3.5 runtime validation must identify the golden vectors used"
            )
        if not runtime_result.report_paths or any(
            not Path(path).is_file() for path in runtime_result.report_paths
        ):
            raise ExperimentalFeatureError(
                "Qwen3.5 runtime validation must emit reopenable report files"
            )
        report_digests = []
        for report_path in runtime_result.report_paths:
            report_digests.append(
                f"{Path(report_path)}:{hashlib.sha256(Path(report_path).read_bytes()).hexdigest()}"
            )

        evidence = self._mint_qwen35_evidence(
            structural_gates=structural_gates,
            standalone_vs_joint_passed=True,
            structural_digest=structural_digest,
            approved_context_keys=[context_key],
            diagnostic_contexts=diagnostics,
            runtime_report_paths=runtime_result.report_paths,
            notes=tuple(
                item
                for item in (runtime_result.details, *report_digests)
                if item
            ),
        )
        return Qwen35DerivationValidation(
            evidence=evidence,
            diagnostic_contexts=tuple(diagnostics),
            structural_report_path=structural_report_path,
            runtime_results=(runtime_result,),
        )

    @staticmethod
    def _validate_compiler_target(config: Any, target: TargetEntry) -> None:
        """Refuse a compile whose resolved device target is not the named one.

        An empty device-config list is the SDK's "could not set soc model for
        chipset ... skipping device config creation" path, which leaves the
        compiler on its own defaults. Those defaults are ``dsp_arch v79`` with
        ``soc_model 69`` -- exactly the registered SM8750 tuple -- so for that
        target a resolved-value check cannot tell an intended target from a
        silent fallback. The empty list therefore fails closed in its own right,
        whichever target was named.
        """

        device_configs = getattr(config, "device_custom_configs", None)
        if not device_configs:
            raise QairtConfigurationError(
                "CompileConfig produced no device configuration for "
                f"{target.chipset}; QAIRT skips device-config creation when "
                "it cannot resolve the requested SoC and would compile against "
                "its own default target"
            )
        for device_config in device_configs:
            soc_model = getattr(device_config, "soc_model", None)
            dsp_arch = getattr(device_config, "dsp_arch", None)
            dsp_value = getattr(dsp_arch, "value", dsp_arch)
            if (
                int(soc_model) != target.soc_model
                or str(dsp_value).lower() != target.dsp_arch
            ):
                raise QairtConfigurationError(
                    "CompileConfig resolved "
                    f"{dsp_value}/soc_model {soc_model} instead of the named "
                    f"{target.tuple_text}; refusing an SDK fallback"
                )

    def compile_context(
        self,
        models: Sequence[ConvertedModelArtifact | str | Path | Any],
        *,
        output_path: str | Path,
        graph_names: Sequence[str],
        ar_values: Sequence[int],
        source_kinds: Sequence[str],
        target_soc: str,
        dsp_arch: str,
        soc_model: int,
        family: FamilyProfile | FamilyId | str | None = None,
        slice_name: str | None = None,
        weight_sharing: bool = True,
        native_kv_config: str | Path | Mapping[str, Any] | None = None,
        native_kv_expectations: Sequence[NativeKvGraphExpectation] = (),
        expect_native_kv: bool = False,
        context_length: int | None = None,
        qwen35_validation_evidence: Qwen35ValidationEvidence | None = None,
        compile_config_options: Mapping[str, Any] | None = None,
    ) -> CompiledContextArtifact:
        """Compile one slice+CL, optionally sharing its AR graph weights."""

        resolved_profile = _profile(family)
        selected_models = tuple(models)
        selected_graph_names = tuple(str(name) for name in graph_names)
        selected_ars = tuple(int(value) for value in ar_values)
        selected_sources = tuple(str(value).lower() for value in source_kinds)
        target = self._validate_target(target_soc, dsp_arch, soc_model)
        validate_weight_sharing_sources(
            resolved_profile or FamilyId.QWEN3_DENSE,
            selected_sources,
        )

        if not selected_models:
            raise QairtConfigurationError("compile_context requires at least one model")
        if not (
            len(selected_models)
            == len(selected_graph_names)
            == len(selected_ars)
            == len(selected_sources)
        ):
            raise QairtConfigurationError(
                "models, graph_names, ar_values, and source_kinds must have equal length"
            )
        if len(set(selected_graph_names)) != len(selected_graph_names):
            raise QairtConfigurationError("graph_names must be unique within a context")
        if len(set(selected_ars)) != len(selected_ars):
            raise QairtConfigurationError("AR values must be unique within a context")
        if weight_sharing and len(selected_models) < 2:
            raise QairtConfigurationError(
                "weight_sharing requires at least two AR graph variants"
            )
        if not weight_sharing and len(selected_models) != 1:
            raise QairtConfigurationError(
                "multiple models may only be compiled together in weight-sharing mode"
            )

        artifact_slices = {
            item.slice_name
            for item in selected_models
            if isinstance(item, ConvertedModelArtifact) and item.slice_name is not None
        }
        if len(artifact_slices) > 1:
            raise QairtConfigurationError(
                "compile_context is per slice; models from different slices cannot be mixed"
            )
        inferred_slice = next(iter(artifact_slices), None)
        if slice_name is not None and inferred_slice is not None and slice_name != inferred_slice:
            raise QairtConfigurationError("slice_name does not match converted model artifacts")
        resolved_slice = slice_name or inferred_slice

        artifact_context_lengths = {
            item.context_length
            for item in selected_models
            if isinstance(item, ConvertedModelArtifact) and item.context_length is not None
        }
        if len(artifact_context_lengths) > 1:
            raise QairtConfigurationError(
                "compile_context is per context length; CL variants cannot be mixed"
            )
        inferred_context = next(iter(artifact_context_lengths), None)
        if (
            context_length is not None
            and inferred_context is not None
            and context_length != inferred_context
        ):
            raise QairtConfigurationError("context_length does not match converted artifacts")
        resolved_context = context_length or inferred_context

        evidence = _qwen35_evidence(qwen35_validation_evidence)
        if weight_sharing:
            context_key = self._context_key(
                resolved_slice,
                resolved_context,
                selected_ars,
                selected_graph_names,
            )
            self._validate_qwen35(
                resolved_profile,
                selected_sources,
                evidence,
                context_key=context_key,
                slice_name=resolved_slice,
            )

        native_config_path: Path | None = None
        if expect_native_kv:
            if resolved_context is None or resolved_context % 256 != 0:
                raise NativeKvConfigError(
                    "native-KV Genie contexts require context_length to be a multiple of 256"
                )
            if native_kv_config is None:
                raise NativeKvConfigError(
                    "native-KV was requested but no data-format config was provided"
                )
        if native_kv_config is not None:
            audit = audit_native_kv_config(
                native_kv_config,
                expectations=native_kv_expectations,
                expected_graph_names=(
                    selected_graph_names if expect_native_kv and not native_kv_expectations else ()
                ),
                require_nonempty=expect_native_kv,
            )
            require_native_kv_audit(audit)

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(native_kv_config, Mapping):
            native_config_path = destination.with_suffix(".native_kv.json")
            native_config_path.write_text(
                json.dumps(native_kv_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif native_kv_config is not None:
            native_config_path = Path(native_kv_config)

        unsafe_config_keys = {
            "backend",
            "soc_details",
            "device_custom_configs",
            "graph_custom_configs",
            "context_custom_configs",
        }
        supplied_config_options = dict(compile_config_options or {})
        overlap = unsafe_config_keys.intersection(supplied_config_options)
        if overlap:
            raise QairtConfigurationError(
                f"compile_config_options cannot override pinned target/mode fields: {sorted(overlap)}"
            )

        self._ensure_ready()
        qairt = self._load_module("qairt")
        config = qairt.CompileConfig(
            backend="HTP",
            soc_details=(
                f"chipset:{target.chipset};dsp_arch:{target.dsp_arch};"
                f"soc_model:{target.soc_model}"
            ),
            data_format_config=str(native_config_path) if native_config_path is not None else None,
            **supplied_config_options,
        )
        if weight_sharing:
            config.set_mode(
                "weight_sharing",
                graph_names=list(selected_graph_names),
                soc_model=target.soc_model,
                dsp_arch=target.dsp_arch,
            )
        self._validate_compiler_target(config, target)

        sdk_models: list[Any] = []
        for item in selected_models:
            if isinstance(item, ConvertedModelArtifact) and item.sdk_model is not None:
                sdk_models.append(item.sdk_model)
            elif isinstance(item, (str, Path)):
                sdk_models.append(qairt.load(str(item)))
            elif isinstance(item, ConvertedModelArtifact) and item.model_path is not None:
                sdk_models.append(qairt.load(str(item.model_path)))
            else:
                sdk_models.append(item)

        compile_input: Any = sdk_models if weight_sharing else sdk_models[0]
        compiled = qairt.compile(compile_input, config=config)
        returned_path = compiled.save(str(destination))
        saved_path = Path(returned_path or destination)
        return CompiledContextArtifact(
            context_binary_path=saved_path,
            slice_name=resolved_slice,
            graph_names=selected_graph_names,
            ar_values=selected_ars,
            target_soc=target.chipset,
            dsp_arch=target.dsp_arch,
            soc_model=target.soc_model,
            weight_sharing=weight_sharing,
            native_kv_config_path=native_config_path,
            context_length=resolved_context,
            sdk_compiled_model=compiled,
        )

    @staticmethod
    def _single_graph_name(model: Any) -> str:
        """Read the single graph name from a converted QAIRT model."""

        graphs_info = getattr(model, "graphs_info", None)
        if callable(graphs_info):
            graphs_info = graphs_info()
        names = tuple(
            str(getattr(item, "name"))
            for item in (graphs_info or ())
            if getattr(item, "name", None)
        )
        if not names:
            module = getattr(model, "module", None)
            graph_names = getattr(module, "graph_names", None)
            if callable(graph_names):
                names = tuple(str(name) for name in graph_names())
        if len(names) != 1 or not names[0]:
            raise QairtConfigurationError(
                "standalone ViT conversion must expose exactly one named graph; "
                f"QAIRT reported {names or '<none>'}"
            )
        return names[0]

    def build_standalone_vit(
        self,
        spec: Any,
        effective_config: Any,
        output_dir: str | Path,
    ) -> BuildResult:
        """Convert, quantize, and compile one unsplit ViT with low-level APIs."""

        self._ensure_ready()
        source = _first(
            effective_config,
            (("sources", "text"), ("source", "text"), ("text_source",)),
        )
        if source is None:
            source = _first(
                spec,
                (("sources", "text"), ("source", "text"), ("text_source",)),
            )
        model_value = _first(
            source,
            (("onnx_path",), ("onnx",), ("model_path",), ("path",)),
            default=source if isinstance(source, (str, Path)) else None,
        )
        if model_value is None:
            raise QairtConfigurationError(
                "standalone ViT requires sources.text.onnx_path"
            )
        model_path = Path(model_value)
        encodings_value = _first(
            source,
            (("encodings_path",), ("encodings",), ("aimet_encodings",)),
        )
        encodings_path = _path_or_none(encodings_value)
        missing = [
            f"model_path={model_path}" if not model_path.is_file() else "",
            (
                f"encodings_path={encodings_path}"
                if encodings_path is not None and not encodings_path.is_file()
                else ""
            ),
        ]
        if any(missing):
            raise QairtConfigurationError(
                "standalone ViT input paths do not exist: "
                + ", ".join(item for item in missing if item)
            )

        sequence = _first(effective_config, (("sequence",),), default={})
        ars = tuple(int(value) for value in _read(sequence, "ars", (1,)))
        if ars != (1,):
            raise QairtConfigurationError("standalone ViT requires the exact AR set (1,)")
        if bool(_read(sequence, "weight_sharing", False)) or bool(
            _read(sequence, "native_kv", False)
        ):
            raise QairtConfigurationError(
                "standalone ViT forbids weight sharing and native KV"
            )
        transforms = _first(effective_config, (("transforms",),), default={})
        if bool(_read(transforms, "mha2sha", False)):
            raise QairtConfigurationError(
                "standalone ViT does not run the LLM MHA2SHA transform"
            )

        quantization = _first(
            effective_config, (("quantization",),), default={}
        )
        mode = str(_enum_value(_read(quantization, "mode", "apply_encodings")))
        calibration_config = _read(quantization, "calibration_config")
        if mode == "apply_encodings" and encodings_path is None:
            raise QairtConfigurationError(
                "standalone ViT apply_encodings requires AIMET encodings"
            )
        if mode == "calibrate" and calibration_config is None:
            raise QairtConfigurationError(
                "standalone ViT calibration requires a QAIRT CalibrationConfig"
            )
        if mode not in {"apply_encodings", "calibrate", "float"}:
            raise QairtConfigurationError(
                "standalone ViT quantization mode must be apply_encodings, calibrate, or float"
            )

        target_soc = _first(
            spec, (("target", "chipset"), ("target_soc",)),
            default=ACTIVE_TARGET.chipset,
        )
        dsp_arch = _first(
            spec, (("target", "dsp_arch"), ("dsp_arch",)),
            default=ACTIVE_TARGET.dsp_arch,
        )
        soc_model = int(
            _first(
                spec, (("target", "soc_model"), ("soc_model",)),
                default=ACTIVE_TARGET.soc_model,
            )
        )
        target = self._validate_target(str(target_soc), str(dsp_arch), soc_model)
        compile_options = dict(
            _first(
                effective_config,
                (("compile", "compiler_options"), ("compile_config_options",)),
                default={},
            )
            or {}
        )
        misplaced_output_options = {
            "enable_intermediate_outputs",
            "set_output_tensors",
        }.intersection(compile_options)
        if misplaced_output_options:
            raise QairtConfigurationError(
                "compile.compiler_options cannot contain diagnostic output fields; "
                "use compile.enable_intermediate_outputs or compile.output_tensors"
            )
        enable_intermediate_outputs = bool(
            _first(
                effective_config,
                (("compile", "enable_intermediate_outputs"),),
                default=False,
            )
        )
        output_tensors = tuple(
            _first(
                effective_config,
                (("compile", "output_tensors"),),
                default=(),
            )
            or ()
        )
        diagnostic_compile_options = dict(compile_options)
        if enable_intermediate_outputs:
            diagnostic_compile_options["enable_intermediate_outputs"] = True
        if output_tensors:
            diagnostic_compile_options["set_output_tensors"] = list(output_tensors)
        diagnostic_requested = enable_intermediate_outputs or bool(output_tensors)

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        converted = self.convert(
            model_path,
            encodings_path=encodings_path if mode == "apply_encodings" else None,
            calibration_config=calibration_config if mode == "calibrate" else None,
            output_path=destination / "converted" / "vit.dlc",
        )
        graph_name = self._single_graph_name(converted.sdk_model)
        context = self.compile_context(
            [converted],
            output_path=destination / "contexts" / "vit.bin",
            graph_names=[graph_name],
            ar_values=[1],
            source_kinds=["base"],
            target_soc=str(target_soc),
            dsp_arch=str(dsp_arch),
            soc_model=soc_model,
            family=None,
            slice_name="vit",
            weight_sharing=False,
            expect_native_kv=False,
            compile_config_options=compile_options,
        )
        diagnostic_contexts: tuple[CompiledContextArtifact, ...] = ()
        if diagnostic_requested:
            diagnostic_contexts = (
                self.compile_context(
                    [converted],
                    output_path=(
                        destination / "diagnostic_contexts" / "vit.bin"
                    ),
                    graph_names=[graph_name],
                    ar_values=[1],
                    source_kinds=["base"],
                    target_soc=str(target_soc),
                    dsp_arch=str(dsp_arch),
                    soc_model=soc_model,
                    family=None,
                    slice_name="vit",
                    weight_sharing=False,
                    expect_native_kv=False,
                    compile_config_options=diagnostic_compile_options,
                ),
            )
        policy_path = destination / "config" / "standalone_vit_build.json"
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(
            json.dumps(
                {
                    "schema": "qairt-agent.standalone-vit-build",
                    "lane": "low_level_python_api",
                    "stages": ["qairt.convert", "qairt.compile"],
                    "model_path": str(model_path),
                    "encodings_path": (
                        str(encodings_path) if encodings_path is not None else None
                    ),
                    "quantization_mode": mode,
                    "graph_name": graph_name,
                    "target": {
                        "chipset": target.chipset,
                        "dsp_arch": target.dsp_arch,
                        "soc_model": target.soc_model,
                    },
                    "weight_sharing": False,
                    "native_kv": False,
                    "mha2sha": False,
                    "diagnostic_context": {
                        "requested": diagnostic_requested,
                        "enable_intermediate_outputs": (
                            enable_intermediate_outputs
                        ),
                        "output_tensors": output_tensors,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return BuildResult(
            variants=(),
            transformed_slices=(),
            converted_models=(converted,),
            contexts=(context,),
            diagnostic_contexts=diagnostic_contexts,
            config_artifact_paths=(policy_path,),
        )

    def _compiled_model(self, compiled_model_or_path: Any) -> Any:
        if isinstance(compiled_model_or_path, CompiledContextArtifact):
            if compiled_model_or_path.sdk_compiled_model is not None:
                return compiled_model_or_path.sdk_compiled_model
            compiled_model_or_path = compiled_model_or_path.context_binary_path
        if isinstance(compiled_model_or_path, (str, Path)):
            qairt = self._load_module("qairt")
            return qairt.load(str(compiled_model_or_path))
        return compiled_model_or_path

    def load_compiled(self, context_binary_path: str | Path) -> Any:
        """Load a context once so benchmark setup stays outside timed runs."""

        self._ensure_ready()
        return self._compiled_model(context_binary_path)

    def create_device(self, *, serial: str, server: str) -> Any:
        """Construct the exact Android target expected by QAIRT 2.49.

        ``CompiledModel(..., device=None)`` means local-host execution, so the
        orchestration layer must never pass the raw environment strings or
        omit this object for a real-device stage.
        """

        host, separator, port_text = str(server).rpartition(":")
        if (
            not str(serial).strip()
            or not separator
            or not host
            or not port_text.isdigit()
        ):
            raise QairtConfigurationError(
                "device requires a non-empty serial and server in host:port form"
            )
        self._ensure_ready()
        qairt = self._load_module("qairt")
        identifier = qairt.RemoteDeviceIdentifier(
            serial_id=str(serial),
            hostname=host,
            port=int(port_text),
        )
        return qairt.Device(
            type=qairt.DevicePlatformType.ANDROID,
            identifier=identifier,
        )

    def create_genai_executor(
        self,
        container_path: str | Path,
        *,
        device: Any,
    ) -> Any:
        """Load a saved GenAI container and prepare its public executor.

        Container loading and executor preparation intentionally happen outside
        the benchmark timer.  ``clean_up=False`` keeps lifecycle ownership in
        the orchestration layer, which always calls ``clean_environment`` in a
        ``finally`` block.
        """

        self._ensure_ready()
        factory = self._load_module(
            "qairt.gen_ai_api.containers.container_factory"
        )
        container = factory.load_container(str(Path(container_path)))
        kwargs: dict[str, Any] = {
            "clean_up": False,
            # LLMContainer.get_executor() defaults this to True while
            # WorkflowContainer.get_executor() constructs an unprepared
            # executor.  Passing the explicit flag is harmless for the latter
            # and gives this adapter one consistent prepare/cleanup owner.
            "prepare_environment": False,
        }
        if self._preflight_report is not None and self._preflight_report.sdk_root:
            kwargs["qairt_sdk_root"] = str(self._preflight_report.sdk_root)
        executor = container.get_executor(device=device, **kwargs)
        prepare = getattr(executor, "prepare_environment", None)
        if not callable(prepare):
            raise QairtConfigurationError(
                "GenAI container returned an executor without prepare_environment()"
            )
        prepare()
        return executor

    @staticmethod
    def clean_genai_executor(executor: Any) -> None:
        """Clean all executor-owned device artifacts."""

        cleanup = getattr(executor, "clean_environment", None)
        if not callable(cleanup):
            raise QairtConfigurationError(
                "GenAI executor does not expose clean_environment()"
            )
        cleanup()

    def run_graph(
        self,
        compiled_model_or_path: CompiledContextArtifact | str | Path | Any,
        inputs: Any,
        *,
        graph_name: str,
        device: Any = None,
        native_io: bool = False,
        **execution_options: Any,
    ) -> Any:
        """Execute exactly one named graph; this method never performs chain run."""

        if not graph_name:
            raise QairtConfigurationError("graph_name must be explicit")
        if "graph_names" in execution_options:
            raise QairtConfigurationError("graph_names is controlled by run_graph")
        self._ensure_ready()
        model = self._compiled_model(compiled_model_or_path)
        return model(
            inputs,
            device=device,
            graph_names=[graph_name],
            use_native_input_data=native_io,
            use_native_output_data=native_io,
            **execution_options,
        )

    def profile(
        self,
        compiled_model_or_path: CompiledContextArtifact | str | Path | Any,
        inputs: Any,
        *,
        graph_name: str,
        device: Any = None,
        native_io: bool = False,
        level: str = "detailed",
        option: str = "optrace",
        **execution_options: Any,
    ) -> ProfileResult:
        """Execute one graph under QAIRT's detailed/optrace profiler."""

        self._ensure_ready()
        qairt = self._load_module("qairt")
        model = self._compiled_model(compiled_model_or_path)
        with qairt.Profiler(context={"level": level, "option": option}) as profiler:
            execution_result = self.run_graph(
                model,
                inputs,
                graph_name=graph_name,
                device=device,
                native_io=native_io,
                **execution_options,
            )
        reports = tuple(profiler.generate_reports())
        return ProfileResult(
            execution_result=execution_result,
            reports=reports,
            graph_name=graph_name,
            level=level,
            option=option,
        )

    def create_qwen3_vl_workflow_config(
        self,
        *,
        vision_path: str | Path,
        text_path: str | Path,
        vision_config_path: str | Path | None = None,
        text_config_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
    ) -> Any:
        """Create QAIRT's native Python ``WorkflowGraph`` for ImageT2T."""

        self._ensure_ready()
        workflow_module = self._load_module("qairt.gen_ai_api.configs.workflow")
        vision_node = workflow_module.WorkflowNode(
            name="imageEncoder",
            role=workflow_module.WorkflowNodeRole.IMAGE_ENCODER,
            path=str(vision_path),
            config_path=(
                str(vision_config_path) if vision_config_path is not None else None
            ),
        )
        text_node = workflow_module.WorkflowNode(
            name="textGenerator",
            role=workflow_module.WorkflowNodeRole.TEXT_GENERATOR,
            path=str(text_path),
            config_path=str(text_config_path) if text_config_path is not None else None,
            tokenizer_path=str(tokenizer_path) if tokenizer_path is not None else None,
        )
        return workflow_module.WorkflowGraph(
            nodes=(vision_node, text_node),
            connections=(("imageEncoder", "textGenerator"),),
        )

    @staticmethod
    def _saved_genai_raw_slices(
        container: Any,
        destination: Path,
        ar_values: Sequence[int],
    ) -> tuple[tuple[GenAIRawSliceArtifact, ...], bool, tuple[str, ...]]:
        """Describe public LLMContainer splits for raw tensor diagnostics."""

        models = getattr(container, "models", None)
        if models is None:
            return (
                (),
                False,
                (
                    "saved container does not expose the public LLMContainer.models "
                    "surface",
                ),
            )
        normalized_models = tuple(models)
        if not normalized_models:
            return (), False, ("saved LLMContainer has no compiled models",)

        raw_slices: list[GenAIRawSliceArtifact] = []
        for index, model in enumerate(normalized_models):
            split_dir = destination / "models" / f"split_{index}"
            candidates = tuple(
                path
                for path in (
                    split_dir / "model.bin",
                    split_dir / "model.dlc",
                )
                if path.is_file()
            )
            if len(candidates) != 1:
                return (
                    (),
                    False,
                    (
                        f"split_{index} has {len(candidates)} saved compiled assets; "
                        "expected exactly one model.bin or model.dlc",
                    ),
                )
            graphs = tuple(getattr(model, "graphs_info", ()) or ())
            if not graphs:
                return (
                    (),
                    False,
                    (f"split_{index} exposes no graph metadata",),
                )
            if len(graphs) == len(ar_values):
                graph_names_by_ar = {
                    int(ar): str(graph.name)
                    for ar, graph in zip(ar_values, graphs)
                }
            elif len(graphs) == 1:
                graph_names_by_ar = {
                    int(ar): str(graphs[0].name)
                    for ar in ar_values
                }
            else:
                return (
                    (),
                    False,
                    (
                        f"split_{index} graph count {len(graphs)} cannot be "
                        f"bound to ARs {list(ar_values)}",
                    ),
                )

            input_sets = {
                tuple(str(tensor.name) for tensor in graph.inputs)
                for graph in graphs
            }
            output_sets = {
                tuple(str(tensor.name) for tensor in graph.outputs)
                for graph in graphs
            }
            if len(input_sets) != 1 or len(output_sets) != 1:
                return (
                    (),
                    False,
                    (
                        f"split_{index} changes tensor names between AR graphs; "
                        "the generic chain runner cannot prove a stable ABI",
                    ),
                )
            raw_slices.append(
                GenAIRawSliceArtifact(
                    slice_id=f"split_{index:03d}",
                    context_binary_path=candidates[0],
                    graph_names_by_ar=graph_names_by_ar,
                    input_names=next(iter(input_sets)),
                    output_names=next(iter(output_sets)),
                )
            )
        return (
            tuple(raw_slices),
            True,
            (
                "raw tensor diagnostics use public LLMContainer.models and "
                "CompiledModel graph metadata",
            ),
        )

    def build_genai_container(
        self,
        model_path: str | Path,
        *,
        output_dir: str | Path,
        family: FamilyProfile | FamilyId | str,
        split_plan: SplitPlan,
        encodings_path: str | Path,
        vision_model_path: str | Path | None = None,
        vision_encodings_path: str | Path | None = None,
        vision_config_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        config_path: str | Path | None = None,
        config_dict: Mapping[str, Any] | None = None,
        cache_root: str | Path | None = None,
        ar_values: Sequence[int] = (1, 128),
        context_lengths: Sequence[int] = (4096,),
        native_kv: bool = True,
        weight_sharing: bool = True,
        attached_models_by_ar: Mapping[
            int | str,
            GenAIAttachedModel | Mapping[str, Any],
        ]
        | None = None,
        target: TargetEntry | str | None = None,
        exist_ok: bool = False,
    ) -> GenAIContainerBuildResult:
        """Build and save one production container through QAIRT GenAI Builder.

        This is an explicit alternative production lane, not a wrapper around
        :meth:`build`.  ``GenAIBuilderHTP.build`` owns transform, conversion,
        quantization, and compilation for this call.  The low-level adapter
        stages remain the diagnostic/SQNR path and are never invoked here.

        Qwen3.5 is fail-closed: every requested AR must have an independently
        exported model and encodings entry in ``attached_models_by_ar``.  The
        method never enables the SDK's single-source automatic AR conversion.

        Qwen3-VL builds and saves a two-node ``WorkflowContainer`` but records
        ``runtime_supported=False`` because QAIRT 2.49's
        ``WorkflowContainer.get_executor`` cannot execute workflows with an
        ``IMAGE_ENCODER`` node.  This method never probes or invokes that
        unsupported executor.
        """

        resolved_target = self._resolve_named_target(target)
        resolved_profile = _profile(family)
        assert resolved_profile is not None
        is_multimodal = resolved_profile.family is FamilyId.QWEN3_VL
        if is_multimodal and (
            vision_model_path is None or vision_encodings_path is None
        ):
            raise QairtConfigurationError(
                "Qwen3-VL GenAI packaging requires explicit vision_model_path "
                "and vision_encodings_path; the projector must be inside the "
                "vision ONNX."
            )
        if not is_multimodal and any(
            value is not None
            for value in (
                vision_model_path,
                vision_encodings_path,
                vision_config_path,
            )
        ):
            raise QairtConfigurationError(
                "vision_model_path/vision_encodings_path/vision_config_path "
                "are valid only for Qwen3-VL"
            )
        if not isinstance(split_plan, SplitPlan):
            raise QairtConfigurationError("split_plan must be a families.SplitPlan")
        if config_path is not None and config_dict is not None:
            raise QairtConfigurationError(
                "config_path and config_dict are mutually exclusive"
            )

        normalized_ars = tuple(ar_values)
        if not normalized_ars or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in normalized_ars
        ):
            raise QairtConfigurationError("ar_values must contain positive integers")
        if len(set(normalized_ars)) != len(normalized_ars):
            raise QairtConfigurationError("ar_values must not contain duplicates")

        normalized_context_lengths = tuple(context_lengths)
        if not normalized_context_lengths or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in normalized_context_lengths
        ):
            raise QairtConfigurationError(
                "context_lengths must contain positive integers"
            )
        if len(set(normalized_context_lengths)) != len(normalized_context_lengths):
            raise QairtConfigurationError(
                "context_lengths must not contain duplicates"
            )
        if native_kv:
            unaligned = tuple(
                value for value in normalized_context_lengths if value % 256 != 0
            )
            if unaligned:
                raise QairtConfigurationError(
                    "native_kv requires context lengths divisible by 256; "
                    f"unaligned values: {unaligned}"
                )

        if weight_sharing:
            if len(normalized_ars) != 2 or set(normalized_ars) != {1, 128}:
                raise QairtConfigurationError(
                    "GenAI weight sharing requires exactly AR1 and AR128"
                )
        elif len(normalized_ars) != 1:
            raise QairtConfigurationError(
                "multiple ARs require weight_sharing=True"
            )

        normalized_attachments: dict[int, GenAIAttachedModel] = {}
        for raw_ar, raw_source in (attached_models_by_ar or {}).items():
            if isinstance(raw_ar, bool):
                raise QairtConfigurationError(
                    "attached_models_by_ar keys must be integer AR values"
                )
            try:
                ar = int(raw_ar)
            except (TypeError, ValueError) as error:
                raise QairtConfigurationError(
                    "attached_models_by_ar keys must be integer AR values"
                ) from error
            if ar in normalized_attachments:
                raise QairtConfigurationError(
                    f"duplicate attached model after AR normalization: {ar}"
                )
            if isinstance(raw_source, GenAIAttachedModel):
                source = raw_source
            elif isinstance(raw_source, Mapping):
                attached_model_path = _first(
                    raw_source,
                    (("model_path",), ("onnx_path",), ("model",), ("onnx",)),
                )
                attached_encodings_path = _first(
                    raw_source,
                    (("encodings_path",), ("encodings",), ("aimet_encodings",)),
                )
                if attached_model_path is None:
                    raise QairtConfigurationError(
                        f"attached_models_by_ar[{ar}] is missing model_path"
                    )
                source = GenAIAttachedModel(
                    model_path=Path(attached_model_path),
                    encodings_path=_path_or_none(attached_encodings_path),
                )
            else:
                raise QairtConfigurationError(
                    "attached_models_by_ar values must be GenAIAttachedModel "
                    "or mappings with model_path/encodings_path"
                )
            normalized_attachments[ar] = source

        unexpected_attachments = set(normalized_attachments).difference(
            normalized_ars
        )
        if unexpected_attachments:
            raise QairtConfigurationError(
                "attached_models_by_ar contains ARs not requested by ar_values: "
                f"{sorted(unexpected_attachments)}"
            )
        if resolved_profile.family is FamilyId.QWEN3_5:
            missing = set(normalized_ars).difference(normalized_attachments)
            if missing:
                raise ExperimentalFeatureError(
                    "Qwen3.5 GenAI Builder forbids single-source automatic AR "
                    "conversion; attached_models_by_ar must provide every requested "
                    f"AR, missing: {sorted(missing)}"
                )
            missing_encodings = tuple(
                ar
                for ar in normalized_ars
                if normalized_attachments[ar].encodings_path is None
            )
            if missing_encodings:
                raise ExperimentalFeatureError(
                    "Qwen3.5 attached models require explicit per-AR AIMET "
                    f"encodings, missing for ARs: {missing_encodings}"
                )

        source_model_path = Path(model_path)
        source_encodings_path = Path(encodings_path)
        resolved_vision_model_path = (
            Path(vision_model_path) if vision_model_path is not None else None
        )
        resolved_vision_encodings_path = (
            Path(vision_encodings_path)
            if vision_encodings_path is not None
            else None
        )
        required_paths: list[tuple[str, Path]] = [
            ("model_path", source_model_path),
            ("encodings_path", source_encodings_path),
        ]
        if resolved_vision_model_path is not None:
            required_paths.append(
                ("vision_model_path", resolved_vision_model_path)
            )
        if resolved_vision_encodings_path is not None:
            required_paths.append(
                ("vision_encodings_path", resolved_vision_encodings_path)
            )
        if vision_config_path is not None:
            required_paths.append(
                ("vision_config_path", Path(vision_config_path))
            )
        if tokenizer_path is not None:
            required_paths.append(("tokenizer_path", Path(tokenizer_path)))
        if config_path is not None:
            required_paths.append(("config_path", Path(config_path)))
        for ar, source in normalized_attachments.items():
            required_paths.append(
                (f"attached_models_by_ar[{ar}].model_path", source.model_path)
            )
            if source.encodings_path is not None:
                required_paths.append(
                    (
                        f"attached_models_by_ar[{ar}].encodings_path",
                        source.encodings_path,
                    )
                )
        missing_paths = tuple(
            f"{label}={path}" for label, path in required_paths if not path.exists()
        )
        if missing_paths:
            raise QairtConfigurationError(
                "GenAI Builder input paths do not exist: " + ", ".join(missing_paths)
            )

        destination = Path(output_dir)
        if destination.exists() and not destination.is_dir():
            raise QairtConfigurationError(
                f"GenAI container destination is not a directory: {destination}"
            )
        if destination.exists() and not exist_ok:
            raise QairtConfigurationError(
                f"GenAI container destination already exists: {destination}; "
                "pass exist_ok=True to reuse it"
            )

        self._ensure_ready()
        factory_api = (
            "qairt.gen_ai_api.gen_ai_builder_factory."
            "GenAIBuilderFactory.create"
        )
        builder_constructor_api = (
            "qairt.gen_ai_api.gen_ai_builder_factory."
            "GenAIBuilderFactory.create"
        )
        used_factory = True
        if resolved_profile.family is FamilyId.QWEN3_5:
            # QAIRT 2.49 only lists Qwen3_5ForConditionalGeneration in
            # GenAIBuilderFactory.SupportedLLMs.  A standalone Omni Thinker
            # legitimately carries Qwen3_5OmniThinkerForConditionalGeneration,
            # so routing it through the generic factory silently selects the
            # base GenAIBuilderHTP.  Use the SDK's public, model-specific
            # constructor for the whole Qwen3.5 text family.  The caller's
            # original config path/dict remains unchanged and is still the
            # provenance-bearing input.
            qwen_builder_module = self._load_module(
                "qairt.gen_ai_api.builders.qwen.builder"
            )
            builder_class = getattr(
                qwen_builder_module,
                "Qwen3_5BuilderHTP",
                None,
            )
            from_pretrained = getattr(builder_class, "from_pretrained", None)
            if not callable(from_pretrained):
                raise QairtConfigurationError(
                    "QAIRT SDK does not expose Qwen3_5BuilderHTP.from_pretrained"
                )
            builder = from_pretrained(
                str(source_model_path),
                str(cache_root) if cache_root is not None else None,
                tokenizer_path=(
                    str(tokenizer_path) if tokenizer_path is not None else None
                ),
                config_path=(
                    str(config_path) if config_path is not None else None
                ),
                config_dict=(
                    dict(config_dict) if config_dict is not None else None
                ),
            )
            builder_constructor_api = (
                "qairt.gen_ai_api.builders.qwen.builder."
                "Qwen3_5BuilderHTP.from_pretrained"
            )
            used_factory = False
        else:
            factory_module = self._load_module(
                "qairt.gen_ai_api.gen_ai_builder_factory"
            )
            common_module = self._load_module("qairt.api.configs.common")
            builder = factory_module.GenAIBuilderFactory.create(
                str(source_model_path),
                common_module.BackendType.HTP,
                cache_root=str(cache_root) if cache_root is not None else None,
                tokenizer_path=(
                    str(tokenizer_path) if tokenizer_path is not None else None
                ),
                config_path=(
                    str(config_path) if config_path is not None else None
                ),
                config_dict=(
                    dict(config_dict) if config_dict is not None else None
                ),
            )
        expected_builder_classes = {
            FamilyId.QWEN3_MOE: "Qwen3MoeBuilderHTP",
            FamilyId.QWEN3_5: "Qwen3_5BuilderHTP",
        }
        expected_builder_class = expected_builder_classes.get(
            resolved_profile.family
        )
        if (
            expected_builder_class is not None
            and type(builder).__name__ != expected_builder_class
        ):
            raise QairtConfigurationError(
                "QAIRT did not construct the pinned explicit "
                f"{resolved_profile.family.value} builder: expected "
                f"{expected_builder_class}, got {type(builder).__name__}"
            )
        builder.encodings_path = str(source_encodings_path)
        target_spec = (
            f"chipset:{resolved_target.chipset};"
            f"dsp_arch:{resolved_target.dsp_arch};"
            f"soc_model:{resolved_target.soc_model}"
        )
        builder.set_targets([target_spec])

        if resolved_profile.family is FamilyId.QWEN3_5:
            builder.skip_ar_conversion = False
        builder.set_transformation_options(
            options={
                "arn": list(normalized_ars),
                "context_length": list(normalized_context_lengths),
                "split.num_splits": split_plan.num_splits,
                "split.split_embedding": split_plan.split_embedding,
                "split.split_lm_head": split_plan.split_lm_head,
                "mha2sha.permute_kv_cache_io": native_kv,
            }
        )
        for ar in normalized_ars:
            source = normalized_attachments.get(ar)
            if source is None:
                continue
            builder.attach_model_for_arn(
                ar,
                str(source.model_path),
                (
                    str(source.encodings_path)
                    if source.encodings_path is not None
                    else None
                ),
            )
        builder.native_kv = native_kv
        builder.weight_sharing = weight_sharing

        vision_builder: Any = None
        if is_multimodal:
            assert resolved_vision_model_path is not None
            assert resolved_vision_encodings_path is not None
            vision_builder_module = self._load_module(
                "qairt.gen_ai_api.builders.vision_encoder_builder_htp"
            )
            workflow_builder_module = self._load_module(
                "qairt.gen_ai_api.builders.workflow_builder"
            )
            vision_builder = (
                vision_builder_module.VisionEncoderBuilderHTP.from_pretrained(
                    str(resolved_vision_model_path),
                    cache_root=(
                        str(cache_root) if cache_root is not None else None
                    ),
                    config_path=(
                        str(vision_config_path)
                        if vision_config_path is not None
                        else None
                    ),
                )
            )
            vision_builder.encodings_path = str(
                resolved_vision_encodings_path
            )
            vision_builder.set_targets([target_spec])
            workflow_graph = self.create_qwen3_vl_workflow_config(
                vision_path=resolved_vision_model_path,
                text_path=source_model_path,
                vision_config_path=vision_config_path,
                text_config_path=config_path,
                tokenizer_path=tokenizer_path,
            )
            workflow_builder = (
                workflow_builder_module.WorkflowBuilder.from_builders(
                    {
                        "imageEncoder": vision_builder,
                        "textGenerator": builder,
                    },
                    workflow_graph,
                )
            )
            container = workflow_builder.build()
        else:
            container = builder.build()
        container.save(str(destination), exist_ok=exist_ok)
        if not destination.is_dir():
            raise QairtConfigurationError(
                "QAIRT GenAI container save completed without creating "
                f"the destination directory: {destination}"
            )

        factory_support = str(_enum_value(resolved_profile.factory_support))
        compatibility_notes: list[str] = []
        if resolved_profile.family is FamilyId.QWEN3_5:
            compatibility_mode = "explicit_family_builder"
            compatibility_notes.append(
                "Qwen3.5 uses the public Qwen3_5BuilderHTP.from_pretrained "
                "constructor directly because QAIRT 2.49 GenAIBuilderFactory "
                "does not recognize every supported Qwen3.5 architecture name."
            )
        elif factory_support == "generic_fallback":
            compatibility_mode = "generic_fallback_requires_device_validation"
            compatibility_notes.append(
                "QAIRT 2.49 does not explicitly dispatch this architecture; "
                "GenAIBuilderFactory selected its generic HTP compatibility path. "
                "Device golden validation is required before release."
            )
        else:
            compatibility_mode = "explicit_factory"
        if resolved_profile.family is FamilyId.QWEN3_5:
            compatibility_notes.append(
                "All requested ARs use caller-supplied models and AIMET encodings; "
                "single-source automatic AR conversion was disabled."
            )
        runtime_supported = not is_multimodal
        if is_multimodal:
            compatibility_notes.append(
                "QAIRT 2.49 can build and save this IMAGE_ENCODER -> "
                "TEXT_GENERATOR WorkflowContainer, but its get_executor() "
                "rejects IMAGE_ENCODER workflows. Runtime is unsupported."
            )
        (
            raw_slices,
            raw_tensor_runtime_supported,
            raw_tensor_runtime_notes,
        ) = (
            self._saved_genai_raw_slices(
                container,
                destination,
                normalized_ars,
            )
            if runtime_supported
            else (
                (),
                False,
                (
                    "raw tensor runtime is disabled when the container executor "
                    "topology is unsupported",
                ),
            )
        )

        metadata_path = destination / "qairt_agent_genai_build.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "schema": "qairt-agent.genai-build",
                    "lane": "genai_builder_production_packaging",
                    "python_api": {
                        # Kept for readers of the original metadata schema.
                        # The builder_constructor field is authoritative.
                        "factory": factory_api if used_factory else None,
                        "builder_constructor": builder_constructor_api,
                        "vision_factory": (
                            "qairt.gen_ai_api.builders."
                            "vision_encoder_builder_htp."
                            "VisionEncoderBuilderHTP.from_pretrained"
                            if is_multimodal
                            else None
                        ),
                        "workflow_factory": (
                            "qairt.gen_ai_api.builders.workflow_builder."
                            "WorkflowBuilder.from_builders"
                            if is_multimodal
                            else None
                        ),
                        "build": (
                            "qairt.gen_ai_api.builders.workflow_builder."
                            "WorkflowBuilder.build"
                            if is_multimodal
                            else (
                                "qairt.gen_ai_api.builders."
                                "gen_ai_builder_htp."
                                "GenAIBuilderHTP.build"
                            )
                        ),
                        "save": (
                            "qairt.gen_ai_api.containers."
                            "workflow_container."
                            "WorkflowContainer.save"
                            if is_multimodal
                            else (
                                "qairt.gen_ai_api.containers."
                                "llm_container."
                                "LLMContainer.save"
                            )
                        ),
                    },
                    "family": resolved_profile.family.value,
                    "builder_class": type(builder).__name__,
                    "vision_builder_class": (
                        type(vision_builder).__name__
                        if vision_builder is not None
                        else None
                    ),
                    "container_class": type(container).__name__,
                    "capability": {
                        "factory_support": factory_support,
                        "compatibility_mode": compatibility_mode,
                        "runtime_supported": runtime_supported,
                        "notes": compatibility_notes,
                    },
                    "target": {
                        "chipset": resolved_target.chipset,
                        "dsp_arch": resolved_target.dsp_arch,
                        "soc_model": resolved_target.soc_model,
                    },
                    "sequence": {
                        "ar_values": normalized_ars,
                        "context_lengths": normalized_context_lengths,
                        "weight_sharing": weight_sharing,
                        "native_kv": native_kv,
                        "attached_ar_values": tuple(normalized_attachments),
                    },
                    "split": {
                        "num_splits": split_plan.num_splits,
                        "split_embedding": split_plan.split_embedding,
                        "split_lm_head": split_plan.split_lm_head,
                    },
                    "raw_tensor_runtime": {
                        "supported": raw_tensor_runtime_supported,
                        "notes": raw_tensor_runtime_notes,
                        "slices": [
                            {
                                "slice_id": item.slice_id,
                                "context_binary_path": str(
                                    item.context_binary_path
                                ),
                                "graph_names_by_ar": item.graph_names_by_ar,
                                "input_names": item.input_names,
                                "output_names": item.output_names,
                            }
                            for item in raw_slices
                        ],
                    },
                    "workflow": (
                        {
                            "nodes": ("imageEncoder", "textGenerator"),
                            "connections": (
                                ("imageEncoder", "textGenerator"),
                            ),
                            "projector_location": "inside_vision_onnx",
                        }
                        if is_multimodal
                        else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        return GenAIContainerBuildResult(
            container_path=destination,
            metadata_path=metadata_path,
            family=resolved_profile.family.value,
            builder_class=type(builder).__name__,
            container_class=type(container).__name__,
            factory_support=factory_support,
            compatibility_mode=compatibility_mode,
            compatibility_notes=tuple(compatibility_notes),
            ar_values=normalized_ars,
            context_lengths=normalized_context_lengths,
            num_splits=split_plan.num_splits,
            split_embedding=split_plan.split_embedding,
            split_lm_head=split_plan.split_lm_head,
            target_soc=resolved_target.chipset,
            dsp_arch=resolved_target.dsp_arch,
            soc_model=resolved_target.soc_model,
            weight_sharing=weight_sharing,
            native_kv=native_kv,
            runtime_supported=runtime_supported,
            attached_ar_values=tuple(normalized_attachments),
            vision_builder_class=(
                type(vision_builder).__name__
                if vision_builder is not None
                else None
            ),
            raw_slices=raw_slices,
            raw_tensor_runtime_supported=raw_tensor_runtime_supported,
            raw_tensor_runtime_notes=raw_tensor_runtime_notes,
            sdk_container=container,
        )

    def build_qwen35_omni_components(
        self,
        text_model_path: str | Path,
        *,
        audio_model_path: str | Path,
        output_dir: str | Path,
        split_plan: SplitPlan,
        text_encodings_path: str | Path,
        audio_encodings_path: str | Path,
        text_config_path: str | Path,
        audio_config_path: str | Path,
        tokenizer_path: str | Path | None = None,
        cache_root: str | Path | None = None,
        ar_values: Sequence[int] = (1, 128),
        context_lengths: Sequence[int] = (4096,),
        native_kv: bool = True,
        weight_sharing: bool = True,
        attached_models_by_ar: Mapping[
            int | str,
            GenAIAttachedModel | Mapping[str, Any],
        ]
        | None = None,
        target: TargetEntry | str | None = None,
        exist_ok: bool = False,
    ) -> GenAIContainerBuildResult:
        """Build a Qwen3.5-Omni AUDIO_ENCODER -> TEXT_GENERATOR package.

        The checked-in QAIRT 2.49 SDK has a dedicated
        ``Qwen3OmniAudioEncoderBuilderHTP`` and a Qwen3.5 text builder, but its
        ``WorkflowContainer.get_executor`` does not orchestrate audio.  This
        method therefore packages both SDK containers while explicitly
        reporting ``runtime_supported=False`` and never probes that executor.
        """

        resolved_target = self._resolve_named_target(target)

        if not isinstance(split_plan, SplitPlan):
            raise QairtConfigurationError("split_plan must be a families.SplitPlan")
        normalized_ars = tuple(int(value) for value in ar_values)
        if (
            not normalized_ars
            or any(value <= 0 for value in normalized_ars)
            or len(set(normalized_ars)) != len(normalized_ars)
        ):
            raise QairtConfigurationError(
                "Qwen3.5-Omni ar_values must contain unique positive integers"
            )
        normalized_context_lengths = tuple(int(value) for value in context_lengths)
        if (
            not normalized_context_lengths
            or any(value <= 0 for value in normalized_context_lengths)
            or len(set(normalized_context_lengths)) != len(
                normalized_context_lengths
            )
        ):
            raise QairtConfigurationError(
                "Qwen3.5-Omni context_lengths must contain unique positive integers"
            )
        if native_kv and any(value % 256 for value in normalized_context_lengths):
            raise QairtConfigurationError(
                "Qwen3.5-Omni native_kv requires context lengths divisible by 256"
            )
        if weight_sharing and set(normalized_ars) != {1, 128}:
            raise QairtConfigurationError(
                "Qwen3.5-Omni weight sharing requires exactly AR1 and AR128"
            )
        if not weight_sharing and len(normalized_ars) != 1:
            raise QairtConfigurationError(
                "Qwen3.5-Omni multiple ARs require weight_sharing=True"
            )

        normalized_attachments: dict[int, GenAIAttachedModel] = {}
        for raw_ar, raw_source in (attached_models_by_ar or {}).items():
            if isinstance(raw_ar, bool):
                raise QairtConfigurationError(
                    "attached_models_by_ar keys must be integer AR values"
                )
            try:
                ar = int(raw_ar)
            except (TypeError, ValueError) as error:
                raise QairtConfigurationError(
                    "attached_models_by_ar keys must be integer AR values"
                ) from error
            if ar in normalized_attachments:
                raise QairtConfigurationError(
                    f"duplicate attached model after AR normalization: {ar}"
                )
            if isinstance(raw_source, GenAIAttachedModel):
                source = raw_source
            elif isinstance(raw_source, Mapping):
                attached_model = _first(
                    raw_source,
                    (("model_path",), ("onnx_path",), ("model",), ("onnx",)),
                )
                attached_encodings = _first(
                    raw_source,
                    (("encodings_path",), ("encodings",), ("aimet_encodings",)),
                )
                if attached_model is None or attached_encodings is None:
                    raise QairtConfigurationError(
                        f"attached_models_by_ar[{ar}] requires model_path and encodings_path"
                    )
                source = GenAIAttachedModel(
                    model_path=Path(attached_model),
                    encodings_path=Path(attached_encodings),
                )
            else:
                raise QairtConfigurationError(
                    "attached_models_by_ar values must be mappings or GenAIAttachedModel"
                )
            if source.encodings_path is None:
                raise QairtConfigurationError(
                    f"attached_models_by_ar[{ar}] requires encodings_path"
                )
            normalized_attachments[ar] = source

        missing_ars = set(normalized_ars).difference(normalized_attachments)
        unexpected_ars = set(normalized_attachments).difference(normalized_ars)
        if missing_ars or unexpected_ars:
            raise ExperimentalFeatureError(
                "Qwen3.5-Omni forbids single-source automatic AR conversion; "
                f"missing ARs={sorted(missing_ars)}, unexpected ARs={sorted(unexpected_ars)}"
            )

        text_model = Path(text_model_path)
        audio_model = Path(audio_model_path)
        text_encodings = Path(text_encodings_path)
        audio_encodings = Path(audio_encodings_path)
        text_model_config = Path(text_config_path)
        audio_model_config = Path(audio_config_path)
        required_paths: list[tuple[str, Path]] = [
            ("text_model_path", text_model),
            ("audio_model_path", audio_model),
            ("text_encodings_path", text_encodings),
            ("audio_encodings_path", audio_encodings),
            ("text_config_path", text_model_config),
            ("audio_config_path", audio_model_config),
        ]
        if tokenizer_path is not None:
            required_paths.append(("tokenizer_path", Path(tokenizer_path)))
        for ar, source in normalized_attachments.items():
            required_paths.extend(
                [
                    (f"attached_models_by_ar[{ar}].model_path", source.model_path),
                    (
                        f"attached_models_by_ar[{ar}].encodings_path",
                        source.encodings_path,
                    ),
                ]
            )
        missing_paths = [
            f"{label}={path}" for label, path in required_paths if not path.exists()
        ]
        if missing_paths:
            raise QairtConfigurationError(
                "Qwen3.5-Omni input paths do not exist: " + ", ".join(missing_paths)
            )

        destination = Path(output_dir)
        if destination.exists() and not destination.is_dir():
            raise QairtConfigurationError(
                f"Qwen3.5-Omni destination is not a directory: {destination}"
            )
        if destination.exists() and not exist_ok:
            raise QairtConfigurationError(
                f"Qwen3.5-Omni destination already exists: {destination}; "
                "pass exist_ok=True to reuse it"
            )

        self._ensure_ready()
        factory_module = self._load_module(
            "qairt.gen_ai_api.gen_ai_builder_factory"
        )
        create_audio_encoder = getattr(
            factory_module.GenAIBuilderFactory, "create_audio_encoder", None
        )
        if not callable(create_audio_encoder):
            raise QairtConfigurationError(
                "QAIRT SDK does not expose GenAIBuilderFactory.create_audio_encoder"
            )
        audio_builder = create_audio_encoder(
            str(audio_model),
            cache_root=(
                str(Path(cache_root) / "audio") if cache_root is not None else None
            ),
            config_path=str(audio_model_config),
            encodings_path=str(audio_encodings),
        )
        if type(audio_builder).__name__ != "Qwen3OmniAudioEncoderBuilderHTP":
            raise QairtConfigurationError(
                "GenAIBuilderFactory did not dispatch the pinned Omni audio builder: "
                "expected Qwen3OmniAudioEncoderBuilderHTP, got "
                f"{type(audio_builder).__name__}"
            )

        qwen_builder_module = self._load_module(
            "qairt.gen_ai_api.builders.qwen.builder"
        )
        text_builder_class = getattr(
            qwen_builder_module, "Qwen3_5BuilderHTP", None
        )
        if text_builder_class is None or not callable(
            getattr(text_builder_class, "from_pretrained", None)
        ):
            raise QairtConfigurationError(
                "QAIRT SDK does not expose Qwen3_5BuilderHTP.from_pretrained"
            )
        text_builder = text_builder_class.from_pretrained(
            str(text_model),
            str(Path(cache_root) / "text") if cache_root is not None else None,
            tokenizer_path=(
                str(tokenizer_path) if tokenizer_path is not None else None
            ),
            config_path=str(text_model_config),
        )
        if type(text_builder).__name__ != "Qwen3_5BuilderHTP":
            raise QairtConfigurationError(
                "pinned Qwen3.5 text builder returned an unexpected type: "
                f"{type(text_builder).__name__}"
            )

        audio_config = getattr(audio_builder, "config", None)
        missing_audio_tokens = [
            name
            for name in ("audio_start_token_id", "audio_end_token_id")
            if not isinstance(getattr(audio_config, name, None), int)
            or getattr(audio_config, name) < 0
        ]
        if missing_audio_tokens:
            raise QairtConfigurationError(
                "Qwen3.5-Omni audio config is missing workflow token IDs: "
                + ", ".join(missing_audio_tokens)
            )

        target_spec = (
            f"chipset:{resolved_target.chipset};"
            f"dsp_arch:{resolved_target.dsp_arch};"
            f"soc_model:{resolved_target.soc_model}"
        )
        audio_builder.set_targets([target_spec])
        text_builder.encodings_path = str(text_encodings)
        text_builder.set_targets([target_spec])
        text_builder.skip_ar_conversion = False
        text_builder.set_transformation_options(
            options={
                "arn": list(normalized_ars),
                "context_length": list(normalized_context_lengths),
                "split.num_splits": split_plan.num_splits,
                "split.split_embedding": split_plan.split_embedding,
                "split.split_lm_head": split_plan.split_lm_head,
                "mha2sha.permute_kv_cache_io": native_kv,
            }
        )
        for ar in normalized_ars:
            source = normalized_attachments[ar]
            text_builder.attach_model_for_arn(
                ar,
                str(source.model_path),
                str(source.encodings_path),
            )
        text_builder.native_kv = native_kv
        text_builder.weight_sharing = weight_sharing

        workflow_module = self._load_module(
            "qairt.gen_ai_api.configs.workflow"
        )
        workflow_builder_module = self._load_module(
            "qairt.gen_ai_api.builders.workflow_builder"
        )
        workflow_graph = workflow_module.WorkflowGraph(
            nodes=(
                workflow_module.WorkflowNode(
                    name="audioEncoder",
                    role=workflow_module.WorkflowNodeRole.AUDIO_ENCODER,
                ),
                workflow_module.WorkflowNode(
                    name="textGenerator",
                    role=workflow_module.WorkflowNodeRole.TEXT_GENERATOR,
                ),
            ),
            connections=(("audioEncoder", "textGenerator"),),
        )
        workflow_builder = workflow_builder_module.WorkflowBuilder.from_builders(
            {
                "audioEncoder": audio_builder,
                "textGenerator": text_builder,
            },
            workflow_graph,
        )
        container = workflow_builder.build()
        container.save(str(destination), exist_ok=exist_ok)
        audio_container_path = destination / "audioEncoder"
        text_container_path = destination / "textGenerator"
        missing_components = [
            str(path)
            for path in (audio_container_path, text_container_path)
            if not path.is_dir() or not any(path.iterdir())
        ]
        if missing_components:
            raise QairtConfigurationError(
                "Qwen3.5-Omni workflow save omitted component directories: "
                + ", ".join(missing_components)
            )

        compatibility_notes = (
            "Audio uses Qwen3OmniAudioEncoderBuilderHTP and text uses the pinned "
            "Qwen3_5BuilderHTP; no family aliasing was used.",
            "QAIRT 2.49 WorkflowContainer.get_executor does not orchestrate its "
            "AUDIO_ENCODER node; the package is buildable but end-to-end audio "
            "runtime is not claimed.",
            "All text ARs use caller-supplied ONNX models and AIMET encodings.",
        )
        metadata_path = destination / "qairt_agent_genai_build.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "schema": "qairt-agent.genai-build",
                    "lane": "qwen3_5_omni_component_packaging",
                    "python_api": {
                        "audio_factory": (
                            "qairt.gen_ai_api.gen_ai_builder_factory."
                            "GenAIBuilderFactory.create_audio_encoder"
                        ),
                        # Kept for readers of the original metadata schema.
                        # Qwen3.5 text construction is direct, not factory-owned.
                        "text_factory": None,
                        "text_builder_constructor": (
                            "qairt.gen_ai_api.builders.qwen.builder."
                            "Qwen3_5BuilderHTP.from_pretrained"
                        ),
                        "workflow_factory": (
                            "qairt.gen_ai_api.builders.workflow_builder."
                            "WorkflowBuilder.from_builders"
                        ),
                    },
                    "family": FamilyId.QWEN3_5_OMNI.value,
                    "builders": {
                        "audioEncoder": type(audio_builder).__name__,
                        "textGenerator": type(text_builder).__name__,
                    },
                    "container_class": type(container).__name__,
                    "capability": {
                        "factory_support": "explicit",
                        "compatibility_mode": (
                            "explicit_components_runtime_unsupported"
                        ),
                        "runtime_supported": False,
                        "notes": compatibility_notes,
                    },
                    "target": {
                        "chipset": resolved_target.chipset,
                        "dsp_arch": resolved_target.dsp_arch,
                        "soc_model": resolved_target.soc_model,
                    },
                    "configs": {
                        "audio": str(audio_model_config),
                        "text": str(text_model_config),
                    },
                    "sequence": {
                        "ar_values": normalized_ars,
                        "context_lengths": normalized_context_lengths,
                        "weight_sharing": weight_sharing,
                        "native_kv": native_kv,
                        "attached_ar_values": tuple(normalized_attachments),
                    },
                    "split": {
                        "num_splits": split_plan.num_splits,
                        "split_embedding": split_plan.split_embedding,
                        "split_lm_head": split_plan.split_lm_head,
                    },
                    "workflow": {
                        "nodes": {
                            "audioEncoder": "AUDIO_ENCODER",
                            "textGenerator": "TEXT_GENERATOR",
                        },
                        "connections": (("audioEncoder", "textGenerator"),),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        return GenAIContainerBuildResult(
            container_path=destination,
            metadata_path=metadata_path,
            family=FamilyId.QWEN3_5_OMNI.value,
            builder_class=type(text_builder).__name__,
            audio_builder_class=type(audio_builder).__name__,
            container_class=type(container).__name__,
            factory_support="explicit",
            compatibility_mode="explicit_components_runtime_unsupported",
            compatibility_notes=compatibility_notes,
            ar_values=normalized_ars,
            context_lengths=normalized_context_lengths,
            num_splits=split_plan.num_splits,
            split_embedding=split_plan.split_embedding,
            split_lm_head=split_plan.split_lm_head,
            target_soc=resolved_target.chipset,
            dsp_arch=resolved_target.dsp_arch,
            soc_model=resolved_target.soc_model,
            weight_sharing=weight_sharing,
            native_kv=native_kv,
            runtime_supported=False,
            attached_ar_values=tuple(normalized_attachments),
            audio_container_path=audio_container_path,
            text_container_path=text_container_path,
            sdk_container=container,
        )

    def build(
        self,
        spec: Any,
        effective_config: Any,
        output_dir: str | Path,
        *,
        qwen35_runtime_validator: Callable[
            [Qwen35RuntimeValidationRequest], Qwen35RuntimeValidationResult
        ]
        | None = None,
        qwen35_validation_payload: Any = None,
    ) -> BuildResult:
        """Compose the production path: one context per slice+CL, ARs inside.

        The method reads the canonical nested fields by name but remains
        duck-typed so the contracts module can evolve independently.
        """

        require_preflight(self.preflight(spec))
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        def configured(
            paths: Sequence[tuple[str, ...]],
            *,
            default: Any = None,
        ) -> Any:
            value = _first(effective_config, paths)
            return _first(spec, paths, default=default) if value is None else value

        source = _first(
            effective_config,
            (("sources", "text"), ("source", "text"), ("text_source",)),
        )
        if source is None:
            source = _first(spec, (("sources", "text"), ("source", "text"), ("text_source",)))
        model_path = _first(
            source,
            (("onnx_path",), ("onnx",), ("model",), ("model_path",), ("path",)),
            default=source if isinstance(source, (str, Path)) else None,
        )
        if model_path is None:
            raise QairtConfigurationError("sources.text must identify an ONNX model path")
        encodings_path = _first(
            source,
            (("encodings",), ("encodings_path",), ("aimet_encodings",)),
        )

        family_value = _first(
            effective_config,
            (("profile",), ("family",), ("model_family",)),
        )
        if family_value is None:
            family_value = _first(spec, (("family",), ("model_family",)))
        if isinstance(family_value, FamilyProfile):
            family_profile = family_value
        elif family_value is not None:
            family_profile = get_family_profile(family_value)
        else:
            raise QairtConfigurationError("effective_config must contain a resolved family/profile")

        ars = tuple(
            int(value)
            for value in configured(
                (("sequence", "ars"), ("ar_values",)),
                default=(1, 128),
            )
        )
        configured_context_lengths = configured(
            (("sequence", "context_lengths"), ("context_lengths",)),
        )
        if configured_context_lengths is None:
            configured_context_lengths = (
                configured((("context_length",),), default=4096),
            )
        context_lengths = tuple(
            int(value)
            for value in configured_context_lengths
        )
        weight_sharing = bool(
            configured(
                (("sequence", "weight_sharing"), ("weight_sharing",)),
                default=True,
            )
        )
        native_kv = bool(
            configured(
                (("sequence", "native_kv"), ("native_kv",)),
                default=False,
            )
        )
        qwen35_experimental = bool(
            configured(
                (
                    ("sequence", "qwen35_auto_ar_experimental"),
                    ("sequence", "qwen35_experimental_auto_ar"),
                    ("qwen35_auto_ar_experimental",),
                    ("qwen35_experimental_auto_ar",),
                ),
                default=False,
            )
        )
        mha2sha_enabled = bool(
            configured(
                (("transforms", "mha2sha"), ("mha2sha",)),
                default=True,
            )
        )
        permute_kv_cache_io = bool(
            configured(
                (
                    ("transforms", "permute_kv_cache_io"),
                    ("permute_kv_cache_io",),
                ),
                default=False,
            )
        )
        mha2sha_validate = bool(
            configured(
                (("transforms", "mha2sha_validate"), ("mha2sha_validate",)),
                default=False,
            )
        )
        transform_input_list_path = configured(
            (
                ("transforms", "input_raw_list_path"),
                ("input_raw_list_path",),
            ),
        )
        transform_input_base_dir = configured(
            (
                ("transforms", "input_raw_base_dir"),
                ("input_raw_base_dir",),
            ),
        )
        family_transform_options = dict(
            configured(
                (("transforms", "family_options"), ("family_options",)),
                default={},
            )
            or {}
        )

        split_plan = _first(effective_config, (("split_plan",),))
        embedding_value = configured(
            (("split", "embedding_mode"), ("embedding_mode",)),
            default="compiled",
        )
        embedding_mode = str(_enum_value(embedding_value)).lower()
        if not isinstance(split_plan, SplitPlan):
            num_layers = int(
                configured(
                    (("num_hidden_layers",), ("model", "num_hidden_layers")),
                )
            )
            decoder_slices = int(
                configured(
                    (("split", "decoder_slice_count"), ("decoder_slice_count",)),
                    default=1,
                )
            )
            split_plan = build_split_plan(
                num_layers,
                decoder_slices=decoder_slices,
                split_embedding=embedding_mode
                in {"separate", "split", "true", "lut", "compiled", "external"},
                split_lm_head=bool(
                    configured(
                        (("split", "split_lm_head"), ("split_lm_head",)),
                        default=True,
                    )
                ),
            )

        quantization_mode = str(
            _enum_value(
                configured(
                (("quantization", "mode"), ("quantization_mode",)),
                default="apply_encodings" if encodings_path is not None else "float",
                )
            )
        )
        calibration_config = configured(
            (("quantization", "calibration_config"), ("calibration_config",)),
        )
        if quantization_mode == "apply_encodings" and encodings_path is None:
            raise QairtConfigurationError(
                "quantization.mode='apply_encodings' requires AIMET encodings"
            )
        if quantization_mode == "calibrate" and calibration_config is None:
            raise QairtConfigurationError(
                "quantization.mode='calibrate' requires a QAIRT CalibrationConfig"
            )
        if quantization_mode not in {"apply_encodings", "calibrate", "float"}:
            raise QairtConfigurationError(
                "quantization.mode must be apply_encodings, calibrate, or float"
            )

        target_soc = _first(
            spec,
            (
                ("target_soc",),
                ("target", "soc"),
                ("target", "chipset"),
                ("hardware", "soc"),
                ("hardware", "chipset"),
            ),
        )
        dsp_arch = _first(
            spec,
            (("dsp_arch",), ("target", "dsp_arch"), ("hardware", "dsp_arch")),
        )
        soc_model = _first(
            spec,
            (("soc_model",), ("target", "soc_model"), ("hardware", "soc_model")),
        )
        production_compile_options = dict(
            configured(
                (("compile", "compiler_options"), ("compile_config_options",)),
                default={},
            )
            or {}
        )
        misplaced_output_options = {
            "enable_intermediate_outputs",
            "set_output_tensors",
        }.intersection(production_compile_options)
        if misplaced_output_options:
            raise QairtConfigurationError(
                "compile.compiler_options cannot contain diagnostic output fields; "
                "use compile.enable_intermediate_outputs or compile.output_tensors"
            )
        enable_intermediate_outputs = bool(
            configured(
                (("compile", "enable_intermediate_outputs"),),
                default=False,
            )
        )
        output_tensors = tuple(
            configured((("compile", "output_tensors"),), default=()) or ()
        )
        diagnostic_requested = enable_intermediate_outputs or bool(output_tensors)
        diagnostic_compile_options = dict(production_compile_options)
        if enable_intermediate_outputs:
            diagnostic_compile_options["enable_intermediate_outputs"] = True
        if output_tensors:
            diagnostic_compile_options["set_output_tensors"] = list(output_tensors)

        variants: list[ModelVariantArtifact] = []
        transformed: list[TransformedSliceArtifact] = []
        converted: list[ConvertedModelArtifact] = []
        contexts: list[CompiledContextArtifact] = []
        diagnostic_contexts: list[CompiledContextArtifact] = []
        config_artifact_paths: list[Path] = []
        auxiliary_artifact_paths: list[Path] = []

        config_dir = destination / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        build_policy_path = config_dir / "effective_build_policy.json"
        build_policy_path.write_text(
            json.dumps(
                {
                    "family": family_profile.family.value,
                    "ars": ars,
                    "context_lengths": context_lengths,
                    "weight_sharing": weight_sharing,
                    "native_kv": native_kv,
                    "mha2sha": mha2sha_enabled,
                    "permute_kv_cache_io": native_kv or permute_kv_cache_io,
                    "embedding_mode": embedding_mode,
                    "quantization_mode": quantization_mode,
                    "target": {
                        "chipset": str(target_soc),
                        "dsp_arch": str(dsp_arch),
                        "soc_model": int(soc_model),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        config_artifact_paths.append(build_policy_path)
        if (
            family_profile.family is FamilyId.QWEN3_5
            and len(ars) > 1
            and qwen35_runtime_validator is None
        ):
            raise ExperimentalFeatureError(
                "Qwen3.5 automatic multi-AR derivation is fail-closed without "
                "qwen35_runtime_validator and golden-vector reports"
            )

        for context_length in context_lengths:
            variants_for_cl: list[ModelVariantArtifact] = []
            for ar in ars:
                variant = self.ar_convert(
                    model_path,
                    ar=ar,
                    context_length=context_length,
                    encodings_path=encodings_path,
                    output_dir=destination / "variants" / f"cl{context_length}" / f"ar{ar}",
                    family=family_profile,
                    source_kind="derived",
                    allow_experimental_qwen35=qwen35_experimental,
                )
                variants.append(variant)
                variants_for_cl.append(variant)

            slices_by_name: dict[str, list[TransformedSliceArtifact]] = {}
            for variant in variants_for_cl:
                variant_slices = self.transform(
                    variant,
                    split_plan=split_plan,
                    family=family_profile,
                    output_dir=(
                        destination
                        / "transformed"
                        / f"cl{context_length}"
                        / f"ar{variant.ar}"
                    ),
                    mha2sha=mha2sha_enabled,
                    native_kv=native_kv or permute_kv_cache_io,
                    validate=mha2sha_validate,
                    input_raw_list_path=transform_input_list_path,
                    input_raw_base_dir=transform_input_base_dir,
                    m2s_head_split_map=family_transform_options.get(
                        "m2s_head_split_map"
                    ),
                    adapt_moe=family_transform_options.get("adapt_moe"),
                )
                transformed.extend(variant_slices)
                for slice_artifact in variant_slices:
                    slices_by_name.setdefault(slice_artifact.slice_name, []).append(slice_artifact)

            for slice_name, slice_variants in slices_by_name.items():
                ordered_slices = sorted(
                    slice_variants,
                    key=lambda item: int(item.ar or 0),
                )
                if slice_name == "embedding" and embedding_mode in {"lut", "external"}:
                    boundary_path = (
                        config_dir
                        / f"embedding_{embedding_mode}_cl{context_length}_boundary.json"
                    )
                    boundary_variants: list[dict[str, Any]] = []
                    for artifact in ordered_slices:
                        info = self._onnx_inspector.inspect(artifact.model_path)
                        boundary_variants.append(
                            {
                                "ar": artifact.ar,
                                "model_path": str(artifact.model_path),
                                "encodings_path": (
                                    str(artifact.encodings_path)
                                    if artifact.encodings_path is not None
                                    else None
                                ),
                                "inputs": [
                                    {
                                        "name": tensor.name,
                                        "shape": tensor.shape,
                                        "dtype": tensor.dtype,
                                    }
                                    for tensor in info.inputs
                                ],
                                "outputs": [
                                    {
                                        "name": tensor.name,
                                        "shape": tensor.shape,
                                        "dtype": tensor.dtype,
                                    }
                                    for tensor in info.outputs
                                ],
                            }
                        )
                        auxiliary_artifact_paths.append(artifact.model_path)
                        if artifact.encodings_path is not None:
                            auxiliary_artifact_paths.append(artifact.encodings_path)
                    boundary_path.write_text(
                        json.dumps(
                            {
                                "embedding_mode": embedding_mode,
                                "compiled": False,
                                "contract": (
                                    "extract LUT assets"
                                    if embedding_mode == "lut"
                                    else "external embedding boundary"
                                ),
                                "variants": boundary_variants,
                            },
                            indent=2,
                            sort_keys=True,
                            default=str,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    config_artifact_paths.append(boundary_path)
                    continue

                converted_for_slice: list[ConvertedModelArtifact] = []
                graph_names: list[str] = []
                native_expectations: list[NativeKvGraphExpectation] = []
                for slice_artifact in ordered_slices:
                    converted_artifact = self.convert(
                        slice_artifact,
                        encodings_path=(
                            slice_artifact.encodings_path
                            if quantization_mode == "apply_encodings"
                            else None
                        ),
                        calibration_config=(
                            calibration_config if quantization_mode == "calibrate" else None
                        ),
                        output_path=(
                            destination
                            / "converted"
                            / f"cl{context_length}"
                            / slice_name
                            / f"{slice_artifact.model_path.stem}.dlc"
                        ),
                    )
                    converted.append(converted_artifact)
                    converted_for_slice.append(converted_artifact)
                    graph_name = str(
                        getattr(converted_artifact.sdk_model, "name", "")
                        or slice_artifact.model_path.stem
                    )
                    graph_names.append(graph_name)
                    if native_kv and slice_name.startswith("decoder"):
                        info = self._onnx_inspector.inspect(slice_artifact.model_path)
                        native_expectations.append(
                            NativeKvGraphExpectation(
                                graph_name=graph_name,
                                ar=int(slice_artifact.ar or 0),
                                input_names=tuple(item.name for item in info.inputs),
                                output_names=tuple(item.name for item in info.outputs),
                            )
                        )

                native_config: Mapping[str, Any] | None = None
                expect_slice_native_kv = native_kv and slice_name.startswith("decoder")
                if expect_slice_native_kv:
                    native_config = build_native_kv_config(native_expectations)

                selected_ars = [int(item.ar or 0) for item in ordered_slices]
                qwen35_evidence: Qwen35ValidationEvidence | None = None
                if (
                    family_profile.family is FamilyId.QWEN3_5
                    and slice_name.startswith("decoder")
                    and len(converted_for_slice) > 1
                ):
                    validation = self.validate_qwen35_derivation(
                        variants_for_cl,
                        ordered_slices,
                        converted_for_slice,
                        output_dir=(
                            destination
                            / "diagnostics"
                            / "qwen35"
                            / f"cl{context_length}"
                            / slice_name
                        ),
                        slice_name=slice_name,
                        graph_names=graph_names,
                        ar_values=selected_ars,
                        context_length=context_length,
                        target_soc=str(target_soc),
                        dsp_arch=str(dsp_arch),
                        soc_model=int(soc_model),
                        runtime_validator=qwen35_runtime_validator,
                        validation_payload=qwen35_validation_payload,
                        native_kv_config=native_config,
                        native_kv_expectations=native_expectations,
                        expect_native_kv=expect_slice_native_kv,
                    )
                    qwen35_evidence = validation.evidence
                    diagnostic_contexts.extend(validation.diagnostic_contexts)
                    config_artifact_paths.append(validation.structural_report_path)
                    auxiliary_artifact_paths.extend(qwen35_evidence.runtime_report_paths)

                if weight_sharing:
                    contexts.append(
                        self.compile_context(
                            converted_for_slice,
                            output_path=(
                                destination
                                / "contexts"
                                / f"cl{context_length}"
                                / f"{slice_name}.bin"
                            ),
                            graph_names=graph_names,
                            ar_values=selected_ars,
                            source_kinds=["derived"] * len(converted_for_slice),
                            target_soc=str(target_soc),
                            dsp_arch=str(dsp_arch),
                            soc_model=int(soc_model),
                            family=family_profile,
                            slice_name=slice_name,
                            weight_sharing=True,
                            native_kv_config=native_config,
                            native_kv_expectations=native_expectations,
                            expect_native_kv=expect_slice_native_kv,
                            context_length=context_length,
                            qwen35_validation_evidence=qwen35_evidence,
                            compile_config_options=production_compile_options,
                        )
                    )
                else:
                    for model, graph_name, ar in zip(
                        converted_for_slice,
                        graph_names,
                        selected_ars,
                    ):
                        standalone_expectations = [
                            item
                            for item in native_expectations
                            if item.graph_name == graph_name
                        ]
                        standalone_native_config: Mapping[str, Any] | None = None
                        if expect_slice_native_kv:
                            standalone_native_config = build_native_kv_config(
                                standalone_expectations
                            )
                        contexts.append(
                            self.compile_context(
                                [model],
                                output_path=(
                                    destination
                                    / "contexts"
                                    / f"cl{context_length}"
                                    / f"{slice_name}_ar{ar}.bin"
                                ),
                                graph_names=[graph_name],
                                ar_values=[ar],
                                source_kinds=["derived"],
                                target_soc=str(target_soc),
                                dsp_arch=str(dsp_arch),
                                soc_model=int(soc_model),
                                family=family_profile,
                                slice_name=slice_name,
                                weight_sharing=False,
                                native_kv_config=standalone_native_config,
                                native_kv_expectations=standalone_expectations,
                                expect_native_kv=expect_slice_native_kv,
                                context_length=context_length,
                                compile_config_options=production_compile_options,
                            )
                        )

                if diagnostic_requested:
                    diagnostic_root = (
                        destination
                        / "diagnostic_contexts"
                        / f"cl{context_length}"
                    )
                    if weight_sharing:
                        diagnostic_contexts.append(
                            self.compile_context(
                                converted_for_slice,
                                output_path=diagnostic_root / f"{slice_name}.bin",
                                graph_names=graph_names,
                                ar_values=selected_ars,
                                source_kinds=["derived"] * len(converted_for_slice),
                                target_soc=str(target_soc),
                                dsp_arch=str(dsp_arch),
                                soc_model=int(soc_model),
                                family=family_profile,
                                slice_name=slice_name,
                                weight_sharing=True,
                                native_kv_config=native_config,
                                native_kv_expectations=native_expectations,
                                expect_native_kv=expect_slice_native_kv,
                                context_length=context_length,
                                qwen35_validation_evidence=qwen35_evidence,
                                compile_config_options=diagnostic_compile_options,
                            )
                        )
                    else:
                        for model, graph_name, ar in zip(
                            converted_for_slice,
                            graph_names,
                            selected_ars,
                        ):
                            standalone_expectations = [
                                item
                                for item in native_expectations
                                if item.graph_name == graph_name
                            ]
                            standalone_native_config: Mapping[str, Any] | None = None
                            if expect_slice_native_kv:
                                standalone_native_config = build_native_kv_config(
                                    standalone_expectations
                                )
                            diagnostic_contexts.append(
                                self.compile_context(
                                    [model],
                                    output_path=(
                                        diagnostic_root
                                        / f"{slice_name}_ar{ar}.bin"
                                    ),
                                    graph_names=[graph_name],
                                    ar_values=[ar],
                                    source_kinds=["derived"],
                                    target_soc=str(target_soc),
                                    dsp_arch=str(dsp_arch),
                                    soc_model=int(soc_model),
                                    family=family_profile,
                                    slice_name=slice_name,
                                    weight_sharing=False,
                                    native_kv_config=standalone_native_config,
                                    native_kv_expectations=standalone_expectations,
                                    expect_native_kv=expect_slice_native_kv,
                                    context_length=context_length,
                                    compile_config_options=diagnostic_compile_options,
                                )
                            )

        if family_profile.family is FamilyId.QWEN3_VL:
            vision_source = _first(spec, (("sources", "vision"), ("vision_source",)))
            if vision_source is None:
                raise QairtConfigurationError("Qwen3-VL requires sources.vision")
            projector_location = _first(
                spec,
                (("sources", "vision_projector_location"), ("vision_projector_location",)),
            )
            if projector_location != "inside_vision_onnx":
                raise QairtConfigurationError(
                    "Qwen3-VL vision ONNX must include the projector"
                )
            vision_model_path = _first(
                vision_source,
                (("onnx_path",), ("onnx",), ("model_path",), ("path",)),
            )
            vision_encodings_path = _first(
                vision_source,
                (("encodings_path",), ("encodings",), ("aimet_encodings",)),
            )
            if vision_model_path is None:
                raise QairtConfigurationError("sources.vision must identify an ONNX model")

            text_hidden_width = int(
                configured(
                    (("hidden_size",), ("model", "hidden_size")),
                )
            )
            vision_info = self._onnx_inspector.inspect(vision_model_path)
            output_widths = tuple(
                tensor.shape[-1]
                for tensor in vision_info.outputs
                if tensor.shape and isinstance(tensor.shape[-1], int)
            )
            if text_hidden_width not in output_widths:
                raise QairtConfigurationError(
                    "Qwen3-VL vision+projector output hidden width does not match "
                    f"text hidden_size={text_hidden_width}; observed {output_widths}"
                )

            vision_artifact = TransformedSliceArtifact(
                slice_name="vision_projector",
                split_index=0,
                model_path=Path(vision_model_path),
                encodings_path=_path_or_none(vision_encodings_path),
                ar=None,
                context_length=None,
            )
            converted_vision = self.convert(
                vision_artifact,
                encodings_path=(
                    vision_encodings_path
                    if quantization_mode == "apply_encodings"
                    else None
                ),
                calibration_config=(
                    calibration_config if quantization_mode == "calibrate" else None
                ),
                output_path=destination / "converted" / "vision_projector.dlc",
            )
            converted.append(converted_vision)
            vision_graph_name = str(
                getattr(converted_vision.sdk_model, "name", "")
                or Path(vision_model_path).stem
            )
            vision_context = self.compile_context(
                [converted_vision],
                output_path=destination / "contexts" / "vision_projector.bin",
                graph_names=[vision_graph_name],
                ar_values=[1],
                source_kinds=["base"],
                target_soc=str(target_soc),
                dsp_arch=str(dsp_arch),
                soc_model=int(soc_model),
                family=family_profile,
                slice_name="vision_projector",
                weight_sharing=False,
                compile_config_options=production_compile_options,
            )
            contexts.append(vision_context)
            if diagnostic_requested:
                diagnostic_contexts.append(
                    self.compile_context(
                        [converted_vision],
                        output_path=(
                            destination
                            / "diagnostic_contexts"
                            / "vision_projector.bin"
                        ),
                        graph_names=[vision_graph_name],
                        ar_values=[1],
                        source_kinds=["base"],
                        target_soc=str(target_soc),
                        dsp_arch=str(dsp_arch),
                        soc_model=int(soc_model),
                        family=family_profile,
                        slice_name="vision_projector",
                        weight_sharing=False,
                        compile_config_options=diagnostic_compile_options,
                    )
                )

            vision_config_path = _first(
                vision_source,
                (("config_path",), ("config",)),
            )
            text_config_path = _first(source, (("config_path",), ("config",)))
            tokenizer_path = _first(source, (("tokenizer_path",), ("tokenizer",)))
            workflow = self.create_qwen3_vl_workflow_config(
                vision_path=vision_model_path,
                text_path=model_path,
                vision_config_path=vision_config_path,
                text_config_path=text_config_path,
                tokenizer_path=tokenizer_path,
            )
            workflow_dump = (
                workflow.model_dump(mode="json", exclude_none=True)
                if hasattr(workflow, "model_dump")
                else workflow
            )
            workflow_path = config_dir / "qwen3_vl_image_t2t_workflow.json"
            workflow_path.write_text(
                json.dumps(
                    {
                        "python_api": {
                            "workflow_graph_class": (
                                "qairt.gen_ai_api.configs.workflow.WorkflowGraph"
                            ),
                            "workflow_builder_class": (
                                "qairt.gen_ai_api.builders.workflow_builder.WorkflowBuilder"
                            ),
                            "executor_class": (
                                "qairt.gen_ai_api.executors.image_t2t_executor."
                                "ImageT2TExecutor"
                            ),
                        },
                        "projector_location": "inside_vision_onnx",
                        "vision_output_hidden_width": text_hidden_width,
                        "vision_context": str(vision_context.context_binary_path),
                        "text_contexts": [
                            str(item.context_binary_path)
                            for item in contexts
                            if item.slice_name != "vision_projector"
                        ],
                        "workflow_graph": workflow_dump,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            config_artifact_paths.append(workflow_path)

        return BuildResult(
            variants=tuple(variants),
            transformed_slices=tuple(transformed),
            converted_models=tuple(converted),
            contexts=tuple(contexts),
            diagnostic_contexts=tuple(diagnostic_contexts),
            config_artifact_paths=tuple(dict.fromkeys(config_artifact_paths)),
            auxiliary_artifact_paths=tuple(dict.fromkeys(auxiliary_artifact_paths)),
        )
