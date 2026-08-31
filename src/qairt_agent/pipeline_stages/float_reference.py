"""The ONNX Runtime float-graph reference -- debug only, never a default.

Off unless `stage_configs.validation.float_reference` is present. It publishes
its own artifact and block; the supplied-golden comparison is untouched and
remains the production reference. `granularity` accepts `slice_boundary` and
`layer`; layer compares the tapped intermediates and therefore needs an
executed, hash-verified diagnostic context for every slice in scope, failing
closed and naming the slices that lack one. Internal activations are promoted to
outputs in an in-memory copy; the model on disk is never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from qairt_agent.artifacts import (
    ManifestStore,
    atomic_publish_json,
    canonical_json_bytes,
    verify_artifact,
)
from qairt_agent.contracts import (
    ArtifactKind,
    ArtifactRef,
    BuildSpec,
    EmbeddingMode,
    ModelFamily,
    PipelineKind,
    QuantizationMode,
    RunManifest,
    SqnrMode,
    StageExecutionContext,
    StageRecord,
    StageStatus,
    ToolResult,
    VectorMode,
    preset_id_for_family,
    utc_now,
)
from qairt_agent.diagnostics.device_metrics import (
    DEVICE_EXECUTION_METER,
    DEVICE_EXECUTION_SCHEMA,
    aggregate_device_executions,
)
from qairt_agent.diagnostics.latency import LatencyDiagnoser
from qairt_agent.diagnostics.sqnr import QualityDiagnoser, compute_tensor_quality
from qairt_agent.device import DeviceRuntime
from qairt_agent.errors import ErrorCode, InvalidSpecError, ToolErrorData
from qairt_agent.errors import ManifestConflictError
from qairt_agent.families import (
    FamilyConfigGenerator,
    FamilyCrossCheck,
    GeneratedFamilyConfig,
    OnnxInspector,
    apply_lane_benchmark_defaults,
    cross_check_declared_family,
    effective_benchmark_policy,
)
from qairt_agent.compare import compare_runs
from qairt_agent.contracts_reports import (
    MultiArLatencyReport,
    MultiArSqnrReport,
)
from qairt_agent.harness import load_harness_constraints, resolve_target
from qairt_agent.qairt_adapter import (
    LIVE_SDK_FIELDS,
    NativeKvGraphExpectation,
    QairtAdapterFactory,
    QairtAdapterProtocol,
    QairtSdkAdapter,
    Qwen35ValidationEvidence,
    Qwen35RuntimeValidationResult,
    require_preflight,
)
from qairt_agent.qairt_adapter.errors import (
    ExperimentalFeatureError,
    QairtAdapterError,
    QairtConfigurationError,
    QairtPreflightError,
    QairtSdkImportError,
)
from qairt_agent.runtime.chain import SliceChainRunner, SliceRoute
from qairt_agent.runtime.index import (
    load_runtime_index,
    make_runtime_index,
    select_runtime_binding,
)
from qairt_agent.vector_retarget import (
    VectorRetargetError,
    retarget_vector_manifest,
    validate_provided_ar_manifest,
)
from qairt_agent.vectors import TensorSource, VectorPreparer, sha256_file

# Shared helpers live in pipeline_support so the stage modules can use
# them without importing this facade. Re-exported here because existing
# call sites and tests import them from `qairt_agent.pipeline`.
from qairt_agent.pipeline_stages.benchmark import BenchmarkStage
from qairt_agent.pipeline_stages.diagnose import DiagnoseStage
from qairt_agent.pipeline_stages.execution import ExecutionStage
from qairt_agent.pipeline_support import (  # noqa: F401
    run_directory,
    DEVICE_EXECUTION_METER,
    DEVICE_EXECUTION_SAMPLES,
    DEVICE_EXECUTION_SCHEMA,
    LIVE_SDK_FIELDS,
    _AUTOMATIC_DIAGNOSE_KEYS,
    _DEPLOYABLE_FOOTPRINT_ROLES,
    _DEVICE_EXECUTION_UNAVAILABLE,
    _EXECUTION_ATTEMPT_METADATA,
    _LIVE_SDK_FIELDS,
    _LOW_LEVEL_CHAIN_CONFIG_FIELDS,
    _OUTPUT_ONLY_CONFIG_FIELDS,
    _SDK_GENERATED_TOKEN_COUNT_KEYS,
    _STATIC_FOOTPRINT_SCHEMA,
    _artifact_kind,
    _config_input_artifacts,
    _jsonable,
    _layer_float_reference,
    _output_mapping,
    _path_artifacts,
    _sdk_generated_token_count,
    _stage_key_value,
    _static_footprint,
    _unique_artifacts,
    hashlib,
    json,
    np,
    os,
    re,
)




class FloatReferenceStage:
    """FloatReferenceStage — see the module docstring."""

    def _diagnostic_device_outputs(
        self,
        manifest: RunManifest,
        effective: Mapping[str, Any],
        adapter: Any,
        *,
        device: Any,
        ar: int,
        inputs: Mapping[str, np.ndarray],
        initial_native_state: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Execute the diagnostic contexts and collect their tapped tensors.

        The build has always compiled these contexts and verified their hashes,
        but nothing ever ran them: an ``op_level_dump_available`` claim that
        rests on a context's *existence* is not operator evidence. This is the
        step that turns it into evidence.

        Fails closed rather than degrading -- a layer-level report that quietly
        fell back to slice boundaries would be the same overclaim in a new
        place.
        """

        index = self._runtime_index_for_manifest(manifest)
        cl_key = str(int(effective["context_length"]))
        entries = list((index.get("diagnostic_contexts") or {}).get(cl_key) or ())
        if not entries:
            raise InvalidSpecError(
                "layer-level float reference requires diagnostic contexts, and "
                "this build produced none; set quality."
                "dump_intermediates_on_failure or compile."
                "enable_intermediate_outputs and rebuild",
                stage="validate",
                details={"context_length": cl_key},
            )

        artifacts_by_path = {
            artifact.path.expanduser().resolve(): artifact
            for artifact in manifest.artifacts
        }
        loaded: dict[str, Any] = {}
        graphs: dict[str, str] = {}
        executed: list[dict[str, Any]] = []
        for entry in entries:
            path = Path(str(entry["context_path"])).expanduser().resolve()
            artifact = artifacts_by_path.get(path)
            if artifact is None:
                raise InvalidSpecError(
                    "runtime_index references a diagnostic context that is not "
                    "a verified build artifact",
                    stage="validate",
                    details={"context_path": os.fspath(path)},
                )
            verify_artifact(artifact)
            graph_name = (entry.get("graphs_by_ar") or {}).get(str(int(ar)))
            if graph_name is None:
                raise InvalidSpecError(
                    "the diagnostic context carries no graph for the bound AR",
                    stage="validate",
                    details={
                        "context_path": os.fspath(path),
                        "ar": int(ar),
                        "available_ars": sorted(
                            (entry.get("graphs_by_ar") or {})
                        ),
                    },
                )
            slice_name = str(entry.get("slice") or "model")
            loaded[slice_name] = (
                adapter.load_compiled(path)
                if hasattr(adapter, "load_compiled")
                else path
            )
            graphs[slice_name] = str(graph_name)
            executed.append(
                {
                    "slice": slice_name,
                    "graph_name": str(graph_name),
                    "artifact": _jsonable(artifact),
                }
            )

        routes = effective.get("routes")
        execution_options = self._execution_options(effective)
        native_io = bool(effective.get("native_io", False))
        if routes:
            # Diagnostic contexts come from one build's manifest. A chain
            # assembled from independently built contexts therefore has no
            # diagnostic context for the slices that came from another run, and
            # running it anyway would fail deep inside the chain runner with a
            # missing-executor error instead of naming the cause.
            missing = [
                str(route.get("slice_id"))
                for route in routes
                if str(route.get("slice_id")) not in loaded
            ]
            if missing:
                raise InvalidSpecError(
                    "layer granularity needs a diagnostic context for every "
                    "chain slice, and this build produced none for some of "
                    "them; a chain assembled from separately built contexts "
                    "cannot be drilled into, because diagnostic contexts belong "
                    "to the build that made them",
                    stage="validate",
                    details={
                        "slices_without_diagnostic_context": missing,
                        "slices_with_diagnostic_context": sorted(loaded),
                    },
                )
            # Reuse the production chain wiring with the diagnostic contexts
            # substituted, so each slice is fed exactly what it is fed in a
            # real run instead of a guess at its inputs.
            runner = SliceChainRunner(
                routes,
                self._chain_executors(
                    adapter,
                    loaded,
                    device=device,
                    native_io=native_io,
                    execution_options=execution_options,
                ),
            )
            result = runner.run_device_chain(
                dict(inputs),
                ar=int(ar),
                initial_native_state=dict(initial_native_state or {}),
            )
            outputs = {
                str(name): dict(values)
                for name, values in result.outputs_by_slice().items()
            }
        elif len(loaded) == 1:
            slice_name, compiled = next(iter(loaded.items()))
            raw = adapter.run_graph(
                compiled,
                dict(inputs),
                graph_name=graphs[slice_name],
                device=device,
                native_io=native_io,
                **execution_options,
            )
            outputs = {
                slice_name: dict(
                    _output_mapping(raw, graph_name=graphs[slice_name])
                )
            }
        else:
            raise InvalidSpecError(
                "several diagnostic contexts but no routes: each slice's "
                "diagnostic inputs come from the previous slice, so a chain "
                "definition is required rather than guessed inputs",
                stage="validate",
                details={"diagnostic_slices": sorted(loaded)},
            )

        evidence = {
            "executed_contexts": executed,
            "context_count": len(executed),
            "execution": "device_chain" if routes else "single_graph",
            "tensor_counts": {
                name: len(values) for name, values in outputs.items()
            },
        }
        return outputs, evidence



    def _float_reference_report(
        self,
        manifest: RunManifest,
        effective: Mapping[str, Any],
        *,
        device_outputs: Mapping[str, Mapping[str, Any]] | None,
        output_dir: Path,
        report_suffix: str,
        diagnostic_outputs: Mapping[str, Mapping[str, Any]] | None = None,
        diagnostic_evidence: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...]]:
        """Compare device slice boundaries against an ONNX Runtime float run.

        Debug-only. Nothing here runs unless
        ``stage_configs.validation.float_reference`` is present, and the result
        is published beside the production report rather than replacing any
        part of it: the AIMET golden comparison remains the production
        reference.
        """

        config = effective.get("float_reference")
        if config is None:
            return {}, ()
        if not isinstance(config, Mapping):
            raise InvalidSpecError(
                "stage_configs.validation.float_reference must be an object",
                stage="validate",
            )

        granularity = str(config.get("granularity", "slice_boundary"))
        if granularity not in {"slice_boundary", "layer"}:
            raise InvalidSpecError(
                "float reference granularity must be 'slice_boundary' or "
                "'layer'",
                stage="validate",
                details={"requested_granularity": granularity},
            )
        if granularity == "layer" and not diagnostic_outputs:
            # Degrading to slice boundaries here would republish the exact
            # overclaim this granularity exists to retire.
            raise InvalidSpecError(
                "layer granularity requires executed diagnostic contexts; none "
                "were collected for this run",
                stage="validate",
                details={"requested_granularity": granularity},
            )
        if config.get("ar") is None:
            raise InvalidSpecError(
                "the float reference is a single-AR debug mode: set an "
                "explicit float_reference.ar",
                stage="validate",
            )
        requested_ar = int(config["ar"])
        bound_ar = effective.get("ar")
        if bound_ar is not None and int(bound_ar) != requested_ar:
            raise InvalidSpecError(
                "float_reference.ar must match the AR the validation run is "
                "bound to",
                stage="validate",
                details={
                    "float_reference_ar": requested_ar,
                    "runtime_binding_ar": int(bound_ar),
                },
            )
        if not device_outputs:
            raise InvalidSpecError(
                "the float reference compares device slice boundaries, so it "
                "requires a device chain run; supply routes/contexts or "
                "device_chain_outputs",
                stage="validate",
            )
        if effective.get("vector_manifest") is None:
            raise InvalidSpecError(
                "the float reference must be fed the same inputs as the "
                "device run; no vector manifest is bound",
                stage="validate",
            )

        model_path = (
            config.get("model_path")
            or effective.get("reference_model_path")
            or manifest.build_spec.sources.text.onnx_path
        )
        resolved_model = Path(str(model_path)).expanduser().resolve()
        if not resolved_model.is_file():
            raise InvalidSpecError(
                "the float reference model does not exist",
                stage="validate",
                details={"model_path": os.fspath(resolved_model)},
            )

        supplied_map = config.get("tensor_map") or {}
        if not isinstance(supplied_map, Mapping):
            raise InvalidSpecError(
                "float_reference.tensor_map must be an object",
                stage="validate",
            )

        def mapped_name(slice_name: str, tensor_name: str) -> str | None:
            nested = supplied_map.get(slice_name)
            if isinstance(nested, Mapping) and tensor_name in nested:
                return str(nested[tensor_name])
            direct = supplied_map.get(tensor_name)
            if isinstance(direct, str):
                return direct
            return None

        # Layer granularity compares the tapped intermediates; the boundary
        # tensors stay in the same comparison so a run shows both.
        compared_outputs: dict[str, dict[str, Any]] = {
            str(name): dict(values) for name, values in device_outputs.items()
        }
        # A tensor can legitimately appear from both paths -- the production
        # context's boundary and the diagnostic context's tap. Recording which
        # produced each observation keeps the duplicate readable instead of
        # looking like the same tensor measured twice for no reason.
        tensor_source: dict[tuple[str, str], str] = {
            (str(name), str(tensor)): "production_boundary"
            for name, values in device_outputs.items()
            for tensor in values
        }
        if diagnostic_outputs:
            for name, values in diagnostic_outputs.items():
                compared_outputs.setdefault(str(name), {}).update(dict(values))
                for tensor in values:
                    tensor_source[(str(name), str(tensor))] = "diagnostic_context" 

        producible = VectorPreparer.onnx_producible_tensor_names(resolved_model)
        pairs: list[tuple[str, str, str]] = []
        unmapped: list[dict[str, Any]] = []
        for slice_name, tensors in compared_outputs.items():
            for tensor_name in tensors:
                explicit = mapped_name(str(slice_name), str(tensor_name))
                candidate = explicit if explicit is not None else str(tensor_name)
                if candidate in producible:
                    pairs.append((str(slice_name), str(tensor_name), candidate))
                    continue
                unmapped.append(
                    {
                        "slice": str(slice_name),
                        "tensor": str(tensor_name),
                        "attempted_float_tensor": candidate,
                        "reason": (
                            "explicit tensor_map entry is not produced by the "
                            "float graph"
                            if explicit is not None
                            else "no exact name match in the float graph and no "
                            "tensor_map entry"
                        ),
                    }
                )
        if not pairs:
            raise InvalidSpecError(
                "no device boundary tensor could be bound to the float graph; "
                "supply float_reference.tensor_map — boundary names are never "
                "guessed",
                stage="validate",
                details={"unmapped_tensors": unmapped},
            )

        inputs = self._manifest_inputs(
            effective["vector_manifest"],
            section=str(effective.get("input_section", "inputs")),
            sha256=effective.get("vector_manifest_sha256"),
        )
        providers = tuple(
            str(item) for item in config.get("providers", ("CPUExecutionProvider",))
        )
        captured, provenance = VectorPreparer.capture_onnx_float_activations(
            resolved_model,
            inputs,
            [float_name for _, _, float_name in pairs],
            providers=providers,
        )

        floor = float(effective.get("reference_energy_floor", 0.0))
        observations: list[dict[str, Any]] = []
        for slice_name, tensor_name, float_name in pairs:
            quality = compute_tensor_quality(
                captured[float_name],
                compared_outputs[slice_name][tensor_name],
                reference_energy_floor=floor,
            )
            observations.append(
                {
                    "slice": slice_name,
                    "tensor": tensor_name,
                    "float_tensor": float_name,
                    "device_tensor_source": tensor_source.get(
                        (slice_name, tensor_name), "production_boundary"
                    ),
                    "quality": quality.to_dict(),
                }
            )

        # Ordered by the float graph's own topology so the first divergence is
        # the first row, not something the reader has to search for.
        topology = {name: order for order, name in enumerate(producible)}
        observations.sort(
            key=lambda item: topology.get(item["float_tensor"], len(topology))
        )

        payload = {
            "schema": "qairt-agent.float-reference-report/1",
            "mode": "debug_only",
            "policy": "report_only",
            "granularity": granularity,
            "ar": requested_ar,
            "claim_scope": "first_observed_divergence_not_root_cause",
            "comparison": "device_chain_vs_onnxruntime_float_graph",
            "ordered_by": "float_graph_topology",
            "op_level_dump_available": bool(diagnostic_outputs),
            "tensor_map": {
                slice_name: {tensor_name: float_name}
                for slice_name, tensor_name, float_name in pairs
            },
            "unmapped_tensors": unmapped,
            "observations": observations,
            **provenance,
        }
        if diagnostic_evidence is not None:
            payload["diagnostic_contexts"] = dict(diagnostic_evidence)
        report_ref = atomic_publish_json(
            output_dir / f"float_reference_report{report_suffix}.json",
            payload,
            kind=ArtifactKind.REPORT,
            logical_name=f"float_reference_report{report_suffix}",
        )
        return payload, (report_ref,)



    @classmethod
    def _diagnostic_context_evidence(
        cls,
        manifest: RunManifest,
        effective: Mapping[str, Any],
        *,
        requested: bool,
        divergence_observed: bool,
        slice_reference_refs: Sequence[ArtifactRef],
    ) -> dict[str, Any]:
        """Describe verified diagnostic evidence without overstating its scope."""

        evidence: dict[str, Any] = {
            "requested": bool(requested),
            "triggered": bool(requested and divergence_observed),
            "reason": (
                "observed_numerical_divergence"
                if requested and divergence_observed
                else "not_triggered"
            ),
        }
        if not requested or not divergence_observed:
            return evidence

        index = cls._runtime_index_for_manifest(manifest)
        cl_key = str(int(effective["context_length"]))
        entries = list(
            (index.get("diagnostic_contexts") or {}).get(cl_key) or ()
        )
        artifacts_by_path = {
            artifact.path.expanduser().resolve(): artifact
            for artifact in manifest.artifacts
        }
        contexts: list[dict[str, Any]] = []
        for entry in entries:
            path = Path(str(entry["context_path"])).expanduser().resolve()
            artifact = artifacts_by_path.get(path)
            if artifact is None:
                raise InvalidSpecError(
                    "runtime_index references a diagnostic context that is "
                    "not a verified build artifact",
                    stage="validate",
                    details={"context_path": os.fspath(path)},
                )
            verify_artifact(artifact)
            contexts.append(
                {
                    **dict(entry),
                    "artifact": _jsonable(artifact),
                }
            )
        if contexts:
            evidence.update(
                {
                    "status": "ready",
                    "evidence_scope": "op_intermediate_contexts",
                    "op_level_dump_available": True,
                    "contexts": contexts,
                }
            )
            return evidence

        evidence.update(
            {
                "status": "slice_tensor_only",
                "evidence_scope": "verified_slice_tensor_boundaries",
                "op_level_dump_available": False,
                "contexts": [],
                "slice_reference_manifests": [
                    _jsonable(ref) for ref in slice_reference_refs
                ],
                "limitation": (
                    "the build produced no diagnostic context; the report may "
                    "localize a slice/tensor boundary but cannot claim an "
                    "operator-level intermediate dump"
                ),
            }
        )
        return evidence



    @staticmethod
    def _capture_onnx_reference(
        vector_manifest: str | Path,
        model_path: str | Path,
        output_dir: Path,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> Path:
        """Copy inputs into the run tree and capture an auditable ORT reference."""

        source_path = Path(vector_manifest).expanduser().resolve()
        source = VectorPreparer.load_manifest(
            source_path,
            expected_sha256=expected_manifest_sha256,
        )
        inputs = VectorPreparer.load_tensors(
            source_path,
            section="inputs",
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if not inputs:
            raise ValueError(
                "ONNX Runtime fallback requires a vector manifest with model inputs"
            )
        preparer = VectorPreparer(output_dir / "onnxruntime-reference")
        copied = preparer.prepare_case(
            source.case_id,
            inputs,
            metadata={
                **dict(source.metadata),
                "reference_request": "golden_missing_fallback",
                "source_manifest_path": os.fspath(source_path),
                "source_manifest_sha256": sha256_file(source_path),
            },
        )
        return preparer.capture_onnx(
            copied,
            model_path,
            destination_name="vector_manifest.onnx-reference.json",
        )


