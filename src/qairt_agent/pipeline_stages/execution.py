"""Executing graphs and slice chains on the device, and metering them.

Latency means device time. The wall-clock number these runners produce is host
time around one SDK call -- on the low-level lane QAIRT relaunches
``qnn-net-run`` per call, so per-call context load and HVX/HMX power-on sit
inside the sample -- and is quarantined under ``harness_diagnostics``. The
device number comes from QAIRT's own profiling log, captured **before**
``initialize_execution``, because an initialized model carries an execution
context built with profiling disabled.
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
from qairt_agent.pipeline_stages.diagnose import DiagnoseStage
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



class ExecutionStage:
    """ExecutionStage — see the module docstring."""

    @staticmethod
    def _execution_options(config: Mapping[str, Any]) -> dict[str, Any]:
        options = dict(config.get("execution_options", {}))
        if "device" in options:
            raise ValueError(
                "execution_options.device is controlled by "
                "QAIRT_AGENT_ADB_SERIAL/QAIRT_AGENT_ADB_SERVER"
            )
        return options


    @staticmethod
    def _device_execution_block(
        adapter: Any,
        compiled: Any,
        inputs: Mapping[str, np.ndarray],
        *,
        graph_name: str,
        device: Any,
        native_io: bool,
        execution_options: Mapping[str, Any],
        working_dir: Path,
        repeats: int = DEVICE_EXECUTION_SAMPLES,
    ) -> dict[str, Any] | None:
        """Device-side execute evidence, or ``None`` when unavailable.

        The profiled execute is repeated and averaged, because this is the
        latency metric and one sample cannot show a regression.  An adapter
        that cannot profile still produces a valid benchmark, so a failed
        capture degrades the report rather than failing the stage -- what it
        must never do is publish a device claim it did not measure.
        """

        capture = getattr(adapter, "capture_device_execution", None)
        if not callable(capture):
            return {
                "schema": DEVICE_EXECUTION_SCHEMA,
                "policy": "report_only",
                "available": False,
                "reason": (
                    "the QAIRT adapter does not expose capture_device_execution"
                ),
            }
        try:
            blocks = [
                capture(
                    compiled,
                    inputs,
                    graph_name=graph_name,
                    device=device,
                    native_io=native_io,
                    working_dir=working_dir,
                    **dict(execution_options),
                )
                for _ in range(max(1, int(repeats)))
            ]
            return aggregate_device_executions(
                blocks, requested=max(1, int(repeats))
            )
        except Exception as error:  # report-only: never fail the benchmark
            return {
                "schema": DEVICE_EXECUTION_SCHEMA,
                "policy": "report_only",
                "available": False,
                "reason": f"{type(error).__name__}: {error}",
            }


    def run_graph(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> ToolResult[dict[str, Any]]:
        selected = dict(config or {})

        def operation(
            manifest: RunManifest, adapter: Any, output_dir: Path
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            self._preflight(adapter, manifest.build_spec)
            context_path = selected.get("context_path")
            graph_name = selected.get("graph_name")
            vector_manifest = selected.get("vector_manifest")
            if context_path is None or not graph_name or vector_manifest is None:
                raise ValueError(
                    "run_graph requires context_path, graph_name, and vector_manifest"
                )
            inputs = self._manifest_inputs(
                vector_manifest,
                section=str(selected.get("input_section", "inputs")),
                sha256=selected.get("vector_manifest_sha256"),
            )
            push_files = self._device_stage_files(
                output_dir,
                contexts=(context_path,),
                vector_manifests=(vector_manifest,),
            )
            with self._device_stage(
                manifest,
                adapter,
                stage_name="run_graph",
                input_manifest_sha256=manifest_sha256,
                stage_config=selected,
                push_files=push_files,
            ) as device_stage:
                raw_result = adapter.run_graph(
                    context_path,
                    inputs,
                    graph_name=str(graph_name),
                    device=device_stage.device,
                    native_io=bool(selected.get("native_io", False)),
                    **self._execution_options(selected),
                )
                outputs = _output_mapping(raw_result, graph_name=str(graph_name))
                device_identifier = device_stage.identifier
                device_soc = device_stage.soc_verification
                remote_attempt_dir = device_stage.adb.attempt_dir
            result_manifest = VectorPreparer(output_dir).prepare_case(
                "run_graph",
                inputs,
                goldens=outputs,
                metadata={"graph_name": str(graph_name), "role": "device_outputs"},
            )
            refs = (
                ArtifactRef.from_path(
                    result_manifest,
                    kind=ArtifactKind.TEST_VECTORS,
                    logical_name="run_graph_outputs",
                ),
            )
            return {
                "graph_name": str(graph_name),
                "output_manifest": _jsonable(refs[0]),
                "outputs": _jsonable(outputs),
            }, refs, {
                "output_count": len(outputs),
                "device_identifier": device_identifier,
                "device_soc": device_soc,
                "remote_attempt_dir": remote_attempt_dir,
                "remote_cleanup": "confirmed",
            }

        return self._continuation_operation(
            "run_graph",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
        )


    @staticmethod
    def _chain_executors(
        adapter: Any,
        contexts: Mapping[str, Any],
        *,
        device: Any,
        native_io: bool,
        execution_options: Mapping[str, Any],
    ) -> dict[str, Callable[[Mapping[str, np.ndarray], Any], Mapping[str, np.ndarray]]]:
        executors: dict[
            str, Callable[[Mapping[str, np.ndarray], Any], Mapping[str, np.ndarray]]
        ] = {}
        for slice_name, context in contexts.items():
            def execute(
                inputs: Mapping[str, np.ndarray],
                invocation: Any,
                *,
                compiled_context: Any = context,
            ) -> Mapping[str, np.ndarray]:
                result = adapter.run_graph(
                    compiled_context,
                    inputs,
                    graph_name=invocation.graph_name,
                    device=device,
                    native_io=native_io,
                    **dict(execution_options),
                )
                return _output_mapping(result, graph_name=invocation.graph_name)

            executors[str(slice_name)] = execute
        return executors


    @staticmethod
    def _recording_chain_executors(
        base: Mapping[str, Callable[[Mapping[str, np.ndarray], Any], Mapping[str, np.ndarray]]],
        recorded: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Callable[[Mapping[str, np.ndarray], Any], Mapping[str, np.ndarray]]]:
        """Wrap chain executors so each slice's exact inputs are captured.

        A profiled per-slice execute needs the inputs that slice is really fed,
        and in a chain those come from the previous slice at run time. Recording
        them during one ordinary pass is what lets the device capture replay
        each slice faithfully instead of guessing.

        Every invocation is appended, not overwritten. A sequence (prefill then
        decode, say) runs each slice once per step, and keeping only the last
        left the published block covering the final step while labelled as the
        whole chain.
        """

        wrapped: dict[
            str, Callable[[Mapping[str, np.ndarray], Any], Mapping[str, np.ndarray]]
        ] = {}
        for slice_name, executor in base.items():
            def record(
                inputs: Mapping[str, np.ndarray],
                invocation: Any,
                *,
                bound: str = str(slice_name),
                inner: Any = executor,
            ) -> Mapping[str, np.ndarray]:
                entries = recorded.setdefault(bound, [])
                entries.append(
                    {
                        "step_index": len(entries),
                        "inputs": {
                            str(name): np.asarray(value)
                            for name, value in inputs.items()
                        },
                        "graph_name": str(invocation.graph_name),
                        "ar": getattr(invocation, "ar", None),
                    }
                )
                return inner(inputs, invocation)

            wrapped[str(slice_name)] = record
        return wrapped


    def _chain_device_execution(
        self,
        adapter: Any,
        recorded: Mapping[str, Sequence[Mapping[str, Any]]],
        contexts: Mapping[str, Any],
        *,
        device: Any,
        native_io: bool,
        execution_options: Mapping[str, Any],
        working_dir: Path,
    ) -> dict[str, Any]:
        """Per-slice device execute time for a chain run.

        Each slice is profiled with the inputs it was actually fed, recorded
        during the preceding ordinary chain pass.

        A sequence runs each slice once per step. Every recorded step is
        profiled and the block says how many it covered, so a prefill+decode
        run can never publish decode-only evidence under an unqualified
        ``scope="chain"`` label.
        """

        steps_total = max((len(entries) for entries in recorded.values()), default=0)
        by_step: list[dict[str, Any]] = []
        for step_index in range(steps_total):
            step_slices: dict[str, Any] = {}
            for slice_name, entries in recorded.items():
                if step_index >= len(entries):
                    continue
                context = contexts.get(slice_name)
                if context is None:
                    continue
                entry = entries[step_index]
                step_slices[str(slice_name)] = {
                    **self._device_execution_block(
                        adapter,
                        context,
                        entry["inputs"],
                        graph_name=str(entry["graph_name"]),
                        device=device,
                        native_io=native_io,
                        execution_options=execution_options,
                        working_dir=(
                            working_dir / str(slice_name)
                            if steps_total == 1
                            else working_dir / f"step{step_index}" / str(slice_name)
                        ),
                    ),
                    "graph_name": str(entry["graph_name"]),
                    **({"ar": entry["ar"]} if entry.get("ar") is not None else {}),
                }
            by_step.append({"step_index": step_index, "by_slice": step_slices})

        def _measured(step_slices: Mapping[str, Any]) -> dict[str, Any]:
            return {
                name: value
                for name, value in step_slices.items()
                if isinstance(value, Mapping) and value.get("available") is not False
            }

        def _totals(step_slices: Mapping[str, Any]) -> dict[str, Any] | None:
            measured = _measured(step_slices)
            if not measured or len(measured) != len(step_slices):
                return None
            # Chain slices run sequentially, so their device execute times add.
            # This is a sum of per-slice means, not a measured end-to-end
            # number, and says so.
            return {
                key: sum(
                    float(item[key])
                    for item in measured.values()
                    if isinstance(item.get(key), (int, float))
                )
                for key in (
                    "accelerator_compute_us",
                    "accelerator_execute_us",
                    "qnn_execute_us",
                )
                if all(
                    isinstance(item.get(key), (int, float))
                    for item in measured.values()
                )
            }

        block: dict[str, Any] = {
            "schema": DEVICE_EXECUTION_SCHEMA,
            "meter": DEVICE_EXECUTION_METER,
            "lane": "low_level",
            "policy": "report_only",
            "scope": "chain" if steps_total <= 1 else "chain_sequence",
            "statistic": "mean",
            "steps_total": steps_total,
            "steps_covered": len(by_step),
        }
        if steps_total <= 1:
            # One pass over the slices: the documented one-block-per-slice shape.
            step_slices = by_step[0]["by_slice"] if by_step else {}
            measured = _measured(step_slices)
            block["slice_count"] = len(step_slices)
            block["measured_slice_count"] = len(measured)
            block["by_slice"] = step_slices
            totals = _totals(step_slices)
            if totals is not None:
                block["totals"] = totals
                block["totals_basis"] = (
                    "sum_of_per_slice_means_slices_run_sequentially"
                )
            else:
                block["available"] = False
                block["reason"] = (
                    "not every chain slice produced device evidence; a partial "
                    "chain total would understate the work"
                )
            return block

        # A sequence: per-step evidence, never collapsed into one unlabelled
        # set of slices.
        complete = True
        for step in by_step:
            totals = _totals(step["by_slice"])
            step["slice_count"] = len(step["by_slice"])
            step["measured_slice_count"] = len(_measured(step["by_slice"]))
            if totals is None:
                complete = False
            else:
                step["totals"] = totals
                step["totals_basis"] = (
                    "sum_of_per_slice_means_slices_run_sequentially"
                )
        block["by_step"] = by_step
        if not complete:
            block["available"] = False
            block["reason"] = (
                "not every chain slice produced device evidence in every step; "
                "a partial chain total would understate the work"
            )
        else:
            block["totals"] = {
                key: sum(
                    float(step["totals"][key])
                    for step in by_step
                    if key in step.get("totals", {})
                )
                for key in (
                    "accelerator_compute_us",
                    "accelerator_execute_us",
                    "qnn_execute_us",
                )
                if all(key in step.get("totals", {}) for step in by_step)
            }
            block["totals_basis"] = (
                "sum_over_all_steps_of_per_slice_means_slices_run_sequentially"
            )
        return block


    def run_chain(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> ToolResult[dict[str, Any]]:
        selected = dict(config or {})

        def operation(
            manifest: RunManifest, adapter: Any, output_dir: Path
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            self._preflight(adapter, manifest.build_spec)
            routes = selected.get("routes")
            contexts = selected.get("contexts")
            if not routes or not isinstance(contexts, Mapping):
                raise ValueError("run_chain requires config.routes and config.contexts")
            mode = str(selected.get("mode", "device_chain"))
            inline_cases: list[Mapping[str, Any]] = []
            if "steps" in selected:
                inline_cases.extend(
                    dict(step.get("inputs", {}))
                    for step in selected["steps"]
                    if isinstance(step, Mapping)
                )
            initial_native_state = self._tensor_mapping(
                selected.get("initial_native_state", {})
            )
            if initial_native_state:
                inline_cases.append(initial_native_state)
            vector_manifests = (
                (selected["vector_manifest"],)
                if selected.get("vector_manifest") is not None
                else ()
            )
            push_files = self._device_stage_files(
                output_dir,
                contexts=tuple(contexts.values()),
                vector_manifests=vector_manifests,
                inline_cases=tuple(inline_cases),
            )
            execution_options = self._execution_options(selected)
            with self._device_stage(
                manifest,
                adapter,
                stage_name="run_chain",
                input_manifest_sha256=manifest_sha256,
                stage_config=selected,
                push_files=push_files,
            ) as device_stage:
                runner = SliceChainRunner(
                    routes,
                    self._chain_executors(
                        adapter,
                        contexts,
                        device=device_stage.device,
                        native_io=bool(selected.get("native_io", False)),
                        execution_options=execution_options,
                    ),
                )
                device_identifier = device_stage.identifier
                device_soc = device_stage.soc_verification
                remote_attempt_dir = device_stage.adb.attempt_dir
                if "steps" in selected:
                    result = runner.run_sequence(
                        selected["steps"],
                        mode=mode,
                        initial_native_state=initial_native_state,
                    )
                    last = result.steps[-1]
                else:
                    vector_manifest = selected.get("vector_manifest")
                    if vector_manifest is None:
                        raise ValueError(
                            "run_chain requires vector_manifest when steps are absent"
                        )
                    inputs = self._manifest_inputs(
                        vector_manifest,
                        sha256=selected.get("vector_manifest_sha256"),
                    )
                    ar = int(
                        selected.get("ar", manifest.build_spec.sequence.ars[0])
                    )
                    if mode == "teacher_forced":
                        teacher_inputs = self._slice_tensor_tree(
                            selected.get("teacher_inputs", {}),
                            section="inputs",
                        )
                        result = runner.run_teacher_forced(
                            inputs,
                            teacher_inputs,
                            ar=ar,
                            initial_native_state=initial_native_state,
                        )
                    else:
                        result = runner.run_device_chain(
                            inputs,
                            ar=ar,
                            initial_native_state=initial_native_state,
                        )
                    last = result

            refs: list[ArtifactRef] = []
            output_manifests: dict[str, Any] = {}
            for slice_name, outputs in last.outputs_by_slice().items():
                path = VectorPreparer(output_dir / slice_name).prepare_case(
                    f"{mode}_{slice_name}",
                    outputs,
                    roles={name: "device_output" for name in outputs},
                    metadata={"slice": slice_name, "mode": mode},
                )
                ref = ArtifactRef.from_path(
                    path,
                    kind=ArtifactKind.TEST_VECTORS,
                    logical_name=f"{mode}_{slice_name}_outputs",
                )
                refs.append(ref)
                output_manifests[slice_name] = _jsonable(ref)
            return {
                "mode": mode,
                "output_manifests": output_manifests,
                "final_outputs": _jsonable(last.final_outputs),
                "native_state_slots": list(last.native_state_slots),
            }, tuple(refs), {
                "slice_count": len(refs),
                "device_identifier": device_identifier,
                "device_soc": device_soc,
                "remote_attempt_dir": remote_attempt_dir,
                "remote_cleanup": "confirmed",
            }

        return self._continuation_operation(
            "run_chain",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
        )


    def profile(
        self,
        manifest_uri: str | Path,
        manifest_sha256: str,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> ToolResult[dict[str, Any]]:
        selected = dict(config or {})

        def operation(
            manifest: RunManifest, adapter: Any, output_dir: Path
        ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
            self._preflight(adapter, manifest.build_spec)
            required = ("context_path", "graph_name", "vector_manifest")
            missing = [key for key in required if selected.get(key) is None]
            if missing:
                raise ValueError(f"profile missing config fields: {missing}")
            inputs = self._manifest_inputs(
                selected["vector_manifest"],
                sha256=selected.get("vector_manifest_sha256"),
            )
            push_files = self._device_stage_files(
                output_dir,
                contexts=(selected["context_path"],),
                vector_manifests=(selected["vector_manifest"],),
            )
            with self._device_stage(
                manifest,
                adapter,
                stage_name="profile",
                input_manifest_sha256=manifest_sha256,
                stage_config=selected,
                push_files=push_files,
            ) as device_stage:
                result = adapter.profile(
                    selected["context_path"],
                    inputs,
                    graph_name=str(selected["graph_name"]),
                    device=device_stage.device,
                    native_io=bool(selected.get("native_io", False)),
                    level=str(selected.get("level", "detailed")),
                    option=str(selected.get("option", "optrace")),
                    **self._execution_options(selected),
                )
                device_identifier = device_stage.identifier
                device_soc = device_stage.soc_verification
                remote_attempt_dir = device_stage.adb.attempt_dir
            report_payload = {
                "graph_name": result.graph_name,
                "level": result.level,
                "option": result.option,
                "reports": [_jsonable(report) for report in result.reports],
            }
            report_ref = atomic_publish_json(
                output_dir / "profile_report.json",
                report_payload,
                kind=ArtifactKind.REPORT,
                logical_name="profile_report",
            )
            return report_payload, (report_ref,), {
                "report_count": len(result.reports),
                "device_identifier": device_identifier,
                "device_soc": device_soc,
                "remote_attempt_dir": remote_attempt_dir,
                "remote_cleanup": "confirmed",
            }

        return self._continuation_operation(
            "profile",
            manifest_uri,
            manifest_sha256,
            operation,
            stage_config=selected,
        )

