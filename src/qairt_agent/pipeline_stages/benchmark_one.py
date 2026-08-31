"""Measuring one runtime binding.

One graph invocation scope, one slice chain, or one GenAI generation call. The
multi-AR benchmark drives this once per AR with that AR's config and its own
vector manifest, which is why it takes everything as an argument rather than
reading an enclosing scope: it used to be an 870-line closure inside
``benchmark``, unreadable and undrivable on its own.
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




class BenchmarkOneStage:
    """BenchmarkOneStage — see the module docstring."""

    def _benchmark_one(
        self,
        manifest: RunManifest,
        adapter: Any,
        output_dir: Path,
        selected_config: Mapping[str, Any],
        *,
        manifest_sha256: str,
        report_suffix: str = "",
    ) -> tuple[dict[str, Any], tuple[ArtifactRef, ...], dict[str, Any]]:
        """Measure one runtime binding: one graph, one chain, or one GenAI scope.

        Lifted out of ``benchmark``, where it was an 870-line closure. Everything it
        needs arrives as an argument -- the multi-AR path calls it once per AR with
        that AR's config -- so it can be read, and driven, on its own.
        """

        effective = dict(selected_config)
        explicit_chain = effective.get("routes") is not None
        explicit_graph = all(
            effective.get(key) is not None
            for key in ("context_path", "graph_name", "vector_manifest")
        )
        explicit_genai = effective.get("container_path") is not None
        if not explicit_chain and not explicit_graph and not explicit_genai:
            effective = self._automatic_runtime_binding(
                manifest,
                effective,
            )
        effective = self._enforce_qwen3_vl_runtime_scope(
            manifest,
            effective,
            stage="benchmark",
        )
        self._reject_genai_chain_keys(effective, stage="benchmark")

        spec = manifest.build_spec
        optrace_enabled = bool(
            effective.get("optrace", spec.benchmark.optrace)
        )
        profile_level = str(
            effective.get("profile_level", "detailed")
        )
        profile_option = str(
            effective.get("profile_option", "optrace")
        )
        if optrace_enabled and profile_option.lower() != "optrace":
            raise InvalidSpecError(
                "benchmark.optrace requires profile_option='optrace'",
                stage="benchmark",
                details={"profile_option": profile_option},
            )
        profile_entries: list[dict[str, Any]] = []
        normalized_profile_ops: list[dict[str, Any]] = []
        profile_source_refs: list[ArtifactRef] = []
        self._preflight(adapter, spec)
        execution_options = self._execution_options(effective)
        native_io = bool(effective.get("native_io", False))
        scope = "graph"
        generation_metrics: dict[str, Any] | None = None
        generated_text_sha256: str | None = None
        generated_text_length: int | None = None
        profile_routes: Sequence[Mapping[str, Any]] | None = None
        profile_contexts: Mapping[str, Any] | None = None
        profile_inputs: dict[str, np.ndarray] | None = None
        profile_ar: int | None = None
        profile_initial_native_state: dict[str, np.ndarray] = {}
        device_execution: dict[str, Any] | None = None
        execution_owner: Any = None
        chain_slice_inputs: dict[str, list[dict[str, Any]]] = {}
        loaded_contexts: dict[str, Any] = {}
        profile_claim_scope = (
            "production_runtime_optrace"
        )
        if effective.get("lane") == "genai_builder" or explicit_genai:
            if not effective.get("runtime_supported", True):
                raise InvalidSpecError(
                    "the saved GenAI container is marked runtime unsupported "
                    "by the selected QAIRT workflow",
                    stage="benchmark",
                    details={
                        "family": effective.get("family"),
                        "container_path": effective.get("container_path"),
                    },
                )
            if effective.get("container_path") is None:
                raise ValueError(
                    "GenAI benchmark requires container_path"
                )
            if effective.get("prompt") is not None:
                prompt: Any = effective["prompt"]
            elif effective.get("prompt_path") is not None:
                prompt = Path(effective["prompt_path"]).expanduser().resolve()
                if not prompt.is_file():
                    raise FileNotFoundError(
                        f"GenAI prompt_path does not exist: {prompt}"
                    )
            else:
                raise InvalidSpecError(
                    "GenAI benchmark requires stage_configs.benchmark.prompt "
                    "or prompt_path so the measured workload is explicit",
                    stage="benchmark",
                )
            tensor_runtime = effective.get("tensor_runtime")
            if optrace_enabled:
                if not isinstance(tensor_runtime, Mapping):
                    raise InvalidSpecError(
                        "GenAI benchmark.optrace requires an auditable "
                        "public raw CompiledModel tensor runtime",
                        stage="benchmark",
                        details={
                            "family": effective.get("family"),
                            "container_path": effective.get(
                                "container_path"
                            ),
                        },
                    )
                raw_routes = tensor_runtime.get("routes")
                raw_contexts = tensor_runtime.get("contexts")
                if (
                    not isinstance(raw_routes, Sequence)
                    or isinstance(
                        raw_routes,
                        (str, bytes, bytearray),
                    )
                    or not raw_routes
                    or not isinstance(raw_contexts, Mapping)
                    or not raw_contexts
                ):
                    raise InvalidSpecError(
                        "GenAI tensor runtime is missing raw routes or "
                        "compiled contexts required for optrace",
                        stage="benchmark",
                    )
                if effective.get("vector_manifest") is None:
                    raise InvalidSpecError(
                        "GenAI benchmark.optrace requires the exact per-AR "
                        "vector manifest selected by runtime_index",
                        stage="benchmark",
                    )
                profile_routes = list(raw_routes)
                profile_contexts = dict(raw_contexts)
                profile_inputs = self._manifest_inputs(
                    effective["vector_manifest"],
                    sha256=effective.get("vector_manifest_sha256"),
                )
                profile_initial_native_state = (
                    self._initial_native_state_from_routes(
                        profile_routes,
                        profile_inputs,
                    )
                )
                profile_ar = int(
                    effective.get("ar", spec.sequence.ars[0])
                )
                profile_claim_scope = (
                    "raw_compiled_slices_not_generation_wall_latency"
                )
            context_paths = (
                tuple(profile_contexts.values())
                if profile_contexts is not None
                else ()
            )
            vector_manifests = (
                (effective["vector_manifest"],)
                if profile_inputs is not None
                else ()
            )
            inline_cases: list[Mapping[str, Any]] = []
            if profile_initial_native_state:
                inline_cases.append(profile_initial_native_state)
            scope = "genai_generation"
        elif effective.get("routes") is not None:
            contexts = effective.get("contexts")
            if not isinstance(contexts, Mapping) or not contexts:
                raise ValueError(
                    "chain benchmark requires config.contexts mapped by slice"
                )
            context_paths = tuple(contexts.values())
            vector_manifests = (
                (effective["vector_manifest"],)
                if effective.get("vector_manifest") is not None
                else ()
            )
            inline_cases = [
                dict(step.get("inputs", {}))
                for step in effective.get("steps", ())
                if isinstance(step, Mapping)
            ]
            initial_native_state = self._tensor_mapping(
                effective.get("initial_native_state", {})
            )
            if (
                not initial_native_state
                and effective.get("vector_manifest") is not None
            ):
                manifest_inputs = self._manifest_inputs(
                    effective["vector_manifest"],
                    sha256=effective.get("vector_manifest_sha256"),
                )
                initial_native_state = self._initial_native_state_from_routes(
                    effective["routes"],
                    manifest_inputs,
                )
            if initial_native_state:
                inline_cases.append(initial_native_state)
            profile_routes = list(effective["routes"])
            profile_contexts = dict(contexts)
            profile_initial_native_state = dict(initial_native_state)

        else:
            required = ("context_path", "graph_name", "vector_manifest")
            missing = [key for key in required if effective.get(key) is None]
            if missing:
                raise ValueError(f"benchmark missing config fields: {missing}")
            context_paths = (effective["context_path"],)
            vector_manifests = (effective["vector_manifest"],)
            inline_cases = []

        push_files = self._device_stage_files(
            output_dir,
            contexts=context_paths,
            vector_manifests=vector_manifests,
            inline_cases=tuple(inline_cases),
        )

        warmup = int(effective.get("warmup_runs", spec.benchmark.warmup_runs))
        repeats = int(
            effective.get("measured_runs", spec.benchmark.measured_runs)
        )
        diagnoser = LatencyDiagnoser()
        with self._device_stage(
            manifest,
            adapter,
            stage_name="benchmark",
            input_manifest_sha256=manifest_sha256,
            stage_config=effective,
            push_files=push_files,
        ) as device_stage:
            if effective.get("lane") == "genai_builder" or explicit_genai:
                executor = adapter.create_genai_executor(
                    effective["container_path"],
                    device=device_stage.device,
                )
                last_generation: dict[str, Any] = {}

                def invoke() -> Any:
                    result = executor.generate(prompt)
                    last_generation["result"] = result
                    return result

                try:
                    measurement = diagnoser.measure(
                        invoke,
                        warmup=warmup,
                        repeats=repeats,
                    )
                    aa_measurement = (
                        diagnoser.calibrate_aa(
                            invoke,
                            warmup=warmup,
                            repeats=repeats,
                        )
                        if bool(effective.get("aa_calibration", True))
                        else None
                    )
                    result = last_generation.get("result")
                    metrics_value = getattr(result, "metrics", None)
                    if metrics_value is not None:
                        generation_metrics = _jsonable(metrics_value)
                    generated_text = getattr(result, "generated_text", None)
                    if isinstance(generated_text, str):
                        generated_text_length = len(generated_text)
                        generated_text_sha256 = hashlib.sha256(
                            generated_text.encode("utf-8")
                        ).hexdigest()
                finally:
                    adapter.clean_genai_executor(executor)
            elif effective.get("routes") is not None:
                for slice_name, context_path in contexts.items():
                    if hasattr(adapter, "load_compiled"):
                        loaded_contexts[str(slice_name)] = (
                            adapter.load_compiled(context_path)
                        )
                    elif hasattr(adapter, "_compiled_model"):
                        loaded_contexts[str(slice_name)] = (
                            adapter._compiled_model(context_path)
                        )
                    else:
                        loaded_contexts[str(slice_name)] = context_path
                runner = SliceChainRunner(
                    effective["routes"],
                    self._recording_chain_executors(
                        self._chain_executors(
                            adapter,
                            loaded_contexts,
                            device=device_stage.device,
                            native_io=native_io,
                            execution_options=execution_options,
                        ),
                        chain_slice_inputs,
                    ),
                )
                scope = (
                    "chain_sequence" if "steps" in effective else "chain"
                )
                if "steps" in effective:
                    steps = effective["steps"]

                    def invoke() -> Any:
                        return runner.run_sequence(
                            steps,
                            mode="device_chain",
                            initial_native_state=initial_native_state,
                        )

                else:
                    if effective.get("vector_manifest") is None:
                        raise ValueError(
                            "chain benchmark requires vector_manifest "
                            "when steps are absent"
                        )
                    inputs = self._manifest_inputs(
                        effective["vector_manifest"],
                        sha256=effective.get("vector_manifest_sha256"),
                    )
                    ar = int(
                        effective.get("ar", spec.sequence.ars[0])
                    )
                    profile_inputs = dict(inputs)
                    profile_ar = ar

                    def invoke() -> Any:
                        return runner.run_device_chain(
                            inputs,
                            ar=ar,
                            initial_native_state=initial_native_state,
                        )

            else:
                inputs = self._manifest_inputs(
                    effective["vector_manifest"],
                    sha256=effective.get("vector_manifest_sha256"),
                )
                if hasattr(adapter, "load_compiled"):
                    compiled = adapter.load_compiled(
                        effective["context_path"]
                    )
                elif hasattr(adapter, "_compiled_model"):
                    compiled = adapter._compiled_model(
                        effective["context_path"]
                    )
                else:
                    compiled = effective["context_path"]

                def invoke() -> Any:
                    return adapter.run_graph(
                        compiled,
                        inputs,
                        graph_name=str(effective["graph_name"]),
                        device=device_stage.device,
                        native_io=native_io,
                        **execution_options,
                    )
                profile_inputs = dict(inputs)
                profile_ar = int(
                    effective.get("ar", spec.sequence.ars[0])
                )
                # Ask the device what it actually spent, before the model
                # is initialized: an initialized model carries an execution
                # context created with profiling disabled, and QAIRT would
                # then report no profiling data at all.
                device_execution = self._device_execution_block(
                    adapter,
                    compiled,
                    inputs,
                    graph_name=str(effective["graph_name"]),
                    device=device_stage.device,
                    native_io=native_io,
                    execution_options=execution_options,
                    working_dir=output_dir / "device_profiling",
                )
                # Without initialize(), QAIRT rebuilds the backend and the
                # inferencer on every call.  This is its documented way to
                # execute repeatedly against one model/backend/device.
                execution_owner = self._initialize_execution(
                    adapter,
                    compiled,
                    device=device_stage.device,
                )

            if scope != "genai_generation":
                # Our own setup -- context loading, Device construction, ADB
                # staging, graph-runner setup -- is complete before the
                # timer.  What QAIRT does inside one call is not: on the
                # low-level lane it relaunches qnn-net-run per call.
                try:
                    measurement = diagnoser.measure(
                        invoke,
                        warmup=warmup,
                        repeats=repeats,
                    )
                    aa_measurement = (
                        diagnoser.calibrate_aa(
                            invoke,
                            warmup=warmup,
                            repeats=repeats,
                        )
                        if bool(effective.get("aa_calibration", True))
                        else None
                    )
                finally:
                    if execution_owner is not None:
                        adapter.release_execution(execution_owner)
                        execution_owner = None

            if device_execution is None and chain_slice_inputs:
                # The measured chain pass recorded what each slice was
                # actually fed, so each one can now be profiled with its
                # real inputs rather than a guess.
                device_execution = self._chain_device_execution(
                    adapter,
                    chain_slice_inputs,
                    loaded_contexts,
                    device=device_stage.device,
                    native_io=native_io,
                    execution_options=execution_options,
                    working_dir=output_dir / "device_profiling",
                )
            if device_execution is None:
                # Latency is device time, so a scope with no device meter
                # says so with a cause rather than quietly omitting the
                # block and leaving the wall number to look like latency.
                device_execution = {
                    "schema": DEVICE_EXECUTION_SCHEMA,
                    "policy": "report_only",
                    "available": False,
                    "reason": _DEVICE_EXECUTION_UNAVAILABLE.get(
                        scope, "no device meter is wired for this scope"
                    ),
                }
            if optrace_enabled:
                if not hasattr(adapter, "profile"):
                    raise InvalidSpecError(
                        "the selected QAIRT adapter does not expose the "
                        "public profile API required by benchmark.optrace",
                        stage="benchmark",
                    )

                def capture_profile(
                    compiled_context: Any,
                    profile_inputs: Mapping[str, np.ndarray],
                    *,
                    slice_id: str,
                    graph_name: str,
                    step_index: int,
                ) -> dict[str, np.ndarray]:
                    profiled = adapter.profile(
                        compiled_context,
                        profile_inputs,
                        graph_name=graph_name,
                        device=device_stage.device,
                        native_io=native_io,
                        level=profile_level,
                        option=profile_option,
                        **execution_options,
                    )
                    raw_reports: list[Any] = []
                    source_refs: list[ArtifactRef] = []
                    profile_call_index = len(profile_entries)
                    for report_index, raw_report in enumerate(
                        tuple(getattr(profiled, "reports", ()) or ())
                    ):
                        report_payload, source_ref = (
                            self._profile_report_payload(raw_report)
                        )
                        raw_reports.append(report_payload)
                        captured_ref = atomic_publish_json(
                            output_dir
                            / "optrace"
                            / (
                                f"profile-{profile_call_index:04d}-"
                                f"report-{report_index:04d}.json"
                            ),
                            {
                                "schema": (
                                    "qairt-agent.captured-profile-report.v1"
                                ),
                                "source_artifact": (
                                    _jsonable(source_ref)
                                    if source_ref is not None
                                    else None
                                ),
                                "report": report_payload,
                            },
                            kind=ArtifactKind.REPORT,
                        )
                        source_refs.append(captured_ref)
                    records = self._normalize_optrace_records(
                        raw_reports,
                        slice_id=slice_id,
                        graph_name=graph_name,
                        step_index=step_index,
                    )
                    profile_entries.append(
                        {
                            "slice_id": slice_id,
                            "graph_name": graph_name,
                            "step_index": step_index,
                            "level": str(
                                getattr(
                                    profiled,
                                    "level",
                                    profile_level,
                                )
                            ),
                            "option": str(
                                getattr(
                                    profiled,
                                    "option",
                                    profile_option,
                                )
                            ),
                            "reports": raw_reports,
                            "report_artifacts": [
                                _jsonable(ref) for ref in source_refs
                            ],
                            "normalized_op_count": len(records),
                        }
                    )
                    normalized_profile_ops.extend(records)
                    profile_source_refs.extend(source_refs)
                    return _output_mapping(
                        getattr(profiled, "execution_result", None),
                        graph_name=graph_name,
                    )

                if profile_routes is not None:
                    assert profile_contexts is not None
                    loaded_profile_contexts: dict[str, Any] = {}
                    for (
                        slice_name,
                        context_path,
                    ) in profile_contexts.items():
                        if hasattr(adapter, "load_compiled"):
                            loaded_profile_contexts[str(slice_name)] = (
                                adapter.load_compiled(context_path)
                            )
                        elif hasattr(adapter, "_compiled_model"):
                            loaded_profile_contexts[str(slice_name)] = (
                                adapter._compiled_model(context_path)
                            )
                        else:
                            loaded_profile_contexts[str(slice_name)] = (
                                context_path
                            )
                    profile_executors: dict[
                        str,
                        Callable[
                            [Mapping[str, np.ndarray], Any],
                            Mapping[str, np.ndarray],
                        ],
                    ] = {}
                    for (
                        slice_name,
                        compiled_context,
                    ) in loaded_profile_contexts.items():
                        def execute_profile(
                            slice_inputs: Mapping[str, np.ndarray],
                            invocation: Any,
                            *,
                            context: Any = compiled_context,
                            bound_slice: str = str(slice_name),
                        ) -> Mapping[str, np.ndarray]:
                            return capture_profile(
                                context,
                                slice_inputs,
                                slice_id=bound_slice,
                                graph_name=invocation.graph_name,
                                step_index=int(invocation.step_index),
                            )

                        profile_executors[str(slice_name)] = (
                            execute_profile
                        )
                    profile_runner = SliceChainRunner(
                        profile_routes,
                        profile_executors,
                    )
                    if "steps" in effective:
                        profile_runner.run_sequence(
                            steps,
                            mode="device_chain",
                            initial_native_state=initial_native_state,
                        )
                    else:
                        if profile_inputs is None or profile_ar is None:
                            raise InvalidSpecError(
                                "optrace chain execution is missing exact "
                                "inputs or AR selection",
                                stage="benchmark",
                            )
                        profile_runner.run_device_chain(
                            profile_inputs,
                            ar=profile_ar,
                            initial_native_state=(
                                profile_initial_native_state
                            ),
                        )
                else:
                    capture_profile(
                        compiled,
                        inputs,
                        slice_id=str(
                            effective.get("slice_id") or "model"
                        ),
                        graph_name=str(effective["graph_name"]),
                        step_index=0,
                    )
                if not normalized_profile_ops:
                    raise InvalidSpecError(
                        "benchmark.optrace produced no structured per-op "
                        "cycle records; raw profiler reports cannot support "
                        "automatic latency attribution",
                        stage="benchmark",
                        details={
                            "profile_count": len(profile_entries),
                            "required_fields": [
                                "op_id/name",
                                "cycles or thread_cycles",
                            ],
                        },
                    )
            device_identifier = device_stage.identifier
            device_soc = device_stage.soc_verification
            remote_attempt_dir = device_stage.adb.attempt_dir
        optrace_ref: ArtifactRef | None = None
        if optrace_enabled:
            runtime_indexes = [
                artifact
                for artifact in manifest.artifacts
                if artifact.logical_name == "runtime_index"
            ]
            if len(runtime_indexes) != 1:
                raise InvalidSpecError(
                    "benchmark.optrace requires exactly one build "
                    "runtime_index artifact",
                    stage="benchmark",
                    details={
                        "runtime_index_count": len(runtime_indexes)
                    },
                )
            verify_artifact(runtime_indexes[0])
            optrace_ref = atomic_publish_json(
                output_dir / f"optrace_evidence{report_suffix}.json",
                {
                    "schema": "qairt-agent.optrace-evidence.v1",
                    "source_manifest_sha256": manifest_sha256,
                    "runtime_index": _jsonable(runtime_indexes[0]),
                    "runtime_binding": {
                        key: _jsonable(effective.get(key))
                        for key in (
                            "lane",
                            "family",
                            "ar",
                            "context_length",
                            "scope",
                            "route_manifest",
                            "context_path",
                            "graph_name",
                            "component",
                            "coverage",
                        )
                        if effective.get(key) is not None
                    },
                    "device_identifier": device_identifier,
                    "device_soc": device_soc,
                    "profile_level": profile_level,
                    "profile_option": profile_option,
                    "profiles": profile_entries,
                    "ops": normalized_profile_ops,
                    "profile_scope": profile_claim_scope,
                    "claim_scope": (
                        "reported_op_work_not_additive_wall_latency"
                    ),
                },
                kind=ArtifactKind.REPORT,
                logical_name=f"optrace_evidence{report_suffix}",
            )
        payload: dict[str, Any] = {
            "policy": "report_only",
            "scope": scope,
            # Latency means device time. The host wall number is kept for
            # diagnosing the harness -- it still detects ADB, container and
            # transport degradation -- but it is not this report's latency
            # and never grounds a regression verdict.
            "latency_metric": (
                "device_execution"
                if device_execution is not None
                and device_execution.get("available") is not False
                else "unavailable"
            ),
            "harness_diagnostics": {
                "metric_name": "host_orchestrated_call_latency",
                "not_latency": True,
                "note": (
                    "host wall time around one SDK call, kept to detect "
                    "harness and transport degradation; the reported "
                    "latency is the block named by 'latency_metric'"
                ),
                "measurement": measurement.to_dict(),
                # Our own setup is outside the timer; what the SDK does
                # inside one call is not, and on the low-level lane that
                # includes a fresh qnn-net-run process per call.  Reporting
                # a single "setup_excluded" flag conflated the two.
                "harness_setup_excluded": True,
                "sdk_per_call_setup_included": (
                    "unverified"
                    if scope == "genai_generation"
                    else True
                ),
                "measurement_scope": {
                    "clock": "host_perf_counter_ns",
                    "includes": "host_to_sdk_to_device_round_trip",
                    "device_side_sync_barrier": False,
                    "note": (
                        "the QAIRT Python API exposes no device-side "
                        "synchronization barrier, so each sample is the warmed "
                        "host wall time around one call, including whatever "
                        "per-call setup the SDK performs inside it; this is not "
                        "device execution time -- see 'device_execution'"
                    ),
                    "excluded_from_timer": [
                        "context_loading",
                        "device_construction",
                        "adb_staging",
                        "graph_runner_setup",
                    ],
                    "included_in_sample": (
                        []
                        if scope == "genai_generation"
                        else [
                            "qnn_net_run_process_launch",
                            "per_call_context_load",
                            "hvx_hmx_power_on_and_acquire",
                            "per_call_deinit",
                            "adb_input_push_and_output_pull",
                        ]
                    ),
                    "sample_unit": (
                        "generate_call"
                        if scope == "genai_generation"
                        else "graph_invocation"
                    ),
                },
            },
            "runtime_binding": {
                key: _jsonable(effective.get(key))
                for key in (
                    "lane",
                    "family",
                    "ar",
                    "context_length",
                    "scope",
                    "route_manifest",
                    "context_path",
                    "graph_name",
                    "container_path",
                    "component",
                    "coverage",
                    "excluded_components",
                    "graph_ar",
                )
                if effective.get(key) is not None
            },
        }
        requested_ar_values = [
            int(value) for value in spec.sequence.ars
        ]
        if scope == "genai_generation":
            payload["coverage"] = {
                "mode": "executor_managed_generation",
                "requested_ars": requested_ar_values,
                "executed_ars": "not_observable_via_public_generation_api",
                "complete": None,
                "prefill_decode_scope": True,
                "graph_ar_coverage_proven": False,
                "limitation": (
                    "the public GenAI generation executor reports end-to-end "
                    "generation latency but does not expose which AR graph "
                    "served each prefill/decode step"
                ),
            }
        elif effective.get("ar") is not None:
            executed_ar = int(effective["ar"])
            payload["coverage"] = {
                "mode": (
                    "single_ar_override"
                    if selected_config.get("ar") is not None
                    and not report_suffix
                    else "single_ar"
                ),
                "requested_ars": requested_ar_values,
                "executed_ars": [executed_ar],
                "missing_ars": [
                    ar
                    for ar in requested_ar_values
                    if ar != executed_ar
                ],
                "complete": requested_ar_values == [executed_ar],
            }
        else:
            payload["coverage"] = {
                "mode": "explicit_custom_runtime",
                "requested_ars": requested_ar_values,
                "executed_ars": "caller_defined",
                "complete": None,
            }
        if device_execution is not None:
            payload["device_execution"] = device_execution
        if generation_metrics is not None:
            payload["generation_metrics"] = generation_metrics
        if generated_text_sha256 is not None:
            payload["generated_text"] = {
                "sha256": generated_text_sha256,
                "character_count": generated_text_length,
            }
        token_count = effective.get("token_count")
        token_source: str | None = None
        if token_count is not None:
            normalized_tokens = int(token_count)
            if normalized_tokens <= 0:
                raise ValueError("benchmark token_count must be positive")
            token_source = "caller"
        else:
            reported = _sdk_generated_token_count(generation_metrics)
            if reported is not None:
                normalized_tokens = reported
                token_source = "sdk_metrics"
        if token_source is not None:
            payload["token_count"] = normalized_tokens
            payload["ms_per_token_source"] = token_source
            # Derived from the host wall samples, so it lives with them:
            # it inherits everything that makes them not device latency.
            payload["harness_diagnostics"]["p50_ms_per_token"] = (
                measurement.summary.p50_ms / normalized_tokens
            )
        if aa_measurement is not None:
            # A/A calibrated host noise, which is no longer the metric.
            payload["harness_diagnostics"]["aa_calibration"] = (
                aa_measurement.to_dict()
            )
        if optrace_ref is not None:
            payload["optrace_evidence"] = _jsonable(optrace_ref)
        footprint = self._build_static_footprint(manifest)
        if footprint is not None:
            payload["static_footprint"] = footprint
        report_ref = atomic_publish_json(
            output_dir / f"latency_report{report_suffix}.json",
            payload,
            kind=ArtifactKind.REPORT,
            logical_name=f"latency_report{report_suffix}",
        )
        output_refs = (
            tuple(profile_source_refs)
            + ((optrace_ref,) if optrace_ref is not None else ())
            + (report_ref,)
        )
        return payload, output_refs, {
            "warmup_runs": warmup,
            "measured_runs": repeats,
            "optrace": optrace_enabled,
            "optrace_profile_count": len(profile_entries),
            "optrace_op_count": len(normalized_profile_ops),
            "policy": "report_only",
            "device_identifier": device_identifier,
            "device_soc": device_soc,
            "remote_attempt_dir": remote_attempt_dir,
            "remote_cleanup": "confirmed",
        }

