"""Default stage runner: adapts workflow stages to the synchronous engine.

This is the production runner used when the QAIRT SDK is available.  It wraps
:class:`qairt_agent.pipeline.QairtAgent` and maps each workflow stage to the
corresponding engine method, threading the immutable manifest from one stage to
the next.  Stage failures surface as structured tool errors so the worker can
record a failed receipt.
"""

from __future__ import annotations

import inspect
from typing import Any

from qairt_agent.artifacts import verify_artifact, verify_manifest_graph
from qairt_agent.contracts import ArtifactRef, StageExecutionContext, ToolResult
from qairt_agent.errors import InvalidSpecError, ManifestInvalidError, ToolError
from qairt_agent.jobs.worker import StageContext, StageOutcome


def _outcome_from_result(result: ToolResult[Any], stage_name: str) -> StageOutcome:
    if not result.ok:
        if result.error is not None:
            raise ToolError(result.error)
        raise InvalidSpecError(f"stage '{stage_name}' failed without a structured error", stage=stage_name)
    data = result.data
    metrics = dict(data) if isinstance(data, dict) else {"data": data}
    metrics["stage"] = stage_name
    artifacts: tuple[ArtifactRef, ...] = ()
    if result.manifest is not None:
        try:
            artifacts = verify_manifest_graph(result.manifest)
        except ManifestInvalidError:
            # Legacy/custom test engines historically returned opaque JSON
            # tagged as MANIFEST. Production ManifestStore references carry the
            # run-manifest logical name and must always pass graph validation.
            if (result.manifest.logical_name or "").startswith("run-manifest-r"):
                raise
            verify_artifact(result.manifest)
    return StageOutcome(
        artifacts=artifacts,
        metrics=metrics,
        manifest=result.manifest,
        verified=True,
    )


def _require_manifest(ctx: StageContext) -> ArtifactRef:
    if ctx.current_manifest is None:
        raise InvalidSpecError(
            f"stage '{ctx.stage_name}' requires a prior build manifest",
            stage=ctx.stage_name,
        )
    return ctx.current_manifest


class EngineStageRunner:
    """Run workflow stages against a synchronous :class:`QairtAgent` engine."""

    def __init__(
        self,
        *,
        engine: Any,
        build_spec: dict[str, Any],
        pipeline: str,
        vector_manifest: str | None = None,
        config: dict[str, Any] | None = None,
        build_config: dict[str, Any] | None = None,
        validate_config: dict[str, Any] | None = None,
        benchmark_config: dict[str, Any] | None = None,
        diagnose_config: dict[str, Any] | None = None,
        diagnose_kind: str = "quality",
    ) -> None:
        self.engine = engine
        self.build_spec = build_spec
        self.pipeline = pipeline
        self.vector_manifest = vector_manifest
        # ``config`` is retained as a compatibility fallback for direct users
        # of EngineStageRunner.  Native workflows pass explicit per-stage
        # mappings so unrelated continuation inputs cannot leak across stages.
        legacy_config = dict(config or {})
        self.build_config = dict(
            legacy_config if build_config is None else build_config
        )
        self.validate_config = dict(
            legacy_config if validate_config is None else validate_config
        )
        self.benchmark_config = dict(
            legacy_config if benchmark_config is None else benchmark_config
        )
        self.diagnose_config = dict(
            legacy_config if diagnose_config is None else diagnose_config
        )
        normalized_diagnose_kind = getattr(diagnose_kind, "value", diagnose_kind)
        if normalized_diagnose_kind not in {"quality", "latency"}:
            raise InvalidSpecError(
                "diagnose_kind must be 'quality' or 'latency'",
                stage="diagnose",
                details={"diagnose_kind": str(normalized_diagnose_kind)},
            )
        self.diagnose_kind = str(normalized_diagnose_kind)

    @staticmethod
    def _invoke(
        method: Any,
        *args: Any,
        execution_context: StageExecutionContext,
        **kwargs: Any,
    ) -> Any:
        """Pass attempt-local context when the engine supports the new API."""

        try:
            parameters = inspect.signature(method).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        supports_context = any(
            parameter.name == "execution_context"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if supports_context:
            kwargs["execution_context"] = execution_context
        return method(*args, **kwargs)

    def __call__(self, ctx: StageContext) -> StageOutcome:
        stage = ctx.stage_name
        execution_context = StageExecutionContext(
            output_dir=ctx.output_dir,
            attempt=ctx.attempt,
        )
        if stage == "build":
            return self._build(execution_context)
        if stage == "validate":
            manifest = _require_manifest(ctx)
            result = self._invoke(
                self.engine.validate,
                str(manifest.path),
                manifest.sha256,
                vector_manifest=self.vector_manifest,
                config=self.validate_config,
                execution_context=execution_context,
            )
            return _outcome_from_result(result, stage)
        if stage == "benchmark":
            manifest = _require_manifest(ctx)
            result = self._invoke(
                self.engine.benchmark,
                str(manifest.path),
                manifest.sha256,
                config=self.benchmark_config,
                execution_context=execution_context,
            )
            return _outcome_from_result(result, stage)
        if stage == "diagnose":
            manifest = _require_manifest(ctx)
            diagnose = (
                self.engine.diagnose_latency
                if self.diagnose_kind == "latency"
                else self.engine.diagnose_quality
            )
            result = self._invoke(
                diagnose,
                str(manifest.path),
                manifest.sha256,
                config=self.diagnose_config,
                execution_context=execution_context,
            )
            return _outcome_from_result(result, stage)
        raise InvalidSpecError(f"unknown workflow stage '{stage}'", stage=stage)

    def _build(self, execution_context: StageExecutionContext) -> StageOutcome:
        if self.pipeline == "genai_builder":
            config = dict(self.build_config)
            attached = self.build_spec.get("metadata", {}).get("attached_models_by_ar")
            if attached is not None:
                config.setdefault("attached_models_by_ar", attached)
            result = self._invoke(
                self.engine.build_genai_container,
                self.build_spec,
                config=config,
                execution_context=execution_context,
            )
        else:
            result = self._invoke(
                self.engine.build,
                self.build_spec,
                execution_context=execution_context,
            )
        return _outcome_from_result(result, "build")


__all__ = ["EngineStageRunner"]
