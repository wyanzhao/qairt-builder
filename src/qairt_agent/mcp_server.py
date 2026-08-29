"""MCP facade for the QAIRT agent.

By default the server exposes only four short, asynchronous tools backed by the
file job journal:

    submit_job(spec, stages?, from_job?)  -> {job_id, state, status_path}
    get_job(job_id, after_seq?)           -> {status, events}
    cancel_job(job_id)                    -> {ok, job_id}
    resume_job(job_id)                    -> {job_id, state, status_path}

The agent no longer orchestrates a dozen fine-grained tools.  The original
~18 synchronous tools remain available behind ``legacy=True`` (or
``QAIRT_AGENT_MCP_LEGACY=1``) and are marked deprecated.
"""

import os
from collections.abc import Callable
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    FastMCP = None  # type: ignore[assignment,misc]
    _MCP_IMPORT_ERROR: Exception | None = exc
else:
    _MCP_IMPORT_ERROR = None

from qairt_agent.agent import QairtAgentClient
from qairt_agent.pipeline import QairtAgent


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _error_payload(exc: Exception, stage: str | None = None) -> dict[str, Any]:
    error_data = getattr(exc, "as_dict", None)
    if callable(error_data):
        return error_data()
    return {
        "code": type(exc).__name__.upper(),
        "stage": stage,
        "message": str(exc),
        "retryable": False,
    }


def _safe(call: Callable[[], Any]) -> dict[str, Any]:
    try:
        result = call()
    except Exception as exc:  # the MCP boundary must always return structured JSON
        return {"ok": False, "error": _error_payload(exc)}
    return result if isinstance(result, dict) else {"ok": True, "data": result}


def _invoke(method: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Legacy synchronous dispatch: a fresh orchestrator per call."""

    try:
        agent = QairtAgent()
        result = getattr(agent, method)(*args, **kwargs)
        data = _jsonable(result)
        return data if isinstance(data, dict) else {"ok": True, "data": data}
    except Exception as exc:  # FastMCP should always return a structured result
        return {"ok": False, "error": _error_payload(exc, stage=method)}


# --------------------------------------------------------------------------- #
# default asynchronous tools
# --------------------------------------------------------------------------- #


def _register_async_tools(server: Any, *, jobs_root: str | None, client: QairtAgentClient | None) -> None:
    if client is None:
        client = QairtAgentClient(jobs_root=jobs_root or ".qairt-agent/jobs", background=True)

    @server.tool(
        description=(
            "Submit an asynchronous workflow job. Returns job_id/state/status_path "
            "immediately; the job keeps running after this call returns."
        )
    )
    def submit_job(
        spec: dict[str, Any],
        stages: list[str] | None = None,
        from_job: str | None = None,
    ) -> dict[str, Any]:
        return _safe(
            lambda: client.submit(
                spec,
                stages=tuple(stages) if stages else ("build",),
                from_job=from_job,
            ).submission()
        )

    @server.tool(
        description=(
            "Get a job's current status plus any events after a sequence number; "
            "resume a watch by passing the last seq you saw as after_seq."
        )
    )
    def get_job(job_id: str, after_seq: int = 0) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            handle = client.job(job_id)
            return {
                "status": handle.status().model_dump(mode="json"),
                "events": handle.events(after_seq),
            }

        return _safe(call)

    @server.tool(description="Request cancellation of a running job.")
    def cancel_job(job_id: str) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            client.job(job_id).cancel()
            return {"ok": True, "job_id": job_id}

        return _safe(call)

    @server.tool(description="Resume an interrupted (orphaned) job from its last verified receipt.")
    def resume_job(job_id: str) -> dict[str, Any]:
        return _safe(lambda: client.resume(job_id).submission())


# --------------------------------------------------------------------------- #
# deprecated legacy tools
# --------------------------------------------------------------------------- #


def _register_legacy_tools(server: Any) -> None:
    @server.tool(description="DEPRECATED: use submit_job. Validate a BuildSpec and publish a build plan.")
    def qairt_plan(spec: dict[str, Any], offline: bool = False) -> dict[str, Any]:
        return _invoke("plan", spec, offline=offline)

    @server.tool(description="DEPRECATED: generate a family-specific effective QAIRT configuration.")
    def qairt_generate_config(spec: dict[str, Any]) -> dict[str, Any]:
        return _invoke("generate_config", spec)

    @server.tool(description="DEPRECATED: use submit_job. Run the low-level build pipeline.")
    def qairt_build(spec: dict[str, Any]) -> dict[str, Any]:
        return _invoke("build", spec)

    @server.tool(description="DEPRECATED: use submit_job. Build and save a GenAI container.")
    def qairt_build_genai_container(
        spec: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return _invoke("build_genai_container", spec, config=config or {})

    @server.tool(description="DEPRECATED: run full/teacher-forced/device-chain quality validation.")
    def qairt_validate(
        manifest_uri: str,
        manifest_sha256: str,
        vector_manifest: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _invoke(
            "validate", manifest_uri, manifest_sha256, vector_manifest=vector_manifest, config=config or {}
        )

    @server.tool(description="DEPRECATED: measure warmed production graph/chain/token latency.")
    def qairt_benchmark(
        manifest_uri: str,
        manifest_sha256: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _invoke("benchmark", manifest_uri, manifest_sha256, config=config or {})

    @server.tool(description="DEPRECATED: localize a numerical divergence to slice/layer/op.")
    def qairt_diagnose_quality(
        manifest_uri: str,
        manifest_sha256: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _invoke("diagnose_quality", manifest_uri, manifest_sha256, config=config or {})

    @server.tool(description="DEPRECATED: localize a latency issue via production timing/optrace.")
    def qairt_diagnose_latency(
        manifest_uri: str,
        manifest_sha256: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _invoke("diagnose_latency", manifest_uri, manifest_sha256, config=config or {})

    expert_tools: tuple[tuple[str, str], ...] = (
        ("qairt_prepare_vectors", "prepare_vectors"),
        ("qairt_ar_convert", "ar_convert"),
        ("qairt_split", "split"),
        ("qairt_mha2sha", "mha2sha"),
        ("qairt_convert", "convert"),
        ("qairt_quantize", "quantize"),
        ("qairt_compile_context", "compile_context"),
        ("qairt_run_graph", "run_graph"),
        ("qairt_run_chain", "run_chain"),
        ("qairt_profile", "profile"),
    )

    def make_expert(stage_method: str) -> Callable[[str, str, dict[str, Any] | None], dict[str, Any]]:
        def expert(
            manifest_uri: str,
            manifest_sha256: str,
            config: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            return _invoke(stage_method, manifest_uri, manifest_sha256, config=config or {})

        expert.__name__ = stage_method
        return expert

    # Stages that are deprecated beyond the legacy surface itself: production
    # quantization is always AIMET ``apply_encodings``, so the standalone
    # quantizer survives only as a debugging comparison.
    extra_deprecation_notes = {
        "qairt_quantize": (
            " Standalone calibration quantization is not a production path; "
            "supply AIMET encodings through apply_encodings instead."
        ),
    }

    for tool_name, method_name in expert_tools:
        server.tool(
            name=tool_name,
            description=(
                f"DEPRECATED expert synchronous QAIRT stage: {method_name}."
                + extra_deprecation_notes.get(tool_name, "")
            ),
        )(make_expert(method_name))


def create_server(
    *,
    legacy: bool = False,
    jobs_root: str | None = None,
    client: QairtAgentClient | None = None,
) -> Any:
    if FastMCP is None:
        raise RuntimeError("MCP support is not installed; install qairt-agent[mcp]") from _MCP_IMPORT_ERROR

    server = FastMCP(
        "qairt-agent",
        instructions=(
            "Asynchronous QAIRT 2.48 workflow jobs. Submit a job and poll/watch it by id; "
            "the journal persists state, events, and verified stage receipts."
            if not legacy
            else "DEPRECATED synchronous QAIRT tools; prefer the asynchronous submit_job/get_job API."
        ),
    )
    if legacy:
        _register_legacy_tools(server)
    else:
        _register_async_tools(server, jobs_root=jobs_root, client=client)
    return server


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="qairt-agent-mcp")
    parser.add_argument(
        "--legacy",
        action="store_true",
        default=os.environ.get("QAIRT_AGENT_MCP_LEGACY") == "1",
        help="expose the deprecated synchronous tool set",
    )
    parser.add_argument("--jobs-root", default=None, help="job journal root (default .qairt-agent/jobs)")
    args = parser.parse_args(argv)
    create_server(legacy=args.legacy, jobs_root=args.jobs_root).run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
