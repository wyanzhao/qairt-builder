from __future__ import annotations

from pathlib import Path

from qairt_agent.agent import QairtAgentClient
from qairt_agent.jobs.worker import DEFAULT_WORKFLOW_STAGES
from qairt_agent.mcp_server import _invoke, _safe, create_server


def test_default_server_exposes_only_async_tools() -> None:
    server = create_server()
    names = {tool.name for tool in server._tool_manager.list_tools()}
    assert names == {"submit_job", "get_job", "cancel_job", "resume_job"}


def test_legacy_server_exposes_deprecated_tools() -> None:
    server = create_server(legacy=True)
    names = {tool.name for tool in server._tool_manager.list_tools()}
    assert names == {
        "qairt_plan",
        "qairt_generate_config",
        "qairt_build",
        "qairt_build_genai_container",
        "qairt_validate",
        "qairt_benchmark",
        "qairt_diagnose_quality",
        "qairt_diagnose_latency",
        "qairt_prepare_vectors",
        "qairt_ar_convert",
        "qairt_split",
        "qairt_mha2sha",
        "qairt_convert",
        "qairt_quantize",
        "qairt_compile_context",
        "qairt_run_graph",
        "qairt_run_chain",
        "qairt_profile",
    }


def test_legacy_boundary_returns_structured_contract_error() -> None:
    result = _invoke("plan", {"family": "not-a-family"}, offline=True)
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_spec"
    assert result["error"]["retryable"] is False


def test_safe_wraps_exceptions_structurally() -> None:
    def boom() -> None:
        raise RuntimeError("nope")

    result = _safe(boom)
    assert result["ok"] is False
    assert result["error"]["message"] == "nope"

    assert _safe(lambda: {"ok": True}) == {"ok": True}
    assert _safe(lambda: 5) == {"ok": True, "data": 5}


# --------------------------------------------------------------------------- #
# submit_job goes through the detached pinned worker (T15)
# --------------------------------------------------------------------------- #


class _RecordingLauncher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def __call__(self, job_id: str, jobs_root: object) -> int:
        self.calls.append((job_id, jobs_root))
        return 4321


def _spec_dict(tmp_path: Path) -> dict:
    return {
        "family": "qwen3",
        "sources": {
            "text": {
                "onnx_path": "/m/model.onnx",
                "encodings_path": "/m/model.encodings",
            }
        },
        "output_root": str(tmp_path / "artifacts"),
        "vectors": {"mode": "provided", "validation_manifest": "/v/golden.json"},
    }


def _submit_tool(server) -> object:
    return server._tool_manager._tools["submit_job"].fn


def test_submit_job_launches_the_detached_worker(tmp_path) -> None:
    # In-process daemon threads contradict the detached-worker contract: the
    # job dies with the MCP process and never gets the pinned environment.
    launcher = _RecordingLauncher()
    client = QairtAgentClient(jobs_root=tmp_path / "jobs", background=False)
    server = create_server(client=client, launcher=launcher)

    result = _submit_tool(server)(_spec_dict(tmp_path))

    assert result["execution"] == "detached"
    assert result["worker_pid"] == 4321
    assert [job for job, _ in launcher.calls] == [result["job_id"]]


def test_submit_job_defaults_to_the_cli_workflow_stages(tmp_path) -> None:
    launcher = _RecordingLauncher()
    client = QairtAgentClient(jobs_root=tmp_path / "jobs", background=False)
    server = create_server(client=client, launcher=launcher)

    result = _submit_tool(server)(_spec_dict(tmp_path))

    journal = client.job(result["job_id"]).journal
    assert tuple(journal.launcher()["stages"]) == DEFAULT_WORKFLOW_STAGES


def test_submit_job_runs_in_process_only_when_inline_is_explicit(tmp_path) -> None:
    launcher = _RecordingLauncher()
    client = QairtAgentClient(jobs_root=tmp_path / "jobs", background=False)
    server = create_server(client=client, launcher=launcher)

    result = _submit_tool(server)(_spec_dict(tmp_path), None, None, True)

    assert result["execution"] == "inline"
    assert launcher.calls == []
