from __future__ import annotations

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
