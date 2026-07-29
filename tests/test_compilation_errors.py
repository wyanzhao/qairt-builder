"""Tests for QAIRT adapter compilation error diagnostics."""

from __future__ import annotations

import pytest

from qairt_agent.qairt_adapter.adapter import (
    _CREATE_DEVICE_RE,
    _SOC_MODEL_ZERO_RE,
    _wrap_genai_build_error,
)
from qairt_agent.qairt_adapter.errors import (
    QairtAdapterError,
    QairtCompilationError,
)


def test_compilation_error_is_adapter_error() -> None:
    error = QairtCompilationError("failed", details={"soc_model": 87})

    assert isinstance(error, QairtAdapterError)
    assert error.details == {"soc_model": 87}
    assert "failed" in str(error)


def test_compilation_error_defaults_empty_details() -> None:
    error = QairtCompilationError("failed")

    assert error.details == {}


@pytest.mark.parametrize(
    "text",
    [
        "Unknown config socModel 0",
        "unknown config soc_model: 0",
        "unknown config SOC MODEL=0",
    ],
)
def test_socmodel_zero_pattern_drives_structured_diagnostics(text: str) -> None:
    assert _SOC_MODEL_ZERO_RE.search(text)
    error = _wrap_genai_build_error(RuntimeError(text))

    assert "effective socModel 0" in str(error)
    assert error.details["pipeline"] == "genai_builder"


@pytest.mark.parametrize(
    "text",
    [
        "<NetRunErrorCode.CREATE_DEVICE: 11>",
        "Failed to create a device handle",
    ],
)
def test_create_device_pattern_drives_structured_diagnostics(text: str) -> None:
    assert _CREATE_DEVICE_RE.search(text)
    error = _wrap_genai_build_error(RuntimeError(text))

    assert "failed to create a device handle" in str(error).lower()


def test_generic_device_handle_text_does_not_trigger_target_diagnosis() -> None:
    text = "failed to release device handle after shutdown"
    assert not _CREATE_DEVICE_RE.search(text)

    error = _wrap_genai_build_error(RuntimeError(text))
    assert "container build failed" in str(error)


def test_low_level_fallback_keeps_low_level_pipeline_and_wording() -> None:
    error = _wrap_genai_build_error(
        RuntimeError("compiler crashed"),
        extra_diagnostics={"pipeline": "low_level"},
    )

    assert error.details["pipeline"] == "low_level"
    assert "context-binary compilation failed" in str(error)
    assert "container build failed" not in str(error)
