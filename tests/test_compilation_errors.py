"""Tests for QAIRT adapter compilation error diagnostics."""

from __future__ import annotations

import pytest

from qairt_agent.qairt_adapter.errors import (
    QairtAdapterError,
    QairtCompilationError,
)


def test_compilation_error_is_adapter_error() -> None:
    error = QairtCompilationError("failed", details={"soc_model": 660})

    assert isinstance(error, QairtAdapterError)
    assert error.details == {"soc_model": 660}
    assert "failed" in str(error)


def test_compilation_error_defaults_empty_details() -> None:
    error = QairtCompilationError("failed")

    assert error.details == {}


def test_socmodel_zero_pattern_is_recognized() -> None:
    exc_text = (
        "(<NetRunErrorCode.CREATE_DEVICE: 11>, "
        "'Failed to create a device handle') "
        "Unknown config socModel 0"
    )

    assert "socModel 0" in exc_text
    assert "CREATE_DEVICE" in exc_text


def test_create_device_pattern_is_recognized() -> None:
    exc_text = "Failed to create a device handle"

    assert "device handle" in exc_text.lower()
