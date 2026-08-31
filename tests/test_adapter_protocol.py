"""The pipeline<->adapter boundary has one declared shape.

The pipeline consumed ~30 adapter methods through `Callable[[], Any]`, and the
test fake is 500 lines hand-maintained beside it. Drift between the real
adapter, the fake and the call sites was caught only by eye.
"""

from __future__ import annotations

import inspect

import pytest

from qairt_agent.qairt_adapter import (
    QairtAdapterOptionalProtocol,
    QairtAdapterProtocol,
    QairtSdkAdapter,
)

from test_pipeline import FakeAdapter, FakeAdapterFactory


def _uninitialized(cls):
    return cls.__new__(cls)


def _members(protocol) -> set[str]:
    """The methods a Protocol declares, on any supported Python.

    3.12 exposes `__protocol_attrs__`; earlier versions do not, so fall back to
    the annotations-and-callables the class body declares.
    """

    declared = getattr(protocol, "__protocol_attrs__", None)
    if declared is not None:
        return {name for name in declared if not name.startswith("_")}
    return {
        name
        for name, value in vars(protocol).items()
        if not name.startswith("_") and callable(value)
    }


def test_the_real_adapter_satisfies_the_protocol() -> None:
    assert isinstance(_uninitialized(QairtSdkAdapter), QairtAdapterProtocol)
    assert isinstance(_uninitialized(QairtSdkAdapter), QairtAdapterOptionalProtocol)


def test_the_test_fake_satisfies_the_same_protocol() -> None:
    # The fake and the real adapter are checked against one declaration, so a
    # method renamed in one and not the other is a failure, not a surprise.
    assert isinstance(_uninitialized(FakeAdapter), QairtAdapterProtocol)


def test_the_adapter_factory_returns_something_protocol_shaped() -> None:
    assert isinstance(FakeAdapterFactory()(), QairtAdapterProtocol)


@pytest.mark.parametrize(
    "method",
    sorted(_members(QairtAdapterProtocol)),
)
def test_every_required_method_exists_on_both_implementations(method: str) -> None:
    assert callable(getattr(QairtSdkAdapter, method, None)), method
    assert callable(getattr(FakeAdapter, method, None)), method


def test_a_missing_method_fails_the_protocol_check() -> None:
    """A deliberate drift is detected.

    This is the failure the boundary existed without: an adapter that stopped
    providing `compile_context` used to reach the call site and fail there.
    """

    class DriftedAdapter(FakeAdapter):
        compile_context = None  # type: ignore[assignment]

    assert not isinstance(_uninitialized(DriftedAdapter), QairtAdapterProtocol)


def test_optional_methods_are_probed_not_required() -> None:
    # A missing capture_device_execution degrades the report with a reason; it
    # must not make an adapter unusable.
    assert "capture_device_execution" in _members(QairtAdapterOptionalProtocol)
    assert "capture_device_execution" not in _members(QairtAdapterProtocol)


def test_the_protocol_covers_what_the_pipeline_actually_calls() -> None:
    """Every unconditionally called adapter method is declared."""

    import re
    from pathlib import Path

    source = (
        Path(__file__).parents[1] / "src" / "qairt_agent" / "pipeline.py"
    ).read_text(encoding="utf-8")
    called = set(re.findall(r"\badapter\.([a-z_][a-z0-9_]*)\(", source))
    probed = {
        name
        for name in re.findall(r'(?:hasattr|getattr)\(adapter, "([a-z_]+)"', source)
    }
    declared = _members(QairtAdapterProtocol) | _members(
        QairtAdapterOptionalProtocol
    )

    # Private SDK helpers the pipeline reaches for only as a fallback are out
    # of the declared surface by design; everything else must be declared.
    undeclared = {
        name
        for name in called - probed - declared
        if not name.startswith("_")
    }
    assert not undeclared, (
        "pipeline calls adapter methods the Protocol does not declare: "
        f"{sorted(undeclared)}"
    )
