"""Structured errors raised by the QAIRT Python adapter."""

from __future__ import annotations


class QairtAdapterError(RuntimeError):
    """Base error for adapter failures."""


class QairtPreflightError(QairtAdapterError):
    """The pinned SDK/host/target contract is not satisfied."""


class QairtSdkImportError(QairtAdapterError):
    """A required QAIRT Python API cannot be imported."""


class QairtConfigurationError(QairtAdapterError):
    """A requested build configuration is unsafe or inconsistent."""


class NativeKvConfigError(QairtConfigurationError):
    """A native-KV data-format config failed strict audit."""


class ExperimentalFeatureError(QairtConfigurationError):
    """An experimental path was used without its explicit validation gates."""
