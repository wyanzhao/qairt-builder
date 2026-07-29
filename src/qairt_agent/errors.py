"""Structured errors shared by the Python and MCP interfaces.

The public tool boundary returns :class:`ToolErrorData`; exceptions are kept as
an implementation detail and can be converted without exposing a traceback or
non-JSON-serializable SDK objects.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ErrorCode(str, Enum):
    """Stable, machine-readable error codes returned by agent tools."""

    INVALID_SPEC = "invalid_spec"
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    ARTIFACT_HASH_MISMATCH = "artifact_hash_mismatch"
    ARTIFACT_PUBLISH_FAILED = "artifact_publish_failed"
    MANIFEST_CONFLICT = "manifest_conflict"
    MANIFEST_INVALID = "manifest_invalid"
    QAIRT_UNAVAILABLE = "qairt_unavailable"
    QAIRT_VERSION_MISMATCH = "qairt_version_mismatch"
    PREFLIGHT_FAILED = "preflight_failed"
    STAGE_FAILED = "stage_failed"
    INTERNAL_ERROR = "internal_error"
    # native-workflow codes
    PRESET_NOT_FOUND = "preset_not_found"
    UNSUPPORTED_SDK_CAPABILITY = "unsupported_sdk_capability"
    PROJECT_NOT_INITIALIZED = "project_not_initialized"
    JOB_NOT_FOUND = "job_not_found"
    JOB_CONFLICT = "job_conflict"
    JOB_CANCELLED = "job_cancelled"
    JOB_ORPHANED = "job_orphaned"
    DOCKER_UNAVAILABLE = "docker_unavailable"
    APPLE_CONTAINER_UNAVAILABLE = "apple_container_unavailable"
    DEVICE_UNAVAILABLE = "device_unavailable"
    LEASE_CONFLICT = "lease_conflict"
    PICKLE_REJECTED = "pickle_rejected"


def _json_safe(value: Any) -> JsonValue:
    """Convert arbitrary exception context to a bounded JSON-compatible value."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


class ToolErrorData(BaseModel):
    """Serializable error payload returned by a failed tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: ErrorCode
    message: str = Field(min_length=1)
    stage: str | None = None
    retryable: bool = False
    details: dict[str, JsonValue] = Field(default_factory=dict)

    def as_dict(self) -> dict[str, JsonValue]:
        """Return the exact JSON payload expected at the tool boundary."""

        return self.model_dump(mode="json")

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        stage: str | None = None,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> "ToolErrorData":
        """Create a safe tool payload from an exception.

        A :class:`QairtAgentError` retains its declared code and metadata unless
        explicitly overridden with a non-default value.
        """

        if isinstance(exc, ToolError):
            return exc.data
        if isinstance(exc, QairtAgentError):
            effective_code = exc.code if code == ErrorCode.INTERNAL_ERROR else code
            effective_stage = stage if stage is not None else exc.stage
            effective_retryable = retryable or exc.retryable
            merged_details: dict[str, Any] = {**exc.details, **(details or {})}
            return cls(
                code=effective_code,
                message=str(exc),
                stage=effective_stage,
                retryable=effective_retryable,
                details={key: _json_safe(value) for key, value in merged_details.items()},
            )

        return cls(
            code=code,
            message=str(exc) or exc.__class__.__name__,
            stage=stage,
            retryable=retryable,
            details={key: _json_safe(value) for key, value in (details or {}).items()},
        )


class QairtAgentError(Exception):
    """Base exception carrying structured, user-safe context."""

    code = ErrorCode.INTERNAL_ERROR
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        retryable: bool | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        if retryable is not None:
            self.retryable = retryable
        self.details = dict(details or {})

    def to_tool_error(self) -> ToolErrorData:
        return ToolErrorData.from_exception(self)

    def as_dict(self) -> dict[str, JsonValue]:
        """Compatibility hook used by the MCP exception boundary."""

        return self.to_tool_error().as_dict()


class InvalidSpecError(QairtAgentError):
    code = ErrorCode.INVALID_SPEC


class ArtifactNotFoundError(QairtAgentError):
    code = ErrorCode.ARTIFACT_NOT_FOUND


class ArtifactIntegrityError(QairtAgentError):
    code = ErrorCode.ARTIFACT_HASH_MISMATCH


class ArtifactPublishError(QairtAgentError):
    code = ErrorCode.ARTIFACT_PUBLISH_FAILED


class ManifestConflictError(QairtAgentError):
    code = ErrorCode.MANIFEST_CONFLICT


class ManifestInvalidError(QairtAgentError):
    code = ErrorCode.MANIFEST_INVALID


class PresetNotFoundError(QairtAgentError):
    code = ErrorCode.PRESET_NOT_FOUND


class UnsupportedSdkCapabilityError(QairtAgentError):
    code = ErrorCode.UNSUPPORTED_SDK_CAPABILITY


class ProjectNotInitializedError(QairtAgentError):
    code = ErrorCode.PROJECT_NOT_INITIALIZED


class JobNotFoundError(QairtAgentError):
    code = ErrorCode.JOB_NOT_FOUND


class JobConflictError(QairtAgentError):
    code = ErrorCode.JOB_CONFLICT


class JobCancelledError(QairtAgentError):
    code = ErrorCode.JOB_CANCELLED


class JobOrphanedError(QairtAgentError):
    code = ErrorCode.JOB_ORPHANED


class DockerUnavailableError(QairtAgentError):
    code = ErrorCode.DOCKER_UNAVAILABLE


class AppleContainerUnavailableError(QairtAgentError):
    code = ErrorCode.APPLE_CONTAINER_UNAVAILABLE


class WorkerCommandError(QairtAgentError):
    """A container backend ran successfully but its workload failed."""

    code = ErrorCode.STAGE_FAILED


class DeviceUnavailableError(QairtAgentError):
    code = ErrorCode.DEVICE_UNAVAILABLE


class LeaseConflictError(QairtAgentError):
    code = ErrorCode.LEASE_CONFLICT


class PickleRejectedError(QairtAgentError):
    code = ErrorCode.PICKLE_REJECTED


class ToolError(QairtAgentError):
    """Exception wrapper for an already structured tool error."""

    def __init__(self, data: ToolErrorData) -> None:
        super().__init__(
            data.message,
            stage=data.stage,
            retryable=data.retryable,
            details=data.details,
        )
        self.data = data
        self.code = data.code


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactNotFoundError",
    "ArtifactPublishError",
    "AppleContainerUnavailableError",
    "DeviceUnavailableError",
    "DockerUnavailableError",
    "ErrorCode",
    "InvalidSpecError",
    "JobCancelledError",
    "JobConflictError",
    "JobNotFoundError",
    "JobOrphanedError",
    "LeaseConflictError",
    "ManifestConflictError",
    "ManifestInvalidError",
    "PickleRejectedError",
    "PresetNotFoundError",
    "ProjectNotInitializedError",
    "QairtAgentError",
    "ToolError",
    "ToolErrorData",
    "WorkerCommandError",
    "UnsupportedSdkCapabilityError",
]
