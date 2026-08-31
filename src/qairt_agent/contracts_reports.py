"""Typed models for the published report payloads.

Report payloads were schema-string-tagged dicts, validated by isinstance guards
scattered across the consumption sites -- unlike the spec and manifest
contracts, which have been typed all along. A reader could not tell which keys a
schema promises without reading the constructor, and a consumer could not tell
whether a missing key meant "absent" or "this is a different schema".

The models here are **round-trip exact**: a payload validated and dumped back
must be byte-identical to what was published, because these documents are
content-addressed evidence and their hashes are recorded. Every model therefore
allows extra keys and preserves them, and `tests/test_contracts_reports.py`
proves the round trip on real published reports.

Migration is by report family. The multi-AR aggregates come first because they
are the ones consumed programmatically -- by diagnosis attribution and by
`qairt-agent compare` -- and the per-AR reports they wrap stay plain payloads
until their own landing.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MULTI_AR_SQNR_SCHEMA = "qairt-agent.multi-ar-sqnr-report.v1"
MULTI_AR_LATENCY_SCHEMA = "qairt-agent.multi-ar-latency-report.v1"
DEVICE_EXECUTION_SCHEMA_V2 = "qairt-agent.device-execution/2"


class ReportModel(BaseModel):
    """Base for every published report payload.

    ``extra="allow"`` is not laxness: a published report is evidence whose hash
    is recorded, so a model that silently dropped an unmodelled key would make
    the round trip lossy and the recorded hash unreproducible. Unknown keys are
    preserved verbatim and surface as attributes.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    def to_payload(self) -> dict[str, Any]:
        """The published JSON body, extras included, in declaration order."""

        return self.model_dump(mode="json")


class CoverageBlock(ReportModel):
    """Which ARs a multi-AR stage actually executed."""

    mode: str
    requested_ars: list[int]
    executed_ars: list[int]
    missing_ars: list[int] = Field(default_factory=list)
    complete: bool
    context_lengths: list[int] = Field(default_factory=list)


class LatencyCoverageBlock(CoverageBlock):
    """Coverage plus which ARs actually carried a device meter.

    An aggregate that claimed ``device_execution`` for the whole set while one
    AR had degraded to ``available=false`` is exactly what these fields exist to
    make impossible to write by accident.
    """

    metered_ars: list[int] = Field(default_factory=list)
    unmetered_ars: list[int] = Field(default_factory=list)
    device_meter_complete: bool = True


class ArResultEntry(ReportModel):
    """One AR's report plus the artifact it was published as."""

    report: dict[str, Any]
    report_artifact: dict[str, Any]


class MultiArSqnrReport(ReportModel):
    """`qairt-agent.multi-ar-sqnr-report.v1`."""

    schema_: Literal["qairt-agent.multi-ar-sqnr-report.v1"] = Field(
        alias="schema", serialization_alias="schema"
    )
    policy: Literal["report_only"]
    coverage: CoverageBlock
    results_by_ar: dict[str, ArResultEntry]
    reference_source: str
    reference_sources_by_ar: dict[str, Any]
    first_teacher_error: Any = None
    first_teacher_error_ar: int | None = None
    first_chain_error: Any = None
    first_chain_error_ar: int | None = None
    divergence_observed: bool

    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


class MultiArLatencyReport(ReportModel):
    """`qairt-agent.multi-ar-latency-report.v1`."""

    schema_: Literal["qairt-agent.multi-ar-latency-report.v1"] = Field(
        alias="schema", serialization_alias="schema"
    )
    policy: Literal["report_only"]
    # "partial" when any requested AR lost its device meter; never an
    # unconditional device claim over a degraded set.
    latency_metric: Literal["device_execution", "partial", "unavailable"]
    harness_diagnostics: dict[str, Any]
    coverage: LatencyCoverageBlock
    results_by_ar: dict[str, ArResultEntry]

    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


class DeviceExecutionBlock(ReportModel):
    """`qairt-agent.device-execution/2` -- the only latency metric.

    ``available=False`` carries a ``reason`` and nothing else; a block that
    measured something carries the meter, the statistic, and the per-sample
    values behind the mean.
    """

    schema_: str = Field(alias="schema", serialization_alias="schema")
    policy: Literal["report_only"] = "report_only"
    available: bool | None = None
    reason: str | None = None
    meter: str | None = None
    lane: str | None = None
    scope: str | None = None
    statistic: str | None = None
    production_latency_us: float | None = None
    production_latency_source: str | None = None
    production_latency_cv_percent: float | None = None
    samples_requested: int | None = None
    samples_used: int | None = None
    partial: bool | None = None

    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)

    @property
    def measured(self) -> bool:
        """Whether this block actually carries a device number."""

        return self.available is not False and self.production_latency_us is not None

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


__all__ = [
    "ArResultEntry",
    "CoverageBlock",
    "DEVICE_EXECUTION_SCHEMA_V2",
    "DeviceExecutionBlock",
    "LatencyCoverageBlock",
    "MULTI_AR_LATENCY_SCHEMA",
    "MULTI_AR_SQNR_SCHEMA",
    "MultiArLatencyReport",
    "MultiArSqnrReport",
    "ReportModel",
]
