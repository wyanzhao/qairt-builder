"""Fail-fast QAIRT 2.48 Linux host and target validation."""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from qairt_agent.harness import (
    DEFAULT_CONSTRAINTS,
    HarnessConstraints,
    load_harness_constraints,
)

from .errors import QairtPreflightError
from .types import IssueSeverity, PreflightIssue, PreflightReport


PINNED_QAIRT_VERSION = DEFAULT_CONSTRAINTS.qairt_version
PINNED_QAIRT_BUILD_ID = DEFAULT_CONSTRAINTS.qairt_build_id
PINNED_UBUNTU_VERSION = DEFAULT_CONSTRAINTS.ubuntu_version
PINNED_PYTHON = DEFAULT_CONSTRAINTS.python_version_tuple
PINNED_TARGET_SOC = DEFAULT_CONSTRAINTS.target_chipset
PINNED_DSP_ARCH = DEFAULT_CONSTRAINTS.target_dsp_arch
PINNED_SOC_MODEL = DEFAULT_CONSTRAINTS.target_soc_model


@dataclass(frozen=True)
class PreflightSpec:
    """Standalone preflight input; arbitrary build specs are also accepted."""

    sdk_root: str | Path | None
    target_soc: str
    dsp_arch: str
    soc_model: int = PINNED_SOC_MODEL


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _nested(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        current = _read(current, key)
        if current is None:
            return None
    return current


def _first_path(value: Any, paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        candidate = _nested(value, *path)
        if candidate is not None:
            return candidate
    return None


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    """Parse scalar top-level keys from SDK metadata without a YAML dependency."""

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*([^#]+?)\s*$", line)
        if match:
            values[match.group(1)] = match.group(2).strip().strip("'\"")
    return values


def _read_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip("'\"")
    return values


def _find_htp_config(sdk_root: Path) -> Path | None:
    direct = (
        sdk_root
        / "lib"
        / "python"
        / "qti"
        / "aisw"
        / "converters"
        / "common"
        / "backend_aware_configs"
        / "htp_v2.json"
    )
    if direct.is_file():
        return direct
    candidates = tuple(sdk_root.glob("**/backend_aware_configs/htp_v2.json"))
    return candidates[0] if candidates else None


def _target_mapping(path: Path, target_soc: str) -> str | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # QAIRT 2.48 places this mapping at the top level.  Recursion keeps the
    # check stable if a patched SDK wraps backend-aware configuration in a
    # nested object, while still requiring an actual dictionary mapping.
    pending: list[Any] = [document]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            mapping = current.get("soc_model_to_arch")
            if isinstance(mapping, Mapping) and target_soc in mapping:
                return str(mapping[target_soc])
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return None


class PreflightChecker:
    """Validate the pinned SDK without importing it."""

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        system: Callable[[], str] = platform.system,
        machine: Callable[[], str] = platform.machine,
        python_version: tuple[int, int] | None = None,
        os_release_reader: Callable[[], Mapping[str, str]] = _read_os_release,
        constraints: HarnessConstraints | None = None,
    ) -> None:
        self._environ = environ if environ is not None else os.environ
        self._system = system
        self._machine = machine
        self._python_version = python_version or (sys.version_info.major, sys.version_info.minor)
        self._os_release_reader = os_release_reader
        # Resolve this at checker construction time instead of module import
        # time.  Direct Python/inline CLI users can therefore select a
        # project-owned harness through QAIRT_AGENT_HARNESS_CONSTRAINTS just
        # like detached container workers do.
        self._constraints = constraints or load_harness_constraints()

    def check(self, spec: Any) -> PreflightReport:
        issues: list[PreflightIssue] = []
        constraints = self._constraints

        sdk_root_value = _first_path(
            spec,
            (
                ("sdk_root",),
                ("sdk", "root"),
                ("environment", "sdk_root"),
                ("toolchain", "sdk_root"),
            ),
        )
        if sdk_root_value is None:
            sdk_root_value = self._environ.get("QAIRT_SDK_ROOT") or self._environ.get("QNN_SDK_ROOT")
        sdk_root = Path(sdk_root_value).expanduser() if sdk_root_value else None

        target_soc_value = _first_path(
            spec,
            (
                ("target_soc",),
                ("target", "soc"),
                ("target", "chipset"),
                ("hardware", "soc"),
                ("hardware", "chipset"),
                ("device", "soc"),
                ("device", "chipset"),
            ),
        )
        dsp_arch_value = _first_path(
            spec,
            (
                ("dsp_arch",),
                ("target", "dsp_arch"),
                ("hardware", "dsp_arch"),
                ("device", "dsp_arch"),
            ),
        )
        soc_model_value = _first_path(
            spec,
            (
                ("soc_model",),
                ("target", "soc_model"),
                ("hardware", "soc_model"),
                ("device", "soc_model"),
            ),
        )

        target_soc = str(target_soc_value).upper() if target_soc_value is not None else None
        dsp_arch = str(dsp_arch_value).lower() if dsp_arch_value is not None else None
        try:
            soc_model = int(soc_model_value) if soc_model_value is not None else None
        except (TypeError, ValueError):
            soc_model = None

        if self._system().lower() != "linux":
            issues.append(
                PreflightIssue(
                    "host.os",
                    f"QAIRT {constraints.qairt_version} build execution requires Linux; "
                    f"pinned host is Ubuntu {constraints.ubuntu_version}",
                )
            )
        os_release = dict(self._os_release_reader())
        if os_release:
            distribution = (os_release.get("ID") or "").lower()
            version = os_release.get("VERSION_ID")
            if distribution != "ubuntu" or version != constraints.ubuntu_version:
                issues.append(
                    PreflightIssue(
                        "host.ubuntu",
                        f"expected Ubuntu {constraints.ubuntu_version}, got "
                        f"{distribution or '<unknown>'} {version or '<unknown>'}",
                    )
                )
        elif self._system().lower() == "linux":
            issues.append(
                PreflightIssue(
                    "host.os_release_missing",
                    "cannot verify Ubuntu release because /etc/os-release is unavailable",
                )
            )

        if self._machine().lower() not in {"x86_64", "amd64"}:
            issues.append(
                PreflightIssue("host.arch", f"expected x86_64 host, got {self._machine()}")
            )
        if self._python_version != constraints.python_version_tuple:
            issues.append(
                PreflightIssue(
                    "host.python",
                    f"expected Python {constraints.python_version}, got "
                    f"{self._python_version[0]}.{self._python_version[1]}",
                )
            )

        sdk_version: str | None = None
        sdk_build_id: str | None = None
        if sdk_root is None:
            issues.append(
                PreflightIssue(
                    "sdk.root_missing",
                    "sdk_root must be explicit in the build spec or QAIRT_SDK_ROOT/QNN_SDK_ROOT",
                )
            )
        elif not sdk_root.is_dir():
            issues.append(PreflightIssue("sdk.root_invalid", f"SDK root does not exist: {sdk_root}"))
        else:
            sdk_yaml = sdk_root / "sdk.yaml"
            if not sdk_yaml.is_file():
                issues.append(PreflightIssue("sdk.metadata_missing", f"missing {sdk_yaml}"))
            else:
                metadata = _parse_simple_yaml(sdk_yaml)
                sdk_version = metadata.get("version")
                sdk_build_id = metadata.get("build_id")
                if sdk_version != constraints.qairt_version:
                    issues.append(
                        PreflightIssue(
                            "sdk.version",
                            f"expected QAIRT {constraints.qairt_version}, "
                            f"got {sdk_version or '<missing>'}",
                        )
                    )
                if sdk_build_id != constraints.qairt_build_id:
                    issues.append(
                        PreflightIssue(
                            "sdk.build_id",
                            f"expected QAIRT build {constraints.qairt_build_id}, "
                            f"got {sdk_build_id or '<missing>'}",
                        )
                    )
            python_api = sdk_root / "lib" / "python"
            if not python_api.is_dir():
                issues.append(
                    PreflightIssue("sdk.python_api_missing", f"missing SDK Python API: {python_api}")
                )

            htp_config = _find_htp_config(sdk_root)
            if htp_config is None:
                issues.append(
                    PreflightIssue(
                        "sdk.target_map_unverified",
                        "htp_v2.json was not found; "
                        f"{constraints.target_chipset} to "
                        f"{constraints.target_dsp_arch} SDK mapping could not be verified",
                        IssueSeverity.WARNING,
                    )
                )
            else:
                mapped_arch = _target_mapping(
                    htp_config,
                    constraints.target_chipset,
                )
                if mapped_arch != constraints.target_dsp_arch:
                    issues.append(
                        PreflightIssue(
                            "sdk.target_map",
                            f"{htp_config} must map {constraints.target_chipset} to "
                            f"{constraints.target_dsp_arch}, "
                            f"got {mapped_arch or '<missing>'}",
                        )
                    )

        if target_soc != constraints.target_chipset:
            issues.append(
                PreflightIssue(
                    "target.soc",
                    f"target_soc must be explicit {constraints.target_chipset}; "
                    "no SDK default is allowed",
                )
            )
        if dsp_arch != constraints.target_dsp_arch:
            issues.append(
                PreflightIssue(
                    "target.dsp_arch",
                    f"dsp_arch must be explicit {constraints.target_dsp_arch}; "
                    "SDK fallback is forbidden",
                )
            )
        if soc_model != constraints.target_soc_model:
            issues.append(
                PreflightIssue(
                    "target.soc_model",
                    f"soc_model must be explicit {constraints.target_soc_model} "
                    f"for {constraints.target_chipset}",
                )
            )

        return PreflightReport(
            issues=tuple(issues),
            sdk_root=sdk_root,
            sdk_version=sdk_version,
            sdk_build_id=sdk_build_id,
            target_soc=target_soc,
            dsp_arch=dsp_arch,
            soc_model=soc_model,
        )


def require_preflight(report: PreflightReport) -> PreflightReport:
    """Raise one concise error if a report has any blocking issue."""

    if not report.ok:
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in report.errors)
        raise QairtPreflightError(f"QAIRT preflight failed: {details}")
    return report
