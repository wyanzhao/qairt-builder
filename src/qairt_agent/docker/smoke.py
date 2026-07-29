"""Mounted-SDK smoke test for the pinned Docker worker image.

The SDK is intentionally not part of the image build context.  This module is
therefore executed in a freshly built image with the user's SDK mounted
read-only at ``/opt/qairt``.  It verifies the SDK dependency contract and both
Python API lanes before the image is accepted by ``qairt-agent image build``.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from qairt_agent.harness import HarnessConstraints, load_harness_constraints

_SDK_ROOT = Path(os.environ.get("QAIRT_SDK_ROOT", "/opt/qairt"))
_IMAGE_REQUIREMENTS = Path(
    os.environ.get(
        "QAIRT_AGENT_IMAGE_REQUIREMENTS",
        "/opt/qairt-agent/docker/requirements.txt",
    )
)
_EXACT_REQUIREMENT_RE = re.compile(
    r"^\s*([A-Za-z0-9_.-]+)==([^\s;]+)\s*$"
)


def _check_sdk_dependencies(sdk_root: Path) -> None:
    checker = sdk_root / "bin" / "check-python-dependency"
    if not checker.is_file():
        raise FileNotFoundError(f"QAIRT dependency checker not found: {checker}")
    result = subprocess.run(
        [sys.executable, str(checker), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"QAIRT Python dependency check failed with exit code "
            f"{result.returncode}: {details}"
        )


def _locked_versions(
    requirements_path: Path,
    constraints: HarnessConstraints,
) -> dict[str, str]:
    if not requirements_path.is_file():
        raise FileNotFoundError(
            f"worker dependency lock not found: {requirements_path}"
        )
    expected: dict[str, str] = {}
    for line_number, raw in enumerate(
        requirements_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _EXACT_REQUIREMENT_RE.fullmatch(line)
        if match is None:
            raise RuntimeError(
                f"worker dependency lock line {line_number} must use "
                f"name==version form: {line!r}"
            )
        name, version = match.groups()
        normalized = name.lower().replace("_", "-")
        if normalized in expected and expected[normalized] != version:
            raise RuntimeError(
                f"conflicting worker dependency pins for {name}"
            )
        expected[normalized] = version
    torch_pin = expected.get("torch")
    if torch_pin is not None and torch_pin != constraints.torch_version:
        raise RuntimeError(
            "worker dependency lock torch pin does not match harness "
            f"{constraints.torch_version}: {torch_pin}"
        )
    expected["torch"] = constraints.torch_version
    return expected


def _check_pinned_versions(
    requirements_path: Path,
    constraints: HarnessConstraints,
) -> dict[str, str]:
    expected_versions = _locked_versions(requirements_path, constraints)
    installed: dict[str, str] = {}
    for distribution, expected in sorted(expected_versions.items()):
        actual = importlib.metadata.version(distribution)
        comparable = actual.split("+", 1)[0] if distribution == "torch" else actual
        if comparable != expected:
            raise RuntimeError(
                f"{distribution} version mismatch: expected {expected}, got {actual}"
            )
        installed[distribution] = actual
    return installed


def _import_api_lanes(python_abi: str) -> tuple[str, ...]:
    import qairt.api.configs.common as common_config
    import qairt.api.transforms._transform as transform_api
    import qairt.api.transforms.model_transformer_config as transformer_config
    import qairt.optimizer.onnx as onnx_optimizer
    from qairt import Model, compile, convert
    from qairt.gen_ai_api.builders.qwen.audio_encoder_builder import (
        Qwen3OmniAudioEncoderBuilderHTP,
    )
    from qairt.gen_ai_api.builders.qwen.builder import Qwen3_5BuilderHTP
    from qairt.gen_ai_api.gen_ai_builder_factory import GenAIBuilderFactory
    from qti.aisw import dlc_utils
    from qti.aisw.converters import common as converter_common
    from qti.aisw.core.model_level_api import utils as model_level_utils
    from qti.aisw.genai import genie as native_genie

    # Several higher-level QAIRT modules catch native-extension ImportError and
    # substitute a dummy Genie module.  Import and identify the native modules
    # directly so a missing libpython/libc++/SDK runtime cannot pass smoke.
    required_native_modules = (
        native_genie,
        converter_common.ir_graph,
        converter_common.qnn_ir,
        converter_common.ir_quantizer,
        converter_common.modeltools,
        dlc_utils.modeltools,
        dlc_utils.dlcontainer,
        model_level_utils.py_net_run,
    )
    unexpected = [
        getattr(module, "__name__", repr(module))
        for module in required_native_modules
        if module is None
        or "dummy_lib_genie" in getattr(module, "__name__", "")
        or not getattr(module, "__name__", "").endswith(python_abi)
    ]
    if unexpected:
        raise RuntimeError(
            f"QAIRT Python ABI {python_abi} native extension smoke failed: "
            f"{unexpected}"
        )

    # Referencing the symbols prevents an import-only optimizer from dropping
    # them and makes the intended acceptance contract explicit.
    symbols = (
        Model,
        convert,
        compile,
        GenAIBuilderFactory,
        Qwen3_5BuilderHTP,
        Qwen3OmniAudioEncoderBuilderHTP,
    )
    modules = (
        *required_native_modules,
        onnx_optimizer,
        transform_api,
        transformer_config,
        common_config,
    )
    return (
        *(f"{symbol.__module__}.{symbol.__name__}" for symbol in symbols),
        *(module.__name__ for module in modules),
    )


def _sdk_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = re.match(
            r"^\s*([A-Za-z0-9_]+)\s*:\s*([^#]+?)\s*$",
            raw,
        )
        if match:
            metadata[match.group(1)] = match.group(2).strip().strip("'\"")
    return metadata


def main() -> int:
    constraints = load_harness_constraints()
    expected_python = constraints.python_version_tuple
    actual_python = (sys.version_info.major, sys.version_info.minor)
    if actual_python != expected_python:
        raise RuntimeError(
            f"worker Python mismatch: expected {constraints.python_version}, "
            f"got {actual_python[0]}.{actual_python[1]}"
        )

    sdk_yaml = _SDK_ROOT / "sdk.yaml"
    if not sdk_yaml.is_file():
        raise FileNotFoundError(f"mounted QAIRT SDK metadata not found: {sdk_yaml}")
    metadata = _sdk_metadata(sdk_yaml)
    if (
        metadata.get("version") != constraints.qairt_version
        or metadata.get("build_id") != constraints.qairt_build_id
    ):
        raise RuntimeError(
            "mounted QAIRT SDK does not match harness: expected "
            f"{constraints.qairt_version}/{constraints.qairt_build_id}, got "
            f"{metadata.get('version', '<missing>')}/"
            f"{metadata.get('build_id', '<missing>')}"
        )
    _check_sdk_dependencies(_SDK_ROOT)
    versions = _check_pinned_versions(_IMAGE_REQUIREMENTS, constraints)
    python_abi = "".join(str(part) for part in expected_python)
    symbols = _import_api_lanes(python_abi)
    print(
        json.dumps(
            {
                "ok": True,
                "sdk_root": str(_SDK_ROOT),
                "qairt_version": constraints.qairt_version,
                "qairt_build_id": constraints.qairt_build_id,
                "python_version": constraints.python_version,
                "constraints": str(constraints.source_path),
                "versions": versions,
                "symbols": symbols,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - executed inside the worker image
    raise SystemExit(main())
