"""Native-KV data-format config generation and strict auditing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import NativeKvConfigError
from .types import NativeKvAuditReport, NativeKvGraphExpectation


NATIVE_KV_DATA_FORMAT = "QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT"


def _is_kv_name(name: str) -> bool:
    normalized = name.lower()
    return ("key" in normalized or "value" in normalized) and not (
        "recurrent_state" in normalized or "conv_state" in normalized
    )


def _load_config(config: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    path = Path(config)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NativeKvConfigError(f"cannot read native-KV config {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise NativeKvConfigError("native-KV config root must be a JSON object")
    return document


def build_native_kv_config(
    expectations: Sequence[NativeKvGraphExpectation],
) -> dict[str, list[dict[str, Any]]]:
    """Generate QAIRT 2.48's canonical ``graphs[].tensors[]`` JSON shape."""

    graphs: list[dict[str, Any]] = []
    for expectation in expectations:
        input_names = tuple(name for name in expectation.input_names if _is_kv_name(name))
        output_names = tuple(name for name in expectation.output_names if _is_kv_name(name))
        selected = input_names + output_names if expectation.ar % 32 == 0 else input_names
        if selected:
            graphs.append(
                {
                    "graph_name": expectation.graph_name,
                    "tensors": [
                        {"tensor_name": name, "dataFormat": NATIVE_KV_DATA_FORMAT}
                        for name in selected
                    ],
                }
            )
    return {"graphs": graphs}


def audit_native_kv_config(
    config: str | Path | Mapping[str, Any],
    *,
    expectations: Sequence[NativeKvGraphExpectation] = (),
    expected_graph_names: Sequence[str] = (),
    require_nonempty: bool = True,
) -> NativeKvAuditReport:
    """Audit exact graph names, tensor sets, and HMX layout values."""

    document = _load_config(config)
    issues: list[str] = []
    raw_graphs = document.get("graphs")
    if not isinstance(raw_graphs, list):
        raw_graphs = []
        issues.append("root field 'graphs' must be a list")

    graph_tensors: dict[str, tuple[str, ...]] = {}
    tensor_count = 0
    for index, raw_graph in enumerate(raw_graphs):
        if not isinstance(raw_graph, Mapping):
            issues.append(f"graphs[{index}] must be an object")
            continue
        graph_name = raw_graph.get("graph_name")
        if not isinstance(graph_name, str) or not graph_name:
            issues.append(f"graphs[{index}].graph_name must be a non-empty string")
            continue
        if graph_name in graph_tensors:
            issues.append(f"duplicate graph_name {graph_name!r}")
            continue
        raw_tensors = raw_graph.get("tensors")
        if not isinstance(raw_tensors, list) or not raw_tensors:
            issues.append(f"graph {graph_name!r} must contain a non-empty tensors list")
            graph_tensors[graph_name] = ()
            continue
        names: list[str] = []
        for tensor_index, raw_tensor in enumerate(raw_tensors):
            if not isinstance(raw_tensor, Mapping):
                issues.append(f"{graph_name}.tensors[{tensor_index}] must be an object")
                continue
            tensor_name = raw_tensor.get("tensor_name")
            data_format = raw_tensor.get("dataFormat")
            if not isinstance(tensor_name, str) or not tensor_name:
                issues.append(f"{graph_name}.tensors[{tensor_index}] has invalid tensor_name")
                continue
            if tensor_name in names:
                issues.append(f"graph {graph_name!r} repeats tensor {tensor_name!r}")
            if not _is_kv_name(tensor_name):
                issues.append(
                    f"graph {graph_name!r} tensor {tensor_name!r} is not a key/value cache tensor"
                )
            if data_format != NATIVE_KV_DATA_FORMAT:
                issues.append(
                    f"graph {graph_name!r} tensor {tensor_name!r} must use "
                    f"{NATIVE_KV_DATA_FORMAT}"
                )
            names.append(tensor_name)
            tensor_count += 1
        graph_tensors[graph_name] = tuple(names)

    expected_names = set(expected_graph_names)
    expected_names.update(item.graph_name for item in expectations)
    actual_names = set(graph_tensors)
    for graph_name in sorted(expected_names - actual_names):
        issues.append(f"missing native-KV graph entry {graph_name!r}")
    for graph_name in sorted(actual_names - expected_names) if expected_names else ():
        issues.append(f"unexpected native-KV graph entry {graph_name!r}")

    for expectation in expectations:
        expected_inputs = tuple(name for name in expectation.input_names if _is_kv_name(name))
        expected_outputs = tuple(name for name in expectation.output_names if _is_kv_name(name))
        expected_tensors = (
            expected_inputs + expected_outputs
            if expectation.ar % 32 == 0
            else expected_inputs
        )
        actual = graph_tensors.get(expectation.graph_name, ())
        missing = set(expected_tensors) - set(actual)
        extra = set(actual) - set(expected_tensors)
        if missing:
            issues.append(
                f"graph {expectation.graph_name!r} is missing tensors: {sorted(missing)}"
            )
        if extra:
            issues.append(
                f"graph {expectation.graph_name!r} has unexpected tensors: {sorted(extra)}"
            )
        if not expected_tensors:
            issues.append(
                f"graph {expectation.graph_name!r} has no key/value cache tensors; "
                "native-KV must not be silently enabled"
            )

    if require_nonempty and tensor_count == 0:
        issues.append("native-KV config contains no tensors")

    return NativeKvAuditReport(
        issues=tuple(issues),
        graph_names=tuple(graph_tensors),
        tensor_count=tensor_count,
    )


def require_native_kv_audit(report: NativeKvAuditReport) -> NativeKvAuditReport:
    if not report.ok:
        raise NativeKvConfigError("native-KV config audit failed: " + "; ".join(report.issues))
    return report
