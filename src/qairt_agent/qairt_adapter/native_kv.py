"""Native-KV data-format policy and strict auditing.

The selection rule -- which tensors get the HMX weight layout, and whether a
graph's outputs are included -- belongs to QAIRT, not to this program. It is
read from the SDK's own ``gen_ai_utils.gen_kv_format_config`` at build time
(see ``QairtSdkAdapter._native_kv_config``) rather than reimplemented here: a
reimplementation goes stale the moment the SDK changes it, and nothing says so.
This module previously carried that copy, and it had already drifted -- it was
missing the SDK's ``ar > 0`` guard.

What stays here is the part that is *ours*: a documented subtraction of tensor
names whose role proves they are not caches, and a strict audit that fails
closed when the config the compiler is about to receive disagrees with what the
SDK selected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import NativeKvConfigError
from .types import NativeKvAuditReport, NativeKvGraphExpectation


NATIVE_KV_DATA_FORMAT = "QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT"

# QAIRT selects cache tensors with a bare ``"key" in name or "value" in name``
# substring test. We subtract names whose role vocabulary proves they are not
# caches: marking a padding mask or a position index as an HMX weight layout
# would silently corrupt it, and linear-attention recurrent/conv state is not a
# KV cache at all. This is a deliberate, narrow divergence from the SDK, and
# ``normalize_sdk_kv_config`` reports what it removed rather than hiding it.
_NON_CACHE_ROLE_TOKENS = (
    "recurrent_state",
    "conv_state",
    "mask",
    "padding",
    "position",
    "index",
    "length",
    "scale",
    "offset",
)


def is_non_cache_role(name: str) -> bool:
    """Whether a name QAIRT selected is provably not a KV cache tensor."""

    normalized = name.lower()
    return any(token in normalized for token in _NON_CACHE_ROLE_TOKENS)


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


def normalize_sdk_kv_config(
    sdk_config: Mapping[str, Any],
    expectations: Sequence[NativeKvGraphExpectation] = (),
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    """Adopt QAIRT's selection, renamed to our graph names and role-filtered.

    QAIRT names each graph by its ONNX stem; this program addresses graphs by
    the converted model's graph name, so entries are remapped through the
    expectation that produced them. Returns the config plus every name the role
    filter removed, so the divergence from the SDK is reported, not silent.
    """

    by_stem = {
        Path(str(item.model_path)).stem: item
        for item in expectations
        if item.model_path is not None
    }
    graphs: list[dict[str, Any]] = []
    removed: list[dict[str, str]] = []
    for raw_graph in sdk_config.get("graphs", ()):
        if not isinstance(raw_graph, Mapping):
            raise NativeKvConfigError("QAIRT returned a non-object graph entry")
        stem = str(raw_graph.get("graph_name", ""))
        expectation = by_stem.get(stem)
        tensors: list[dict[str, Any]] = []
        for raw_tensor in raw_graph.get("tensors", ()):
            name = str(raw_tensor.get("tensor_name", ""))
            if is_non_cache_role(name):
                removed.append({"graph_name": stem, "tensor_name": name})
                continue
            tensors.append(
                {
                    "tensor_name": name,
                    "dataFormat": str(
                        raw_tensor.get("dataFormat", NATIVE_KV_DATA_FORMAT)
                    ),
                }
            )
        if tensors:
            graphs.append(
                {
                    "graph_name": (
                        expectation.graph_name if expectation is not None else stem
                    ),
                    "tensors": tensors,
                }
            )
    return {"graphs": graphs}, removed


def expected_tensors(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Per-graph tensor names, used as the audit's expectation."""

    expected: dict[str, tuple[str, ...]] = {}
    for raw_graph in config.get("graphs", ()):
        if not isinstance(raw_graph, Mapping):
            continue
        expected[str(raw_graph.get("graph_name", ""))] = tuple(
            str(item.get("tensor_name", ""))
            for item in raw_graph.get("tensors", ())
            if isinstance(item, Mapping)
        )
    return expected


def audit_native_kv_config(
    config: str | Path | Mapping[str, Any],
    *,
    expected: Mapping[str, Sequence[str]] | None = None,
    expected_graph_names: Sequence[str] = (),
    require_nonempty: bool = True,
) -> NativeKvAuditReport:
    """Audit exact graph names, tensor sets, and HMX layout values.

    ``expected`` is what QAIRT selected for each graph after the role filter.
    Auditing against that, rather than against a locally recomputed rule, keeps
    this a check on the config the compiler receives instead of a second copy of
    the selection logic.
    """

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
            if is_non_cache_role(tensor_name):
                issues.append(
                    f"graph {graph_name!r} tensor {tensor_name!r} is not a key/value "
                    "cache tensor"
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
    if expected is not None:
        expected_names.update(expected)
    actual_names = set(graph_tensors)
    for graph_name in sorted(expected_names - actual_names):
        issues.append(f"missing native-KV graph entry {graph_name!r}")
    for graph_name in sorted(actual_names - expected_names) if expected_names else ():
        issues.append(f"unexpected native-KV graph entry {graph_name!r}")

    for graph_name, tensors in (expected or {}).items():
        actual = graph_tensors.get(graph_name, ())
        missing = set(tensors) - set(actual)
        extra = set(actual) - set(tensors)
        if missing:
            issues.append(
                f"graph {graph_name!r} is missing tensors: {sorted(missing)}"
            )
        if extra:
            issues.append(
                f"graph {graph_name!r} has unexpected tensors: {sorted(extra)}"
            )
        if not tensors:
            issues.append(
                f"graph {graph_name!r} has no key/value cache tensors; "
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
