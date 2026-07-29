"""Durable build-to-runtime bindings.

The build stage publishes one ``runtime_index.json`` so validation and
benchmarking never have to guess which context, graph, route, model variant, or
vector manifest belongs to an AR/CL pair. This module has no QAIRT imports.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

RUNTIME_INDEX_SCHEMA = "qairt-agent.runtime-index.v1"
SLICE_ROUTES_SCHEMA = "qairt-agent.slice-routes"


def _context_component(*, family: str, slice_name: str) -> str:
    """Classify only component boundaries that the build contract defines."""

    if family == "qwen3_vl":
        return "vision" if slice_name == "vision_projector" else "text"
    if family == "vit":
        return "vision"
    return "model"


def _execution_contract(*, lane: str, family: str) -> dict[str, Any]:
    if lane == "low_level" and family == "qwen3_vl":
        return {
            "kind": "multimodal_components",
            "automatic_end_to_end_supported": False,
            "required_components": ["vision", "text"],
            "boundary_binding": "not_executable",
            "reason": (
                "the build contains separate vision/projector and text contexts, "
                "but no audited tensor bridge or ImageT2T executor binding"
            ),
        }
    return {
        "kind": "single_runtime",
        "automatic_end_to_end_supported": True,
        "required_components": [
            "vision" if family == "vit" else "model"
        ],
        "boundary_binding": "not_applicable",
    }


def _state_slot(tensor_name: str) -> str | None:
    lowered = tensor_name.lower()
    if not any(
        token in lowered
        for token in (
            "past_key",
            "past_value",
            "present_key",
            "present_value",
            "key_cache",
            "value_cache",
            "kv_cache",
            "recurrent_state",
            "conv_state",
        )
    ):
        return None
    normalized = lowered
    for old, new in (
        ("present_key", "key"),
        ("past_key", "key"),
        ("present_value", "value"),
        ("past_value", "value"),
        ("_input", ""),
        ("_output", ""),
        ("_in", ""),
        ("_out", ""),
    ):
        normalized = normalized.replace(old, new)
    import re

    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def _path(value: Any) -> str | None:
    if value is None:
        return None
    return os.fspath(Path(value).expanduser().resolve())


def _artifact_path(value: Any) -> str:
    raw = value.get("path") if isinstance(value, Mapping) else getattr(value, "path", value)
    if raw is None:
        raise ValueError("route artifact is missing a path")
    return os.fspath(Path(raw).expanduser().resolve())


def _artifact_sha(value: Any) -> str | None:
    raw = value.get("sha256") if isinstance(value, Mapping) else getattr(value, "sha256", None)
    return str(raw) if raw is not None else None


def make_runtime_index(
    *,
    result: Any,
    lane: str,
    family: str,
    default_ar: int,
    default_context_length: int,
    route_artifacts: Sequence[Any] = (),
    validation_manifest: str | os.PathLike[str] | None = None,
    validation_manifests_by_ar: Mapping[int | str, str | os.PathLike[str]] | None = None,
    runtime_supported: bool = True,
    container_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Serialize a low-level ``BuildResult`` or GenAI container result."""

    variants: dict[str, dict[str, dict[str, Any]]] = {}
    for item in tuple(getattr(result, "variants", ())):
        cl = str(int(getattr(item, "context_length")))
        ar = str(int(getattr(item, "ar")))
        variants.setdefault(cl, {})[ar] = {
            "onnx_path": _path(getattr(item, "model_path")),
            "encodings_path": _path(getattr(item, "encodings_path", None)),
            "source_kind": str(getattr(item, "source_kind", "derived")),
        }

    transformed: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    component_models: dict[str, dict[str, Any]] = {}
    for item in tuple(getattr(result, "transformed_slices", ())):
        raw_cl = getattr(item, "context_length", None)
        raw_ar = getattr(item, "ar", None)
        slice_name = str(getattr(item, "slice_name"))
        component = _context_component(
            family=family,
            slice_name=slice_name,
        )
        if component != "model" and raw_cl is None and raw_ar is None:
            component_models[component] = {
                "slice_id": slice_name,
                "onnx_path": _path(getattr(item, "model_path")),
                "encodings_path": _path(
                    getattr(item, "encodings_path", None)
                ),
            }
        if raw_cl is None or raw_ar is None:
            continue
        cl = str(int(raw_cl))
        ar = str(int(raw_ar))
        transformed.setdefault(cl, {}).setdefault(ar, {})[slice_name] = {
            "onnx_path": _path(getattr(item, "model_path")),
            "encodings_path": _path(getattr(item, "encodings_path", None)),
            "split_index": int(getattr(item, "split_index")),
        }

    contexts: dict[str, dict[str, dict[str, Any]]] = {}
    for ordinal, item in enumerate(tuple(getattr(result, "contexts", ()))):
        raw_cl = getattr(item, "context_length", None)
        cl = str(int(default_context_length if raw_cl is None else raw_cl))
        slice_name = str(getattr(item, "slice_name", None) or f"context_{ordinal:03d}")
        ar_values = tuple(int(value) for value in getattr(item, "ar_values", ()))
        graph_names = tuple(str(value) for value in getattr(item, "graph_names", ()))
        if len(ar_values) != len(graph_names):
            raise ValueError(
                f"context {slice_name!r} has mismatched AR and graph-name counts"
            )
        contexts.setdefault(cl, {})[slice_name] = {
            "context_path": _path(getattr(item, "context_binary_path")),
            "graphs_by_ar": {str(ar): graph for ar, graph in zip(ar_values, graph_names)},
            "weight_sharing": bool(getattr(item, "weight_sharing", False)),
            "component": _context_component(
                family=family,
                slice_name=slice_name,
            ),
            "context_length_scope": (
                "independent" if raw_cl is None else "fixed"
            ),
            "native_kv_config_path": _path(
                getattr(item, "native_kv_config_path", None)
            ),
        }

    diagnostic_contexts: dict[str, list[dict[str, Any]]] = {}
    for item in tuple(getattr(result, "diagnostic_contexts", ())):
        raw_cl = getattr(item, "context_length", None)
        cl = str(int(default_context_length if raw_cl is None else raw_cl))
        diagnostic_contexts.setdefault(cl, []).append(
            {
                "slice": getattr(item, "slice_name", None),
                "context_path": _path(getattr(item, "context_binary_path")),
                "graphs_by_ar": {
                    str(ar): str(graph)
                    for ar, graph in zip(
                        getattr(item, "ar_values", ()),
                        getattr(item, "graph_names", ()),
                    )
                },
            }
        )

    routes: dict[str, dict[str, Any]] = {}
    for artifact in route_artifacts:
        path = Path(_artifact_path(artifact))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load slice route artifact {path}: {exc}") from exc
        if payload.get("schema") != SLICE_ROUTES_SCHEMA:
            raise ValueError(
                f"unexpected slice route schema in {path}: {payload.get('schema')!r}"
            )
        routes[str(int(payload["context_length"]))] = {
            "path": os.fspath(path.resolve()),
            "sha256": _artifact_sha(artifact),
            "component": str(payload.get("component", "model")),
            "coverage": str(payload.get("coverage", "full_model")),
            "excluded_components": [
                str(value)
                for value in payload.get("excluded_components", ())
            ],
        }

    vector_map = {
        str(int(ar)): _path(path)
        for ar, path in (validation_manifests_by_ar or {}).items()
    }
    raw_slices = tuple(getattr(result, "raw_slices", ()) or ())
    raw_contexts: dict[str, str] = {}
    raw_routes: list[dict[str, Any]] = []
    available_outputs: set[str] = set()
    for item in raw_slices:
        slice_id = str(getattr(item, "slice_id"))
        input_names = tuple(str(name) for name in getattr(item, "input_names"))
        output_names = tuple(str(name) for name in getattr(item, "output_names"))
        state_inputs = {
            name: slot
            for name in input_names
            if (slot := _state_slot(name)) is not None
        }
        state_outputs = {
            name: slot
            for name in output_names
            if (slot := _state_slot(name)) is not None
        }
        from_previous = {
            name: name
            for name in input_names
            if name in available_outputs and name not in state_inputs
        }
        raw_contexts[slice_id] = _path(
            getattr(item, "context_binary_path")
        ) or ""
        raw_routes.append(
            {
                "slice_id": slice_id,
                "input_names": list(input_names),
                "output_names": list(output_names),
                "graph_names": {
                    str(int(ar)): str(graph)
                    for ar, graph in dict(
                        getattr(item, "graph_names_by_ar")
                    ).items()
                },
                "from_previous": from_previous,
                "state_inputs": state_inputs,
                "state_outputs": state_outputs,
                "unresolved_external_inputs": [
                    name
                    for name in input_names
                    if name not in from_previous and name not in state_inputs
                ],
            }
        )
        available_outputs.update(output_names)
    return {
        "schema": RUNTIME_INDEX_SCHEMA,
        "lane": lane,
        "family": family,
        "runtime_supported": bool(runtime_supported),
        "execution_contract": _execution_contract(
            lane=lane,
            family=family,
        ),
        "defaults": {
            "ar": int(default_ar),
            "context_length": int(default_context_length),
        },
        "variants": variants,
        "transformed_slices": transformed,
        "component_models": component_models,
        "contexts": contexts,
        "diagnostic_contexts": diagnostic_contexts,
        "routes": routes,
        "vectors": {
            "validation_manifest": _path(validation_manifest),
            "validation_manifests_by_ar": vector_map,
            "reference_priority": "exact_ar_golden_then_onnxruntime",
        },
        "container_path": _path(container_path),
        "genai_tensor_runtime": {
            "supported": bool(
                getattr(result, "raw_tensor_runtime_supported", False)
            ),
            "notes": list(
                getattr(result, "raw_tensor_runtime_notes", ()) or ()
            ),
            "contexts": raw_contexts,
            "routes": raw_routes,
        },
    }


def load_runtime_index(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and minimally validate a published runtime index."""

    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load runtime index {resolved}: {exc}") from exc
    if payload.get("schema") != RUNTIME_INDEX_SCHEMA:
        raise ValueError(
            f"unsupported runtime index schema {payload.get('schema')!r}"
        )
    if payload.get("lane") not in {"low_level", "genai_builder"}:
        raise ValueError(f"invalid runtime lane {payload.get('lane')!r}")
    if (
        payload.get("lane") == "low_level"
        and payload.get("family") == "qwen3_vl"
    ):
        execution_contract = payload.get("execution_contract")
        if (
            not isinstance(execution_contract, Mapping)
            or execution_contract.get("kind") != "multimodal_components"
            or execution_contract.get("automatic_end_to_end_supported")
            is not False
        ):
            raise ValueError(
                "Qwen3-VL runtime index is missing its fail-closed "
                "multimodal execution contract"
            )
    defaults = payload.get("defaults")
    if not isinstance(defaults, Mapping):
        raise ValueError("runtime index is missing defaults")
    int(defaults["ar"])
    int(defaults["context_length"])
    return payload


def select_runtime_binding(
    index: Mapping[str, Any],
    *,
    ar: int | None = None,
    context_length: int | None = None,
    component: str | None = None,
) -> dict[str, Any]:
    """Resolve one exact AR/CL execution binding from an index."""

    defaults = index["defaults"]
    selected_ar = int(defaults["ar"] if ar is None else ar)
    selected_cl = int(
        defaults["context_length"] if context_length is None else context_length
    )
    cl_key = str(selected_cl)
    ar_key = str(selected_ar)
    execution_contract = dict(index.get("execution_contract") or {})
    requested_component = (
        str(component).strip().lower() if component is not None else None
    )
    is_qwen3_vl_components = (
        index.get("lane") == "low_level"
        and index.get("family") == "qwen3_vl"
    )
    if is_qwen3_vl_components:
        if (
            execution_contract.get("kind") != "multimodal_components"
            or execution_contract.get("automatic_end_to_end_supported")
            is not False
        ):
            raise ValueError(
                "Qwen3-VL runtime index is missing its fail-closed "
                "multimodal execution contract"
            )
        if requested_component is None:
            raise ValueError(
                "Qwen3-VL automatic end-to-end execution is unavailable: "
                "the runtime index contains separate vision/projector and text "
                "contexts but no audited boundary bridge. Select component='text' "
                "for an explicitly text-only run, or provide an explicit "
                "vision graph binding and vision-only vectors."
            )
        if requested_component not in {"text", "vision"}:
            raise ValueError(
                "Qwen3-VL runtime component must be 'text' or 'vision'; "
                "end-to-end execution is not supported"
            )
    vectors = dict(index.get("vectors") or {})
    exact_vectors = dict(vectors.get("validation_manifests_by_ar") or {})
    vector_manifest = exact_vectors.get(ar_key) or vectors.get("validation_manifest")

    binding: dict[str, Any] = {
        "lane": index["lane"],
        "family": index["family"],
        "runtime_supported": bool(index.get("runtime_supported", False)),
        "ar": selected_ar,
        "context_length": selected_cl,
        "vector_manifest": vector_manifest,
        "reference_model_path": (
            ((index.get("variants") or {}).get(cl_key) or {}).get(ar_key) or {}
        ).get("onnx_path"),
        "container_path": index.get("container_path"),
    }
    if requested_component is not None:
        binding["component"] = requested_component
    if index["lane"] == "genai_builder":
        tensor_runtime = dict(index.get("genai_tensor_runtime") or {})
        if tensor_runtime.get("supported"):
            routes = list(tensor_runtime.get("routes") or ())
            contexts = dict(tensor_runtime.get("contexts") or {})
            if not routes or not contexts:
                raise ValueError(
                    "GenAI tensor runtime is marked supported without routes/contexts"
                )
            for route in routes:
                if ar_key not in dict(route.get("graph_names") or {}):
                    raise ValueError(
                        f"GenAI slice {route.get('slice_id')!r} has no "
                        f"AR{selected_ar} graph"
                    )
            binding["tensor_runtime"] = {
                "scope": "chain" if len(routes) > 1 else "graph",
                "routes": routes,
                "contexts": contexts,
                "notes": list(tensor_runtime.get("notes") or ()),
            }
        return binding

    if requested_component == "vision":
        contexts = dict((index.get("contexts") or {}).get(cl_key) or {})
        candidates = [
            (slice_name, value)
            for slice_name, value in contexts.items()
            if value.get("component") == "vision"
            and dict(value.get("graphs_by_ar") or {})
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Qwen3-VL runtime index cannot select one vision component; "
                f"candidates={[name for name, _ in candidates]}"
            )
        slice_name, context = candidates[0]
        graphs = dict(context["graphs_by_ar"])
        if len(graphs) != 1:
            raise ValueError(
                "Qwen3-VL vision component must expose exactly one "
                f"AR-independent graph, found {sorted(graphs)}"
            )
        graph_ar, graph_name = next(iter(graphs.items()))
        vision_model = dict(
            (index.get("component_models") or {}).get("vision") or {}
        )
        if not vision_model.get("onnx_path"):
            raise ValueError(
                "Qwen3-VL runtime index has a vision context but no "
                "content-addressed vision ONNX reference"
            )
        binding.update(
            {
                "scope": "graph",
                "coverage": "vision_only",
                "context_path": context["context_path"],
                "graph_name": graph_name,
                "graph_ar": int(graph_ar),
                "slice_id": slice_name,
                "reference_model_path": vision_model["onnx_path"],
                # Per-AR manifests target the text graph. Reusing one for the
                # vision graph would pass unrelated token/KV inputs.
                "vector_manifest": None,
            }
        )
        return binding

    route_entry = (index.get("routes") or {}).get(cl_key)
    if route_entry is not None:
        if (
            requested_component == "text"
            and route_entry.get("component") != "text"
        ):
            raise ValueError(
                "Qwen3-VL runtime index has no reviewed text-component route "
                f"for CL{selected_cl}"
            )
        route_path = Path(route_entry["path"]).expanduser().resolve()
        route_payload = json.loads(route_path.read_text(encoding="utf-8"))
        if route_payload.get("schema") != SLICE_ROUTES_SCHEMA:
            raise ValueError(f"invalid slice route schema in {route_path}")
        routes = list(route_payload.get("routes") or ())
        contexts = dict(route_payload.get("contexts") or {})
        if not routes or not contexts:
            raise ValueError(
                f"slice route artifact {route_path} has no executable routes"
            )
        for route in routes:
            graph = (route.get("graph_names") or {}).get(ar_key)
            if not graph:
                raise ValueError(
                    f"slice {route.get('slice_id')!r} has no AR{selected_ar} graph"
                )
        route_binding = {
            "component": route_entry.get("component", "model"),
            "coverage": route_entry.get("coverage", "full_model"),
            "excluded_components": list(
                route_entry.get("excluded_components") or ()
            ),
        }
        if len(routes) > 1:
            binding.update(
                {
                    "scope": "chain",
                    "routes": routes,
                    "contexts": contexts,
                    "route_manifest": os.fspath(route_path),
                    **route_binding,
                }
            )
        else:
            route = routes[0]
            slice_id = str(route["slice_id"])
            binding.update(
                {
                    "scope": "graph",
                    "route_manifest": os.fspath(route_path),
                    "context_path": contexts[slice_id],
                    "graph_name": route["graph_names"][ar_key],
                    **route_binding,
                }
            )
        return binding

    contexts = dict((index.get("contexts") or {}).get(cl_key) or {})
    candidates = [
        (slice_name, value)
        for slice_name, value in contexts.items()
        if ar_key in dict(value.get("graphs_by_ar") or {})
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"runtime index cannot select one graph for CL{selected_cl}/AR{selected_ar}; "
            f"candidates={[name for name, _ in candidates]}"
        )
    slice_name, context = candidates[0]
    binding.update(
        {
            "scope": "graph",
            "context_path": context["context_path"],
            "graph_name": context["graphs_by_ar"][ar_key],
            "slice_id": slice_name,
        }
    )
    return binding


__all__ = [
    "RUNTIME_INDEX_SCHEMA",
    "load_runtime_index",
    "make_runtime_index",
    "select_runtime_binding",
]
