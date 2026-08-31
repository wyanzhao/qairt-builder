from __future__ import annotations

from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pytest

from qairt_agent.runtime.chain import (
    ChainExecutionError,
    SequenceStep,
    SliceChainRunner,
    SliceInvocation,
    SliceRoute,
)


def _three_slice_runner() -> SliceChainRunner:
    routes = [
        SliceRoute(
            slice_id="embedding",
            input_names=("tokens",),
            output_names=("embedding_out",),
            graph_names={1: "embedding_ar1", 128: "embedding_ar128"},
        ),
        SliceRoute(
            slice_id="decoder",
            input_names=("hidden_in",),
            output_names=("decoder_out",),
            graph_names={1: "decoder_ar1", 128: "decoder_ar128"},
            from_previous={"hidden_in": "embedding_out"},
        ),
        SliceRoute(
            slice_id="lm_head",
            input_names=("hidden_in",),
            output_names=("logits",),
            graph_names={1: "lm_head_ar1", 128: "lm_head_ar128"},
            from_previous={"hidden_in": "decoder_out"},
        ),
    ]

    def embedding(
        inputs: Mapping[str, np.ndarray], invocation: SliceInvocation
    ) -> Mapping[str, np.ndarray]:
        assert invocation.graph_name == f"embedding_ar{invocation.ar}"
        return {"embedding_out": inputs["tokens"] + 1}

    def decoder(
        inputs: Mapping[str, np.ndarray], invocation: SliceInvocation
    ) -> Mapping[str, np.ndarray]:
        return {"decoder_out": inputs["hidden_in"] * 2}

    def lm_head(
        inputs: Mapping[str, np.ndarray], invocation: SliceInvocation
    ) -> Mapping[str, np.ndarray]:
        return {"logits": inputs["hidden_in"] - 3}

    return SliceChainRunner(
        routes,
        {"embedding": embedding, "decoder": decoder, "lm_head": lm_head},
    )


def test_device_chain_routes_actual_slice_outputs() -> None:
    runner = _three_slice_runner()
    result = runner.run_device_chain({"tokens": np.array([2.0])}, ar=1)

    np.testing.assert_array_equal(result.final_outputs["logits"], np.array([3.0]))
    assert [item.graph_name for item in result.slices] == [
        "embedding_ar1",
        "decoder_ar1",
        "lm_head_ar1",
    ]


def test_teacher_forced_uses_explicit_boundary_goldens() -> None:
    runner = _three_slice_runner()
    result = runner.run_teacher_forced(
        {"tokens": np.array([2.0])},
        {
            "decoder": {"hidden_in": np.array([10.0])},
            "lm_head": {"hidden_in": np.array([100.0])},
        },
        ar=128,
    )

    np.testing.assert_array_equal(
        result.outputs_by_slice()["decoder"]["decoder_out"],
        np.array([20.0]),
    )
    np.testing.assert_array_equal(result.final_outputs["logits"], np.array([97.0]))
    assert result.slices[-1].graph_name == "lm_head_ar128"


def test_teacher_forced_rejects_implicit_device_boundary() -> None:
    runner = _three_slice_runner()
    with pytest.raises(ChainExecutionError, match="has no explicit golden"):
        runner.run_teacher_forced(
            {"tokens": np.array([2.0])},
            {"decoder": {"hidden_in": np.array([10.0])}},
            ar=1,
        )


def test_native_state_is_sequence_local_and_does_not_leak() -> None:
    route = SliceRoute(
        slice_id="decoder",
        input_names=("token", "kv_in"),
        output_names=("logits", "kv_out"),
        graph_names={1: "decode", 128: "prefill"},
        state_inputs={"kv_in": "decoder.kv"},
        state_outputs={"kv_out": "decoder.kv"},
    )

    def decoder(
        inputs: Mapping[str, np.ndarray], invocation: SliceInvocation
    ) -> Mapping[str, np.ndarray]:
        next_state = inputs["kv_in"] + inputs["token"]
        return {"logits": next_state.copy(), "kv_out": next_state}

    runner = SliceChainRunner([route], {"decoder": decoder})
    steps = [
        SequenceStep(inputs={"token": np.array([2.0])}, ar=128),
        SequenceStep(inputs={"token": np.array([3.0])}, ar=1),
    ]

    first = runner.run_sequence(
        steps,
        initial_native_state={"decoder.kv": np.array([0.0])},
    )
    second = runner.run_sequence(
        steps,
        initial_native_state={"decoder.kv": np.array([0.0])},
    )

    np.testing.assert_array_equal(first.final_outputs["logits"], np.array([5.0]))
    np.testing.assert_array_equal(second.final_outputs["logits"], np.array([5.0]))
    assert "native_state" not in vars(runner)


def test_missing_ar_graph_fails_closed() -> None:
    runner = _three_slice_runner()
    with pytest.raises(ChainExecutionError, match="no explicit graph for AR32"):
        runner.run_device_chain({"tokens": np.array([1.0])}, ar=32)


def test_teacher_forced_native_state_requires_explicit_golden() -> None:
    route = SliceRoute(
        slice_id="decoder",
        input_names=("token", "kv_in"),
        output_names=("logits", "kv_out"),
        graph_names={1: "decode"},
        state_inputs={"kv_in": "decoder.kv"},
        state_outputs={"kv_out": "decoder.kv"},
    )

    def decoder(
        inputs: Mapping[str, np.ndarray], invocation: SliceInvocation
    ) -> Mapping[str, np.ndarray]:
        return {"logits": inputs["token"], "kv_out": inputs["kv_in"]}

    runner = SliceChainRunner([route], {"decoder": decoder})
    with pytest.raises(ChainExecutionError, match="native-state input"):
        runner.run_teacher_forced(
            {"token": np.array([1.0])},
            {"decoder": {}},
            ar=1,
            initial_native_state={"decoder.kv": np.array([0.0])},
        )


def test_runner_can_read_routes_from_manifest_metadata() -> None:
    route = {
        "slice_id": "head",
        "input_names": ["x"],
        "output_names": ["y"],
        "graph_names": {"1": "head_ar1"},
    }

    def head(
        inputs: Mapping[str, np.ndarray], invocation: SliceInvocation
    ) -> Mapping[str, np.ndarray]:
        return {"y": inputs["x"] + 1}

    manifest = SimpleNamespace(metadata={"slice_routes": [route]})
    runner = SliceChainRunner.from_manifest(manifest, {"head": head})
    result = runner.run_device_chain({"x": np.array([1.0])}, ar=1)
    np.testing.assert_array_equal(result.final_outputs["y"], np.array([2.0]))


# --------------------------------------------------------------------------- #
# Chain-step device provenance (T16)
# --------------------------------------------------------------------------- #


class _CountingAdapter:
    """Records every profiled execute so coverage can be asserted exactly."""

    def __init__(self) -> None:
        self.captures: list[tuple[str, float]] = []

    def capture_device_execution(
        self, compiled, inputs, *, graph_name, device, native_io, working_dir, **_
    ):
        marker = float(np.asarray(next(iter(inputs.values()))).ravel()[0])
        self.captures.append((graph_name, marker))
        return {
            "schema": "qairt-agent.device-execution/2",
            "accelerator_compute_us": 10.0 + marker,
            "accelerator_execute_us": 100.0,
            "qnn_execute_us": 200.0,
        }


def _record(step_inputs: list[tuple[str, str, float]]) -> dict:
    """Build the recorded structure one chain pass would have produced."""

    recorded: dict[str, list[dict]] = {}
    for slice_name, graph_name, marker in step_inputs:
        entries = recorded.setdefault(slice_name, [])
        entries.append(
            {
                "step_index": len(entries),
                "inputs": {"x": np.array([marker], dtype=np.float32)},
                "graph_name": graph_name,
                "ar": 1 if graph_name.endswith("ar1") else 128,
            }
        )
    return recorded


def test_a_two_step_chain_publishes_per_step_device_evidence(tmp_path) -> None:
    """Keeping only the last step made a prefill+decode run look like a chain.

    The block used to carry ``scope="chain"`` over decode-only evidence; a
    reader had no way to see that prefill was never measured.
    """

    from qairt_agent.pipeline import QairtAgent

    adapter = _CountingAdapter()
    recorded = _record(
        [
            ("embedding", "embedding_ar128", 1.0),
            ("decoder", "decoder_ar128", 2.0),
            ("embedding", "embedding_ar1", 3.0),
            ("decoder", "decoder_ar1", 4.0),
        ]
    )

    block = QairtAgent._chain_device_execution(
        QairtAgent.__new__(QairtAgent),
        adapter,
        recorded,
        {"embedding": object(), "decoder": object()},
        device=object(),
        native_io=False,
        execution_options={},
        working_dir=tmp_path,
    )

    assert block["scope"] == "chain_sequence"
    assert block["steps_total"] == 2
    assert block["steps_covered"] == 2
    assert [step["step_index"] for step in block["by_step"]] == [0, 1]
    # Both steps were profiled with their own inputs and their own AR graph.
    assert block["by_step"][0]["by_slice"]["embedding"]["ar"] == 128
    assert block["by_step"][1]["by_slice"]["embedding"]["ar"] == 1
    # Every recorded invocation was profiled with its own recorded inputs.
    assert sorted({marker for _, marker in adapter.captures}) == [1.0, 2.0, 3.0, 4.0]
    assert sorted({graph for graph, _ in adapter.captures}) == [
        "decoder_ar1",
        "decoder_ar128",
        "embedding_ar1",
        "embedding_ar128",
    ]
    # No unqualified per-slice collapse: a sequence publishes steps.
    assert "by_slice" not in block


def test_a_single_pass_chain_keeps_the_documented_per_slice_shape(tmp_path) -> None:
    from qairt_agent.pipeline import QairtAgent

    adapter = _CountingAdapter()
    recorded = _record(
        [("embedding", "embedding_ar1", 1.0), ("decoder", "decoder_ar1", 2.0)]
    )

    block = QairtAgent._chain_device_execution(
        QairtAgent.__new__(QairtAgent),
        adapter,
        recorded,
        {"embedding": object(), "decoder": object()},
        device=object(),
        native_io=False,
        execution_options={},
        working_dir=tmp_path,
    )

    assert block["scope"] == "chain"
    assert block["steps_total"] == 1
    assert set(block["by_slice"]) == {"embedding", "decoder"}
    assert block["totals_basis"] == (
        "sum_of_per_slice_means_slices_run_sequentially"
    )


def test_the_recorder_keeps_every_invocation_not_just_the_last() -> None:
    from qairt_agent.pipeline import QairtAgent

    recorded: dict[str, list[dict]] = {}
    calls: list[str] = []

    def executor(inputs, invocation):
        calls.append(invocation.graph_name)
        return {"out": np.array([1.0], dtype=np.float32)}

    wrapped = QairtAgent._recording_chain_executors({"decoder": executor}, recorded)
    for graph in ("decoder_ar128", "decoder_ar1"):
        wrapped["decoder"](
            {"x": np.array([1.0], dtype=np.float32)},
            SimpleNamespace(graph_name=graph, ar=1 if graph.endswith("ar1") else 128),
        )

    assert [entry["graph_name"] for entry in recorded["decoder"]] == [
        "decoder_ar128",
        "decoder_ar1",
    ]
    assert [entry["step_index"] for entry in recorded["decoder"]] == [0, 1]
