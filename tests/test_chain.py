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
