"""Stateless, slice-aware execution and explicit tensor routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


class ChainExecutionError(RuntimeError):
    """Raised when an explicit slice route cannot be executed."""


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


@dataclass(frozen=True)
class SliceInvocation:
    """The complete routing request passed to a slice executor."""

    slice_id: str
    ar: int
    graph_name: str
    output_names: tuple[str, ...]
    mode: str
    step_index: int


class SliceExecutor(Protocol):
    def __call__(
        self, inputs: Mapping[str, np.ndarray], invocation: SliceInvocation
    ) -> Mapping[str, np.ndarray]: ...


@dataclass(frozen=True)
class SliceRoute:
    """Explicit IO, state, and AR-graph routing for one model slice.

    ``from_previous`` maps this slice's input name to an upstream output name.
    ``state_inputs`` maps this slice's input name to an invocation-local state
    slot. ``state_outputs`` maps an output name back to such a state slot.
    """

    slice_id: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    graph_names: Mapping[int, str]
    from_previous: Mapping[str, str] = field(default_factory=dict)
    state_inputs: Mapping[str, str] = field(default_factory=dict)
    state_outputs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.slice_id:
            raise ValueError("slice_id cannot be empty")
        if not self.input_names:
            raise ValueError(f"Slice {self.slice_id!r} has no declared inputs")
        if not self.output_names:
            raise ValueError(f"Slice {self.slice_id!r} has no declared outputs")

        graphs = {int(ar): str(name) for ar, name in self.graph_names.items()}
        if not graphs or any(not name for name in graphs.values()):
            raise ValueError(f"Slice {self.slice_id!r} requires explicit, non-empty graph names")
        if set(self.from_previous) - set(self.input_names):
            raise ValueError(f"Slice {self.slice_id!r} has from_previous entries for undeclared inputs")
        if set(self.state_inputs) - set(self.input_names):
            raise ValueError(f"Slice {self.slice_id!r} has state entries for undeclared inputs")
        if set(self.state_outputs) - set(self.output_names):
            raise ValueError(f"Slice {self.slice_id!r} has state entries for undeclared outputs")
        overlap = set(self.from_previous) & set(self.state_inputs)
        if overlap:
            raise ValueError(
                f"Slice {self.slice_id!r} inputs cannot be both boundaries "
                f"and native state: {sorted(overlap)}"
            )

        object.__setattr__(self, "input_names", tuple(str(name) for name in self.input_names))
        object.__setattr__(self, "output_names", tuple(str(name) for name in self.output_names))
        object.__setattr__(self, "graph_names", graphs)
        object.__setattr__(
            self, "from_previous", {str(name): str(source) for name, source in self.from_previous.items()}
        )
        object.__setattr__(
            self, "state_inputs", {str(name): str(slot) for name, slot in self.state_inputs.items()}
        )
        object.__setattr__(
            self, "state_outputs", {str(name): str(slot) for name, slot in self.state_outputs.items()}
        )

    @classmethod
    def from_object(cls, value: Any) -> "SliceRoute":
        """Coerce a mapping or contract-like object without importing contracts."""

        if isinstance(value, cls):
            return value
        slice_id = _field(value, "slice_id", "id", "name")
        input_names = _field(value, "input_names", "inputs", default=())
        output_names = _field(value, "output_names", "outputs", default=())
        graph_names = _field(value, "graph_names", "graphs_by_ar", default={})
        return cls(
            slice_id=str(slice_id or ""),
            input_names=tuple(str(name) for name in input_names),
            output_names=tuple(str(name) for name in output_names),
            graph_names={int(ar): str(name) for ar, name in dict(graph_names).items()},
            from_previous=dict(
                _field(value, "from_previous", "input_from_previous", "boundary_inputs", default={})
            ),
            state_inputs=dict(_field(value, "state_inputs", "native_state_inputs", default={})),
            state_outputs=dict(_field(value, "state_outputs", "native_state_outputs", default={})),
        )

    def graph_for_ar(self, ar: int) -> str:
        try:
            return self.graph_names[int(ar)]
        except KeyError as error:
            raise ChainExecutionError(
                f"Slice {self.slice_id!r} has no explicit graph for AR{ar}; "
                f"available ARs: {sorted(self.graph_names)}"
            ) from error


@dataclass(frozen=True)
class SliceExecution:
    slice_id: str
    graph_name: str
    outputs: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ChainResult:
    """Outputs from one chain step with no separate hidden-state snapshot.

    State-designated graph outputs remain visible for quality diagnostics, but
    the runner never retains them after the enclosing public call.
    """

    mode: str
    ar: int
    step_index: int
    slices: tuple[SliceExecution, ...]
    final_outputs: Mapping[str, np.ndarray]
    native_state_slots: tuple[str, ...]

    def outputs_by_slice(self) -> dict[str, Mapping[str, np.ndarray]]:
        return {item.slice_id: item.outputs for item in self.slices}


@dataclass(frozen=True)
class SequenceStep:
    inputs: Mapping[str, np.ndarray]
    ar: int
    teacher_inputs: Mapping[str, Mapping[str, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def from_object(cls, value: Any) -> "SequenceStep":
        if isinstance(value, cls):
            return value
        return cls(
            inputs=dict(_field(value, "inputs", default={})),
            ar=int(_field(value, "ar")),
            teacher_inputs=dict(_field(value, "teacher_inputs", "golden_inputs", default={})),
        )


@dataclass(frozen=True)
class ChainSequenceResult:
    """A full prefill/decode sequence completed within one method call."""

    mode: str
    steps: tuple[ChainResult, ...]
    final_outputs: Mapping[str, np.ndarray]


class SliceChainRunner:
    """Run explicit slice routes without retaining tensor or native-KV state.

    Executors are immutable dependencies; all tensor state is local to
    ``run_device_chain``, ``run_teacher_forced``, or ``run_sequence``.
    """

    def __init__(
        self,
        routes: Sequence[SliceRoute | Mapping[str, Any] | Any],
        executors: Mapping[str, SliceExecutor | Any],
    ) -> None:
        self.routes = tuple(SliceRoute.from_object(route) for route in routes)
        if not self.routes:
            raise ValueError("At least one slice route is required")
        slice_ids = [route.slice_id for route in self.routes]
        if len(slice_ids) != len(set(slice_ids)):
            raise ValueError(f"Duplicate slice IDs are not allowed: {slice_ids}")

        self.executors = dict(executors)
        missing = set(slice_ids) - set(self.executors)
        if missing:
            raise ValueError(f"Missing executors for slices: {sorted(missing)}")

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any] | Any,
        executors: Mapping[str, SliceExecutor | Any],
    ) -> "SliceChainRunner":
        routes = _field(manifest, "slice_routes", "slices", default=())
        if not routes:
            metadata = _field(manifest, "metadata", default={})
            routes = _field(metadata, "slice_routes", "slices", default=())
        return cls(routes, executors)

    @staticmethod
    def _as_array_mapping(value: Mapping[str, Any], *, context: str) -> dict[str, np.ndarray]:
        if not isinstance(value, Mapping):
            raise ChainExecutionError(f"{context} must be a tensor mapping")
        return {str(name): np.asarray(tensor) for name, tensor in value.items()}

    def _invoke(
        self,
        route: SliceRoute,
        inputs: Mapping[str, np.ndarray],
        invocation: SliceInvocation,
    ) -> dict[str, np.ndarray]:
        executor = self.executors[route.slice_id]
        if hasattr(executor, "execute_slice"):
            raw_outputs = executor.execute_slice(inputs, invocation)
        elif callable(executor):
            raw_outputs = executor(inputs, invocation)
        else:
            raise ChainExecutionError(
                f"Executor for slice {route.slice_id!r} is neither callable nor execute_slice-capable"
            )
        outputs = self._as_array_mapping(raw_outputs, context=f"Outputs of slice {route.slice_id!r}")
        missing = set(route.output_names) - set(outputs)
        if missing:
            raise ChainExecutionError(
                f"Slice {route.slice_id!r} did not produce declared outputs: {sorted(missing)}"
            )
        return {name: outputs[name] for name in route.output_names}

    @staticmethod
    def _copy_tensor_map(value: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
        return {str(name): np.asarray(tensor) for name, tensor in (value or {}).items()}

    def _run_once(
        self,
        initial_inputs: Mapping[str, Any],
        *,
        ar: int,
        mode: str,
        teacher_inputs: Mapping[str, Mapping[str, Any]] | None,
        native_state: dict[str, np.ndarray],
        step_index: int,
    ) -> tuple[ChainResult, dict[str, np.ndarray]]:
        if mode not in {"device_chain", "teacher_forced"}:
            raise ValueError("mode must be 'device_chain' or 'teacher_forced'")

        available = self._copy_tensor_map(initial_inputs)
        teacher_by_slice = {
            str(slice_id): self._copy_tensor_map(tensors)
            for slice_id, tensors in (teacher_inputs or {}).items()
        }
        executions: list[SliceExecution] = []

        for route in self.routes:
            forced = teacher_by_slice.get(route.slice_id, {})
            slice_inputs: dict[str, np.ndarray] = {}
            for input_name in route.input_names:
                if mode == "teacher_forced" and input_name in forced:
                    slice_inputs[input_name] = forced[input_name]
                    continue

                if input_name in route.state_inputs:
                    if mode == "teacher_forced":
                        raise ChainExecutionError(
                            f"Teacher-forced native-state input "
                            f"{route.slice_id}.{input_name} has no explicit golden"
                        )
                    state_slot = route.state_inputs[input_name]
                    try:
                        slice_inputs[input_name] = native_state[state_slot]
                    except KeyError as error:
                        raise ChainExecutionError(
                            f"Slice {route.slice_id!r} requires missing native state slot {state_slot!r}"
                        ) from error
                    continue

                if input_name in route.from_previous:
                    if mode == "teacher_forced":
                        raise ChainExecutionError(
                            f"Teacher-forced input {route.slice_id}.{input_name} has no explicit golden"
                        )
                    source_name = route.from_previous[input_name]
                    try:
                        slice_inputs[input_name] = available[source_name]
                    except KeyError as error:
                        raise ChainExecutionError(
                            f"Slice {route.slice_id!r} boundary input {input_name!r} "
                            f"cannot resolve upstream output {source_name!r}"
                        ) from error
                    continue

                try:
                    slice_inputs[input_name] = available[input_name]
                except KeyError as error:
                    raise ChainExecutionError(
                        f"Slice {route.slice_id!r} requires missing external input {input_name!r}"
                    ) from error

            graph_name = route.graph_for_ar(ar)
            invocation = SliceInvocation(
                slice_id=route.slice_id,
                ar=int(ar),
                graph_name=graph_name,
                output_names=route.output_names,
                mode=mode,
                step_index=step_index,
            )
            outputs = self._invoke(route, slice_inputs, invocation)
            available.update(outputs)
            available.update(
                {f"{route.slice_id}.{output_name}": tensor for output_name, tensor in outputs.items()}
            )
            for output_name, state_slot in route.state_outputs.items():
                native_state[state_slot] = outputs[output_name]
            executions.append(
                SliceExecution(slice_id=route.slice_id, graph_name=graph_name, outputs=outputs)
            )

        final_outputs = executions[-1].outputs
        result = ChainResult(
            mode=mode,
            ar=int(ar),
            step_index=step_index,
            slices=tuple(executions),
            final_outputs=final_outputs,
            native_state_slots=tuple(sorted(native_state)),
        )
        return result, native_state

    def run_device_chain(
        self,
        initial_inputs: Mapping[str, Any],
        *,
        ar: int,
        initial_native_state: Mapping[str, Any] | None = None,
    ) -> ChainResult:
        """Run one explicit device-output-to-next-slice chain."""

        result, _ = self._run_once(
            initial_inputs,
            ar=ar,
            mode="device_chain",
            teacher_inputs=None,
            native_state=self._copy_tensor_map(initial_native_state),
            step_index=0,
        )
        return result

    def run_teacher_forced(
        self,
        initial_inputs: Mapping[str, Any],
        teacher_inputs: Mapping[str, Mapping[str, Any]],
        *,
        ar: int,
        initial_native_state: Mapping[str, Any] | None = None,
    ) -> ChainResult:
        """Run slices with explicit golden boundary inputs for local attribution."""

        result, _ = self._run_once(
            initial_inputs,
            ar=ar,
            mode="teacher_forced",
            teacher_inputs=teacher_inputs,
            native_state=self._copy_tensor_map(initial_native_state),
            step_index=0,
        )
        return result

    def run_sequence(
        self,
        steps: Sequence[SequenceStep | Mapping[str, Any] | Any],
        *,
        mode: str = "device_chain",
        initial_native_state: Mapping[str, Any] | None = None,
    ) -> ChainSequenceResult:
        """Run a prefill/decode sequence while keeping native state call-local."""

        normalized_steps = tuple(SequenceStep.from_object(step) for step in steps)
        if not normalized_steps:
            raise ValueError("At least one sequence step is required")

        native_state = self._copy_tensor_map(initial_native_state)
        results: list[ChainResult] = []
        for index, step in enumerate(normalized_steps):
            result, native_state = self._run_once(
                step.inputs,
                ar=step.ar,
                mode=mode,
                teacher_inputs=step.teacher_inputs if mode == "teacher_forced" else None,
                native_state=native_state,
                step_index=index,
            )
            results.append(result)

        # native_state intentionally falls out of scope here and is never kept on self.
        return ChainSequenceResult(
            mode=mode,
            steps=tuple(results),
            final_outputs=results[-1].final_outputs,
        )


__all__ = [
    "ChainExecutionError",
    "ChainResult",
    "ChainSequenceResult",
    "SequenceStep",
    "SliceChainRunner",
    "SliceExecution",
    "SliceInvocation",
    "SliceRoute",
]
