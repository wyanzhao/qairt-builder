"""Deterministic embedding/decoder/lm-head split planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class SliceKind(str, Enum):
    EMBEDDING = "embedding"
    DECODER = "decoder"
    LM_HEAD = "lm_head"


@dataclass(frozen=True)
class SliceSpec:
    """One logical model slice.

    Layer ranges are half-open: ``[layer_start, layer_end)``.  A decoder slice
    always carries one; the lm_head slice carries the single decoder layer
    ``split_llm`` folds into it.
    """

    index: int
    name: str
    kind: SliceKind
    layer_start: int | None = None
    layer_end: int | None = None

    @property
    def layer_count(self) -> int:
        if self.layer_start is None or self.layer_end is None:
            return 0
        return self.layer_end - self.layer_start


@dataclass(frozen=True)
class SplitPlan:
    """A complete, ordered split plan accepted by QAIRT ``split_llm``."""

    num_decoder_layers: int
    slices: tuple[SliceSpec, ...]
    split_embedding: bool
    split_lm_head: bool

    @property
    def num_splits(self) -> int:
        return len(self.slices)

    @property
    def decoder_slices(self) -> tuple[SliceSpec, ...]:
        return tuple(item for item in self.slices if item.kind is SliceKind.DECODER)

    @property
    def distributed_decoder_layers(self) -> int:
        """Layers spread across the decoder slices, excluding any fold."""

        return sum(item.layer_count for item in self.decoder_slices)

    @property
    def folded_lm_head_layer(self) -> int | None:
        """The decoder layer ``split_llm`` folds into the lm_head split."""

        for item in self.slices:
            if item.kind is SliceKind.LM_HEAD and item.layer_start is not None:
                return item.layer_start
        return None

    def __iter__(self) -> Iterator[SliceSpec]:
        return iter(self.slices)

    def to_qairt_kwargs(self) -> dict[str, int | bool]:
        """Return the exact public ``split_llm`` argument contract."""

        return {
            "num_splits": self.num_splits,
            "split_embedding": self.split_embedding,
            "split_lm_head": self.split_lm_head,
        }


def build_split_plan(
    num_decoder_layers: int,
    *,
    decoder_slices: int = 1,
    split_embedding: bool = True,
    split_lm_head: bool = True,
) -> SplitPlan:
    """Reproduce the decoder ranges QAIRT's ``split_llm`` produces.

    ``split_llm`` pops the final layer's post-FFN residual add before it
    distributes boundaries, so with ``split_lm_head`` the last decoder layer
    belongs to the lm_head split and only ``num_decoder_layers - 1`` layers are
    spread across the decoder slices.  The remainder is front-loaded: the first
    ``len % slices`` decoder slices each take one extra layer.

    Verified against QAIRT 2.49.0.260730
    ``qairt/optimizer/onnx/passes/splitters/llm_splitter.py``
    (``lm_head = residual_adds.pop()`` followed by the
    ``layers_per_split``/``extra_layers`` distribution loop).
    """

    if num_decoder_layers <= 0:
        raise ValueError("num_decoder_layers must be positive")
    if decoder_slices <= 0:
        raise ValueError("decoder_slices must be positive")
    distributed_layers = num_decoder_layers - (1 if split_lm_head else 0)
    if decoder_slices > distributed_layers:
        raise ValueError(
            f"decoder_slices ({decoder_slices}) cannot exceed the "
            f"{distributed_layers} layers split_llm distributes across decoder "
            f"splits (num_decoder_layers={num_decoder_layers}, "
            f"split_lm_head={split_lm_head})"
        )

    slices: list[SliceSpec] = []
    if split_embedding:
        slices.append(SliceSpec(len(slices), "embedding", SliceKind.EMBEDDING))

    base, remainder = divmod(distributed_layers, decoder_slices)
    layer_start = 0
    decoder_width = max(2, len(str(decoder_slices - 1)))
    for decoder_index in range(decoder_slices):
        layer_count = base + (1 if decoder_index < remainder else 0)
        layer_end = layer_start + layer_count
        slices.append(
            SliceSpec(
                index=len(slices),
                name=f"decoder_{decoder_index:0{decoder_width}d}",
                kind=SliceKind.DECODER,
                layer_start=layer_start,
                layer_end=layer_end,
            )
        )
        layer_start = layer_end

    if split_lm_head:
        slices.append(
            SliceSpec(
                index=len(slices),
                name="lm_head",
                kind=SliceKind.LM_HEAD,
                layer_start=distributed_layers,
                layer_end=num_decoder_layers,
            )
        )

    return SplitPlan(
        num_decoder_layers=num_decoder_layers,
        slices=tuple(slices),
        split_embedding=split_embedding,
        split_lm_head=split_lm_head,
    )
