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

    Decoder ranges are half-open: ``[layer_start, layer_end)``.
    """

    index: int
    name: str
    kind: SliceKind
    layer_start: int | None = None
    layer_end: int | None = None

    @property
    def layer_count(self) -> int:
        if self.kind is not SliceKind.DECODER:
            return 0
        assert self.layer_start is not None and self.layer_end is not None
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
    """Build balanced decoder ranges with optional edge slices."""

    if num_decoder_layers <= 0:
        raise ValueError("num_decoder_layers must be positive")
    if decoder_slices <= 0:
        raise ValueError("decoder_slices must be positive")
    if decoder_slices > num_decoder_layers:
        raise ValueError("decoder_slices cannot exceed num_decoder_layers")

    slices: list[SliceSpec] = []
    if split_embedding:
        slices.append(SliceSpec(len(slices), "embedding", SliceKind.EMBEDDING))

    base, remainder = divmod(num_decoder_layers, decoder_slices)
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
        slices.append(SliceSpec(len(slices), "lm_head", SliceKind.LM_HEAD))

    return SplitPlan(
        num_decoder_layers=num_decoder_layers,
        slices=tuple(slices),
        split_embedding=split_embedding,
        split_lm_head=split_lm_head,
    )
