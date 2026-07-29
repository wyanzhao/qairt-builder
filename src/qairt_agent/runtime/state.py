"""Shared native-state tensor naming."""

from __future__ import annotations

import re


def state_slot(tensor_name: str) -> str | None:
    """Map compatible KV/recurrent input and output names to one slot."""

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
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


__all__ = ["state_slot"]
