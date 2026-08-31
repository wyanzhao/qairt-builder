"""The one place a model family is declared.

Family identity used to live in four hand-synced registries: ``ModelFamily``
plus ``_FAMILY_TO_PRESET`` in ``contracts``, ``FamilyId``/``FamilyProfile`` in
``families.profiles``, ``PRESET_REGISTRY``/``_PRESET_TO_FAMILY`` in
``families.presets``, and ``_FAMILY_ALIASES`` in ``vector_retarget``. They had
already drifted -- ``qwen3_4b`` was known to one, ``qwen_3`` to another, and ViT
was missing from a third -- so adding a family was a scavenger hunt and a
mis-spelled alias failed in one place while working in the rest.

This module declares each family once. Everything else derives its own view from
these records, so the enums and lookup tables keep their existing types and call
sites are unchanged.

It deliberately imports nothing from the package: ``families.profiles`` must
stay importable on a host with no SDK, and ``contracts`` imports this, so a
dependency in either direction would be a cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyRecord:
    """One model family, in every spelling and every registry that needs it.

    ``model_family`` is the coarse build lane (``contracts.ModelFamily``);
    ``profile_id`` is the finer decoder profile (``families.profiles.FamilyId``)
    and is ``None`` for a family with no decoder, such as standalone ViT. Two
    records may share a ``model_family`` -- Omni Thinker builds through the
    Qwen3.5 lane but is its own family for routing and vector retargeting.
    """

    key: str
    model_family: str
    profile_id: str | None
    preset_ids: tuple[str, ...]
    canonical_name: str
    aliases: tuple[str, ...]
    retarget_allowed: bool
    kv_state: bool
    allowed_ars: frozenset[int] | None = None

    #: Whether the family builds through the decoder lane at all. Standalone
    #: ViT does not: one ONNX, AR1 only, no split/MHA2SHA/KV/weight sharing.
    has_decoder_lane: bool = True
    #: Whether a vision graph must be supplied and its projector-integrated
    #: output width checked against the text hidden size.
    has_vision_component: bool = False
    #: Whether the workflow packages an audio encoder beside the text lane.
    has_audio_component: bool = False
    #: Whether deriving several ARs from one export requires runtime-validated
    #: derivation evidence before the joint compile is authorized.
    requires_derivation_evidence: bool = False

    @property
    def spellings(self) -> tuple[str, ...]:
        """Canonical name plus every alias, deduplicated, order preserved."""

        seen: dict[str, None] = {}
        for value in (self.canonical_name, *self.aliases):
            seen.setdefault(value, None)
        return tuple(seen)


#: Every family, declared once.
#:
#: The alias sets are the *union* of what the four registries knew separately,
#: which is what stops a spelling from resolving in one entry point and failing
#: in another. Both hyphen and dot spellings are listed because the consumers
#: normalize differently: the profile table folds ``_`` to ``-`` and keeps dots,
#: while vector retargeting strips every separator.
FAMILY_RECORDS: tuple[FamilyRecord, ...] = (
    FamilyRecord(
        key="qwen3_dense",
        model_family="qwen3",
        profile_id="qwen3",
        preset_ids=("qwen3_dense",),
        canonical_name="qwen3",
        aliases=("qwen3-dense", "qwen3_dense", "qwen_3", "qwen3-4b", "qwen3_4b"),
        retarget_allowed=True,
        kv_state=True,
    ),
    FamilyRecord(
        key="qwen3_moe",
        model_family="qwen3_moe",
        profile_id="qwen3-moe",
        preset_ids=("qwen3_moe",),
        canonical_name="qwen3-moe",
        aliases=("qwen3_moe", "qwen3moe"),
        retarget_allowed=True,
        kv_state=True,
    ),
    FamilyRecord(
        key="qwen3_vl",
        model_family="qwen3_vl",
        profile_id="qwen3-vl",
        preset_ids=("qwen3_vl",),
        canonical_name="qwen3-vl",
        aliases=("qwen3_vl", "qwen3vl"),
        retarget_allowed=True,
        kv_state=True,
        has_vision_component=True,
    ),
    FamilyRecord(
        key="qwen3_5",
        model_family="qwen3_5",
        profile_id="qwen3.5",
        preset_ids=("qwen3_5",),
        canonical_name="qwen3.5",
        aliases=("qwen3-5", "qwen35", "qwen3_5"),
        retarget_allowed=False,
        kv_state=True,
        allowed_ars=frozenset({1, 128}),
        requires_derivation_evidence=True,
    ),
    FamilyRecord(
        key="qwen3_5_omni_thinker",
        model_family="qwen3_5",
        # The Thinker text lane is the Qwen3.5 decoder profile: its
        # architecture names are declared there.
        profile_id="qwen3.5",
        preset_ids=("qwen3_5_omni_thinker",),
        canonical_name="qwen3.5-omni-thinker",
        aliases=(
            "qwen3-5-omni-thinker",
            "qwen35-omni-thinker",
            "qwen3_5_omni_thinker",
        ),
        retarget_allowed=False,
        kv_state=True,
        allowed_ars=frozenset({1, 128}),
    ),
    FamilyRecord(
        key="qwen3_5_omni",
        model_family="qwen3_5_omni",
        profile_id="qwen3.5-omni",
        preset_ids=("qwen3_5_omni",),
        canonical_name="qwen3.5-omni",
        aliases=("qwen3-5-omni", "qwen35-omni", "qwen3_5_omni"),
        retarget_allowed=False,
        kv_state=True,
        allowed_ars=frozenset({1, 128}),
        has_audio_component=True,
    ),
    FamilyRecord(
        key="vit",
        model_family="vit",
        # Standalone ViT has no decoder profile: there is no AR/KV/split policy
        # to describe, which is exactly what a FamilyProfile is for.
        profile_id=None,
        preset_ids=("vit",),
        canonical_name="vit",
        aliases=("vision-transformer", "vision_transformer"),
        retarget_allowed=True,
        kv_state=False,
        allowed_ars=frozenset({1}),
        has_decoder_lane=False,
    ),
)

FAMILY_BY_KEY: dict[str, FamilyRecord] = {
    record.key: record for record in FAMILY_RECORDS
}

PRESET_TO_RECORD: dict[str, FamilyRecord] = {
    preset_id: record
    for record in FAMILY_RECORDS
    for preset_id in record.preset_ids
}


def _collapsed(value: str) -> str:
    """Every separator removed -- the vector-retarget normalization."""

    return re.sub(r"[-_.\s]+", "", value.strip().lower())


def _hyphenated(value: str) -> str:
    """Underscores folded to hyphens -- the family-profile normalization."""

    return value.strip().lower().replace("_", "-")


def aliases_for_profile(profile_id: str) -> tuple[str, ...]:
    """Every spelling that should resolve to a decoder profile."""

    seen: dict[str, None] = {}
    for record in FAMILY_RECORDS:
        if record.profile_id != profile_id:
            continue
        for spelling in record.spellings:
            seen.setdefault(spelling, None)
    return tuple(seen)


def resolve(value: object) -> FamilyRecord | None:
    """Resolve any declared spelling to its record, or ``None``.

    Both normalizations are tried, so a caller does not have to know which
    registry's spelling conventions it happens to be holding.
    """

    raw = str(getattr(value, "value", value))
    collapsed = _collapsed(raw)
    hyphenated = _hyphenated(raw)
    for record in FAMILY_RECORDS:
        for spelling in record.spellings:
            if _collapsed(spelling) == collapsed or _hyphenated(spelling) == hyphenated:
                return record
    return None


def _record_for(family: object) -> FamilyRecord:
    record = resolve(family)
    if record is None:
        raise ValueError(f"unknown model family {family!r}")
    return record


def has_decoder_lane(family: object) -> bool:
    """Whether this family builds through the decoder lane.

    Stage bodies ask this instead of comparing against ``ModelFamily.VIT``: the
    branch is about a capability the family either has or lacks, and naming it
    that way means a second decoder-less family needs no new comparison.
    """

    return _record_for(family).has_decoder_lane


def has_vision_component(family: object) -> bool:
    """Whether a vision graph must be supplied and boundary-checked."""

    return _record_for(family).has_vision_component


def has_audio_component(family: object) -> bool:
    """Whether the workflow packages an audio encoder beside the text lane."""

    return _record_for(family).has_audio_component


def requires_derivation_evidence(family: object, ar_count: int) -> bool:
    """Whether a multi-AR build must produce runtime-validated evidence first."""

    return _record_for(family).requires_derivation_evidence and ar_count > 1


__all__ = [
    "FAMILY_BY_KEY",
    "FAMILY_RECORDS",
    "FamilyRecord",
    "PRESET_TO_RECORD",
    "aliases_for_profile",
    "has_audio_component",
    "has_decoder_lane",
    "has_vision_component",
    "requires_derivation_evidence",
    "resolve",
]
