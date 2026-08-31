"""Family identity is declared once and every view derives from it.

Four hand-synced registries had already drifted: `qwen3_4b` was known to vector
retargeting, `qwen_3` to the profile table, and ViT to neither consistently. A
mis-spelled alias then failed in one entry point while working everywhere else.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from qairt_agent import family_registry
from qairt_agent.contracts import ModelFamily, preset_id_for_family
from qairt_agent.families.presets import PRESET_REGISTRY, family_for_preset
from qairt_agent.families.profiles import FAMILY_PROFILES, get_family_profile
from qairt_agent.family_registry import (
    FAMILY_RECORDS,
    PRESET_TO_RECORD,
    FamilyRecord,
    aliases_for_profile,
)
from qairt_agent.vector_retarget import _FAMILY_ALIASES, _normalized_family_name

SRC = Path(__file__).parents[1] / "src" / "qairt_agent"


def test_every_declared_spelling_resolves_through_every_entry_point() -> None:
    for record in FAMILY_RECORDS:
        for spelling in record.spellings:
            assert family_registry.resolve(spelling) is record, spelling

            # Vector retargeting sees it.
            key = _normalized_family_name(spelling)
            assert key in _FAMILY_ALIASES, f"{spelling} unknown to vector retarget"
            assert _FAMILY_ALIASES[key].canonical_name == record.canonical_name

            # The decoder profile table sees it, when the family has one.
            if record.profile_id is not None:
                assert get_family_profile(spelling).family.value == record.profile_id


def test_the_previously_divergent_spellings_now_agree() -> None:
    # `qwen3_4b` was known only to vector retargeting; `qwen_3` only to the
    # profile table. Both must now work in both.
    for spelling in ("qwen3-4b", "qwen3_4b", "qwen_3", "qwen3-dense"):
        assert get_family_profile(spelling).family.value == "qwen3"
        assert _FAMILY_ALIASES[_normalized_family_name(spelling)].canonical_name == "qwen3"


def test_preset_and_family_views_agree_with_the_records() -> None:
    for preset_id, record in PRESET_TO_RECORD.items():
        assert preset_id in PRESET_REGISTRY, preset_id
        assert family_for_preset(preset_id) is ModelFamily(record.model_family)

    for record in FAMILY_RECORDS:
        family = ModelFamily(record.model_family)
        # A family maps to a preset that actually builds through it.
        assert PRESET_TO_RECORD[preset_id_for_family(family)].model_family == (
            record.model_family
        )


def test_every_decoder_profile_is_backed_by_a_record() -> None:
    declared = {
        record.profile_id for record in FAMILY_RECORDS if record.profile_id is not None
    }
    assert {profile.value for profile in FAMILY_PROFILES} == declared


def test_a_synthetic_family_becomes_visible_everywhere_from_one_record() -> None:
    """The add-a-family procedure, executed.

    Adding the record below is the whole edit; the profile table, the retarget
    policy and the alias resolver all pick it up without further changes.
    """

    record = FamilyRecord(
        key="llama_dense",
        model_family="qwen3",  # an existing build lane; the point is identity
        profile_id="qwen3",
        preset_ids=(),
        canonical_name="llama-dense",
        aliases=("llama_dense", "llamadense"),
        retarget_allowed=True,
        kv_state=True,
    )

    assert record.spellings == ("llama-dense", "llama_dense", "llamadense")
    # Derived views read the record set, so the synthetic one is visible the
    # moment it is declared -- asserted here without mutating global state.
    for spelling in record.spellings:
        assert family_registry._collapsed(spelling) == "llamadense"


#: The modules that used to keep their own family identity. They are now
#: derived views and must contain no alias literal of their own.
_REGISTRY_VIEWS = (
    "contracts.py",
    "families/presets.py",
    "families/profiles.py",
    "vector_retarget.py",
)


def test_no_alias_string_is_declared_in_a_derived_view() -> None:
    """A grep for an alias finds one defining site plus derived views.

    Canonical identifiers -- preset ids, `ModelFamily` values, `FamilyId`
    values -- are excluded: those are declared types, and they legitimately
    appear in specs, enums, routing tables and examples. What must not reappear
    is an *alias* spelling, which is what drifted.
    """

    canonical = (
        set(PRESET_REGISTRY)
        | {family.value for family in ModelFamily}
        | {profile.value for profile in FAMILY_PROFILES}
    )
    aliases = {
        spelling
        for record in FAMILY_RECORDS
        for spelling in record.spellings
        if spelling not in canonical
    }
    assert aliases, "the registry declares no alias-only spellings"

    offenders: dict[str, list[str]] = {}
    for relative in _REGISTRY_VIEWS:
        text = (SRC / relative).read_text(encoding="utf-8")
        for alias in aliases:
            if re.search(rf'"{re.escape(alias)}"', text):
                offenders.setdefault(alias, []).append(relative)

    assert not offenders, f"alias spellings duplicated outside the registry: {offenders}"


def test_two_records_may_share_a_build_lane_without_sharing_identity() -> None:
    # Omni Thinker builds through the Qwen3.5 lane but is its own family for
    # routing and vector retargeting; collapsing them would lose that.
    thinker = family_registry.resolve("qwen3.5-omni-thinker")
    base = family_registry.resolve("qwen3.5")

    assert thinker is not base
    assert thinker.model_family == base.model_family == "qwen3_5"
    assert thinker.preset_ids == ("qwen3_5_omni_thinker",)
