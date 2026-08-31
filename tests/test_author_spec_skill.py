"""The spec-authoring skill states hard constraints; they must stay true.

`.claude/skills/qairt-author-spec` tells an agent what to ask the user and what
each answer commits them to. Those claims are load-bearing -- an agent repeats
them to a person who then makes a decision -- so they are asserted against the
contracts rather than trusted to stay accurate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from qairt_agent.contracts import ModelFamily, SequenceSpec
from qairt_agent.families.presets import get_preset

SKILL = (
    Path(__file__).parents[1]
    / ".claude"
    / "skills"
    / "qairt-author-spec"
    / "SKILL.md"
)


def test_the_skill_exists_and_is_registered() -> None:
    assert SKILL.is_file()
    contract = (Path(__file__).parents[1] / "CLAUDE.md").read_text(encoding="utf-8")
    assert "qairt-author-spec" in contract, (
        "a skill the contract does not list is a skill nobody finds"
    )


def test_weight_sharing_defaults_on_which_is_why_the_skill_asks() -> None:
    # Silence means yes, so the agent must ask rather than let the default
    # decide for the user.
    assert SequenceSpec().weight_sharing is True
    assert "Weight sharing packages AR1 and AR128" in SKILL.read_text(encoding="utf-8")


def test_weight_sharing_requires_exactly_ar_1_and_128() -> None:
    SequenceSpec(ars=(1, 128))  # accepted

    for refused in ((1,), (128,), (1, 64, 128), (1, 32)):
        with pytest.raises(ValidationError):
            SequenceSpec(ars=refused)


def test_turning_weight_sharing_off_releases_the_ar_set() -> None:
    # The skill tells the user that adding AR64 means turning weight sharing
    # off; that has to actually work.
    spec = SequenceSpec(ars=(1, 64, 128), weight_sharing=False)

    assert spec.ars == (1, 64, 128)


def test_native_kv_requires_context_lengths_divisible_by_256() -> None:
    SequenceSpec(context_lengths=(4096,))
    SequenceSpec(context_lengths=(8192,))

    with pytest.raises(ValidationError):
        SequenceSpec(context_lengths=(4000,))


def test_every_ar_must_fit_every_context_length() -> None:
    with pytest.raises(ValidationError):
        SequenceSpec(ars=(1, 128), context_lengths=(64,))


def test_standalone_vit_refuses_weight_sharing_and_native_kv() -> None:
    preset = get_preset("vit")

    assert preset.default_weight_sharing is False
    assert preset.default_decoder_slices == 1


def test_the_decoder_presets_default_to_four_slices() -> None:
    for preset_id in ("qwen3_dense", "qwen3_moe"):
        assert get_preset(preset_id).default_decoder_slices == 4


def test_the_skill_does_not_claim_weight_sharing_is_hardware_proven() -> None:
    """No device acceptance run has exercised weight sharing.

    The smoke fixture is single-AR with weight sharing off, and the config cell
    that turns it on needs a real model export (T21). The skill must say so.
    """

    text = SKILL.read_text(encoding="utf-8")
    assert "no device acceptance run has exercised it" in text

    fixture = (
        Path(__file__).parents[1] / "tools" / "make_smoke_fixture.py"
    ).read_text(encoding="utf-8")
    assert '"weight_sharing": False' in fixture, (
        "if the fixture starts exercising weight sharing, the skill's caveat "
        "and this test must be revisited"
    )
