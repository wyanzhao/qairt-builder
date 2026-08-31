"""Stable model-family profiles used by the build planner.

The profiles deliberately describe policy, not SDK objects.  Keeping this
module free of QAIRT imports lets an agent inspect and plan a build on a host
where the pinned SDK is not installed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from qairt_agent.family_registry import aliases_for_profile


class FamilyId(str, Enum):
    """Model families supported by the first-party configuration generator."""

    QWEN3_DENSE = "qwen3"
    QWEN3_MOE = "qwen3-moe"
    QWEN3_VL = "qwen3-vl"
    QWEN3_5 = "qwen3.5"
    QWEN3_5_OMNI = "qwen3.5-omni"


class AutoArPolicy(str, Enum):
    """Whether a single exported graph may be rewritten to another AR."""

    SUPPORTED = "supported"
    EXPERIMENTAL_FAIL_CLOSED = "experimental_fail_closed"


class QairtFactorySupport(str, Enum):
    """Historical GenAI builder capability label.

    ``EXPLICIT`` means the adapter has a pinned, family-specific SDK builder
    binding.  It is not evidence that ``GenAIBuilderFactory`` recognizes every
    architecture name in that family.  In particular, Qwen3.5 and Omni Thinker
    use ``Qwen3_5BuilderHTP.from_pretrained`` directly.
    """

    EXPLICIT = "explicit"
    GENERIC_FALLBACK = "generic_fallback"


@dataclass(frozen=True)
class MhaStartPointSpec:
    """A family-specific MHA2SHA split start point."""

    output_name_regex: str
    axis: int
    split_map: Mapping[int, int] | None = None
    note: str = ""


def start_point_fingerprint(
    points: "Sequence[MhaStartPointSpec] | Sequence[tuple[str, int, Mapping[int, int] | None]]",
) -> str:
    """Hash a start-point set so a change in the SDK's copy is never silent.

    Normalizes to ``[(pattern, axis, {int: int})]`` before hashing, so the
    fingerprint compares the *meaning* rather than an object repr that could
    drift with an unrelated SDK refactor.
    """

    normalized: list[list[Any]] = []
    for item in points:
        if isinstance(item, MhaStartPointSpec):
            pattern, axis, split_map = item.output_name_regex, item.axis, item.split_map
        else:
            pattern, axis, split_map = item
        normalized.append(
            [
                str(pattern),
                int(axis),
                (
                    {str(int(key)): int(value) for key, value in dict(split_map).items()}
                    if split_map
                    else None
                ),
            ]
        )
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SdkStartPointSource:
    """Where the SDK keeps a family's own MHA2SHA start points.

    The values are read from the SDK at transform time rather than copied here,
    so an upper-layer change does not have to be mirrored by hand. The
    fingerprint is the guard: it only ever changes when the SDK pin moves, which
    is already a reviewed event with a device acceptance run, and a mismatch
    fails closed naming exactly what changed instead of silently reslicing the
    graph.
    """

    module: str
    qualname: str
    reviewed_sha256: str
    note: str = ""


@dataclass(frozen=True)
class FamilyProfile:
    """Declarative family capabilities and strict safety policy."""

    family: FamilyId
    architecture_names: tuple[str, ...]
    model_type_names: tuple[str, ...]
    aliases: tuple[str, ...]
    factory_support: QairtFactorySupport
    auto_ar_policy: AutoArPolicy
    is_moe: bool = False
    is_multimodal: bool = False
    is_hybrid_attention: bool = False
    config_container: str | None = None
    sdk_mha_start_points: SdkStartPointSource | None = None

    @property
    def experimental_auto_ar(self) -> bool:
        return self.auto_ar_policy is AutoArPolicy.EXPERIMENTAL_FAIL_CLOSED


QWEN3_DENSE = FamilyProfile(
    family=FamilyId.QWEN3_DENSE,
    architecture_names=("Qwen3ForCausalLM",),
    model_type_names=("qwen3",),
    aliases=aliases_for_profile("qwen3"),
    factory_support=QairtFactorySupport.GENERIC_FALLBACK,
    auto_ar_policy=AutoArPolicy.SUPPORTED,
)

QWEN3_MOE = FamilyProfile(
    family=FamilyId.QWEN3_MOE,
    architecture_names=("Qwen3MoeForCausalLM",),
    model_type_names=("qwen3_moe", "qwen3-moe"),
    aliases=aliases_for_profile("qwen3-moe"),
    factory_support=QairtFactorySupport.EXPLICIT,
    auto_ar_policy=AutoArPolicy.SUPPORTED,
    is_moe=True,
)

QWEN3_VL = FamilyProfile(
    family=FamilyId.QWEN3_VL,
    architecture_names=(
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLModel",
    ),
    model_type_names=("qwen3_vl", "qwen3-vl"),
    aliases=aliases_for_profile("qwen3-vl"),
    factory_support=QairtFactorySupport.GENERIC_FALLBACK,
    auto_ar_policy=AutoArPolicy.SUPPORTED,
    is_multimodal=True,
    config_container="text_config",
)

QWEN3_5 = FamilyProfile(
    family=FamilyId.QWEN3_5,
    architecture_names=(
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen3_5OmniThinkerForConditionalGeneration",
        "Qwen3_5OmniThinkerModel",
    ),
    model_type_names=(
        "qwen3_5",
        "qwen3.5",
        "qwen3_5_text",
        "qwen3_5_omni_thinker",
    ),
    aliases=aliases_for_profile("qwen3.5"),
    # The capability is explicit through Qwen3_5BuilderHTP.from_pretrained;
    # GenAIBuilderFactory does not recognize every architecture listed above.
    factory_support=QairtFactorySupport.EXPLICIT,
    auto_ar_policy=AutoArPolicy.EXPERIMENTAL_FAIL_CLOSED,
    is_hybrid_attention=True,
    # Not copied here on purpose. The SDK's own Qwen3.5 builder carries these,
    # and duplicating them means an upper-layer change silently reslices the
    # graph. The adapter reads them from the SDK at transform time; the
    # fingerprint below is what makes a change loud instead of invisible.
    sdk_mha_start_points=SdkStartPointSource(
        module="qairt.gen_ai_api.builders.qwen.builder",
        qualname="Qwen3_5BuilderHTP._QWEN3_5_START_POINTS",
        reviewed_sha256=(
            "e6276e42e66ae5826d91e552709ff99371d03b87624d4c3315898627923ea960"
        ),
        note=(
            "Linear-attention norm output (axis 1), full-attention output "
            "(axis 2, 4096 -> 256 per KV head), and the recurrent/conv state "
            "outputs (axis 1). Reviewed against QAIRT 2.49.0.260730."
        ),
    ),
)

QWEN3_5_OMNI = FamilyProfile(
    family=FamilyId.QWEN3_5_OMNI,
    architecture_names=("Qwen3OmniForConditionalGeneration",),
    model_type_names=("qwen3_omni", "qwen3.5_omni", "qwen3_5_omni"),
    aliases=aliases_for_profile("qwen3.5-omni"),
    factory_support=QairtFactorySupport.EXPLICIT,
    auto_ar_policy=AutoArPolicy.EXPERIMENTAL_FAIL_CLOSED,
    is_multimodal=True,
    is_hybrid_attention=True,
    config_container="text_config",
    sdk_mha_start_points=QWEN3_5.sdk_mha_start_points,
)


FAMILY_PROFILES: Mapping[FamilyId, FamilyProfile] = {
    profile.family: profile
    for profile in (QWEN3_DENSE, QWEN3_MOE, QWEN3_VL, QWEN3_5, QWEN3_5_OMNI)
}


def _normalized(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def get_family_profile(family: FamilyId | str) -> FamilyProfile:
    """Resolve a stable family id or a friendly alias."""

    if isinstance(family, FamilyId):
        return FAMILY_PROFILES[family]
    candidate = _normalized(str(getattr(family, "value", family)))
    for profile in FAMILY_PROFILES.values():
        names = {
            _normalized(profile.family.value),
            *(_normalized(alias) for alias in profile.aliases),
        }
        if candidate in names:
            return profile
    supported = ", ".join(profile.family.value for profile in FAMILY_PROFILES.values())
    raise ValueError(f"unsupported model family {family!r}; expected one of: {supported}")


def _config_identity(config: Any) -> tuple[frozenset[str], frozenset[str]]:
    """The architecture and model_type names a Hugging Face config declares.

    Multimodal configs sometimes put the architecture at the outer level and
    all decoder dimensions under ``text_config``, so both levels are read.
    """

    architectures = _read(config, "architectures", ()) or ()
    if isinstance(architectures, str):
        architectures = (architectures,)
    model_type = _read(config, "model_type")

    architecture_candidates = {str(item) for item in architectures}
    model_type_candidates = {str(model_type)} if model_type else set()
    text_config = _read(config, "text_config")
    if text_config is not None:
        text_architectures = _read(text_config, "architectures", ()) or ()
        if isinstance(text_architectures, str):
            text_architectures = (text_architectures,)
        architecture_candidates.update(str(item) for item in text_architectures)
        text_model_type = _read(text_config, "model_type")
        if text_model_type:
            model_type_candidates.add(str(text_model_type))
    return frozenset(architecture_candidates), frozenset(model_type_candidates)


def resolve_family_profile(config: Any, family: FamilyId | str | None = None) -> FamilyProfile:
    """Resolve a family from an explicit id or Hugging Face-style config."""

    if family is not None:
        return get_family_profile(family)

    architecture_candidates, model_type_candidates = _config_identity(config)

    # An explicit architecture is more specific than a nested decoder
    # ``model_type`` (Qwen3-VL legitimately nests model_type="qwen3").
    for profile in FAMILY_PROFILES.values():
        if architecture_candidates.intersection(profile.architecture_names):
            return profile
    for profile in FAMILY_PROFILES.values():
        if model_type_candidates.intersection(profile.model_type_names):
            return profile

    candidates = architecture_candidates | model_type_candidates
    rendered = ", ".join(sorted(candidates)) or "<missing architecture/model_type>"
    raise ValueError(f"cannot resolve a supported Qwen family from config values: {rendered}")


class FamilyCrossCheckStatus(str, Enum):
    """What the supplied config had to say about the declared family."""

    AGREES = "agrees"
    SILENT = "silent"
    UNKNOWN_ARCHITECTURE = "unknown_architecture"
    MODEL_TYPE_DISAGREES = "model_type_disagrees"
    CONTRADICTS = "contradicts"
    NO_PROFILE = "no_profile"


@dataclass(frozen=True)
class FamilyCrossCheck:
    """Whether the supplied HF config corroborates the declared preset.

    The preset stays the routing authority by program decision. Authority
    without a cross-check, though, means a mis-declared export (a Qwen3.5
    hybrid export declared ``qwen3_dense``) silently bypasses every
    family-specific gate, so what the config actually says is read and
    compared. Only an *architecture* naming a different known family is a
    contradiction: a nested ``model_type`` is legitimately ambiguous
    (Qwen3-VL nests ``model_type="qwen3"``), and an architecture the table has
    never seen must not block a new family.
    """

    status: FamilyCrossCheckStatus
    declared_family: str
    architectures: tuple[str, ...]
    model_types: tuple[str, ...]
    implied_family: str | None = None
    config_path: str | None = None

    @property
    def contradicts(self) -> bool:
        return self.status is FamilyCrossCheckStatus.CONTRADICTS

    @property
    def is_warning(self) -> bool:
        return self.status in {
            FamilyCrossCheckStatus.UNKNOWN_ARCHITECTURE,
            FamilyCrossCheckStatus.MODEL_TYPE_DISAGREES,
        }

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status.value,
            "declared_family": self.declared_family,
            "architectures": list(self.architectures),
            "model_types": list(self.model_types),
        }
        if self.implied_family is not None:
            payload["implied_family"] = self.implied_family
        if self.config_path is not None:
            payload["config_path"] = self.config_path
        return payload


def cross_check_declared_family(
    config: Any,
    family: FamilyId | str,
    *,
    config_path: str | None = None,
) -> FamilyCrossCheck:
    """Compare a declared family against what the supplied config declares."""

    declared = str(getattr(family, "value", family))
    try:
        declared_profile = get_family_profile(family)
    except ValueError:
        # Standalone ViT and anything else without a decoder profile: there is
        # no architecture table to compare against, so there is nothing to say.
        return FamilyCrossCheck(
            status=FamilyCrossCheckStatus.NO_PROFILE,
            declared_family=declared,
            architectures=(),
            model_types=(),
            config_path=config_path,
        )

    architectures, model_types = _config_identity(config)

    def result(
        status: FamilyCrossCheckStatus, implied: str | None = None
    ) -> FamilyCrossCheck:
        return FamilyCrossCheck(
            status=status,
            declared_family=declared,
            architectures=tuple(sorted(architectures)),
            model_types=tuple(sorted(model_types)),
            implied_family=implied,
            config_path=config_path,
        )

    if not architectures and not model_types:
        return result(FamilyCrossCheckStatus.SILENT)

    if architectures:
        for profile in FAMILY_PROFILES.values():
            if architectures.intersection(profile.architecture_names):
                if profile.family is declared_profile.family:
                    return result(FamilyCrossCheckStatus.AGREES)
                return result(
                    FamilyCrossCheckStatus.CONTRADICTS, profile.family.value
                )

    if model_types:
        for profile in FAMILY_PROFILES.values():
            if model_types.intersection(profile.model_type_names):
                if profile.family is declared_profile.family:
                    return result(FamilyCrossCheckStatus.AGREES)
                # A model_type pointing elsewhere is the weaker signal --
                # Qwen3-VL legitimately nests model_type="qwen3" -- so it is
                # reported, never failed on.
                return result(
                    FamilyCrossCheckStatus.MODEL_TYPE_DISAGREES,
                    profile.family.value,
                )

    return result(FamilyCrossCheckStatus.UNKNOWN_ARCHITECTURE)


def validate_weight_sharing_sources(
    profile: FamilyProfile | FamilyId | str,
    source_kinds: tuple[str, ...] | list[str],
) -> None:
    """Validate source labels without forbidding Qwen3.5 single-source AR.

    Qwen3.5's canonical flow may derive AR variants from one base graph.  Its
    fail-closed policy is enforced by adapter validation evidence (AR/state/MHA/
    initializer/standalone-vs-joint gates), not by requiring attached models.
    """

    _ = profile if isinstance(profile, FamilyProfile) else get_family_profile(profile)
    normalized = tuple(str(kind).strip().lower() for kind in source_kinds)
    supported = {"base", "derived", "attached"}
    invalid = tuple(kind for kind in normalized if kind not in supported)
    if invalid:
        raise ValueError(
            f"unsupported model source_kind values: {invalid}; expected base, derived, or attached"
        )
