"""Generate family-aware build configuration from model metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .profiles import AutoArPolicy, FamilyId, FamilyProfile, resolve_family_profile
from .split_plan import SplitPlan, build_split_plan


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _first(value: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        candidate = _read(value, key)
        if candidate is not None:
            return candidate
    return default


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class GeneratedFamilyConfig:
    """Normalized fields needed by splitting, AR conversion, and MHA2SHA."""

    profile: FamilyProfile
    architecture: str
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    context_length: int
    vocab_size: int | None
    num_experts: int | None
    ar_values: tuple[int, ...]
    split_plan: SplitPlan
    source_config_container: str | None
    warnings: tuple[str, ...] = ()

    @property
    def family(self) -> FamilyId:
        return self.profile.family

    @property
    def auto_ar_policy(self) -> AutoArPolicy:
        return self.profile.auto_ar_policy

    @property
    def native_kv_head_shape(self) -> tuple[int, int]:
        return self.num_key_value_heads, self.head_dim

    def to_dict(self) -> dict[str, Any]:
        value = _jsonable(asdict(self))
        value["family"] = self.family.value
        value["auto_ar_policy"] = self.auto_ar_policy.value
        return value


class FamilyConfigGenerator:
    """Normalize Hugging Face-like configs into a deterministic build plan."""

    def generate(
        self,
        config: Any,
        *,
        family: FamilyId | str | None = None,
        ar_values: Sequence[int] = (1, 128),
        context_length: int | None = None,
        decoder_slices: int = 1,
        split_embedding: bool = True,
        split_lm_head: bool = True,
    ) -> GeneratedFamilyConfig:
        profile = resolve_family_profile(config, family)
        source = config
        if profile.config_container:
            nested = _read(config, profile.config_container)
            if nested is None:
                raise ValueError(
                    f"{profile.family.value} requires decoder config in "
                    f"{profile.config_container!r}"
                )
            source = nested

        architectures = _read(config, "architectures", ()) or _read(source, "architectures", ())
        if isinstance(architectures, str):
            architectures = (architectures,)
        architecture = str(
            architectures[0] if architectures else profile.architecture_names[0]
        )
        model_type = str(_first(config, "model_type", default=_read(source, "model_type", "")))

        hidden_size = int(_first(source, "hidden_size", "d_model"))
        num_layers = int(_first(source, "num_hidden_layers", "n_layer", "num_layers"))
        num_heads = int(_first(source, "num_attention_heads", "n_head"))
        num_kv_heads = int(
            _first(source, "num_key_value_heads", "num_kv_heads", default=num_heads)
        )
        explicit_head_dim = _first(source, "head_dim")
        if explicit_head_dim is None:
            if hidden_size % num_heads:
                raise ValueError("hidden_size must be divisible by num_attention_heads")
            head_dim = hidden_size // num_heads
        else:
            head_dim = int(explicit_head_dim)

        requested_ars = tuple(int(value) for value in ar_values)
        if not requested_ars or any(value <= 0 for value in requested_ars):
            raise ValueError("ar_values must contain positive integers")
        if len(set(requested_ars)) != len(requested_ars):
            raise ValueError("ar_values must be unique")

        resolved_context = int(
            context_length
            if context_length is not None
            else _first(
                source,
                "max_position_embeddings",
                "seq_length",
                "context_length",
                default=4096,
            )
        )
        if resolved_context <= 0:
            raise ValueError("context_length must be positive")

        warnings: list[str] = []
        if profile.experimental_auto_ar and len(requested_ars) > 1:
            warnings.append(
                "Qwen3.5 automatic single-source AR conversion is experimental and "
                "requires strict AR/state/MHA/initializer/standalone-vs-joint validation"
            )
        if profile.factory_support.value == "generic_fallback":
            warnings.append(
                f"QAIRT 2.49 does not have an explicit {profile.family.value} GenAI "
                "builder entry; validate generic-factory graph compatibility"
            )

        return GeneratedFamilyConfig(
            profile=profile,
            architecture=architecture,
            model_type=model_type,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            context_length=resolved_context,
            vocab_size=(
                int(value) if (value := _first(source, "vocab_size")) is not None else None
            ),
            num_experts=(
                int(value)
                if (value := _first(source, "num_experts", "num_local_experts")) is not None
                else None
            ),
            ar_values=requested_ars,
            split_plan=build_split_plan(
                num_layers,
                decoder_slices=decoder_slices,
                split_embedding=split_embedding,
                split_lm_head=split_lm_head,
            ),
            source_config_container=profile.config_container,
            warnings=tuple(warnings),
        )
