from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any


SCHEMA_VERSION = "v2_complete_spell_only"

SUBJECT_KINDS = ["summon_material"]
COLOR_LABELS = [
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "white",
    "black",
    "gray",
    "gold",
    "silver",
    "brown",
]
STATE_LABELS = ["solid", "liquid", "gas"]
REACTION_KIND_LABELS = ["none", "burn", "corrode", "freeze", "poison", "grow"]
REACTION_DIRECTION_LABELS = ["none", "up", "down", "forward", "outward"]
REACTION_MASK_LABELS = ["solid", "liquid", "gas", "living", "terrain"]
RELEASE_PROFILE_LABELS = ["burst", "stream", "spray", "pool", "beam"]
MOTION_TEMPLATE_LABELS = ["none", "fixed", "flow", "vortex", "rotation", "vibration"]
MOTION_DIRECTION_LABELS = ["forward", "backward", "up", "down", "target", "self", "none"]
ORIGIN_LABELS = ["self", "front_enemy_random", "back", "front_up", "front_down"]
DIRECTION_MODE_LABELS = ["to_enemy", "to_self", "forward", "backward", "none"]
VALUE_BINS_7 = ["very_low", "low", "mid_low", "mid", "mid_high", "high", "very_high"]
STYLE_AXIS_FIELDS = [
    "classicalness",
    "literaryness",
    "rituality",
    "colloquiality",
    "modernity",
    "directness",
]

MANDATORY_RUNTIME_FIELDS = [
    ("subject", "color"),
    ("subject", "state"),
    ("subject", "density"),
    ("subject", "temperature"),
    ("subject", "amount"),
    ("release", "release_profile"),
    ("release", "release_speed"),
    ("motion", "motion_template"),
    ("motion", "force_strength"),
    ("motion", "carrier_velocity"),
    ("motion", "motion_direction"),
    ("targeting", "origin"),
    ("targeting", "direction_mode"),
]

BINNED_SLOT_FIELDS = [
    ("subject", "density"),
    ("subject", "temperature"),
    ("subject", "amount"),
    ("subject", "reaction_rate"),
    ("subject", "hardness"),
    ("subject", "friction"),
    ("subject", "viscosity"),
    ("release", "release_speed"),
    ("release", "release_spread"),
    ("release", "release_duration"),
    ("motion", "force_strength"),
    ("motion", "carrier_velocity"),
]

OPTIONAL_FIELD_DEPENDENCIES = {
    ("subject", "reaction_rate"): ("subject", "reaction_kind", "none"),
    ("subject", "reaction_mask"): ("subject", "reaction_kind", "none"),
    ("subject", "reaction_direction"): ("subject", "reaction_kind", "none"),
    ("subject", "hardness"): ("subject", "state", "solid"),
    ("subject", "friction"): ("subject", "state", "solid"),
    ("subject", "viscosity"): ("subject", "state", "liquid"),
}

LABEL_SPACES: dict[tuple[str, str], list[str]] = {
    ("subject", "color"): COLOR_LABELS,
    ("subject", "state"): STATE_LABELS,
    ("subject", "reaction_kind"): REACTION_KIND_LABELS,
    ("subject", "reaction_direction"): REACTION_DIRECTION_LABELS,
    ("release", "release_profile"): RELEASE_PROFILE_LABELS,
    ("motion", "motion_template"): MOTION_TEMPLATE_LABELS,
    ("motion", "motion_direction"): MOTION_DIRECTION_LABELS,
    ("targeting", "origin"): ORIGIN_LABELS,
    ("targeting", "direction_mode"): DIRECTION_MODE_LABELS,
}


@dataclass(frozen=True)
class SlotSpec:
    path: tuple[str, str]
    labels: list[str]


CATEGORICAL_SPECS = [SlotSpec(path, labels) for path, labels in LABEL_SPACES.items()]
BINNED_SPECS = [SlotSpec(path, VALUE_BINS_7) for path in BINNED_SLOT_FIELDS]

BIN_NORMALIZATION = {
    "very low": "very_low",
    "very_low": "very_low",
    "low": "low",
    "medium low": "mid_low",
    "medium_low": "mid_low",
    "mid low": "mid_low",
    "mid_low": "mid_low",
    "medium": "mid",
    "mid": "mid",
    "medium high": "mid_high",
    "medium_high": "mid_high",
    "mid high": "mid_high",
    "mid_high": "mid_high",
    "high": "high",
    "very high": "very_high",
    "very_high": "very_high",
}

FIELD_SPECIFIC_BIN_NORMALIZATION: dict[tuple[str, str], dict[str, str]] = {
    ("subject", "temperature"): {
        "freezing": "very_low",
        "very_cold": "very_low",
        "ice_cold": "very_low",
        "cold": "low",
        "cool": "mid_low",
        "warm": "mid_high",
        "hot": "high",
        "very_hot": "very_high",
        "scorching": "very_high",
        "blazing": "very_high",
    },
    ("subject", "density"): {
        "weightless": "very_low",
        "very_light": "very_low",
        "light": "low",
        "medium_density": "mid",
        "heavy": "high",
        "very_heavy": "very_high",
    },
    ("subject", "amount"): {
        "tiny": "very_low",
        "small": "low",
        "little": "low",
        "medium_amount": "mid",
        "large": "high",
        "huge": "very_high",
    },
    ("release", "release_speed"): {
        "slow": "low",
        "fast": "high",
        "very_fast": "very_high",
    },
    ("motion", "force_strength"): {
        "weak": "low",
        "strong": "high",
        "very_strong": "very_high",
    },
    ("motion", "carrier_velocity"): {
        "slow": "low",
        "fast": "high",
        "very_fast": "very_high",
    },
}

COLOR_NORMALIZATION = {
    "grey": "gray",
    "dark_gray": "gray",
    "light": "white",
    "bright": "white",
    "pale": "white",
    "transparent": "white",
    "amber": "orange",
    "red_orange": "orange",
    "blue_white": "white",
    "golden": "gold",
}
STYLE_AXIS_ALIASES = {"colloquialness": "colloquiality"}
MOTION_TEMPLATE_NORMALIZATION = {"stationary": "fixed", "forward": "flow", "backward": "flow", "up": "flow", "down": "flow", "target": "flow", "self": "flow"}
DIRECTION_MODE_NORMALIZATION = {"to_target": "to_enemy", "aim_enemy": "to_enemy", "aim_self": "to_self"}
REACTION_DIRECTION_NORMALIZATION = {"through": "forward"}
SUBJECT_KIND_NORMALIZATION = {"manipulate_material": "summon_material"}
ORIGIN_NORMALIZATION = {"front": "front_enemy_random"}
REACTION_MASK_ALIASES = {
    "enemy": "living",
    "enemies": "living",
    "person": "living",
    "people": "living",
    "human": "living",
    "body": "living",
    "creature": "living",
    "ground": "terrain",
    "wall": "terrain",
    "floor": "terrain",
    "road": "terrain",
    "obstacle": "terrain",
    "surface": "terrain",
    "terrain": "terrain",
    "armor": "solid",
    "shield": "solid",
    "metal": "solid",
    "stone": "solid",
    "water": "liquid",
    "liquid": "liquid",
    "air": "gas",
    "smoke": "gas",
    "mist": "gas",
}
REACTION_MASK_FALLBACK = {
    "burn": ["living", "terrain"],
    "corrode": ["solid", "terrain"],
    "freeze": ["living", "terrain"],
    "poison": ["living"],
    "grow": ["terrain", "living"],
}


def get_nested(mapping: dict[str, Any] | None, path: tuple[str, str]) -> Any:
    if mapping is None:
        return None
    section = mapping.get(path[0])
    if section is None:
        return None
    return section.get(path[1])


def set_nested(mapping: dict[str, Any], path: tuple[str, str], value: Any) -> None:
    mapping.setdefault(path[0], {})[path[1]] = value


def _validate_style_axes(style_axes: dict[str, Any] | None) -> None:
    if style_axes is None:
        return
    if not isinstance(style_axes, dict):
        raise ValueError("expression.style_axes must be an object")
    missing = [field for field in STYLE_AXIS_FIELDS if field not in style_axes]
    if missing:
        raise ValueError(f"missing style axes: {missing}")
    for field in STYLE_AXIS_FIELDS:
        value = style_axes[field]
        if not isinstance(value, (int, float)):
            raise ValueError(f"style axis {field} must be numeric")
        numeric = float(value)
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"style axis {field} out of range: {numeric}")


def normalize_runtime_b(runtime_b: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(runtime_b)
    subject_kind = normalized.get("subject_kind")
    if isinstance(subject_kind, str):
        normalized["subject_kind"] = SUBJECT_KIND_NORMALIZATION.get(subject_kind.strip().lower(), subject_kind)
    for path in BINNED_SLOT_FIELDS:
        value = get_nested(normalized, path)
        if isinstance(value, str):
            lowered = value.strip().lower()
            normalized_value = FIELD_SPECIFIC_BIN_NORMALIZATION.get(path, {}).get(lowered)
            if normalized_value is None:
                normalized_value = BIN_NORMALIZATION.get(lowered)
            if normalized_value is not None:
                set_nested(normalized, path, normalized_value)
    color = get_nested(normalized, ("subject", "color"))
    if isinstance(color, str):
        lowered = color.strip().lower()
        set_nested(normalized, ("subject", "color"), COLOR_NORMALIZATION.get(lowered, color))
    origin = get_nested(normalized, ("targeting", "origin"))
    if isinstance(origin, str):
        lowered = origin.strip().lower()
        mapped = ORIGIN_NORMALIZATION.get(lowered)
        if mapped is not None:
            set_nested(normalized, ("targeting", "origin"), mapped)
    motion_template = get_nested(normalized, ("motion", "motion_template"))
    if isinstance(motion_template, str):
        mapped = MOTION_TEMPLATE_NORMALIZATION.get(motion_template.strip().lower())
        if mapped is not None:
            set_nested(normalized, ("motion", "motion_template"), mapped)
    reaction_direction = get_nested(normalized, ("subject", "reaction_direction"))
    if isinstance(reaction_direction, str):
        mapped = REACTION_DIRECTION_NORMALIZATION.get(reaction_direction.strip().lower())
        if mapped is not None:
            set_nested(normalized, ("subject", "reaction_direction"), mapped)
    direction_mode = get_nested(normalized, ("targeting", "direction_mode"))
    if isinstance(direction_mode, str):
        mapped = DIRECTION_MODE_NORMALIZATION.get(direction_mode.strip().lower())
        if mapped is not None:
            set_nested(normalized, ("targeting", "direction_mode"), mapped)
    targeting = normalized.get("targeting")
    if isinstance(targeting, dict):
        targeting.pop("target_mode", None)
    reaction_mask = get_nested(normalized, ("subject", "reaction_mask"))
    reaction_kind = get_nested(normalized, ("subject", "reaction_kind"))
    if isinstance(reaction_mask, list):
        mapped_mask: list[str] = []
        for item in reaction_mask:
            lowered = str(item).strip().lower()
            mapped = REACTION_MASK_ALIASES.get(lowered, lowered)
            if mapped in REACTION_MASK_LABELS and mapped not in mapped_mask:
                mapped_mask.append(mapped)
        if isinstance(reaction_kind, str) and reaction_kind == "none":
            normalized["subject"].pop("reaction_mask", None)
        elif not mapped_mask and isinstance(reaction_kind, str):
            mapped_mask = list(REACTION_MASK_FALLBACK.get(reaction_kind, []))
            set_nested(normalized, ("subject", "reaction_mask"), mapped_mask)
        else:
            set_nested(normalized, ("subject", "reaction_mask"), mapped_mask)
    expression = normalized.get("expression")
    if isinstance(expression, dict):
        style_axes = expression.get("style_axes")
        if isinstance(style_axes, dict):
            for alias, canonical in STYLE_AXIS_ALIASES.items():
                if alias in style_axes and canonical not in style_axes:
                    style_axes[canonical] = style_axes.pop(alias)
    return normalized


def validate_runtime_b(runtime_b: dict[str, Any] | None, *, allow_none: bool = False) -> None:
    if runtime_b is None:
        if allow_none:
            return
        raise ValueError("runtime_b cannot be None")
    if runtime_b.get("subject_kind") not in SUBJECT_KINDS:
        raise ValueError(f"invalid subject_kind: {runtime_b.get('subject_kind')}")
    for path in MANDATORY_RUNTIME_FIELDS:
        if get_nested(runtime_b, path) is None:
            raise ValueError(f"missing mandatory field: {path}")
    for path, labels in LABEL_SPACES.items():
        value = get_nested(runtime_b, path)
        if value is None:
            continue
        if value not in labels:
            raise ValueError(f"invalid value for {path}: {value}")
    for path in BINNED_SLOT_FIELDS:
        value = get_nested(runtime_b, path)
        if value is None:
            continue
        if value not in VALUE_BINS_7:
            raise ValueError(f"invalid bin value for {path}: {value}")
    reaction_mask = get_nested(runtime_b, ("subject", "reaction_mask"))
    if reaction_mask is not None:
        if not isinstance(reaction_mask, list):
            raise ValueError("reaction_mask must be a list")
        unknown = [item for item in reaction_mask if item not in REACTION_MASK_LABELS]
        if unknown:
            raise ValueError(f"invalid reaction_mask items: {unknown}")
    for path, dependency in OPTIONAL_FIELD_DEPENDENCIES.items():
        value = get_nested(runtime_b, path)
        section, key, disallowed_when = dependency
        base_value = get_nested(runtime_b, (section, key))
        if path == ("subject", "reaction_rate") and base_value == disallowed_when and value is not None:
            raise ValueError("reaction_rate must be omitted when reaction_kind is none")
        if path == ("subject", "reaction_mask") and base_value == disallowed_when and value is not None:
            raise ValueError("reaction_mask must be omitted when reaction_kind is none")
        if path == ("subject", "reaction_direction") and base_value == disallowed_when and value is not None:
            raise ValueError("reaction_direction must be omitted when reaction_kind is none")
        if path == ("subject", "hardness") and base_value != disallowed_when and value is not None:
            raise ValueError("hardness only applies to solid state")
        if path == ("subject", "friction") and base_value != disallowed_when and value is not None:
            raise ValueError("friction only applies to solid state")
        if path == ("subject", "viscosity") and base_value != disallowed_when and value is not None:
            raise ValueError("viscosity only applies to liquid state")
    _validate_style_axes(get_nested(runtime_b, ("expression", "style_axes")))
