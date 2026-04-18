from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ONTOLOGY_VERSION = "v1_text_qwen06b"

STATUS_LABELS = ["success", "unstable", "backfire"]
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
]
STATE_LABELS = ["solid", "liquid", "gas"]
REACTION_KIND_LABELS = ["none", "burn", "corrode", "freeze", "poison", "grow"]
REACTION_DIRECTION_LABELS = ["none", "up", "down", "forward", "outward"]
REACTION_MASK_LABELS = ["solid", "liquid", "gas", "living", "terrain"]
RELEASE_PROFILE_LABELS = ["burst", "stream", "spray", "pool", "beam"]
MOTION_TEMPLATE_LABELS = ["fixed", "flow", "vortex"]
MOTION_DIRECTION_LABELS = ["forward", "backward", "up", "down", "target", "self", "none"]
ORIGIN_LABELS = ["self", "front_enemy_random", "back", "front_up", "front_down"]
TARGET_MODE_LABELS = ["aim_enemy", "aim_self", "none"]
DIRECTION_MODE_LABELS = ["to_target", "to_self", "forward", "none"]
STYLE_BINS = ["very_low", "low", "mid", "high", "very_high"]
VALUE_BINS_7 = ["very_low", "low", "mid_low", "mid", "mid_high", "high", "very_high"]

MATERIAL_ARCHETYPE_LABELS = [
    "holy_fire",
    "wildfire",
    "acid_slime",
    "poison_mist",
    "frost_breath",
    "ice_shard",
    "stone_lance",
    "iron_sand",
    "thunder_arc",
    "storm_spark",
    "shadow_tar",
    "radiant_dust",
    "lava_bloom",
    "bramble_growth",
    "quicksilver_stream",
    "ash_cloud",
]

DEFAULT_COLOR_BY_ARCHETYPE = {
    "holy_fire": "gold",
    "wildfire": "orange",
    "acid_slime": "green",
    "poison_mist": "purple",
    "frost_breath": "cyan",
    "ice_shard": "white",
    "stone_lance": "gray",
    "iron_sand": "silver",
    "thunder_arc": "blue",
    "storm_spark": "purple",
    "shadow_tar": "black",
    "radiant_dust": "gold",
    "lava_bloom": "orange",
    "bramble_growth": "green",
    "quicksilver_stream": "silver",
    "ash_cloud": "gray",
}

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
    ("targeting", "origin"),
    ("targeting", "target_mode"),
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

STYLE_FIELDS = [
    ("expression", "curvature"),
    ("expression", "politeness"),
    ("expression", "elegance"),
]

CATEGORICAL_SLOT_FIELDS = [
    ("subject", "color"),
    ("subject", "state"),
    ("subject", "reaction_kind"),
    ("subject", "reaction_direction"),
    ("release", "release_profile"),
    ("motion", "motion_template"),
    ("motion", "motion_direction"),
    ("targeting", "origin"),
    ("targeting", "target_mode"),
    ("targeting", "direction_mode"),
]

OPTIONAL_FIELD_DEPENDENCIES = {
    ("subject", "reaction_rate"): ("subject", "reaction_kind", "none"),
    ("subject", "reaction_mask"): ("subject", "reaction_kind", "none"),
    ("subject", "reaction_direction"): ("subject", "reaction_kind", "none"),
    ("subject", "hardness"): ("subject", "state", "solid"),
    ("subject", "friction"): ("subject", "state", "solid"),
    ("subject", "viscosity"): ("subject", "state", "liquid"),
}


LABEL_SPACES = {
    ("status",): STATUS_LABELS,
    ("subject", "material_archetype"): MATERIAL_ARCHETYPE_LABELS + ["none"],
    ("subject", "color"): COLOR_LABELS,
    ("subject", "state"): STATE_LABELS,
    ("subject", "reaction_kind"): REACTION_KIND_LABELS,
    ("subject", "reaction_direction"): REACTION_DIRECTION_LABELS,
    ("release", "release_profile"): RELEASE_PROFILE_LABELS,
    ("motion", "motion_template"): MOTION_TEMPLATE_LABELS,
    ("motion", "motion_direction"): MOTION_DIRECTION_LABELS,
    ("targeting", "origin"): ORIGIN_LABELS,
    ("targeting", "target_mode"): TARGET_MODE_LABELS,
    ("targeting", "direction_mode"): DIRECTION_MODE_LABELS,
}


def get_label_space(path: tuple[str, ...]) -> list[str]:
    return LABEL_SPACES[path]


def get_nested(mapping: dict[str, Any] | None, path: tuple[str, str]) -> Any:
    if mapping is None:
        return None
    section = mapping.get(path[0])
    if section is None:
        return None
    return section.get(path[1])


def set_nested(mapping: dict[str, Any], path: tuple[str, str], value: Any) -> None:
    mapping.setdefault(path[0], {})[path[1]] = value


def validate_runtime_b(runtime_b: dict[str, Any] | None, *, allow_none: bool = False) -> None:
    if runtime_b is None:
        if allow_none:
            return
        raise ValueError("runtime_b cannot be None")
    if runtime_b.get("subject_kind") not in SUBJECT_KINDS:
        raise ValueError(f"invalid subject_kind: {runtime_b.get('subject_kind')}")
    for path, labels in LABEL_SPACES.items():
        if path == ("status",):
            continue
        value = get_nested(runtime_b, path) if len(path) == 2 else runtime_b.get(path[0])
        if value is None:
            continue
        if value not in labels:
            raise ValueError(f"invalid value for {path}: {value}")
    reaction_mask = get_nested(runtime_b, ("subject", "reaction_mask"))
    if reaction_mask is not None:
        if not isinstance(reaction_mask, list):
            raise ValueError("reaction_mask must be a list")
        unknown = [item for item in reaction_mask if item not in REACTION_MASK_LABELS]
        if unknown:
            raise ValueError(f"invalid reaction_mask items: {unknown}")
    for path in BINNED_SLOT_FIELDS:
        value = get_nested(runtime_b, path)
        if value is None:
            continue
        if value not in VALUE_BINS_7:
            raise ValueError(f"invalid bin value for {path}: {value}")
    for path in STYLE_FIELDS:
        value = get_nested(runtime_b, path)
        if value is None:
            continue
        if value not in STYLE_BINS:
            raise ValueError(f"invalid style value for {path}: {value}")
@dataclass(frozen=True)
class SlotSpec:
    path: tuple[str, str]
    labels: list[str]


CATEGORICAL_SPECS = [
    SlotSpec(("subject", "color"), COLOR_LABELS),
    SlotSpec(("subject", "state"), STATE_LABELS),
    SlotSpec(("subject", "reaction_kind"), REACTION_KIND_LABELS),
    SlotSpec(("subject", "reaction_direction"), REACTION_DIRECTION_LABELS),
    SlotSpec(("release", "release_profile"), RELEASE_PROFILE_LABELS),
    SlotSpec(("motion", "motion_template"), MOTION_TEMPLATE_LABELS),
    SlotSpec(("motion", "motion_direction"), MOTION_DIRECTION_LABELS),
    SlotSpec(("targeting", "origin"), ORIGIN_LABELS),
    SlotSpec(("targeting", "target_mode"), TARGET_MODE_LABELS),
    SlotSpec(("targeting", "direction_mode"), DIRECTION_MODE_LABELS),
]

BINNED_SPECS = [SlotSpec(path, VALUE_BINS_7) for path in BINNED_SLOT_FIELDS]
STYLE_SPECS = [SlotSpec(path, STYLE_BINS) for path in STYLE_FIELDS]
