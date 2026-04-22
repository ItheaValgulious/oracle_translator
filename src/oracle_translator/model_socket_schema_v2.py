from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "v2_model_socket_template_first"

SUBJECT_KINDS = ["summon_material"]

MATERIAL_TEMPLATE_LABELS = [
    "granite",
    "obsidian",
    "earth",
    "sand",
    "water",
    "ice",
    "steam",
    "fire",
    "light",
    "wind",
    "lightning",
    "acid",
    "poison_slurry",
    "tar",
    "explosive_slurry",
    "grass",
    "wood",
    "glass",
    "iron",
    "quicksilver",
]

REACTION_TEMPLATE_LABELS = ["none", "burn", "corrode", "freeze", "poison", "grow"]
RELEASE_TEMPLATE_LABELS = ["spray", "appear"]
MOTION_TEMPLATE_LABELS = ["none", "fixed", "flow", "vortex", "rotation", "vibration"]
MOTION_DIRECTION_LABELS = ["forward", "backward", "up", "down", "target", "self", "front_up", "front_down"]
ORIGIN_LABELS = ["self", "back", "front_up", "front_down"]
TARGET_LABELS = ["self", "enemy", "none"]

LABEL_SPACES: dict[tuple[str, str], list[str]] = {
    ("subject", "material_template"): MATERIAL_TEMPLATE_LABELS,
    ("reaction", "reaction_template"): REACTION_TEMPLATE_LABELS,
    ("release", "release_template"): RELEASE_TEMPLATE_LABELS,
    ("motion", "motion_template"): MOTION_TEMPLATE_LABELS,
    ("motion", "motion_direction"): MOTION_DIRECTION_LABELS,
    ("motion", "origin"): ORIGIN_LABELS,
    ("motion", "target"): TARGET_LABELS,
}


@dataclass(frozen=True)
class SlotSpec:
    path: tuple[str, str]
    labels: list[str]


CATEGORICAL_SPECS = [SlotSpec(path, labels) for path, labels in LABEL_SPACES.items()]


def get_nested(mapping: dict[str, Any] | None, path: tuple[str, str]) -> Any:
    if mapping is None:
        return None
    section = mapping.get(path[0])
    if section is None:
        return None
    return section.get(path[1])


def set_nested(mapping: dict[str, Any], path: tuple[str, str], value: Any) -> None:
    mapping.setdefault(path[0], {})[path[1]] = value


SUBJECT_KIND_NORMALIZATION = {"manipulate_material": "summon_material"}
MATERIAL_TEMPLATE_NORMALIZATION = {
    "stone": "granite",
    "rock": "granite",
    "acid_liquid": "acid",
    "poison_smoke": "poison_slurry",
    "sticky_tar": "tar",
    "liquid_metal": "quicksilver",
    "thorn_vines": "wood",
    "radiant_dust": "light",
    "lava_pool": "explosive_slurry",
    "paper_flood": "wood",
    "laser_like_light": "light",
}
REACTION_TEMPLATE_NORMALIZATION = {
    "self": "grow",
    "empty": "corrode",
}
RELEASE_TEMPLATE_NORMALIZATION = {
    "burst": "appear",
    "stream": "spray",
    "pool": "appear",
    "beam": "appear",
    "spray": "spray",
    "appear": "appear",
}
TARGET_NORMALIZATION = {"to_enemy": "enemy", "enemy_target": "enemy", "to_self": "self"}
ORIGIN_NORMALIZATION = {"front_down": "front_down", "front_up": "front_up", "front": "front_up"}


def normalize_model_socket(model_socket: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(model_socket)
    subject_kind = normalized.get("subject_kind")
    if isinstance(subject_kind, str):
        normalized["subject_kind"] = SUBJECT_KIND_NORMALIZATION.get(subject_kind.strip().lower(), subject_kind)
    material_template = get_nested(normalized, ("subject", "material_template"))
    if isinstance(material_template, str):
        lowered = material_template.strip().lower()
        set_nested(normalized, ("subject", "material_template"), MATERIAL_TEMPLATE_NORMALIZATION.get(lowered, lowered))
    reaction_template = get_nested(normalized, ("reaction", "reaction_template"))
    if isinstance(reaction_template, str):
        lowered = reaction_template.strip().lower()
        set_nested(normalized, ("reaction", "reaction_template"), REACTION_TEMPLATE_NORMALIZATION.get(lowered, lowered))
    release_template = get_nested(normalized, ("release", "release_template"))
    if isinstance(release_template, str):
        lowered = release_template.strip().lower()
        set_nested(normalized, ("release", "release_template"), RELEASE_TEMPLATE_NORMALIZATION.get(lowered, lowered))
    motion_direction = get_nested(normalized, ("motion", "motion_direction"))
    if isinstance(motion_direction, str):
        set_nested(normalized, ("motion", "motion_direction"), motion_direction.strip().lower())
    motion_template = get_nested(normalized, ("motion", "motion_template"))
    if isinstance(motion_template, str):
        set_nested(normalized, ("motion", "motion_template"), motion_template.strip().lower())
    origin = get_nested(normalized, ("motion", "origin"))
    if isinstance(origin, str):
        lowered = origin.strip().lower()
        set_nested(normalized, ("motion", "origin"), ORIGIN_NORMALIZATION.get(lowered, lowered))
    target = get_nested(normalized, ("motion", "target"))
    if isinstance(target, str):
        lowered = target.strip().lower()
        set_nested(normalized, ("motion", "target"), TARGET_NORMALIZATION.get(lowered, lowered))
    powerness = get_nested(normalized, ("expression", "powerness"))
    if isinstance(powerness, str):
        try:
            set_nested(normalized, ("expression", "powerness"), float(powerness))
        except ValueError:
            pass
    expression = normalized.get("expression")
    if isinstance(expression, dict):
        expression.pop("politeness", None)
        expression.pop("style_axes", None)
    release = normalized.get("release")
    if isinstance(release, dict):
        release.pop("release_speed", None)
        release.pop("release_spread", None)
        release.pop("release_duration", None)
    return normalized


def validate_model_socket(model_socket: dict[str, Any] | None, *, allow_none: bool = False) -> None:
    if model_socket is None:
        if allow_none:
            return
        raise ValueError("model_socket cannot be None")
    if model_socket.get("subject_kind") not in SUBJECT_KINDS:
        raise ValueError(f"invalid subject_kind: {model_socket.get('subject_kind')}")
    for path, labels in LABEL_SPACES.items():
        value = get_nested(model_socket, path)
        if value is None:
            raise ValueError(f"missing mandatory field: {path}")
        if value not in labels:
            raise ValueError(f"invalid value for {path}: {value}")
    powerness = get_nested(model_socket, ("expression", "powerness"))
    if powerness is None:
        raise ValueError("missing expression.powerness")
    if not isinstance(powerness, (int, float)):
        raise ValueError("expression.powerness must be numeric")
    if float(powerness) < 0.0 or float(powerness) > 1.0:
        raise ValueError(f"expression.powerness out of range: {powerness}")
