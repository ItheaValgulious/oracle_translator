"""Spell System: Model Socket -> Magic Socket expander + catalog + SLM adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.engine.demo_app import DEFAULT_TICK_RATE_HZ
from src.game import config as cfg


# ── Level -> value mappings ──
AMOUNT_LEVEL = {
    "very_small": 1, "small": 2, "mid": 3, "mid_high": 4,
    "large": 5, "very_large": 6,
}
TEMP_LEVEL = {
    "very_low": -40.0, "low": 0.0, "room": 20.0,
    "high": 300.0, "very_high": 600.0,
}
DENSITY_LEVEL = {
    "very_low": 0.2, "low": 0.5, "mid": 1.0,
    "high": 1.3, "very_high": 2.8,
}
SPEED_LEVEL = {
    "very_slow": 0.1, "slow": 0.3, "mid": 0.5,
    "high": 0.8, "very_high": 1.0,
}
RELEASE_SPEED_LEVEL = {
    "very_slow": 0.5, "slow": 1.0, "mid": 2.0,
    "high": 4.0, "very_high": 8.0,
}
SPREAD_LEVEL = {
    "none": 0, "very_low": 1, "low": 2, "mid": 3,
    "mid_low": 2.5, "high": 5,
}
FORCE_LEVEL = {
    "very_weak": 0.5, "weak": 1.0, "mid": 2.0,
    "mid_high": 3.0, "high": 4.0, "very_high": 6.0,
}
CARRIER_LEVEL = {
    "none": 0.0, "low": 2.0, "mid": 4.0,
    "high": 6.0, "very_high": 10.0,
}


# ── Subject expansion table ──
SUBJECT_EXPANSION_TABLE: dict[str, dict[str, Any]] = {
    "fire":       {"family_id": "fire",         "variant_id": "fire",              "color": "orange",      "state": "gas",    "density": "low",       "temperature": "high",       "amount": "mid_high", "hardness": None,   "friction": None,  "viscosity": None},
    "water":      {"family_id": "water",        "variant_id": "water",             "color": "blue",        "state": "liquid", "density": "mid",       "temperature": "room",       "amount": "mid",      "hardness": None,   "friction": None,  "viscosity": "low"},
    "ice":        {"family_id": "water",        "variant_id": "ice",               "color": "cyan",        "state": "solid",  "density": "mid",       "temperature": "very_low",   "amount": "mid",      "hardness": "mid",   "friction": "mid",  "viscosity": None},
    "steam":      {"family_id": "water",        "variant_id": "steam",             "color": "white",       "state": "gas",    "density": "very_low",  "temperature": "high",       "amount": "mid_high", "hardness": None,   "friction": None,  "viscosity": None},
    "granite":    {"family_id": "stone",        "variant_id": "stone_platform",    "color": "grey",        "state": "solid",  "density": "very_high", "temperature": "room",       "amount": "mid",      "hardness": "very_high", "friction": "high", "viscosity": None},
    "obsidian":   {"family_id": "obsidian",     "variant_id": "obsidian_platform",   "color": "dark",        "state": "solid",  "density": "very_high", "temperature": "room",       "amount": "mid",      "hardness": "very_high", "friction": "high", "viscosity": None},
    "sand":       {"family_id": "sand",         "variant_id": "sand_powder",         "color": "tan",         "state": "solid",  "density": "mid",       "temperature": "room",       "amount": "mid",      "hardness": "very_low","friction": "mid",  "viscosity": None},
    "acid":       {"family_id": "magic_acid",   "variant_id": "magic_acid_liquid",   "color": "green",       "state": "liquid", "density": "high",      "temperature": "room",       "amount": "mid",      "hardness": None,   "friction": None,  "viscosity": "mid"},
    "poison_slurry": {"family_id": "poison",    "variant_id": "poison_liquid",       "color": "toxic_green", "state": "liquid", "density": "mid",       "temperature": "room",       "amount": "mid",      "hardness": None,   "friction": None,  "viscosity": "mid"},
    "tar":        {"family_id": "tar",          "variant_id": "tar_liquid",          "color": "black",       "state": "liquid", "density": "high",      "temperature": "room",       "amount": "mid",      "hardness": None,   "friction": None,  "viscosity": "very_high"},
    "explosive_slurry": {"family_id": "stone",  "variant_id": "magma",               "color": "red",         "state": "liquid", "density": "very_high", "temperature": "very_high",  "amount": "mid",      "hardness": None,   "friction": None,  "viscosity": "very_high"},
    "grass":      {"family_id": "grass",        "variant_id": "grass_platform",      "color": "green",       "state": "solid",  "density": "very_low",  "temperature": "room",       "amount": "mid",      "hardness": "low",   "friction": "mid",  "viscosity": None},
    "wood":       {"family_id": "wood",         "variant_id": "wood_platform",       "color": "brown",       "state": "solid",  "density": "low",       "temperature": "room",       "amount": "mid",      "hardness": "mid",   "friction": "mid",  "viscosity": None},
    "glass":      {"family_id": "glass",        "variant_id": "glass_platform",      "color": "transparent", "state": "solid",  "density": "high",      "temperature": "room",       "amount": "mid",      "hardness": "mid",   "friction": "mid",  "viscosity": None},
    "iron":       {"family_id": "iron",         "variant_id": "iron_platform",         "color": "metallic",    "state": "solid",  "density": "very_high", "temperature": "room",       "amount": "mid",      "hardness": "very_high","friction": "high","viscosity": None},
    "earth":      {"family_id": "stone",        "variant_id": "stone_falling",       "color": "brown_grey",  "state": "solid",  "density": "high",      "temperature": "room",       "amount": "mid_high","hardness": "mid",   "friction": "mid",  "viscosity": None},
    "quicksilver":{"family_id": "iron",         "variant_id": "molten_iron",         "color": "silver",      "state": "liquid", "density": "very_high", "temperature": "high",       "amount": "mid",      "hardness": None,   "friction": None,  "viscosity": "mid"},
    "unknown":    {"family_id": "stone",        "variant_id": "stone_platform",      "color": "grey",        "state": "solid",  "density": "mid",       "temperature": "room",       "amount": "mid",      "hardness": "mid",   "friction": "mid",  "viscosity": None},
}


# ── Reaction expansion table ──
REACTION_EXPANSION_TABLE: dict[str, dict[str, Any]] = {
    "none":    {"convert_mode": "none",  "reaction_speed": None,      "reaction_mask": [],                           "reaction_direction": None,   "generation_limit": 0},
    "burn":    {"convert_mode": "self",  "reaction_speed": "high",    "reaction_mask": ["living", "terrain"],        "reaction_direction": "up",    "generation_limit": 2},
    "corrode": {"convert_mode": "empty","reaction_speed": "mid",     "reaction_mask": ["terrain", "living"],        "reaction_direction": "forward","generation_limit": 2},
    "freeze":  {"convert_mode": "self",  "reaction_speed": "mid",     "reaction_mask": ["living", "terrain", "water"], "reaction_direction": "down",  "generation_limit": 2},
    "poison":  {"convert_mode": "self",  "reaction_speed": "slow",    "reaction_mask": ["living"],                     "reaction_direction": "forward","generation_limit": 3},
    "grow":    {"convert_mode": "self",  "reaction_speed": "mid",     "reaction_mask": ["terrain"],                   "reaction_direction": "up",    "generation_limit": 4},
}


# ── Release expansion table ──
RELEASE_EXPANSION_TABLE: dict[str, dict[str, Any]] = {
    "spray":  {"release_profile": "stream", "release_speed": "high",    "release_spread": "mid_low"},
    "appear": {"release_profile": "burst",  "release_speed": "very_high","release_spread": "low"},
}


# ── Motion expansion table ──
MOTION_EXPANSION_TABLE: dict[str, dict[str, Any]] = {
    "none":      {"force_strength": "none",     "carrier_velocity": "none"},
    "fixed":     {"force_strength": "none",     "carrier_velocity": "none"},
    "flow":      {"force_strength": "mid_high", "carrier_velocity": "high"},
    "vortex":    {"force_strength": "mid",      "carrier_velocity": "mid"},
    "rotation":  {"force_strength": "mid",      "carrier_velocity": "mid"},
    "vibration": {"force_strength": "mid",      "carrier_velocity": "low"},
}


@dataclass
class MagicSocket:
    """Fully expanded spell parameters for the engine."""

    subject_family: str
    subject_variant: str
    subject_temperature: float
    brush_radius: int
    reaction_template: str
    reaction_convert_mode: str
    reaction_speed: float
    reaction_generation_limit: int
    release_profile: str
    release_speed: float
    release_spread: float
    force_strength: float
    carrier_velocity: float
    direction: tuple[float, float]
    origin: tuple[float, float]
    powerness: float


@dataclass
class ActiveStream:
    """Tracks a stream-type spell injecting cells over multiple ticks."""

    magic: MagicSocket
    remaining_ticks: int
    ticks_total: int


def _scale_amount(base_key: str, powerness: float) -> int:
    base = AMOUNT_LEVEL.get(base_key, 3)
    return max(1, int(base * (0.5 + powerness * 0.5)))


def _scale_speed(base_key: str, powerness: float) -> float:
    base = SPEED_LEVEL.get(base_key, 0.5)
    return base * (0.3 + powerness * 0.7)


def _scale_force(base_key: str, powerness: float) -> float:
    base = FORCE_LEVEL.get(base_key, 2.0)
    return base * (0.3 + powerness * 0.7)


def _scale_release_speed(base_key: str, powerness: float) -> float:
    base = RELEASE_SPEED_LEVEL.get(base_key, 2.0)
    return base * (0.3 + powerness * 0.7)


def _scale_spread(base_key: str, powerness: float) -> float:
    base = SPREAD_LEVEL.get(base_key, 2.0)
    return base * (0.5 + powerness * 0.5)


def _direction_vector(motion_direction: str, origin_type: str, target_type: str, facing_right: bool) -> tuple[float, float]:
    """Map motion direction + origin + target to a direction vector."""
    facing_x = 1.0 if facing_right else -1.0
    dirs: dict[str, tuple[float, float]] = {
        "forward": (facing_x, 0.0),
        "backward": (-facing_x, 0.0),
        "up": (0.0, -1.0),
        "down": (0.0, 1.0),
        "self": (0.0, 0.0),
        "target": (facing_x, 0.0),
        "front_up": (facing_x, -0.5),
        "front_down": (facing_x, 0.5),
    }
    vec = dirs.get(motion_direction, (0.0, 0.0))
    length = math.hypot(vec[0], vec[1])
    if length > 0:
        return (vec[0] / length, vec[1] / length)
    return (0.0, 0.0)


def _origin_offset(origin_type: str, hero_x: float, hero_y: float, facing_right: bool) -> tuple[float, float]:
    """Map origin type to a world position relative to the hero."""
    facing = 1.0 if facing_right else -1.0
    offsets: dict[str, tuple[float, float]] = {
        "self": (0.0, 0.0),
        "back": (-facing * cfg.HERO_WIDTH, 0.0),
        "front_up": (facing * cfg.HERO_WIDTH, -cfg.HERO_HEIGHT * 0.5),
        "front_down": (facing * cfg.HERO_WIDTH, 0.0),
    }
    ox, oy = offsets.get(origin_type, (0.0, 0.0))
    return (hero_x + ox, hero_y + oy)


def expand_model_socket(model_socket: dict[str, Any], hero_x: float = 0.0, hero_y: float = 0.0, facing_right: bool = True) -> MagicSocket:
    """Expand a Model Socket dict into a Magic Socket."""
    powerness = float(model_socket.get("politeness", 0.5))

    # Subject
    subject = SUBJECT_EXPANSION_TABLE[model_socket["material"]].copy()
    brush_radius = _scale_amount(subject.get("amount", "mid"), powerness)
    subject_temp = TEMP_LEVEL.get(subject.get("temperature", "room"), 20.0)

    # Reaction
    reaction = REACTION_EXPANSION_TABLE[model_socket["reaction"]].copy()
    reaction_speed = _scale_speed(reaction.get("reaction_speed", "mid") or "mid", powerness)

    # Release
    release = RELEASE_EXPANSION_TABLE[model_socket["release"]].copy()
    release_speed = _scale_release_speed(release.get("release_speed", "mid") or "mid", powerness)
    release_spread = _scale_spread(release.get("release_spread", "mid") or "mid", powerness)

    # Motion
    motion_template = MOTION_EXPANSION_TABLE[model_socket["motion"]].copy()
    force_strength = _scale_force(motion_template.get("force_strength", "mid") or "mid", powerness)
    carrier_velocity = CARRIER_LEVEL.get(motion_template.get("carrier_velocity", "mid") or "mid", 0.0)
    carrier_velocity *= (0.3 + powerness * 0.7)

    # Direction and origin
    motion_direction = model_socket.get("motion_direction", "forward")
    origin_type = model_socket.get("origin", "self")
    target_type = model_socket.get("target", "none")
    direction = _direction_vector(motion_direction, origin_type, target_type, facing_right)
    origin = _origin_offset(origin_type, hero_x, hero_y, facing_right)

    return MagicSocket(
        subject_family=subject["family_id"],
        subject_variant=subject["variant_id"],
        subject_temperature=subject_temp,
        brush_radius=brush_radius,
        reaction_template=model_socket["reaction"],
        reaction_convert_mode=reaction["convert_mode"],
        reaction_speed=reaction_speed,
        reaction_generation_limit=int(reaction.get("generation_limit", 0)),
        release_profile=release["release_profile"],
        release_speed=release_speed,
        release_spread=release_spread,
        force_strength=force_strength,
        carrier_velocity=carrier_velocity,
        direction=direction,
        origin=origin,
        powerness=powerness,
    )


# ── v1 Spell Catalog (Model Socket JSON configs) ──
SPELL_CATALOG: list[dict[str, Any]] = [
    {"name": "Fireball",       "material": "fire",         "reaction": "burn",    "release": "spray",  "motion": "flow",      "motion_direction": "forward",   "origin": "self",       "target": "enemy", "politeness": 0.5, "mp": 10},
    {"name": "Water Wall",     "material": "water",        "reaction": "none",    "release": "appear", "motion": "fixed",     "motion_direction": "forward",   "origin": "front_down", "target": "none",  "politeness": 0.5, "mp": 10},
    {"name": "Ice Shield",     "material": "ice",          "reaction": "freeze",  "release": "appear", "motion": "fixed",     "motion_direction": "self",      "origin": "self",       "target": "none",  "politeness": 0.6, "mp": 8},
    {"name": "Sandstorm",      "material": "sand",         "reaction": "none",    "release": "spray",  "motion": "flow",      "motion_direction": "forward",   "origin": "self",       "target": "none",  "politeness": 0.3, "mp": 5},
    {"name": "Magic Acid",     "material": "acid",         "reaction": "corrode", "release": "spray",  "motion": "flow",      "motion_direction": "forward",   "origin": "self",       "target": "enemy", "politeness": 0.7, "mp": 15},
    {"name": "Stone Platform", "material": "granite",      "reaction": "none",    "release": "appear", "motion": "fixed",     "motion_direction": "down",      "origin": "front_down", "target": "none",  "politeness": 0.5, "mp": 12},
    {"name": "Oil Pool",       "material": "tar",          "reaction": "burn",    "release": "appear", "motion": "none",      "motion_direction": "down",      "origin": "front_down", "target": "none",  "politeness": 0.3, "mp": 8},
    {"name": "Wooden Wall",    "material": "wood",         "reaction": "grow",    "release": "appear", "motion": "fixed",     "motion_direction": "forward",   "origin": "front_down", "target": "none",  "politeness": 0.5, "mp": 8},
]


def execute_magic_socket(magic: MagicSocket, world: Any, registry: Any) -> None | ActiveStream:
    """Execute a fully expanded Magic Socket by injecting cells into the world.

    Uses world.paint_world() for correct world→local coordinate conversion.
    Returns an ActiveStream for stream-type spells, None for burst-type.
    """
    origin_x, origin_y = magic.origin
    overrides: dict[str, Any] = {}
    if magic.subject_temperature != 20.0:
        overrides["temperature"] = magic.subject_temperature
    if magic.carrier_velocity > 0.0 or magic.force_strength > 0.0:
        overrides["vel_x"] = magic.direction[0] * magic.carrier_velocity
        overrides["vel_y"] = magic.direction[1] * magic.carrier_velocity

    # Attach reaction parameters to injected cells
    if magic.reaction_convert_mode and magic.reaction_convert_mode != "none":
        overrides["spell_convert_mode"] = magic.reaction_convert_mode
    if magic.reaction_generation_limit > 0:
        overrides["spell_generation_limit"] = magic.reaction_generation_limit
    reaction = REACTION_EXPANSION_TABLE.get(magic.reaction_template, REACTION_EXPANSION_TABLE["none"])
    mask = reaction.get("reaction_mask", [])
    if mask:
        overrides["spell_damage_mask"] = tuple(mask)

    if magic.release_profile == "burst":
        world.paint_world(
            int(origin_x), int(origin_y), magic.brush_radius,
            magic.subject_family, magic.subject_variant,
            overrides=overrides,
        )
        return None

    # Stream: return an ActiveStream for multi-frame injection
    release_speed = magic.release_speed
    stream_duration = max(0.5, magic.brush_radius / release_speed)
    from src.engine.demo_app import DEFAULT_TICK_RATE_HZ
    ticks_total = max(1, int(stream_duration * DEFAULT_TICK_RATE_HZ))
    return ActiveStream(magic=magic, remaining_ticks=ticks_total, ticks_total=ticks_total)


def inject_stream_tick(stream: ActiveStream, world: Any, registry: Any, hero_x: float, hero_y: float, facing_right: bool) -> None:
    """Inject one tick's worth of cells for a stream spell.

    For flow-type motion, origin follows the hero each tick.
    For fixed-type motion, origin stays at the initial cast point.
    """
    magic = stream.magic
    if magic.force_strength > 0.0 or magic.carrier_velocity > 0.0:
        origin_x = hero_x + (1.0 if facing_right else -1.0) * cfg.HERO_WIDTH
        origin_y = hero_y + cfg.HERO_HEIGHT * 0.5
    else:
        origin_x, origin_y = magic.origin

    # Per-tick radius: small burst each frame, accumulates over stream duration
    per_tick_radius = max(1, int(math.ceil(magic.brush_radius / stream.ticks_total)))

    overrides: dict[str, Any] = {}
    if magic.subject_temperature != 20.0:
        overrides["temperature"] = magic.subject_temperature
    overrides["vel_x"] = magic.direction[0] * magic.carrier_velocity
    overrides["vel_y"] = magic.direction[1] * magic.carrier_velocity

    if magic.reaction_convert_mode and magic.reaction_convert_mode != "none":
        overrides["spell_convert_mode"] = magic.reaction_convert_mode
    if magic.reaction_generation_limit > 0:
        overrides["spell_generation_limit"] = magic.reaction_generation_limit
    reaction = REACTION_EXPANSION_TABLE.get(magic.reaction_template, REACTION_EXPANSION_TABLE["none"])
    mask = reaction.get("reaction_mask", [])
    if mask:
        overrides["spell_damage_mask"] = tuple(mask)

    world.paint_world(
        int(origin_x), int(origin_y), per_tick_radius,
        magic.subject_family, magic.subject_variant,
        overrides=overrides,
    )
    stream.remaining_ticks -= 1


# ── SLM Adapter ──
SLM_AVAILABLE = False

try:
    from src.slm.model_socket_schema import normalize_model_socket, validate_model_socket
    SLM_AVAILABLE = True
except ImportError:
    pass


def slm_to_model_socket(text: str) -> dict[str, Any] | None:
    """Convert raw text input to a Model Socket via SLM inference.

    Returns None if SLM is not available or inference fails.
    """
    if not SLM_AVAILABLE:
        return None
    return None