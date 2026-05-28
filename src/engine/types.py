from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, IntFlag
from typing import Any


class MatterState(str, Enum):
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"


class LifetimeMode(str, Enum):
    NONE = "none"
    DECAY_WITH_AGE = "decay_with_age"


class CellFlag(IntFlag):
    NONE = 0
    FIXPOINT = 1 << 0


# Bitmask values for damage_mask — determines what targets a reaction can affect.
DAMAGE_MASK_TERRAIN = 1   # affects support-bearing cells
DAMAGE_MASK_LIVING = 2    # affects entity placeholders
DAMAGE_MASK_WATER = 4     # affects water-family cells


@dataclass(frozen=True)
class PhaseRule:
    source_variant: str
    target_variant: str
    target_family_id: str | None = None
    above_temperature: float | None = None
    below_temperature: float | None = None

    def matches(self, variant_id: str, temperature: float) -> bool:
        if self.source_variant != variant_id:
            return False
        if self.above_temperature is not None and temperature < self.above_temperature:
            return False
        if self.below_temperature is not None and temperature > self.below_temperature:
            return False
        return True


@dataclass(frozen=True)
class VariantDef:
    variant_id: str
    matter_state: MatterState
    density: float
    hardness: float
    friction: float
    viscosity: float
    thermal_conductivity: float
    heat_capacity: float
    support_bearing: bool
    support_transmission: bool
    base_temperature: float
    reaction_min_temperature: float = 0.0
    reaction_max_temperature: float = 0.0
    melt_temperature: float | None = None
    freeze_temperature: float | None = None
    boil_temperature: float | None = None
    integrity_decay_from_heat: float = 0.0
    reaction_strength: float = 0.0
    reaction_energy: float = 0.0
    reaction_preserves_self: bool = True
    lifetime_mode: LifetimeMode = LifetimeMode.NONE
    ignite_target_family_id: str | None = None
    ignite_target_variant_id: str | None = None
    damage_mask: tuple[str, ...] = ()
    mobility: float = 1.0
    pressure_response: float = 1.0
    gravity_scale: float = 0.0
    buoyancy_scale: float = 0.0
    thermal_motion_scale: float = 0.0
    wind_coupling: float = 0.0
    wind_vertical_factor: float = 0.0
    downward_blocked_diagonal_fallback: bool = False
    velocity_decay: float = 0.92
    liquid_contact_heat_exchange_multiplier: float = 1.0
    same_variant_heat_exchange_multiplier: float = 1.0
    render_color: tuple[int, int, int] = (255, 255, 255)

    @property
    def damage_mask_bitmask(self) -> int:
        result = 0
        for target in self.damage_mask:
            if target == "terrain":
                result |= DAMAGE_MASK_TERRAIN
            elif target == "living":
                result |= DAMAGE_MASK_LIVING
            elif target == "water":
                result |= DAMAGE_MASK_WATER
        return result


@dataclass(frozen=True)
class MaterialFamily:
    family_id: str
    name: str
    default_variant: str
    collapse_target: str | None
    variants: dict[str, VariantDef]
    phase_map: tuple[PhaseRule, ...] = ()
    reaction_profile: dict[str, Any] = field(default_factory=dict)
    render_profile: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterialRegistry:
    families: dict[str, MaterialFamily]
    variants: dict[tuple[str, str], VariantDef]

    def family(self, family_id: str) -> MaterialFamily:
        return self.families[family_id]

    def variant(self, family_id: str, variant_id: str) -> VariantDef:
        return self.variants[(family_id, variant_id)]


@dataclass
class CellState:
    family_id: str = "empty"
    variant_id: str = "empty"
    vel_x: float = 0.0
    vel_y: float = 0.0
    blocked_x: float = 0.0
    blocked_y: float = 0.0
    temperature: float = 20.0
    support_value: float = 0.0
    integrity: float = 1.0
    generation: int = 0
    age: float = 0.0
    flags: CellFlag = CellFlag.NONE
    # Per-cell spell overrides (empty = use variant defaults)
    spell_damage_mask: tuple[str, ...] = ()
    spell_convert_mode: str = ""
    spell_generation_limit: int = -1

    def copy(self) -> CellState:
        return replace(self)

    @property
    def is_empty(self) -> bool:
        return self.family_id == "empty"


def empty_cell() -> CellState:
    return CellState()