from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, IntFlag
from typing import Any


class SimKind(str, Enum):
    EMPTY = "empty"
    PLATFORM = "platform"
    POWDER = "powder"
    LIQUID = "liquid"
    GAS = "gas"
    FIRE = "fire"
    MOLTEN = "molten"


class ReactionKind(str, Enum):
    NONE = "none"
    HEAT_SOURCE = "heat_source"
    CORROSIVE = "corrosive"
    TOXIC = "toxic"
    FLAMMABLE = "flammable"


class LifetimeMode(str, Enum):
    NONE = "none"
    DECAY_WITH_AGE = "decay_with_age"


class CellFlag(IntFlag):
    NONE = 0
    FIXPOINT = 1 << 0


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
    sim_kind: SimKind
    density: float
    hardness: float
    friction: float
    viscosity: float
    thermal_conductivity: float
    heat_capacity: float
    support_bearing: bool
    support_transmission: bool
    base_temperature: float
    ignite_temperature: float | None = None
    melt_temperature: float | None = None
    freeze_temperature: float | None = None
    boil_temperature: float | None = None
    decompose_temperature: float | None = None
    integrity_decay_from_heat: float = 0.0
    reaction_kind: ReactionKind = ReactionKind.NONE
    reaction_strength: float = 0.0
    lifetime_mode: LifetimeMode = LifetimeMode.NONE
    render_color: tuple[int, int, int] = (255, 255, 255)


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

    def copy(self) -> CellState:
        return replace(self)

    @property
    def is_empty(self) -> bool:
        return self.family_id == "empty"


def empty_cell() -> CellState:
    return CellState()
