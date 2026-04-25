from __future__ import annotations

from array import array
from dataclasses import dataclass
import struct

import moderngl

from .grid import Grid
from .render import DebugViewMode
from .support import SUPPORT_FAILURE_THRESHOLD, SUPPORT_SOURCE_VALUE
from .types import CellFlag, CellState, LifetimeMode, MaterialRegistry, MatterState, MotionMode, ReactionKind


WORKGROUP_SIZE = 8
LIQUID_RELAXATION_PASSES = 0
FORCE_WAVE_DECAY = 0.82
INT_MAX_VALUE = 2_147_483_647
PHASE_FLAG_ABOVE = 1
PHASE_FLAG_BELOW = 1 << 1
PHASE_FLAG_COOL_TO_SOLID = 1 << 2

MATTER_STATE_CODES = {
    MatterState.SOLID: 0,
    MatterState.LIQUID: 1,
    MatterState.GAS: 2,
}

MOTION_MODE_CODES = {
    MotionMode.STATIC: 0,
    MotionMode.POWDER: 1,
    MotionMode.FLUID: 2,
}

REACTION_KIND_CODES = {
    ReactionKind.NONE: 0,
    ReactionKind.HEAT_SOURCE: 1,
    ReactionKind.CORROSIVE: 2,
    ReactionKind.TOXIC: 3,
    ReactionKind.FLAMMABLE: 4,
}

LIFETIME_MODE_CODES = {
    LifetimeMode.NONE: 0,
    LifetimeMode.DECAY_WITH_AGE: 1,
}

VARIANT_PACK_FORMAT = "<8i20f"
FAMILY_PACK_FORMAT = "<4i4f"
PHASE_PACK_FORMAT = "<4i4f"


class ComputeBackendUnavailable(RuntimeError):
    """Raised when the active OpenGL context cannot run the compute backend."""


@dataclass(frozen=True)
class GpuMaterialTables:
    family_ids: tuple[str, ...]
    variant_keys: tuple[tuple[str, str], ...]
    family_index_by_id: dict[str, int]
    variant_index_by_key: dict[tuple[str, str], int]
    family_buffer_data: bytes
    phase_buffer_data: bytes
    variant_buffer_data: bytes
    empty_variant_index: int

    @classmethod
    def from_registry(cls, registry: MaterialRegistry) -> GpuMaterialTables:
        family_ids = tuple(registry.families.keys())
        family_index_by_id = {family_id: index for index, family_id in enumerate(family_ids)}

        variant_keys = tuple(
            (family_id, variant_id)
            for family_id, family in registry.families.items()
            for variant_id in family.variants.keys()
        )
        variant_index_by_key = {key: index for index, key in enumerate(variant_keys)}

        phase_ranges: dict[str, tuple[int, int]] = {}
        phase_buffer = bytearray()
        for family_id, family in registry.families.items():
            offset = len(phase_buffer) // struct.calcsize(PHASE_PACK_FORMAT)
            for rule in family.phase_map:
                source_variant = registry.variant(family_id, rule.source_variant)
                target_family_id = rule.target_family_id or family_id
                flags = 0
                if rule.above_temperature is not None:
                    flags |= PHASE_FLAG_ABOVE
                if rule.below_temperature is not None:
                    flags |= PHASE_FLAG_BELOW
                if source_variant.variant_id in {"magma", "molten_glass", "molten_iron"} and rule.below_temperature is not None:
                    flags |= PHASE_FLAG_COOL_TO_SOLID
                phase_buffer.extend(
                    struct.pack(
                        PHASE_PACK_FORMAT,
                        variant_index_by_key[(family_id, rule.source_variant)],
                        variant_index_by_key[(target_family_id, rule.target_variant)],
                        flags,
                        0,
                        float(rule.above_temperature or 0.0),
                        float(rule.below_temperature or 0.0),
                        0.0,
                        0.0,
                    )
                )
            count = len(phase_buffer) // struct.calcsize(PHASE_PACK_FORMAT) - offset
            phase_ranges[family_id] = (offset, count)

        family_buffer = bytearray()
        for family_id in family_ids:
            family = registry.family(family_id)
            phase_offset, phase_count = phase_ranges[family_id]
            family_buffer.extend(
                struct.pack(
                    FAMILY_PACK_FORMAT,
                    variant_index_by_key[(family_id, family.default_variant)],
                    variant_index_by_key[(family_id, family.collapse_target)] if family.collapse_target else -1,
                    phase_offset,
                    phase_count,
                    float(family.reaction_profile.get("max_age", 0.0)),
                    0.0,
                    0.0,
                    0.0,
                )
            )

        variant_buffer = bytearray()
        for family_id, variant_id in variant_keys:
            variant = registry.variant(family_id, variant_id)
            red, green, blue = variant.render_color
            variant_buffer.extend(
                struct.pack(
                    VARIANT_PACK_FORMAT,
                    family_index_by_id[family_id],
                    REACTION_KIND_CODES[variant.reaction_kind],
                    LIFETIME_MODE_CODES[variant.lifetime_mode],
                    MOTION_MODE_CODES[variant.motion_mode],
                    int(variant.support_bearing),
                    int(variant.support_transmission),
                    int(variant.reaction_preserves_self),
                    MATTER_STATE_CODES[variant.matter_state],
                    float(variant.density),
                    float(variant.hardness),
                    float(variant.friction),
                    float(variant.viscosity),
                    float(variant.thermal_conductivity),
                    float(variant.heat_capacity),
                    float(variant.base_temperature),
                    float(variant.reaction_strength),
                    float(variant.ignite_temperature or 0.0),
                    float(variant.melt_temperature or 0.0),
                    float(variant.freeze_temperature or 0.0),
                    float(variant.boil_temperature or 0.0),
                    float(variant.decompose_temperature or 0.0),
                    float(variant.integrity_decay_from_heat),
                    float(red),
                    float(green),
                    float(blue),
                    float(variant.reaction_energy),
                    0.0,
                    0.0,
                )
            )

        return cls(
            family_ids=family_ids,
            variant_keys=variant_keys,
            family_index_by_id=family_index_by_id,
            variant_index_by_key=variant_index_by_key,
            family_buffer_data=bytes(family_buffer),
            phase_buffer_data=bytes(phase_buffer),
            variant_buffer_data=bytes(variant_buffer),
            empty_variant_index=variant_index_by_key[("empty", "empty")],
        )


def pack_grid_state(grid: Grid, tables: GpuMaterialTables) -> tuple[bytes, bytes, bytes]:
    state_int = array("i")
    state_vec = array("f")
    state_misc = array("f")
    for cell in grid.cells:
        state_int.extend(
            (
                tables.variant_index_by_key[(cell.family_id, cell.variant_id)],
                cell.generation,
                int(cell.flags),
                0,
            )
        )
        state_vec.extend((cell.vel_x, cell.vel_y, cell.blocked_x, cell.blocked_y))
        state_misc.extend((cell.temperature, cell.support_value, cell.integrity, cell.age))
    return (state_int.tobytes(), state_vec.tobytes(), state_misc.tobytes())


def unpack_grid_state(
    width: int,
    height: int,
    tables: GpuMaterialTables,
    state_int_data: bytes,
    state_vec_data: bytes,
    state_misc_data: bytes,
) -> Grid:
    state_int = array("i")
    state_vec = array("f")
    state_misc = array("f")
    state_int.frombytes(state_int_data)
    state_vec.frombytes(state_vec_data)
    state_misc.frombytes(state_misc_data)

    grid = Grid(width=width, height=height)
    for index in range(width * height):
        int_offset = index * 4
        vec_offset = index * 4
        family_id, variant_id = tables.variant_keys[state_int[int_offset]]
        grid.cells[index] = CellState(
            family_id=family_id,
            variant_id=variant_id,
            generation=state_int[int_offset + 1],
            flags=CellFlag(state_int[int_offset + 2]),
            vel_x=state_vec[vec_offset],
            vel_y=state_vec[vec_offset + 1],
            blocked_x=state_vec[vec_offset + 2],
            blocked_y=state_vec[vec_offset + 3],
            temperature=state_misc[vec_offset],
            support_value=state_misc[vec_offset + 1],
            integrity=state_misc[vec_offset + 2],
            age=state_misc[vec_offset + 3],
        )
    return grid


def _dispatch_groups(width: int, height: int) -> tuple[int, int]:
    return (
        (width + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE,
        (height + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE,
    )


def _set_uniform_if_present(program: moderngl.ComputeShader, name: str, value: object) -> None:
    try:
        program[name].value = value
    except KeyError:
        return


def _build_common_glsl(tables: GpuMaterialTables) -> str:
    return f"""
#version 430

#define CELL_FLAG_FIXPOINT {int(CellFlag.FIXPOINT)}
#define EMPTY_VARIANT_INDEX {tables.empty_variant_index}
#define INT_MAX_VALUE {INT_MAX_VALUE}

#define MATTER_STATE_SOLID {MATTER_STATE_CODES[MatterState.SOLID]}
#define MATTER_STATE_LIQUID {MATTER_STATE_CODES[MatterState.LIQUID]}
#define MATTER_STATE_GAS {MATTER_STATE_CODES[MatterState.GAS]}

#define MOTION_MODE_STATIC {MOTION_MODE_CODES[MotionMode.STATIC]}
#define MOTION_MODE_POWDER {MOTION_MODE_CODES[MotionMode.POWDER]}
#define MOTION_MODE_FLUID {MOTION_MODE_CODES[MotionMode.FLUID]}

#define REACTION_NONE {REACTION_KIND_CODES[ReactionKind.NONE]}
#define REACTION_HEAT_SOURCE {REACTION_KIND_CODES[ReactionKind.HEAT_SOURCE]}
#define REACTION_CORROSIVE {REACTION_KIND_CODES[ReactionKind.CORROSIVE]}
#define REACTION_TOXIC {REACTION_KIND_CODES[ReactionKind.TOXIC]}
#define REACTION_FLAMMABLE {REACTION_KIND_CODES[ReactionKind.FLAMMABLE]}

#define FIRE_FAMILY_INDEX {tables.family_index_by_id["fire"]}

#define SUPPORT_TIMEOUT_SECONDS 10.0
#define SUPPORT_SOURCE_VALUE SUPPORT_TIMEOUT_SECONDS
#define SUPPORT_FAILURE_THRESHOLD {SUPPORT_FAILURE_THRESHOLD}
#define INTEGRITY_DECAY_UNSUPPORTED 0.18

#define AIR_PRESSURE 1.0
#define PRESSURE_RELAXATION 0.22
#define PRESSURE_FORCE_SCALE 0.55
#define FORCE_WAVE_DECAY 0.82
#define LIQUID_LATERAL_PRESSURE_BOOST 1.35
#define LIQUID_BLOCKED_LATERAL_BOOST 2.6
#define LIQUID_SURFACE_SIDEFLOW_BONUS 0.8
#define LIQUID_PRESSURE_HEAD_BOOST 0.25
#define LIQUID_PRESSURE_HEAD_BOOST_CAP 2.5
#define LIQUID_RELAXATION_HEAD_THRESHOLD 1.5
#define LIQUID_RELAXATION_NEIGHBOR_THRESHOLD 2
#define LIQUID_VERTICAL_PRESSURE_SCALE 0.8
#define GAS_PRESSURE_FORCE_SCALE 0.65
#define MAX_HEAT_EXCHANGE 120.0
#define THERMAL_CONDUCTION_RATE 8.0

#define GRAVITY_ACCELERATION 1.2
#define GAS_BUOYANCY_SCALE 10.0
#define VELOCITY_DECAY 0.92
#define STATIC_VELOCITY_DECAY 0.96
#define BLOCKED_DECAY 0.72
#define LIQUID_RANDOM_GAIN 0.012
#define LIQUID_RANDOM_VERTICAL_FACTOR 0.18
#define GAS_RANDOM_GAIN 0.42
#define GAS_RANDOM_VERTICAL_FACTOR 0.85
#define DIRECTION_JITTER_GAIN 0.06
#define KELVIN_OFFSET 273.15
#define MIN_THERMAL_KELVIN 80.0
#define EMPTY_MOTION_TEMPERATURE_THRESHOLD 0.5
#define GAS_THERMAL_TEMPERATURE_SPAN 120.0
#define LIQUID_THERMAL_TEMPERATURE_SPAN 400.0
#define GAS_RANDOM_FLOOR_FACTOR 0.35
#define LIQUID_RANDOM_FLOOR_FACTOR 0.15

#define VIEW_MODE_MATERIAL 0
#define VIEW_MODE_TEMPERATURE 1
#define VIEW_MODE_PRESSURE 2

#define PHASE_FLAG_ABOVE {PHASE_FLAG_ABOVE}
#define PHASE_FLAG_BELOW {PHASE_FLAG_BELOW}
#define PHASE_FLAG_COOL_TO_SOLID {PHASE_FLAG_COOL_TO_SOLID}

struct VariantData {{
    ivec4 ints0;
    ivec4 ints1;
    vec4 floats0;
    vec4 floats1;
    vec4 floats2;
    vec4 floats3;
    vec4 floats4;
}};

struct FamilyData {{
    ivec4 ints0;
    vec4 floats0;
}};

struct PhaseRuleData {{
    ivec4 ints0;
    vec4 floats0;
}};

layout(std430, binding = 0) readonly buffer VariantTable {{
    VariantData variant_table[];
}};

layout(std430, binding = 1) readonly buffer FamilyTable {{
    FamilyData family_table[];
}};

layout(std430, binding = 2) readonly buffer PhaseTable {{
    PhaseRuleData phase_table[];
}};

layout(r32f, binding = 12) uniform readonly image2D pressure_tex;

uniform ivec2 grid_size;
uniform float dt;
uniform int step_index;
uniform int liquids_only;
uniform int liquid_brownian_enabled;
uniform int blocked_impulse_enabled;

const ivec2 NEIGHBORS_8[8] = ivec2[](
    ivec2(-1, -1),
    ivec2(0, -1),
    ivec2(1, -1),
    ivec2(-1, 0),
    ivec2(1, 0),
    ivec2(-1, 1),
    ivec2(0, 1),
    ivec2(1, 1)
);

const ivec2 HEAT_NEIGHBORS[4] = ivec2[](
    ivec2(-1, 0),
    ivec2(1, 0),
    ivec2(0, -1),
    ivec2(0, 1)
);

bool in_bounds(ivec2 coord) {{
    return coord.x >= 0 && coord.y >= 0 && coord.x < grid_size.x && coord.y < grid_size.y;
}}

int linear_index(ivec2 coord) {{
    return coord.y * grid_size.x + coord.x;
}}

int family_index_for_variant(int variant_index) {{
    return variant_table[variant_index].ints0.x;
}}

int reaction_kind_for_variant(int variant_index) {{
    return variant_table[variant_index].ints0.y;
}}

int lifetime_mode_for_variant(int variant_index) {{
    return variant_table[variant_index].ints0.z;
}}

int motion_mode_for_variant(int variant_index) {{
    return variant_table[variant_index].ints0.w;
}}

bool support_bearing_for_variant(int variant_index) {{
    return variant_table[variant_index].ints1.x != 0;
}}

bool support_transmission_for_variant(int variant_index) {{
    return variant_table[variant_index].ints1.y != 0;
}}

int matter_state_for_variant(int variant_index) {{
    return variant_table[variant_index].ints1.w;
}}

float thermal_conductivity_for_variant(int variant_index) {{
    return variant_table[variant_index].floats1.x;
}}

float heat_capacity_for_variant(int variant_index) {{
    return variant_table[variant_index].floats1.y;
}}

float base_temperature_for_variant(int variant_index) {{
    return variant_table[variant_index].floats1.z;
}}

float reaction_strength_for_variant(int variant_index) {{
    return variant_table[variant_index].floats1.w;
}}

float density_for_variant(int variant_index) {{
    return variant_table[variant_index].floats0.x;
}}

float ignite_temperature_for_variant(int variant_index) {{
    return variant_table[variant_index].floats2.x;
}}

float melt_temperature_for_variant(int variant_index) {{
    return variant_table[variant_index].floats2.y;
}}

float decompose_temperature_for_variant(int variant_index) {{
    return variant_table[variant_index].floats3.x;
}}

float integrity_decay_from_heat_for_variant(int variant_index) {{
    return variant_table[variant_index].floats3.y;
}}

bool reaction_preserves_self_for_variant(int variant_index) {{
    return variant_table[variant_index].ints1.z != 0;
}}

float reaction_energy_for_variant(int variant_index) {{
    return variant_table[variant_index].floats4.y;
}}

vec3 render_color_for_variant(int variant_index) {{
    return vec3(
        variant_table[variant_index].floats3.z,
        variant_table[variant_index].floats3.w,
        variant_table[variant_index].floats4.x
    );
}}

ivec4 empty_cell_int() {{
    return ivec4(EMPTY_VARIANT_INDEX, 0, 0, 0);
}}

vec4 empty_cell_vec() {{
    return vec4(0.0, 0.0, 0.0, 0.0);
}}

vec4 empty_cell_misc() {{
    return vec4(base_temperature_for_variant(EMPTY_VARIANT_INDEX), 0.0, 1.0, 0.0);
}}

float clamp_channel(float value) {{
    return clamp(round(value), 0.0, 255.0);
}}

bool can_move_variant(int variant_index) {{
    return motion_mode_for_variant(variant_index) != MOTION_MODE_STATIC;
}}

bool can_move_in_current_pass(int variant_index) {{
    if (!can_move_variant(variant_index)) {{
        return false;
    }}
    return true;
}}

vec2 normalized_or_default(vec2 value, vec2 fallback_value) {{
    float length_value = length(value);
    if (length_value == 0.0) {{
        float fallback_length = length(fallback_value);
        if (fallback_length == 0.0) {{
            return vec2(0.0, 0.0);
        }}
        return fallback_value / fallback_length;
    }}
    return value / length_value;
}}

float direction_score(ivec2 direction, vec2 desired) {{
    vec2 direction_f = normalized_or_default(vec2(direction), vec2(0.0, 1.0));
    vec2 desired_f = normalized_or_default(desired, vec2(0.0, 1.0));
    return dot(direction_f, desired_f);
}}

float axis_remaining_after_step(float desired_component, float realized_component) {{
    if (desired_component == 0.0) {{
        return 0.0;
    }}
    if (realized_component == 0.0 || desired_component * realized_component <= 0.0) {{
        return desired_component;
    }}
    return sign(desired_component) * max(0.0, abs(desired_component) - abs(realized_component));
}}

vec2 blocked_intent(vec2 blocked) {{
    return blocked_impulse_enabled != 0 ? blocked : vec2(0.0);
}}

float hash01(ivec2 coord, int salt) {{
    uint value = uint(step_index + 1) * 374761393u;
    value += uint(coord.x + 11) * 668265263u;
    value += uint(coord.y + 17) * 2147483647u;
    value += uint(salt + 23) * 1274126177u;
    value ^= value >> 13;
    value *= 1274126177u;
    return float(value) / 4294967295.0;
}}

float direction_jitter(ivec2 coord, ivec2 direction, float jitter_gain) {{
    ivec2 jitter_coord = coord + ivec2(direction.x * 17, direction.y * 31);
    return (hash01(jitter_coord, 97) - 0.5) * jitter_gain;
}}

int preferred_side(float desired_x, ivec2 coord, int salt) {{
    if (desired_x > 0.0) {{
        return 1;
    }}
    if (desired_x < 0.0) {{
        return -1;
    }}
    salt = salt;
    return ((coord.x + coord.y + step_index) & 1) == 0 ? -1 : 1;
}}

int state_rank(int matter_state) {{
    if (matter_state == MATTER_STATE_SOLID) {{
        return 3;
    }}
    if (matter_state == MATTER_STATE_LIQUID) {{
        return 2;
    }}
    return 1;
}}

bool target_is_lighter(int current_variant, int target_variant) {{
    if (target_variant == EMPTY_VARIANT_INDEX) {{
        return true;
    }}
    int current_rank = state_rank(matter_state_for_variant(current_variant));
    int target_rank = state_rank(matter_state_for_variant(target_variant));
    if (current_rank != target_rank) {{
        return target_rank < current_rank;
    }}
    return density_for_variant(target_variant) < density_for_variant(current_variant);
}}

bool is_gas_state(int matter_state) {{
    return matter_state == MATTER_STATE_GAS;
}}

float ambient_temperature() {{
    return base_temperature_for_variant(EMPTY_VARIANT_INDEX);
}}

float temperature_kelvin(float temperature) {{
    return max(MIN_THERMAL_KELVIN, temperature + KELVIN_OFFSET);
}}

float temperature_motion_factor(float temperature, float span) {{
    return clamp((temperature - ambient_temperature()) / max(span, 0.001), 0.0, 1.0);
}}

float effective_density_for_variant_and_temperature(int variant_index, float temperature) {{
    if (variant_index == EMPTY_VARIANT_INDEX || is_gas_state(matter_state_for_variant(variant_index))) {{
        return density_for_variant(variant_index) * temperature_kelvin(ambient_temperature()) / temperature_kelvin(temperature);
    }}
    return density_for_variant(variant_index);
}}

float thermal_random_gain_for_cell(int variant_index, float temperature) {{
    if (variant_index == EMPTY_VARIANT_INDEX) {{
        return GAS_RANDOM_GAIN * temperature_motion_factor(temperature, GAS_THERMAL_TEMPERATURE_SPAN);
    }}
    int matter_state = matter_state_for_variant(variant_index);
    if (matter_state == MATTER_STATE_GAS) {{
        float factor = temperature_motion_factor(temperature, GAS_THERMAL_TEMPERATURE_SPAN);
        return GAS_RANDOM_GAIN * (GAS_RANDOM_FLOOR_FACTOR + (1.0 - GAS_RANDOM_FLOOR_FACTOR) * factor);
    }}
    if (matter_state == MATTER_STATE_LIQUID && liquid_brownian_enabled != 0) {{
        float factor = temperature_motion_factor(temperature, LIQUID_THERMAL_TEMPERATURE_SPAN);
        return LIQUID_RANDOM_GAIN * (LIQUID_RANDOM_FLOOR_FACTOR + (1.0 - LIQUID_RANDOM_FLOOR_FACTOR) * factor);
    }}
    return 0.0;
}}

bool empty_can_move(vec4 cell_vec, vec4 cell_misc) {{
    return abs(cell_misc.x - ambient_temperature()) > EMPTY_MOTION_TEMPERATURE_THRESHOLD
        || length(cell_vec.xy) > 0.01
        || length(cell_vec.zw) > 0.01;
}}

vec2 pressure_force_at(ivec2 coord) {{
    float current_pressure = imageLoad(pressure_tex, coord).x;
    vec2 force = vec2(0.0);
    for (int index = 0; index < 4; index += 1) {{
        ivec2 direction = HEAT_NEIGHBORS[index];
        ivec2 neighbor_coord = coord + direction;
        if (!in_bounds(neighbor_coord)) {{
            continue;
        }}
        float neighbor_pressure = imageLoad(pressure_tex, neighbor_coord).x;
        force += vec2(direction) * (current_pressure - neighbor_pressure);
    }}
    return force;
}}

vec2 local_pressure_force_for_variant(int variant_index, ivec2 coord, bool downward_exchange_available) {{
    int matter_state = matter_state_for_variant(variant_index);
    vec2 pressure_force = pressure_force_at(coord);
    float current_pressure = imageLoad(pressure_tex, coord).x;
    float pressure_scale_x = PRESSURE_FORCE_SCALE;
    float pressure_scale_y = PRESSURE_FORCE_SCALE;
    if (matter_state == MATTER_STATE_LIQUID) {{
        pressure_scale_x *= LIQUID_LATERAL_PRESSURE_BOOST;
        pressure_scale_y *= LIQUID_VERTICAL_PRESSURE_SCALE;
        if (!downward_exchange_available) {{
            pressure_scale_x *= LIQUID_BLOCKED_LATERAL_BOOST;
            float pressure_head = max(0.0, current_pressure - AIR_PRESSURE);
            pressure_scale_x *= 1.0 + min(LIQUID_PRESSURE_HEAD_BOOST_CAP, sqrt(pressure_head) * LIQUID_PRESSURE_HEAD_BOOST);
        }}
    }} else if (matter_state == MATTER_STATE_GAS) {{
        pressure_scale_x *= GAS_PRESSURE_FORCE_SCALE;
        pressure_scale_y *= GAS_PRESSURE_FORCE_SCALE;
    }}

    return vec2(pressure_force.x * pressure_scale_x, pressure_force.y * pressure_scale_y);
}}

vec2 apply_forces_to_velocity(
    int variant_index,
    ivec2 coord,
    vec2 velocity,
    bool downward_exchange_available,
    vec2 propagated_force,
    float temperature
) {{
    int motion_mode = motion_mode_for_variant(variant_index);
    vec2 local_pressure_force = local_pressure_force_for_variant(variant_index, coord, downward_exchange_available);
    velocity += (local_pressure_force + propagated_force) * dt;
    if (motion_mode == MOTION_MODE_STATIC) {{
        return velocity;
    }}
    if (motion_mode == MOTION_MODE_POWDER) {{
        velocity.y += GRAVITY_ACCELERATION * dt;
        return velocity;
    }}
    int matter_state = matter_state_for_variant(variant_index);
    float random_gain = thermal_random_gain_for_cell(variant_index, temperature);
    velocity.x += (hash01(coord, 11) - 0.5) * random_gain;
    float vertical_factor = (variant_index == EMPTY_VARIANT_INDEX || is_gas_state(matter_state))
        ? GAS_RANDOM_VERTICAL_FACTOR
        : LIQUID_RANDOM_VERTICAL_FACTOR;
    velocity.y += (hash01(coord, 13) - 0.5) * random_gain * vertical_factor;
    if (variant_index == EMPTY_VARIANT_INDEX || is_gas_state(matter_state)) {{
        velocity.y += (
            effective_density_for_variant_and_temperature(variant_index, temperature)
            - density_for_variant(EMPTY_VARIANT_INDEX)
        ) * GAS_BUOYANCY_SCALE * dt;
    }} else {{
        velocity.y += GRAVITY_ACCELERATION * dt;
    }}
    return velocity;
}}

bool can_swap_with_target(int current_state, float current_density, int target_state, float target_density, float diffusion_roll, int surrounding_count) {{
    diffusion_roll = diffusion_roll;
    surrounding_count = surrounding_count;
    if (target_density == density_for_variant(EMPTY_VARIANT_INDEX) && target_state == MATTER_STATE_GAS) {{
        return true;
    }}
    if (state_rank(target_state) >= state_rank(current_state)) {{
        return false;
    }}
    if (current_state == target_state && target_density >= current_density) {{
        return false;
    }}
    return true;
}}

vec3 temperature_color(float temperature) {{
    if (temperature <= -40.0) {{
        return vec3(20.0, 28.0, 70.0);
    }}
    if (temperature <= 0.0) {{
        float factor = (temperature + 40.0) / 40.0;
        return mix(vec3(20.0, 28.0, 70.0), vec3(50.0, 120.0, 220.0), factor);
    }}
    if (temperature <= 20.0) {{
        float factor = temperature / 20.0;
        return mix(vec3(50.0, 120.0, 220.0), vec3(70.0, 180.0, 255.0), factor);
    }}
    if (temperature <= 100.0) {{
        float factor = (temperature - 20.0) / 80.0;
        return mix(vec3(70.0, 180.0, 255.0), vec3(90.0, 220.0, 140.0), factor);
    }}
    if (temperature <= 300.0) {{
        float factor = (temperature - 100.0) / 200.0;
        return mix(vec3(90.0, 220.0, 140.0), vec3(240.0, 220.0, 80.0), factor);
    }}
    if (temperature <= 800.0) {{
        float factor = (temperature - 300.0) / 500.0;
        return mix(vec3(240.0, 220.0, 80.0), vec3(255.0, 120.0, 40.0), factor);
    }}
    if (temperature <= 1400.0) {{
        float factor = (temperature - 800.0) / 600.0;
        return mix(vec3(255.0, 120.0, 40.0), vec3(255.0, 245.0, 235.0), factor);
    }}
    return vec3(255.0, 245.0, 235.0);
}}

vec3 pressure_color(float pressure) {{
    float excess_pressure = max(0.0, pressure - AIR_PRESSURE);
    float factor = clamp(log(1.0 + excess_pressure) / log(129.0), 0.0, 1.0);
    if (factor <= 0.15) {{
        float local_factor = factor / 0.15;
        return mix(vec3(8.0, 12.0, 28.0), vec3(25.0, 70.0, 150.0), local_factor);
    }}
    if (factor <= 0.35) {{
        float local_factor = (factor - 0.15) / 0.20;
        return mix(vec3(25.0, 70.0, 150.0), vec3(45.0, 150.0, 235.0), local_factor);
    }}
    if (factor <= 0.60) {{
        float local_factor = (factor - 0.35) / 0.25;
        return mix(vec3(45.0, 150.0, 235.0), vec3(100.0, 220.0, 170.0), local_factor);
    }}
    if (factor <= 0.82) {{
        float local_factor = (factor - 0.60) / 0.22;
        return mix(vec3(100.0, 220.0, 170.0), vec3(245.0, 210.0, 70.0), local_factor);
    }}
    float local_factor = (factor - 0.82) / 0.18;
    return mix(vec3(245.0, 210.0, 70.0), vec3(255.0, 245.0, 235.0), local_factor);
}}
"""


def _pressure_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 64, local_size_y = 1) in;

layout(rgba32i, binding = 3) uniform readonly iimage2D state_int_src;
layout(rgba32f, binding = 5) uniform readonly image2D state_misc_src;
layout(r32f, binding = 12) uniform readonly image2D pressure_prev_tex;
layout(r32f, binding = 13) uniform writeonly image2D pressure_next_tex;

void main() {
    int x = int(gl_GlobalInvocationID.x);
    if (x >= grid_size.x) {
        return;
    }

    float above_pressure = AIR_PRESSURE;
    for (int y = 0; y < grid_size.y; y += 1) {
        ivec2 coord = ivec2(x, y);
        int variant_index = imageLoad(state_int_src, coord).x;
        float previous_pressure = imageLoad(pressure_prev_tex, coord).x;
        float cell_temperature = imageLoad(state_misc_src, coord).x;
        float pressure = previous_pressure + (AIR_PRESSURE - previous_pressure) * PRESSURE_RELAXATION;
        if (variant_index == EMPTY_VARIANT_INDEX) {
            pressure = previous_pressure + (AIR_PRESSURE - previous_pressure) * PRESSURE_RELAXATION;
        } else if (matter_state_for_variant(variant_index) == MATTER_STATE_GAS) {
            pressure = AIR_PRESSURE + effective_density_for_variant_and_temperature(variant_index, cell_temperature);
        } else if (matter_state_for_variant(variant_index) == MATTER_STATE_LIQUID) {
            pressure = above_pressure + density_for_variant(variant_index);
        }
        imageStore(pressure_next_tex, coord, vec4(pressure, 0.0, 0.0, 0.0));
        above_pressure = pressure;
    }
}
"""


def _source_force_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32i, binding = 3) uniform readonly iimage2D state_int_src;
layout(r32f, binding = 12) uniform readonly image2D pressure_prev_tex;
layout(rg32f, binding = 13) uniform writeonly image2D force_tex;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }

    int variant_index = imageLoad(state_int_src, coord).x;
    if (variant_index == EMPTY_VARIANT_INDEX) {
        imageStore(force_tex, coord, vec4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    bool downward_exchange_available = false;
    ivec2 below_coord = coord + ivec2(0, 1);
    if (in_bounds(below_coord)) {
        int below_variant = imageLoad(state_int_src, below_coord).x;
        downward_exchange_available = target_is_lighter(variant_index, below_variant);
    }

    vec2 local_force = local_pressure_force_for_variant(variant_index, coord, downward_exchange_available);
    imageStore(force_tex, coord, vec4(local_force, 0.0, 0.0));
}
"""


def _force_wave_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32i, binding = 3) uniform readonly iimage2D state_int_src;
layout(rg32f, binding = 13) uniform readonly image2D source_force_tex;
layout(rg32f, binding = 14) uniform readonly image2D prev_source_force_tex;
layout(rg32f, binding = 15) uniform readonly image2D prev_wave_force_tex;
layout(rg32f, binding = 16) uniform writeonly image2D next_wave_force_tex;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }

    int variant_index = imageLoad(state_int_src, coord).x;
    if (variant_index == EMPTY_VARIANT_INDEX || matter_state_for_variant(variant_index) != MATTER_STATE_LIQUID) {
        imageStore(next_wave_force_tex, coord, vec4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    vec2 incoming = vec2(0.0);
    if (coord.x > 0) {
        ivec2 left_coord = coord + ivec2(-1, 0);
        int left_variant = imageLoad(state_int_src, left_coord).x;
        if (left_variant != EMPTY_VARIANT_INDEX && matter_state_for_variant(left_variant) == MATTER_STATE_LIQUID) {
            incoming.x += max(0.0, imageLoad(prev_wave_force_tex, left_coord).x) * FORCE_WAVE_DECAY;
        }
    }
    if (coord.x < grid_size.x - 1) {
        ivec2 right_coord = coord + ivec2(1, 0);
        int right_variant = imageLoad(state_int_src, right_coord).x;
        if (right_variant != EMPTY_VARIANT_INDEX && matter_state_for_variant(right_variant) == MATTER_STATE_LIQUID) {
            incoming.x += min(0.0, imageLoad(prev_wave_force_tex, right_coord).x) * FORCE_WAVE_DECAY;
        }
    }
    if (coord.y > 0) {
        ivec2 above_coord = coord + ivec2(0, -1);
        int above_variant = imageLoad(state_int_src, above_coord).x;
        if (above_variant != EMPTY_VARIANT_INDEX && matter_state_for_variant(above_variant) == MATTER_STATE_LIQUID) {
            incoming.y += max(0.0, imageLoad(prev_wave_force_tex, above_coord).y) * FORCE_WAVE_DECAY;
        }
    }
    if (coord.y < grid_size.y - 1) {
        ivec2 below_coord = coord + ivec2(0, 1);
        int below_variant = imageLoad(state_int_src, below_coord).x;
        if (below_variant != EMPTY_VARIANT_INDEX && matter_state_for_variant(below_variant) == MATTER_STATE_LIQUID) {
            incoming.y += min(0.0, imageLoad(prev_wave_force_tex, below_coord).y) * FORCE_WAVE_DECAY;
        }
    }

    vec2 source_force = imageLoad(source_force_tex, coord).xy;
    vec2 prev_source_force = imageLoad(prev_source_force_tex, coord).xy;
    vec2 delta_source = source_force - prev_source_force * FORCE_WAVE_DECAY;
    imageStore(next_wave_force_tex, coord, vec4(incoming + delta_source, 0.0, 0.0));
}
"""


def _support_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32i, binding = 3) uniform readonly iimage2D state_int_src;
layout(rgba32f, binding = 4) uniform readonly image2D state_vec_src;
layout(rgba32f, binding = 5) uniform readonly image2D state_misc_src;
layout(rgba32i, binding = 6) uniform writeonly iimage2D state_int_dst;
layout(rgba32f, binding = 7) uniform writeonly image2D state_vec_dst;
layout(rgba32f, binding = 8) uniform writeonly image2D state_misc_dst;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }

    ivec4 cell_int = imageLoad(state_int_src, coord);
    vec4 cell_vec = imageLoad(state_vec_src, coord);
    vec4 cell_misc = imageLoad(state_misc_src, coord);
    vec4 out_misc = cell_misc;
    int variant_index = cell_int.x;

    int incoming_generation = cell_int.y;
    if ((cell_int.z & CELL_FLAG_FIXPOINT) != 0) {
        incoming_generation = step_index + 1;
    } else if (support_transmission_for_variant(variant_index)) {
        for (int index = 0; index < 8; index += 1) {
            ivec2 neighbor_coord = coord + NEIGHBORS_8[index];
            if (!in_bounds(neighbor_coord)) {
                continue;
            }
            ivec4 neighbor_int = imageLoad(state_int_src, neighbor_coord);
            if ((neighbor_int.z & CELL_FLAG_FIXPOINT) != 0) {
                incoming_generation = max(incoming_generation, step_index + 1);
                continue;
            }
            if (support_transmission_for_variant(neighbor_int.x)) {
                incoming_generation = max(incoming_generation, neighbor_int.y);
            }
        }
    }

    if (incoming_generation > cell_int.y) {
        out_misc.y = SUPPORT_SOURCE_VALUE;
        cell_int.y = incoming_generation;
    } else if (support_transmission_for_variant(variant_index)) {
        out_misc.y = max(0.0, cell_misc.y - dt);
    } else {
        out_misc.y = 0.0;
        cell_int.y = 0;
    }

    if (support_bearing_for_variant(variant_index) && out_misc.y <= SUPPORT_FAILURE_THRESHOLD) {
        float unsupported_dt = cell_misc.y <= SUPPORT_FAILURE_THRESHOLD ? dt : max(0.0, dt - cell_misc.y);
        out_misc.z = max(0.0, out_misc.z - INTEGRITY_DECAY_UNSUPPORTED * unsupported_dt);
    }

    imageStore(state_int_dst, coord, cell_int);
    imageStore(state_vec_dst, coord, cell_vec);
    imageStore(state_misc_dst, coord, out_misc);
}
"""


def _motion_plan_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32i, binding = 3) uniform readonly iimage2D state_int_src;
layout(rgba32f, binding = 4) uniform readonly image2D state_vec_src;
layout(rgba32f, binding = 5) uniform readonly image2D state_misc_src;
layout(rg32f, binding = 15) uniform readonly image2D force_wave_tex;
layout(rgba32i, binding = 9) uniform writeonly iimage2D motion_plan_tex;

int liquid_neighbor_count_local(ivec2 coord) {
    int count = 0;
    for (int index = 0; index < 8; index += 1) {
        ivec2 neighbor_coord = coord + NEIGHBORS_8[index];
        if (!in_bounds(neighbor_coord)) {
            continue;
        }
        int neighbor_variant = imageLoad(state_int_src, neighbor_coord).x;
        if (neighbor_variant == EMPTY_VARIANT_INDEX) {
            continue;
        }
        if (matter_state_for_variant(neighbor_variant) == MATTER_STATE_LIQUID) {
            count += 1;
        }
    }
    return count;
}

bool downward_exchange_available_local(ivec2 coord, int variant_index) {
    ivec2 below_coord = coord + ivec2(0, 1);
    if (!in_bounds(below_coord)) {
        return false;
    }
    int below_variant = imageLoad(state_int_src, below_coord).x;
    return target_is_lighter(variant_index, below_variant);
}

bool motion_target_is_lighter(int current_variant, float current_temperature, int target_variant, float target_temperature) {
    if (current_variant == EMPTY_VARIANT_INDEX) {
        return target_variant == EMPTY_VARIANT_INDEX;
    }
    if (target_variant == EMPTY_VARIANT_INDEX) {
        return true;
    }
    int current_rank = state_rank(matter_state_for_variant(current_variant));
    int target_rank = state_rank(matter_state_for_variant(target_variant));
    if (current_rank != target_rank) {
        return target_rank < current_rank;
    }
    return effective_density_for_variant_and_temperature(target_variant, target_temperature)
        < effective_density_for_variant_and_temperature(current_variant, current_temperature);
}

int surface_outflow_direction_local(ivec2 coord, int variant_index) {
    if (matter_state_for_variant(variant_index) != MATTER_STATE_LIQUID) {
        return 0;
    }
    bool above_is_empty = coord.y > 0 && imageLoad(state_int_src, coord + ivec2(0, -1)).x == EMPTY_VARIANT_INDEX;
    bool downward_blocked = !downward_exchange_available_local(coord, variant_index);
    if (!above_is_empty && !downward_blocked) {
        return 0;
    }
    bool left_is_empty = coord.x > 0 && imageLoad(state_int_src, coord + ivec2(-1, 0)).x == EMPTY_VARIANT_INDEX;
    bool right_is_empty = coord.x < grid_size.x - 1 && imageLoad(state_int_src, coord + ivec2(1, 0)).x == EMPTY_VARIANT_INDEX;
    if (right_is_empty && !left_is_empty) {
        return 1;
    }
    if (left_is_empty && !right_is_empty) {
        return -1;
    }
    return 0;
}

ivec4 find_exchange_plan(ivec2 coord, int variant_index, float current_temperature, vec2 desired) {
    float best_score = -1000000.0;
    ivec2 best_direction = ivec2(0, 0);
    int best_target = -1;
    int best_swap = 0;
    int surface_outflow_direction = surface_outflow_direction_local(coord, variant_index);
    float jitter_gain = DIRECTION_JITTER_GAIN;
    if (length(desired) < 0.05) {
        jitter_gain = 0.0;
    } else if (matter_state_for_variant(variant_index) == MATTER_STATE_LIQUID && liquid_brownian_enabled == 0) {
        jitter_gain = 0.0;
    }
    for (int index = 0; index < 8; index += 1) {
        ivec2 direction = NEIGHBORS_8[index];
        ivec2 neighbor_coord = coord + direction;
        if (!in_bounds(neighbor_coord)) {
            continue;
        }
        int target_variant = imageLoad(state_int_src, neighbor_coord).x;
        float target_temperature = imageLoad(state_misc_src, neighbor_coord).x;
        if (!motion_target_is_lighter(variant_index, current_temperature, target_variant, target_temperature)) {
            continue;
        }
        float score = direction_score(direction, desired) + direction_jitter(coord, direction, jitter_gain);
        if (surface_outflow_direction != 0 && direction.x == surface_outflow_direction && direction.y == 0) {
            score += LIQUID_SURFACE_SIDEFLOW_BONUS;
        }
        if (score > best_score) {
            best_score = score;
            best_direction = direction;
            best_target = linear_index(neighbor_coord);
            best_swap = (variant_index != EMPTY_VARIANT_INDEX && target_variant == EMPTY_VARIANT_INDEX) ? 0 : 1;
        }
    }
    return best_target >= 0 ? ivec4(best_target, best_direction.x, best_direction.y, best_swap) : ivec4(-1, 0, 0, 0);
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }

    ivec4 cell_int = imageLoad(state_int_src, coord);
    vec4 cell_vec = imageLoad(state_vec_src, coord);
    vec4 cell_misc = imageLoad(state_misc_src, coord);
    int variant_index = cell_int.x;

    if (!can_move_in_current_pass(variant_index)) {
        imageStore(motion_plan_tex, coord, ivec4(-1, 0, 0, 0));
        return;
    }
    if (variant_index == EMPTY_VARIANT_INDEX && !empty_can_move(cell_vec, cell_misc)) {
        imageStore(motion_plan_tex, coord, ivec4(-1, 0, 0, 0));
        return;
    }
    if (liquids_only != 0) {
        if (matter_state_for_variant(variant_index) != MATTER_STATE_LIQUID) {
            imageStore(motion_plan_tex, coord, ivec4(-1, 0, 0, 0));
            return;
        }
        float pressure_head = imageLoad(pressure_tex, coord).x - AIR_PRESSURE;
        if (pressure_head < LIQUID_RELAXATION_HEAD_THRESHOLD || liquid_neighbor_count_local(coord) < LIQUID_RELAXATION_NEIGHBOR_THRESHOLD) {
            imageStore(motion_plan_tex, coord, ivec4(-1, 0, 0, 0));
            return;
        }
    }

    bool downward_exchange_available = downward_exchange_available_local(coord, variant_index);
    vec2 propagated_force = imageLoad(force_wave_tex, coord).xy;
    vec2 base_velocity = apply_forces_to_velocity(
        variant_index,
        coord,
        vec2(cell_vec.x, cell_vec.y),
        downward_exchange_available,
        propagated_force,
        cell_misc.x
    );
    vec2 desired = base_velocity + blocked_intent(vec2(cell_vec.z, cell_vec.w));
    ivec4 plan = find_exchange_plan(coord, variant_index, cell_misc.x, desired);
    imageStore(motion_plan_tex, coord, plan);
}
"""


def _clear_r32i_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(r32i, binding = 9) uniform writeonly iimage2D target_tex;
uniform int clear_value;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }
    imageStore(target_tex, coord, ivec4(clear_value, 0, 0, 0));
}
"""


def _motion_claim_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32i, binding = 9) uniform readonly iimage2D motion_plan_tex;
layout(r32i, binding = 10) uniform coherent iimage2D motion_claim_tex;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }

    ivec4 plan = imageLoad(motion_plan_tex, coord);
    if (plan.x < 0) {
        return;
    }

    ivec2 target_coord = ivec2(plan.x % grid_size.x, plan.x / grid_size.x);
    imageAtomicMin(motion_claim_tex, target_coord, linear_index(coord));
}
"""


def _motion_resolve_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32i, binding = 3) uniform readonly iimage2D state_int_src;
layout(rgba32f, binding = 4) uniform readonly image2D state_vec_src;
layout(rgba32f, binding = 5) uniform readonly image2D state_misc_src;
layout(rgba32i, binding = 6) uniform writeonly iimage2D state_int_dst;
layout(rgba32f, binding = 7) uniform writeonly image2D state_vec_dst;
layout(rgba32f, binding = 8) uniform writeonly image2D state_misc_dst;
layout(rgba32i, binding = 9) uniform readonly iimage2D motion_plan_tex;
layout(r32i, binding = 10) uniform readonly iimage2D motion_claim_tex;
layout(rg32f, binding = 15) uniform readonly image2D force_wave_tex;

int liquid_neighbor_count_local(ivec2 coord) {
    int count = 0;
    for (int index = 0; index < 8; index += 1) {
        ivec2 neighbor_coord = coord + NEIGHBORS_8[index];
        if (!in_bounds(neighbor_coord)) {
            continue;
        }
        int neighbor_variant = imageLoad(state_int_src, neighbor_coord).x;
        if (neighbor_variant == EMPTY_VARIANT_INDEX) {
            continue;
        }
        if (matter_state_for_variant(neighbor_variant) == MATTER_STATE_LIQUID) {
            count += 1;
        }
    }
    return count;
}

bool downward_exchange_available_local(ivec2 coord, int variant_index) {
    ivec2 below_coord = coord + ivec2(0, 1);
    if (!in_bounds(below_coord)) {
        return false;
    }
    int below_variant = imageLoad(state_int_src, below_coord).x;
    return target_is_lighter(variant_index, below_variant);
}

bool source_displaced_by_swap_local(ivec2 source_coord) {
    int incoming_winner = imageLoad(motion_claim_tex, source_coord).x;
    if (incoming_winner == INT_MAX_VALUE) {
        return false;
    }
    ivec2 incoming_coord = ivec2(incoming_winner % grid_size.x, incoming_winner / grid_size.x);
    ivec4 incoming_plan = imageLoad(motion_plan_tex, incoming_coord);
    return incoming_plan.x == linear_index(source_coord) && incoming_plan.w != 0;
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }

    int winner_index = imageLoad(motion_claim_tex, coord).x;
    if (winner_index != INT_MAX_VALUE) {
        ivec2 source_coord = ivec2(winner_index % grid_size.x, winner_index / grid_size.x);
        if (!source_displaced_by_swap_local(source_coord)) {
            ivec4 source_int = imageLoad(state_int_src, source_coord);
            vec4 source_vec = imageLoad(state_vec_src, source_coord);
            vec4 source_misc = imageLoad(state_misc_src, source_coord);
            ivec4 plan = imageLoad(motion_plan_tex, source_coord);
            int variant_index = source_int.x;
            bool downward_exchange_available = downward_exchange_available_local(source_coord, variant_index);
            vec2 propagated_force = imageLoad(force_wave_tex, source_coord).xy;
            vec2 base_velocity = apply_forces_to_velocity(
                variant_index,
                source_coord,
                vec2(source_vec.x, source_vec.y),
                downward_exchange_available,
                propagated_force,
                source_misc.x
            );
            vec2 desired = base_velocity + blocked_intent(vec2(source_vec.z, source_vec.w));
            vec4 out_vec = vec4(0.0);
            vec4 out_misc = source_misc;
            out_vec.x = base_velocity.x * 0.92;
            out_vec.y = base_velocity.y * 0.92;
            if (blocked_impulse_enabled != 0) {
                out_vec.z = axis_remaining_after_step(desired.x, float(plan.y));
                out_vec.w = axis_remaining_after_step(desired.y, float(plan.z));
            }
            imageStore(state_int_dst, coord, source_int);
            imageStore(state_vec_dst, coord, out_vec);
            imageStore(state_misc_dst, coord, out_misc);
            return;
        }
    }

    ivec4 cell_int = imageLoad(state_int_src, coord);
    vec4 cell_vec = imageLoad(state_vec_src, coord);
    vec4 cell_misc = imageLoad(state_misc_src, coord);
    ivec4 plan = imageLoad(motion_plan_tex, coord);
    bool moved_away = false;
    if (plan.x >= 0) {
        ivec2 target_coord = ivec2(plan.x % grid_size.x, plan.x / grid_size.x);
        moved_away = imageLoad(motion_claim_tex, target_coord).x == linear_index(coord);
    }

    if (moved_away) {
        if (moved_away && plan.w != 0) {
            ivec2 target_coord = ivec2(plan.x % grid_size.x, plan.x / grid_size.x);
            ivec4 displaced_int = imageLoad(state_int_src, target_coord);
            vec4 displaced_vec = imageLoad(state_vec_src, target_coord);
            vec4 displaced_misc = imageLoad(state_misc_src, target_coord);
            if (motion_mode_for_variant(displaced_int.x) == MOTION_MODE_STATIC) {
                bool displaced_downward_exchange = downward_exchange_available_local(target_coord, displaced_int.x);
                vec2 displaced_propagated_force = imageLoad(force_wave_tex, target_coord).xy;
                vec2 displaced_base_velocity = apply_forces_to_velocity(
                    displaced_int.x,
                    target_coord,
                    vec2(displaced_vec.x, displaced_vec.y),
                    displaced_downward_exchange,
                    displaced_propagated_force,
                    displaced_misc.x
                );
                displaced_vec.x = displaced_base_velocity.x * STATIC_VELOCITY_DECAY;
                displaced_vec.y = displaced_base_velocity.y * STATIC_VELOCITY_DECAY;
                if (blocked_impulse_enabled != 0) {
                    displaced_vec.z *= BLOCKED_DECAY;
                    displaced_vec.w *= BLOCKED_DECAY;
                } else {
                    displaced_vec.z = 0.0;
                    displaced_vec.w = 0.0;
                }
            } else {
                bool displaced_downward_exchange = downward_exchange_available_local(target_coord, displaced_int.x);
                vec2 displaced_propagated_force = imageLoad(force_wave_tex, target_coord).xy;
                vec2 displaced_base_velocity = apply_forces_to_velocity(
                    displaced_int.x,
                    target_coord,
                    vec2(displaced_vec.x, displaced_vec.y),
                    displaced_downward_exchange,
                    displaced_propagated_force,
                    displaced_misc.x
                );
                vec2 displaced_desired = displaced_base_velocity + blocked_intent(vec2(displaced_vec.z, displaced_vec.w));
                displaced_vec.x = displaced_base_velocity.x * VELOCITY_DECAY;
                displaced_vec.y = displaced_base_velocity.y * VELOCITY_DECAY;
                if (blocked_impulse_enabled != 0) {
                    displaced_vec.z = axis_remaining_after_step(displaced_desired.x, float(-plan.y));
                    displaced_vec.w = axis_remaining_after_step(displaced_desired.y, float(-plan.z));
                } else {
                    displaced_vec.z = 0.0;
                    displaced_vec.w = 0.0;
                }
            }
            imageStore(state_int_dst, coord, displaced_int);
            imageStore(state_vec_dst, coord, displaced_vec);
            imageStore(state_misc_dst, coord, displaced_misc);
        } else {
            ivec2 target_coord = ivec2(plan.x % grid_size.x, plan.x / grid_size.x);
            vec4 target_misc = imageLoad(state_misc_src, target_coord);
            vec4 empty_misc = empty_cell_misc();
            empty_misc.x = max(cell_misc.x, target_misc.x);
            imageStore(state_int_dst, coord, empty_cell_int());
            imageStore(state_vec_dst, coord, empty_cell_vec());
            imageStore(state_misc_dst, coord, empty_misc);
        }
        return;
    }

    int variant_index = cell_int.x;
    vec4 out_vec = vec4(0.0);
    if (liquids_only != 0) {
        if (matter_state_for_variant(variant_index) != MATTER_STATE_LIQUID) {
            imageStore(state_int_dst, coord, cell_int);
            imageStore(state_vec_dst, coord, cell_vec);
            imageStore(state_misc_dst, coord, cell_misc);
            return;
        }
        float pressure_head = imageLoad(pressure_tex, coord).x - AIR_PRESSURE;
        if (pressure_head < LIQUID_RELAXATION_HEAD_THRESHOLD || liquid_neighbor_count_local(coord) < LIQUID_RELAXATION_NEIGHBOR_THRESHOLD) {
            imageStore(state_int_dst, coord, cell_int);
            imageStore(state_vec_dst, coord, cell_vec);
            imageStore(state_misc_dst, coord, cell_misc);
            return;
        }
    }

    if (!can_move_variant(variant_index)) {
        bool downward_exchange_available = downward_exchange_available_local(coord, variant_index);
        vec2 propagated_force = imageLoad(force_wave_tex, coord).xy;
        vec2 base_velocity = apply_forces_to_velocity(
            variant_index,
            coord,
            vec2(cell_vec.x, cell_vec.y),
            downward_exchange_available,
            propagated_force,
            cell_misc.x
        );
        out_vec.x = base_velocity.x * STATIC_VELOCITY_DECAY;
        out_vec.y = base_velocity.y * STATIC_VELOCITY_DECAY;
        if (blocked_impulse_enabled != 0) {
            out_vec.z = cell_vec.z * BLOCKED_DECAY;
            out_vec.w = cell_vec.w * BLOCKED_DECAY;
        }
        imageStore(state_int_dst, coord, cell_int);
        imageStore(state_vec_dst, coord, out_vec);
        imageStore(state_misc_dst, coord, cell_misc);
        return;
    }

    bool downward_exchange_available = downward_exchange_available_local(coord, variant_index);
    vec2 propagated_force = imageLoad(force_wave_tex, coord).xy;
    vec2 base_velocity = apply_forces_to_velocity(
        variant_index,
        coord,
        vec2(cell_vec.x, cell_vec.y),
        downward_exchange_available,
        propagated_force,
        cell_misc.x
    );
    vec2 desired = base_velocity + blocked_intent(vec2(cell_vec.z, cell_vec.w));
    out_vec.x = base_velocity.x * 0.92;
    out_vec.y = base_velocity.y * 0.92;
    if (blocked_impulse_enabled != 0) {
        out_vec.z = desired.x * BLOCKED_DECAY;
        out_vec.w = desired.y * BLOCKED_DECAY;
    }

    imageStore(state_int_dst, coord, cell_int);
    imageStore(state_vec_dst, coord, out_vec);
    imageStore(state_misc_dst, coord, cell_misc);
}
"""


def _thermal_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32i, binding = 3) uniform readonly iimage2D state_int_src;
layout(rgba32f, binding = 4) uniform readonly image2D state_vec_src;
layout(rgba32f, binding = 5) uniform readonly image2D state_misc_src;
layout(rgba32i, binding = 6) uniform writeonly iimage2D state_int_dst;
layout(rgba32f, binding = 7) uniform writeonly image2D state_vec_dst;
layout(rgba32f, binding = 8) uniform writeonly image2D state_misc_dst;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }

    ivec4 cell_int = imageLoad(state_int_src, coord);
    vec4 cell_vec = imageLoad(state_vec_src, coord);
    vec4 cell_misc = imageLoad(state_misc_src, coord);
    int variant_index = cell_int.x;

    float next_temperature = cell_misc.x;
    for (int index = 0; index < 4; index += 1) {
        ivec2 neighbor_coord = coord + HEAT_NEIGHBORS[index];
        if (!in_bounds(neighbor_coord)) {
            continue;
        }
        ivec4 neighbor_int = imageLoad(state_int_src, neighbor_coord);
        vec4 neighbor_misc = imageLoad(state_misc_src, neighbor_coord);
        float conductivity = (thermal_conductivity_for_variant(variant_index) + thermal_conductivity_for_variant(neighbor_int.x)) * 0.5;
        float capacity = max((heat_capacity_for_variant(variant_index) + heat_capacity_for_variant(neighbor_int.x)) * 0.5, 0.001);
        float delta = (neighbor_misc.x - cell_misc.x) * conductivity * THERMAL_CONDUCTION_RATE * dt / capacity;
        next_temperature += clamp(delta, -MAX_HEAT_EXCHANGE, MAX_HEAT_EXCHANGE);
    }

    vec4 out_misc = cell_misc;
    out_misc.x = next_temperature;
    imageStore(state_int_dst, coord, cell_int);
    imageStore(state_vec_dst, coord, cell_vec);
    imageStore(state_misc_dst, coord, out_misc);
}
"""


def _phase_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32i, binding = 3) uniform readonly iimage2D state_int_src;
layout(rgba32f, binding = 4) uniform readonly image2D state_vec_src;
layout(rgba32f, binding = 5) uniform readonly image2D state_misc_src;
layout(rgba32i, binding = 6) uniform writeonly iimage2D state_int_dst;
layout(rgba32f, binding = 7) uniform writeonly image2D state_vec_dst;
layout(rgba32f, binding = 8) uniform writeonly image2D state_misc_dst;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }

    ivec4 cell_int = imageLoad(state_int_src, coord);
    vec4 cell_vec = imageLoad(state_vec_src, coord);
    vec4 cell_misc = imageLoad(state_misc_src, coord);
    ivec4 out_int = cell_int;
    vec4 out_misc = cell_misc;

    int family_index = family_index_for_variant(cell_int.x);
    FamilyData family = family_table[family_index];
    for (int offset = 0; offset < family.ints0.w; offset += 1) {
        PhaseRuleData rule = phase_table[family.ints0.z + offset];
        if (rule.ints0.x != cell_int.x) {
            continue;
        }
        if ((rule.ints0.z & PHASE_FLAG_ABOVE) != 0 && cell_misc.x < rule.floats0.x) {
            continue;
        }
        if ((rule.ints0.z & PHASE_FLAG_BELOW) != 0 && cell_misc.x > rule.floats0.y) {
            continue;
        }

        int target_variant = rule.ints0.y;
        if ((rule.ints0.z & PHASE_FLAG_COOL_TO_SOLID) != 0) {
            target_variant = cell_misc.y > SUPPORT_FAILURE_THRESHOLD
                ? family.ints0.x
                : (family.ints0.y >= 0 ? family.ints0.y : family.ints0.x);
        }

        out_int.x = target_variant;
        out_misc.x = cell_misc.x;
        out_misc.w = 0.0;
        if (!support_bearing_for_variant(target_variant)) {
            out_int.z &= ~CELL_FLAG_FIXPOINT;
            out_int.y = 0;
            out_misc.y = 0.0;
        }
        break;
    }

    imageStore(state_int_dst, coord, out_int);
    imageStore(state_vec_dst, coord, cell_vec);
    imageStore(state_misc_dst, coord, out_misc);
}
"""


def _reaction_plan_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32i, binding = 3) uniform readonly iimage2D state_int_src;
layout(rgba32f, binding = 5) uniform readonly image2D state_misc_src;
layout(r32i, binding = 9) uniform coherent iimage2D reaction_claim_tex;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }

    ivec4 cell_int = imageLoad(state_int_src, coord);
    vec4 cell_misc = imageLoad(state_misc_src, coord);
    int variant_index = cell_int.x;
    int family_index = family_index_for_variant(variant_index);

    if (family_index != %d) {
        return;
    }
    if (cell_misc.x < decompose_temperature_for_variant(variant_index)) {
        return;
    }

    for (int index = 0; index < 8; index += 1) {
        ivec2 target_coord = coord + NEIGHBORS_8[index];
        if (!in_bounds(target_coord)) {
            continue;
        }
        if (imageLoad(state_int_src, target_coord).x != EMPTY_VARIANT_INDEX) {
            continue;
        }
        imageAtomicMin(reaction_claim_tex, target_coord, linear_index(coord));
        break;
    }
}
""" % tables.family_index_by_id["poison"]


def _reaction_resolve_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32i, binding = 3) uniform readonly iimage2D state_int_src;
layout(rgba32f, binding = 4) uniform readonly image2D state_vec_src;
layout(rgba32f, binding = 5) uniform readonly image2D state_misc_src;
layout(rgba32i, binding = 6) uniform writeonly iimage2D state_int_dst;
layout(rgba32f, binding = 7) uniform writeonly image2D state_vec_dst;
layout(rgba32f, binding = 8) uniform writeonly image2D state_misc_dst;
layout(r32i, binding = 9) uniform readonly iimage2D reaction_claim_tex;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }

    ivec4 cell_int = imageLoad(state_int_src, coord);
    vec4 cell_vec = imageLoad(state_vec_src, coord);
    vec4 cell_misc = imageLoad(state_misc_src, coord);

    ivec4 out_int = cell_int;
    vec4 out_vec = cell_vec;
    vec4 out_misc = cell_misc;
    int variant_index = cell_int.x;
    int family_index = family_index_for_variant(variant_index);

    float gathered_corrosion = 0.0;
    bool adjacent_fire = false;
    bool caused_corrosion = false;
    for (int index = 0; index < 8; index += 1) {
        ivec2 neighbor_coord = coord + NEIGHBORS_8[index];
        if (!in_bounds(neighbor_coord)) {
            continue;
        }
        ivec4 neighbor_int = imageLoad(state_int_src, neighbor_coord);
        int neighbor_variant = neighbor_int.x;
        int neighbor_reaction = reaction_kind_for_variant(neighbor_variant);
        if (reaction_kind_for_variant(variant_index) == REACTION_CORROSIVE && support_bearing_for_variant(neighbor_variant)) {
            caused_corrosion = true;
        }
        if (neighbor_reaction == REACTION_HEAT_SOURCE) {
            adjacent_fire = adjacent_fire || family_index_for_variant(neighbor_variant) == %d;
        }
        if (neighbor_reaction == REACTION_CORROSIVE && support_bearing_for_variant(variant_index)) {
            float hardness = max(variant_table[variant_index].floats0.y, 0.05);
            gathered_corrosion += reaction_strength_for_variant(neighbor_variant) * dt / hardness;
        }
    }

    if (reaction_kind_for_variant(variant_index) == REACTION_HEAT_SOURCE) {
        out_misc.x += reaction_energy_for_variant(variant_index) * dt / max(heat_capacity_for_variant(variant_index), 0.001);
        out_misc.w = cell_misc.w + dt;
        float max_age = family_table[family_index].floats0.x;
        if (out_misc.w >= max_age) {
            vec4 empty_misc = empty_cell_misc();
            empty_misc.x = out_misc.x;
            imageStore(state_int_dst, coord, empty_cell_int());
            imageStore(state_vec_dst, coord, empty_cell_vec());
            imageStore(state_misc_dst, coord, empty_misc);
            return;
        }
    }

    if (family_index == %d) {
        bool ignited = cell_misc.x >= ignite_temperature_for_variant(variant_index) || adjacent_fire;
        if (ignited) {
            out_int = ivec4(%d, cell_int.y + 1, 0, 0);
            out_vec = vec4(0.0);
            out_misc = vec4(600.0, 0.0, 1.0, 0.0);
            imageStore(state_int_dst, coord, out_int);
            imageStore(state_vec_dst, coord, out_vec);
            imageStore(state_misc_dst, coord, out_misc);
            return;
        }
        if (cell_misc.x >= 140.0) {
            out_int.x = %d;
        }
    }

    if (family_index == %d && cell_misc.x >= decompose_temperature_for_variant(variant_index)) {
        out_int.x = %d;
    }

    if (support_bearing_for_variant(variant_index) && integrity_decay_from_heat_for_variant(variant_index) > 0.0) {
        float heat_start = base_temperature_for_variant(variant_index) + 60.0;
        float melt_temperature = melt_temperature_for_variant(variant_index);
        if (melt_temperature > 0.0) {
            heat_start = min(heat_start, melt_temperature * 0.7);
        }
        if (cell_misc.x > heat_start) {
            out_misc.z = max(
                0.0,
                out_misc.z - (cell_misc.x - heat_start) * integrity_decay_from_heat_for_variant(variant_index) * dt
            );
        }
    }

    if (gathered_corrosion > 0.0) {
        out_misc.z = max(0.0, out_misc.z - gathered_corrosion);
    }
    if (caused_corrosion && !reaction_preserves_self_for_variant(variant_index)) {
        vec4 empty_misc = empty_cell_misc();
        empty_misc.x = out_misc.x;
        imageStore(state_int_dst, coord, empty_cell_int());
        imageStore(state_vec_dst, coord, empty_cell_vec());
        imageStore(state_misc_dst, coord, empty_misc);
        return;
    }

    int winner_index = imageLoad(reaction_claim_tex, coord).x;
    if (winner_index != INT_MAX_VALUE && cell_int.x == EMPTY_VARIANT_INDEX) {
        ivec2 source_coord = ivec2(winner_index %% grid_size.x, winner_index / grid_size.x);
        ivec4 source_int = imageLoad(state_int_src, source_coord);
        imageStore(state_int_dst, coord, ivec4(%d, source_int.y + 1, 0, 0));
        imageStore(state_vec_dst, coord, vec4(0.0));
        imageStore(state_misc_dst, coord, vec4(600.0, 0.0, 1.0, 0.0));
        return;
    }

    imageStore(state_int_dst, coord, out_int);
    imageStore(state_vec_dst, coord, out_vec);
    imageStore(state_misc_dst, coord, out_misc);
}
""" % (
        tables.family_index_by_id["fire"],
        tables.family_index_by_id["tar"],
        tables.variant_index_by_key[("fire", "fire")],
        tables.variant_index_by_key[("tar", "tar_smoke")],
        tables.family_index_by_id["poison"],
        tables.variant_index_by_key[("poison", "poison_gas")],
        tables.variant_index_by_key[("fire", "fire")],
    )


def _collapse_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32i, binding = 3) uniform readonly iimage2D state_int_src;
layout(rgba32f, binding = 4) uniform readonly image2D state_vec_src;
layout(rgba32f, binding = 5) uniform readonly image2D state_misc_src;
layout(rgba32i, binding = 6) uniform writeonly iimage2D state_int_dst;
layout(rgba32f, binding = 7) uniform writeonly image2D state_vec_dst;
layout(rgba32f, binding = 8) uniform writeonly image2D state_misc_dst;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }

    ivec4 cell_int = imageLoad(state_int_src, coord);
    vec4 cell_vec = imageLoad(state_vec_src, coord);
    vec4 cell_misc = imageLoad(state_misc_src, coord);

    if (!support_bearing_for_variant(cell_int.x) || cell_misc.z > 0.0) {
        imageStore(state_int_dst, coord, cell_int);
        imageStore(state_vec_dst, coord, cell_vec);
        imageStore(state_misc_dst, coord, cell_misc);
        return;
    }

    FamilyData family = family_table[family_index_for_variant(cell_int.x)];
    if (family.ints0.y < 0) {
        imageStore(state_int_dst, coord, empty_cell_int());
        imageStore(state_vec_dst, coord, empty_cell_vec());
        imageStore(state_misc_dst, coord, empty_cell_misc());
        return;
    }

    ivec4 out_int = cell_int;
    vec4 out_vec = cell_vec;
    vec4 out_misc = cell_misc;
    out_int.x = family.ints0.y;
    out_int.y = 0;
    out_int.z &= ~CELL_FLAG_FIXPOINT;
    out_vec = vec4(0.0);
    out_misc.y = 0.0;
    out_misc.z = 0.6;
    out_misc.w = 0.0;
    imageStore(state_int_dst, coord, out_int);
    imageStore(state_vec_dst, coord, out_vec);
    imageStore(state_misc_dst, coord, out_misc);
}
"""


def _render_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32i, binding = 3) uniform readonly iimage2D state_int_src;
layout(rgba32f, binding = 5) uniform readonly image2D state_misc_src;
layout(rgba8, binding = 11) uniform writeonly image2D frame_tex;
uniform int view_mode;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }

    ivec4 cell_int = imageLoad(state_int_src, coord);
    vec4 cell_misc = imageLoad(state_misc_src, coord);
    int variant_index = cell_int.x;
    vec3 color = vec3(10.0, 12.0, 16.0);

    if (view_mode == VIEW_MODE_TEMPERATURE) {
        color = temperature_color(cell_misc.x);
    } else if (view_mode == VIEW_MODE_PRESSURE) {
        color = pressure_color(imageLoad(pressure_tex, coord).x);
    } else if (variant_index != EMPTY_VARIANT_INDEX) {
        color = render_color_for_variant(variant_index);
        if (family_index_for_variant(variant_index) == FIRE_FAMILY_INDEX) {
            float pulse = min(1.0, cell_misc.w / 6.0);
            color.r = clamp_channel(color.r + 25.0 * (1.0 - pulse));
            color.g = clamp_channel(color.g - 20.0 * pulse);
            color.b = clamp_channel(color.b - 10.0 * pulse);
        }

        if ((cell_int.z & CELL_FLAG_FIXPOINT) != 0) {
            color.r = clamp_channel(color.r + 35.0);
            color.g = clamp_channel(color.g + 35.0);
        }

        if (support_bearing_for_variant(variant_index)) {
            float support_mix = clamp(cell_misc.y / SUPPORT_SOURCE_VALUE, 0.0, 1.0);
            float scale = 0.85 + support_mix * 0.15;
            color *= scale;
        }

        float integrity_tint = clamp(cell_misc.z, 0.2, 1.0);
        color.r = clamp_channel(color.r * integrity_tint + 30.0 * (1.0 - integrity_tint));
        color.g = clamp_channel(color.g * integrity_tint);
        color.b = clamp_channel(color.b * integrity_tint);

        if (cell_misc.x > 120.0 && family_index_for_variant(variant_index) != FIRE_FAMILY_INDEX) {
            float heat = min(1.0, (cell_misc.x - 120.0) / 900.0);
            color.r = clamp_channel(color.r + 120.0 * heat);
            color.g = clamp_channel(color.g + 30.0 * heat);
        }
    }

    ivec2 target_coord = ivec2(coord.x, grid_size.y - 1 - coord.y);
    imageStore(frame_tex, target_coord, vec4(color / 255.0, 1.0));
}
"""


def _paint_shader_source(tables: GpuMaterialTables) -> str:
    return _build_common_glsl(tables) + """
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32i, binding = 3) uniform iimage2D state_int_image;
layout(rgba32f, binding = 4) uniform image2D state_vec_image;
layout(rgba32f, binding = 5) uniform image2D state_misc_image;

uniform ivec2 paint_center;
uniform int paint_radius;
uniform int paint_variant_index;
uniform float paint_temperature;
uniform float paint_support_value;
uniform float paint_integrity;
uniform int paint_generation;
uniform float paint_age;
uniform int paint_flags;
uniform int erase_mode;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (!in_bounds(coord)) {
        return;
    }

    ivec2 delta = coord - paint_center;
    if (delta.x * delta.x + delta.y * delta.y > paint_radius * paint_radius) {
        return;
    }

    if (erase_mode != 0) {
        imageStore(state_int_image, coord, empty_cell_int());
        imageStore(state_vec_image, coord, empty_cell_vec());
        imageStore(state_misc_image, coord, empty_cell_misc());
        return;
    }

    imageStore(state_int_image, coord, ivec4(paint_variant_index, paint_generation, paint_flags, 0));
    imageStore(state_vec_image, coord, vec4(0.0));
    imageStore(state_misc_image, coord, vec4(paint_temperature, paint_support_value, paint_integrity, paint_age));
}
"""


class GpuSimulator:
    def __init__(self, ctx: moderngl.Context, grid: Grid, registry: MaterialRegistry) -> None:
        if getattr(ctx, "version_code", 0) < 430:
            raise ComputeBackendUnavailable("OpenGL 4.3 or newer is required for compute shaders.")

        self.ctx = ctx
        self.width = grid.width
        self.height = grid.height
        self.registry = registry
        self.tables = GpuMaterialTables.from_registry(registry)
        self.group_x, self.group_y = _dispatch_groups(self.width, self.height)
        self.pressure_group_x = (self.width + 63) // 64
        self.front_index = 0
        self.pressure_front_index = 0
        self.step_index = grid.step_id
        self.liquid_brownian_enabled = bool(grid.liquid_brownian_enabled)
        self.blocked_impulse_enabled = bool(grid.blocked_impulse_enabled)

        self.variant_buffer = self.ctx.buffer(self.tables.variant_buffer_data)
        self.family_buffer = self.ctx.buffer(self.tables.family_buffer_data)
        self.phase_buffer = self.ctx.buffer(self.tables.phase_buffer_data)
        self.variant_buffer.bind_to_storage_buffer(0)
        self.family_buffer.bind_to_storage_buffer(1)
        self.phase_buffer.bind_to_storage_buffer(2)

        self.state_int = [self._make_texture(4, "i4"), self._make_texture(4, "i4")]
        self.state_vec = [self._make_texture(4, "f4"), self._make_texture(4, "f4")]
        self.state_misc = [self._make_texture(4, "f4"), self._make_texture(4, "f4")]
        self.pressure_tex = [self._make_texture(1, "f4"), self._make_texture(1, "f4")]
        self.source_force_tex = [self._make_texture(2, "f4"), self._make_texture(2, "f4")]
        self.wave_force_tex = [self._make_texture(2, "f4"), self._make_texture(2, "f4")]
        self.source_force_front_index = 0
        self.wave_force_front_index = 0
        self.motion_plan = self._make_texture(4, "i4")
        self.motion_claim = self._make_texture(1, "i4")
        self.reaction_claim = self._make_texture(1, "i4")
        self.frame_texture = self._make_texture(4, "f1")
        self.frame_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.frame_texture.repeat_x = False
        self.frame_texture.repeat_y = False

        self.pressure_shader = self.ctx.compute_shader(_pressure_shader_source(self.tables))
        self.source_force_shader = self.ctx.compute_shader(_source_force_shader_source(self.tables))
        self.force_wave_shader = self.ctx.compute_shader(_force_wave_shader_source(self.tables))
        self.support_shader = self.ctx.compute_shader(_support_shader_source(self.tables))
        self.motion_plan_shader = self.ctx.compute_shader(_motion_plan_shader_source(self.tables))
        self.clear_r32i_shader = self.ctx.compute_shader(_clear_r32i_shader_source(self.tables))
        self.motion_claim_shader = self.ctx.compute_shader(_motion_claim_shader_source(self.tables))
        self.motion_resolve_shader = self.ctx.compute_shader(_motion_resolve_shader_source(self.tables))
        self.thermal_shader = self.ctx.compute_shader(_thermal_shader_source(self.tables))
        self.phase_shader = self.ctx.compute_shader(_phase_shader_source(self.tables))
        self.reaction_plan_shader = self.ctx.compute_shader(_reaction_plan_shader_source(self.tables))
        self.reaction_resolve_shader = self.ctx.compute_shader(_reaction_resolve_shader_source(self.tables))
        self.collapse_shader = self.ctx.compute_shader(_collapse_shader_source(self.tables))
        self.render_shader = self.ctx.compute_shader(_render_shader_source(self.tables))
        self.paint_shader = self.ctx.compute_shader(_paint_shader_source(self.tables))

        self._set_common_uniforms(
            self.pressure_shader,
            self.source_force_shader,
            self.force_wave_shader,
            self.support_shader,
            self.motion_plan_shader,
            self.clear_r32i_shader,
            self.motion_claim_shader,
            self.motion_resolve_shader,
            self.thermal_shader,
            self.phase_shader,
            self.reaction_plan_shader,
            self.reaction_resolve_shader,
            self.collapse_shader,
            self.render_shader,
            self.paint_shader,
        )

        self.load_grid(grid)
        self.render()

    def _make_texture(self, components: int, dtype: str) -> moderngl.Texture:
        texture = self.ctx.texture((self.width, self.height), components, data=None, dtype=dtype)
        texture.repeat_x = False
        texture.repeat_y = False
        return texture

    def _set_common_uniforms(self, *programs: moderngl.ComputeShader) -> None:
        for program in programs:
            _set_uniform_if_present(program, "grid_size", (self.width, self.height))
            _set_uniform_if_present(program, "step_index", self.step_index)
            _set_uniform_if_present(program, "liquid_brownian_enabled", int(self.liquid_brownian_enabled))
            _set_uniform_if_present(program, "blocked_impulse_enabled", int(self.blocked_impulse_enabled))

    def set_liquid_brownian_enabled(self, enabled: bool) -> None:
        self.liquid_brownian_enabled = bool(enabled)
        self._set_common_uniforms(
            self.pressure_shader,
            self.source_force_shader,
            self.force_wave_shader,
            self.support_shader,
            self.motion_plan_shader,
            self.clear_r32i_shader,
            self.motion_claim_shader,
            self.motion_resolve_shader,
            self.thermal_shader,
            self.phase_shader,
            self.reaction_plan_shader,
            self.reaction_resolve_shader,
            self.collapse_shader,
            self.render_shader,
            self.paint_shader,
        )

    def set_blocked_impulse_enabled(self, enabled: bool) -> None:
        self.blocked_impulse_enabled = bool(enabled)
        self._set_common_uniforms(
            self.pressure_shader,
            self.source_force_shader,
            self.force_wave_shader,
            self.support_shader,
            self.motion_plan_shader,
            self.clear_r32i_shader,
            self.motion_claim_shader,
            self.motion_resolve_shader,
            self.thermal_shader,
            self.phase_shader,
            self.reaction_plan_shader,
            self.reaction_resolve_shader,
            self.collapse_shader,
            self.render_shader,
            self.paint_shader,
        )

    def _pressure_bytes(self, value: float = 1.0) -> bytes:
        return array("f", [value for _ in range(self.width * self.height)]).tobytes()

    def _force_bytes(self) -> bytes:
        return array("f", [0.0 for _ in range(self.width * self.height * 2)]).tobytes()

    def _bind_state(self, src_index: int, dst_index: int) -> None:
        self.state_int[src_index].bind_to_image(3, read=True, write=False)
        self.state_vec[src_index].bind_to_image(4, read=True, write=False)
        self.state_misc[src_index].bind_to_image(5, read=True, write=False)
        self.state_int[dst_index].bind_to_image(6, read=False, write=True)
        self.state_vec[dst_index].bind_to_image(7, read=False, write=True)
        self.state_misc[dst_index].bind_to_image(8, read=False, write=True)

    def _run_stage(self, shader: moderngl.ComputeShader, dt: float, *, swap_after: bool = True) -> None:
        src_index = self.front_index
        dst_index = 1 - self.front_index
        self._bind_state(src_index, dst_index)
        _set_uniform_if_present(shader, "dt", dt)
        _set_uniform_if_present(shader, "step_index", self.step_index)
        shader.run(group_x=self.group_x, group_y=self.group_y, group_z=1)
        self.ctx.memory_barrier()
        if swap_after:
            self.front_index = dst_index

    def _run_support(self, dt: float) -> None:
        self._run_stage(self.support_shader, dt)

    def _clear_r32i_texture(self, texture: moderngl.Texture, clear_value: int) -> None:
        texture.bind_to_image(9, read=False, write=True)
        _set_uniform_if_present(self.clear_r32i_shader, "dt", 0.0)
        _set_uniform_if_present(self.clear_r32i_shader, "step_index", self.step_index)
        self.clear_r32i_shader["clear_value"].value = clear_value
        self.clear_r32i_shader.run(group_x=self.group_x, group_y=self.group_y, group_z=1)
        self.ctx.memory_barrier()

    def _run_pressure(self, *, step_seed: int | None = None) -> None:
        self.state_int[self.front_index].bind_to_image(3, read=True, write=False)
        self.state_misc[self.front_index].bind_to_image(5, read=True, write=False)
        self.pressure_tex[self.pressure_front_index].bind_to_image(12, read=True, write=False)
        self.pressure_tex[1 - self.pressure_front_index].bind_to_image(13, read=False, write=True)
        _set_uniform_if_present(self.pressure_shader, "dt", 0.0)
        _set_uniform_if_present(self.pressure_shader, "step_index", self.step_index if step_seed is None else step_seed)
        self.pressure_shader.run(group_x=self.pressure_group_x, group_y=1, group_z=1)
        self.ctx.memory_barrier()
        self.pressure_front_index = 1 - self.pressure_front_index

    def _run_source_force(self, *, step_seed: int | None = None) -> None:
        current_index = 1 - self.source_force_front_index
        self.state_int[self.front_index].bind_to_image(3, read=True, write=False)
        self.pressure_tex[self.pressure_front_index].bind_to_image(12, read=True, write=False)
        self.source_force_tex[current_index].bind_to_image(13, read=False, write=True)
        _set_uniform_if_present(self.source_force_shader, "dt", 0.0)
        _set_uniform_if_present(self.source_force_shader, "step_index", self.step_index if step_seed is None else step_seed)
        self.source_force_shader.run(group_x=self.group_x, group_y=self.group_y, group_z=1)
        self.ctx.memory_barrier()
        self.source_force_front_index = current_index

    def _run_force_wave(self, *, step_seed: int | None = None) -> int:
        current_index = 1 - self.wave_force_front_index
        self.state_int[self.front_index].bind_to_image(3, read=True, write=False)
        self.source_force_tex[self.source_force_front_index].bind_to_image(13, read=True, write=False)
        self.source_force_tex[1 - self.source_force_front_index].bind_to_image(14, read=True, write=False)
        self.wave_force_tex[self.wave_force_front_index].bind_to_image(15, read=True, write=False)
        self.wave_force_tex[current_index].bind_to_image(16, read=False, write=True)
        _set_uniform_if_present(self.force_wave_shader, "dt", 0.0)
        _set_uniform_if_present(self.force_wave_shader, "step_index", self.step_index if step_seed is None else step_seed)
        self.force_wave_shader.run(group_x=self.group_x, group_y=self.group_y, group_z=1)
        self.ctx.memory_barrier()
        return current_index

    def _run_motion(self, dt: float, *, step_seed: int | None = None, liquids_only: bool = False) -> None:
        self.state_int[self.front_index].bind_to_image(3, read=True, write=False)
        self.state_vec[self.front_index].bind_to_image(4, read=True, write=False)
        self.state_misc[self.front_index].bind_to_image(5, read=True, write=False)
        self.pressure_tex[self.pressure_front_index].bind_to_image(12, read=True, write=False)
        self.wave_force_tex[self.wave_force_front_index].bind_to_image(15, read=True, write=False)
        self.motion_plan.bind_to_image(9, read=False, write=True)
        _set_uniform_if_present(self.motion_plan_shader, "dt", dt)
        _set_uniform_if_present(self.motion_plan_shader, "step_index", self.step_index if step_seed is None else step_seed)
        _set_uniform_if_present(self.motion_plan_shader, "liquids_only", int(liquids_only))
        self.motion_plan_shader.run(group_x=self.group_x, group_y=self.group_y, group_z=1)
        self.ctx.memory_barrier()

        self._clear_r32i_texture(self.motion_claim, INT_MAX_VALUE)

        self.motion_plan.bind_to_image(9, read=True, write=False)
        self.motion_claim.bind_to_image(10, read=True, write=True)
        _set_uniform_if_present(self.motion_claim_shader, "dt", dt)
        _set_uniform_if_present(self.motion_claim_shader, "step_index", self.step_index if step_seed is None else step_seed)
        _set_uniform_if_present(self.motion_claim_shader, "liquids_only", int(liquids_only))
        self.motion_claim_shader.run(group_x=self.group_x, group_y=self.group_y, group_z=1)
        self.ctx.memory_barrier()

        self._bind_state(self.front_index, 1 - self.front_index)
        self.motion_plan.bind_to_image(9, read=True, write=False)
        self.motion_claim.bind_to_image(10, read=True, write=False)
        self.pressure_tex[self.pressure_front_index].bind_to_image(12, read=True, write=False)
        self.wave_force_tex[self.wave_force_front_index].bind_to_image(15, read=True, write=False)
        _set_uniform_if_present(self.motion_resolve_shader, "dt", dt)
        _set_uniform_if_present(self.motion_resolve_shader, "step_index", self.step_index if step_seed is None else step_seed)
        _set_uniform_if_present(self.motion_resolve_shader, "liquids_only", int(liquids_only))
        self.motion_resolve_shader.run(group_x=self.group_x, group_y=self.group_y, group_z=1)
        self.ctx.memory_barrier()
        self.front_index = 1 - self.front_index

    def _run_reactions(self, dt: float) -> None:
        self._clear_r32i_texture(self.reaction_claim, INT_MAX_VALUE)

        self.state_int[self.front_index].bind_to_image(3, read=True, write=False)
        self.state_misc[self.front_index].bind_to_image(5, read=True, write=False)
        self.reaction_claim.bind_to_image(9, read=True, write=True)
        _set_uniform_if_present(self.reaction_plan_shader, "dt", dt)
        _set_uniform_if_present(self.reaction_plan_shader, "step_index", self.step_index)
        self.reaction_plan_shader.run(group_x=self.group_x, group_y=self.group_y, group_z=1)
        self.ctx.memory_barrier()

        self._bind_state(self.front_index, 1 - self.front_index)
        self.reaction_claim.bind_to_image(9, read=True, write=False)
        _set_uniform_if_present(self.reaction_resolve_shader, "dt", dt)
        _set_uniform_if_present(self.reaction_resolve_shader, "step_index", self.step_index)
        self.reaction_resolve_shader.run(group_x=self.group_x, group_y=self.group_y, group_z=1)
        self.ctx.memory_barrier()
        self.front_index = 1 - self.front_index

    def load_grid(self, grid: Grid) -> None:
        if grid.width != self.width or grid.height != self.height:
            raise ValueError("Grid dimensions do not match the GPU simulator.")
        self.set_liquid_brownian_enabled(grid.liquid_brownian_enabled)
        self.set_blocked_impulse_enabled(grid.blocked_impulse_enabled)
        state_int_data, state_vec_data, state_misc_data = pack_grid_state(grid, self.tables)
        for texture in self.state_int:
            texture.write(state_int_data)
        for texture in self.state_vec:
            texture.write(state_vec_data)
        for texture in self.state_misc:
            texture.write(state_misc_data)
        pressure_bytes = self._pressure_bytes()
        for texture in self.pressure_tex:
            texture.write(pressure_bytes)
        force_bytes = self._force_bytes()
        for texture in self.source_force_tex:
            texture.write(force_bytes)
        for texture in self.wave_force_tex:
            texture.write(force_bytes)
        self.front_index = 0
        self.pressure_front_index = 0
        self.source_force_front_index = 0
        self.wave_force_front_index = 0
        self.step_index = grid.step_id

    def step(self, dt: float) -> None:
        self._run_support(dt)
        self._run_reactions(dt)
        self._run_stage(self.thermal_shader, dt)
        self._run_stage(self.phase_shader, dt)
        self._run_pressure(step_seed=self.step_index)
        self._run_source_force(step_seed=self.step_index)
        next_wave_index = self._run_force_wave(step_seed=self.step_index)
        self._run_motion(dt, step_seed=self.step_index, liquids_only=False)
        self.wave_force_front_index = next_wave_index
        if LIQUID_RELAXATION_PASSES > 0:
            relaxation_dt = dt / LIQUID_RELAXATION_PASSES
            for pass_index in range(LIQUID_RELAXATION_PASSES):
                step_seed = self.step_index * (LIQUID_RELAXATION_PASSES + 1) + 1 + pass_index
                self._run_pressure(step_seed=step_seed)
                self._run_motion(relaxation_dt, step_seed=step_seed, liquids_only=True)
        self._run_stage(self.collapse_shader, dt)
        self.step_index += 1

    def render(self, view_mode: DebugViewMode = DebugViewMode.MATERIAL) -> moderngl.Texture:
        self.state_int[self.front_index].bind_to_image(3, read=True, write=False)
        self.state_misc[self.front_index].bind_to_image(5, read=True, write=False)
        self.pressure_tex[self.pressure_front_index].bind_to_image(12, read=True, write=False)
        self.frame_texture.bind_to_image(11, read=False, write=True)
        _set_uniform_if_present(self.render_shader, "dt", 0.0)
        _set_uniform_if_present(self.render_shader, "step_index", self.step_index)
        if view_mode == DebugViewMode.MATERIAL:
            self.render_shader["view_mode"].value = 0
        elif view_mode == DebugViewMode.TEMPERATURE:
            self.render_shader["view_mode"].value = 1
        else:
            self.render_shader["view_mode"].value = 2
        self.render_shader.run(group_x=self.group_x, group_y=self.group_y, group_z=1)
        self.ctx.memory_barrier()
        return self.frame_texture

    def paint_circle(
        self,
        center_x: int,
        center_y: int,
        radius: int,
        family_id: str | None,
        variant_id: str | None,
        *,
        overrides: dict[str, object] | None = None,
    ) -> None:
        center_x = int(center_x)
        center_y = int(center_y)
        radius = int(radius)
        overrides = overrides or {}
        erase_mode = 1 if family_id is None or variant_id is None else 0
        variant_index = self.tables.empty_variant_index if erase_mode else self.tables.variant_index_by_key[(family_id, variant_id)]
        base_temperature = self.registry.variant(family_id, variant_id).base_temperature if not erase_mode else 20.0
        flags = int(overrides.get("flags", CellFlag.NONE))
        support_value = float(overrides.get("support_value", SUPPORT_SOURCE_VALUE if flags & CellFlag.FIXPOINT else 0.0))
        integrity = float(overrides.get("integrity", 1.0))
        generation = int(overrides.get("generation", 0))
        age = float(overrides.get("age", 0.0))

        self.state_int[self.front_index].bind_to_image(3, read=True, write=True)
        self.state_vec[self.front_index].bind_to_image(4, read=True, write=True)
        self.state_misc[self.front_index].bind_to_image(5, read=True, write=True)
        _set_uniform_if_present(self.paint_shader, "dt", 0.0)
        _set_uniform_if_present(self.paint_shader, "step_index", self.step_index)
        self.paint_shader["paint_center"].value = (center_x, center_y)
        self.paint_shader["paint_radius"].value = radius
        self.paint_shader["paint_variant_index"].value = variant_index
        self.paint_shader["paint_temperature"].value = float(overrides.get("temperature", base_temperature))
        self.paint_shader["paint_support_value"].value = support_value
        self.paint_shader["paint_integrity"].value = integrity
        self.paint_shader["paint_generation"].value = generation
        self.paint_shader["paint_age"].value = age
        self.paint_shader["paint_flags"].value = flags
        self.paint_shader["erase_mode"].value = erase_mode
        self.paint_shader.run(group_x=self.group_x, group_y=self.group_y, group_z=1)
        self.ctx.memory_barrier()

    def readback_grid(self) -> Grid:
        grid = unpack_grid_state(
            self.width,
            self.height,
            self.tables,
            self.state_int[self.front_index].read(),
            self.state_vec[self.front_index].read(),
            self.state_misc[self.front_index].read(),
        )
        grid.step_id = self.step_index
        return grid
