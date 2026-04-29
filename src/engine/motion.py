from __future__ import annotations

from math import copysign, cos, hypot, radians, sqrt

from .atmosphere import ambient_air_temperature_for_row
from .grid import Grid
from .types import MaterialRegistry, MatterState


AIR_PRESSURE = 1.0
PRESSURE_RELAXATION = 0.22
PRESSURE_FORCE_SCALE = 0.55
LIQUID_RELAXATION_PASSES = 2
FORCE_WAVE_DECAY = 0.82
LIQUID_LATERAL_PRESSURE_BOOST = 1.35
LIQUID_BLOCKED_LATERAL_BOOST = 2.6
LIQUID_SURFACE_SIDEFLOW_BONUS = 3.0
LIQUID_PRESSURE_HEAD_BOOST = 0.55
LIQUID_PRESSURE_HEAD_BOOST_CAP = 4.0
LIQUID_RELAXATION_HEAD_THRESHOLD = 1.5
LIQUID_RELAXATION_NEIGHBOR_THRESHOLD = 2
LIQUID_VERTICAL_PRESSURE_SCALE = 0.8
GAS_PRESSURE_FORCE_SCALE = 0.65
GRAVITY_ACCELERATION = 1.2
GAS_BUOYANCY_SCALE = 10.0
INTENT_DECAY = 0.72
BLOCKED_IMPULSE_MAX = 1.25
LIQUID_RANDOM_GAIN = 0.012
LIQUID_RANDOM_VERTICAL_FACTOR = 0.18
GAS_RANDOM_GAIN = 0.42
GAS_RANDOM_VERTICAL_FACTOR = 0.85
DIRECTION_JITTER_GAIN = 0.06
DIRECTION_TIE_BREAK_GAIN = 1e-4
DIRECTION_FALLBACK_ALIGNMENT_EPSILON = 1e-6
KELVIN_OFFSET = 273.15
MIN_THERMAL_KELVIN = 80.0
EMPTY_MOTION_TEMPERATURE_THRESHOLD = 0.5
GAS_THERMAL_TEMPERATURE_SPAN = 120.0
LIQUID_THERMAL_TEMPERATURE_SPAN = 400.0
GAS_RANDOM_FLOOR_FACTOR = 0.35
GAS_RANDOM_HEAT_FACTOR = 1.25
LIQUID_RANDOM_FLOOR_FACTOR = 0.15
DENSITY_EPSILON = 1e-6
GAS_NEIGHBOR_WIND_DIAGONAL_WEIGHT = 0.7


def _normalize(x: float, y: float) -> tuple[float, float]:
    length = hypot(x, y)
    if length == 0.0:
        return (0.0, 0.0)
    return (x / length, y / length)


def _direction_score(direction: tuple[int, int], desired: tuple[float, float]) -> float:
    ndx, ndy = _normalize(direction[0], direction[1])
    return ndx * desired[0] + ndy * desired[1]


def _hash01(step_id: int, x: int, y: int, salt: int) -> float:
    value = (
        (step_id + 1) * 374_761_393
        + (x + 11) * 668_265_263
        + (y + 17) * 2_147_483_647
        + (salt + 23) * 1_274_126_177
    ) & 0xFFFFFFFF
    value ^= value >> 13
    value = (value * 1_274_126_177) & 0xFFFFFFFF
    return value / 0xFFFFFFFF


def _state_rank(state: MatterState) -> int:
    if state == MatterState.SOLID:
        return 3
    if state == MatterState.LIQUID:
        return 2
    return 1


def _variant_can_translate(variant) -> bool:
    return (
        variant.mobility > 1e-9
        or variant.pressure_response > 1e-9
        or variant.gravity_scale > 1e-9
        or variant.buoyancy_scale > 1e-9
        or variant.thermal_motion_scale > 1e-9
        or variant.wind_coupling > 1e-9
    )


def _ambient_air_temperature(grid: Grid, registry: MaterialRegistry, y: int) -> float:
    return ambient_air_temperature_for_row(
        grid.height,
        y,
        registry.variant("empty", "empty").base_temperature,
    )


def _temperature_kelvin(temperature: float) -> float:
    return max(MIN_THERMAL_KELVIN, temperature + KELVIN_OFFSET)


def _temperature_motion_factor(temperature: float, ambient_temperature: float, *, span: float) -> float:
    return min(1.0, max(0.0, temperature - ambient_temperature) / max(span, 0.001))


def _effective_density(cell, variant, *, ambient_temperature: float) -> float:
    if variant.matter_state == MatterState.GAS:
        ambient_kelvin = _temperature_kelvin(ambient_temperature)
        return variant.density * ambient_kelvin / _temperature_kelvin(cell.temperature)
    return variant.density


def _thermal_random_gain(grid: Grid, registry: MaterialRegistry, y: int, cell, variant) -> float:
    if variant.thermal_motion_scale <= 0.0:
        return 0.0
    ambient_temperature = _ambient_air_temperature(grid, registry, y)
    if variant.matter_state == MatterState.GAS:
        factor = _temperature_motion_factor(
            cell.temperature,
            ambient_temperature,
            span=GAS_THERMAL_TEMPERATURE_SPAN,
        )
        ambient_density = registry.variant("empty", "empty").density
        density_ratio = ambient_density / max(variant.density, DENSITY_EPSILON)
        gain_scale = min(1.35, max(0.85, density_ratio))
        return variant.thermal_motion_scale * GAS_RANDOM_GAIN * gain_scale * (
            GAS_RANDOM_FLOOR_FACTOR + (1.0 - GAS_RANDOM_FLOOR_FACTOR) * factor * GAS_RANDOM_HEAT_FACTOR
        )
    if variant.matter_state == MatterState.LIQUID and grid.liquid_brownian_enabled:
        factor = _temperature_motion_factor(
            cell.temperature,
            ambient_temperature,
            span=LIQUID_THERMAL_TEMPERATURE_SPAN,
        )
        return variant.thermal_motion_scale * LIQUID_RANDOM_GAIN * (
            LIQUID_RANDOM_FLOOR_FACTOR + (1.0 - LIQUID_RANDOM_FLOOR_FACTOR) * factor
        )
    return 0.0


def _random_step_factor(dt: float) -> float:
    return min(1.0, max(0.0, dt * 60.0))


def _thermal_buoyancy(grid: Grid, registry: MaterialRegistry, y: int, cell, variant) -> float:
    if variant.matter_state != MatterState.GAS:
        return 0.0
    ambient_density = registry.variant("empty", "empty").density
    ambient_temperature = _ambient_air_temperature(grid, registry, y)
    return (_effective_density(cell, variant, ambient_temperature=ambient_temperature) - ambient_density) * GAS_BUOYANCY_SCALE


def _local_wind_velocity(grid: Grid, registry: MaterialRegistry, x: int, y: int) -> tuple[float, float]:
    wind_x = 0.0
    wind_y = 0.0
    total_weight = 0.0

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if not grid.in_bounds(nx, ny):
                continue

            neighbor = grid.get_cell(nx, ny)
            neighbor_variant = registry.variant(neighbor.family_id, neighbor.variant_id)
            if neighbor_variant.matter_state != MatterState.GAS:
                continue

            weight = GAS_NEIGHBOR_WIND_DIAGONAL_WEIGHT if dx != 0 and dy != 0 else 1.0
            wind_x += weight * (neighbor.vel_x + neighbor.blocked_x)
            wind_y += weight * (neighbor.vel_y + neighbor.blocked_y)
            total_weight += weight

    if total_weight <= 0.0:
        return (0.0, 0.0)
    return (wind_x / total_weight, wind_y / total_weight)


def _wind_coupling_for_cell(grid: Grid, registry: MaterialRegistry, y: int, cell, variant) -> float:
    if variant.wind_coupling <= 0.0:
        return 0.0
    ambient_density = registry.variant("empty", "empty").density
    ambient_temperature = _ambient_air_temperature(grid, registry, y)
    current_density = _effective_density(cell, variant, ambient_temperature=ambient_temperature)
    density_ratio = ambient_density / max(current_density, DENSITY_EPSILON)
    return variant.wind_coupling * min(1.0, density_ratio)


def _gas_cell_can_move(grid: Grid, registry: MaterialRegistry, x: int, y: int, current, variant) -> bool:
    ambient_temperature = _ambient_air_temperature(grid, registry, y)
    ambient_density = registry.variant("empty", "empty").density
    index = grid.index(x, y)
    return (
        abs(variant.density - ambient_density) > DENSITY_EPSILON
        or abs(current.temperature - ambient_temperature) > EMPTY_MOTION_TEMPERATURE_THRESHOLD
        or hypot(current.vel_x, current.vel_y) > 0.01
        or hypot(current.blocked_x, current.blocked_y) > 0.01
        or hypot(grid.source_force_x[index] + grid.force_wave_x[index], grid.source_force_y[index] + grid.force_wave_y[index]) > 0.01
    )


def _gas_like_target_can_exchange(
    current,
    current_variant,
    target,
    target_variant,
    *,
    current_ambient_temperature: float,
    target_ambient_temperature: float,
    direction: tuple[int, int],
) -> bool:
    current_density = _effective_density(current, current_variant, ambient_temperature=current_ambient_temperature)
    target_density = _effective_density(target, target_variant, ambient_temperature=target_ambient_temperature)
    if direction[1] < 0:
        return current_density < target_density - DENSITY_EPSILON
    if direction[1] > 0:
        return current_density > target_density + DENSITY_EPSILON
    return True


def _target_can_exchange(
    current,
    current_variant,
    target,
    target_variant,
    *,
    grid: Grid,
    registry: MaterialRegistry,
    current_y: int,
    target_y: int,
    direction: tuple[int, int],
) -> bool:
    if current_variant.matter_state == MatterState.LIQUID and target_variant.matter_state == MatterState.GAS and direction[1] < 0:
        return False
    if current_variant.matter_state == MatterState.GAS and target_variant.matter_state == MatterState.GAS:
        if current.is_empty and not target.is_empty:
            return False
        if direction[1] == 0:
            return True
        return _gas_like_target_can_exchange(
            current,
            current_variant,
            target,
            target_variant,
            current_ambient_temperature=_ambient_air_temperature(grid, registry, current_y),
            target_ambient_temperature=_ambient_air_temperature(grid, registry, target_y),
            direction=direction,
        )
    current_rank = _state_rank(current_variant.matter_state)
    target_rank = _state_rank(target_variant.matter_state)
    if current_rank != target_rank:
        return target_rank < current_rank
    return _effective_density(
        target,
        target_variant,
        ambient_temperature=_ambient_air_temperature(grid, registry, target_y),
    ) < _effective_density(
        current,
        current_variant,
        ambient_temperature=_ambient_air_temperature(grid, registry, current_y),
    )


def _axis_remaining_after_step(desired_component: float, realized_component: float) -> float:
    if desired_component == 0.0:
        return 0.0
    if realized_component == 0.0 or desired_component * realized_component <= 0.0:
        return desired_component
    return copysign(max(0.0, abs(desired_component) - abs(realized_component)), desired_component)


def _clamp_blocked_impulse(value: float) -> float:
    return max(-BLOCKED_IMPULSE_MAX, min(BLOCKED_IMPULSE_MAX, value))


def _decayed_blocked_impulse(value: float) -> float:
    return _clamp_blocked_impulse(value * INTENT_DECAY)


def _downward_exchange_available(grid: Grid, registry: MaterialRegistry, x: int, y: int, variant) -> bool:
    current = grid.get_cell(x, y)
    ny = y + 1
    if not grid.in_bounds(x, ny):
        return False
    target = grid.get_cell(x, ny)
    target_variant = registry.variant(target.family_id, target.variant_id)
    return _target_can_exchange(current, variant, target, target_variant, grid=grid, registry=registry, current_y=y, target_y=ny, direction=(0, 1))


def _compute_pressure_field(grid: Grid, registry: MaterialRegistry) -> None:
    next_pressure = [AIR_PRESSURE for _ in range(grid.width * grid.height)]
    previous_pressure = grid.pressure

    for y in range(grid.height):
        for x in range(grid.width):
            index = grid.index(x, y)
            cell = grid.get_cell(x, y)
            variant = registry.variant(cell.family_id, cell.variant_id)
            ambient_temperature = _ambient_air_temperature(grid, registry, y)
            if variant.matter_state == MatterState.GAS:
                next_pressure[index] = AIR_PRESSURE + _effective_density(cell, variant, ambient_temperature=ambient_temperature)
            else:
                next_pressure[index] = previous_pressure[index] + (AIR_PRESSURE - previous_pressure[index]) * PRESSURE_RELAXATION

    for x in range(grid.width):
        for y in range(grid.height):
            index = grid.index(x, y)
            cell = grid.get_cell(x, y)
            variant = registry.variant(cell.family_id, cell.variant_id)
            if variant.matter_state != MatterState.LIQUID:
                continue
            above_pressure = AIR_PRESSURE if y == 0 else next_pressure[grid.index(x, y - 1)]
            next_pressure[index] = above_pressure + variant.density

    grid.pressure = next_pressure


def _pressure_force(grid: Grid, x: int, y: int) -> tuple[float, float]:
    current_pressure = grid.pressure[grid.index(x, y)]
    force_x = 0.0
    force_y = 0.0
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx = x + dx
        ny = y + dy
        if not grid.in_bounds(nx, ny):
            continue
        neighbor_pressure = grid.pressure[grid.index(nx, ny)]
        delta = current_pressure - neighbor_pressure
        force_x += dx * delta
        force_y += dy * delta
    return (force_x, force_y)


def _local_pressure_force(
    grid: Grid,
    registry: MaterialRegistry,
    x: int,
    y: int,
    variant,
) -> tuple[float, float]:
    pressure_x, pressure_y = _pressure_force(grid, x, y)
    current_pressure = grid.pressure[grid.index(x, y)]
    pressure_scale_x = PRESSURE_FORCE_SCALE
    pressure_scale_y = PRESSURE_FORCE_SCALE

    if variant.matter_state == MatterState.LIQUID:
        pressure_scale_x *= LIQUID_LATERAL_PRESSURE_BOOST
        pressure_scale_y *= LIQUID_VERTICAL_PRESSURE_SCALE
        if not _downward_exchange_available(grid, registry, x, y, variant):
            pressure_scale_x *= LIQUID_BLOCKED_LATERAL_BOOST
            pressure_head = max(0.0, current_pressure - AIR_PRESSURE)
            pressure_scale_x *= 1.0 + min(LIQUID_PRESSURE_HEAD_BOOST_CAP, sqrt(pressure_head) * LIQUID_PRESSURE_HEAD_BOOST)
    elif variant.matter_state == MatterState.GAS:
        pressure_scale_x *= GAS_PRESSURE_FORCE_SCALE
        pressure_scale_y *= GAS_PRESSURE_FORCE_SCALE

    return (pressure_x * pressure_scale_x, pressure_y * pressure_scale_y)


def _compute_source_force_field(grid: Grid, registry: MaterialRegistry) -> None:
    next_source_x = [0.0 for _ in range(grid.width * grid.height)]
    next_source_y = [0.0 for _ in range(grid.width * grid.height)]

    for y in range(grid.height):
        for x in range(grid.width):
            index = grid.index(x, y)
            cell = grid.get_cell(x, y)
            variant = registry.variant(cell.family_id, cell.variant_id)
            next_source_x[index], next_source_y[index] = _local_pressure_force(grid, registry, x, y, variant)

    grid.source_force_x = next_source_x
    grid.source_force_y = next_source_y


def _next_force_wave(grid: Grid, registry: MaterialRegistry) -> tuple[list[float], list[float]]:
    next_wave_x = [0.0 for _ in range(grid.width * grid.height)]
    next_wave_y = [0.0 for _ in range(grid.width * grid.height)]

    for y in range(grid.height):
        for x in range(grid.width):
            index = grid.index(x, y)
            cell = grid.get_cell(x, y)
            variant = registry.variant(cell.family_id, cell.variant_id)
            if variant.matter_state != MatterState.LIQUID:
                continue

            incoming_x = 0.0
            incoming_y = 0.0
            if x > 0:
                left_index = grid.index(x - 1, y)
                left_cell = grid.get_cell(x - 1, y)
                left_variant = registry.variant(left_cell.family_id, left_cell.variant_id)
                if left_variant.matter_state == MatterState.LIQUID:
                    incoming_x += max(0.0, grid.force_wave_x[left_index]) * FORCE_WAVE_DECAY
            if x < grid.width - 1:
                right_index = grid.index(x + 1, y)
                right_cell = grid.get_cell(x + 1, y)
                right_variant = registry.variant(right_cell.family_id, right_cell.variant_id)
                if right_variant.matter_state == MatterState.LIQUID:
                    incoming_x += min(0.0, grid.force_wave_x[right_index]) * FORCE_WAVE_DECAY
            if y > 0:
                above_index = grid.index(x, y - 1)
                above_cell = grid.get_cell(x, y - 1)
                above_variant = registry.variant(above_cell.family_id, above_cell.variant_id)
                if above_variant.matter_state == MatterState.LIQUID:
                    incoming_y += max(0.0, grid.force_wave_y[above_index]) * FORCE_WAVE_DECAY
            if y < grid.height - 1:
                below_index = grid.index(x, y + 1)
                below_cell = grid.get_cell(x, y + 1)
                below_variant = registry.variant(below_cell.family_id, below_cell.variant_id)
                if below_variant.matter_state == MatterState.LIQUID:
                    incoming_y += min(0.0, grid.force_wave_y[below_index]) * FORCE_WAVE_DECAY

            delta_x = grid.source_force_x[index] - FORCE_WAVE_DECAY * grid.prev_source_force_x[index]
            delta_y = grid.source_force_y[index] - FORCE_WAVE_DECAY * grid.prev_source_force_y[index]
            next_wave_x[index] = incoming_x + delta_x
            next_wave_y[index] = incoming_y + delta_y

    return (next_wave_x, next_wave_y)


def _liquid_neighbor_count(grid: Grid, registry: MaterialRegistry, x: int, y: int) -> int:
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if not grid.in_bounds(nx, ny):
                continue
            neighbor = grid.get_cell(nx, ny)
            neighbor_variant = registry.variant(neighbor.family_id, neighbor.variant_id)
            if neighbor_variant.matter_state == MatterState.LIQUID:
                count += 1
    return count


def _liquid_surface_outflow_direction(grid: Grid, registry: MaterialRegistry, x: int, y: int, variant) -> int:
    if variant.matter_state != MatterState.LIQUID:
        return 0
    above_is_empty = y > 0 and grid.get_cell(x, y - 1).is_empty
    downward_blocked = not _downward_exchange_available(grid, registry, x, y, variant)
    left_is_empty = x > 0 and grid.get_cell(x - 1, y).is_empty
    right_is_empty = x < grid.width - 1 and grid.get_cell(x + 1, y).is_empty
    if not above_is_empty and not downward_blocked:
        return 0
    if right_is_empty and not left_is_empty:
        return 1
    if left_is_empty and not right_is_empty:
        return -1
    return 0


def _liquid_relaxation_active(grid: Grid, registry: MaterialRegistry, x: int, y: int, variant) -> bool:
    if variant.matter_state != MatterState.LIQUID:
        return False
    pressure_head = grid.pressure[grid.index(x, y)] - AIR_PRESSURE
    if pressure_head < LIQUID_RELAXATION_HEAD_THRESHOLD:
        return False
    return _liquid_neighbor_count(grid, registry, x, y) >= LIQUID_RELAXATION_NEIGHBOR_THRESHOLD


def _base_velocity(grid: Grid, registry: MaterialRegistry, x: int, y: int, cell, variant, dt: float, *, step_seed: int) -> tuple[float, float]:
    index = grid.index(x, y)
    pressure_force_x = grid.source_force_x[index] + grid.force_wave_x[index]
    pressure_force_y = grid.source_force_y[index] + grid.force_wave_y[index]
    velocity_x = cell.vel_x * variant.mobility + pressure_force_x * dt * variant.pressure_response
    velocity_y = cell.vel_y * variant.mobility + pressure_force_y * dt * variant.pressure_response

    if variant.matter_state == MatterState.SOLID:
        wind_x, wind_y = _local_wind_velocity(grid, registry, x, y)
        wind_coupling = _wind_coupling_for_cell(grid, registry, y, cell, variant)
        velocity_x += wind_x * wind_coupling
        velocity_y += wind_y * wind_coupling * variant.wind_vertical_factor
        velocity_y += GRAVITY_ACCELERATION * dt * variant.gravity_scale
    else:
        random_gain = _thermal_random_gain(grid, registry, y, cell, variant) * _random_step_factor(dt)
        velocity_x += (_hash01(step_seed, x, y, 11) - 0.5) * random_gain
        vertical_factor = GAS_RANDOM_VERTICAL_FACTOR if variant.matter_state == MatterState.GAS else LIQUID_RANDOM_VERTICAL_FACTOR
        velocity_y += (_hash01(step_seed, x, y, 13) - 0.5) * random_gain * vertical_factor
        if variant.matter_state == MatterState.GAS:
            velocity_y += _thermal_buoyancy(grid, registry, y, cell, variant) * dt * variant.buoyancy_scale
        else:
            velocity_y += GRAVITY_ACCELERATION * dt * variant.gravity_scale

    return (velocity_x, velocity_y)


def _candidate_jitter(step_id: int, x: int, y: int, dx: int, dy: int, *, jitter_gain: float) -> float:
    return (_hash01(step_id, x + dx * 17, y + dy * 31, 97) - 0.5) * jitter_gain


def _preferred_side(step_id: int, x: int, y: int, desired_x: float) -> int:
    if desired_x > 1e-9:
        return 1
    if desired_x < -1e-9:
        return -1
    return -1 if ((step_id + x + y) & 1) == 0 else 1


def _direction_tie_break(step_id: int, x: int, y: int, dx: int, *, desired_x: float) -> float:
    return dx * _preferred_side(step_id, x, y, desired_x) * DIRECTION_TIE_BREAK_GAIN


def _sweep_range(length: int, *, reverse: bool) -> range:
    if reverse:
        return range(length - 1, -1, -1)
    return range(length)


def _directional_fallback_alignment_threshold(angle_limit_degrees: float) -> float:
    clamped_limit = max(0.0, min(180.0, float(angle_limit_degrees)))
    return cos(radians(clamped_limit))


def _limit_directional_fallback_candidates(
    candidates: list[tuple[tuple[int, int], float]],
    *,
    desired_x: float,
    desired_y: float,
    angle_limit_degrees: float,
) -> list[tuple[tuple[int, int], float]]:
    if len(candidates) <= 1:
        return candidates
    desired = _normalize(desired_x, desired_y)
    threshold = _directional_fallback_alignment_threshold(angle_limit_degrees)
    limited_candidates = [candidates[0]]
    for direction, score in candidates[1:]:
        if _direction_score(direction, desired) >= threshold - DIRECTION_FALLBACK_ALIGNMENT_EPSILON:
            limited_candidates.append((direction, score))
    return limited_candidates


def _downward_release_candidates(
    grid: Grid,
    registry: MaterialRegistry,
    x: int,
    y: int,
    variant,
    *,
    desired_x: float,
    desired_y: float,
    step_id: int,
    jitter_gain: float,
) -> list[tuple[tuple[int, int], float]] | None:
    if not variant.downward_blocked_diagonal_fallback:
        return None
    if desired_y <= 1e-9:
        return None
    if desired_y + DIRECTION_FALLBACK_ALIGNMENT_EPSILON < abs(desired_x):
        return None
    if _downward_exchange_available(grid, registry, x, y, variant):
        return None
    if grid.directional_fallback_angle_limit_degrees + DIRECTION_FALLBACK_ALIGNMENT_EPSILON < 45.0:
        return []

    desired = _normalize(desired_x, desired_y)
    preferred_side = _preferred_side(step_id, x, y, desired_x)
    directions = ((preferred_side, 1), (-preferred_side, 1))
    return [
        (
            direction,
            _direction_score(direction, desired)
            + _candidate_jitter(step_id, x, y, direction[0], direction[1], jitter_gain=jitter_gain)
            + _direction_tie_break(step_id, x, y, direction[0], desired_x=desired_x),
        )
        for direction in directions
    ]


def _sorted_candidates(
    desired_x: float,
    desired_y: float,
    *,
    step_id: int,
    x: int,
    y: int,
    jitter_gain: float,
    surface_outflow_direction: int,
) -> list[tuple[tuple[int, int], float]]:
    desired = _normalize(desired_x, desired_y)
    return sorted(
        (
            (
                (dx, dy),
                _direction_score((dx, dy), desired)
                + _candidate_jitter(step_id, x, y, dx, dy, jitter_gain=jitter_gain)
                + _direction_tie_break(step_id, x, y, dx, desired_x=desired_x)
                + (
                    LIQUID_SURFACE_SIDEFLOW_BONUS
                    if surface_outflow_direction != 0 and dx == surface_outflow_direction and dy == 0
                    else 0.0
                ),
            )
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
        ),
        key=lambda item: item[1],
        reverse=True,
    )


def _resolved_cell_after_forces(
    grid: Grid,
    registry: MaterialRegistry,
    x: int,
    y: int,
    cell,
    variant,
    dt: float,
    *,
    realized_step: tuple[int, int] | None,
    step_seed: int,
):
    updated = cell.copy()
    base_velocity_x, base_velocity_y = _base_velocity(grid, registry, x, y, cell, variant, dt, step_seed=step_seed)
    blocked_enabled = grid.blocked_impulse_enabled
    desired_x = base_velocity_x + (cell.blocked_x * variant.mobility if blocked_enabled else 0.0)
    desired_y = base_velocity_y + (cell.blocked_y * variant.mobility if blocked_enabled else 0.0)
    updated.vel_x = base_velocity_x * variant.velocity_decay
    updated.vel_y = base_velocity_y * variant.velocity_decay

    if not blocked_enabled:
        updated.blocked_x = 0.0
        updated.blocked_y = 0.0
    elif realized_step is None:
        updated.blocked_x = _decayed_blocked_impulse(desired_x)
        updated.blocked_y = _decayed_blocked_impulse(desired_y)
    else:
        updated.blocked_x = _decayed_blocked_impulse(_axis_remaining_after_step(desired_x, realized_step[0]))
        updated.blocked_y = _decayed_blocked_impulse(_axis_remaining_after_step(desired_y, realized_step[1]))

    return updated


def _apply_motion_pass(
    grid: Grid,
    registry: MaterialRegistry,
    dt: float,
    *,
    step_seed: int,
    liquids_only: bool,
    dense_only: bool = False,
    gas_like_only: bool = False,
) -> None:
    grid.copy_cells_to_scratch()
    size = grid.width * grid.height
    processed = [False] * size
    claimed = [False] * size
    x_range = _sweep_range(grid.width, reverse=bool(step_seed & 1))
    y_range = _sweep_range(grid.height, reverse=bool(step_seed & 2))

    for y in y_range:
        for x in x_range:
            current_index = grid.index(x, y)
            if processed[current_index]:
                continue

            current = grid.get_cell(x, y)
            variant = registry.variant(current.family_id, current.variant_id)
            is_gas_like_current = variant.matter_state == MatterState.GAS
            if dense_only and is_gas_like_current:
                continue
            if gas_like_only and not is_gas_like_current:
                continue
            if variant.matter_state == MatterState.GAS and not _gas_cell_can_move(grid, registry, x, y, current, variant):
                continue

            if liquids_only and not _liquid_relaxation_active(grid, registry, x, y, variant):
                continue

            if not _variant_can_translate(variant):
                updated = _resolved_cell_after_forces(
                    grid,
                    registry,
                    x,
                    y,
                    current,
                    variant,
                    dt,
                    realized_step=None,
                    step_seed=step_seed,
                )
                grid.set_cell(x, y, updated, use_scratch=True)
                processed[current_index] = True
                continue

            base_velocity_x, base_velocity_y = _base_velocity(grid, registry, x, y, current, variant, dt, step_seed=step_seed)
            desired_x = base_velocity_x + (current.blocked_x * variant.mobility if grid.blocked_impulse_enabled else 0.0)
            desired_y = base_velocity_y + (current.blocked_y * variant.mobility if grid.blocked_impulse_enabled else 0.0)
            if hypot(desired_x, desired_y) <= 1e-9:
                updated = _resolved_cell_after_forces(
                    grid,
                    registry,
                    x,
                    y,
                    current,
                    variant,
                    dt,
                    realized_step=None,
                    step_seed=step_seed,
                )
                grid.set_cell(x, y, updated, use_scratch=True)
                processed[current_index] = True
                continue
            jitter_gain = DIRECTION_JITTER_GAIN
            if hypot(desired_x, desired_y) < 0.05:
                jitter_gain = 0.0
            elif variant.matter_state == MatterState.LIQUID and not grid.liquid_brownian_enabled:
                jitter_gain = 0.0
            if grid.directional_fallback_enabled:
                candidates = _downward_release_candidates(
                    grid,
                    registry,
                    x,
                    y,
                    variant,
                    desired_x=desired_x,
                    desired_y=desired_y,
                    step_id=step_seed,
                    jitter_gain=jitter_gain,
                )
            else:
                candidates = None
            if candidates is None:
                surface_outflow_direction = _liquid_surface_outflow_direction(grid, registry, x, y, variant)
                candidates = _sorted_candidates(
                    desired_x,
                    desired_y,
                    step_id=step_seed,
                    x=x,
                    y=y,
                    jitter_gain=jitter_gain,
                    surface_outflow_direction=surface_outflow_direction,
                )
                if not grid.directional_fallback_enabled:
                    candidates = candidates[:1]
                else:
                    candidates = _limit_directional_fallback_candidates(
                        candidates,
                        desired_x=desired_x,
                        desired_y=desired_y,
                        angle_limit_degrees=grid.directional_fallback_angle_limit_degrees,
                    )
            moved = False

            for (dx, dy), score in candidates:
                if score < -0.5:
                    continue
                nx = x + dx
                ny = y + dy
                if not grid.in_bounds(nx, ny):
                    continue

                target_index = grid.index(nx, ny)
                if processed[target_index] or claimed[target_index]:
                    continue

                target = grid.get_cell(nx, ny)
                target_variant = registry.variant(target.family_id, target.variant_id)
                if not _target_can_exchange(
                    current,
                    variant,
                    target,
                    target_variant,
                    grid=grid,
                    registry=registry,
                    current_y=y,
                    target_y=ny,
                    direction=(dx, dy),
                ):
                    continue

                updated = _resolved_cell_after_forces(
                    grid,
                    registry,
                    x,
                    y,
                    current,
                    variant,
                    dt,
                    realized_step=(dx, dy),
                    step_seed=step_seed,
                )

                if target.is_empty:
                    displaced = target.copy()
                    if not current.is_empty:
                        displaced.temperature = max(target.temperature, current.temperature)
                else:
                    displaced = _resolved_cell_after_forces(
                        grid,
                        registry,
                        nx,
                        ny,
                        target,
                        target_variant,
                        dt,
                        realized_step=(-dx, -dy),
                        step_seed=step_seed,
                    )

                grid.set_cell(nx, ny, updated, use_scratch=True)
                grid.set_cell(x, y, displaced, use_scratch=True)
                claimed[target_index] = True
                processed[current_index] = True
                processed[target_index] = True
                moved = True
                break

            if moved:
                continue

            updated = _resolved_cell_after_forces(grid, registry, x, y, current, variant, dt, realized_step=None, step_seed=step_seed)
            grid.set_cell(x, y, updated, use_scratch=True)
            processed[current_index] = True

    grid.swap_buffers()


def apply_motion(grid: Grid, registry: MaterialRegistry, dt: float) -> None:
    _compute_pressure_field(grid, registry)
    _compute_source_force_field(grid, registry)
    next_wave_x, next_wave_y = _next_force_wave(grid, registry)
    _apply_motion_pass(
        grid,
        registry,
        dt,
        step_seed=grid.step_id,
        liquids_only=False,
        dense_only=True,
    )
    _apply_motion_pass(
        grid,
        registry,
        dt,
        step_seed=grid.step_id + 4096,
        liquids_only=False,
        gas_like_only=True,
    )
    grid.force_wave_x = next_wave_x
    grid.force_wave_y = next_wave_y
    grid.prev_source_force_x = grid.source_force_x[:]
    grid.prev_source_force_y = grid.source_force_y[:]
    if LIQUID_RELAXATION_PASSES <= 0:
        return
    relaxation_dt = dt / LIQUID_RELAXATION_PASSES
    for pass_index in range(LIQUID_RELAXATION_PASSES):
        _apply_motion_pass(
            grid,
            registry,
            relaxation_dt,
            step_seed=grid.step_id * (LIQUID_RELAXATION_PASSES + 1) + 1 + pass_index,
            liquids_only=True,
        )
