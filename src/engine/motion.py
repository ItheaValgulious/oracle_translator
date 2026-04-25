from __future__ import annotations

from math import copysign, hypot, sqrt

from .grid import Grid
from .types import MaterialRegistry, MatterState, MotionMode


AIR_PRESSURE = 1.0
PRESSURE_RELAXATION = 0.22
PRESSURE_FORCE_SCALE = 0.55
LIQUID_RELAXATION_PASSES = 0
FORCE_WAVE_DECAY = 0.82
LIQUID_LATERAL_PRESSURE_BOOST = 1.35
LIQUID_BLOCKED_LATERAL_BOOST = 2.6
LIQUID_SURFACE_SIDEFLOW_BONUS = 0.8
LIQUID_PRESSURE_HEAD_BOOST = 0.25
LIQUID_PRESSURE_HEAD_BOOST_CAP = 2.5
LIQUID_RELAXATION_HEAD_THRESHOLD = 1.5
LIQUID_RELAXATION_NEIGHBOR_THRESHOLD = 2
LIQUID_VERTICAL_PRESSURE_SCALE = 0.8
GAS_PRESSURE_FORCE_SCALE = 0.65
GRAVITY_ACCELERATION = 1.2
GAS_BUOYANCY_SCALE = 10.0
VELOCITY_DECAY = 0.92
STATIC_VELOCITY_DECAY = 0.96
INTENT_DECAY = 0.72
LIQUID_RANDOM_GAIN = 0.012
LIQUID_RANDOM_VERTICAL_FACTOR = 0.18
GAS_RANDOM_GAIN = 0.42
GAS_RANDOM_VERTICAL_FACTOR = 0.85
DIRECTION_JITTER_GAIN = 0.06
KELVIN_OFFSET = 273.15
MIN_THERMAL_KELVIN = 80.0
EMPTY_MOTION_TEMPERATURE_THRESHOLD = 0.5
GAS_THERMAL_TEMPERATURE_SPAN = 120.0
LIQUID_THERMAL_TEMPERATURE_SPAN = 400.0
GAS_RANDOM_FLOOR_FACTOR = 0.35
LIQUID_RANDOM_FLOOR_FACTOR = 0.15


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


def _ambient_air_temperature(registry: MaterialRegistry) -> float:
    return registry.variant("empty", "empty").base_temperature


def _temperature_kelvin(temperature: float) -> float:
    return max(MIN_THERMAL_KELVIN, temperature + KELVIN_OFFSET)


def _temperature_motion_factor(temperature: float, ambient_temperature: float, *, span: float) -> float:
    return min(1.0, max(0.0, temperature - ambient_temperature) / max(span, 0.001))


def _effective_density(cell, variant, registry: MaterialRegistry) -> float:
    if cell.is_empty or variant.matter_state == MatterState.GAS:
        ambient_kelvin = _temperature_kelvin(_ambient_air_temperature(registry))
        return variant.density * ambient_kelvin / _temperature_kelvin(cell.temperature)
    return variant.density


def _thermal_random_gain(grid: Grid, registry: MaterialRegistry, cell, variant) -> float:
    ambient_temperature = _ambient_air_temperature(registry)
    if cell.is_empty:
        return GAS_RANDOM_GAIN * _temperature_motion_factor(
            cell.temperature,
            ambient_temperature,
            span=GAS_THERMAL_TEMPERATURE_SPAN,
        )
    if variant.matter_state == MatterState.GAS:
        factor = _temperature_motion_factor(
            cell.temperature,
            ambient_temperature,
            span=GAS_THERMAL_TEMPERATURE_SPAN,
        )
        return GAS_RANDOM_GAIN * (GAS_RANDOM_FLOOR_FACTOR + (1.0 - GAS_RANDOM_FLOOR_FACTOR) * factor)
    if variant.matter_state == MatterState.LIQUID and grid.liquid_brownian_enabled:
        factor = _temperature_motion_factor(
            cell.temperature,
            ambient_temperature,
            span=LIQUID_THERMAL_TEMPERATURE_SPAN,
        )
        return LIQUID_RANDOM_GAIN * (LIQUID_RANDOM_FLOOR_FACTOR + (1.0 - LIQUID_RANDOM_FLOOR_FACTOR) * factor)
    return 0.0


def _thermal_buoyancy(registry: MaterialRegistry, cell, variant) -> float:
    if not (cell.is_empty or variant.matter_state == MatterState.GAS):
        return 0.0
    ambient_density = registry.variant("empty", "empty").density
    return (_effective_density(cell, variant, registry) - ambient_density) * GAS_BUOYANCY_SCALE


def _empty_cell_can_move(current, registry: MaterialRegistry) -> bool:
    ambient_temperature = _ambient_air_temperature(registry)
    return (
        abs(current.temperature - ambient_temperature) > EMPTY_MOTION_TEMPERATURE_THRESHOLD
        or hypot(current.vel_x, current.vel_y) > 0.01
        or hypot(current.blocked_x, current.blocked_y) > 0.01
    )


def _target_is_lighter(current, current_variant, target, target_variant, *, registry: MaterialRegistry) -> bool:
    if current.is_empty:
        return target.is_empty
    if target.is_empty:
        return True
    current_rank = _state_rank(current_variant.matter_state)
    target_rank = _state_rank(target_variant.matter_state)
    if current_rank != target_rank:
        return target_rank < current_rank
    return _effective_density(target, target_variant, registry) < _effective_density(current, current_variant, registry)


def _axis_remaining_after_step(desired_component: float, realized_component: float) -> float:
    if desired_component == 0.0:
        return 0.0
    if realized_component == 0.0 or desired_component * realized_component <= 0.0:
        return desired_component
    return copysign(max(0.0, abs(desired_component) - abs(realized_component)), desired_component)


def _downward_exchange_available(grid: Grid, registry: MaterialRegistry, x: int, y: int, variant) -> bool:
    current = grid.get_cell(x, y)
    ny = y + 1
    if not grid.in_bounds(x, ny):
        return False
    target = grid.get_cell(x, ny)
    target_variant = registry.variant(target.family_id, target.variant_id)
    return _target_is_lighter(current, variant, target, target_variant, registry=registry)


def _compute_pressure_field(grid: Grid, registry: MaterialRegistry) -> None:
    next_pressure = [AIR_PRESSURE for _ in range(grid.width * grid.height)]
    previous_pressure = grid.pressure

    for y in range(grid.height):
        for x in range(grid.width):
            index = grid.index(x, y)
            cell = grid.get_cell(x, y)
            variant = registry.variant(cell.family_id, cell.variant_id)
            if cell.is_empty:
                next_pressure[index] = previous_pressure[index] + (AIR_PRESSURE - previous_pressure[index]) * PRESSURE_RELAXATION
            elif variant.matter_state == MatterState.GAS:
                next_pressure[index] = AIR_PRESSURE + _effective_density(cell, variant, registry)
            else:
                next_pressure[index] = previous_pressure[index] + (AIR_PRESSURE - previous_pressure[index]) * PRESSURE_RELAXATION

    for x in range(grid.width):
        for y in range(grid.height):
            index = grid.index(x, y)
            cell = grid.get_cell(x, y)
            variant = registry.variant(cell.family_id, cell.variant_id)
            if variant.matter_state != MatterState.LIQUID or cell.is_empty:
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
            if cell.is_empty:
                continue
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
            if cell.is_empty or variant.matter_state != MatterState.LIQUID:
                continue

            incoming_x = 0.0
            incoming_y = 0.0
            if x > 0:
                left_index = grid.index(x - 1, y)
                left_cell = grid.get_cell(x - 1, y)
                left_variant = registry.variant(left_cell.family_id, left_cell.variant_id)
                if not left_cell.is_empty and left_variant.matter_state == MatterState.LIQUID:
                    incoming_x += max(0.0, grid.force_wave_x[left_index]) * FORCE_WAVE_DECAY
            if x < grid.width - 1:
                right_index = grid.index(x + 1, y)
                right_cell = grid.get_cell(x + 1, y)
                right_variant = registry.variant(right_cell.family_id, right_cell.variant_id)
                if not right_cell.is_empty and right_variant.matter_state == MatterState.LIQUID:
                    incoming_x += min(0.0, grid.force_wave_x[right_index]) * FORCE_WAVE_DECAY
            if y > 0:
                above_index = grid.index(x, y - 1)
                above_cell = grid.get_cell(x, y - 1)
                above_variant = registry.variant(above_cell.family_id, above_cell.variant_id)
                if not above_cell.is_empty and above_variant.matter_state == MatterState.LIQUID:
                    incoming_y += max(0.0, grid.force_wave_y[above_index]) * FORCE_WAVE_DECAY
            if y < grid.height - 1:
                below_index = grid.index(x, y + 1)
                below_cell = grid.get_cell(x, y + 1)
                below_variant = registry.variant(below_cell.family_id, below_cell.variant_id)
                if not below_cell.is_empty and below_variant.matter_state == MatterState.LIQUID:
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
            if neighbor_variant.matter_state == MatterState.LIQUID and not neighbor.is_empty:
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
    velocity_x = cell.vel_x + pressure_force_x * dt
    velocity_y = cell.vel_y + pressure_force_y * dt

    if variant.motion_mode == MotionMode.POWDER:
        velocity_y += GRAVITY_ACCELERATION * dt
    elif variant.motion_mode == MotionMode.FLUID:
        random_gain = _thermal_random_gain(grid, registry, cell, variant)
        velocity_x += (_hash01(step_seed, x, y, 11) - 0.5) * random_gain
        vertical_factor = GAS_RANDOM_VERTICAL_FACTOR if (cell.is_empty or variant.matter_state == MatterState.GAS) else LIQUID_RANDOM_VERTICAL_FACTOR
        velocity_y += (_hash01(step_seed, x, y, 13) - 0.5) * random_gain * vertical_factor
        if cell.is_empty or variant.matter_state == MatterState.GAS:
            velocity_y += _thermal_buoyancy(registry, cell, variant) * dt
        else:
            velocity_y += GRAVITY_ACCELERATION * dt

    return (velocity_x, velocity_y)


def _candidate_jitter(step_id: int, x: int, y: int, dx: int, dy: int, *, jitter_gain: float) -> float:
    return (_hash01(step_id, x + dx * 17, y + dy * 31, 97) - 0.5) * jitter_gain


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

    if variant.motion_mode == MotionMode.STATIC:
        updated.vel_x = base_velocity_x * STATIC_VELOCITY_DECAY
        updated.vel_y = base_velocity_y * STATIC_VELOCITY_DECAY
        if blocked_enabled:
            updated.blocked_x *= INTENT_DECAY
            updated.blocked_y *= INTENT_DECAY
        else:
            updated.blocked_x = 0.0
            updated.blocked_y = 0.0
        return updated

    desired_x = base_velocity_x + (cell.blocked_x if blocked_enabled else 0.0)
    desired_y = base_velocity_y + (cell.blocked_y if blocked_enabled else 0.0)
    updated.vel_x = base_velocity_x * VELOCITY_DECAY
    updated.vel_y = base_velocity_y * VELOCITY_DECAY

    if not blocked_enabled:
        updated.blocked_x = 0.0
        updated.blocked_y = 0.0
    elif realized_step is None:
        updated.blocked_x = desired_x * INTENT_DECAY
        updated.blocked_y = desired_y * INTENT_DECAY
    else:
        updated.blocked_x = _axis_remaining_after_step(desired_x, realized_step[0])
        updated.blocked_y = _axis_remaining_after_step(desired_y, realized_step[1])

    return updated


def _apply_motion_pass(grid: Grid, registry: MaterialRegistry, dt: float, *, step_seed: int, liquids_only: bool) -> None:
    grid.copy_cells_to_scratch()
    size = grid.width * grid.height
    processed = [False] * size
    claimed = [False] * size

    for y in range(grid.height):
        for x in range(grid.width):
            current_index = grid.index(x, y)
            if processed[current_index]:
                continue

            current = grid.get_cell(x, y)
            variant = registry.variant(current.family_id, current.variant_id)
            if current.is_empty and not _empty_cell_can_move(current, registry):
                processed[current_index] = True
                continue

            if liquids_only and not _liquid_relaxation_active(grid, registry, x, y, variant):
                grid.set_cell(x, y, current.copy(), use_scratch=True)
                processed[current_index] = True
                continue

            if variant.motion_mode == MotionMode.STATIC:
                updated = _resolved_cell_after_forces(grid, registry, x, y, current, variant, dt, realized_step=None, step_seed=step_seed)
                grid.set_cell(x, y, updated, use_scratch=True)
                processed[current_index] = True
                continue

            base_velocity_x, base_velocity_y = _base_velocity(grid, registry, x, y, current, variant, dt, step_seed=step_seed)
            desired_x = base_velocity_x + (current.blocked_x if grid.blocked_impulse_enabled else 0.0)
            desired_y = base_velocity_y + (current.blocked_y if grid.blocked_impulse_enabled else 0.0)
            jitter_gain = DIRECTION_JITTER_GAIN
            if hypot(desired_x, desired_y) < 0.05:
                jitter_gain = 0.0
            elif variant.matter_state == MatterState.LIQUID and not grid.liquid_brownian_enabled:
                jitter_gain = 0.0
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
                if not _target_is_lighter(current, variant, target, target_variant, registry=registry):
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
