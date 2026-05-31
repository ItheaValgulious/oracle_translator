"""Hero overlay entity."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.game import config as cfg

log = logging.getLogger(__name__)


CAST_DURATION = 0.3


@dataclass
class Hero:
    """Player-controlled hero with float position and grid interaction."""

    x: float = 0.0
    y: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    facing_right: bool = True
    hp: float = cfg.HERO_MAX_HP
    mp: float = cfg.HERO_MAX_MP
    on_ground: bool = False
    state: str = "idle"  # idle, walk, jump, chant, cast
    state_timer: float = 0.0
    selected_spell_idx: int = 0

    input_left: bool = False
    input_right: bool = False
    input_jump: bool = False
    input_chant_held: bool = False

    anim_frame: int = 0
    anim_timer: float = 0.0

    @property
    def width(self) -> float:
        return cfg.HERO_WIDTH

    @property
    def height(self) -> float:
        return cfg.HERO_HEIGHT

    @property
    def left(self) -> float:
        return self.x - self.width / 2.0

    @property
    def right(self) -> float:
        return self.x + self.width / 2.0

    @property
    def bottom(self) -> float:
        return self.y

    @property
    def top(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    @property
    def is_alive(self) -> bool:
        return self.hp > 0.0

    def reset(self, x: float, y: float) -> None:
        """Reset hero to a spawn position."""
        self.x = x
        self.y = y
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.hp = cfg.HERO_MAX_HP
        self.mp = cfg.HERO_MAX_MP
        self.on_ground = False
        self.state = "idle"
        self.state_timer = 0.0
        self.selected_spell_idx = 0

    def take_damage(self, amount: float) -> None:
        self.hp = max(0.0, self.hp - amount)

    def heal(self, amount: float) -> None:
        self.hp = min(cfg.HERO_MAX_HP, self.hp + amount)

    def consume_mp(self, amount: float) -> bool:
        if self.mp >= amount:
            self.mp -= amount
            return True
        return False

    def regen_mp(self, dt: float) -> None:
        self.mp = min(cfg.HERO_MAX_MP, self.mp + cfg.MP_REGEN_PER_SEC * dt)

    def get_placeholder_cells(self) -> list[tuple[int, int]]:
        """Return integer grid cells occupied by this hero."""
        cells: list[tuple[int, int]] = []
        for iy in range(int(self.bottom), int(self.top) + 1):
            for ix in range(int(self.left), int(self.right) + 1):
                cells.append((ix, iy))
        return cells

    def update(
        self,
        dt: float,
        *,
        grid_feedback: GridFeedback | None = None,
        local_snapshot: LocalCellSnapshot | None = None,
        world_width: float | None = None,
    ) -> None:
        """Update hero physics and state."""
        self.regen_mp(dt)
        self.state_timer += dt

        feedback = grid_feedback
        if local_snapshot is not None:
            feedback = feedback_from_snapshot(
                snapshot=local_snapshot,
                center_x=self.x,
                bottom_y=self.y,
                width=self.width,
                height=self.height,
                world_width=world_width if world_width is not None else (grid_feedback.world_width if grid_feedback else 0.0),
            )
        blocked_below = bool(feedback and feedback.blocked_below)
        blocked_left = bool(feedback and feedback.blocked_left)
        blocked_right = bool(feedback and feedback.blocked_right)
        blocked_up = bool(feedback and feedback.blocked_up)

        if self.input_chant_held and self.state != "cast":
            if self.state in ("idle", "walk", "jump"):
                self.state = "chant"
                self.state_timer = 0.0

        if self.state == "chant":
            self.vel_x = 0.0
            if not self.input_chant_held:
                self.state = "cast"
                self.state_timer = 0.0

        if self.state == "cast" and self.state_timer >= CAST_DURATION:
            self.state = "idle"
            self.state_timer = 0.0

        move_dir = 0
        if self.state in ("idle", "walk", "jump"):
            if self.input_left and not self.input_right:
                move_dir = -1
                self.facing_right = False
            elif self.input_right and not self.input_left:
                move_dir = 1
                self.facing_right = True

        if self.input_jump and self.on_ground and self.state in ("idle", "walk") and not blocked_up:
            self.vel_y = -cfg.HERO_JUMP_VELOCITY
            self.on_ground = False
            self.state = "jump"
            self.state_timer = 0.0

        if not self.on_ground and self.state != "chant":
            old_vy = self.vel_y
            self.vel_y += cfg.HERO_GRAVITY * dt
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "[hero] gravity: vel_y %.3f -> %.3f (dt=%.4f on_ground=%s state=%s)",
                    old_vy,
                    self.vel_y,
                    dt,
                    self.on_ground,
                    self.state,
                )

        step_climbed = False
        if feedback is not None:
            # Landing / lift-off detection runs before de-stuck and slope climb.
            if self.vel_y < 0.0 and blocked_up:
                self.vel_y = 0.0
            elif self.vel_y >= 0.0 and blocked_below:
                self.vel_y = 0.0
                if not self.on_ground:
                    log.debug("[hero] LANDING: y=%.2f vel_y=%.2f -> on_ground=True", self.y, self.vel_y)
                self.on_ground = True
            elif not blocked_below:
                if self.on_ground:
                    log.debug("[hero] LIFT_OFF: y=%.2f vel_y=%.2f -> on_ground=False", self.y, self.vel_y)
                self.on_ground = False

            # Slope climbing: blocked below + wall ahead at foot level -> lift 1 cell.
            if blocked_below and move_dir != 0 and not blocked_up:
                is_step = False
                if move_dir > 0 and feedback.blocked_right_ahead:
                    is_step = True
                elif move_dir < 0 and feedback.blocked_left_ahead:
                    is_step = True
                if is_step:
                    self.y -= 1.0
                    self.vel_y = 0.0
                    blocked_below = False
                    step_climbed = True
                    self.on_ground = True

            # De-stuck: body is embedded while the head row is free -> push up 1 cell.
            if feedback.embedded and not blocked_up and (blocked_below or self.on_ground or self.vel_y >= 0.0):
                self.y -= 1.0
                if self.vel_y > 0.0:
                    self.vel_y = 0.0
                blocked_below = False

        if self.state in ("idle", "walk", "jump"):
            blocked_by_wall = (
                (move_dir < 0 and blocked_left and not step_climbed)
                or (move_dir > 0 and blocked_right and not step_climbed)
            )
            self.vel_x = 0.0 if move_dir == 0 or blocked_by_wall else move_dir * cfg.HERO_WALK_SPEED

        self.x += self.vel_x * dt
        self.y += self.vel_y * dt

        if feedback is not None:
            if feedback.world_width > 0.0:
                self.x = max(0.0, min(feedback.world_width - self.width / 2.0 - 1.0, self.x))

            if feedback.in_liquid and self.vel_y > 0.0:
                self.vel_y *= 0.5

            if feedback.damage > 0.0:
                self.take_damage(feedback.damage * cfg.HERO_PLACEHOLDER_DAMAGE_SCALE)

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "[hero] feedback: state=%s y=%.2f vel_y=%.3f on_ground=%s "
                    "blocked_below=%s blocked_up=%s blocked_left=%s blocked_right=%s",
                    self.state,
                    self.y,
                    self.vel_y,
                    self.on_ground,
                    feedback.blocked_below,
                    feedback.blocked_up,
                    feedback.blocked_left,
                    feedback.blocked_right,
                )

        if self.state == "jump" and self.on_ground:
            self.state = "idle"
            self.state_timer = 0.0
        elif feedback is not None and self.state in ("idle", "walk") and not self.on_ground and abs(self.vel_y) > 0.0:
            self.state = "jump"
            self.state_timer = 0.0
        elif self.state in ("idle", "walk"):
            self.state = "walk" if self.vel_x != 0.0 else "idle"

        self.anim_timer += dt
        if self.anim_timer >= 1.0 / cfg.ENTITY_ANIM_FPS:
            self.anim_timer = 0.0
            self.anim_frame = (self.anim_frame + 1) % 4


@dataclass
class GridFeedback:
    """Feedback from the grid to the hero after a simulation step."""

    world_width: float = 0.0
    blocked_below: bool = False
    blocked_left: bool = False
    blocked_right: bool = False
    blocked_up: bool = False
    blocked_left_ahead: bool = False   # wall at foot level on left
    blocked_right_ahead: bool = False  # wall at foot level on right
    embedded: bool = False             # row above feet has terrain (body embedded)
    in_liquid: bool = False
    damage: float = 0.0


@dataclass
class LocalCellSnapshot:
    """GPU-read local cell snapshot around an entity."""

    entity_id: str
    origin_x: int
    origin_y: int
    width: int
    height: int
    variant_indices: tuple[tuple[int, ...], ...]
    velocities: tuple[tuple[tuple[float, float], ...], ...]
    temperatures: tuple[tuple[float, ...], ...]
    variant_families: tuple[str, ...]
    empty_variant_index: int
    tick_id: int = 0
    age_frames: int = 0

    def _local_coords(self, world_x: int, world_y: int) -> tuple[int, int] | None:
        local_x = world_x - self.origin_x
        local_y = world_y - self.origin_y
        if local_x < 0 or local_y < 0 or local_x >= self.width or local_y >= self.height:
            return None
        if local_y >= len(self.variant_indices):
            return None
        row = self.variant_indices[local_y]
        if local_x >= len(row):
            return None
        return (local_x, local_y)

    def variant_index_at_world(self, world_x: int, world_y: int) -> int | None:
        coords = self._local_coords(world_x, world_y)
        if coords is None:
            return None
        local_x, local_y = coords
        return self.variant_indices[local_y][local_x]

    def family_id_at_world(self, world_x: int, world_y: int) -> str | None:
        variant_index = self.variant_index_at_world(world_x, world_y)
        if variant_index is None or variant_index < 0 or variant_index >= len(self.variant_families):
            return None
        return self.variant_families[variant_index]

    def temperature_at_world(self, world_x: int, world_y: int) -> float | None:
        coords = self._local_coords(world_x, world_y)
        if coords is None:
            return None
        local_x, local_y = coords
        if local_y >= len(self.temperatures):
            return None
        row = self.temperatures[local_y]
        if local_x >= len(row):
            return None
        return row[local_x]

    def velocity_at_world(self, world_x: int, world_y: int) -> tuple[float, float] | None:
        coords = self._local_coords(world_x, world_y)
        if coords is None:
            return None
        local_x, local_y = coords
        if local_y >= len(self.velocities):
            return None
        row = self.velocities[local_y]
        if local_x >= len(row):
            return None
        return row[local_x]


def _solid_cell(snapshot: LocalCellSnapshot | None, world_x: int, world_y: int) -> bool:
    if snapshot is None:
        return False
    variant_index = snapshot.variant_index_at_world(world_x, world_y)
    return variant_index is not None and variant_index != snapshot.empty_variant_index


def feedback_from_snapshot(
    *,
    snapshot: LocalCellSnapshot | None,
    center_x: float,
    bottom_y: float,
    width: float,
    height: float,
    world_width: float,
) -> GridFeedback:
    feedback = GridFeedback(world_width=world_width)
    if snapshot is None:
        return feedback

    left = int(center_x - width / 2.0)
    right = int(center_x + width / 2.0)
    bottom = int(bottom_y)
    top = int(bottom_y + height)
    foot_y = top + 1
    head_y = bottom - 1
    left_wall_x = left - 1
    right_wall_x = right + 1

    feedback.blocked_below = any(_solid_cell(snapshot, x, foot_y) for x in range(left, right + 1))
    feedback.blocked_up = any(_solid_cell(snapshot, x, head_y) for x in range(left, right + 1))
    feedback.blocked_left = any(_solid_cell(snapshot, left_wall_x, y) for y in range(bottom, top + 1))
    feedback.blocked_right = any(_solid_cell(snapshot, right_wall_x, y) for y in range(bottom, top + 1))

    clearance_y = max(bottom, top - 1)
    feedback.blocked_left_ahead = _solid_cell(snapshot, left_wall_x, top)
    feedback.blocked_right_ahead = _solid_cell(snapshot, right_wall_x, top)
    feedback.embedded = any(
        _solid_cell(snapshot, x, y)
        for x in range(left, right + 1)
        for y in (top, clearance_y)
    )

    hazard = 0.0
    in_liquid = False
    for x in range(left, right + 1):
        for y in range(bottom, top + 1):
            family_id = snapshot.family_id_at_world(x, y)
            if family_id is None or family_id == "empty":
                continue
            temperature = snapshot.temperature_at_world(x, y)
            if family_id in {"water", "tar", "acid", "magic_acid", "poison"}:
                in_liquid = True
            if family_id in {"fire", "acid", "magic_acid"}:
                hazard += 0.1
            velocity = snapshot.velocity_at_world(x, y)
            if velocity is not None:
                vel_x, vel_y = velocity
                speed = abs(vel_x) + abs(vel_y)
                if speed >= 30.0:
                    hazard += min(0.18, (speed - 30.0) * 0.0015)
            if temperature is None:
                continue
            if temperature >= 80.0:
                hazard += min(0.24, (temperature - 80.0) * 0.00045)
            if temperature <= -35.0:
                hazard += min(0.12, (-35.0 - temperature) * 0.00018)
    feedback.in_liquid = in_liquid
    feedback.damage = min(0.35, hazard)
    return feedback
