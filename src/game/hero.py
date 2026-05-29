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

    def update(self, dt: float, *, grid_feedback: GridFeedback | None = None) -> None:
        """Update hero physics and state."""
        self.regen_mp(dt)
        self.state_timer += dt

        feedback = grid_feedback
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

        if self.state in ("idle", "walk", "jump"):
            if self.input_left and not blocked_left:
                self.vel_x = -cfg.HERO_WALK_SPEED
                self.facing_right = False
            elif self.input_right and not blocked_right:
                self.vel_x = cfg.HERO_WALK_SPEED
                self.facing_right = True
            else:
                self.vel_x = 0.0

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

        if feedback is not None:
            if self.vel_x < 0.0 and blocked_left:
                self.vel_x = 0.0
            elif self.vel_x > 0.0 and blocked_right:
                self.vel_x = 0.0

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

        self.x += self.vel_x * dt
        self.y += self.vel_y * dt

        if feedback is not None:
            if feedback.world_width > 0.0:
                self.x = max(0.0, min(feedback.world_width - self.width / 2.0 - 1.0, self.x))

            if feedback.in_liquid and self.vel_y > 0.0:
                self.vel_y *= 0.5

            if feedback.damage > 0.0:
                self.take_damage(feedback.damage * dt)

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
    in_liquid: bool = False
    damage: float = 0.0
