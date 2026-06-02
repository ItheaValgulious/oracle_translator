"""Enemy entities: EnemyBase, EnemyA (archer), EnemyB (flying bomber), EnemyC (boss)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.game import config as cfg
from src.game.hero import LocalCellSnapshot, feedback_from_snapshot

if TYPE_CHECKING:
    from src.game.hero import GridFeedback

log = logging.getLogger(__name__)


def _stable_token(text: str) -> int:
    return sum((index + 1) * ord(ch) for index, ch in enumerate(text))


def _patrol_direction_for_cycle(entity_id: str, cycle: int) -> float:
    return 1.0 if ((_stable_token(entity_id) + cycle * 97) & 1) == 0 else -1.0


def _patrol_duration_for_cycle(entity_id: str, cycle: int) -> float:
    span = max(0.1, cfg.ENEMY_A_PATROL_MAX_DURATION - cfg.ENEMY_A_PATROL_MIN_DURATION)
    seed = (_stable_token(entity_id) * 131 + cycle * 977) & 0xFFFF
    return cfg.ENEMY_A_PATROL_MIN_DURATION + span * (seed / 0xFFFF)


@dataclass
class EnemyBase:
    """Base class for all enemies."""

    entity_id: str = ""
    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0
    hp: float = 0.0
    max_hp: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    facing_right: bool = True
    state: str = "idle"
    state_timer: float = 0.0
    anim_frame: int = 0
    anim_timer: float = 0.0
    is_alive: bool = True
    damage_flash_timer: float = 0.0
    biome: str = "plains"
    on_ground: bool = False

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

    def take_damage(self, amount: float) -> None:
        self.hp = max(0.0, self.hp - amount)
        self.damage_flash_timer = cfg.DAMAGE_FLASH_DURATION
        if self.hp <= 0.0:
            self.is_alive = False

    def heal(self, amount: float) -> None:
        self.hp = min(self.max_hp, self.hp + amount)

    def distance_to(self, other_x: float, other_y: float) -> float:
        dx = other_x - self.x
        dy = other_y - self.y
        return (dx * dx + dy * dy) ** 0.5

    def update(
        self,
        dt: float,
        *,
        grid_feedback=None,
        local_snapshot: LocalCellSnapshot | None = None,
        world_width: float | None = None,
        hero_x=0.0,
        hero_y=0.0,
    ) -> None:
        del grid_feedback, local_snapshot, world_width, hero_x, hero_y
        self.state_timer += dt
        self.anim_timer += dt
        if self.anim_timer >= 1.0 / cfg.ENTITY_ANIM_FPS:
            self.anim_timer = 0.0
            self.anim_frame = (self.anim_frame + 1) % 4
        if self.damage_flash_timer > 0.0:
            self.damage_flash_timer -= dt


@dataclass
class EnemyA(EnemyBase):
    """Ground archer: patrols, shoots arrows at hero when in range."""

    arrow_cooldown: float = 0.0
    patrol_direction: float = 1.0
    patrol_origin_x: float = 0.0
    patrol_time_remaining: float = 0.0
    patrol_cycle: int = 0
    _last_step_key: tuple[int, int] | None = None

    @staticmethod
    def create(entity_id: str, x: float, y: float, biome: str = "plains") -> EnemyA:
        return EnemyA(
            entity_id=entity_id,
            x=x,
            y=y,
            width=cfg.ENEMY_A_WIDTH,
            height=cfg.ENEMY_A_HEIGHT,
            hp=cfg.ENEMY_A_MAX_HP,
            max_hp=cfg.ENEMY_A_MAX_HP,
            state="patrol",
            biome=biome,
            patrol_origin_x=x,
            patrol_direction=_patrol_direction_for_cycle(entity_id, 0),
            patrol_time_remaining=_patrol_duration_for_cycle(entity_id, 0),
        )

    def _advance_patrol(self, *, reverse: bool = False) -> None:
        self.patrol_cycle += 1
        if reverse:
            self.patrol_direction *= -1.0
        else:
            self.patrol_direction = _patrol_direction_for_cycle(self.entity_id, self.patrol_cycle)
        self.patrol_time_remaining = _patrol_duration_for_cycle(self.entity_id, self.patrol_cycle)

    def update(
        self,
        dt: float,
        *,
        grid_feedback=None,
        local_snapshot: LocalCellSnapshot | None = None,
        world_width: float | None = None,
        hero_x=0.0,
        hero_y=0.0,
    ) -> None:
        super().update(
            dt,
            grid_feedback=grid_feedback,
            local_snapshot=local_snapshot,
            world_width=world_width,
            hero_x=hero_x,
            hero_y=hero_y,
        )
        if not self.is_alive:
            return

        feedback = grid_feedback
        if local_snapshot is not None:
            feedback = feedback_from_snapshot(
                snapshot=local_snapshot,
                center_x=self.x,
                bottom_y=self.y,
                width=self.width,
                height=self.height,
                world_width=(
                    float(world_width)
                    if world_width is not None
                    else float(getattr(grid_feedback, "world_width", 0.0) or 0.0)
                ),
            )

        blocked_left = bool(feedback and feedback.blocked_left)
        blocked_right = bool(feedback and feedback.blocked_right)
        blocked_below = bool(feedback and feedback.blocked_below)
        blocked_up = bool(feedback and feedback.blocked_up)
        blocked_left_ahead = bool(feedback and feedback.blocked_left_ahead)
        blocked_right_ahead = bool(feedback and feedback.blocked_right_ahead)
        embedded = bool(feedback and feedback.embedded)

        self.arrow_cooldown -= dt
        dist = self.distance_to(hero_x, hero_y)
        self.facing_right = hero_x > self.x
        self.state = "attack" if dist < cfg.ENEMY_A_DETECT_RANGE else "patrol"

        move_dir = 0.0
        if self.state == "patrol":
            self.patrol_time_remaining -= dt
            if self.patrol_time_remaining <= 0.0:
                self._advance_patrol()
            if abs(self.x - self.patrol_origin_x) > cfg.ENEMY_A_PATROL_RANGE:
                self._advance_patrol(reverse=True)
            move_dir = self.patrol_direction
            self.facing_right = move_dir > 0.0

        if self.vel_y < 0.0 and blocked_up:
            self.vel_y = 0.0
        if blocked_below:
            self.vel_y = 0.0
            self.on_ground = True
        else:
            self.on_ground = False
            self.vel_y += cfg.HERO_GRAVITY * dt

        step_climbed = False
        if blocked_below and move_dir != 0.0 and not blocked_up:
            can_step = (
                (move_dir > 0.0 and blocked_right_ahead)
                or (move_dir < 0.0 and blocked_left_ahead)
            )
            step_key = (1 if move_dir > 0.0 else -1, int(self.x))
            if can_step and step_key != self._last_step_key:
                self.y -= min(float(cfg.ENEMY_SLOPE_MAX_CLIMB), 1.0)
                self.vel_y = 0.0
                self.on_ground = True
                step_climbed = True
                self._last_step_key = step_key
        else:
            self._last_step_key = None

        if embedded and not blocked_up and (blocked_below or self.on_ground or self.vel_y >= 0.0):
            self.y -= cfg.ENEMY_DESTUCK_LIFT
            if self.vel_y > 0.0:
                self.vel_y = 0.0

        blocked_by_wall = (
            (move_dir < 0.0 and blocked_left and not step_climbed)
            or (move_dir > 0.0 and blocked_right and not step_climbed)
        )
        if self.state == "patrol" and blocked_by_wall:
            self._advance_patrol(reverse=True)
            move_dir = self.patrol_direction
            self.facing_right = move_dir > 0.0

        self.vel_x = 0.0 if self.state == "attack" else move_dir * cfg.ENEMY_A_PATROL_SPEED
        self.x += self.vel_x * dt
        self.y += self.vel_y * dt

    def should_fire_arrow(self) -> bool:
        return self.state == "attack" and self.arrow_cooldown <= 0.0

    def fire_arrow(self) -> None:
        self.arrow_cooldown = cfg.ENEMY_A_ARROW_INTERVAL


@dataclass
class EnemyB(EnemyBase):
    """Flying bomber: hovers, charges at hero, explodes on terrain contact."""

    hover_y: float = 0.0

    @staticmethod
    def create(entity_id: str, x: float, y: float, biome: str = "hillside") -> EnemyB:
        return EnemyB(
            entity_id=entity_id,
            x=x,
            y=y,
            width=cfg.ENEMY_B_WIDTH,
            height=cfg.ENEMY_B_HEIGHT,
            hp=cfg.ENEMY_B_MAX_HP,
            max_hp=cfg.ENEMY_B_MAX_HP,
            state="hover",
            biome=biome,
            hover_y=y,
        )

    def update(
        self,
        dt: float,
        *,
        grid_feedback=None,
        local_snapshot: LocalCellSnapshot | None = None,
        world_width: float | None = None,
        hero_x=0.0,
        hero_y=0.0,
    ) -> None:
        self.state_timer += dt
        self.anim_timer += dt
        if self.anim_timer >= 1.0 / cfg.ENTITY_ANIM_FPS:
            self.anim_timer = 0.0
            self.anim_frame = (self.anim_frame + 1) % 4
        if self.damage_flash_timer > 0.0:
            self.damage_flash_timer -= dt
        if not self.is_alive:
            return

        dist = self.distance_to(hero_x, hero_y)
        self.facing_right = hero_x > self.x
        self.state = "charge" if dist < cfg.ENEMY_B_DETECT_RANGE else "hover"

        if self.state == "hover":
            dy = self.hover_y - self.y
            self.vel_y = dy * 0.5
            self.vel_x = cfg.ENEMY_B_DRIFT_SPEED * (1 if self.facing_right else -1)
        else:
            dx = hero_x - self.x
            dy = hero_y - self.y
            dist_sq = dx * dx + dy * dy
            if dist_sq > 0.0:
                norm = dist_sq ** 0.5
                self.vel_x = cfg.ENEMY_B_CHARGE_SPEED * dx / norm
                self.vel_y = cfg.ENEMY_B_CHARGE_SPEED * dy / norm

        self.x += self.vel_x * dt
        self.y += self.vel_y * dt

        feedback = grid_feedback
        if local_snapshot is not None:
            feedback = feedback_from_snapshot(
                snapshot=local_snapshot,
                center_x=self.x,
                bottom_y=self.y,
                width=self.width,
                height=self.height,
                world_width=(
                    float(world_width)
                    if world_width is not None
                    else float(getattr(grid_feedback, "world_width", 0.0) or 0.0)
                ),
            )

        if feedback is not None:
            if feedback.blocked_left or feedback.blocked_right or feedback.blocked_below or feedback.blocked_up:
                self.is_alive = False
                return
        if self.overlaps_hero(hero_x, hero_y):
            self.is_alive = False

    def overlaps_hero(self, hero_x: float, hero_y: float) -> bool:
        hero_left = hero_x - cfg.HERO_WIDTH / 2.0
        hero_right = hero_x + cfg.HERO_WIDTH / 2.0
        hero_bottom = hero_y
        hero_top = hero_y + cfg.HERO_HEIGHT
        return (
            self.left < hero_right
            and self.right > hero_left
            and self.bottom < hero_top
            and self.top > hero_bottom
        )

    @property
    def should_explode(self) -> bool:
        return not self.is_alive


@dataclass
class EnemyC(EnemyBase):
    """Boss: three-attack cycle (oil spray -> fireball -> collapse)."""

    attack_phase: int = 0
    phase_timer: float = 0.0

    @staticmethod
    def create(entity_id: str, x: float, y: float, biome: str = "underground") -> EnemyC:
        return EnemyC(
            entity_id=entity_id,
            x=x,
            y=y,
            width=cfg.ENEMY_C_WIDTH,
            height=cfg.ENEMY_C_HEIGHT,
            hp=cfg.ENEMY_C_MAX_HP,
            max_hp=cfg.ENEMY_C_MAX_HP,
            state="attack",
            biome=biome,
        )

    def update(
        self,
        dt: float,
        *,
        grid_feedback=None,
        local_snapshot: LocalCellSnapshot | None = None,
        world_width: float | None = None,
        hero_x=0.0,
        hero_y=0.0,
    ) -> None:
        super().update(
            dt,
            grid_feedback=grid_feedback,
            local_snapshot=local_snapshot,
            world_width=world_width,
            hero_x=hero_x,
            hero_y=hero_y,
        )
        if not self.is_alive:
            return
        self.facing_right = hero_x > self.x
        self.phase_timer += dt
        if self.phase_timer >= cfg.ENEMY_C_ATTACK_CYCLE:
            self.phase_timer = 0.0
            self.attack_phase = (self.attack_phase + 1) % 3
        feedback = grid_feedback
        if local_snapshot is not None:
            feedback = feedback_from_snapshot(
                snapshot=local_snapshot,
                center_x=self.x,
                bottom_y=self.y,
                width=self.width,
                height=self.height,
                world_width=(
                    float(world_width)
                    if world_width is not None
                    else float(getattr(grid_feedback, "world_width", 0.0) or 0.0)
                ),
            )
        if feedback is None:
            return
        if self.vel_y < 0.0 and feedback.blocked_up:
            self.vel_y = 0.0
        if feedback.blocked_below:
            self.vel_y = 0.0
            self.on_ground = True
        else:
            self.on_ground = False
            self.vel_y += cfg.HERO_GRAVITY * dt
        if feedback.embedded and not feedback.blocked_up and (feedback.blocked_below or self.vel_y >= 0.0):
            self.y -= cfg.ENEMY_DESTUCK_LIFT
            if self.vel_y > 0.0:
                self.vel_y = 0.0
        self.y += self.vel_y * dt

    @property
    def should_spray_oil(self) -> bool:
        return self.attack_phase == 0 and self.phase_timer < 0.1

    @property
    def should_fire_fireball(self) -> bool:
        return self.attack_phase == 1 and self.phase_timer < 0.1

    @property
    def should_collapse(self) -> bool:
        return self.attack_phase == 2 and self.phase_timer < 0.1
