"""Projectile entities: Arrow and Fireball."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.game import config as cfg

log = logging.getLogger(__name__)


@dataclass
class ProjectileBase:
    """Base class for all projectiles."""

    x: float = 0.0
    y: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    damage: float = 0.0
    owner: str = ""
    is_alive: bool = True
    age: float = 0.0
    max_age: float = 10.0
    projectile_type: int = 0  # 0=arrow, 1=fireball

    @property
    def grid_x(self) -> int:
        return int(self.x)

    @property
    def grid_y(self) -> int:
        return int(self.y)

    def update(self, dt: float) -> None:
        if not self.is_alive:
            return
        self.age += dt
        if self.age >= self.max_age:
            self.is_alive = False
            return
        self.x += self.vel_x * dt
        self.y += self.vel_y * dt


@dataclass
class Arrow(ProjectileBase):
    """Arrow: fired by EnemyA, parabolic arc toward hero.

    Uses ballistic trajectory solve: given start, target, gravity, and flight
    time, compute initial velocity so the arrow arcs and lands on the hero.
    """

    @staticmethod
    def create(start_x: float, start_y: float, target_x: float, target_y: float, facing_right: bool = True) -> Arrow:
        dx = target_x - start_x
        dy = target_y - start_y
        T = cfg.ARROW_FLIGHT_TIME
        g = cfg.ARROW_GRAVITY

        # Ballistic solve: v0x = dx/T, v0y = (dy - 0.5*g*T²)/T
        vx = dx / T
        vy = (dy - 0.5 * g * T * T) / T

        return Arrow(
            x=start_x,
            y=start_y + 3.0,  # shoot from upper body
            vel_x=vx,
            vel_y=vy,
            damage=cfg.ARROW_DAMAGE,
            owner="enemy_a",
            projectile_type=0,
            max_age=T + 1.0,  # slightly longer than flight time
        )

    def update(self, dt: float) -> None:
        if not self.is_alive:
            return
        self.age += dt
        if self.age >= self.max_age:
            self.is_alive = False
            return
        self.vel_y += cfg.ARROW_GRAVITY * dt
        self.x += self.vel_x * dt
        self.y += self.vel_y * dt


@dataclass
class Fireball(ProjectileBase):
    """Fireball: fired by EnemyC (boss), no gravity, explodes on terrain or hero contact."""

    @staticmethod
    def create(start_x: float, start_y: float, target_x: float, target_y: float) -> Fireball:
        dx = target_x - start_x
        dy = target_y - start_y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1.0:
            dist = 1.0
        vx = cfg.BOSS_FIREBALL_SPEED * dx / dist
        vy = cfg.BOSS_FIREBALL_SPEED * dy / dist
        return Fireball(
            x=start_x,
            y=start_y + 6.0,  # shoot from upper body (boss is tall)
            vel_x=vx,
            vel_y=vy,
            damage=cfg.BOSS_FIREBALL_DAMAGE,
            owner="enemy_c",
            projectile_type=1,
            max_age=cfg.BOSS_FIREBALL_MAX_AGE,
        )
