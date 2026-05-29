"""Entity runtime backed by GPU collision queries and GPU occupancy mask."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

from src.engine.gpu_backend import FeedbackQueryPoint
from src.engine.world import ActiveWorldWindow
from src.game.hero import Hero, GridFeedback

MAX_GPU_QUERY_POINTS = 64


@dataclass
class DebugCollisionInfo:
    """Cells sampled for collision queries, in world coordinates."""

    hero_cells: list[tuple[int, int]] = field(default_factory=list)
    hero_rects: list[tuple[int, int, int, int]] = field(default_factory=list)
    ground_cells: list[tuple[int, int]] = field(default_factory=list)
    ceiling_cells: list[tuple[int, int]] = field(default_factory=list)
    left_wall_cells: list[tuple[int, int]] = field(default_factory=list)
    right_wall_cells: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Entity:
    """A positioned entity with dimensions and optional custom query points."""

    entity_id: str
    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0
    query_points: list[tuple[int, int, int]] | None = None

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


def _entity_tag(entity_id: str) -> int:
    return max(1, hash(entity_id) & 0x7FFFFFFF)


def _entity_world_aabb(entity: Entity) -> tuple[int, int, int, int]:
    wx0 = int(entity.left)
    wy0 = int(entity.bottom)
    wx1 = int(entity.right) + 1
    wy1 = int(entity.top) + 1
    return (wx0, wy0, wx1, wy1)


def _clipped_local_rect(world: ActiveWorldWindow, entity: Entity) -> tuple[int, int, int, int] | None:
    ax = world.active_origin_x
    ay = world.active_origin_y
    aw = world.active_width
    ah = world.active_height
    wx0, wy0, wx1, wy1 = _entity_world_aabb(entity)
    lx0 = max(0, wx0 - ax)
    ly0 = max(0, wy0 - ay)
    lx1 = min(aw, wx1 - ax)
    ly1 = min(ah, wy1 - ay)
    if lx0 >= lx1 or ly0 >= ly1:
        return None
    return (lx0, ly0, lx1, ly1)


def _append_limited(points: list[FeedbackQueryPoint], candidates, budget: int) -> None:
    for point in candidates:
        if len(points) >= budget:
            return
        points.append(point)


def _sample_span(start: int, stop: int, max_count: int) -> list[int]:
    if stop <= start or max_count <= 0:
        return []
    count = stop - start
    if count <= max_count:
        return list(range(start, stop))
    if max_count == 1:
        return [(start + stop - 1) // 2]
    values: list[int] = []
    for index in range(max_count):
        offset = round(index * (count - 1) / (max_count - 1))
        values.append(start + offset)
    return values


@dataclass
class EntityManager:
    """Manages entity lifecycle, GPU occupancy mask, and async collision feedback."""

    hero: Hero = field(default_factory=lambda: Hero())
    _entities: dict[str, Entity] = field(default_factory=dict)
    _feedback_tokens: dict[str, object] = field(default_factory=dict)
    _last_feedback: dict[str, GridFeedback] = field(default_factory=dict)
    debug_collision: bool = False
    last_debug: DebugCollisionInfo | None = None

    def register_entity(self, entity: Entity) -> None:
        self._entities[entity.entity_id] = entity

    def unregister_entity(self, entity_id: str) -> None:
        self._entities.pop(entity_id, None)
        self._feedback_tokens.pop(entity_id, None)
        self._last_feedback.pop(entity_id, None)

    def _build_query_points(
        self,
        entity: Entity,
        world: ActiveWorldWindow,
        clip: tuple[int, int, int, int],
    ) -> list[FeedbackQueryPoint]:
        if entity.query_points:
            world_x0, world_y0, _, _ = _entity_world_aabb(entity)
            points: list[FeedbackQueryPoint] = []
            for dx, dy, kind in entity.query_points:
                if len(points) >= MAX_GPU_QUERY_POINTS:
                    break
                points.append((
                    world_x0 + int(dx) - world.active_origin_x,
                    world_y0 + int(dy) - world.active_origin_y,
                    int(kind),
                ))
            return points

        lx0, ly0, lx1, ly1 = clip
        points: list[FeedbackQueryPoint] = []
        width = lx1 - lx0
        height = ly1 - ly0
        horizontal_budget = min(width, 16)
        vertical_budget = min(height, 16)
        xs = _sample_span(lx0, lx1, horizontal_budget)
        ys = _sample_span(ly0, ly1, vertical_budget)
        _append_limited(points, ((x, ly1, 0) for x in xs), MAX_GPU_QUERY_POINTS)
        _append_limited(points, ((x, ly0 - 1, 3) for x in xs), MAX_GPU_QUERY_POINTS)
        _append_limited(points, ((lx0 - 1, y, 1) for y in ys), MAX_GPU_QUERY_POINTS)
        _append_limited(points, ((lx1, y, 2) for y in ys), MAX_GPU_QUERY_POINTS)
        if len(points) < MAX_GPU_QUERY_POINTS:
            interior_budget = MAX_GPU_QUERY_POINTS - len(points)
            interior_cols = max(1, min(width, 8))
            interior_rows = max(1, min(height, max(1, interior_budget // interior_cols)))
            for y in _sample_span(ly0, ly1, interior_rows):
                for x in _sample_span(lx0, lx1, interior_cols):
                    if len(points) >= MAX_GPU_QUERY_POINTS:
                        break
                    points.append((x, y, 4))
        return points

    def _update_debug_collision(
        self,
        entity: Entity,
        world: ActiveWorldWindow,
        clip: tuple[int, int, int, int] | None,
        points: list[FeedbackQueryPoint],
    ) -> None:
        if not self.debug_collision:
            return
        debug = DebugCollisionInfo()
        if clip is not None:
            lx0, ly0, lx1, ly1 = clip
            debug.hero_rects.append((
                lx0 + world.active_origin_x,
                ly0 + world.active_origin_y,
                lx1 + world.active_origin_x,
                ly1 + world.active_origin_y,
            ))
        for x, y, kind in points:
            wx = x + world.active_origin_x
            wy = y + world.active_origin_y
            if kind == 0:
                debug.ground_cells.append((wx, wy))
            elif kind == 1:
                debug.left_wall_cells.append((wx, wy))
            elif kind == 2:
                debug.right_wall_cells.append((wx, wy))
            elif kind == 3:
                debug.ceiling_cells.append((wx, wy))
        self.last_debug = debug

    def update_gpu_entity_mask(self, world: ActiveWorldWindow) -> None:
        if world.gpu_simulator is None:
            return
        rects: list[tuple[int, int, int, int, int]] = []
        for entity in self._entities.values():
            clip = _clipped_local_rect(world, entity)
            if clip is None:
                continue
            rects.append((*clip, _entity_tag(entity.entity_id)))
        world.gpu_simulator.update_entity_mask(rects)

    def tick(self, world: ActiveWorldWindow, dt: float) -> None:
        del dt
        self.update_gpu_entity_mask(world)

    def schedule_feedback(self, world: ActiveWorldWindow) -> None:
        if world.gpu_simulator is None:
            return
        if "hero" in self._feedback_tokens:
            return
        entity = self._entities.get("hero")
        if entity is None:
            return
        clip = _clipped_local_rect(world, entity)
        if clip is None:
            self._last_feedback["hero"] = GridFeedback(world_width=world.world_width)
            if self.debug_collision:
                self.last_debug = DebugCollisionInfo()
            return
        points = self._build_query_points(entity, world, clip)
        self._update_debug_collision(entity, world, clip, points)
        self._feedback_tokens["hero"] = world.gpu_simulator.request_entity_feedback(
            clip=clip,
            query_points=points,
            entity_tag=_entity_tag(entity.entity_id),
            world_width=world.world_width,
        )

    def poll_scheduled_feedback(self, world: ActiveWorldWindow) -> GridFeedback | None:
        token = self._feedback_tokens.get("hero")
        if token is None or world.gpu_simulator is None:
            return self._last_feedback.get("hero")
        feedback = world.gpu_simulator.poll_entity_feedback(token)
        if feedback is not None:
            self._last_feedback["hero"] = feedback
            self._feedback_tokens.pop("hero", None)
        return self._last_feedback.get("hero")

    def read_feedback_and_update(
        self,
        world: ActiveWorldWindow,
        dt: float,
        *,
        feedback: GridFeedback | None = None,
    ) -> None:
        hero_entity = self._entities.get("hero")
        resolved_feedback = feedback or self._last_feedback.get("hero") or GridFeedback(world_width=world.world_width)
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "[entity] gpu_feedback: blocked_below=%s blocked_up=%s blocked_left=%s blocked_right=%s in_liquid=%s damage=%.2f",
                resolved_feedback.blocked_below,
                resolved_feedback.blocked_up,
                resolved_feedback.blocked_left,
                resolved_feedback.blocked_right,
                resolved_feedback.in_liquid,
                resolved_feedback.damage,
            )
        self.hero.update(dt, grid_feedback=resolved_feedback)
        if hero_entity is not None:
            hero_entity.x = self.hero.x
            hero_entity.y = self.hero.y

    def post_step(self, world: ActiveWorldWindow, dt: float) -> None:
        feedback = self.poll_scheduled_feedback(world)
        self.schedule_feedback(world)
        self.read_feedback_and_update(world, dt, feedback=feedback)
