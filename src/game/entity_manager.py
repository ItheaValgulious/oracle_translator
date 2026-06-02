"""Entity runtime backed by GPU local snapshots and GPU occupancy mask."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from src.engine.gpu_backend import FeedbackQueryPoint, GpuFeedbackBatchToken, _same_feedback_token
from src.engine.snapshot_mailbox import SnapshotMailboxRegistry
from src.engine.world import ActiveWorldWindow
from src.game.enemy import EnemyBase
from src.game.hero import Hero, GridFeedback, LocalCellSnapshot, feedback_from_snapshot

log = logging.getLogger(__name__)

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
    return (int(entity.left), int(entity.bottom), int(entity.right) + 1, int(entity.top) + 1)


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


def _is_viewport_visible(entity: Entity, world: ActiveWorldWindow) -> bool:
    ax = world.active_origin_x
    ay = world.active_origin_y
    aw = world.active_width
    ah = world.active_height
    return (
        entity.right >= ax
        and entity.left <= ax + aw
        and entity.top >= ay
        and entity.bottom <= ay + ah
    )


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


def _coerce_local_snapshot(snapshot: object, *, entity_id: str | None = None, age_frames: int = 0) -> LocalCellSnapshot | None:
    if isinstance(snapshot, LocalCellSnapshot):
        snapshot.age_frames = int(age_frames)
        return snapshot
    origin_x = getattr(snapshot, "origin_x", getattr(snapshot, "world_x", None))
    origin_y = getattr(snapshot, "origin_y", getattr(snapshot, "world_y", None))
    width = getattr(snapshot, "width", None)
    height = getattr(snapshot, "height", None)
    variant_indices = getattr(snapshot, "variant_indices", None)
    velocities = getattr(snapshot, "velocities", None)
    temperatures = getattr(snapshot, "temperatures", None)
    variant_families = getattr(snapshot, "variant_families", None)
    empty_variant_index = getattr(snapshot, "empty_variant_index", None)
    tick_id = getattr(snapshot, "tick_id", getattr(snapshot, "issued_step", 0))
    if any(value is None for value in (
        origin_x,
        origin_y,
        width,
        height,
        variant_indices,
        velocities,
        temperatures,
        variant_families,
        empty_variant_index,
    )):
        return None
    return LocalCellSnapshot(
        entity_id=str(getattr(snapshot, "entity_id", entity_id or "")),
        origin_x=int(origin_x),
        origin_y=int(origin_y),
        width=int(width),
        height=int(height),
        variant_indices=variant_indices,
        velocities=velocities,
        temperatures=temperatures,
        variant_families=variant_families,
        empty_variant_index=int(empty_variant_index),
        tick_id=int(tick_id or 0),
        age_frames=int(age_frames),
    )


@dataclass
class EntityManager:
    """Manages entity lifecycle, GPU occupancy mask, and local snapshots."""

    hero: Hero = field(default_factory=lambda: Hero())
    _entities: dict[str, Entity] = field(default_factory=dict)
    _enemies: dict[str, EnemyBase] = field(default_factory=dict)
    _last_feedback: dict[str, GridFeedback] = field(default_factory=dict)
    _latest_snapshots: dict[str, LocalCellSnapshot] = field(default_factory=dict)
    _pending_feedback_tokens: deque[GpuFeedbackBatchToken] = field(default_factory=deque)
    debug_collision: bool = False
    last_debug: DebugCollisionInfo | None = None
    _snapshot_registry: SnapshotMailboxRegistry = field(default_factory=SnapshotMailboxRegistry)

    def register_entity(self, entity: Entity, world: ActiveWorldWindow | None = None) -> None:
        self._entities[entity.entity_id] = entity
        if world is not None and world.gpu_simulator is not None:
            world.gpu_simulator.register_entity_shape(entity.entity_id, int(entity.width), int(entity.height))

    def unregister_entity(self, entity_id: str) -> None:
        self._entities.pop(entity_id, None)
        self._last_feedback.pop(entity_id, None)
        self._latest_snapshots.pop(entity_id, None)
        self._snapshot_registry.drop(entity_id)

    def register_enemy(self, enemy: EnemyBase) -> None:
        self._enemies[enemy.entity_id] = enemy
        self.register_entity(Entity(
            entity_id=enemy.entity_id,
            x=enemy.x,
            y=enemy.y,
            width=enemy.width,
            height=enemy.height,
        ))

    def unregister_enemy(self, entity_id: str) -> None:
        self._enemies.pop(entity_id, None)
        self.unregister_entity(entity_id)

    def register_entity_shapes(self, world: ActiveWorldWindow) -> None:
        if world.gpu_simulator is None:
            return
        for entity_id, width, height in self.entity_shape_payloads():
            world.gpu_simulator.register_entity_shape(entity_id, width, height)

    def entity_shape_payloads(self) -> list[tuple[str, int, int]]:
        return [
            (entity.entity_id, int(entity.width), int(entity.height))
            for entity in self._entities.values()
        ]

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
        horizontal_budget = min(width, 12)
        vertical_budget = min(height, 8)
        xs = _sample_span(lx0, lx1, horizontal_budget)
        ys = _sample_span(ly0, ly1, vertical_budget)
        _append_limited(points, ((x, ly1, 0) for x in xs), MAX_GPU_QUERY_POINTS)
        _append_limited(points, ((x, ly0 - 1, 3) for x in xs), MAX_GPU_QUERY_POINTS)
        _append_limited(points, ((lx0 - 1, y, 1) for y in ys), MAX_GPU_QUERY_POINTS)
        _append_limited(points, ((lx1, y, 2) for y in ys), MAX_GPU_QUERY_POINTS)
        foot_y = ly1 - 1
        clearance_y = ly1 - 2
        _append_limited(points, ((lx0 - 1, foot_y, 6),), MAX_GPU_QUERY_POINTS)
        _append_limited(points, ((lx1, foot_y, 7),), MAX_GPU_QUERY_POINTS)
        _append_limited(points, ((lx0 - 1, clearance_y, 9),), MAX_GPU_QUERY_POINTS)
        _append_limited(points, ((lx1, clearance_y, 10),), MAX_GPU_QUERY_POINTS)
        body_xs = _sample_span(lx0, lx1, min(width, 3))
        body_ys = _sample_span(ly0, ly1, min(height, 3))
        _append_limited(
            points,
            ((x, y, 4) for y in body_ys for x in body_xs),
            MAX_GPU_QUERY_POINTS,
        )
        embedded_xs = _sample_span(lx0, lx1, min(width, 6))
        embedded_rows = [row for row in (foot_y, clearance_y) if ly0 <= row < ly1]
        _append_limited(
            points,
            ((x, row, 8) for row in embedded_rows for x in embedded_xs),
            MAX_GPU_QUERY_POINTS,
        )
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
        rects = self.entity_mask_payload(world)
        world.gpu_simulator.update_entity_mask(rects)

    def entity_mask_payload(self, world: ActiveWorldWindow) -> list[tuple[int, int, int, int, int]]:
        rects: list[tuple[int, int, int, int, int]] = []
        for entity in self._entities.values():
            clip = _clipped_local_rect(world, entity)
            if clip is None:
                continue
            rects.append((*clip, _entity_tag(entity.entity_id)))
        return rects

    def _upload_entity_states(self, world: ActiveWorldWindow) -> None:
        if world.gpu_simulator is None:
            return
        states = self.entity_state_payload(world)
        if states:
            world.gpu_simulator.upload_entity_states(
                states,
                origin_x=world.active_origin_x,
                origin_y=world.active_origin_y,
            )

    def entity_state_payload(self, world: ActiveWorldWindow) -> list[tuple[str, int, int, bool, float]] | None:
        visible_ids = {entity.entity_id for entity in self.visible_entities(world)}
        states: list[tuple[str, int, int, bool, float]] = []
        hero_entity = self._entities.get("hero")
        if hero_entity is not None:
            states.append(("hero", int(self.hero.x), int(self.hero.y), self.hero.facing_right, 1.0))
        for eid, enemy in self._enemies.items():
            if not enemy.is_alive:
                continue
            if eid not in visible_ids:
                continue
            states.append((eid, int(enemy.x), int(enemy.y), enemy.facing_right, 1.0))
        # Cache check: skip upload if states haven't changed
        cached = getattr(self, "_cached_entity_states", None)
        if cached == states:
            return None
        self._cached_entity_states = list(states)
        return states

    def tick(self, world: ActiveWorldWindow, dt: float) -> None:
        del dt
        self._upload_entity_states(world)
        self.update_gpu_entity_mask(world)

    def schedule_feedback(self, world: ActiveWorldWindow) -> None:
        if world.gpu_simulator is None:
            return
        active_entities = self._active_feedback_entities(world)
        entities_info: list[tuple[str, tuple[int, int, int, int], list[FeedbackQueryPoint], int]] = []
        for entity in active_entities:
            clip = _clipped_local_rect(world, entity)
            if clip is None:
                continue
            query_points = self._build_query_points(entity, world, clip)
            entities_info.append((entity.entity_id, clip, query_points, _entity_tag(entity.entity_id)))
            if entity.entity_id == "hero":
                self._update_debug_collision(entity, world, clip, query_points)
        if not entities_info:
            return
        self._pending_feedback_tokens.append(world.gpu_simulator.request_batched_entity_feedback(
            entities_info,
            world_width=world.world_width,
        ))

    def _active_feedback_entities(self, world: ActiveWorldWindow) -> list[Entity]:
        active_entities = self.visible_entities(world)
        hero_entity = self._entities.get("hero")
        if hero_entity is not None and hero_entity not in active_entities:
            active_entities.append(hero_entity)
        return active_entities

    def poll_all_feedback(self, world: ActiveWorldWindow) -> dict[str, GridFeedback]:
        active_entities = self._active_feedback_entities(world)
        resolved = dict(self._last_feedback)
        if world.gpu_simulator is not None and self._pending_feedback_tokens:
            ready_feedback: dict[str, GridFeedback] | None = None
            while len(self._pending_feedback_tokens) > 1:
                newest = self._pending_feedback_tokens[-1]
                polled = world.gpu_simulator.poll_batched_entity_feedback(newest)
                if polled is None:
                    break
                ready_feedback = polled
                while self._pending_feedback_tokens:
                    stale = self._pending_feedback_tokens.popleft()
                    if not _same_feedback_token(stale, newest):
                        world.gpu_simulator.release_batched_entity_feedback(stale)
                    else:
                        break
            if ready_feedback is None and self._pending_feedback_tokens:
                token = self._pending_feedback_tokens[0]
                polled = world.gpu_simulator.poll_batched_entity_feedback(token)
                if polled is not None:
                    self._pending_feedback_tokens.popleft()
                    ready_feedback = polled
            if ready_feedback is not None:
                resolved.update(ready_feedback)
        for entity in active_entities:
            resolved.setdefault(entity.entity_id, GridFeedback(world_width=world.world_width))
        self._last_feedback = resolved
        return self._last_feedback

    def read_feedback_and_update(
        self,
        world: ActiveWorldWindow,
        dt: float,
        *,
        feedback: dict[str, GridFeedback] | None = None,
    ) -> None:
        del feedback
        hero_entity = self._entities.get("hero")
        gpu = getattr(world, "gpu_simulator", None)
        current_tick = int(
            getattr(world, "gpu_step_index", getattr(gpu, "step_index", 0)) or 0
        )
        hero_snapshot = self.latest_snapshot_for("hero", current_tick=current_tick)
        self.hero.update(
            dt,
            local_snapshot=hero_snapshot,
            world_width=world.world_width,
        )
        if hero_entity is not None:
            hero_entity.x = self.hero.x
            hero_entity.y = self.hero.y

    def post_step(self, world: ActiveWorldWindow, dt: float) -> None:
        feedback = self.poll_all_feedback(world)
        self.read_feedback_and_update(world, dt, feedback=feedback)

    def set_latest_snapshot(self, entity_id: str, snapshot: LocalCellSnapshot) -> None:
        self._latest_snapshots[entity_id] = snapshot

    def latest_snapshot_for(self, entity_id: str, *, current_tick: int = 0) -> LocalCellSnapshot | None:
        cached = self._latest_snapshots.get(entity_id)
        envelope = self._snapshot_registry.consume(entity_id, current_tick=current_tick)
        if envelope is not None:
            packed = _coerce_local_snapshot(
                envelope.packed,
                entity_id=entity_id,
                age_frames=int(envelope.age_frames),
            )
            if packed is not None:
                if cached is None or int(packed.tick_id) >= int(cached.tick_id):
                    self._latest_snapshots[entity_id] = packed
                    cached = packed
                if cached is packed:
                    return packed
        snapshot = cached
        if snapshot is None:
            return None
        snapshot.age_frames = max(0, int(current_tick) - int(snapshot.tick_id))
        return snapshot

    def visible_entities(self, world: ActiveWorldWindow) -> list[Entity]:
        result: list[Entity] = []
        for entity in self._entities.values():
            if _is_viewport_visible(entity, world):
                result.append(entity)
        return result
