"""Entity-Grid Hybrid: placeholder write/clear + collision + entity updates.

All grid operations use active_grid (local coordinates) to stay in sync
with the simulation pipeline. World→local conversion mirrors
ActiveWorldWindow.paint_world().

Placeholder cells use a single "entity_placeholder" family. The owning
entity's ID is stored in CellState.generation (unused for placeholders).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.engine.types import CellState, empty_cell
from src.engine.world import ActiveWorldWindow
from src.game import config as cfg
from src.game.hero import Hero, GridFeedback


PLACEHOLDER_FAMILY = "entity_placeholder"
PLACEHOLDER_VARIANT = "placeholder"


def _clipped_local_rect(world: ActiveWorldWindow, entity: Entity) -> tuple[int, int, int, int] | None:
    """Return (lx0, ly0, lx1, ly1) in local coords, clipped to active grid bounds.
    Returns None if entity is completely outside the active window."""
    ax = world.active_origin_x
    ay = world.active_origin_y
    aw = world.active_width
    ah = world.active_height
    # Entity world bounds
    wx0 = int(entity.left)
    wy0 = int(entity.bottom)
    wx1 = int(entity.right) + 1
    wy1 = int(entity.top) + 1
    # Clip to active rect
    lx0 = max(0, wx0 - ax)
    ly0 = max(0, wy0 - ay)
    lx1 = min(aw, wx1 - ax)
    ly1 = min(ah, wy1 - ay)
    if lx0 >= lx1 or ly0 >= ly1:
        return None
    return (lx0, ly0, lx1, ly1)


def _is_solid_variant(registry, cell: CellState) -> bool:
    if cell.is_empty or cell.family_id == PLACEHOLDER_FAMILY:
        return False
    try:
        variant = registry.variant(cell.family_id, cell.variant_id)
        return variant.matter_state.value == "solid"
    except KeyError:
        return False


def _is_liquid_variant(registry, cell: CellState) -> bool:
    if cell.is_empty or cell.family_id == PLACEHOLDER_FAMILY:
        return False
    try:
        variant = registry.variant(cell.family_id, cell.variant_id)
        return variant.matter_state.value == "liquid"
    except KeyError:
        return False


# _clipped_local_rect is defined above, replacing _world_to_local/_in_active_rect


@dataclass
class Entity:
    """A positioned entity with dimensions."""

    entity_id: str
    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0

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


@dataclass
class EntityManager:
    """Manages entity lifecycle, placeholder cells, and grid feedback.

    All operations target active_grid (local coords) to stay synchronized
    with the simulation step which also operates on active_grid.

    Multiple entities are supported via register_entity(). Each entity gets
    placeholder cells tagged with its entity_id (stored in CellState.generation).
    """

    hero: Hero = field(default_factory=lambda: Hero())
    _entities: dict[str, Entity] = field(default_factory=dict)
    _saved_cells: dict[tuple[int, int], CellState] = field(default_factory=dict)

    def register_entity(self, entity: Entity) -> None:
        self._entities[entity.entity_id] = entity

    def unregister_entity(self, entity_id: str) -> None:
        self._entities.pop(entity_id, None)

    def _entity_local_cells(self, entity: Entity, world: ActiveWorldWindow) -> list[tuple[int, int]]:
        """Return local grid cells occupied by an entity."""
        clip = _clipped_local_rect(world, entity)
        if clip is None:
            return []
        lx0, ly0, lx1, ly1 = clip
        return [(lx, ly) for ly in range(ly0, ly1) for lx in range(lx0, lx1)]

    def clear_placeholder(self, world: ActiveWorldWindow) -> None:
        """Remove all placeholder cells, restoring saved originals."""
        grid = world.active_grid
        for (lx, ly), original in self._saved_cells.items():
            grid.set_cell(lx, ly, original)
        self._saved_cells.clear()

    def write_placeholder(self, world: ActiveWorldWindow) -> None:
        """Save original cells, then write placeholder at each entity's position."""
        grid = world.active_grid
        for entity in self._entities.values():
            clip = _clipped_local_rect(world, entity)
            if clip is None:
                continue
            lx0, ly0, lx1, ly1 = clip
            gen = hash(entity.entity_id) & 0x7FFFFFFF
            saved = self._saved_cells
            for ly in range(ly0, ly1):
                for lx in range(lx0, lx1):
                    key = (lx, ly)
                    if key not in saved:
                        saved[key] = grid.get_cell(lx, ly)
                    grid.set_cell(lx, ly, CellState(
                        family_id=PLACEHOLDER_FAMILY,
                        variant_id=PLACEHOLDER_VARIANT,
                        generation=gen,
                    ))

    def read_entity_feedback(self, entity: Entity, world: ActiveWorldWindow) -> GridFeedback:
        """Read grid state around an entity and return feedback.

        Damage is read from placeholder integrity loss caused by
        the reaction system (damage_mask "living" -> entity_placeholder).
        """
        feedback = GridFeedback(world_width=world.world_width)
        registry = world.registry
        grid = world.active_grid
        ax = world.active_origin_x
        ay = world.active_origin_y
        aw = world.active_width
        ah = world.active_height

        # Entity bounds in world coords
        ent_wx0 = int(entity.left)
        ent_wx1 = int(entity.right) + 1
        ent_wy0 = int(entity.bottom)
        ent_wy1 = int(entity.top) + 1

        # Check cell just below entity's bottom for solid ground
        ground_wy = ent_wy0 - 1
        ground_ly = ground_wy - ay
        if 0 <= ground_ly < ah:
            for wx in range(ent_wx0, ent_wx1):
                lx = wx - ax
                if 0 <= lx < aw:
                    cell = grid.get_cell(lx, ground_ly)
                    if _is_solid_variant(registry, cell):
                        feedback.blocked_below = True
                        break

        # Check walls: middle body only, skip top 2 cells (ground surface overlap)
        wall_check_bottom = ent_wy0 + 2
        # Left wall
        wall_wx = ent_wx0 - 1
        wall_lx = wall_wx - ax
        if 0 <= wall_lx < aw:
            for wy in range(wall_check_bottom, ent_wy1):
                ly = wy - ay
                if 0 <= ly < ah:
                    cell = grid.get_cell(wall_lx, ly)
                    if _is_solid_variant(registry, cell):
                        feedback.blocked_left = True
                        break
        # Right wall
        wall_wx = ent_wx1
        wall_lx = wall_wx - ax
        if 0 <= wall_lx < aw:
            for wy in range(wall_check_bottom, ent_wy1):
                ly = wy - ay
                if 0 <= ly < ah:
                    cell = grid.get_cell(wall_lx, ly)
                    if _is_solid_variant(registry, cell):
                        feedback.blocked_right = True
                        break

        # Check if entity is in liquid (saved originals under placeholders)
        for original in self._saved_cells.values():
            if _is_liquid_variant(registry, original):
                feedback.in_liquid = True
                break

        # Damage from reaction system: read placeholder integrity loss.
        entity_tag = hash(entity.entity_id) & 0x7FFFFFFF
        clip = _clipped_local_rect(world, entity)
        if clip is not None:
            lx0, ly0, lx1, ly1 = clip
            placeholder_integrity_total = 0.0
            placeholder_count = 0
            for ly in range(ly0, ly1):
                for lx in range(lx0, lx1):
                    cell = grid.get_cell(lx, ly)
                    if cell.family_id == PLACEHOLDER_FAMILY and cell.generation == entity_tag:
                        placeholder_integrity_total += cell.integrity
                        placeholder_count += 1
            if placeholder_count > 0:
                integrity_loss = max(0.0, 1.0 - (placeholder_integrity_total / placeholder_count))
                if integrity_loss > 0.01:
                    feedback.damage = integrity_loss * 100.0

        return feedback

    def tick(self, world: ActiveWorldWindow, dt: float) -> None:
        """Entity-Grid Hybrid 6-step tick per plan:

        1. Clear old placeholders (restore saved originals)
        2. Write new placeholders at current entity positions
        3. (External) world.step(dt) -- simulation runs with placeholders in grid
        4. Read grid feedback (blocked, liquid, damage via reaction system)
        5. Update entity positions based on input + feedback
        6. Clear placeholders (restore originals), prepare next tick
        """
        # Step 1: clear old placeholders (restore originals)
        self.clear_placeholder(world)

        # Step 2: write placeholders at current positions
        self.write_placeholder(world)

    def post_step(self, world: ActiveWorldWindow, dt: float) -> None:
        """Steps 4-5: read feedback after simulation, update hero.

        Note: placeholders are NOT cleared here. The next tick() clears them
        before writing new ones. This is critical because clearing here would
        restore destroyed terrain (e.g. burned ground) before the hero's
        feedback read, preventing the hero from falling through holes.
        """
        hero_entity = self._entities.get("hero")
        if hero_entity is not None:
            feedback = self.read_entity_feedback(hero_entity, world)
        else:
            feedback = GridFeedback(world_width=world.world_width)

        # Step 5: update hero position with input + feedback
        self.hero.update(dt, grid_feedback=feedback)

        # Sync hero entity position
        if hero_entity is not None:
            hero_entity.x = self.hero.x
            hero_entity.y = self.hero.y