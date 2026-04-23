# sim.py

这个文件编排第一版体素模拟的步进顺序。

## 对外接口

- `step(grid, registry, dt)`
  - 执行一次完整模拟步进。
- `inject_cells(grid, brush_or_cells, family_id, variant_id, overrides=None, registry=None)`
  - 向网格中注入测试物质或结构。

## 依赖的对外接口

- `engine.grid.Grid`
- `engine.materials.build_material_registry`
- `engine.motion.apply_motion`
- `engine.phases.apply_phase_transitions`
- `engine.reactions.apply_reactions`
- `engine.support.apply_support`
- `engine.thermal.apply_thermal`
- `engine.types.CellFlag`
- `engine.types.CellState`
- `engine.types.MaterialRegistry`

## 主要功能

- 统一按顺序执行:
  - `support`
  - `motion`
  - `thermal`
  - `phases`
  - `reactions`
  - `collapse`
- 提供手工注入接口给后续 `pyglet + ModernGL` 壳和测试场景使用。
