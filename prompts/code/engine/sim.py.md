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
  - `reactions`
  - `thermal`
  - `phases`
  - `motion`
  - `collapse`
- 每次完整步进后推进 `grid.step_id`,供后续步的确定性随机运动使用。
- 提供手工注入接口给后续 `pyglet + ModernGL` 壳和测试场景使用; 注入 `fixpoint` 时会初始化 10 秒支撑新鲜度。
- 当注入 `empty` 且没有显式给出温度覆盖时,当前会按目标行的环境温度层结初始化该空气格。
- 当结构完全移除成空格时,留下的 `empty` 也会按所在高度回到本地环境温度,避免凭空制造整列统一 `20°C` 的假空气。
