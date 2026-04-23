# phases.py

这个文件实现家族内和少量跨家族的相变。

## 对外接口

- `apply_phase_transitions(grid, registry, dt)`
  - 按 `phase_map` 切换变体或目标家族。

## 依赖的对外接口

- `engine.grid.Grid`
- `engine.support.SUPPORT_FAILURE_THRESHOLD`
- `engine.types.CellFlag`
- `engine.types.CellState`
- `engine.types.MaterialFamily`
- `engine.types.MaterialRegistry`

## 主要功能

- 处理:
  - `stone <-> magma`
  - `sand -> molten_glass`
  - `glass <-> molten_glass`
  - `iron <-> molten_iron`
  - `water <-> ice <-> steam`
  - `acid_liquid -> acid_gas`
  - `poison_liquid -> poison_gas`
- 在熔融体冷却时,按当前支撑状态决定回 `platform` 还是同家族碎屑。
