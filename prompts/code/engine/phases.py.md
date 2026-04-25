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
- 在熔融体冷却时,按 `support_value > 0` 的当前支撑新鲜度决定回 `platform` 还是同家族碎屑。
- 相变只改变 cell 的家族/变体身份,不会把当前温度重置或夹到目标变体的 `base_temperature`。
