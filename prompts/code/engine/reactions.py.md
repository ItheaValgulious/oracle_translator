# reactions.py

这个文件实现第一版基础反应系统。

## 对外接口

- `apply_reactions(grid, registry, dt)`
  - 处理 `fire/acid/poison/tar` 的反应与热损伤。

## 依赖的对外接口

- `engine.grid.Grid`
- `engine.types.CellState`
- `engine.types.MaterialRegistry`
- `engine.types.ReactionKind`

## 主要功能

- `fire`:
  - 按变体的 `reaction_energy` 直接改变自身温度
  - 不再在反应阶段直接给邻格注入温度
  - 消费 `age`
  - 熄灭时留下当前温度的热空气,不会把位置重置成默认空气温度
- `acid`:
  - 腐蚀可承重结构的 `integrity`
  - 成功腐蚀后可按变体配置自耗,自耗时保留当前位置温度
- `poison`:
  - 当前主要表现为受热挥发成毒气,并能生成短火团
- `tar`:
  - 高温或邻近火焰时点燃
- 结构材质:
  - 不持续燃烧
  - 只通过热量下降 `integrity`
- 热量向邻格的扩散不在这里做,统一交给 `thermal` 按邻格温差处理
