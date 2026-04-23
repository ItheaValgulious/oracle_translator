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
  - 提供热源
  - 消费 `age`
- `acid`:
  - 腐蚀可承重结构的 `integrity`
- `poison`:
  - 受热分解为毒雾,并能生成短火团
- `tar`:
  - 高温或邻近火焰时点燃
- 结构材质:
  - 不持续燃烧
  - 只通过热量下降 `integrity`
