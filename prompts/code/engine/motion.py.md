# motion.py

这个文件实现第一版 `8 邻域` 运动。

## 对外接口

- `apply_motion(grid, registry, dt)`
  - 基于 `vel + blocked_impulse + gravity` 更新移动结果。

## 依赖的对外接口

- `engine.grid.Grid`
- `engine.types.MaterialRegistry`
- `engine.types.SimKind`

## 主要功能

- 对 `powder/liquid/gas/fire/molten` 这几类可动物质做局部移动。
- 从主方向开始,按与 `drive` 的夹角从小到大搜索候选空位。
- 把未实现的那部分主方向需求累积到 `blocked_impulse`,而不是做反向回弹。
