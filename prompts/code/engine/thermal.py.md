# thermal.py

这个文件实现热传导。

## 对外接口

- `apply_thermal(grid, registry, dt)`
  - 更新相邻格子的热交换。

## 依赖的对外接口

- `engine.grid.Grid`
- `engine.types.MaterialRegistry`

## 主要功能

- 按变体的导热率和热容量做局部热交换,包括 `platform/fixpoint` 这类静态固体。
- 允许热量通过 `empty` 代表的空气和固体一起传播。
- `base_temperature` 只作为创建 cell 时的初始温度来源,热传导不会把已有 cell 拉回材质基础温度。
- 让不同材质以不同速度升温和降温。
- 给相变和结构热损伤提供温度来源。
