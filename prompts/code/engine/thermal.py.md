# thermal.py

这个文件实现热传导。

## 对外接口

- `apply_thermal(grid, registry, dt)`
  - 更新相邻格子的热交换和基础温度回归。

## 依赖的对外接口

- `engine.grid.Grid`
- `engine.types.MaterialRegistry`

## 主要功能

- 按变体的导热率和热容量做局部热交换。
- 让不同材质以不同速度升温和降温。
- 给相变和结构热损伤提供温度来源。
