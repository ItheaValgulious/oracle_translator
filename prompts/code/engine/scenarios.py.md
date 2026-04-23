# scenarios.py

这个文件提供第一版 demo 的预置场景。

## 对外接口

- `populate_demo_scene(grid, registry=None)`
  - 在空网格里放置桥、支撑点、液体池、可燃物和测试物质。

## 依赖的对外接口

- `engine.grid.Grid`
- `engine.sim.inject_cells`
- `engine.types.CellFlag`

## 主要功能

- 快速构造一个能同时展示:
  - 支撑传播
  - 失撑粉化
  - 液体流动
  - 热与相变
  - 腐蚀与点燃
  的最小可玩场景。
