# grid.py

这个文件提供体素网格容器和双缓冲工具。

## 对外接口

- `Grid`
  - 提供:
    - `index(x, y)`
    - `in_bounds(x, y)`
    - `get_cell(x, y, use_scratch=False)`
    - `set_cell(x, y, cell, use_scratch=False)`
    - `copy_cells_to_scratch()`
    - `clear_scratch()`
    - `swap_buffers()`
- `create_grid(width, height)`
  - 创建一个空网格。

## 依赖的对外接口

- `engine.types.CellState`
- `engine.types.empty_cell`

## 主要功能

- 维护当前缓冲区与 scratch 缓冲区。
- 维护一个轻量 `step_id`,供运动层做确定性随机扰动。
- 维护 `liquid_brownian_enabled` 这样的运行时模拟开关,供 CPU 路径直接读取。
- 维护 `blocked_impulse_enabled` 这样的运行时模拟开关,供 CPU 路径决定是否继续使用残余移动意图。
- 维护一个持久化 `pressure` 标量场,供 CPU 路径做逐步回归和液柱累积。
- 维护 `source_force`、`prev_source_force` 和 `force_wave` 这样的持久化力场,供跨帧传播液体内部受力。
- 给支撑、运动、热、相变、反应各个 pass 提供统一的网格读写接口。
