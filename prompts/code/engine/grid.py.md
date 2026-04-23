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
- 给支撑、运动、热、相变、反应各个 pass 提供统一的网格读写接口。
