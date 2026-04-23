# render.py

这个文件负责把当前网格状态转换成 demo 渲染用的 RGBA 帧缓冲。

## 对外接口

- `build_rgba_frame(grid, registry)`
  - 返回当前网格的 RGBA 字节流。

## 依赖的对外接口

- `engine.grid.Grid`
- `engine.types.CellFlag`
- `engine.types.MaterialRegistry`
- `engine.types.SimKind`

## 主要功能

- 根据当前变体颜色、支撑值、完整度、温度和 `fixpoint` 标记生成调试可读的颜色输出。
- 为 `pyglet + ModernGL` demo 提供直接可上传的纹理数据。
