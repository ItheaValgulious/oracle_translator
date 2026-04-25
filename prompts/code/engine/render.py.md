# render.py

这个文件负责把当前网格状态转换成 demo 渲染用的 RGBA 帧缓冲。

## 对外接口

- `DebugViewMode`
  - 调试视图枚举,当前包含 `material`、`temperature` 与 `pressure`。
- `build_rgba_frame(grid, registry, view_mode=DebugViewMode.MATERIAL)`
  - 返回当前网格在指定调试视图下的 RGBA 字节流。

## 依赖的对外接口

- `engine.grid.Grid`
- `engine.types.CellFlag`
- `engine.types.MaterialRegistry`

## 主要功能

- 根据当前调试视图生成调试可读的颜色输出。
- 在 `material` 视图下,使用当前变体颜色、按 10 秒窗口归一化后的支撑新鲜度、完整度、温度和 `fixpoint` 标记着色。
- 在 `temperature` 视图下,把当前温度映射成冷到热的调色带。
- 在 `pressure` 视图下,把当前压力标量场映射成由低压到高压的调色带。
- 作为 CPU reference / fallback 路径的调试着色实现。
- 为 `pyglet + ModernGL` demo 在无 compute shader 时提供直接可上传的纹理数据。
