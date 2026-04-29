# world.py

这个文件提供第一版大世界流式加载层。

## 对外接口

- `GridSlice`
  - 表示一块按矩形读取或写回的世界 `cell state` 数据。
- `WorldRect`
  - 表示 world space 下的矩形区域。
- `WorldChunkStore`
  - 提供:
    - `read_rect(world_x, world_y, width, height)`
    - `write_rect(world_x, world_y, slice)`
    - `recompute_anchored_support(registry)`
    - `anchored_support_at(world_x, world_y)`
- `ActiveWorldWindow`
  - 提供:
    - `step(dt)`
    - `pan_camera(dx, dy)`
    - `screen_to_world(...)`
    - `paint_world(...)`
    - `set_liquid_brownian_enabled(enabled)`
    - `set_blocked_impulse_enabled(enabled)`
    - `set_directional_fallback_enabled(enabled)`
    - `set_directional_fallback_angle_limit_degrees(angle_limit_degrees)`
    - `ensure_resident_for_camera()`
    - `render(...)`
    - `readback_active_grid()`

## 依赖的对外接口

- `engine.grid.Grid`
- `engine.gpu_backend.GpuSimulator`
- `engine.render.DebugViewMode`
- `engine.sim.step`
- `engine.support.SUPPORT_SOURCE_VALUE`
- `engine.types.CellState`
- `engine.types.MaterialRegistry`

## 主要功能

- 把有限大地图按 `320x320` chunk 稀疏存储。
- 视野外 chunk 默认冻结,只保存 `cell state`,不持续推进 `pressure / source_force / force_wave`。
- 维护一个带 halo 的连续活动模拟窗,并把相机 viewport 映射到这个活动窗上。
- 当相机接近安全边界时,按固定分页单位平移活动窗。
- 平移活动窗时:
  - CPU 路径会直接把被逐出的区域写回 `WorldChunkStore`
  - GPU 路径会先把被逐出的区域放进 staging 队列
  - 保留重叠区域的当前 `cell state`
  - 从 world store 增量加载新暴露条带
- 活动窗边界当前会根据 world store 里冻结区的承重连通信息生成 `external_support_anchors`,让 support 可以穿过冻结区边界持续补给。
- 额外维护冻结承重网络的外部锚定信息,让 support 穿越冻结区时不断供。
- GPU 路径当前会在后续空闲 tick 里慢速 flush staging 队列,避免用户触发换页时立即做大块 GPU 读回。
