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
    - `rect_has_stored_cells(rect)`
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
- `WorldChunkStore.read_rect()` 和 `write_rect()` 当前按 chunk 行段处理矩形,避免整块条带逐格走 `_cell_ref/get_cell/set_cell` 的 Python 开销。
- 视野外 chunk 默认冻结,只保存 `cell state`,不持续推进 `pressure / source_force / force_wave`。
- 维护一个带 halo 的连续活动模拟窗,并把相机 viewport 映射到这个活动窗上。
- 当相机接近安全边界时,按固定分页单位平移活动窗。
- 相机当前会优先采用更小步长、更高频率的分页平移,降低单次切页卡顿。
- 平移活动窗时:
  - CPU 路径会直接把被逐出的区域写回 `WorldChunkStore`
  - GPU 路径会先把被逐出的区域放进 staging 队列
  - 保留重叠区域的当前 `cell state`
  - 从 world store 增量加载新暴露条带
  - 如果 incoming strip 对应空世界默认区,会直接在 GPU 上按 world row 环境温度填成默认 `empty`
  - `rect_has_stored_cells(rect)` 当前只会在矩形内真实命中存储 cell 时才走非空路径,不会因为同 chunk 其他位置有内容误判整条 strip
  - 对确实命中存储 cell 的 incoming strip,GPU 路径当前会直接把 store 的稀疏 chunk 内容打包成 GPU state bytes 写入目标条带
- GPU 路径当前只有在相机空闲后才会逐步 flush staging 队列,移动过程中不会主动写回,即使 pending 队列暂时超过上限也只记压力不做同步 flush。
- pending staging writeback 当前会按固定 cell budget 分片 readback + `write_rect`,避免停下后一次性 flush 整条 strip。
- GPU staging 区 flush 时,会把 region 内 support-transmission cell 的当前 `support_value / FIXPOINT` 状态转成冻结区 anchored support snapshot 一并写回,不再在 flush 当帧触发全世界 `recompute_anchored_support()`。
- 活动窗边界当前会根据 world store 里冻结区的承重连通信息生成 `external_support_anchors`,让 support 可以穿过冻结区边界持续补给。
- external support anchor 当前只重建活动窗四条边,并压成 compact edge buffer 交给 GPU support shader 读取,不再切页上传整张 anchor 纹理。
- external support anchor 的边条构建当前按“活动窗外侧一行/一列的 anchored source 投影”来算,不再对每个边界 cell 做 8 邻域 Python 查询。
- 额外维护冻结承重网络的外部锚定信息,让 support 穿越冻结区时不断供。
- 额外维护分页次数、最近/最大切页耗时、pending writeback 压力和分页分阶段耗时等调试统计,便于在 demo overlay 里观察流送性能。
