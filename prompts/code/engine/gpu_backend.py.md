# gpu_backend.py

这个文件提供第一版 compute shader 体素模拟 backend。

## 对外接口

- `ComputeBackendUnavailable`
  - 当当前 OpenGL context 不支持 compute shader 时抛出。
- `GpuMaterialTables`
  - 负责把 `MaterialRegistry` 编译成 GPU 侧使用的 family / variant / phase 表。
- `pack_grid_state(grid, tables)`
  - 把 CPU `Grid` 打包成 GPU 纹理上传用的字节流。
- `unpack_grid_state(width, height, tables, state_int_data, state_vec_data, state_misc_data)`
  - 把 GPU 当前状态回读成 CPU `Grid`。
- `GpuSimulator`
  - 提供:
    - `load_grid(grid)`
    - `set_liquid_brownian_enabled(enabled)`
    - `set_blocked_impulse_enabled(enabled)`
    - `set_directional_fallback_enabled(enabled)`
    - `set_directional_fallback_angle_limit_degrees(angle_limit_degrees)`
    - `set_external_support_anchors(anchors)`
    - `read_region(x, y, width, height)`
    - `write_region(x, y, region, buffer_index=None)`
    - `copy_region(src_x, src_y, width, height, dst_x, dst_y, src_buffer_index=None, dst_buffer_index=None)`
    - `copy_transient_region(src_x, src_y, width, height, dst_x, dst_y)`
    - `stage_region(x, y, width, height)`
    - `copy_from_staged_region(staged, src_x, src_y, width, height, dst_x, dst_y, dst_buffer_index=None)`
    - `read_staged_region(staged)`
    - `release_staged_region(staged)`
    - `clear_region_transients(x, y, width, height)`
    - `step(dt)`
    - `render(view_mode=DebugViewMode.MATERIAL)`
    - `paint_circle(center_x, center_y, radius, family_id, variant_id, overrides=None)`
    - `readback_grid()`

## 依赖的对外接口

- 第三方:
  - `moderngl`
- `engine.grid.Grid`
- `engine.render.DebugViewMode`
- `engine.support.SUPPORT_FAILURE_THRESHOLD`
- `engine.types.CellFlag`
- `engine.types.CellState`
- `engine.types.LifetimeMode`
- `engine.types.MaterialRegistry`
- `engine.types.MatterState`
- `engine.types.ReactionKind`

## 主要功能

- 把 CPU 材质表打包成 GPU 可直接读取的结构化缓冲。
- 变体缓冲当前除了基础物理参数,还会一起打包:
  - `reaction_energy`
  - `liquid_contact_heat_exchange_multiplier`
  - `same_variant_heat_exchange_multiplier`
  - `mobility`
  - `pressure_response`
  - `gravity_scale`
  - `buoyancy_scale`
  - `thermal_motion_scale`
  - `wind_coupling`
  - `wind_vertical_factor`
  - `velocity_decay`
  - `downward_blocked_diagonal_fallback`
- 把 cell 动态状态拆成整数纹理和浮点纹理,并做双缓冲。
- 额外维护一张 `external_support_anchor` 纹理,供大世界活动窗把冻结区外部支撑源映射到当前边界。
- 维护 GPU 侧持久化 `pressure` 标量场、`source_force` 和 `force_wave` 纹理。
- 提供面向大世界活动窗的区域接口:
  - 局部读回 `cell state`
  - 局部写入 `cell state`
  - 在 GPU 内把重叠区域从旧位置复制到新位置
  - 在活动窗平移时复制 overlap 的 transient 力场,并只清空新暴露条带
  - 把被逐出的区域临时 stage 在 GPU 纹理里,供后续慢速回写或快速装回
- 用 compute shader 执行:
  - `support`
  - `reactions`
  - `thermal`
  - `phases`
  - `pressure`
  - `source_force`
  - `force_wave`
  - 交换式 `motion`
  - `collapse`
- GPU `support` 当前除了真实 `FIXPOINT`,还会把 `external_support_anchor` 当作虚拟外部支撑源。
- GPU `motion` 和 CPU 一样,求解器只按 `matter_state` 分支; 同一物态内部的差异全部由打包后的通用运动参数控制。
- GPU 里的液体布朗运动、`blocked_impulse` 和方向 fallback 都带运行时开关,语义与 CPU 保持一致。
- GPU 当前每个外部 `step` 会先跑一次“非气态物质”交换,再跑一次“空气/气体”交换。
- `thermal` 阶段使用数据驱动的导热加速倍率,不再把 `steam` 写成 shader 里的固定分支。
- `reaction` 阶段当前只通过 `reaction_energy` 改变当前格自身温度,不会直接给邻格写热量。
- `phases` 阶段只切换变体身份,不会用目标变体 `base_temperature` 重置当前温度。
- 用 GPU 调试着色 pass 直接输出 demo 采样的最终纹理,支持材质视图、温度视图和压力视图。
