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
- 维护 GPU 侧持久化 `pressure` 标量场、`source_force` 和 `force_wave` 纹理。
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
- GPU `motion` 和 CPU 一样,求解器只按 `matter_state` 分支; 同一物态内部的差异全部由打包后的通用运动参数控制。
- GPU `motion` 也会先用这些通用参数判断某个变体当前是否具备平移能力; 对不可平移的变体,shader 会提前跳过 motion 候选与 claim 热路径,只保留轻量的速度/blocked impulse 结算。
- 所有运动都按“和可交换目标换位”实现,包括与空背景以及更轻流体的交换。
- 同为气体/空气的垂直交换使用方向感知的温度修正密度: 更轻者只能向上换入更重目标,更重者只能向下换入更轻目标。
- 开启 `downward_blocked_diagonal_fallback` 的变体,在主方向仍向下且正下方受阻时,shader 会把 fallback 收窄成左下和右下两个 `<= 45°` 候选。
- GPU 里的液体布朗运动、`blocked_impulse` 和方向 fallback 都带运行时开关,语义与 CPU 保持一致。
- GPU 里的环境 `empty` 空气也会参与 pressure、热运动和对流,但不会主动去替换显式气体包,避免背景空气反向吞掉蒸汽或毒气团。
- GPU 当前每个外部 `step` 会先跑一次“非气态物质”交换,再跑一次“空气/气体”交换。
- occupied-target 交换时,被顶开的目标格不会只做简单复制回填,而是会先应用自身受力和剩余意图后再落回 source。
- `thermal` 阶段使用数据驱动的导热加速倍率,不再把 `steam` 写成 shader 里的固定分支。
- `reaction` 阶段当前只通过 `reaction_energy` 改变当前格自身温度,不会直接给邻格写热量。
- `phases` 阶段只切换变体身份,不会用目标变体 `base_temperature` 重置当前温度。
- 用 GPU 调试着色 pass 直接输出 demo 采样的最终纹理,支持材质视图、温度视图和压力视图。
