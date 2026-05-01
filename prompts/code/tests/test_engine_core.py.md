# test_engine_core.py

这个文件提供第一版体素引擎原型的基础验证。

## 对外接口

- 无库接口。
- 这是 `unittest` 测试文件。

## 依赖的对外接口

- `engine.gpu_backend.GpuMaterialTables`
- `engine.gpu_backend.GpuSimulator`
- `engine.gpu_backend.pack_grid_state`
- `engine.gpu_backend.unpack_grid_state`
- `engine.grid.create_grid`
- `engine.materials.build_material_registry`
- `engine.phases.apply_phase_transitions`
- `engine.sim.inject_cells`
- `engine.sim.step`
- `engine.types.CellFlag`
- `engine.types.CellState`
- `engine.world.ActiveWorldWindow`
- `engine.world.WorldChunkStore`
- `scripts.run_engine_demo.parse_args`
- `slm.data_generation`
- `slm.io_utils`
- `slm.model_socket_schema`

## 主要功能

- 验证 `slm` 包改名后导入仍可用。
- 验证基础重力、对角释放、液体铺开、热空气、相变、腐蚀、自耗、GPU 打包与回读等核心行为。
- 验证大世界流式层:
  - `WorldChunkStore` 的矩形读写
  - `WorldChunkStore.rect_has_stored_cells()` 对默认空区与真实存储区的区分
  - `WorldChunkStore.rect_has_stored_cells()` 不会因为同 chunk 其他位置有内容而误判当前 incoming rect
  - `WorldChunkStore.write_rect()` 会把 `GridSlice.anchored_support_mask` 一并落到冻结区 anchored support
  - `ActiveWorldWindow` 平移时对 overlap cell state 的保留
  - 冻结区外部支撑锚点对活动窗边界的续撑
  - 相机持续上下移动时不会回卷,并且分页后的 world 行保持连续
  - 相机移动时 staging 队列不会立刻 flush,即使 pending 超限也只记录压力,空闲后才会逐步写回
  - staged region flush 不再依赖全图 `recompute_anchored_support`,而是直接使用 region 支撑快照写回
  - staged region flush 当前会按 slice 分多次完成,不会一口气读回整条 strip
  - incoming rect 命中真实存储 cell 时,当前会走 GPU 直写 store 内容路径,而不是 `read_rect + write_region`
- 验证方向 fallback 开关和 `directional_fallback_angle_limit_degrees` 的边界语义。
- 验证 CPU/GPU 都会用交替平局打破和镜像扫描顺序减少固定 `45°` 偏置。
- 验证热空气、冷空气和显式气体团的运动不会破坏质量守恒。
- 验证热传导、相变和空气背景层结在 CPU/GPU 两条路径上保持一致。
- 额外覆盖“求解器不再写材质名特判”的回归:
  - 非 `water` 液体也应复用同一套 `liquid` 侧向铺开规则
  - 非承重固体也应复用同一套“正下受阻时只试左右下对角”的规则
  - 对应 CPU 和 GPU 两侧都各有测试
- GPU 测试在可创建 OpenGL 4.3 context 时验证:
  - backend 能执行相变、寿命衰减、支撑传播、交换式运动和热传导
  - backend 的区域读写与复制接口保持 cell state 位置正确
  - external support anchor 的兼容单边更新接口会正确写入 compact edge buffer
  - external support anchor 的边界输入仍能在 support 阶段作为外部锚点生效
  - empty strip 的 GPU 直填会写出正确的环境温度
  - staged overlap 会在 empty fill 之前优先复用
  - `fire + water` 不会因为扩散而自发增殖
  - 空气/气体运动不会吞掉或复制显式气体团
  - 状态回读与调试视图输出保持正确
