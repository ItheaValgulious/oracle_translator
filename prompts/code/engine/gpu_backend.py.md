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
- `engine.types.MotionMode`
- `engine.types.ReactionKind`

## 主要功能

- 把 CPU 材质表打包成 GPU 可直接读取的结构化缓冲。
- 变体缓冲当前也携带 `reaction_energy`,用于 reaction shader 表达吸热/放热。
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
- 运动判定直接使用 `matter_state + motion_mode + density + velocity + blocked_impulse`,不再以旧的 `sim_kind` 作为核心分支依据。
- 先在 GPU 上更新统一压力场,再由压力梯度生成当前帧的局部 `source_force`。
- GPU 当前通过持久化 `force_wave` 和 `delta = source - lambda * prev_source` 的方式,把液体内部受力跨帧往更远处传播。
- 所有运动都按“和可交换目标换位”实现,包括与空背景以及更轻流体的交换。
- 同为气体/空气的垂直交换使用方向感知的温度修正密度: 更轻者只能向上换入更重目标,更重者只能向下换入更轻目标,避免热/冷空气方向在 GPU 并发交换里反转。
- GPU 里的液体会弱化布朗运动,改用更强的横向压力响应来加快地表铺开和液面整平。
- GPU 里的液体布朗运动当前也受温度影响,但量级仍然很小。
- GPU 里的液体布朗运动当前带有运行时开关,可由 demo 或上层逻辑直接启停,用于对比液体抖动对行为和 FPS 的影响。
- GPU 里的 `blocked_impulse` 当前也带有运行时开关; 关闭后 shader 不再把旧的 `blocked_x/y` 参与目标选择,并会把已有残余意图清零。
- GPU 里的深水在下方受阻时,还会按压头额外增强横向压力响应,减少大弧顶长时间挂住的现象。
- GPU 里的气体会保留更强的布朗运动,避免只被浮力和压差单向拉走。
- GPU 里的气体浮力当前按温度修正后的等效密度计算,温度越高越轻。
- GPU 里的 `empty` 热空气当前也会在 `motion` 里参与交换,但只在温度明显偏离环境或已有残余速度时才会主动移动。
- GPU 当前的主流程里不再额外运行独立 `swap` pass; 分层交换已并入交换式 `motion`。
- GPU 当前每个外部 `step` 会在完整 `motion` 后追加 2 次 `liquids_only` relaxation,用于让大水堆在有限步数内降低峰值; 远程传力仍主要依赖跨帧 `force_wave`。
- `motion` 当前吃的是“局部 source force + 持久化 propagated wave”的总受力,而不是只吃本地压力梯度。
- occupied-target 交换时,被顶开的目标格不会只做简单复制回填,而是会先应用自身受力和剩余意图后再落回 source。
- 运动和交换会把 `state_misc` 中的当前温度随 cell 一起搬运。
- 物质移入 `empty` 时,GPU 会让原位置空气保留源物质和目标空气二者中较高的温度,避免空气温度被重置成创建温度。
- 没有参与本轮运动的普通 `empty` 格,在 `motion` resolve 里会直接保留自己当前的温度状态,不会被当成新空气重建。
- 对液体自由表面或外侧壁,GPU 当前还会给“纯水平向外”的候选方向一个额外偏置,尽量减少边缘格总是优先选 `右下/左下` 造成的锯齿滴落。
- 液体作为当前格选择目标时不会主动向上爬进气体/空气,避免 GPU 并发交换把自由表面抽成高水刺。
- 方向选择带有轻量哈希扰动,用于消除固定邻域枚举顺序带来的左上偏置。
- 当液体布朗运动被关闭,或当前期望速度很小时,GPU 也会抑制这层方向扰动,避免存在额外的隐藏随机源。
- GPU 的随机扰动按 `dt` 归一化,`dt=0` 不会凭空生成新布朗运动,但已有速度/残余意图仍可释放。
- 在 `support` 阶段每个外部 `step` 只运行一次 compute pass,把支撑波次向相邻 `platform` 推进一格。
- `support` 阶段同 CPU 一样用 `generation` 记录最新波次 id,用 `support_value` 记录 10 秒失撑倒计时。
- `thermal` 阶段只做相邻格热交换,不会把已有 cell 拉回材质 `base_temperature`。
- `thermal` 阶段对液体-可凝结物质气体相界面施加额外传热倍率,让被水包裹的蒸汽团能更快冷却凝结,但不会放大 `fire` 对水的直接加热。
- `reaction` 阶段当前只通过 `reaction_energy` 改变当前格自身温度,不会直接给邻格写热量。
- `phases` 阶段只切换变体身份,不会用目标变体 `base_temperature` 重置当前温度。
- 在 `reaction` pass 中支持反应后自耗的变体语义。
- `fire` 寿命结束或反应自耗变为空气时,会把当前温度留给 `empty`。
- 用 GPU 调试着色 pass 直接输出 demo 采样的最终纹理,支持材质视图、温度视图和压力视图。
- 提供笔刷写入和状态回读接口,方便 demo 交互和测试验证。
