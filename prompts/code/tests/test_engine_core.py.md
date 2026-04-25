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
- `scripts.run_engine_demo.parse_args`
- `slm.data_generation`
- `slm.io_utils`
- `slm.model_socket_schema`

## 主要功能

- 验证 `slm` 包改名后导入仍可用。
- 验证 `powder` 下落。
- 验证纯重力下落不会凭空制造反向冲量。
- 验证对角意图会先消耗剩余轴向需求,而不是连续重复走对角。
- 验证液体会在地面上侧向铺开。
- 验证关闭液体布朗运动后,单格水不会再凭空获得横向随机速度。
- 验证关闭 `blocked_impulse` 后,已有残余意图会在下一步被清零,不再继续影响运动。
- 验证气体的布朗运动会带来横向位置变化,而不只是单向浮沉; 该测试会先把环境空气设到不触发冷凝的温度,避免被相变旁路。
- 验证重液能与轻液交换并下沉。
- 验证 `fire + water` 的扩散不会无端制造额外质量。
- 验证下落粉体不会把单格水一路顶高成细柱。
- 验证支撑网络断开后结构逐步粉化。
- 验证支撑信号每步只传播一格、能跨多帧沿平台链传播,且连通平台链上不随距离衰减。
- 验证 `platform` 在 10 秒支撑超时后才开始掉 `integrity`。
- 验证水相变与 `sand -> molten_glass`。
- 验证 `fire` 使用 `age` 衰减。
- 验证温度视图会输出不同于材质视图的调色结果。
- 验证压力视图会根据当前压力标量场输出不同于材质视图的调色结果。
- 验证已有 cell 温度不会自动回归材质创建温度。
- 验证 `platform` 之间会导热。
- 验证 `fire` 会先用 `reaction_energy` 加热自身,随后再通过热传导把热量扩散出去。
- 验证热空气与周围空气的交换足够明显,不会只在数值上微弱变化到热力图几乎看不出来。
- 验证热空气除了导热,还会在空空气中向上漂移。
- 验证冷空气会向下漂移,避免冷热空气方向被反转。
- 验证被水包裹的蒸汽团会在有限步数内冷却并凝结。
- 验证单格蒸汽在普通空气中不会几帧内立刻凝结。
- 验证冰块在常温空气中能持续存在一段时间。
- 验证 `blocked_impulse` 成功移动后的残余会被夹紧。
- 验证移动中的 cell 会把当前温度带到新位置。
- 验证物质移入热空气时,留下的空气不会重置成创建温度。
- 验证 `fire` 移走或熄灭后会留下热空气。
- 验证 `fire` 离开后留下的热空气会继续按邻格温差逐步降温。
- 验证 `fire` 会在运动前加热相邻水,也能通过空气间隙让水升温。
- 验证热量能通过空气间隙继续传导。
- 验证酸液在成功腐蚀后会自耗。
- 验证 GPU 状态打包与回读不会破坏格子状态。
- GPU 测试创建 context 时会先试 standalone,再试隐藏窗口 context; 两者都失败会直接报错而不是静默跳过。
- 在可创建 OpenGL 4.3 context 时,验证 GPU backend 能执行相变和寿命衰减。
- 在可创建 OpenGL 4.3 context 时,验证 GPU 支撑信号每步只传播一格,并按 10 秒超时后才开始掉 `integrity`。
- 在可创建 OpenGL 4.3 context 时,验证关闭液体布朗运动后,GPU 下单格水不会再凭空获得横向随机速度。
- 在可创建 OpenGL 4.3 context 时,验证关闭 `blocked_impulse` 后,GPU 下已有残余意图会在下一步被清零。
- 在可创建 OpenGL 4.3 context 时,验证 GPU 压力视图会输出不同于材质视图的调色结果。
- 验证 GPU 下温度不会自动回归、`platform` 会导热、运动会携带温度,且 `reaction_energy` 与空气热扩散语义和 CPU 保持一致。
- 在可创建 OpenGL 4.3 context 时,验证 GPU motion 和 CPU motion 一样会先消耗剩余轴向意图。
- 在可创建 OpenGL 4.3 context 时,验证 GPU acid 也会按配置自耗。
- 在可创建 OpenGL 4.3 context 时,验证 GPU 下 `fire + water` 也不会因为扩散而自发增殖。
- 在可创建 OpenGL 4.3 context 时,验证 GPU 下水体下落不会丢格。
- 在可创建 OpenGL 4.3 context 时,验证 GPU 下液体遇到底板会立即侧向铺开。
- 在可创建 OpenGL 4.3 context 时,验证 GPU 下气体布朗运动也会带来横向游走。
- 在可创建 OpenGL 4.3 context 时,验证中等尺寸蓄水池能在有限步数内沿地面快速铺开。
- 在可创建 OpenGL 4.3 context 时,验证较大液体堆的峰值会在有限步数内明显下降。
- 在可创建 OpenGL 4.3 context 时,验证引入大液体区域专用 relaxation 后,基础的单步液体侧移和分层交换语义仍然成立。
- 在可创建 OpenGL 4.3 context 时,验证 GPU 下冷空气会向下漂移,水包蒸汽团会冷却凝结。
- 在可创建 OpenGL 4.3 context 时,验证 GPU 下空气中的蒸汽不会立即凝结、冰块能持续存在,以及 `blocked_impulse` 成功移动后会被夹紧。
- 在可创建 OpenGL 4.3 context 时,验证 GPU 下重液会与轻液交换分层。
- 在可创建 OpenGL 4.3 context 时,验证 GPU 下液体会与更轻的气体交换分层。
- 在可创建 OpenGL 4.3 context 时,验证下落粉体不会把水一路抬高到空气中,即使水最终未必横向移开。
- 验证 GPU 笔刷接受来自窗口事件的浮点坐标时不会崩溃。
- 验证 demo 命令行参数能正确解析网格分辨率、窗口尺寸、子步进数、液体布朗运动开关和 `blocked_impulse` 开关。
- 验证 RGBA 帧输出长度与 demo 场景中关键物质分布。
